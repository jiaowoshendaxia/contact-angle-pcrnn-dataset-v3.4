"""LS-PSRMoE v4.1 probe-assisted training and nested source validation."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import platform
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from torch.nn import functional as F

from .data import V4Tables, clean_text
from .features import (
    CONDITION_CATEGORICAL,
    CONDITION_NUMERIC,
    LIQUID_NUMERIC,
    SURFACE_CATEGORICAL,
    SURFACE_NUMERIC,
    CategoryEncoder,
    NumericScaler,
)
from .metrics import paired_cluster_bootstrap, regression_metrics
from .model import PhysicsSummaryResidualExpert
from .uncertainty import AdaptiveConformal


PHYSICS_NUMERIC = [
    "nnls_dispersion_mj_m2",
    "nnls_polar_mj_m2",
    "nnls_physical_prediction_deg",
    "nnls_residual_norm",
    "n_unique_liquids",
]
FIT_STATUSES = ["interior_fit", "boundary_fit"]
EXTERNAL_SPLITS = ["internal_test", "legacy_external", "prospective_open_external"]


def _json_default(value: Any) -> Any:
    """Convert NumPy scalars and paths without silently stringifying values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


@dataclass(frozen=True)
class SearchConfig:
    max_delta_cos: float
    source_balance: bool
    residual_penalty: float

    @property
    def key(self) -> str:
        balance = "balanced" if self.source_balance else "unbalanced"
        return f"delta{self.max_delta_cos:.2f}_{balance}_rp{self.residual_penalty:.2f}"


@dataclass
class V41Encoded:
    sample_ids: list[str]
    record_ids: list[str]
    source_group_ids: list[str]
    surface_group_ids: list[str]
    target_liquid_ids: list[str]
    splits: list[str]
    solid_families: list[str]
    n_probes: np.ndarray
    surface_numeric: np.ndarray
    categorical: np.ndarray
    target_liquid_numeric: np.ndarray
    condition_numeric: np.ndarray
    physics_summary: np.ndarray
    physics_cosine: np.ndarray
    physics_angle: np.ndarray
    target_angle: np.ndarray
    target_cosine: np.ndarray
    tabular: np.ndarray
    ood_features: np.ndarray
    unknown_category: np.ndarray

    def __len__(self) -> int:
        return len(self.sample_ids)

    def subset(self, indices: Sequence[int] | np.ndarray) -> "V41Encoded":
        indices = np.asarray(indices, dtype=int)
        kwargs: dict[str, Any] = {}
        list_fields = [
            "sample_ids", "record_ids", "source_group_ids", "surface_group_ids",
            "target_liquid_ids", "splits", "solid_families",
        ]
        for field in list_fields:
            values = getattr(self, field)
            kwargs[field] = [values[index] for index in indices]
        for field in [
            "n_probes", "surface_numeric", "categorical", "target_liquid_numeric",
            "condition_numeric", "physics_summary", "physics_cosine", "physics_angle",
            "target_angle", "target_cosine", "tabular", "ood_features", "unknown_category",
        ]:
            kwargs[field] = getattr(self, field)[indices]
        return V41Encoded(**kwargs)

    def without_physics_summary(self) -> "V41Encoded":
        clone = copy.deepcopy(self)
        clone.physics_summary = np.zeros_like(clone.physics_summary)
        return clone


class V41Preprocessor:
    """Train-source-only preprocessing for the compact physics-summary model."""

    def __init__(self) -> None:
        self.surface_scaler: NumericScaler | None = None
        self.liquid_scaler: NumericScaler | None = None
        self.condition_scaler: NumericScaler | None = None
        self.physics_scaler: NumericScaler | None = None
        self.category_encoder: CategoryEncoder | None = None

    @staticmethod
    def _lookups(tables: V4Tables) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return (
            tables.surfaces.set_index("surface_group_id", drop=False),
            tables.liquids.set_index("liquid_id", drop=False),
            tables.measurements.set_index("measurement_id", drop=False),
        )

    def fit(self, tables: V4Tables, samples: pd.DataFrame) -> "V41Preprocessor":
        surfaces, liquids, measurements = self._lookups(tables)
        surface_rows = surfaces.loc[sorted(set(samples["surface_group_id"].astype(str)))]
        liquid_rows = liquids.loc[sorted(set(samples["target_liquid_id"].astype(str)))]
        target_rows = measurements.loc[samples["target_measurement_id"].astype(str)]
        self.surface_scaler = NumericScaler.fit(surface_rows, SURFACE_NUMERIC)
        self.liquid_scaler = NumericScaler.fit(liquid_rows, LIQUID_NUMERIC)
        self.condition_scaler = NumericScaler.fit(target_rows, CONDITION_NUMERIC)
        self.physics_scaler = NumericScaler.fit(samples, PHYSICS_NUMERIC)
        category_frame = pd.concat([
            surface_rows[SURFACE_CATEGORICAL].reset_index(drop=True),
            target_rows[CONDITION_CATEGORICAL].reset_index(drop=True),
        ], axis=1)
        self.category_encoder = CategoryEncoder.fit(
            category_frame, SURFACE_CATEGORICAL + CONDITION_CATEGORICAL
        )
        return self

    def _require_fit(self) -> None:
        if any(value is None for value in [
            self.surface_scaler, self.liquid_scaler, self.condition_scaler,
            self.physics_scaler, self.category_encoder,
        ]):
            raise RuntimeError("V41Preprocessor must be fitted on training sources")

    @property
    def categorical_cardinalities(self) -> list[int]:
        self._require_fit()
        assert self.category_encoder is not None
        return self.category_encoder.cardinalities(SURFACE_CATEGORICAL + CONDITION_CATEGORICAL)

    def transform(self, tables: V4Tables, samples: pd.DataFrame) -> V41Encoded:
        self._require_fit()
        assert self.surface_scaler and self.liquid_scaler and self.condition_scaler
        assert self.physics_scaler and self.category_encoder
        surfaces, liquids, measurements = self._lookups(tables)
        rows: dict[str, list[Any]] = {name: [] for name in [
            "sample_ids", "record_ids", "source_group_ids", "surface_group_ids",
            "target_liquid_ids", "splits", "solid_families", "n_probes",
            "surface_numeric", "categorical", "target_liquid_numeric", "condition_numeric",
            "physics_summary", "physics_cosine", "physics_angle", "target_angle",
            "tabular", "ood_features", "unknown_category",
        ]}
        for sample in samples.itertuples(index=False):
            surface = surfaces.loc[str(sample.surface_group_id)]
            liquid = liquids.loc[str(sample.target_liquid_id)]
            measurement = measurements.loc[str(sample.target_measurement_id)]
            surface_values = self.surface_scaler.transform_row(surface, SURFACE_NUMERIC)
            liquid_values = self.liquid_scaler.transform_row(liquid, LIQUID_NUMERIC)
            condition_values = self.condition_scaler.transform_row(measurement, CONDITION_NUMERIC)
            category_row = pd.concat([surface, measurement])
            category_values = self.category_encoder.transform_row(
                category_row, SURFACE_CATEGORICAL + CONDITION_CATEGORICAL
            )
            category_one_hot = self.category_encoder.one_hot(
                category_values, SURFACE_CATEGORICAL + CONDITION_CATEGORICAL
            )
            physics_values = self.physics_scaler.transform_row(
                pd.Series(sample._asdict()), PHYSICS_NUMERIC
            )
            fit_status = str(sample.fit_status)
            status_one_hot = [float(fit_status == status) for status in FIT_STATUSES]
            flags = [
                float(str(sample.boundary_fit).casefold() == "yes"),
                1.0,
                float(str(sample.has_independent_sfe).casefold() == "yes"),
            ]
            physics_summary = physics_values + status_one_hot + flags
            physics_angle = float(sample.nnls_physical_prediction_deg)
            if not 0.0 <= physics_angle <= 180.0:
                raise ValueError(f"Invalid NNLS physical angle for {sample.sample_id}")
            if min(float(sample.nnls_dispersion_mj_m2), float(sample.nnls_polar_mj_m2)) < 0.0:
                raise ValueError(f"Negative NNLS SFE for {sample.sample_id}")
            target = float(sample.target_contact_angle_deg)
            tabular = (
                surface_values + category_one_hot + liquid_values + condition_values
                + physics_summary
            )
            ood_features = surface_values + liquid_values + condition_values + physics_summary
            values = {
                "sample_ids": str(sample.sample_id),
                "record_ids": str(sample.record_id),
                "source_group_ids": str(sample.source_group_id),
                "surface_group_ids": str(sample.surface_group_id),
                "target_liquid_ids": str(sample.target_liquid_id),
                "splits": str(sample.v4_split),
                "solid_families": clean_text(surface.get("solid_family")) or "unknown",
                "n_probes": int(sample.n_probes),
                "surface_numeric": surface_values,
                "categorical": category_values,
                "target_liquid_numeric": liquid_values,
                "condition_numeric": condition_values,
                "physics_summary": physics_summary,
                "physics_cosine": math.cos(math.radians(physics_angle)),
                "physics_angle": physics_angle,
                "target_angle": target,
                "tabular": tabular,
                "ood_features": ood_features,
                "unknown_category": any(code == 0 for code in category_values),
            }
            for name, value in values.items():
                rows[name].append(value)
        target = np.asarray(rows["target_angle"], dtype=np.float32)
        return V41Encoded(
            sample_ids=rows["sample_ids"], record_ids=rows["record_ids"],
            source_group_ids=rows["source_group_ids"], surface_group_ids=rows["surface_group_ids"],
            target_liquid_ids=rows["target_liquid_ids"], splits=rows["splits"],
            solid_families=rows["solid_families"], n_probes=np.asarray(rows["n_probes"], dtype=np.int16),
            surface_numeric=np.asarray(rows["surface_numeric"], dtype=np.float32),
            categorical=np.asarray(rows["categorical"], dtype=np.int64),
            target_liquid_numeric=np.asarray(rows["target_liquid_numeric"], dtype=np.float32),
            condition_numeric=np.asarray(rows["condition_numeric"], dtype=np.float32),
            physics_summary=np.asarray(rows["physics_summary"], dtype=np.float32),
            physics_cosine=np.asarray(rows["physics_cosine"], dtype=np.float32),
            physics_angle=np.asarray(rows["physics_angle"], dtype=np.float32),
            target_angle=target, target_cosine=np.cos(np.deg2rad(target)).astype(np.float32),
            tabular=np.asarray(rows["tabular"], dtype=np.float32),
            ood_features=np.asarray(rows["ood_features"], dtype=np.float32),
            unknown_category=np.asarray(rows["unknown_category"], dtype=bool),
        )

    def to_dict(self) -> dict[str, Any]:
        self._require_fit()
        assert self.surface_scaler and self.liquid_scaler and self.condition_scaler
        assert self.physics_scaler and self.category_encoder
        return {
            "surface_scaler": asdict(self.surface_scaler),
            "liquid_scaler": asdict(self.liquid_scaler),
            "condition_scaler": asdict(self.condition_scaler),
            "physics_scaler": asdict(self.physics_scaler),
            "category_encoder": self.category_encoder.vocabularies,
            "physics_fields": PHYSICS_NUMERIC,
            "fit_statuses": FIT_STATUSES,
        }

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


@dataclass
class CompactPrediction:
    physics: np.ndarray
    neural: np.ndarray
    residual: np.ndarray
    sigma: np.ndarray


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _tensor(values: np.ndarray, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
    return torch.as_tensor(values, dtype=dtype, device=device)


def _new_model(
    data: V41Encoded,
    preprocessor: V41Preprocessor,
    search: SearchConfig,
    config: dict[str, Any],
) -> PhysicsSummaryResidualExpert:
    model = config["model"]
    return PhysicsSummaryResidualExpert(
        surface_numeric_dim=data.surface_numeric.shape[1],
        liquid_numeric_dim=data.target_liquid_numeric.shape[1],
        condition_numeric_dim=data.condition_numeric.shape[1],
        physics_summary_dim=data.physics_summary.shape[1],
        categorical_cardinalities=preprocessor.categorical_cardinalities,
        categorical_embedding_dim=int(model["categorical_embedding_dim"]),
        dropout=float(model["dropout"]),
        max_delta_cos=search.max_delta_cos,
    )


def _forward(
    model: PhysicsSummaryResidualExpert,
    data: V41Encoded,
    indices: np.ndarray,
    device: torch.device,
):
    return model(
        _tensor(data.surface_numeric[indices], device),
        _tensor(data.categorical[indices], device, torch.long),
        _tensor(data.target_liquid_numeric[indices], device),
        _tensor(data.condition_numeric[indices], device),
        _tensor(data.physics_summary[indices], device),
        _tensor(data.physics_cosine[indices], device),
    )


def _source_weights(data: V41Encoded) -> np.ndarray:
    counts = Counter(data.source_group_ids)
    values = np.asarray([1.0 / math.sqrt(counts[source]) for source in data.source_group_ids])
    return (values / values.mean()).astype(np.float32)


def _loss(
    output: Any,
    data: V41Encoded,
    indices: np.ndarray,
    sample_weights: np.ndarray,
    search: SearchConfig,
    config: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    observed = _tensor(data.target_angle[indices], device)
    target_cosine = _tensor(data.target_cosine[indices], device)
    weights = _tensor(sample_weights[indices], device)
    weights = weights / weights.mean().clamp_min(1e-8)
    angle = F.huber_loss(output.theta_neural, observed, delta=5.0, reduction="none")
    cosine = (output.cosine_neural - target_cosine).square()
    residual = output.residual_cosine.square()
    scaled_error = (observed - output.theta_neural) / 10.0
    nll = 0.5 * (torch.exp(-output.log_variance) * scaled_error.square() + output.log_variance)
    loss_config = config["loss"]
    per_sample = (
        float(loss_config["angle_huber"]) * angle
        + float(loss_config["cosine_mse"]) * cosine
        + search.residual_penalty * residual
        + float(loss_config["gaussian_nll"]) * nll
    )
    return torch.mean(weights * per_sample)


@torch.no_grad()
def predict_compact(
    model: PhysicsSummaryResidualExpert,
    data: V41Encoded,
    device: torch.device,
    batch_size: int = 512,
) -> CompactPrediction:
    model.eval()
    values: dict[str, list[np.ndarray]] = {name: [] for name in ["physics", "neural", "residual", "sigma"]}
    for start in range(0, len(data), batch_size):
        indices = np.arange(start, min(start + batch_size, len(data)))
        output = _forward(model, data, indices, device)
        batch = {
            "physics": output.theta_physics,
            "neural": output.theta_neural,
            "residual": output.residual_cosine,
            "sigma": 10.0 * torch.exp(0.5 * output.log_variance),
        }
        for name, tensor in batch.items():
            values[name].append(tensor.detach().cpu().numpy())
    return CompactPrediction(**{name: np.concatenate(parts) for name, parts in values.items()})


def train_compact(
    train: V41Encoded,
    validation: V41Encoded | None,
    preprocessor: V41Preprocessor,
    search: SearchConfig,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
    epochs: int,
    patience: int,
    fixed_epochs: int | None = None,
) -> tuple[PhysicsSummaryResidualExpert, int, list[dict[str, Any]]]:
    set_seed(seed)
    model = _new_model(train, preprocessor, search, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["model"]["learning_rate"]),
        weight_decay=float(config["model"]["weight_decay"]),
    )
    batch_size = int(config["model"]["batch_size"])
    train_weights = _source_weights(train)
    validation_weights = _source_weights(validation) if validation is not None else None
    source_counts = Counter(train.source_group_ids)
    if search.source_balance:
        probability = np.asarray([1.0 / source_counts[source] for source in train.source_group_ids], dtype=float)
        probability /= probability.sum()
    else:
        probability = None
    run_epochs = int(fixed_epochs or epochs)
    rng = np.random.default_rng(seed)
    best_state = copy.deepcopy(model.state_dict())
    best_value = float("inf")
    best_epoch = run_epochs
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(run_epochs):
        model.train()
        if probability is None:
            ordering = rng.permutation(len(train))
        else:
            ordering = rng.choice(len(train), size=len(train), replace=True, p=probability)
        batch_losses: list[float] = []
        for start in range(0, len(ordering), batch_size):
            indices = np.asarray(ordering[start:start + batch_size], dtype=int)
            optimizer.zero_grad(set_to_none=True)
            output = _forward(model, train, indices, device)
            loss = _loss(output, train, indices, train_weights, search, config, device)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite v4.1 loss for {train.sample_ids[indices[0]]}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        if validation is None:
            history.append({"epoch": epoch + 1, "train_loss": float(np.mean(batch_losses))})
            continue
        model.eval()
        with torch.no_grad():
            indices = np.arange(len(validation))
            output = _forward(model, validation, indices, device)
            assert validation_weights is not None
            validation_loss = float(_loss(
                output, validation, indices, validation_weights, search, config, device
            ).detach().cpu())
        history.append({
            "epoch": epoch + 1, "train_loss": float(np.mean(batch_losses)),
            "validation_loss": validation_loss,
        })
        if validation_loss < best_value - 1e-6:
            best_value = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if validation is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, history


def _fit_tree_models(
    features: np.ndarray,
    target: np.ndarray,
    physics: np.ndarray,
    seeds: Sequence[int],
) -> dict[str, list[Any]]:
    from xgboost import XGBRegressor

    models: dict[str, list[Any]] = {
        "xgboost": [], "random_forest": [],
        "xgboost_direct": [], "random_forest_direct": [],
    }
    for seed in seeds:
        xgb_options = dict(
            n_estimators=400, max_depth=3, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
            reg_lambda=2.0, objective="reg:squarederror", random_state=int(seed),
            n_jobs=1, tree_method="hist",
        )
        rf_options = dict(
            n_estimators=500, max_features="sqrt", min_samples_leaf=2,
            random_state=int(seed), n_jobs=1,
        )
        residual_target = np.asarray(target) - np.asarray(physics)
        xgb_residual = XGBRegressor(**xgb_options).fit(features, residual_target)
        rf_residual = RandomForestRegressor(**rf_options).fit(features, residual_target)
        xgb_direct = XGBRegressor(**xgb_options).fit(features, target)
        rf_direct = RandomForestRegressor(**rf_options).fit(features, target)
        models["xgboost"].append(xgb_residual)
        models["random_forest"].append(rf_residual)
        models["xgboost_direct"].append(xgb_direct)
        models["random_forest_direct"].append(rf_direct)
    return models


def _predict_tree_member(
    name: str,
    model: Any,
    features: np.ndarray,
    physics: np.ndarray,
) -> np.ndarray:
    prediction = np.asarray(model.predict(features), dtype=float)
    if name in {"xgboost", "random_forest"}:
        prediction = np.asarray(physics, dtype=float) + prediction
    return np.clip(prediction, 0.0, 180.0)


def _predict_tree_models(
    models: dict[str, list[Any]],
    features: np.ndarray,
    physics: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        name: np.mean([
            _predict_tree_member(name, model, features, physics) for model in fitted
        ], axis=0)
        for name, fitted in models.items()
    }


def _huber_numpy(error: np.ndarray, delta: float = 5.0) -> np.ndarray:
    absolute = np.abs(error)
    return np.where(absolute <= delta, 0.5 * np.square(error), delta * (absolute - 0.5 * delta))


def fit_simplex_weights(experts: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Fit auditable nonnegative global expert weights with a unit-sum constraint."""
    experts = np.asarray(experts, dtype=float)
    target = np.asarray(target, dtype=float)
    if experts.ndim != 2 or experts.shape[0] != len(target):
        raise ValueError("Expert matrix and target length do not match")
    n_experts = experts.shape[1]
    start = np.full(n_experts, 1.0 / n_experts)

    def objective(weights: np.ndarray) -> float:
        prediction = experts @ weights
        return float(np.mean(_huber_numpy(prediction - target)))

    result = minimize(
        objective, start, method="SLSQP", bounds=[(0.0, 1.0)] * n_experts,
        constraints={"type": "eq", "fun": lambda value: float(value.sum() - 1.0)},
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"Simplex expert optimization failed: {result.message}")
    weights = np.clip(np.asarray(result.x, dtype=float), 0.0, 1.0)
    weights /= weights.sum()
    return weights


def _expert_matrix(predictions: dict[str, np.ndarray], expert_set: Sequence[str]) -> np.ndarray:
    return np.column_stack([predictions[name] for name in expert_set])


def _candidate_grid(config: dict[str, Any]) -> list[SearchConfig]:
    model = config["model"]
    return [
        SearchConfig(float(delta), bool(balance), float(penalty))
        for delta in model["max_delta_candidates"]
        for balance in model["source_balance_candidates"]
        for penalty in model["residual_penalty_candidates"]
    ]


def _load_primary_samples(
    project_root: Path,
    config: dict[str, Any],
) -> tuple[V4Tables, pd.DataFrame, pd.DataFrame]:
    data_dir = project_root / config["project"]["output_data_dir"]
    tables = V4Tables.load(data_dir)
    samples = pd.read_csv(data_dir / "samples_v4.csv", encoding="utf-8-sig")
    audit = pd.read_csv(project_root / config["project"]["audit_file"], encoding="utf-8-sig")
    audit_fields = audit[[
        "sample_id", "fit_status", "boundary_fit", "nnls_residual_norm",
        "n_unique_liquids", "nnls_dispersion_mj_m2", "nnls_polar_mj_m2",
        "nnls_physical_prediction_deg",
    ]].copy()
    samples = samples.drop(columns=[
        "nnls_dispersion_mj_m2", "nnls_polar_mj_m2", "nnls_physical_prediction_deg"
    ], errors="ignore").merge(audit_fields, on="sample_id", how="left", validate="one_to_one")
    eligibility = tables.measurements[["measurement_id", "target_eligible"]]
    samples = samples.merge(
        eligibility, left_on="target_measurement_id", right_on="measurement_id",
        how="left", validate="many_to_one",
    ).drop(columns=["measurement_id"])
    probe = samples.loc[
        (samples["prediction_mode"] == "probe_assisted")
        & (samples["v4_split"] != "excluded_review")
        & (samples["target_eligible"] == "yes")
    ].copy()
    reasons: list[str] = []
    for row in probe.itertuples(index=False):
        target = str(row.target_liquid_id)
        probe_liquids = {value for value in str(row.probe_liquid_ids).split(";") if value and value != "nan"}
        if target in probe_liquids or str(row.target_liquid_removed).casefold() != "yes":
            raise ValueError(f"Target liquid leakage detected in {row.sample_id}")
        if int(row.n_unique_liquids) < int(config["sample"]["minimum_unique_probes"]):
            reasons.append("fewer_than_two_unique_probes")
        elif str(row.fit_status) not in set(config["sample"]["feasible_fit_status"]):
            reasons.append(str(row.fit_status))
        elif not np.isfinite([
            row.nnls_dispersion_mj_m2, row.nnls_polar_mj_m2,
            row.nnls_physical_prediction_deg, row.nnls_residual_norm,
        ]).all():
            reasons.append("nonfinite_physics_summary")
        elif min(float(row.nnls_dispersion_mj_m2), float(row.nnls_polar_mj_m2)) < 0.0:
            reasons.append("negative_nnls_component")
        else:
            reasons.append("")
    probe["v4_1_exclusion_reason"] = reasons
    probe["v4_1_primary_eligible"] = np.where(probe["v4_1_exclusion_reason"] == "", "yes", "no")
    primary = probe.loc[probe["v4_1_primary_eligible"] == "yes"].reset_index(drop=True)
    allowed_splits = set(config["sample"]["development_splits"] + config["sample"]["confirmation_splits"])
    primary = primary.loc[primary["v4_split"].isin(allowed_splits)].reset_index(drop=True)
    return tables, primary, probe


@dataclass
class InnerCache:
    outer_fold: int
    outer_train: pd.DataFrame
    outer_holdout: pd.DataFrame
    base_oof: dict[str, np.ndarray]
    neural_oof: dict[str, np.ndarray]
    best_epochs: dict[str, list[int]]


def _inner_oof_cache(
    tables: V4Tables,
    outer_fold: int,
    outer_train: pd.DataFrame,
    outer_holdout: pd.DataFrame,
    candidates: Sequence[SearchConfig],
    config: dict[str, Any],
    device: torch.device,
) -> InnerCache:
    allowed = set(config["sample"]["development_splits"])
    observed = set(outer_train["v4_split"]) | set(outer_holdout["v4_split"])
    if not observed <= allowed:
        raise ValueError(f"External split reached model selection: {sorted(observed - allowed)}")
    groups = outer_train["source_group_id"].astype(str).to_numpy()
    n_folds = min(int(config["validation"]["inner_folds"]), len(np.unique(groups)))
    if n_folds < 2:
        raise RuntimeError("Nested validation requires at least two inner source groups")
    base_oof = {
        "physics": np.full(len(outer_train), np.nan),
        "xgboost": np.full(len(outer_train), np.nan),
        "random_forest": np.full(len(outer_train), np.nan),
    }
    neural_oof = {candidate.key: np.full(len(outer_train), np.nan) for candidate in candidates}
    best_epochs = {candidate.key: [] for candidate in candidates}
    seeds = [int(seed) for seed in config["model"]["search_seeds"]]
    splitter = GroupKFold(n_splits=n_folds)
    for inner_fold, (fit_index, holdout_index) in enumerate(splitter.split(outer_train, groups=groups)):
        fit_samples = outer_train.iloc[fit_index].reset_index(drop=True)
        holdout_samples = outer_train.iloc[holdout_index].reset_index(drop=True)
        preprocessor = V41Preprocessor().fit(tables, fit_samples)
        fit_data = preprocessor.transform(tables, fit_samples)
        holdout_data = preprocessor.transform(tables, holdout_samples)
        base_oof["physics"][holdout_index] = holdout_data.physics_angle
        tree_models = _fit_tree_models(
            fit_data.tabular, fit_data.target_angle, fit_data.physics_angle, seeds
        )
        tree_predictions = _predict_tree_models(
            tree_models, holdout_data.tabular, holdout_data.physics_angle
        )
        for name in ["xgboost", "random_forest"]:
            base_oof[name][holdout_index] = tree_predictions[name]
        for candidate in candidates:
            predictions: list[np.ndarray] = []
            for seed in seeds:
                model, best_epoch, _ = train_compact(
                    fit_data, holdout_data, preprocessor, candidate, config, seed,
                    device, int(config["model"]["search_epochs"]),
                    int(config["model"]["search_patience"]),
                )
                predictions.append(predict_compact(model, holdout_data, device).neural)
                best_epochs[candidate.key].append(best_epoch)
            neural_oof[candidate.key][holdout_index] = np.mean(predictions, axis=0)
        print(
            f"v4.1 outer {outer_fold}: completed inner fold {inner_fold + 1}/{n_folds}",
            flush=True,
        )
    arrays = list(base_oof.values()) + list(neural_oof.values())
    if any(np.isnan(values).any() for values in arrays):
        raise RuntimeError(f"Missing inner OOF predictions in outer fold {outer_fold}")
    return InnerCache(
        outer_fold=outer_fold, outer_train=outer_train, outer_holdout=outer_holdout,
        base_oof=base_oof, neural_oof=neural_oof, best_epochs=best_epochs,
    )


def _score_inner_cache(
    cache: InnerCache,
    candidates: Sequence[SearchConfig],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    target = cache.outer_train["target_contact_angle_deg"].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        predictions = dict(cache.base_oof)
        predictions["neural"] = cache.neural_oof[candidate.key]
        for expert_set in config["model"]["expert_sets"]:
            matrix = _expert_matrix(predictions, expert_set)
            weights = fit_simplex_weights(matrix, target)
            fusion = matrix @ weights
            rows.append({
                "outer_fold": cache.outer_fold,
                "config_key": candidate.key,
                "max_delta_cos": candidate.max_delta_cos,
                "source_balance": candidate.source_balance,
                "residual_penalty": candidate.residual_penalty,
                "expert_set": "+".join(expert_set),
                "n_experts": len(expert_set),
                "inner_oof_mae_deg": float(mean_absolute_error(target, fusion)),
                "inner_oof_source_median_mae_deg": float(pd.DataFrame({
                    "source": cache.outer_train["source_group_id"].astype(str),
                    "ae": np.abs(target - fusion),
                }).groupby("source")["ae"].mean().median()),
                "weights_json": json.dumps(dict(zip(expert_set, weights)), sort_keys=True),
                "median_best_epoch": int(np.median(cache.best_epochs[candidate.key])),
            })
    return rows


def _select_row(rows: pd.DataFrame, tolerance: float) -> pd.Series:
    best = float(rows["inner_oof_mae_deg"].min())
    competitive = rows.loc[rows["inner_oof_mae_deg"] <= best + tolerance].copy()
    return competitive.sort_values([
        "n_experts", "inner_oof_source_median_mae_deg", "inner_oof_mae_deg", "config_key"
    ]).iloc[0]


def _raw_ood_distance(train: np.ndarray, values: np.ndarray, neighbors: int = 5) -> np.ndarray:
    count = min(neighbors, len(train))
    model = NearestNeighbors(n_neighbors=count).fit(train)
    return model.kneighbors(values, return_distance=True)[0][:, -1]


def _fold_prediction(
    tables: V4Tables,
    cache: InnerCache,
    candidate: SearchConfig,
    expert_set: Sequence[str],
    config: dict[str, Any],
    device: torch.device,
    seeds: Sequence[int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    preprocessor = V41Preprocessor().fit(tables, cache.outer_train)
    train_data = preprocessor.transform(tables, cache.outer_train)
    holdout_data = preprocessor.transform(tables, cache.outer_holdout)
    fixed_epochs = max(10, int(np.median(cache.best_epochs[candidate.key])))
    neural_predictions: list[CompactPrediction] = []
    models: list[PhysicsSummaryResidualExpert] = []
    for seed in seeds:
        model, _, _ = train_compact(
            train_data, None, preprocessor, candidate, config, int(seed), device,
            int(config["model"]["search_epochs"]), int(config["model"]["search_patience"]),
            fixed_epochs=fixed_epochs,
        )
        models.append(model)
        neural_predictions.append(predict_compact(model, holdout_data, device))
    tree_models = _fit_tree_models(
        train_data.tabular, train_data.target_angle, train_data.physics_angle, seeds
    )
    tree_predictions = _predict_tree_models(
        tree_models, holdout_data.tabular, holdout_data.physics_angle
    )
    predictions = {
        "physics": holdout_data.physics_angle,
        "neural": np.mean([prediction.neural for prediction in neural_predictions], axis=0),
        **tree_predictions,
    }
    inner_predictions = dict(cache.base_oof)
    inner_predictions["neural"] = cache.neural_oof[candidate.key]
    inner_matrix = _expert_matrix(inner_predictions, expert_set)
    inner_target = cache.outer_train["target_contact_angle_deg"].to_numpy(dtype=float)
    weights = fit_simplex_weights(inner_matrix, inner_target)
    fusion = _expert_matrix(predictions, expert_set) @ weights
    seed_fusions: list[np.ndarray] = []
    for seed_index in range(len(seeds)):
        per_seed = {
            "physics": holdout_data.physics_angle,
            "neural": neural_predictions[seed_index].neural,
            "xgboost": _predict_tree_member(
                "xgboost", tree_models["xgboost"][seed_index],
                holdout_data.tabular, holdout_data.physics_angle,
            ),
            "random_forest": _predict_tree_member(
                "random_forest", tree_models["random_forest"][seed_index],
                holdout_data.tabular, holdout_data.physics_angle,
            ),
        }
        seed_fusions.append(_expert_matrix(per_seed, expert_set) @ weights)
    frame = pd.DataFrame({
        "sample_id": holdout_data.sample_ids,
        "record_id": holdout_data.record_ids,
        "source_group_id": holdout_data.source_group_ids,
        "surface_group_id": holdout_data.surface_group_ids,
        "target_liquid_id": holdout_data.target_liquid_ids,
        "solid_family": holdout_data.solid_families,
        "v4_split": holdout_data.splits,
        "n_probes": holdout_data.n_probes,
        "outer_fold": cache.outer_fold,
        "theta_observed_deg": holdout_data.target_angle,
        "theta_pred_deg": fusion,
        "physics_prediction": predictions["physics"],
        "neural_prediction": predictions["neural"],
        "xgboost_prediction": predictions["xgboost"],
        "random_forest_prediction": predictions["random_forest"],
        "xgboost_direct_prediction": predictions["xgboost_direct"],
        "random_forest_direct_prediction": predictions["random_forest_direct"],
        "ensemble_std_deg": np.std(np.stack(seed_fusions), axis=0),
        "aleatoric_sigma_deg": np.mean([prediction.sigma for prediction in neural_predictions], axis=0),
        "residual_shift_cosine": np.mean([prediction.residual for prediction in neural_predictions], axis=0),
        "raw_ood_distance": _raw_ood_distance(train_data.ood_features, holdout_data.ood_features),
        "unknown_category": np.where(holdout_data.unknown_category, "yes", "no"),
        "config_key": candidate.key,
        "expert_set": "+".join(expert_set),
        "fixed_epochs": fixed_epochs,
    })
    for name in ["physics", "neural", "xgboost", "random_forest"]:
        frame[f"weight_{name}"] = weights[list(expert_set).index(name)] if name in expert_set else 0.0
    return frame, {
        "preprocessor": preprocessor, "models": models, "tree_models": tree_models,
        "weights": weights, "fixed_epochs": fixed_epochs,
    }


def _select_consensus(
    selected_rows: list[pd.Series],
    all_rows: pd.DataFrame,
) -> tuple[str, str]:
    pairs = [(str(row["config_key"]), str(row["expert_set"])) for row in selected_rows]
    counts = Counter(pairs)
    most = max(counts.values())
    candidates = [pair for pair, count in counts.items() if count == most]
    if len(candidates) == 1:
        return candidates[0]
    ranked = []
    for config_key, expert_set in candidates:
        subset = all_rows.loc[
            (all_rows["config_key"] == config_key) & (all_rows["expert_set"] == expert_set)
        ]
        ranked.append((float(subset["inner_oof_mae_deg"].mean()), len(expert_set.split("+")), config_key, expert_set))
    _, _, config_key, expert_set = sorted(ranked)[0]
    return config_key, expert_set


def _crossvalidated_conformal(
    nested: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[str, AdaptiveConformal, pd.DataFrame, np.ndarray, np.ndarray]:
    level = float(config["uncertainty"]["confidence_level"])
    floor = float(config["uncertainty"]["adaptive_scale_floor_deg"])
    methods = ["absolute", "ensemble", "adaptive"]
    rows: list[dict[str, Any]] = []
    interval_by_method: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    predictive = np.sqrt(
        np.square(nested["ensemble_std_deg"].to_numpy(dtype=float))
        + np.square(nested["aleatoric_sigma_deg"].to_numpy(dtype=float))
    )
    for method in methods:
        if method == "absolute":
            scale = np.ones(len(nested))
        elif method == "ensemble":
            scale = np.maximum(nested["ensemble_std_deg"].to_numpy(dtype=float), floor)
        else:
            scale = np.maximum(predictive, floor)
        lower = np.zeros(len(nested), dtype=float)
        upper = np.zeros(len(nested), dtype=float)
        for fold in sorted(nested["outer_fold"].unique()):
            calibration_mask = (nested["outer_fold"] != fold).to_numpy()
            holdout_mask = (nested["outer_fold"] == fold).to_numpy()
            conformal = AdaptiveConformal(level=level).fit(
                nested.loc[calibration_mask, "theta_observed_deg"].to_numpy(),
                nested.loc[calibration_mask, "theta_pred_deg"].to_numpy(),
                scale[calibration_mask],
            )
            low, high = conformal.interval(
                nested.loc[holdout_mask, "theta_pred_deg"].to_numpy(), scale[holdout_mask]
            )
            lower[holdout_mask], upper[holdout_mask] = low, high
        metric = regression_metrics(
            nested["theta_observed_deg"].to_numpy(), nested["theta_pred_deg"].to_numpy(), lower, upper
        )
        rows.append({"method": method, **metric})
        interval_by_method[method] = (lower, upper)
    comparison = pd.DataFrame(rows)
    lower_target = float(config["uncertainty"]["coverage_lower"])
    upper_target = float(config["uncertainty"]["coverage_upper"])
    valid = comparison.loc[
        comparison["interval_coverage"].between(lower_target, upper_target, inclusive="both")
    ]
    if len(valid):
        selected_method = str(valid.sort_values("mean_interval_width_deg").iloc[0]["method"])
    else:
        comparison["coverage_distance"] = np.abs(comparison["interval_coverage"] - level)
        selected_method = str(comparison.sort_values([
            "coverage_distance", "mean_interval_width_deg"
        ]).iloc[0]["method"])
    if selected_method == "absolute":
        final_scale = np.ones(len(nested))
    elif selected_method == "ensemble":
        final_scale = np.maximum(nested["ensemble_std_deg"].to_numpy(dtype=float), floor)
    else:
        final_scale = np.maximum(predictive, floor)
    final = AdaptiveConformal(level=level).fit(
        nested["theta_observed_deg"].to_numpy(), nested["theta_pred_deg"].to_numpy(), final_scale
    )
    lower, upper = interval_by_method[selected_method]
    return selected_method, final, comparison, lower, upper


def _apply_conformal(
    conformal: AdaptiveConformal,
    method: str,
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictive = np.sqrt(
        np.square(frame["ensemble_std_deg"].to_numpy(dtype=float))
        + np.square(frame["aleatoric_sigma_deg"].to_numpy(dtype=float))
    )
    if method == "ensemble":
        scale = np.maximum(frame["ensemble_std_deg"].to_numpy(dtype=float), float(
            config["uncertainty"]["adaptive_scale_floor_deg"]
        ))
    elif method == "adaptive":
        scale = np.maximum(predictive, float(config["uncertainty"]["adaptive_scale_floor_deg"]))
    else:
        scale = np.ones(len(frame))
    lower, upper = conformal.interval(frame["theta_pred_deg"].to_numpy(), scale)
    return lower, upper, predictive


def _fit_final_model(
    tables: V4Tables,
    primary: pd.DataFrame,
    development: pd.DataFrame,
    candidate: SearchConfig,
    expert_set: Sequence[str],
    gate_weights: np.ndarray,
    fixed_epochs: int,
    config: dict[str, Any],
    device: torch.device,
    output_dir: Path,
) -> tuple[pd.DataFrame, V41Preprocessor, dict[str, Any]]:
    preprocessor = V41Preprocessor().fit(tables, development)
    development_data = preprocessor.transform(tables, development)
    all_data = preprocessor.transform(tables, primary)
    seeds = [int(seed) for seed in config["model"]["final_seeds"]]
    neural_predictions: list[CompactPrediction] = []
    neural_models: list[PhysicsSummaryResidualExpert] = []
    histories: list[dict[str, Any]] = []
    for seed in seeds:
        model, _, history = train_compact(
            development_data, None, preprocessor, candidate, config, seed, device,
            int(config["model"]["final_epochs"]), int(config["model"]["final_patience"]),
            fixed_epochs=fixed_epochs,
        )
        neural_models.append(model)
        neural_predictions.append(predict_compact(model, all_data, device))
        for row in history:
            histories.append({"seed": seed, **row})
        torch.save({
            "model_state": model.state_dict(), "seed": seed,
            "model_version": config["project"]["model_version"],
            "search_config": asdict(candidate), "fixed_epochs": fixed_epochs,
        }, output_dir / f"neural_seed_{seed}.pt")
    tree_models = _fit_tree_models(
        development_data.tabular, development_data.target_angle,
        development_data.physics_angle, seeds,
    )
    for name, models in tree_models.items():
        for seed, model in zip(seeds, models):
            joblib.dump(model, output_dir / f"{name}_seed_{seed}.joblib")
    tree_predictions = _predict_tree_models(tree_models, all_data.tabular, all_data.physics_angle)
    predictions = {
        "physics": all_data.physics_angle,
        "neural": np.mean([item.neural for item in neural_predictions], axis=0),
        **tree_predictions,
    }
    fusion = _expert_matrix(predictions, expert_set) @ gate_weights
    seed_fusions: list[np.ndarray] = []
    for seed_index in range(len(seeds)):
        per_seed = {
            "physics": all_data.physics_angle,
            "neural": neural_predictions[seed_index].neural,
            "xgboost": _predict_tree_member(
                "xgboost", tree_models["xgboost"][seed_index], all_data.tabular, all_data.physics_angle
            ),
            "random_forest": _predict_tree_member(
                "random_forest", tree_models["random_forest"][seed_index],
                all_data.tabular, all_data.physics_angle,
            ),
        }
        seed_fusions.append(_expert_matrix(per_seed, expert_set) @ gate_weights)
    frame = pd.DataFrame({
        "sample_id": all_data.sample_ids,
        "record_id": all_data.record_ids,
        "source_group_id": all_data.source_group_ids,
        "surface_group_id": all_data.surface_group_ids,
        "target_liquid_id": all_data.target_liquid_ids,
        "solid_family": all_data.solid_families,
        "prediction_mode": "probe_assisted",
        "v4_split": all_data.splits,
        "n_probes": all_data.n_probes,
        "theta_observed_deg": all_data.target_angle,
        "theta_pred_deg": fusion,
        "physics_prediction": predictions["physics"],
        "neural_prediction": predictions["neural"],
        "xgboost_prediction": predictions["xgboost"],
        "random_forest_prediction": predictions["random_forest"],
        "xgboost_direct_prediction": predictions["xgboost_direct"],
        "random_forest_direct_prediction": predictions["random_forest_direct"],
        "ensemble_std_deg": np.std(np.stack(seed_fusions), axis=0),
        "aleatoric_sigma_deg": np.mean([item.sigma for item in neural_predictions], axis=0),
        "residual_shift_cosine": np.mean([item.residual for item in neural_predictions], axis=0),
        "unknown_category": np.where(all_data.unknown_category, "yes", "no"),
        "model_version": config["project"]["model_version"],
    })
    for name in ["physics", "neural", "xgboost", "random_forest"]:
        frame[f"weight_{name}"] = gate_weights[list(expert_set).index(name)] if name in expert_set else 0.0
    tree_weight = frame["weight_xgboost"] + frame["weight_random_forest"]
    frame["tree_prediction"] = np.where(
        tree_weight > 0,
        (
            frame["weight_xgboost"] * frame["xgboost_prediction"]
            + frame["weight_random_forest"] * frame["random_forest_prediction"]
        ) / np.maximum(tree_weight, 1e-12),
        0.5 * (frame["xgboost_prediction"] + frame["random_forest_prediction"]),
    )
    preprocessor.save(output_dir / "feature_preprocessor_v4_1.json")
    pd.DataFrame(histories).to_csv(output_dir / "training_log_v4_1.csv", index=False, encoding="utf-8-sig")
    return frame, preprocessor, {
        "development_data": development_data, "all_data": all_data,
        "neural_models": neural_models, "tree_models": tree_models,
    }


def _metric_tables(
    nested: pd.DataFrame,
    final: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    model_columns = {
        "physics_expert": "physics_prediction",
        "neural_expert": "neural_prediction",
        "physics_residual_xgboost_expert": "xgboost_prediction",
        "physics_residual_random_forest_expert": "random_forest_prediction",
        "direct_xgboost_baseline": "xgboost_direct_prediction",
        "direct_random_forest_baseline": "random_forest_direct_prediction",
        "ls_psrmoe_fusion": "theta_pred_deg",
    }
    for split, frame in [("nested_source_cv", nested)]:
        for model, column in model_columns.items():
            kwargs = {}
            if model == "ls_psrmoe_fusion":
                kwargs = {
                    "lower": frame["interval_lower_deg"].to_numpy(),
                    "upper": frame["interval_upper_deg"].to_numpy(),
                    "abstain": frame["abstain_flag"].eq("yes").to_numpy(),
                }
            rows.append({"split": split, "prediction_mode": "probe_assisted", "model": model,
                         **regression_metrics(frame["theta_observed_deg"], frame[column], **kwargs)})
    for split in EXTERNAL_SPLITS:
        frame = final.loc[final["v4_split"] == split]
        if not len(frame):
            continue
        for model, column in model_columns.items():
            kwargs = {}
            if model == "ls_psrmoe_fusion":
                kwargs = {
                    "lower": frame["interval_lower_deg"].to_numpy(),
                    "upper": frame["interval_upper_deg"].to_numpy(),
                    "abstain": frame["abstain_flag"].eq("yes").to_numpy(),
                }
            rows.append({"split": split, "prediction_mode": "probe_assisted", "model": model,
                         **regression_metrics(frame["theta_observed_deg"], frame[column], **kwargs)})

    strata: list[dict[str, Any]] = []
    confirmation = final.loc[final["v4_split"].isin(EXTERNAL_SPLITS)].copy()
    confirmation["angle_bin"] = pd.cut(
        confirmation["theta_observed_deg"], bins=[-0.1, 60, 120, 150, 180.1],
        labels=["hydrophilic", "neutral", "hydrophobic", "superhydrophobic"],
    ).astype(str)
    confirmation["probe_count_bin"] = np.where(
        confirmation["n_probes"] >= 3, "3_or_more", "2"
    )
    for field in ["target_liquid_id", "angle_bin", "solid_family", "probe_count_bin", "source_group_id"]:
        for value, group in confirmation.groupby(field, dropna=False):
            if len(group) < 2:
                continue
            strata.append({
                "stratum_type": field, "stratum": str(value),
                **regression_metrics(group["theta_observed_deg"], group["theta_pred_deg"]),
            })
    return pd.DataFrame(rows), pd.DataFrame(strata)


def _bootstrap_tables(
    nested: pd.DataFrame,
    final: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    comparisons = {
        "physics": "physics_prediction", "neural": "neural_prediction",
        "physics_residual_xgboost": "xgboost_prediction",
        "physics_residual_random_forest": "random_forest_prediction",
        "direct_xgboost": "xgboost_direct_prediction",
    }
    frames = [("nested_source_cv", nested)] + [
        (split, final.loc[final["v4_split"] == split]) for split in EXTERNAL_SPLITS
    ]
    for split, frame in frames:
        if len(frame) < 2:
            continue
        for cluster in ["surface_group_id", "source_group_id"]:
            for name, column in comparisons.items():
                result = paired_cluster_bootstrap(
                    frame["theta_observed_deg"].to_numpy(), frame["theta_pred_deg"].to_numpy(),
                    frame[column].to_numpy(), frame[cluster].astype(str).to_numpy(),
                    int(config["statistics"]["bootstrap_resamples"]),
                    int(config["project"]["seed"]),
                )
                rows.append({
                    "split": split, "comparison": f"fusion_minus_{name}",
                    "cluster": cluster, **result,
                })
    return pd.DataFrame(rows)


def _ablation_table(
    search_results: pd.DataFrame,
    final: pd.DataFrame,
    no_summary_prediction: np.ndarray,
    expert_set: Sequence[str],
    weights: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for balance, group in search_results.groupby("source_balance"):
        best_by_fold = group.groupby("outer_fold")["inner_oof_mae_deg"].min()
        rows.append({
            "ablation": "source_balance", "setting": str(bool(balance)).lower(),
            "split": "nested_inner_oof", "n": len(best_by_fold),
            "mae_deg": float(best_by_fold.mean()),
        })
    for split in EXTERNAL_SPLITS:
        mask = (final["v4_split"] == split).to_numpy()
        if not mask.any():
            continue
        rows.append({
            "ablation": "physics_summary", "setting": "removed_from_neural",
            "split": split,
            **regression_metrics(final.loc[mask, "theta_observed_deg"], no_summary_prediction[mask]),
        })
        expert_predictions = {
            "physics": final.loc[mask, "physics_prediction"].to_numpy(),
            "neural": final.loc[mask, "neural_prediction"].to_numpy(),
            "xgboost": final.loc[mask, "xgboost_prediction"].to_numpy(),
            "random_forest": final.loc[mask, "random_forest_prediction"].to_numpy(),
        }
        for removed in expert_set:
            keep = [index for index, name in enumerate(expert_set) if name != removed]
            if not keep:
                continue
            reduced_weights = weights[keep] / weights[keep].sum()
            reduced = _expert_matrix(expert_predictions, [expert_set[index] for index in keep]) @ reduced_weights
            rows.append({
                "ablation": "expert", "setting": f"remove_{removed}", "split": split,
                **regression_metrics(final.loc[mask, "theta_observed_deg"], reduced),
            })
    return pd.DataFrame(rows)


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_acceptance_report(
    nested: pd.DataFrame,
    final: pd.DataFrame,
    metrics: pd.DataFrame,
    config: dict[str, Any],
    final_weights: dict[str, float],
) -> dict[str, Any]:
    nested_physics = float(np.mean(np.abs(
        nested["theta_observed_deg"] - nested["physics_prediction"]
    )))
    nested_residual_tree = float(np.mean(np.abs(
        nested["theta_observed_deg"] - nested["xgboost_prediction"]
    )))
    nested_tree_bootstrap = {
        cluster: paired_cluster_bootstrap(
            nested["theta_observed_deg"].to_numpy(),
            nested["xgboost_prediction"].to_numpy(),
            nested["physics_prediction"].to_numpy(),
            nested[cluster].astype(str).to_numpy(),
            int(config["statistics"]["bootstrap_resamples"]),
            int(config["project"]["seed"]),
        ) for cluster in ["surface_group_id", "source_group_id"]
    }
    prospective = final.loc[final["v4_split"] == "prospective_open_external"]
    prospective_fusion = float(np.mean(np.abs(
        prospective["theta_observed_deg"] - prospective["theta_pred_deg"]
    )))
    prospective_tree = float(np.mean(np.abs(
        prospective["theta_observed_deg"] - prospective["xgboost_prediction"]
    )))
    uncertainty = metrics.loc[
        (metrics["split"] == "nested_source_cv")
        & (metrics["model"] == "ls_psrmoe_fusion")
    ].iloc[0]
    checks = {
        "nested_residual_tree_improves_physics_by_at_least_1_deg": (
            nested_residual_tree <= nested_physics - 1.0
        ),
        "nested_surface_bootstrap_ci_below_zero": (
            nested_tree_bootstrap["surface_group_id"]["ci95_upper_deg"] < 0.0
        ),
        "external_fusion_mae_at_most_16_deg": prospective_fusion <= 16.0,
        "nested_interval_coverage_85_to_95_percent": (
            0.85 <= float(uncertainty["interval_coverage"]) <= 0.95
        ),
        "nested_mean_interval_width_at_most_80_deg": (
            float(uncertainty["mean_interval_width_deg"]) <= 80.0
        ),
        "retain_80_percent_and_reduce_mae": (
            float(uncertainty["retained_fraction"]) >= 0.80
            and float(uncertainty["retained_mae_deg"]) < float(uncertainty["mae_deg"])
        ),
        "neural_expert_has_nonzero_final_weight": final_weights.get("neural", 0.0) > 1e-3,
    }
    full_acceptance = all(checks.values())
    return {
        "status": "full_acceptance" if full_acceptance else "conditional_paper_ready",
        "recommended_primary_model": "physics_residual_xgboost_expert",
        "recommended_secondary_model": "ls_psrmoe_fusion",
        "reason": (
            "The physics-residual tree is the strongest leakage-safe cross-source model; "
            "the neural branch was automatically pruned and must not be claimed as a performance driver."
        ),
        "nested_source_cv": {
            "n": len(nested), "n_sources": int(nested["source_group_id"].nunique()),
            "physics_mae_deg": nested_physics,
            "physics_residual_xgboost_mae_deg": nested_residual_tree,
            "improvement_deg": nested_physics - nested_residual_tree,
            "bootstrap": nested_tree_bootstrap,
        },
        "fixed_external_confirmation": {
            "n": len(prospective), "fusion_mae_deg": prospective_fusion,
            "physics_residual_xgboost_mae_deg": prospective_tree,
        },
        "uncertainty_and_rejection": {
            "coverage": float(uncertainty["interval_coverage"]),
            "mean_width_deg": float(uncertainty["mean_interval_width_deg"]),
            "retained_fraction": float(uncertainty["retained_fraction"]),
            "retained_mae_deg": float(uncertainty["retained_mae_deg"]),
        },
        "final_weights": final_weights,
        "checks": checks,
        "full_acceptance": full_acceptance,
    }


def run_v41(config_path: Path) -> dict[str, Any]:
    started = time.time()
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = project_root / config["project"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    tables, primary, eligibility_audit = _load_primary_samples(project_root, config)
    eligibility_audit.to_csv(
        output_dir / "primary_eligibility_audit_v4_1.csv", index=False, encoding="utf-8-sig"
    )
    development = primary.loc[
        primary["v4_split"].isin(config["sample"]["development_splits"])
    ].reset_index(drop=True)
    development_groups = development["source_group_id"].astype(str).to_numpy()
    external_mask = primary["v4_split"].isin(config["sample"]["confirmation_splits"])
    if set(development["source_group_id"]) & set(primary.loc[external_mask, "source_group_id"]):
        raise RuntimeError("Development and external confirmation sources overlap")
    outer_folds = min(int(config["validation"]["outer_folds"]), len(np.unique(development_groups)))
    if outer_folds < 2:
        raise RuntimeError("At least two development sources are required")
    candidates = _candidate_grid(config)
    candidate_by_key = {candidate.key: candidate for candidate in candidates}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outer_splitter = GroupKFold(n_splits=outer_folds)
    caches: list[InnerCache] = []
    search_rows: list[dict[str, Any]] = []
    selected_rows: list[pd.Series] = []
    for outer_fold, (train_index, holdout_index) in enumerate(
        outer_splitter.split(development, groups=development_groups)
    ):
        cache = _inner_oof_cache(
            tables, outer_fold,
            development.iloc[train_index].reset_index(drop=True),
            development.iloc[holdout_index].reset_index(drop=True),
            candidates, config, device,
        )
        caches.append(cache)
        fold_rows = pd.DataFrame(_score_inner_cache(cache, candidates, config))
        search_rows.extend(fold_rows.to_dict(orient="records"))
        selected_rows.append(_select_row(fold_rows, float(config["model"]["simpler_tie_deg"])))
        print(
            f"v4.1 outer {outer_fold}: selected {selected_rows[-1]['config_key']} "
            f"with {selected_rows[-1]['expert_set']}", flush=True,
        )
    search_results = pd.DataFrame(search_rows)
    search_results.to_csv(output_dir / "nested_cv_results.csv", index=False, encoding="utf-8-sig")
    consensus_key, consensus_expert_text = _select_consensus(selected_rows, search_results)
    consensus_candidate = candidate_by_key[consensus_key]
    consensus_experts = consensus_expert_text.split("+")

    nested_frames: list[pd.DataFrame] = []
    consensus_frames: list[pd.DataFrame] = []
    search_seeds = [int(seed) for seed in config["model"]["search_seeds"]]
    for cache, selected in zip(caches, selected_rows):
        selected_candidate = candidate_by_key[str(selected["config_key"])]
        selected_experts = str(selected["expert_set"]).split("+")
        selected_frame, _ = _fold_prediction(
            tables, cache, selected_candidate, selected_experts, config, device, search_seeds
        )
        nested_frames.append(selected_frame)
        if selected_candidate.key == consensus_key and selected_experts == consensus_experts:
            consensus_frames.append(selected_frame.copy())
        else:
            consensus_frame, _ = _fold_prediction(
                tables, cache, consensus_candidate, consensus_experts, config, device, search_seeds
            )
            consensus_frames.append(consensus_frame)
    nested = pd.concat(nested_frames, ignore_index=True).sort_values("sample_id").reset_index(drop=True)
    consensus_nested = pd.concat(consensus_frames, ignore_index=True).sort_values("sample_id").reset_index(drop=True)
    if nested["sample_id"].duplicated().any() or len(nested) != len(development):
        raise RuntimeError("Nested OOF predictions are incomplete or duplicated")
    nested.to_csv(output_dir / "nested_oof_predictions_v4_1.csv", index=False, encoding="utf-8-sig")
    consensus_nested.to_csv(
        output_dir / "consensus_oof_predictions_v4_1.csv", index=False, encoding="utf-8-sig"
    )

    consensus_prediction_map = {
        "physics": consensus_nested["physics_prediction"].to_numpy(),
        "neural": consensus_nested["neural_prediction"].to_numpy(),
        "xgboost": consensus_nested["xgboost_prediction"].to_numpy(),
        "random_forest": consensus_nested["random_forest_prediction"].to_numpy(),
    }
    final_gate_weights = fit_simplex_weights(
        _expert_matrix(consensus_prediction_map, consensus_experts),
        consensus_nested["theta_observed_deg"].to_numpy(),
    )
    fixed_epochs = int(np.median(consensus_nested["fixed_epochs"]))
    final, preprocessor, artifacts = _fit_final_model(
        tables, primary, development, consensus_candidate, consensus_experts,
        final_gate_weights, fixed_epochs, config, device, output_dir,
    )

    method, conformal, conformal_comparison, nested_lower, nested_upper = _crossvalidated_conformal(
        nested, config
    )
    nested["interval_lower_deg"] = nested_lower
    nested["interval_upper_deg"] = nested_upper
    nested_width = nested_upper - nested_lower
    abstention_score_threshold = float(np.quantile(
        nested["ensemble_std_deg"], float(config["uncertainty"]["abstention_score_quantile"])
    ))
    ood_threshold = max(float(np.quantile(
        nested["raw_ood_distance"], float(config["uncertainty"]["ood_quantile"])
    )), 1e-8)
    nested["ood_score"] = nested["raw_ood_distance"] / ood_threshold
    nested_abstain = nested["ensemble_std_deg"] > abstention_score_threshold
    nested["abstain_flag"] = np.where(nested_abstain, "yes", "no")

    development_data = artifacts["development_data"]
    all_data = artifacts["all_data"]
    ood_neighbors = min(5, len(development_data))
    ood_model = NearestNeighbors(n_neighbors=ood_neighbors).fit(development_data.ood_features)
    raw_ood = ood_model.kneighbors(all_data.ood_features, return_distance=True)[0][:, -1]
    final["ood_score"] = raw_ood / ood_threshold
    final_lower, final_upper, predictive_scale = _apply_conformal(conformal, method, final, config)
    final["interval_lower_deg"] = final_lower
    final["interval_upper_deg"] = final_upper
    final["confidence_level"] = float(config["uncertainty"]["confidence_level"])
    final_width = final_upper - final_lower
    final_abstain = final["ensemble_std_deg"] > abstention_score_threshold
    final["abstain_flag"] = np.where(final_abstain, "yes", "no")
    final["risk_level"] = np.where(
        final_abstain, "high",
        np.where(
            final["unknown_category"].eq("yes") | (final["ood_score"] > 0.75)
            | (final["ensemble_std_deg"] > 0.75 * abstention_score_threshold), "medium", "low"
        ),
    )
    final["predictive_scale_deg"] = predictive_scale
    final["expert_weights"] = json.dumps(
        dict(zip(consensus_experts, final_gate_weights)), sort_keys=True
    )
    prediction_path = output_dir / "predictions_v4_1.csv"
    final.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    joblib.dump({"model": ood_model, "threshold": ood_threshold}, output_dir / "ood_detector_v4_1.joblib")

    no_summary_development = development_data.without_physics_summary()
    no_summary_all = all_data.without_physics_summary()
    no_summary_neural: list[np.ndarray] = []
    for seed in [int(value) for value in config["model"]["final_seeds"]]:
        model, _, _ = train_compact(
            no_summary_development, None, preprocessor, consensus_candidate, config,
            seed, device, int(config["model"]["final_epochs"]),
            int(config["model"]["final_patience"]), fixed_epochs=fixed_epochs,
        )
        no_summary_neural.append(predict_compact(model, no_summary_all, device).neural)
    no_summary_map = {
        "physics": final["physics_prediction"].to_numpy(),
        "neural": np.mean(no_summary_neural, axis=0),
        "xgboost": final["xgboost_prediction"].to_numpy(),
        "random_forest": final["random_forest_prediction"].to_numpy(),
    }
    no_summary_fusion = _expert_matrix(no_summary_map, consensus_experts) @ final_gate_weights

    metrics, stratified = _metric_tables(nested, final)
    bootstrap = _bootstrap_tables(nested, final, config)
    ablations = _ablation_table(
        search_results, final, no_summary_fusion, consensus_experts, final_gate_weights
    )
    metrics_path = output_dir / "metrics_v4_1.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    stratified.to_csv(output_dir / "stratified_metrics_v4_1.csv", index=False, encoding="utf-8-sig")
    bootstrap.to_csv(output_dir / "bootstrap_v4_1.csv", index=False, encoding="utf-8-sig")
    ablations.to_csv(output_dir / "ablation_v4_1.csv", index=False, encoding="utf-8-sig")
    acceptance = build_acceptance_report(
        nested, final, metrics, config, dict(zip(consensus_experts, final_gate_weights))
    )
    (output_dir / "acceptance_report_v4_1.json").write_text(
        json.dumps(acceptance, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )

    calibration = {
        "confidence_level": float(config["uncertainty"]["confidence_level"]),
        "selected_method": method,
        "conformal_quantile": conformal.quantile,
        "crossvalidated_comparison": conformal_comparison.to_dict(orient="records"),
        "ood_raw_distance_threshold": ood_threshold,
        "abstention_score": "ensemble_std_deg",
        "abstention_score_threshold_deg": abstention_score_threshold,
    }
    (output_dir / "calibration_v4_1.json").write_text(
        json.dumps(calibration, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    selection = {
        "model_version": config["project"]["model_version"],
        "primary_mode": "probe_assisted",
        "selection_evidence": "four-fold outer, three-fold inner source-group CV",
        "fold_selections": [row.to_dict() for row in selected_rows],
        "consensus_search_config": asdict(consensus_candidate),
        "consensus_expert_set": consensus_experts,
        "consensus_weights": dict(zip(consensus_experts, final_gate_weights)),
        "fixed_epochs": fixed_epochs,
        "external_sets_used_for_selection": False,
        "external_label": "fixed cross-source external confirmation",
    }
    (output_dir / "selected_config.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    with pd.ExcelWriter(output_dir / "model_comparison_v4_1.xlsx", engine="openpyxl") as writer:
        metrics.to_excel(writer, sheet_name="Main Metrics", index=False)
        bootstrap.to_excel(writer, sheet_name="Bootstrap", index=False)
        ablations.to_excel(writer, sheet_name="Ablations", index=False)
        stratified.to_excel(writer, sheet_name="Stratified", index=False)
        search_results.to_excel(writer, sheet_name="Nested CV Search", index=False)
        conformal_comparison.to_excel(writer, sheet_name="Calibration", index=False)

    legacy_predictions = project_root / "outputs" / "experiments" / "predictions_v4.csv"
    if legacy_predictions.exists():
        frozen = pd.read_csv(legacy_predictions, encoding="utf-8-sig")
        frozen = frozen.loc[frozen["prediction_mode"] == "zero_shot"].copy()
        frozen["status"] = "frozen_v4_0_auxiliary_reference"
        frozen.to_csv(output_dir / "zero_shot_reference_v4_0.csv", index=False, encoding="utf-8-sig")

    input_files = [
        project_root / config["project"]["output_data_dir"] / name for name in [
            "sources_v4.csv", "surfaces_v4.csv", "liquids_v4.csv",
            "measurements_v4.csv", "splits_v4.csv", "samples_v4.csv",
        ]
    ]
    manifest = {
        "status": "complete",
        "model_version": config["project"]["model_version"],
        "device": str(device),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "runtime_seconds": time.time() - started,
        "n_primary_samples": len(primary),
        "n_development_samples": len(development),
        "n_development_sources": int(development["source_group_id"].nunique()),
        "n_confirmation_samples": int(external_mask.sum()),
        "selected_config": asdict(consensus_candidate),
        "selected_experts": consensus_experts,
        "selected_weights": dict(zip(consensus_experts, final_gate_weights)),
        "input_hashes": {path.name: _hash_file(path) for path in input_files},
        "output_hashes": {
            prediction_path.name: _hash_file(prediction_path),
            metrics_path.name: _hash_file(metrics_path),
            "selected_config.json": _hash_file(output_dir / "selected_config.json"),
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    return manifest
