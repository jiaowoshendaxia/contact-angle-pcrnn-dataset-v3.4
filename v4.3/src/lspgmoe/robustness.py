"""Source-level robustness analyses for the locked LS-PSRMoE v4.1 task."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from .metrics import paired_cluster_bootstrap, regression_metrics
from .v41 import V41Preprocessor, _json_default, _load_primary_samples


MODEL_COLUMNS = {
    "nnls_physics": "physics_prediction",
    "global_mean_residual": "global_mean_residual_prediction",
    "liquid_mean_residual": "liquid_mean_residual_prediction",
    "ridge_residual": "ridge_residual_prediction",
    "direct_xgboost": "direct_xgboost_prediction",
    "physics_residual_xgboost": "xgboost_residual_prediction",
    "source_weighted_physics_residual_xgboost": "weighted_xgboost_residual_prediction",
}


def source_balance_weights(source_ids: Sequence[str]) -> np.ndarray:
    """Return mean-one 1/sqrt(source-size) sample weights."""
    source_ids = np.asarray(source_ids, dtype=str)
    if source_ids.size == 0:
        raise ValueError("Source-balanced weights require at least one sample")
    unique, counts = np.unique(source_ids, return_counts=True)
    count_map = dict(zip(unique, counts))
    weights = np.asarray([1.0 / np.sqrt(count_map[value]) for value in source_ids], dtype=float)
    return weights / weights.mean()


def residual_location_predictions(
    train_target: np.ndarray,
    train_physics: np.ndarray,
    train_liquids: Sequence[str],
    holdout_liquids: Sequence[str],
    liquid_shrinkage: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit global and target-liquid residual location baselines on training rows only."""
    train_target = np.asarray(train_target, dtype=float)
    train_physics = np.asarray(train_physics, dtype=float)
    train_liquids = np.asarray(train_liquids, dtype=str)
    holdout_liquids = np.asarray(holdout_liquids, dtype=str)
    if len(train_target) != len(train_physics) or len(train_target) != len(train_liquids):
        raise ValueError("Training target, physics, and liquid arrays must have equal length")
    if liquid_shrinkage < 0.0:
        raise ValueError("Liquid residual shrinkage must be nonnegative")
    residual = train_target - train_physics
    global_mean = float(np.mean(residual))
    liquid_stats = (
        pd.DataFrame({"liquid": train_liquids, "residual": residual})
        .groupby("liquid")["residual"]
        .agg(["mean", "count"])
    )
    liquid_residual = []
    liquid_counts = []
    for liquid in holdout_liquids:
        if liquid not in liquid_stats.index:
            liquid_residual.append(global_mean)
            liquid_counts.append(0)
            continue
        row = liquid_stats.loc[liquid]
        count = int(row["count"])
        estimate = (
            count * float(row["mean"]) + liquid_shrinkage * global_mean
        ) / (count + liquid_shrinkage)
        liquid_residual.append(estimate)
        liquid_counts.append(count)
    return (
        np.full(len(holdout_liquids), global_mean, dtype=float),
        np.asarray(liquid_residual, dtype=float),
        np.asarray(liquid_counts, dtype=int),
    )


def build_loso_splits(
    development: pd.DataFrame,
    allowed_splits: Sequence[str],
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Build deterministic leave-one-source-out indices and reject external rows."""
    observed = set(development["v4_split"].astype(str))
    if not observed <= set(allowed_splits):
        raise ValueError(f"External split reached LOSO validation: {sorted(observed - set(allowed_splits))}")
    sources = np.sort(development["source_group_id"].astype(str).unique())
    if len(sources) < 3:
        raise ValueError("LOSO validation requires at least three independent sources")
    output = []
    groups = development["source_group_id"].astype(str).to_numpy()
    for source in sources:
        holdout = np.flatnonzero(groups == source)
        training = np.flatnonzero(groups != source)
        if not len(holdout) or not len(training):
            raise RuntimeError(f"Invalid LOSO fold for source {source}")
        if set(groups[training]) & set(groups[holdout]):
            raise RuntimeError(f"Source leakage in LOSO fold {source}")
        output.append((source, training, holdout))
    return output


def _xgb_options(seed: int, config: dict[str, Any]) -> dict[str, Any]:
    options = config["robustness"]["xgboost"]
    return {
        "n_estimators": int(options["n_estimators"]),
        "max_depth": int(options["max_depth"]),
        "learning_rate": float(options["learning_rate"]),
        "subsample": float(options["subsample"]),
        "colsample_bytree": float(options["colsample_bytree"]),
        "min_child_weight": float(options["min_child_weight"]),
        "reg_lambda": float(options["reg_lambda"]),
        "objective": "reg:squarederror",
        "random_state": int(seed),
        "n_jobs": 1,
        "tree_method": "hist",
    }


def _xgb_ensemble_prediction(
    train_features: np.ndarray,
    train_target: np.ndarray,
    holdout_features: np.ndarray,
    seeds: Sequence[int],
    config: dict[str, Any],
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    from xgboost import XGBRegressor

    predictions = []
    for seed in seeds:
        model = XGBRegressor(**_xgb_options(int(seed), config))
        model.fit(train_features, train_target, sample_weight=sample_weight)
        predictions.append(np.asarray(model.predict(holdout_features), dtype=float))
    return np.mean(predictions, axis=0)


def clip_angle(physics: np.ndarray, residual: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(physics, dtype=float) + np.asarray(residual, dtype=float), 0.0, 180.0)


def _select_ridge_alpha(
    tables: Any,
    train_samples: pd.DataFrame,
    alphas: Sequence[float],
) -> float:
    """Select Ridge strength using inner source-group CV within an outer LOSO fold."""
    groups = train_samples["source_group_id"].astype(str).to_numpy()
    n_splits = min(3, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("Ridge selection requires at least two inner sources")
    scores = {float(alpha): [] for alpha in alphas}
    for fit_index, validation_index in GroupKFold(n_splits=n_splits).split(
        train_samples, groups=groups
    ):
        fit_samples = train_samples.iloc[fit_index].reset_index(drop=True)
        validation_samples = train_samples.iloc[validation_index].reset_index(drop=True)
        preprocessor = V41Preprocessor().fit(tables, fit_samples)
        fit = preprocessor.transform(tables, fit_samples)
        validation = preprocessor.transform(tables, validation_samples)
        fit_residual = fit.target_angle.astype(float) - fit.physics_angle.astype(float)
        for alpha in scores:
            model = Ridge(alpha=alpha).fit(fit.tabular, fit_residual)
            prediction = clip_angle(validation.physics_angle, model.predict(validation.tabular))
            scores[alpha].append(float(np.mean(np.abs(validation.target_angle - prediction))))
    ranked = sorted(
        ((float(np.mean(values)), -alpha, alpha) for alpha, values in scores.items())
    )
    return float(ranked[0][2])


def run_loso(
    tables: Any,
    development: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Run frozen-parameter LOSO analyses without reading confirmation cohorts."""
    seeds = [int(value) for value in config["robustness"]["seeds"]]
    liquid_shrinkage = float(config["robustness"]["liquid_residual_shrinkage"])
    ridge_alphas = [float(value) for value in config["robustness"]["ridge_alphas"]]
    folds = build_loso_splits(development, config["sample"]["development_splits"])
    frames = []
    for fold_index, (held_out_source, train_index, holdout_index) in enumerate(folds):
        train_samples = development.iloc[train_index].reset_index(drop=True)
        holdout_samples = development.iloc[holdout_index].reset_index(drop=True)
        preprocessor = V41Preprocessor().fit(tables, train_samples)
        train = preprocessor.transform(tables, train_samples)
        holdout = preprocessor.transform(tables, holdout_samples)
        train_residual = train.target_angle.astype(float) - train.physics_angle.astype(float)

        global_residual, liquid_residual, liquid_count = residual_location_predictions(
            train.target_angle,
            train.physics_angle,
            train.target_liquid_ids,
            holdout.target_liquid_ids,
            liquid_shrinkage,
        )
        ridge_alpha = _select_ridge_alpha(tables, train_samples, ridge_alphas)
        ridge = Ridge(alpha=ridge_alpha).fit(train.tabular, train_residual)
        ridge_residual = np.asarray(ridge.predict(holdout.tabular), dtype=float)
        xgb_residual = _xgb_ensemble_prediction(
            train.tabular, train_residual, holdout.tabular, seeds, config
        )
        weighted_xgb_residual = _xgb_ensemble_prediction(
            train.tabular,
            train_residual,
            holdout.tabular,
            seeds,
            config,
            sample_weight=source_balance_weights(train.source_group_ids),
        )
        direct_xgb = _xgb_ensemble_prediction(
            train.tabular, train.target_angle, holdout.tabular, seeds, config
        )
        frame = pd.DataFrame({
            "sample_id": holdout.sample_ids,
            "record_id": holdout.record_ids,
            "held_out_source": held_out_source,
            "source_group_id": holdout.source_group_ids,
            "surface_group_id": holdout.surface_group_ids,
            "target_liquid_id": holdout.target_liquid_ids,
            "solid_family": holdout.solid_families,
            "v4_split": holdout.splits,
            "n_probes": holdout.n_probes,
            "loso_fold": fold_index,
            "theta_observed_deg": holdout.target_angle,
            "physics_prediction": holdout.physics_angle,
            "global_mean_residual_prediction": clip_angle(holdout.physics_angle, global_residual),
            "liquid_mean_residual_prediction": clip_angle(holdout.physics_angle, liquid_residual),
            "ridge_residual_prediction": clip_angle(holdout.physics_angle, ridge_residual),
            "direct_xgboost_prediction": np.clip(direct_xgb, 0.0, 180.0),
            "xgboost_residual_prediction": clip_angle(holdout.physics_angle, xgb_residual),
            "weighted_xgboost_residual_prediction": clip_angle(
                holdout.physics_angle, weighted_xgb_residual
            ),
            "liquid_training_count": liquid_count,
            "ridge_alpha": ridge_alpha,
            "unknown_category": np.where(holdout.unknown_category, "yes", "no"),
        })
        frames.append(frame)
        print(
            f"LOSO {fold_index + 1}/{len(folds)}: held out {held_out_source} "
            f"({len(frame)} samples)",
            flush=True,
        )
    predictions = pd.concat(frames, ignore_index=True).sort_values("sample_id").reset_index(drop=True)
    if predictions["sample_id"].duplicated().any() or len(predictions) != len(development):
        raise RuntimeError("LOSO predictions are incomplete or duplicated")
    return predictions


def _metric_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_rows = []
    source_rows = []
    observed = predictions["theta_observed_deg"].to_numpy(dtype=float)
    for model, column in MODEL_COLUMNS.items():
        overall_rows.append({"evaluation": "development_loso", "model": model, **regression_metrics(
            observed, predictions[column].to_numpy(dtype=float)
        )})
        for source, frame in predictions.groupby("source_group_id", sort=True):
            source_rows.append({
                "source_group_id": source,
                "model": model,
                **regression_metrics(
                    frame["theta_observed_deg"].to_numpy(dtype=float),
                    frame[column].to_numpy(dtype=float),
                ),
            })
    return pd.DataFrame(overall_rows), pd.DataFrame(source_rows)


def run_fixed_confirmation(
    tables: Any,
    primary: pd.DataFrame,
    development: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Fit on all development sources and run one fixed confirmation inference."""
    confirmation = primary.loc[
        primary["v4_split"].isin(config["sample"]["confirmation_splits"])
    ].reset_index(drop=True)
    if set(development["source_group_id"]) & set(confirmation["source_group_id"]):
        raise RuntimeError("Development and fixed confirmation sources overlap")
    preprocessor = V41Preprocessor().fit(tables, development)
    train = preprocessor.transform(tables, development)
    holdout = preprocessor.transform(tables, confirmation)
    train_residual = train.target_angle.astype(float) - train.physics_angle.astype(float)
    seeds = [int(value) for value in config["robustness"]["seeds"]]
    unweighted_residual = _xgb_ensemble_prediction(
        train.tabular, train_residual, holdout.tabular, seeds, config
    )
    weighted_residual = _xgb_ensemble_prediction(
        train.tabular,
        train_residual,
        holdout.tabular,
        seeds,
        config,
        sample_weight=source_balance_weights(train.source_group_ids),
    )
    direct = _xgb_ensemble_prediction(
        train.tabular, train.target_angle, holdout.tabular, seeds, config
    )
    return pd.DataFrame({
        "sample_id": holdout.sample_ids,
        "record_id": holdout.record_ids,
        "source_group_id": holdout.source_group_ids,
        "surface_group_id": holdout.surface_group_ids,
        "target_liquid_id": holdout.target_liquid_ids,
        "solid_family": holdout.solid_families,
        "v4_split": holdout.splits,
        "n_probes": holdout.n_probes,
        "theta_observed_deg": holdout.target_angle,
        "physics_prediction": holdout.physics_angle,
        "direct_xgboost_prediction": np.clip(direct, 0.0, 180.0),
        "xgboost_residual_prediction": clip_angle(holdout.physics_angle, unweighted_residual),
        "weighted_xgboost_residual_prediction": clip_angle(
            holdout.physics_angle, weighted_residual
        ),
        "unknown_category": np.where(holdout.unknown_category, "yes", "no"),
    })


def _confirmation_tables(
    predictions: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_columns = {
        "nnls_physics": "physics_prediction",
        "direct_xgboost": "direct_xgboost_prediction",
        "physics_residual_xgboost": "xgboost_residual_prediction",
        "source_weighted_physics_residual_xgboost": "weighted_xgboost_residual_prediction",
    }
    metric_rows = []
    bootstrap_rows = []
    for split, frame in predictions.groupby("v4_split", sort=True):
        observed = frame["theta_observed_deg"].to_numpy(dtype=float)
        for model, column in model_columns.items():
            metric_rows.append({
                "split": split,
                "model": model,
                **regression_metrics(observed, frame[column].to_numpy(dtype=float)),
            })
        comparisons = [
            ("source_weighted_physics_residual_xgboost", "nnls_physics"),
            ("source_weighted_physics_residual_xgboost", "physics_residual_xgboost"),
        ]
        for model_a, model_b in comparisons:
            for cluster in ["source_group_id", "surface_group_id"]:
                bootstrap_rows.append({
                    "split": split,
                    "model_a": model_a,
                    "model_b": model_b,
                    "cluster": cluster,
                    **paired_cluster_bootstrap(
                        observed,
                        frame[model_columns[model_a]].to_numpy(dtype=float),
                        frame[model_columns[model_b]].to_numpy(dtype=float),
                        frame[cluster].astype(str).to_numpy(),
                        int(config["robustness"]["bootstrap_resamples"]),
                        int(config["project"]["seed"]) + 2,
                    ),
                })
    return pd.DataFrame(metric_rows), pd.DataFrame(bootstrap_rows)


def _bootstrap_table(predictions: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    observed = predictions["theta_observed_deg"].to_numpy(dtype=float)
    physics = predictions[MODEL_COLUMNS["nnls_physics"]].to_numpy(dtype=float)
    for model, column in MODEL_COLUMNS.items():
        if model == "nnls_physics":
            continue
        for cluster in ["source_group_id", "surface_group_id"]:
            rows.append({
                "model_a": model,
                "model_b": "nnls_physics",
                "cluster": cluster,
                **paired_cluster_bootstrap(
                    observed,
                    predictions[column].to_numpy(dtype=float),
                    physics,
                    predictions[cluster].astype(str).to_numpy(),
                    int(config["robustness"]["bootstrap_resamples"]),
                    int(config["project"]["seed"]),
                ),
            })
    for cluster in ["source_group_id", "surface_group_id"]:
        rows.append({
            "model_a": "source_weighted_physics_residual_xgboost",
            "model_b": "physics_residual_xgboost",
            "cluster": cluster,
            **paired_cluster_bootstrap(
                observed,
                predictions[MODEL_COLUMNS["source_weighted_physics_residual_xgboost"]].to_numpy(dtype=float),
                predictions[MODEL_COLUMNS["physics_residual_xgboost"]].to_numpy(dtype=float),
                predictions[cluster].astype(str).to_numpy(),
                int(config["robustness"]["bootstrap_resamples"]),
                int(config["project"]["seed"]) + 1,
            ),
        })
    return pd.DataFrame(rows)


def _source_figure(source_metrics: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    pivot = source_metrics.pivot(index="source_group_id", columns="model", values="mae_deg")
    unweighted_delta = (
        pivot["physics_residual_xgboost"] - pivot["nnls_physics"]
    ).sort_values()
    weighted_delta = (
        pivot.loc[unweighted_delta.index, "source_weighted_physics_residual_xgboost"]
        - pivot.loc[unweighted_delta.index, "nnls_physics"]
    )
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    positions = np.arange(len(unweighted_delta))
    axis.barh(
        positions - 0.18, unweighted_delta.values, height=0.34,
        color="#6f879c", label="Unweighted residual XGBoost",
    )
    axis.barh(
        positions + 0.18, weighted_delta.values, height=0.34,
        color="#2f7f6f", label="Source-weighted residual XGBoost",
    )
    axis.axvline(0.0, color="#333333", linewidth=1.0)
    axis.set_yticks(positions, labels=unweighted_delta.index)
    axis.set_xlabel("MAE difference from NNLS physics (degrees)")
    axis.set_ylabel("Held-out source")
    axis.legend(
        frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _probe_text(sample: pd.Series, tables: Any) -> str:
    measurements = tables.measurements.set_index("measurement_id", drop=False)
    liquids = tables.liquids.set_index("liquid_id", drop=False)
    parts = []
    for measurement_id in str(sample["probe_measurement_ids"]).split(";"):
        if not measurement_id or measurement_id == "nan" or measurement_id not in measurements.index:
            continue
        measurement = measurements.loc[measurement_id]
        liquid_id = str(measurement["liquid_id"])
        liquid_name = str(liquids.loc[liquid_id, "liquid_name"]) if liquid_id in liquids.index else liquid_id
        parts.append(f"{liquid_name}: {float(measurement['contact_angle_deg']):.1f} deg")
    return "; ".join(parts)


def _application_cases(
    project_root: Path,
    tables: Any,
    primary: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    prediction_path = project_root / config["robustness"]["final_prediction_file"]
    if not prediction_path.exists():
        return pd.DataFrame()
    predictions = pd.read_csv(prediction_path, encoding="utf-8-sig")
    candidates = predictions.loc[
        predictions["v4_split"] == "prospective_open_external"
    ].copy()
    if candidates.empty:
        return pd.DataFrame()
    sample_fields = primary[[
        "sample_id", "probe_measurement_ids", "nnls_dispersion_mj_m2",
        "nnls_polar_mj_m2", "fit_status",
    ]]
    candidates = candidates.merge(sample_fields, on="sample_id", how="left", validate="one_to_one")
    candidates["uncertainty_rank"] = candidates["ensemble_std_deg"].rank(pct=True, method="average")
    targets = [("low", 0.10), ("medium", 0.50), ("high", 0.90)]
    selected = []
    used_surfaces: set[str] = set()
    primary_lookup = primary.set_index("sample_id", drop=False)
    liquid_lookup = tables.liquids.set_index("liquid_id", drop=False)
    for label, quantile in targets:
        ordered = candidates.assign(
            distance=(candidates["uncertainty_rank"] - quantile).abs()
        ).sort_values(["distance", "sample_id"])
        row = next(
            item for item in ordered.itertuples(index=False)
            if str(item.surface_group_id) not in used_surfaces
        )
        used_surfaces.add(str(row.surface_group_id))
        sample = primary_lookup.loc[str(row.sample_id)]
        target_liquid = (
            str(liquid_lookup.loc[str(row.target_liquid_id), "liquid_name"])
            if str(row.target_liquid_id) in liquid_lookup.index else str(row.target_liquid_id)
        )
        selected.append({
            "uncertainty_example": label,
            "sample_id": row.sample_id,
            "source_group_id": row.source_group_id,
            "surface_group_id": row.surface_group_id,
            "solid_family": row.solid_family,
            "target_liquid": target_liquid,
            "non_target_probes": _probe_text(sample, tables),
            "nnls_dispersion_mj_m2": row.nnls_dispersion_mj_m2,
            "nnls_polar_mj_m2": row.nnls_polar_mj_m2,
            "fit_status": row.fit_status,
            "theta_observed_deg": row.theta_observed_deg,
            "physics_prediction_deg": row.physics_prediction,
            "residual_xgboost_prediction_deg": row.xgboost_prediction,
            "fusion_prediction_deg": row.theta_pred_deg,
            "interval_lower_deg": row.interval_lower_deg,
            "interval_upper_deg": row.interval_upper_deg,
            "risk_level": row.risk_level,
            "abstain_flag": row.abstain_flag,
            "ensemble_std_deg": row.ensemble_std_deg,
        })
    return pd.DataFrame(selected)


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_model_strengthening(config_path: Path) -> dict[str, Any]:
    started = time.time()
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = project_root / config["project"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    tables, primary, _ = _load_primary_samples(project_root, config)
    development = primary.loc[
        primary["v4_split"].isin(config["sample"]["development_splits"])
    ].reset_index(drop=True)

    predictions = run_loso(tables, development, config)
    metrics, source_metrics = _metric_tables(predictions)
    bootstrap = _bootstrap_table(predictions, config)
    confirmation_predictions = run_fixed_confirmation(tables, primary, development, config)
    confirmation_metrics, confirmation_bootstrap = _confirmation_tables(
        confirmation_predictions, config
    )
    cases = _application_cases(project_root, tables, primary, config)
    cases = cases.merge(
        confirmation_predictions[["sample_id", "weighted_xgboost_residual_prediction"]],
        on="sample_id", how="left", validate="one_to_one",
    )

    paths = {
        "predictions": output_dir / "loso_predictions_v4_2.csv",
        "metrics": output_dir / "loso_metrics_v4_2.csv",
        "source_metrics": output_dir / "loso_source_metrics_v4_2.csv",
        "bootstrap": output_dir / "loso_bootstrap_v4_2.csv",
        "cases": output_dir / "application_case_studies_v4_2.csv",
        "confirmation_predictions": output_dir / "confirmation_predictions_v4_2.csv",
        "confirmation_metrics": output_dir / "confirmation_metrics_v4_2.csv",
        "confirmation_bootstrap": output_dir / "confirmation_bootstrap_v4_2.csv",
        "figure": output_dir / "Figure_LOSO_source_transfer_v4_2.png",
    }
    predictions.to_csv(paths["predictions"], index=False, encoding="utf-8-sig")
    metrics.to_csv(paths["metrics"], index=False, encoding="utf-8-sig")
    source_metrics.to_csv(paths["source_metrics"], index=False, encoding="utf-8-sig")
    bootstrap.to_csv(paths["bootstrap"], index=False, encoding="utf-8-sig")
    cases.to_csv(paths["cases"], index=False, encoding="utf-8-sig")
    confirmation_predictions.to_csv(
        paths["confirmation_predictions"], index=False, encoding="utf-8-sig"
    )
    confirmation_metrics.to_csv(paths["confirmation_metrics"], index=False, encoding="utf-8-sig")
    confirmation_bootstrap.to_csv(
        paths["confirmation_bootstrap"], index=False, encoding="utf-8-sig"
    )
    _source_figure(source_metrics, paths["figure"])
    with pd.ExcelWriter(output_dir / "model_strengthening_v4_2.xlsx", engine="openpyxl") as writer:
        metrics.to_excel(writer, sheet_name="Overall_LOSO", index=False)
        source_metrics.to_excel(writer, sheet_name="Source_LOSO", index=False)
        bootstrap.to_excel(writer, sheet_name="Bootstrap", index=False)
        confirmation_metrics.to_excel(writer, sheet_name="Fixed_Confirmation", index=False)
        confirmation_bootstrap.to_excel(writer, sheet_name="Confirmation_Bootstrap", index=False)
        cases.to_excel(writer, sheet_name="Application_Cases", index=False)

    metric_map = metrics.set_index("model")["mae_deg"].to_dict()
    source_win = source_metrics.pivot(
        index="source_group_id", columns="model", values="mae_deg"
    )
    primary_bootstrap = bootstrap.loc[
        (bootstrap["model_a"] == "physics_residual_xgboost")
        & (bootstrap["cluster"] == "source_group_id")
    ].iloc[0]
    weighted_bootstrap = bootstrap.loc[
        (bootstrap["model_a"] == "source_weighted_physics_residual_xgboost")
        & (bootstrap["model_b"] == "physics_residual_xgboost")
        & (bootstrap["cluster"] == "source_group_id")
    ].iloc[0]
    confirmation_map = {
        split: {
            row.model: float(row.mae_deg)
            for row in frame.itertuples(index=False)
        }
        for split, frame in confirmation_metrics.groupby("split", sort=True)
    }
    open_confirmation = confirmation_map["prospective_open_external"]
    weighting_degrades_open = (
        open_confirmation["source_weighted_physics_residual_xgboost"]
        > open_confirmation["physics_residual_xgboost"]
    )
    report = {
        "status": "complete",
        "model_version": config["project"]["model_version"],
        "n_samples": int(len(predictions)),
        "n_sources": int(predictions["source_group_id"].nunique()),
        "external_sets_used_for_model_fitting_or_selection": False,
        "external_sets_used_once_for_fixed_confirmation": True,
        "external_case_table_is_presentation_only": True,
        "frozen_primary_model": "physics_residual_xgboost",
        "final_model_decision": "retain_unweighted_physics_residual_xgboost",
        "decision_reason": (
            "Source weighting improved development LOSO but degraded all three fixed "
            "confirmation cohorts, including the open external set. No further weighting "
            "variants may be tuned on these confirmation labels."
        ),
        "mae_deg": {key: float(value) for key, value in metric_map.items()},
        "source_win_fraction_vs_physics": float(np.mean(
            source_win["physics_residual_xgboost"] < source_win["nnls_physics"]
        )),
        "weighted_source_win_fraction_vs_physics": float(np.mean(
            source_win["source_weighted_physics_residual_xgboost"] < source_win["nnls_physics"]
        )),
        "weighted_source_win_fraction_vs_unweighted_xgboost": float(np.mean(
            source_win["source_weighted_physics_residual_xgboost"]
            < source_win["physics_residual_xgboost"]
        )),
        "source_cluster_bootstrap_vs_physics": primary_bootstrap.to_dict(),
        "source_cluster_bootstrap_weighted_vs_unweighted": weighted_bootstrap.to_dict(),
        "fixed_confirmation_mae_deg": confirmation_map,
        "checks": {
            "residual_xgboost_beats_nnls": (
                metric_map["physics_residual_xgboost"] < metric_map["nnls_physics"]
            ),
            "residual_xgboost_beats_direct_xgboost": (
                metric_map["physics_residual_xgboost"] < metric_map["direct_xgboost"]
            ),
            "residual_xgboost_beats_ridge_residual": (
                metric_map["physics_residual_xgboost"] < metric_map["ridge_residual"]
            ),
            "source_cluster_ci_below_zero": float(primary_bootstrap["ci95_upper_deg"]) < 0.0,
            "source_weighting_improves_unweighted_xgboost": (
                metric_map["source_weighted_physics_residual_xgboost"]
                < metric_map["physics_residual_xgboost"]
            ),
            "source_weighting_does_not_degrade_open_external": not weighting_degrades_open,
        },
        "no_further_weight_tuning_on_confirmation_sets": True,
    }
    report_path = output_dir / "model_strengthening_report_v4_2.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8"
    )
    manifest = {
        "status": "complete",
        "runtime_seconds": time.time() - started,
        "config": str(config_path.relative_to(project_root)).replace("\\", "/"),
        "development_only_for_loso": True,
        "input_hashes": {
            name: _hash_file(project_root / config["project"]["output_data_dir"] / name)
            for name in ["sources_v4.csv", "surfaces_v4.csv", "liquids_v4.csv", "measurements_v4.csv", "splits_v4.csv", "samples_v4.csv"]
        },
        "output_hashes": {
            path.name: _hash_file(path) for path in [
                paths["predictions"], paths["metrics"], paths["source_metrics"],
                paths["bootstrap"], paths["cases"], report_path,
                paths["confirmation_predictions"], paths["confirmation_metrics"],
                paths["confirmation_bootstrap"],
            ]
        },
    }
    (output_dir / "run_manifest_v4_2.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8"
    )
    return report
