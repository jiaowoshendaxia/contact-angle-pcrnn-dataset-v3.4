"""End-to-end dual-mode training, OOF gating, calibration, and result locking."""

from __future__ import annotations

import copy
import hashlib
import json
import platform
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from torch.nn import functional as F

from .data import V4Tables
from .features import EncodedSamples, FeaturePreprocessor
from .metrics import paired_cluster_bootstrap, regression_metrics
from .model import ExpertGate, PhysicsGuidedNeuralExpert
from .uncertainty import AdaptiveConformal, KNNOODDetector


@dataclass
class NeuralPrediction:
    physics: np.ndarray
    neural: np.ndarray
    cosine_physics: np.ndarray
    cosine_neural: np.ndarray
    residual_cosine: np.ndarray
    sfe_dispersion: np.ndarray
    sfe_polar: np.ndarray
    sigma_deg: np.ndarray


@dataclass
class TreeFit:
    model: Any
    backend: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _model_from_data(data: EncodedSamples, preprocessor: FeaturePreprocessor, config: dict[str, Any]) -> PhysicsGuidedNeuralExpert:
    model_config = config["model"]
    return PhysicsGuidedNeuralExpert(
        surface_numeric_dim=data.surface_numeric.shape[1],
        liquid_numeric_dim=data.target_liquid_numeric.shape[1],
        condition_numeric_dim=data.condition_numeric.shape[1],
        categorical_cardinalities=preprocessor.categorical_cardinalities,
        categorical_embedding_dim=int(model_config["categorical_embedding_dim"]),
        dropout=float(model_config["dropout"]),
        max_delta_cos=float(model_config["max_delta_cos"]),
        use_probe_encoder=bool(model_config.get("use_probe_encoder", True)),
        use_physics_decoder=bool(model_config.get("use_physics_decoder", True)),
    )


def _tensor(value: np.ndarray, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype, device=device)


def _forward(model: PhysicsGuidedNeuralExpert, data: EncodedSamples, indices: np.ndarray, device: torch.device):
    return model(
        _tensor(data.surface_numeric[indices], device),
        _tensor(data.categorical[indices], device, torch.long),
        _tensor(data.target_liquid_numeric[indices], device),
        _tensor(data.target_liquid_physical[indices], device),
        _tensor(data.condition_numeric[indices], device),
        _tensor(data.probes[indices], device),
        _tensor(data.probe_mask[indices], device, torch.bool),
        _tensor(data.independent_sfe[indices], device),
        _tensor(data.independent_sfe_mask[indices], device),
        _tensor(data.nnls_sfe[indices], device),
        _tensor(data.nnls_sfe_mask[indices], device),
    )


def _loss(output: Any, data: EncodedSamples, indices: np.ndarray, config: dict[str, Any], device: torch.device, phase: str) -> torch.Tensor:
    weights = config["loss"]
    target_angle = _tensor(data.target_angle[indices], device)
    target_cosine = _tensor(data.target_cosine[indices], device)
    sfe_target = _tensor(data.independent_sfe[indices], device)
    sfe_mask = _tensor(data.independent_sfe_mask[indices], device).bool()

    physical_reconstruction = F.huber_loss(output.theta_physics, target_angle, delta=5.0)
    physical_cosine = F.mse_loss(output.cosine_physics, target_cosine)
    if sfe_mask.any():
        predicted_sfe = torch.stack([output.sfe_dispersion, output.sfe_polar], dim=-1)
        sfe_loss = F.mse_loss(predicted_sfe[sfe_mask] / 100.0, sfe_target[sfe_mask] / 100.0)
    else:
        sfe_loss = torch.zeros((), device=device)
    if phase == "physics":
        return (
            float(weights["masked_probe"]) * physical_reconstruction
            + float(weights["cosine_mse"]) * physical_cosine
            + float(weights["independent_sfe"]) * sfe_loss
        )

    angle_loss = F.huber_loss(output.theta_neural, target_angle, delta=5.0)
    cosine_loss = F.mse_loss(output.cosine_neural, target_cosine)
    residual_penalty = torch.mean(output.residual_cosine.square())
    scaled_error = (target_angle - output.theta_neural) / 10.0
    gaussian_nll = 0.5 * torch.mean(torch.exp(-output.log_variance) * scaled_error.square() + output.log_variance)
    return (
        float(weights["angle_huber"]) * angle_loss
        + float(weights["cosine_mse"]) * cosine_loss
        + float(weights["masked_probe"]) * physical_reconstruction
        + float(weights["independent_sfe"]) * sfe_loss
        + float(weights["residual_penalty"]) * residual_penalty
        + float(weights["gaussian_nll"]) * gaussian_nll
    )


@torch.no_grad()
def predict_neural(model: PhysicsGuidedNeuralExpert, data: EncodedSamples, device: torch.device, batch_size: int = 256) -> NeuralPrediction:
    model.eval()
    names = [
        "physics", "neural", "cosine_physics", "cosine_neural", "residual_cosine",
        "sfe_dispersion", "sfe_polar", "sigma_deg",
    ]
    accumulated: dict[str, list[np.ndarray]] = {name: [] for name in names}
    for start in range(0, len(data), batch_size):
        indices = np.arange(start, min(start + batch_size, len(data)))
        output = _forward(model, data, indices, device)
        values = {
            "physics": output.theta_physics,
            "neural": output.theta_neural,
            "cosine_physics": output.cosine_physics,
            "cosine_neural": output.cosine_neural,
            "residual_cosine": output.residual_cosine,
            "sfe_dispersion": output.sfe_dispersion,
            "sfe_polar": output.sfe_polar,
            "sigma_deg": 10.0 * torch.exp(0.5 * output.log_variance),
        }
        for name, value in values.items():
            accumulated[name].append(value.detach().cpu().numpy())
    return NeuralPrediction(**{name: np.concatenate(values) for name, values in accumulated.items()})


def train_neural(
    train: EncodedSamples,
    validation: EncodedSamples,
    preprocessor: FeaturePreprocessor,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[PhysicsGuidedNeuralExpert, list[dict[str, float | int | str]]]:
    set_seed(seed)
    model = _model_from_data(train, preprocessor, config).to(device)
    model_config = config["model"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    epochs = int(model_config["epochs"])
    patience = int(model_config["patience"])
    batch_size = int(model_config["batch_size"])
    physics_epochs = max(1, epochs // 3) if bool(model_config.get("use_physics_decoder", True)) else 0
    best_state = copy.deepcopy(model.state_dict())
    best_validation = float("inf")
    stale = 0
    history: list[dict[str, float | int | str]] = []
    rng = np.random.default_rng(seed)

    for epoch in range(epochs):
        phase = "physics" if epoch < physics_epochs else "residual"
        if physics_epochs and epoch == physics_epochs:
            # Validation losses from the two stages are not on the same scale.
            best_validation = float("inf")
            stale = 0
        model.train()
        ordering = rng.permutation(len(train))
        train_losses: list[float] = []
        for start in range(0, len(train), batch_size):
            indices = ordering[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            output = _forward(model, train, indices, device)
            loss = _loss(output, train, indices, config, device, phase)
            if not torch.isfinite(loss):
                failing = ", ".join(train.sample_ids[index] for index in indices[:5])
                raise FloatingPointError(f"Non-finite training loss in phase {phase}; samples: {failing}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            indices = np.arange(len(validation))
            output = _forward(model, validation, indices, device)
            validation_loss = float(_loss(output, validation, indices, config, device, phase).cpu())
            if not np.isfinite(validation_loss):
                raise FloatingPointError(f"Non-finite validation loss in phase {phase}")
        history.append({
            "seed": seed, "epoch": epoch + 1, "phase": phase,
            "train_loss": float(np.mean(train_losses)), "validation_loss": validation_loss,
        })
        if validation_loss < best_validation - 1e-6:
            best_validation = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch >= physics_epochs and stale >= patience:
            break
    model.load_state_dict(best_state)
    return model, history


def fit_tree(features: np.ndarray, target: np.ndarray, seed: int) -> TreeFit:
    try:
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=500, max_depth=4, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.85,
            objective="reg:squarederror", random_state=seed,
            n_jobs=-1, tree_method="hist",
        )
        backend = "official_xgboost"
    except ImportError:
        model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.04, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=seed,
        )
        backend = "sklearn_hist_gradient_boosting_fallback"
    model.fit(features, target)
    return TreeFit(model=model, backend=backend)


def train_gate(
    experts: np.ndarray,
    context: np.ndarray,
    target: np.ndarray,
    seed: int,
    max_epochs: int = 1000,
    patience: int = 100,
) -> ExpertGate:
    set_seed(seed)
    gate = ExpertGate(context_dim=context.shape[1], hidden_dim=32, n_experts=experts.shape[1])
    optimizer = torch.optim.AdamW(gate.parameters(), lr=3e-3, weight_decay=1e-4)
    expert_tensor = torch.as_tensor(experts, dtype=torch.float32)
    context_tensor = torch.as_tensor(context, dtype=torch.float32)
    target_tensor = torch.as_tensor(np.array(target, dtype=np.float32, copy=True))
    best_state = copy.deepcopy(gate.state_dict())
    best_loss = float("inf")
    stale = 0
    for _ in range(max_epochs):
        optimizer.zero_grad(set_to_none=True)
        prediction, _ = gate(expert_tensor, context_tensor)
        loss = F.huber_loss(prediction, target_tensor, delta=5.0)
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        if value < best_loss - 1e-7:
            best_loss = value
            best_state = copy.deepcopy(gate.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    gate.load_state_dict(best_state)
    gate.eval()
    return gate


@torch.no_grad()
def predict_gate(gate: ExpertGate, experts: np.ndarray, context: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prediction, weights = gate(
        torch.as_tensor(experts, dtype=torch.float32),
        torch.as_tensor(context, dtype=torch.float32),
    )
    return prediction.numpy(), weights.numpy()


def _eligible_samples(tables: V4Tables, samples: pd.DataFrame) -> pd.DataFrame:
    eligibility = tables.measurements[["measurement_id", "target_eligible"]]
    output = samples.merge(
        eligibility, left_on="target_measurement_id", right_on="measurement_id",
        how="left", validate="many_to_one",
    )
    return output.loc[
        (output["target_eligible"] == "yes") & (output["v4_split"] != "excluded_review")
    ].drop(columns=["measurement_id"]).reset_index(drop=True)


def _oof_experts(
    tables: V4Tables,
    train_samples: pd.DataFrame,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    groups = train_samples["source_group_id"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    folds = min(int(config["model"].get("oof_folds", 5)), len(unique_groups))
    if folds < 2:
        raise RuntimeError("At least two training sources are required for OOF gating")
    oof = np.full((len(train_samples), 3), np.nan, dtype=np.float32)
    context = np.zeros((len(train_samples), 7), dtype=np.float32)
    sample_ids = train_samples["sample_id"].astype(str).tolist()
    backends: set[str] = set()
    splitter = GroupKFold(n_splits=folds)
    for fold, (fit_indices, holdout_indices) in enumerate(splitter.split(train_samples, groups=groups)):
        fit_samples = train_samples.iloc[fit_indices].reset_index(drop=True)
        holdout_samples = train_samples.iloc[holdout_indices].reset_index(drop=True)
        preprocessor = FeaturePreprocessor().fit(tables, fit_samples)
        fit_data = preprocessor.transform(tables, fit_samples)
        holdout_data = preprocessor.transform(tables, holdout_samples)
        model, _ = train_neural(
            fit_data, holdout_data, preprocessor, config, seed + fold, device
        )
        neural = predict_neural(model, holdout_data, device)
        tree = fit_tree(fit_data.tabular, fit_data.target_angle, seed + fold)
        tree_prediction = np.clip(tree.model.predict(holdout_data.tabular), 0.0, 180.0)
        backends.add(tree.backend)
        oof[holdout_indices] = np.column_stack([neural.physics, neural.neural, tree_prediction])
        context[holdout_indices] = holdout_data.gate_context
    if np.isnan(oof).any():
        raise RuntimeError("OOF expert matrix contains missing predictions")
    return oof, context, sample_ids, "+".join(sorted(backends))


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_experiment(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    data_dir = project_root / config["project"]["output_data_dir"]
    output_dir = project_root / config["project"]["output_dir"] / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = V4Tables.load(data_dir)
    samples = pd.read_csv(data_dir / "samples_v4.csv", encoding="utf-8-sig")
    samples = _eligible_samples(tables, samples)
    train_samples = samples.loc[samples["v4_split"] == "train"].reset_index(drop=True)
    validation_samples = samples.loc[samples["v4_split"] == "validation"].reset_index(drop=True)
    if not len(train_samples) or not len(validation_samples):
        raise RuntimeError("Both training and validation samples are required")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(seed) for seed in config["model"]["seeds"]]
    oof, oof_context, oof_ids, oof_tree_backend = _oof_experts(
        tables, train_samples, config, seeds[0], device
    )
    gate = train_gate(
        oof, oof_context, train_samples["target_contact_angle_deg"].to_numpy(), seeds[0],
        max_epochs=int(config["model"].get("gate_epochs", 1000)),
        patience=int(config["model"].get("gate_patience", 100)),
    )
    oof_fusion, oof_weights = predict_gate(gate, oof, oof_context)
    torch.save(
        {"model_state": gate.state_dict(), "context_dim": oof_context.shape[1], "n_experts": oof.shape[1]},
        output_dir / "expert_gate.pt",
    )
    pd.DataFrame({
        "sample_id": oof_ids,
        "physics_oof_deg": oof[:, 0], "neural_oof_deg": oof[:, 1], "tree_oof_deg": oof[:, 2],
        "fusion_oof_deg": oof_fusion,
        "weight_physics": oof_weights[:, 0], "weight_neural": oof_weights[:, 1],
        "weight_tree": oof_weights[:, 2],
    }).to_csv(output_dir / "oof_expert_predictions.csv", index=False, encoding="utf-8-sig")

    preprocessor = FeaturePreprocessor().fit(tables, train_samples)
    all_data = preprocessor.transform(tables, samples)
    split_array = np.asarray(all_data.splits)
    train_indices = np.flatnonzero(split_array == "train")
    validation_indices = np.flatnonzero(split_array == "validation")
    train_data = all_data.subset(train_indices)
    validation_data = all_data.subset(validation_indices)
    preprocessor.save(output_dir / "feature_preprocessor.json")

    neural_runs: list[NeuralPrediction] = []
    tree_runs: list[np.ndarray] = []
    tree_models: list[TreeFit] = []
    training_rows: list[dict[str, Any]] = []
    tree_backends: set[str] = set()
    for seed in seeds:
        model, history = train_neural(train_data, validation_data, preprocessor, config, seed, device)
        neural_runs.append(predict_neural(model, all_data, device))
        training_rows.extend(history)
        torch.save({
            "model_state": model.state_dict(), "seed": seed,
            "model_version": config["project"]["model_version"],
        }, output_dir / f"neural_seed_{seed}.pt")
        tree = fit_tree(train_data.tabular, train_data.target_angle, seed)
        tree_models.append(tree)
        tree_runs.append(np.clip(tree.model.predict(all_data.tabular), 0.0, 180.0))
        tree_backends.add(tree.backend)
        joblib.dump(tree.model, output_dir / f"tree_seed_{seed}.joblib")
    pd.DataFrame(training_rows).to_csv(output_dir / "training_log.csv", index=False, encoding="utf-8-sig")

    physics_by_seed = np.stack([run.physics for run in neural_runs])
    neural_by_seed = np.stack([run.neural for run in neural_runs])
    tree_by_seed = np.stack(tree_runs)
    sigma_by_seed = np.stack([run.sigma_deg for run in neural_runs])
    sfe_d_by_seed = np.stack([run.sfe_dispersion for run in neural_runs])
    sfe_p_by_seed = np.stack([run.sfe_polar for run in neural_runs])
    residual_by_seed = np.stack([run.residual_cosine for run in neural_runs])
    experts = np.column_stack([
        physics_by_seed.mean(axis=0), neural_by_seed.mean(axis=0), tree_by_seed.mean(axis=0)
    ])
    fusion, weights = predict_gate(gate, experts, all_data.gate_context)
    fused_seed_predictions = []
    for seed_index in range(len(seeds)):
        seed_experts = np.column_stack([
            physics_by_seed[seed_index], neural_by_seed[seed_index], tree_by_seed[seed_index]
        ])
        seed_fusion, _ = predict_gate(gate, seed_experts, all_data.gate_context)
        fused_seed_predictions.append(seed_fusion)
    ensemble_std = np.std(np.stack(fused_seed_predictions), axis=0)
    aleatoric_sigma = sigma_by_seed.mean(axis=0)
    predictive_scale = np.sqrt(np.square(ensemble_std) + np.square(aleatoric_sigma)).clip(1e-3)

    conformal = AdaptiveConformal(level=float(config["uncertainty"]["conformal_level"]))
    conformal_scale = predictive_scale if bool(config["uncertainty"].get("adaptive_scale", False)) else np.ones_like(predictive_scale)
    conformal.fit(
        all_data.target_angle[validation_indices], fusion[validation_indices], conformal_scale[validation_indices]
    )
    lower, upper = conformal.interval(fusion, conformal_scale)
    ood = KNNOODDetector(quantile=float(config["uncertainty"]["ood_quantile"]))
    ood.fit(all_data.tabular[train_indices], all_data.tabular[validation_indices])
    joblib.dump(ood, output_dir / "ood_detector.joblib")
    ood_score = ood.score(all_data.tabular)
    unknown_category = (all_data.categorical == 0).any(axis=1)
    interval_width = upper - lower
    width_threshold = float(np.quantile(
        interval_width[validation_indices], float(config["uncertainty"]["abstention_width_quantile"])
    ))
    # Unknown categories increase risk, but they are not an automatic rejection:
    # otherwise every genuinely new material in the prospective set is discarded.
    # Use a strict width comparison so values clipped exactly at the validation
    # P90 threshold are not rejected merely because of floating-point ties.
    abstain = (ood_score > 1.0) | (interval_width > width_threshold)
    risk = np.where(
        abstain, "high",
        np.where(
            unknown_category | (ood_score > 0.75) | (interval_width > 0.75 * width_threshold),
            "medium", "low"
        ),
    )

    predictions = pd.DataFrame({
        "sample_id": all_data.sample_ids,
        "record_id": all_data.record_ids,
        "source_group_id": all_data.source_group_ids,
        "surface_group_id": all_data.surface_group_ids,
        "target_liquid_id": all_data.target_liquid_ids,
        "prediction_mode": all_data.modes,
        "v4_split": all_data.splits,
        "theta_observed_deg": all_data.target_angle,
        "theta_pred_deg": fusion,
        "interval_lower_deg": lower,
        "interval_upper_deg": upper,
        "confidence_level": float(config["uncertainty"]["conformal_level"]),
        "risk_level": risk,
        "abstain_flag": np.where(abstain, "yes", "no"),
        "physics_prediction": experts[:, 0],
        "neural_prediction": experts[:, 1],
        "tree_prediction": experts[:, 2],
        "weight_physics": weights[:, 0],
        "weight_neural": weights[:, 1],
        "weight_tree": weights[:, 2],
        "ood_score": ood_score,
        "unknown_category": np.where(unknown_category, "yes", "no"),
        "ensemble_std_deg": ensemble_std,
        "aleatoric_sigma_deg": aleatoric_sigma,
        "latent_sfe_dispersion_mj_m2": sfe_d_by_seed.mean(axis=0),
        "latent_sfe_polar_mj_m2": sfe_p_by_seed.mean(axis=0),
        "residual_shift_cosine": residual_by_seed.mean(axis=0),
        "model_version": config["project"]["model_version"],
    })
    prediction_path = output_dir / "predictions_v4.csv"
    predictions.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    calibration = {
        "confidence_level": float(config["uncertainty"]["conformal_level"]),
        "conformal_quantile": conformal.quantile,
        "ood_threshold_normalized": 1.0,
        "abstention_width_threshold_deg": width_threshold,
        "model_version": config["project"]["model_version"],
        "seeds": seeds,
    }
    (output_dir / "calibration.json").write_text(
        json.dumps(calibration, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    metric_rows: list[dict[str, Any]] = []
    model_predictions = {
        "physics_expert": predictions["physics_prediction"].to_numpy(),
        "neural_expert": predictions["neural_prediction"].to_numpy(),
        "tree_expert": predictions["tree_prediction"].to_numpy(),
        "lspgmoe_fusion": predictions["theta_pred_deg"].to_numpy(),
    }
    for split in ["validation", "internal_test", "legacy_external", "prospective_open_external"]:
        for mode in ["zero_shot", "probe_assisted"]:
            mask = ((predictions["v4_split"] == split) & (predictions["prediction_mode"] == mode)).to_numpy()
            if not mask.any():
                continue
            for model_name, values in model_predictions.items():
                kwargs = {}
                if model_name == "lspgmoe_fusion":
                    kwargs = {"lower": lower[mask], "upper": upper[mask], "abstain": abstain[mask]}
                metric_rows.append({
                    "split": split, "prediction_mode": mode, "model": model_name,
                    **regression_metrics(all_data.target_angle[mask], values[mask], **kwargs),
                })
    metrics_path = output_dir / "metrics_v4.csv"
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False, encoding="utf-8-sig")

    bootstrap_rows: list[dict[str, Any]] = []
    for split in ["internal_test", "legacy_external", "prospective_open_external"]:
        for mode in ["zero_shot", "probe_assisted"]:
            mask = ((predictions["v4_split"] == split) & (predictions["prediction_mode"] == mode)).to_numpy()
            if mask.sum() < 2:
                continue
            for cluster_name, clusters in [
                ("surface_group_id", np.asarray(all_data.surface_group_ids)[mask]),
                ("source_group_id", np.asarray(all_data.source_group_ids)[mask]),
            ]:
                for baseline_name, baseline_index in [
                    ("physics", 0), ("neural", 1), ("tree", 2)
                ]:
                    result = paired_cluster_bootstrap(
                        all_data.target_angle[mask], fusion[mask], experts[mask, baseline_index], clusters,
                        int(config["statistics"]["bootstrap_resamples"]), int(config["project"]["seed"]),
                    )
                    bootstrap_rows.append({
                        "split": split, "prediction_mode": mode,
                        "comparison": f"lspgmoe_fusion_minus_{baseline_name}", "cluster": cluster_name,
                        **result,
                    })
    pd.DataFrame(bootstrap_rows).to_csv(
        output_dir / "paired_cluster_bootstrap.csv", index=False, encoding="utf-8-sig"
    )

    manifest = {
        "status": "complete",
        "model_version": config["project"]["model_version"],
        "device": str(device),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seeds": seeds,
        "tree_backend": "+".join(sorted(tree_backends)),
        "oof_tree_backend": oof_tree_backend,
        "n_eligible_samples": len(samples),
        "n_training_samples": len(train_samples),
        "n_validation_samples": len(validation_samples),
        "conformal_quantile": conformal.quantile,
        "ood_threshold": ood.threshold,
        "abstention_width_threshold_deg": width_threshold,
        "input_hashes": {
            name: _hash_file(data_dir / name) for name in [
                "sources_v4.csv", "surfaces_v4.csv", "liquids_v4.csv",
                "measurements_v4.csv", "splits_v4.csv", "samples_v4.csv",
            ]
        },
        "output_hashes": {
            prediction_path.name: _hash_file(prediction_path), metrics_path.name: _hash_file(metrics_path),
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def run_smoke_experiment(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return run_experiment(config, config_path.parent.parent)
