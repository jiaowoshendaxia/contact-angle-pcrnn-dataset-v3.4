"""Required deep-module, fusion, uncertainty, and legacy-physics ablations."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from .data import V4Tables
from .features import FeaturePreprocessor
from .metrics import regression_metrics
from .training import predict_gate, predict_neural, train_gate, train_neural


def _eligible(tables: V4Tables, samples: pd.DataFrame) -> pd.DataFrame:
    eligibility = tables.measurements[["measurement_id", "target_eligible"]]
    samples = samples.merge(
        eligibility, left_on="target_measurement_id", right_on="measurement_id",
        how="left", validate="many_to_one",
    )
    return samples.loc[
        (samples["target_eligible"] == "yes") & (samples["v4_split"] != "excluded_review")
    ].drop(columns=["measurement_id"]).reset_index(drop=True)


def _deep_variant(
    name: str,
    tables: V4Tables,
    data,
    train_data,
    validation_data,
    preprocessor: FeaturePreprocessor,
    config: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    variant = copy.deepcopy(config)
    if name == "no_probe_encoder":
        variant["model"]["use_probe_encoder"] = False
    elif name == "deepsets_no_physics":
        variant["model"]["use_physics_decoder"] = False
        variant["loss"]["masked_probe"] = 0.0
        variant["loss"]["independent_sfe"] = 0.0
    else:
        raise ValueError(name)
    runs = []
    for seed in variant["model"]["seeds"]:
        model, _ = train_neural(
            train_data, validation_data, preprocessor, variant, int(seed), device
        )
        runs.append(predict_neural(model, data, device).neural)
    return np.mean(np.stack(runs), axis=0)


def run_ablations(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    root = config_path.parent.parent
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_dir = root / config["project"]["output_data_dir"]
    output_dir = root / config["project"]["output_dir"] / "ablations"
    experiment_dir = root / config["project"]["output_dir"] / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = V4Tables.load(data_dir)
    samples = _eligible(tables, pd.read_csv(data_dir / "samples_v4.csv", encoding="utf-8-sig"))
    train_samples = samples.loc[samples["v4_split"] == "train"].reset_index(drop=True)
    preprocessor = FeaturePreprocessor().fit(tables, train_samples)
    data = preprocessor.transform(tables, samples)
    splits = np.asarray(data.splits)
    train_data = data.subset(np.flatnonzero(splits == "train"))
    validation_data = data.subset(np.flatnonzero(splits == "validation"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full = pd.read_csv(experiment_dir / "predictions_v4.csv", encoding="utf-8-sig")
    full = full.set_index("sample_id").loc[data.sample_ids].reset_index()
    outputs: dict[str, np.ndarray] = {
        "full_lspgmoe": full["theta_pred_deg"].to_numpy(),
        "latent_sfe_physics_no_moe": full["neural_prediction"].to_numpy(),
        "fixed_average_fusion": full[[
            "physics_prediction", "neural_prediction", "tree_prediction"
        ]].mean(axis=1).to_numpy(),
        "no_uncertainty_or_rejection": full["theta_pred_deg"].to_numpy(),
    }
    for name in ["no_probe_encoder", "deepsets_no_physics"]:
        outputs[name] = _deep_variant(
            name, tables, data, train_data, validation_data, preprocessor, config, device
        )

    oof = pd.read_csv(experiment_dir / "oof_expert_predictions.csv", encoding="utf-8-sig")
    ordered_train = train_samples.set_index("sample_id").loc[oof["sample_id"]].reset_index()
    oof_context = preprocessor.transform(tables, ordered_train).gate_context
    oof_target = ordered_train["target_contact_angle_deg"].to_numpy()
    final_experts = full[["physics_prediction", "neural_prediction", "tree_prediction"]].to_numpy()
    oof_matrix = oof[["physics_oof_deg", "neural_oof_deg", "tree_oof_deg"]].to_numpy()
    for name, selected in {
        "no_physics_expert": [1, 2],
        "no_xgboost_expert": [0, 1],
    }.items():
        gate = train_gate(
            oof_matrix[:, selected], oof_context, oof_target, int(config["project"]["seed"]),
            max_epochs=int(config["model"].get("gate_epochs", 1000)),
            patience=int(config["model"].get("gate_patience", 100)),
        )
        outputs[name] = predict_gate(gate, final_experts[:, selected], data.gate_context)[0]

    frame = pd.DataFrame({
        "sample_id": data.sample_ids, "source_group_id": data.source_group_ids,
        "surface_group_id": data.surface_group_ids, "prediction_mode": data.modes,
        "v4_split": data.splits, "theta_observed_deg": data.target_angle,
        **outputs,
    })
    frame.to_csv(output_dir / "ablation_predictions_v4.csv", index=False, encoding="utf-8-sig")
    rows: list[dict[str, Any]] = []
    for split in ["validation", "internal_test", "legacy_external", "prospective_open_external"]:
        for mode in ["zero_shot", "probe_assisted"]:
            mask = ((frame["v4_split"] == split) & (frame["prediction_mode"] == mode)).to_numpy()
            for name, prediction in outputs.items():
                rows.append({
                    "split": split, "prediction_mode": mode, "ablation": name,
                    **regression_metrics(data.target_angle[mask], prediction[mask]),
                })

    audit = pd.read_csv(
        root / config["project"]["output_dir"] / "audit" / "loo_nnls_audit_v4.csv",
        encoding="utf-8-sig",
    )
    for split in ["validation", "internal_test", "legacy_external"]:
        subset = audit.loc[
            (audit["v4_split"] == split)
            & audit["nnls_physical_prediction_deg"].notna()
            & audit["legacy_squared_physical_prediction_deg"].notna()
        ]
        for name, column in {
            "nnls_owrk": "nnls_physical_prediction_deg",
            "legacy_negative_square_owrk": "legacy_squared_physical_prediction_deg",
        }.items():
            if len(subset):
                rows.append({
                    "split": split, "prediction_mode": "probe_assisted_common_subset",
                    "ablation": name,
                    **regression_metrics(
                        subset["target_contact_angle_deg"].to_numpy(), subset[column].to_numpy()
                    ),
                })
    pd.DataFrame(rows).to_csv(output_dir / "ablation_metrics_v4.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "status": "complete", "device": str(device), "variants": list(outputs),
        "legacy_physics_comparison": ["nnls_owrk", "legacy_negative_square_owrk"],
    }
    (output_dir / "ablation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest
