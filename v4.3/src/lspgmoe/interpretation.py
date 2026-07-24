"""Post-hoc interpretation and error stratification for locked v4 predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from .data import V4Tables
from .features import (
    CONDITION_NUMERIC, LIQUID_NUMERIC, SURFACE_CATEGORICAL, SURFACE_NUMERIC,
    CONDITION_CATEGORICAL, FeaturePreprocessor,
)
from .metrics import regression_metrics


def _eligible_samples(tables: V4Tables, samples: pd.DataFrame) -> pd.DataFrame:
    output = samples.merge(
        tables.measurements[["measurement_id", "target_eligible"]],
        left_on="target_measurement_id", right_on="measurement_id", how="left",
        validate="many_to_one",
    )
    return output.loc[
        (output["target_eligible"] == "yes") & (output["v4_split"] != "excluded_review")
    ].drop(columns=["measurement_id"]).reset_index(drop=True)


def _tabular_feature_names(preprocessor: FeaturePreprocessor) -> list[str]:
    names = [f"surface:{column}" for column in SURFACE_NUMERIC]
    names += [f"surface_missing:{column}" for column in SURFACE_NUMERIC]
    assert preprocessor.category_encoder is not None
    for column in SURFACE_CATEGORICAL + CONDITION_CATEGORICAL:
        size = len(preprocessor.category_encoder.vocabularies[column]) + 2
        names.extend(f"category:{column}[{index}]" for index in range(size))
    names += [f"liquid:{column}" for column in LIQUID_NUMERIC]
    names += [f"liquid_missing:{column}" for column in LIQUID_NUMERIC]
    names += [f"condition:{column}" for column in CONDITION_NUMERIC]
    names += [f"condition_missing:{column}" for column in CONDITION_NUMERIC]
    names += [
        "probe:n_probes", "probe:available", "probe:angle_mean",
        "probe:angle_std", "probe:angle_min", "probe:angle_max",
        "sfe:dispersion", "sfe:polar", "sfe:independent_available",
        "nnls:dispersion", "nnls:polar", "nnls:available", "nnls:physics_angle",
    ]
    return names


def _error_rows(frame: pd.DataFrame, group_column: str, group_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode, mode_frame in frame.groupby("prediction_mode", dropna=False):
        for group_value, group in mode_frame.groupby(group_column, dropna=False):
            for model_name, prediction_column in [
                ("fusion", "theta_pred_deg"), ("physics", "physics_prediction"),
                ("neural", "neural_prediction"), ("tree", "tree_prediction"),
            ]:
                metrics = regression_metrics(
                    group["theta_observed_deg"].to_numpy(), group[prediction_column].to_numpy()
                )
                rows.append({
                    "prediction_mode": str(mode), "stratifier": group_name, "group": str(group_value),
                    "model": model_name, **metrics,
                })
    return rows


def run_interpretation(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    root = config_path.parent.parent
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_dir = root / config["project"]["output_data_dir"]
    experiment_dir = root / config["project"]["output_dir"] / "experiments"
    output_dir = root / config["project"]["output_dir"] / "interpretation"
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = V4Tables.load(data_dir)
    samples = _eligible_samples(tables, pd.read_csv(data_dir / "samples_v4.csv", encoding="utf-8-sig"))
    train_samples = samples.loc[samples["v4_split"] == "train"].reset_index(drop=True)
    preprocessor = FeaturePreprocessor().fit(tables, train_samples)
    encoded = preprocessor.transform(tables, samples)
    predictions = pd.read_csv(experiment_dir / "predictions_v4.csv", encoding="utf-8-sig")
    predictions = predictions.set_index("sample_id").loc[encoded.sample_ids].reset_index()
    surface_frame = tables.surfaces[["surface_group_id", "solid_family", "surface_state"]]
    liquid_frame = tables.liquids[["liquid_id", "liquid_name"]]
    sample_aux = samples[["sample_id", "n_probes"]]
    frame = predictions.merge(surface_frame, on="surface_group_id", how="left", validate="many_to_one")
    frame = frame.merge(liquid_frame, left_on="target_liquid_id", right_on="liquid_id", how="left")
    frame = frame.merge(sample_aux, on="sample_id", how="left", validate="one_to_one")
    frame["angle_bin"] = pd.cut(
        frame["theta_observed_deg"], bins=[-0.1, 60.0, 120.0, 150.0, 180.0],
        labels=["hydrophilic_0_60", "neutral_60_120", "hydrophobic_120_150", "superhydrophobic_150_180"],
    ).astype(str)
    prospective = frame.loc[frame["v4_split"] == "prospective_open_external"].copy()
    prospective.to_csv(output_dir / "prospective_interpretation_frame.csv", index=False, encoding="utf-8-sig")

    rows: list[dict[str, Any]] = []
    for column, label in [
        ("liquid_name", "target_liquid"), ("solid_family", "solid_family"),
        ("surface_state", "surface_state"), ("angle_bin", "angle_bin"),
        ("source_group_id", "source_group"), ("n_probes", "n_probes"),
    ]:
        rows.extend(_error_rows(prospective, column, label))
    pd.DataFrame(rows).to_csv(output_dir / "error_stratification.csv", index=False, encoding="utf-8-sig")

    sensitivity_rows: list[dict[str, Any]] = []
    expert_columns = {
        "physics": "weight_physics", "neural": "weight_neural", "tree": "weight_tree",
    }
    for mode in ["zero_shot", "probe_assisted"]:
        subset = prospective.loc[prospective["prediction_mode"] == mode].copy()
        for removed, weight_column in expert_columns.items():
            remaining = [name for name in expert_columns if name != removed]
            denominator = subset[[expert_columns[name] for name in remaining]].sum(axis=1).clip(lower=1e-8)
            prediction = sum(
                subset[expert_columns[name]] * subset[f"{name}_prediction"] for name in remaining
            ) / denominator
            sensitivity_rows.append({
                "prediction_mode": mode, "removed_expert": removed,
                **regression_metrics(subset["theta_observed_deg"].to_numpy(), prediction.to_numpy()),
            })
    pd.DataFrame(sensitivity_rows).to_csv(output_dir / "expert_sensitivity.csv", index=False, encoding="utf-8-sig")

    shap_report: dict[str, Any]
    try:
        import shap

        model = joblib.load(experiment_dir / "tree_seed_7.joblib")
        mask = (np.asarray(encoded.splits) == "prospective_open_external")
        features = encoded.tabular[mask]
        names = _tabular_feature_names(preprocessor)
        if len(names) != features.shape[1]:
            raise ValueError(f"SHAP feature-name count {len(names)} != matrix width {features.shape[1]}")
        values = shap.TreeExplainer(model)(features).values
        if values.ndim == 3:
            values = values[..., 0]
        importance = pd.DataFrame({
            "feature": names,
            "mean_abs_shap": np.mean(np.abs(values), axis=0),
            "mean_shap": np.mean(values, axis=0),
        }).sort_values("mean_abs_shap", ascending=False)
        importance.to_csv(output_dir / "xgboost_shap_importance.csv", index=False, encoding="utf-8-sig")
        shap_report = {"status": "complete", "n_samples": int(features.shape[0]), "n_features": int(features.shape[1])}
    except Exception as error:  # Keep interpretation available if an optional SHAP backend changes.
        from sklearn.inspection import permutation_importance

        model = joblib.load(experiment_dir / "tree_seed_7.joblib")
        mask = (np.asarray(encoded.splits) == "prospective_open_external")
        features = encoded.tabular[mask]
        target = encoded.target_angle[mask]
        names = _tabular_feature_names(preprocessor)
        permutation = permutation_importance(
            model, features, target, scoring="neg_mean_absolute_error",
            n_repeats=10, random_state=int(config["project"]["seed"]), n_jobs=1,
        )
        importance = pd.DataFrame({
            "feature": names,
            "mean_importance": permutation.importances_mean,
            "std_importance": permutation.importances_std,
        }).sort_values("mean_importance", ascending=False)
        importance.to_csv(output_dir / "xgboost_permutation_importance.csv", index=False, encoding="utf-8-sig")
        shap_report = {
            "status": "complete_permutation_fallback", "shap_error": str(error),
            "n_samples": int(features.shape[0]), "n_features": int(features.shape[1]),
        }
    (output_dir / "interpretation_manifest.json").write_text(
        json.dumps({"status": "complete", "n_prospective": int(len(prospective)), "shap": shap_report}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {"status": "complete", "n_prospective": int(len(prospective)), "shap": shap_report}
