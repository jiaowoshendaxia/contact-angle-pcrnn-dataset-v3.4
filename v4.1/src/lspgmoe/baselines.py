"""Official and transparent baseline runners for the v4 benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from .data import V4Tables
from .features import EncodedSamples
from .metrics import regression_metrics
from .training import fit_tree


@dataclass
class BaselineResult:
    predictions: dict[str, np.ndarray]
    backends: dict[str, str]


def run_tabular_baselines(
    train: EncodedSamples,
    evaluation: EncodedSamples,
    seeds: list[int] | tuple[int, ...] = (7, 19, 42, 67, 99),
) -> BaselineResult:
    """Fit five-seed XGBoost, RF, MLP, and no-SFE controls on training sources only."""
    tree_predictions: list[np.ndarray] = []
    rf_predictions: list[np.ndarray] = []
    mlp_predictions: list[np.ndarray] = []
    no_sfe_tree_predictions: list[np.ndarray] = []
    no_sfe_mlp_predictions: list[np.ndarray] = []
    backends: dict[str, str] = {}
    no_sfe_train = train.tabular[:, :-7]
    no_sfe_evaluation = evaluation.tabular[:, :-7]
    for seed in seeds:
        xgb = fit_tree(train.tabular, train.target_angle, int(seed))
        tree_predictions.append(np.clip(xgb.model.predict(evaluation.tabular), 0.0, 180.0))
        backends["xgboost"] = xgb.backend

        rf = RandomForestRegressor(
            n_estimators=500, max_features="sqrt", min_samples_leaf=2,
            random_state=int(seed), n_jobs=-1,
        )
        rf.fit(train.tabular, train.target_angle)
        rf_predictions.append(np.clip(rf.predict(evaluation.tabular), 0.0, 180.0))
        backends["random_forest"] = "sklearn_random_forest"

        mlp = MLPRegressor(
            hidden_layer_sizes=(128, 64), activation="relu", alpha=1e-4,
            learning_rate_init=1e-3, max_iter=1000, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=50, random_state=int(seed),
        )
        mlp.fit(train.tabular, train.target_angle)
        mlp_predictions.append(np.clip(mlp.predict(evaluation.tabular), 0.0, 180.0))
        backends["mlp"] = "sklearn_mlp"

        no_sfe_xgb = fit_tree(no_sfe_train, train.target_angle, int(seed))
        no_sfe_tree_predictions.append(np.clip(
            no_sfe_xgb.model.predict(no_sfe_evaluation), 0.0, 180.0
        ))
        backends["xgboost_no_sfe"] = no_sfe_xgb.backend

        no_sfe_mlp = MLPRegressor(
            hidden_layer_sizes=(128, 64), activation="relu", alpha=1e-4,
            learning_rate_init=1e-3, max_iter=1000, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=50, random_state=int(seed),
        )
        no_sfe_mlp.fit(no_sfe_train, train.target_angle)
        no_sfe_mlp_predictions.append(np.clip(
            no_sfe_mlp.predict(no_sfe_evaluation), 0.0, 180.0
        ))
        backends["mlp_no_sfe"] = "sklearn_mlp"
    return BaselineResult(
        predictions={
            "xgboost": np.mean(tree_predictions, axis=0),
            "random_forest": np.mean(rf_predictions, axis=0),
            "mlp": np.mean(mlp_predictions, axis=0),
            "xgboost_no_sfe": np.mean(no_sfe_tree_predictions, axis=0),
            "mlp_no_sfe": np.mean(no_sfe_mlp_predictions, axis=0),
        },
        backends=backends,
    )


def run_baselines(config_path: Path) -> dict:
    """CLI benchmark using only source-group training rows."""
    import yaml

    config_path = config_path.resolve()
    root = config_path.parent.parent
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_dir = root / config["project"]["output_data_dir"]
    output_dir = root / config["project"]["output_dir"] / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = V4Tables.load(data_dir)
    samples = pd.read_csv(data_dir / "samples_v4.csv", encoding="utf-8-sig")
    samples = samples.merge(
        tables.measurements[["measurement_id", "target_eligible"]],
        left_on="target_measurement_id", right_on="measurement_id", how="left",
        validate="many_to_one",
    )
    samples = samples.loc[
        (samples["target_eligible"] == "yes") & (samples["v4_split"] != "excluded_review")
    ].drop(columns=["measurement_id"]).reset_index(drop=True)
    train_samples = samples.loc[samples["v4_split"] == "train"].reset_index(drop=True)
    from .features import FeaturePreprocessor
    preprocessor = FeaturePreprocessor().fit(tables, train_samples)
    encoded = preprocessor.transform(tables, samples)
    train = encoded.subset(np.flatnonzero(np.asarray(encoded.splits) == "train"))
    seeds = [int(value) for value in config["model"]["seeds"]]
    result = run_tabular_baselines(train, encoded, seeds=seeds)
    rows = []
    for split in ["validation", "internal_test", "legacy_external", "prospective_open_external"]:
        for mode in ["zero_shot", "probe_assisted"]:
            mask = (np.asarray(encoded.splits) == split) & (np.asarray(encoded.modes) == mode)
            if not mask.any():
                continue
            for name, values in result.predictions.items():
                rows.append({
                    "split": split, "prediction_mode": mode, "model": name,
                    **regression_metrics(encoded.target_angle[mask], values[mask]),
                })
    pd.DataFrame(rows).to_csv(output_dir / "baseline_metrics_v4.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"sample_id": encoded.sample_ids, **result.predictions}).to_csv(
        output_dir / "baseline_predictions_v4.csv", index=False, encoding="utf-8-sig"
    )
    return {"status": "complete", "backends": result.backends, "n_samples": len(encoded)}
