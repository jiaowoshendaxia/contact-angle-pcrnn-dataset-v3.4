from pathlib import Path

import numpy as np
import pandas as pd

from lspgmoe.robustness import (
    build_loso_splits,
    clip_angle,
    residual_location_predictions,
    source_balance_weights,
)


ROOT = Path(__file__).resolve().parents[1]


def test_source_balance_weights_are_mean_one_and_favor_small_sources():
    weights = source_balance_weights(["large", "large", "large", "large", "small"])
    assert np.isclose(weights.mean(), 1.0)
    assert weights[-1] > weights[0]


def test_residual_location_baselines_use_training_liquids_with_global_fallback():
    global_residual, liquid_residual, counts = residual_location_predictions(
        train_target=np.array([12.0, 22.0, 40.0]),
        train_physics=np.array([10.0, 20.0, 30.0]),
        train_liquids=["water", "water", "dim"],
        holdout_liquids=["water", "unknown"],
        liquid_shrinkage=0.0,
    )
    np.testing.assert_allclose(global_residual, [14.0 / 3.0, 14.0 / 3.0])
    np.testing.assert_allclose(liquid_residual, [2.0, 14.0 / 3.0])
    np.testing.assert_array_equal(counts, [2, 0])


def test_loso_folds_are_source_disjoint_and_cover_every_row_once():
    frame = pd.DataFrame({
        "source_group_id": ["A", "A", "B", "C"],
        "v4_split": ["train", "train", "validation", "train"],
    })
    folds = build_loso_splits(frame, ["train", "validation"])
    held_out = []
    for source, train, holdout in folds:
        assert set(frame.iloc[train]["source_group_id"]).isdisjoint(
            set(frame.iloc[holdout]["source_group_id"])
        )
        assert set(frame.iloc[holdout]["source_group_id"]) == {source}
        held_out.extend(holdout.tolist())
    assert sorted(held_out) == list(range(len(frame)))


def test_loso_rejects_external_rows():
    frame = pd.DataFrame({
        "source_group_id": ["A", "B", "C"],
        "v4_split": ["train", "validation", "legacy_external"],
    })
    try:
        build_loso_splits(frame, ["train", "validation"])
    except ValueError as error:
        assert "External split" in str(error)
    else:
        raise AssertionError("LOSO accepted an external confirmation row")


def test_clip_angle_preserves_physical_anchor_and_bounds():
    prediction = clip_angle(np.array([10.0, 170.0]), np.array([-20.0, 30.0]))
    np.testing.assert_allclose(prediction, [0.0, 180.0])


def test_locked_loso_metrics_recompute_from_predictions_when_available():
    output = ROOT / "outputs" / "v4_2_model_strengthening"
    prediction_path = output / "loso_predictions_v4_2.csv"
    metric_path = output / "loso_metrics_v4_2.csv"
    if not prediction_path.exists() or not metric_path.exists():
        return
    predictions = pd.read_csv(prediction_path, encoding="utf-8-sig")
    metrics = pd.read_csv(metric_path, encoding="utf-8-sig")
    assert len(predictions) == 240
    assert not predictions["sample_id"].duplicated().any()
    row = metrics.loc[
        metrics["model"] == "source_weighted_physics_residual_xgboost"
    ].iloc[0]
    recomputed = np.mean(np.abs(
        predictions["theta_observed_deg"]
        - predictions["weighted_xgboost_residual_prediction"]
    ))
    assert np.isclose(recomputed, row["mae_deg"])


def test_fixed_confirmation_contains_only_declared_evaluation_splits_when_available():
    path = ROOT / "outputs" / "v4_2_model_strengthening" / "confirmation_predictions_v4_2.csv"
    if not path.exists():
        return
    predictions = pd.read_csv(path, encoding="utf-8-sig")
    assert set(predictions["v4_split"]) == {
        "internal_test", "legacy_external", "prospective_open_external"
    }
    assert len(predictions) == 347
