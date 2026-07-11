import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import GroupKFold

from lspgmoe.model import PhysicsSummaryResidualExpert
from lspgmoe.v41 import (
    V41Preprocessor,
    _json_default,
    _load_primary_samples,
    _predict_tree_member,
    fit_simplex_weights,
    set_seed,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "v4_1_main.yaml"


def _config():
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_v41_primary_subset_is_target_masked_and_nnls_feasible():
    tables, primary, audit = _load_primary_samples(ROOT, _config())
    assert len(primary) > 0
    assert (primary["n_unique_liquids"] >= 2).all()
    assert primary["fit_status"].isin(["interior_fit", "boundary_fit"]).all()
    assert (primary[["nnls_dispersion_mj_m2", "nnls_polar_mj_m2"]].to_numpy() >= 0).all()
    for row in primary.itertuples(index=False):
        assert str(row.target_liquid_id) not in set(str(row.probe_liquid_ids).split(";"))
    assert (audit.loc[audit["n_unique_liquids"] < 2, "v4_1_primary_eligible"] == "no").all()


def test_v41_development_source_folds_are_disjoint():
    _, primary, _ = _load_primary_samples(ROOT, _config())
    development = primary.loc[primary["v4_split"].isin(["train", "validation"])].reset_index(drop=True)
    groups = development["source_group_id"].astype(str).to_numpy()
    for fit, holdout in GroupKFold(n_splits=4).split(development, groups=groups):
        assert set(groups[fit]).isdisjoint(set(groups[holdout]))
    external = primary.loc[primary["v4_split"].isin([
        "internal_test", "legacy_external", "prospective_open_external"
    ])]
    assert set(development["source_group_id"]).isdisjoint(set(external["source_group_id"]))


def test_physics_summary_does_not_depend_on_probe_order():
    tables, primary, _ = _load_primary_samples(ROOT, _config())
    train = primary.loc[primary["v4_split"] == "train"].reset_index(drop=True)
    preprocessor = V41Preprocessor().fit(tables, train)
    sample = train.iloc[[0]].copy()
    original = preprocessor.transform(tables, sample)
    sample.loc[:, "probe_measurement_ids"] = ";".join(
        reversed(str(sample.iloc[0]["probe_measurement_ids"]).split(";"))
    )
    permuted = preprocessor.transform(tables, sample)
    np.testing.assert_allclose(original.physics_summary, permuted.physics_summary)


def test_simplex_weights_are_nonnegative_and_sum_to_one():
    experts = np.array([[10, 20, 30], [30, 20, 10], [15, 25, 35]], dtype=float)
    weights = fit_simplex_weights(experts, np.array([18, 22, 20], dtype=float))
    assert np.all(weights >= 0)
    assert np.isclose(weights.sum(), 1.0)


def test_numpy_scalars_are_serialized_as_native_json_values():
    payload = json.dumps({"accepted": np.bool_(True)}, default=_json_default)
    assert payload == '{"accepted": true}'


def test_compact_residual_is_bounded_and_reproducible():
    kwargs = dict(
        surface_numeric_dim=6, liquid_numeric_dim=12, condition_numeric_dim=8,
        physics_summary_dim=15, categorical_cardinalities=[4] * 7,
        max_delta_cos=0.25, dropout=0.0,
    )
    inputs = (
        torch.randn(3, 6), torch.zeros(3, 7, dtype=torch.long),
        torch.randn(3, 12), torch.randn(3, 8), torch.randn(3, 15),
        torch.tensor([0.1, -0.2, 0.3]),
    )
    set_seed(42)
    first = PhysicsSummaryResidualExpert(**kwargs).eval()(*inputs)
    set_seed(42)
    second = PhysicsSummaryResidualExpert(**kwargs).eval()(*inputs)
    torch.testing.assert_close(first.theta_neural, second.theta_neural)
    assert torch.all(first.residual_cosine.abs() <= 0.25)


def test_residual_tree_prediction_is_anchored_to_physics():
    class ConstantResidual:
        def predict(self, features):
            return np.full(len(features), 5.0)

    prediction = _predict_tree_member(
        "xgboost", ConstantResidual(), np.zeros((2, 3)), np.array([70.0, 179.0])
    )
    np.testing.assert_allclose(prediction, [75.0, 180.0])


def test_locked_main_metrics_recompute_from_predictions_when_available():
    output = ROOT / "results"
    if not output.exists():
        output = ROOT / "outputs" / "v4_1_final"
    prediction_path = output / "predictions_v4_1.csv"
    metric_path = output / "metrics_v4_1.csv"
    if not prediction_path.exists() or not metric_path.exists():
        return
    predictions = pd.read_csv(prediction_path, encoding="utf-8-sig")
    metrics = pd.read_csv(metric_path, encoding="utf-8-sig")
    row = metrics.loc[
        (metrics["split"] == "prospective_open_external")
        & (metrics["model"] == "ls_psrmoe_fusion")
    ].iloc[0]
    subset = predictions.loc[predictions["v4_split"] == "prospective_open_external"]
    recomputed = np.mean(np.abs(subset["theta_observed_deg"] - subset["theta_pred_deg"]))
    assert np.isclose(recomputed, row["mae_deg"])
