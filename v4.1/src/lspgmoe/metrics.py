"""Evaluation metrics and grouped bootstrap statistics."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    abstain: np.ndarray | None = None,
) -> dict[str, float | int]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    absolute_error = np.abs(observed - predicted)
    output: dict[str, float | int] = {
        "n": int(len(observed)),
        "mae_deg": float(mean_absolute_error(observed, predicted)),
        "rmse_deg": float(np.sqrt(mean_squared_error(observed, predicted))),
        "r2": float(r2_score(observed, predicted)) if len(observed) > 1 else float("nan"),
        "medae_deg": float(np.median(absolute_error)),
        "p90ae_deg": float(np.quantile(absolute_error, 0.90)),
    }
    if lower is not None and upper is not None:
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        output["interval_coverage"] = float(np.mean((observed >= lower) & (observed <= upper)))
        output["mean_interval_width_deg"] = float(np.mean(upper - lower))
    if abstain is not None:
        abstain = np.asarray(abstain, dtype=bool)
        keep = ~abstain
        output["retained_fraction"] = float(np.mean(keep))
        output["retained_mae_deg"] = (
            float(mean_absolute_error(observed[keep], predicted[keep])) if keep.any() else float("nan")
        )
    return output


def paired_cluster_bootstrap(
    observed: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    clusters: np.ndarray,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap paired MAE(A)-MAE(B), resampling whole clusters."""
    observed = np.asarray(observed, dtype=float)
    prediction_a = np.asarray(prediction_a, dtype=float)
    prediction_b = np.asarray(prediction_b, dtype=float)
    clusters = np.asarray(clusters)
    unique = np.unique(clusters)
    rng = np.random.default_rng(seed)
    differences = np.empty(resamples, dtype=float)
    cluster_indices = {cluster: np.flatnonzero(clusters == cluster) for cluster in unique}
    for iteration in range(resamples):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([cluster_indices[cluster] for cluster in sampled])
        mae_a = np.mean(np.abs(observed[indices] - prediction_a[indices]))
        mae_b = np.mean(np.abs(observed[indices] - prediction_b[indices]))
        differences[iteration] = mae_a - mae_b
    return {
        "n_clusters": int(len(unique)),
        "n_resamples": int(resamples),
        "mae_difference_a_minus_b_deg": float(
            np.mean(np.abs(observed - prediction_a)) - np.mean(np.abs(observed - prediction_b))
        ),
        "ci95_lower_deg": float(np.quantile(differences, 0.025)),
        "ci95_upper_deg": float(np.quantile(differences, 0.975)),
        "probability_a_better": float(np.mean(differences < 0.0)),
    }
