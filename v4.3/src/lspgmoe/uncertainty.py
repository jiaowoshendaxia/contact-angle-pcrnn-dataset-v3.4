"""Split-conformal intervals and feature-space OOD scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass
class AdaptiveConformal:
    level: float = 0.90
    quantile: float | None = None

    def fit(self, observed: np.ndarray, predicted: np.ndarray, scale: np.ndarray) -> "AdaptiveConformal":
        observed = np.asarray(observed, dtype=float)
        predicted = np.asarray(predicted, dtype=float)
        scale = np.maximum(np.asarray(scale, dtype=float), 1e-3)
        scores = np.abs(observed - predicted) / scale
        n = len(scores)
        probability = min(1.0, np.ceil((n + 1) * self.level) / n)
        self.quantile = float(np.quantile(scores, probability, method="higher"))
        return self

    def interval(self, predicted: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.quantile is None:
            raise RuntimeError("AdaptiveConformal must be fitted on validation data")
        predicted = np.asarray(predicted, dtype=float)
        half_width = self.quantile * np.maximum(np.asarray(scale, dtype=float), 1e-3)
        return np.clip(predicted - half_width, 0.0, 180.0), np.clip(predicted + half_width, 0.0, 180.0)


class KNNOODDetector:
    def __init__(self, quantile: float = 0.95, neighbors: int = 5) -> None:
        self.quantile = quantile
        self.neighbors = neighbors
        self.model: NearestNeighbors | None = None
        self.threshold: float | None = None

    def fit(self, train: np.ndarray, validation: np.ndarray) -> "KNNOODDetector":
        neighbors = min(self.neighbors, len(train))
        self.model = NearestNeighbors(n_neighbors=neighbors).fit(train)
        distances = self.model.kneighbors(validation, return_distance=True)[0][:, -1]
        self.threshold = max(float(np.quantile(distances, self.quantile)), 1e-8)
        return self

    def score(self, values: np.ndarray) -> np.ndarray:
        if self.model is None or self.threshold is None:
            raise RuntimeError("KNNOODDetector must be fitted first")
        distances = self.model.kneighbors(values, return_distance=True)[0][:, -1]
        return distances / self.threshold
