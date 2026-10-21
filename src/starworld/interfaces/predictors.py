from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class PredictionInterface(ABC):
    name: str

    @abstractmethod
    def predict(self, throughput_history_mbps: np.ndarray) -> tuple[float, float]:
        """Return capacity estimate and uncertainty in Mbps."""


class NoPredictor(PredictionInterface):
    name = "none"

    def predict(self, throughput_history_mbps: np.ndarray) -> tuple[float, float]:
        del throughput_history_mbps
        return 0.0, 0.0


class PointCapacityPredictor(PredictionInterface):
    name = "point"

    def __init__(self, history: int = 8):
        self.history = history

    def predict(self, throughput_history_mbps: np.ndarray) -> tuple[float, float]:
        values = np.asarray(throughput_history_mbps, dtype=float)[-self.history :]
        if values.size == 0:
            return 0.0, 0.0
        return float(np.mean(values)), float(np.std(values))


class SafeCapacityPredictor(PredictionInterface):
    name = "safe_capacity"

    def __init__(self, quantile: float = 0.1, history: int = 12):
        if not 0 < quantile < 1:
            raise ValueError("quantile must lie in (0, 1)")
        self.quantile = quantile
        self.history = history

    def predict(self, throughput_history_mbps: np.ndarray) -> tuple[float, float]:
        values = np.asarray(throughput_history_mbps, dtype=float)[-self.history :]
        if values.size == 0:
            return 0.0, 0.0
        estimate = float(np.quantile(values, self.quantile))
        return estimate, float(max(np.mean(values) - estimate, 0.0))
