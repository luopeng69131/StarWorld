from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class FeatureNormalizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, arrays: list[np.ndarray], epsilon: float = 1e-6) -> "FeatureNormalizer":
        values = np.concatenate(arrays, axis=0).astype(np.float64)
        mean = np.nanmean(values, axis=0)
        scale = np.nanstd(values, axis=0)
        scale = np.where(scale < epsilon, 1.0, scale)
        return cls(mean.astype(np.float32), scale.astype(np.float32))

    def transform(self, value: np.ndarray) -> np.ndarray:
        return ((value - self.mean) / self.scale).astype(np.float32)

    def inverse(self, value: np.ndarray) -> np.ndarray:
        return (value * self.scale + self.mean).astype(np.float32)

    def transform_tensor(self, value: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, device=value.device, dtype=value.dtype)
        scale = torch.as_tensor(self.scale, device=value.device, dtype=value.dtype)
        return (value - mean) / scale

    def inverse_tensor(self, value: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, device=value.device, dtype=value.dtype)
        scale = torch.as_tensor(self.scale, device=value.device, dtype=value.dtype)
        return value * scale + mean

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, values: dict[str, list[float]]) -> "FeatureNormalizer":
        return cls(np.asarray(values["mean"], np.float32), np.asarray(values["scale"], np.float32))
