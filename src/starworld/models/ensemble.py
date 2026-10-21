from __future__ import annotations

import copy

import torch
from torch import nn

from starworld.config import ModelConfig
from starworld.models.world_model import StarWorldModel, WorldModelOutput


class WorldModelEnsemble(nn.Module):
    def __init__(self, config: ModelConfig, size: int | None = None):
        super().__init__()
        ensemble_size = size or config.ensemble_size
        if ensemble_size < 1:
            raise ValueError("ensemble size must be positive")
        self.members = nn.ModuleList([StarWorldModel(copy.deepcopy(config)) for _ in range(ensemble_size)])

    def forward_members(self, *args: torch.Tensor, **kwargs: object) -> list[WorldModelOutput]:
        return [member(*args, **kwargs) for member in self.members]

    @staticmethod
    def aggregate(outputs: list[WorldModelOutput]) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        for field in ("network_mean", "prediction_mean", "streaming_mean", "event_logits"):
            values = torch.stack([getattr(output, field) for output in outputs], dim=0)
            result[field] = values.mean(dim=0)
            result[f"{field}_epistemic_std"] = values.std(dim=0, unbiased=False)
        for field in ("network_log_std", "prediction_log_std", "streaming_log_std"):
            result[field] = torch.stack([getattr(output, field) for output in outputs], dim=0).mean(dim=0)
        return result
