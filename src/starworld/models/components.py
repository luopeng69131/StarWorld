from __future__ import annotations

import torch
from torch import nn


class GaussianHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.mean = nn.Linear(input_dim, output_dim)
        self.log_std = nn.Linear(input_dim, output_dim)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean(value)
        log_std = self.log_std(value).clamp(-5.0, 2.0)
        return mean, log_std


def sample_diagonal_gaussian(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    sample: bool,
) -> torch.Tensor:
    """Sample a diagonal Gaussian, or return its mean for deterministic use."""
    if not sample:
        return mean
    return mean + torch.randn_like(mean) * log_std.exp()


def diagonal_gaussian_kl(
    posterior_mean: torch.Tensor,
    posterior_log_std: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_log_std: torch.Tensor,
) -> torch.Tensor:
    """KL(q || p) per sample and time step, summed over latent dimensions."""
    posterior_variance = (2.0 * posterior_log_std).exp()
    prior_variance = (2.0 * prior_log_std).exp()
    squared_mean_difference = (posterior_mean - prior_mean).pow(2)
    return (
        prior_log_std
        - posterior_log_std
        + 0.5 * (posterior_variance + squared_mean_difference) / prior_variance
        - 0.5
    ).sum(dim=-1)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)
