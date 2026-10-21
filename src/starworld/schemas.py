from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


NETWORK_FEATURES = (
    "observed_window_mean_mbps",
    "observed_window_p05_mbps",
    "observed_window_min_mbps",
    "history_mean_mbps",
    "history_p05_mbps",
    "history_p25_mbps",
    "history_std_mbps",
    "window_progress",
)
PREDICTION_FEATURES = (
    "point_prediction_mbps",
    "safe_capacity_mbps",
    "prediction_spread_mbps",
    "prediction_horizon_s",
)
STREAMING_FEATURES = (
    "buffer_s",
    "last_bitrate_mbps",
    "cumulative_rebuffer_s",
    "last_download_s",
    "playback_progress",
)
ACTION_FEATURES = ("bitrate_mbps", "intervention", "decision_count")
CONFIG_FEATURES = (
    "controller_bola",
    "controller_robustmpc",
    "controller_pensieve",
    "controller_safesabr",
    "uses_safe_capacity",
    "bitrate_cap_mbps",
    "chunk_duration_s",
)
CONTEXT_FEATURES = (
    "region_us",
    "region_osn",
    "region_vic",
)


@dataclass
class Episode:
    episode_id: str
    source_run_id: str
    trace_id: str
    branch_id: str
    region: str
    controller: str
    time_s: np.ndarray
    network: np.ndarray
    prediction: np.ndarray
    streaming: np.ndarray
    action: np.ndarray
    config: np.ndarray
    context: np.ndarray
    targets: dict[str, np.ndarray]
    metadata: dict[str, Any]

    @property
    def length(self) -> int:
        return int(self.time_s.shape[0])

    def validate(self) -> None:
        length = self.length
        for name in ("network", "prediction", "streaming", "action"):
            value = getattr(self, name)
            if value.ndim != 2 or value.shape[0] != length:
                raise ValueError(f"{name} has invalid shape {value.shape}; expected T={length}")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")
        if self.config.ndim != 1 or self.context.ndim != 1:
            raise ValueError("config and context must be one-dimensional")
        if length < 2:
            raise ValueError("episode must contain at least two steps")


@dataclass
class SplitManifest:
    split_by_source_run: dict[str, str]
    train_ids: list[str]
    validation_ids: list[str]
    test_ids: list[str]
    excluded_ids: list[str] | None = None
    protocol: str = "source_run_random"
    holdout_value: str | None = None
