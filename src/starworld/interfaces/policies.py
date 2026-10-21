from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from starworld.interfaces.streaming import StreamingState


DEFAULT_LADDER = np.asarray([0.3, 0.75, 1.2, 1.85, 2.85, 4.3, 8.0, 15.0, 30.0, 60.0, 120.0])


def _floor_ladder(target: float, ladder: np.ndarray) -> float:
    available = ladder[ladder <= target]
    return float(available[-1] if available.size else ladder[0])


class StreamingPolicy(ABC):
    name: str

    @abstractmethod
    def action(
        self,
        state: StreamingState,
        capacity_mbps: float,
        uncertainty_mbps: float = 0.0,
    ) -> float:
        """Return the next chunk bitrate in Mbps."""


class FixedBitratePolicy(StreamingPolicy):
    name = "fixed"

    def __init__(self, bitrate_mbps: float):
        self.bitrate_mbps = bitrate_mbps

    def action(self, state: StreamingState, capacity_mbps: float, uncertainty_mbps: float = 0.0) -> float:
        del state, capacity_mbps, uncertainty_mbps
        return self.bitrate_mbps


class ThroughputFractionPolicy(StreamingPolicy):
    name = "throughput_fraction"

    def __init__(self, fraction: float = 0.85, ladder: np.ndarray = DEFAULT_LADDER):
        self.fraction = fraction
        self.ladder = np.asarray(ladder)

    def action(self, state: StreamingState, capacity_mbps: float, uncertainty_mbps: float = 0.0) -> float:
        del state
        target = max(capacity_mbps - uncertainty_mbps, 0.0) * self.fraction
        return _floor_ladder(target, self.ladder)


class BufferPolicy(StreamingPolicy):
    name = "buffer"

    def __init__(self, reservoir_s: float = 8.0, cushion_s: float = 24.0, ladder: np.ndarray = DEFAULT_LADDER):
        self.reservoir_s = reservoir_s
        self.cushion_s = cushion_s
        self.ladder = np.asarray(ladder)

    def action(self, state: StreamingState, capacity_mbps: float, uncertainty_mbps: float = 0.0) -> float:
        del uncertainty_mbps
        position = np.clip((state.buffer_s - self.reservoir_s) / max(self.cushion_s, 1e-3), 0.0, 1.0)
        buffer_target = self.ladder[0] + position * (self.ladder[-1] - self.ladder[0])
        return _floor_ladder(min(buffer_target, max(capacity_mbps, self.ladder[0])), self.ladder)
