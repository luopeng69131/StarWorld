from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ChunkLog:
    raw_time_s: np.ndarray
    time_s: np.ndarray
    decision_time_s: np.ndarray
    bitrate_kbps: np.ndarray
    buffer_s: np.ndarray
    rebuffer_s: np.ndarray
    chunk_size_bytes: np.ndarray
    delay_ms: np.ndarray
    reward: np.ndarray
    intervention: np.ndarray
    safe_capacity_kbps: np.ndarray
    trace_origin_s: float

    @property
    def length(self) -> int:
        return int(self.time_s.size)

    @property
    def terminal_time_s(self) -> float:
        return float(self.time_s[-1])


def parse_chunk_log(path: str | Path) -> ChunkLog:
    """Parse a native ABR log and normalize its clock to the current trace.

    Pensieve and SafeSABR runners keep a process-level timestamp across traces
    even though they write one file per trace.  Every supported runner writes
    the first record after downloading the first chunk, so the trace origin is
    the first raw timestamp minus that first download delay.  This also leaves
    already trace-local BOLA and RobustMPC logs unchanged.
    """
    rows: list[list[float]] = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                rows.append([float(value) for value in parts])
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"no valid chunk records in {path}")

    width = max(len(row) for row in rows)
    # Keep timestamps and delays in float64.  Learned-controller logs can use
    # process-level clocks of tens of thousands of seconds, for which float32
    # subtraction would noticeably perturb the recovered trace origin.
    values = np.full((len(rows), width), np.nan, dtype=np.float64)
    for index, row in enumerate(rows):
        values[index, : len(row)] = row
    shield_log = width >= 12
    reward_column = 7 if shield_log else 6
    intervention = np.nan_to_num(values[:, 10], nan=0.0) if shield_log else np.zeros(len(rows))
    safe_capacity = values[:, 9] if shield_log else np.full(len(rows), np.nan)
    raw_time_s = values[:, 0].astype(np.float64)
    first_delay_s = float(values[0, 5]) / 1000.0
    if not np.isfinite(first_delay_s) or first_delay_s < 0.0:
        raise ValueError(f"invalid first download delay in {path}")
    trace_origin_s = float(raw_time_s[0] - first_delay_s)
    time_s = raw_time_s - trace_origin_s
    time_s[np.abs(time_s) < 1e-9] = 0.0
    if time_s[0] < -1e-6 or np.any(np.diff(time_s) < -1e-6):
        raise ValueError(f"non-monotonic trace-local timestamps in {path}")

    # The action for chunk 0 is chosen at trace start.  Each later action is
    # chosen after the preceding chunk record (including any recorded sleep).
    decision_time_s = np.concatenate(([0.0], time_s[:-1])).astype(np.float64)
    return ChunkLog(
        raw_time_s=raw_time_s,
        time_s=time_s,
        decision_time_s=decision_time_s,
        bitrate_kbps=values[:, 1],
        buffer_s=values[:, 2],
        rebuffer_s=values[:, 3],
        chunk_size_bytes=values[:, 4],
        delay_ms=values[:, 5],
        reward=values[:, reward_column],
        intervention=intervention.astype(np.float32),
        safe_capacity_kbps=safe_capacity.astype(np.float32),
        trace_origin_s=trace_origin_s,
    )


def align_chunks_to_times(
    log: ChunkLog,
    query_time_s: np.ndarray,
    *,
    planned_chunks: int = 48,
) -> dict[str, np.ndarray]:
    """Return causal streaming state and availability masks at query times.

    Before the first completed chunk and after termination, state values are
    neutral zeros and ``streaming_valid`` is zero.  The function never clips a
    pre-start query to future chunk 0 or repeats the terminal state indefinitely.
    Playback progress uses the causal completed-chunk count and the explicit
    pre-episode ``planned_chunks`` value, never the observed final log length.
    """
    if planned_chunks <= 0:
        raise ValueError("planned_chunks must be a positive pre-episode value")
    query_time_s = np.asarray(query_time_s, dtype=np.float64)
    if query_time_s.ndim != 1 or not np.isfinite(query_time_s).all():
        raise ValueError("query_time_s must be a finite one-dimensional array")
    indices = np.searchsorted(log.time_s, query_time_s, side="right") - 1
    pre_start = indices < 0
    post_termination = query_time_s > log.terminal_time_s
    valid = ~(pre_start | post_termination)
    safe_indices = np.clip(indices, 0, log.length - 1)
    cumulative_rebuffer = np.cumsum(log.rebuffer_s)
    completed_chunks = np.maximum(indices + 1, 0)

    def _masked(values: np.ndarray, scale: float = 1.0) -> np.ndarray:
        result = np.asarray(values[safe_indices] / scale, dtype=np.float32)
        result[~valid] = 0.0
        return result

    return {
        "buffer_s": _masked(log.buffer_s),
        "last_bitrate_mbps": _masked(log.bitrate_kbps, 1000.0),
        "cumulative_rebuffer_s": _masked(cumulative_rebuffer),
        "last_download_s": _masked(log.delay_ms, 1000.0),
        "playback_progress": np.where(
            valid,
            np.minimum(completed_chunks / float(planned_chunks), 1.0),
            0.0,
        ).astype(np.float32),
        "bitrate_mbps": _masked(log.bitrate_kbps, 1000.0),
        "intervention": _masked(log.intervention),
        "chunk_rebuffer_s": _masked(log.rebuffer_s),
        "reward": _masked(log.reward),
        "chunk_index": np.where(valid, indices, -1).astype(np.int64),
        "streaming_valid": valid.astype(np.float32),
        "episode_active": (query_time_s < log.terminal_time_s).astype(np.float32),
        "pre_start": pre_start.astype(np.float32),
        "post_termination": post_termination.astype(np.float32),
    }


def align_interval_actions(
    log: ChunkLog,
    start_time_s: np.ndarray,
    end_time_s: np.ndarray,
) -> dict[str, np.ndarray]:
    """Summarize decisions made in each half-open interval ``[start, end)``."""
    start_time_s = np.asarray(start_time_s, dtype=np.float64)
    end_time_s = np.asarray(end_time_s, dtype=np.float64)
    if start_time_s.shape != end_time_s.shape or start_time_s.ndim != 1:
        raise ValueError("interval boundaries must be matching one-dimensional arrays")
    if not (np.isfinite(start_time_s).all() and np.isfinite(end_time_s).all()):
        raise ValueError("interval boundaries must be finite")
    if np.any(end_time_s <= start_time_s):
        raise ValueError("each interval must have positive duration")

    left = np.searchsorted(log.decision_time_s, start_time_s, side="left")
    right = np.searchsorted(log.decision_time_s, end_time_s, side="left")
    decision_count = right - left
    bitrate = np.zeros(len(left), dtype=np.float32)
    intervention = np.zeros(len(left), dtype=np.float32)
    interval_rebuffer = np.zeros(len(left), dtype=np.float32)
    interval_reward = np.zeros(len(left), dtype=np.float32)
    completion_left = np.searchsorted(log.time_s, start_time_s, side="left")
    completion_right = np.searchsorted(log.time_s, end_time_s, side="left")
    for row, (first, stop) in enumerate(zip(left, right)):
        if stop > first:
            # The world-model clock can cover several chunk decisions.  The
            # interval action uses their mean bitrate and an explicit count so
            # the analytical transition can integrate the same number of
            # chunks without pretending that only the final action occurred.
            bitrate[row] = float(np.mean(log.bitrate_kbps[first:stop]) / 1000.0)
            intervention[row] = float(np.max(log.intervention[first:stop]))
        complete_first = completion_left[row]
        complete_stop = completion_right[row]
        if complete_stop > complete_first:
            interval_rebuffer[row] = float(np.sum(log.rebuffer_s[complete_first:complete_stop]))
            interval_reward[row] = float(np.sum(log.reward[complete_first:complete_stop]))
    return {
        "bitrate_mbps": bitrate,
        "intervention": intervention,
        "decision_count": decision_count.astype(np.float32),
        "chunk_rebuffer_s": interval_rebuffer,
        "reward": interval_reward,
    }
