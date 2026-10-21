from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StreamingState:
    buffer_s: float = 0.0
    last_bitrate_mbps: float = 0.0
    cumulative_rebuffer_s: float = 0.0
    last_download_s: float = 0.0
    playback_progress: float = 0.0
    # Internal simulator state. These fields are not model observations, but
    # preserve a chunk download that crosses a world-model interval boundary.
    inflight_remaining_megabits: float = 0.0
    inflight_bitrate_mbps: float = 0.0
    inflight_elapsed_s: float = 0.0
    inflight_rebuffer_s: float = 0.0


@dataclass
class StreamingTransition:
    state: StreamingState
    chunk_rebuffer_s: float
    bitrate_switch_mbps: float
    reward: float


def streaming_auxiliary_vector(state: StreamingState) -> np.ndarray:
    """Serialize analytical download state not exposed to the learned encoder."""
    return np.asarray(
        [
            state.inflight_remaining_megabits,
            state.inflight_bitrate_mbps,
            state.inflight_elapsed_s,
            state.inflight_rebuffer_s,
        ],
        dtype=np.float32,
    )


def apply_streaming_auxiliary(
    state: StreamingState,
    values: np.ndarray,
) -> StreamingState:
    auxiliary = np.asarray(values, dtype=float).reshape(-1)
    if auxiliary.size != 4:
        raise ValueError("streaming auxiliary state must contain four fields")
    state.inflight_remaining_megabits = max(float(auxiliary[0]), 0.0)
    state.inflight_bitrate_mbps = max(float(auxiliary[1]), 0.0)
    state.inflight_elapsed_s = max(float(auxiliary[2]), 0.0)
    state.inflight_rebuffer_s = max(float(auxiliary[3]), 0.0)
    return state


def transition_streaming_state(
    state: StreamingState,
    throughput_mbps: float,
    bitrate_mbps: float,
    chunk_duration_s: float = 4.0,
    total_chunks: int = 48,
    smoothness_weight: float = 1.0,
    rebuffer_weight: float = 4.3,
) -> StreamingTransition:
    throughput = max(float(throughput_mbps), 1e-3)
    bitrate = max(float(bitrate_mbps), 0.0)
    download_s = bitrate * chunk_duration_s / throughput
    chunk_rebuffer = max(download_s - state.buffer_s, 0.0)
    next_buffer = max(state.buffer_s - download_s, 0.0) + chunk_duration_s
    switch = abs(bitrate - state.last_bitrate_mbps)
    reward = bitrate - smoothness_weight * switch - rebuffer_weight * chunk_rebuffer
    next_state = StreamingState(
        buffer_s=next_buffer,
        last_bitrate_mbps=bitrate,
        cumulative_rebuffer_s=state.cumulative_rebuffer_s + chunk_rebuffer,
        last_download_s=download_s,
        playback_progress=min(state.playback_progress + 1.0 / max(total_chunks, 1), 1.0),
    )
    return StreamingTransition(next_state, chunk_rebuffer, switch, reward)


def transition_streaming_interval(
    state: StreamingState,
    throughput_mbps: float,
    bitrate_mbps: float,
    decision_count: int,
    chunk_duration_s: float = 4.0,
    total_chunks: int = 48,
    interval_duration_s: float = 10.0,
) -> StreamingTransition:
    """Apply the known chunk equations for all decisions in one model interval.

    StarWorld uses a 10-second network/prediction clock while native ABR
    controllers may complete several chunk decisions in that interval.  The
    interval action therefore carries both bitrate and decision count.  This
    transition preserves an in-flight chunk across interval boundaries, then
    advances playback for any remaining wall-clock time. A zero-decision
    interval therefore continues an existing download or consumes buffer
    instead of freezing the playback state.
    """
    count = max(int(round(decision_count)), 0)
    interval = max(float(interval_duration_s), 0.0)
    throughput = max(float(throughput_mbps), 1e-3)
    bitrate = max(float(bitrate_mbps), 0.0)
    buffer_s = max(float(state.buffer_s), 0.0)
    last_bitrate = max(float(state.last_bitrate_mbps), 0.0)
    cumulative_rebuffer = max(float(state.cumulative_rebuffer_s), 0.0)
    last_download = max(float(state.last_download_s), 0.0)
    progress = min(max(float(state.playback_progress), 0.0), 1.0)
    inflight_remaining = max(float(state.inflight_remaining_megabits), 0.0)
    inflight_bitrate = max(float(state.inflight_bitrate_mbps), 0.0)
    inflight_elapsed = max(float(state.inflight_elapsed_s), 0.0)
    inflight_rebuffer = max(float(state.inflight_rebuffer_s), 0.0)
    total_rebuffer = 0.0
    total_switch = 0.0
    total_reward = 0.0
    remaining_time = interval
    new_chunks = count
    epsilon = 1e-9
    while remaining_time > epsilon and (
        inflight_remaining > epsilon or new_chunks > 0
    ):
        if inflight_remaining <= epsilon:
            inflight_remaining = bitrate * chunk_duration_s
            inflight_bitrate = bitrate
            inflight_elapsed = 0.0
            inflight_rebuffer = 0.0
            new_chunks -= 1

        required_time = inflight_remaining / throughput
        spent = min(required_time, remaining_time)
        interval_rebuffer = max(spent - buffer_s, 0.0)
        buffer_s = max(buffer_s - spent, 0.0)
        cumulative_rebuffer += interval_rebuffer
        total_rebuffer += interval_rebuffer
        inflight_rebuffer += interval_rebuffer
        inflight_remaining = max(inflight_remaining - throughput * spent, 0.0)
        inflight_elapsed += spent
        remaining_time -= spent

        if inflight_remaining <= epsilon:
            bitrate_switch = abs(inflight_bitrate - last_bitrate)
            total_switch += bitrate_switch
            total_reward += (
                inflight_bitrate
                - bitrate_switch
                - 4.3 * inflight_rebuffer
            )
            buffer_s += chunk_duration_s
            last_bitrate = inflight_bitrate
            last_download = inflight_elapsed
            progress = min(
                progress + 1.0 / max(total_chunks, 1),
                1.0,
            )
            inflight_remaining = 0.0
            inflight_bitrate = 0.0
            inflight_elapsed = 0.0
            inflight_rebuffer = 0.0

    if remaining_time > 0:
        interval_rebuffer = max(remaining_time - buffer_s, 0.0)
        buffer_s = max(buffer_s - remaining_time, 0.0)
        cumulative_rebuffer += interval_rebuffer
        total_rebuffer += interval_rebuffer
    current = StreamingState(
        buffer_s=buffer_s,
        last_bitrate_mbps=last_bitrate,
        cumulative_rebuffer_s=cumulative_rebuffer,
        last_download_s=last_download,
        playback_progress=progress,
        inflight_remaining_megabits=inflight_remaining,
        inflight_bitrate_mbps=inflight_bitrate,
        inflight_elapsed_s=inflight_elapsed,
        inflight_rebuffer_s=inflight_rebuffer,
    )
    return StreamingTransition(
        state=current,
        chunk_rebuffer_s=total_rebuffer,
        bitrate_switch_mbps=total_switch,
        reward=total_reward,
    )
