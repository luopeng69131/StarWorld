from __future__ import annotations

import numpy as np

from starworld.interfaces.streaming import (StreamingState, transition_streaming_interval,
                                           streaming_auxiliary_vector)
from starworld.schemas import Episode


def make_synthetic_episodes(
    count: int = 18,
    steps: int = 36,
    seed: int = 11,
) -> list[Episode]:
    """Generate cross-layer sequences for the runnable example and unit tests."""
    rng = np.random.default_rng(seed)
    episodes: list[Episode] = []
    controllers = ("BOLA", "RobustMPC", "Pensieve", "SafeSABR")
    regions = ("US", "OSN", "VIC")
    for index in range(count):
        source_run = f"synthetic_run_{index // 2:02d}"
        trace_id = f"synthetic_trace_{index:03d}"
        region = regions[index % len(regions)]
        controller = controllers[index % len(controllers)]
        throughput = np.zeros(steps, np.float32)
        throughput[0] = rng.uniform(40.0, 120.0)
        for step in range(1, steps):
            throughput[step] = max(
                2.0,
                0.88 * throughput[step - 1]
                + 0.12 * rng.uniform(30.0, 150.0)
                + rng.normal(0.0, 7.0),
            )
        history_mean = np.asarray(
            [throughput[max(0, step - 5) : step + 1].mean() for step in range(steps)]
        )
        history_p05 = np.asarray(
            [np.quantile(throughput[max(0, step - 5) : step + 1], 0.05) for step in range(steps)]
        )
        history_std = np.asarray(
            [throughput[max(0, step - 5) : step + 1].std() for step in range(steps)]
        )
        network = np.column_stack(
            [
                throughput,
                history_p05,
                np.minimum(history_p05, throughput * 0.8),
                history_mean,
                history_p05,
                0.5 * (history_mean + history_p05),
                history_std,
                np.linspace(0.0, 1.0, steps),
            ]
        ).astype(np.float32)
        point = history_mean
        safe = history_p05
        prediction = np.column_stack(
            [point, safe, np.maximum(point - safe, 0.0), np.full(steps, 10.0)]
        ).astype(np.float32)
        state = StreamingState()
        streaming_rows = []
        action_rows = []
        chunk_rebuffer = []
        reward = []
        analytical_rows, auxiliary_rows = [], []
        for step in range(steps):
            bitrate = float(max(0.3, min(120.0, safe[step] * (0.75 + 0.05 * (index % 4)))))
            streaming_rows.append(
                [
                    state.buffer_s,
                    state.last_bitrate_mbps,
                    state.cumulative_rebuffer_s,
                    state.last_download_s,
                    state.playback_progress,
                ]
            )
            decisions = int(np.clip(10.0 * safe[step] / max(4 * bitrate, 1e-3), 1, 8))
            action_rows.append(
                [bitrate, float(controller == "SafeSABR" and state.buffer_s < 8.0), decisions]
            )
            auxiliary_rows.append(streaming_auxiliary_vector(state))
            transition = transition_streaming_interval(state, throughput[min(step + 1, steps - 1)],
                                                       bitrate, decisions)
            state = transition.state
            analytical_rows.append([state.buffer_s, state.last_bitrate_mbps,
                                    state.cumulative_rebuffer_s, state.last_download_s,
                                    state.playback_progress])
            chunk_rebuffer.append([transition.chunk_rebuffer_s])
            reward.append([transition.reward])
        streaming = np.asarray(streaming_rows, np.float32)
        action = np.asarray(action_rows, np.float32)
        config = np.asarray(
            [
                controller == "BOLA",
                controller == "RobustMPC",
                controller == "Pensieve",
                controller == "SafeSABR",
                controller == "SafeSABR",
                120.0,
                4.0,
            ],
            np.float32,
        )
        context = np.asarray(
            [
                region == "US",
                region == "OSN",
                region == "VIC",
            ],
            np.float32,
        )
        future_mean = np.roll(throughput, -1)
        future_mean[-1] = future_mean[-2]
        future_p05 = np.minimum(future_mean, np.roll(history_p05, -1))
        future_min = np.minimum(future_p05, future_mean * 0.8)
        overestimation = (point > 1.25 * future_mean).astype(np.float32)[:, None]
        high_risk = ((future_p05 < 20.0) | (np.asarray(chunk_rebuffer).reshape(-1) > 0.5)).astype(np.float32)[:, None]
        episode = Episode(
            episode_id=f"{trace_id}:{controller}",
            source_run_id=source_run,
            trace_id=trace_id,
            branch_id=controller,
            region=region,
            controller=controller,
            time_s=np.arange(steps, dtype=np.float32) * 10.0,
            network=network,
            prediction=prediction,
            streaming=streaming,
            action=action,
            config=config,
            context=context,
            targets={
                "future_network": np.column_stack([future_mean, future_p05, future_min]).astype(np.float32),
                "overestimation": overestimation,
                "high_risk": high_risk,
                "chunk_rebuffer_s": np.asarray(chunk_rebuffer, np.float32),
                "reward": np.asarray(reward, np.float32),
                "analytical_inflight": np.asarray(auxiliary_rows, np.float32),
                "analytical_streaming": np.asarray(analytical_rows, np.float32),
            },
            metadata={"synthetic_fixture": True},
        )
        episode.validate()
        episodes.append(episode)
    return episodes
