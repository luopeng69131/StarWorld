from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from starworld.models.normalizer import FeatureNormalizer
from starworld.schemas import Episode


@dataclass
class NormalizerBundle:
    network: FeatureNormalizer
    prediction: FeatureNormalizer
    streaming: FeatureNormalizer
    action: FeatureNormalizer
    config: FeatureNormalizer
    context: FeatureNormalizer
    reward: FeatureNormalizer
    rebuffer: FeatureNormalizer
    network_support_center: np.ndarray | None = None
    network_support_precision: np.ndarray | None = None
    network_support_radius: float | None = None
    query_support_center: np.ndarray | None = None
    query_support_precision: np.ndarray | None = None
    query_support_radius: float | None = None

    def to_dict(self) -> dict[str, object]:
        output: dict[str, object] = {
            name: getattr(self, name).to_dict()
            for name in (
                "network",
                "prediction",
                "streaming",
                "action",
                "config",
                "context",
                "reward",
                "rebuffer",
            )
        }
        if self.network_support_center is not None:
            output["network_support_center"] = self.network_support_center.tolist()
        if self.network_support_precision is not None:
            output["network_support_precision"] = self.network_support_precision.tolist()
        if self.network_support_radius is not None:
            output["network_support_radius"] = float(self.network_support_radius)
        if self.query_support_center is not None:
            output["query_support_center"] = self.query_support_center.tolist()
        if self.query_support_precision is not None:
            output["query_support_precision"] = self.query_support_precision.tolist()
        if self.query_support_radius is not None:
            output["query_support_radius"] = float(self.query_support_radius)
        return output

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "NormalizerBundle":
        normalizers = {
            name: FeatureNormalizer.from_dict(values[name])  # type: ignore[arg-type]
            for name in ("network", "prediction", "streaming", "action", "config", "context")
        }
        for name in ("reward", "rebuffer"):
            normalizers[name] = (
                FeatureNormalizer.from_dict(values[name])  # type: ignore[arg-type]
                if name in values
                else FeatureNormalizer(
                    np.zeros(1, np.float32),
                    np.ones(1, np.float32),
                )
            )
        return cls(
            **normalizers,
            network_support_center=(
                None
                if "network_support_center" not in values
                else np.asarray(values["network_support_center"], np.float32)
            ),
            network_support_precision=(
                None
                if "network_support_precision" not in values
                else np.asarray(values["network_support_precision"], np.float32)
            ),
            network_support_radius=(
                None
                if "network_support_radius" not in values
                else float(values["network_support_radius"])  # type: ignore[arg-type]
            ),
            query_support_center=(
                None
                if "query_support_center" not in values
                else np.asarray(values["query_support_center"], np.float32)
            ),
            query_support_precision=(
                None
                if "query_support_precision" not in values
                else np.asarray(values["query_support_precision"], np.float32)
            ),
            query_support_radius=(
                None
                if "query_support_radius" not in values
                else float(values["query_support_radius"])  # type: ignore[arg-type]
            ),
        )


def _support_statistics(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    samples = np.asarray(values, dtype=np.float64)
    center = samples.mean(axis=0)
    centered = samples - center
    covariance = np.atleast_2d(np.cov(centered, rowvar=False))
    regularizer = (
        max(float(np.trace(covariance)) / max(covariance.shape[0], 1), 1.0)
        * 1e-5
    )
    precision = np.linalg.pinv(
        covariance
        + regularizer * np.eye(covariance.shape[0], dtype=np.float64)
    )
    distance = np.sqrt(
        np.maximum(
            np.einsum("ni,ij,nj->n", centered, precision, centered),
            0.0,
        )
    )
    return (
        center.astype(np.float32),
        precision.astype(np.float32),
        max(float(np.quantile(distance, 0.99)), 1e-6),
    )


def fit_normalizers(episodes: list[Episode]) -> NormalizerBundle:
    if not episodes:
        raise ValueError("cannot fit normalizers without episodes")
    # NetworkOnly episodes retain the complete measurement/prediction timeline
    # without inventing an application branch. Prefer them as the one network
    # representative per trace so controller coverage cannot change the
    # network normalizer. Streaming-side normalizers use only observed replay
    # branches; otherwise the intentionally masked zero placeholders in
    # NetworkOnly episodes would distort their scale.
    network_by_trace: dict[str, Episode] = {}
    for episode in episodes:
        previous = network_by_trace.get(episode.trace_id)
        if previous is None or (
            previous.controller != "NetworkOnly"
            and episode.controller == "NetworkOnly"
        ):
            network_by_trace[episode.trace_id] = episode
    network_episodes = list(network_by_trace.values())
    replay_episodes = [
        episode for episode in episodes if episode.controller != "NetworkOnly"
    ]
    if not replay_episodes:
        raise ValueError(
            "streaming-side normalizers require at least one real replay branch"
        )

    network_values = np.concatenate(
        [episode.network for episode in network_episodes],
        axis=0,
    ).astype(np.float64)
    support_center, precision, support_radius = _support_statistics(
        network_values
    )
    query_values = np.concatenate(
        [
            np.concatenate(
                [
                    episode.network,
                    np.repeat(
                        episode.config[None],
                        episode.length,
                        axis=0,
                    ),
                    episode.action,
                ],
                axis=1,
            )
            for episode in replay_episodes
        ],
        axis=0,
    ).astype(np.float64)
    (
        query_support_center,
        query_support_precision,
        query_support_radius,
    ) = _support_statistics(
        query_values
    )
    replay_streaming_values: list[np.ndarray] = []
    for episode in replay_episodes:
        valid = episode.targets.get("streaming_valid")
        if valid is None:
            replay_streaming_values.append(episode.streaming)
            continue
        mask = np.asarray(valid).reshape(-1) > 0.5
        if mask.any():
            replay_streaming_values.append(episode.streaming[mask])
    if not replay_streaming_values:
        raise ValueError("no causally available streaming rows for normalization")
    reward_values = [
        np.asarray(episode.targets["reward"], np.float32)
        for episode in replay_episodes
        if "reward" in episode.targets
    ]
    rebuffer_values = [
        np.asarray(episode.targets["chunk_rebuffer_s"], np.float32)
        for episode in replay_episodes
        if "chunk_rebuffer_s" in episode.targets
    ]
    if not reward_values or not rebuffer_values:
        # Synthetic and legacy fixtures may omit scalar application targets.
        reward_values = reward_values or [np.zeros((1, 1), np.float32)]
        rebuffer_values = rebuffer_values or [np.zeros((1, 1), np.float32)]
    return NormalizerBundle(
        network=FeatureNormalizer.fit(
            [episode.network for episode in network_episodes]
        ),
        prediction=FeatureNormalizer.fit(
            [episode.prediction for episode in network_episodes]
        ),
        streaming=FeatureNormalizer.fit(
            replay_streaming_values
        ),
        action=FeatureNormalizer.fit(
            [episode.action for episode in replay_episodes]
        ),
        config=FeatureNormalizer.fit(
            [episode.config[None] for episode in replay_episodes]
        ),
        context=FeatureNormalizer.fit(
            [episode.context[None] for episode in network_episodes]
        ),
        reward=FeatureNormalizer.fit(reward_values),
        rebuffer=FeatureNormalizer.fit(rebuffer_values),
        network_support_center=support_center.astype(np.float32),
        network_support_precision=precision.astype(np.float32),
        network_support_radius=support_radius,
        query_support_center=query_support_center,
        query_support_precision=query_support_precision,
        query_support_radius=query_support_radius,
    )


class SequenceDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        episodes: list[Episode],
        sequence_length: int,
        normalizers: NormalizerBundle,
        stride: int | None = None,
        use_analytical_baseline: bool = True,
    ):
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        self.episodes = episodes
        self.sequence_length = sequence_length
        self.normalizers = normalizers
        self.use_analytical_baseline = use_analytical_baseline
        self.index: list[tuple[int, int, int]] = []
        step = stride or max(sequence_length // 2, 1)
        branch_count: dict[str, int] = {}
        has_network_only: set[str] = set()
        for episode in episodes:
            branch_count[episode.trace_id] = branch_count.get(episode.trace_id, 0) + 1
            if episode.controller == "NetworkOnly":
                has_network_only.add(episode.trace_id)
        self.network_weights = {
            episode.episode_id: (
                float(episode.controller == "NetworkOnly")
                if episode.trace_id in has_network_only
                else 1.0 / branch_count[episode.trace_id]
            )
            for episode in episodes
        }
        for episode_index, episode in enumerate(episodes):
            transition_count = episode.length - 1
            if transition_count < 1:
                continue
            if transition_count <= sequence_length:
                self.index.append((episode_index, 0, transition_count))
                continue
            maximum_start = transition_count - sequence_length
            starts = list(range(0, maximum_start + 1, step))
            if starts[-1] != maximum_start:
                starts.append(maximum_start)
            self.index.extend(
                (episode_index, start, sequence_length)
                for start in starts
            )
        if not self.index:
            raise ValueError("no episode is long enough for the requested sequence length")
        replay_flags = np.asarray(
            [
                self.episodes[episode_index].controller != "NetworkOnly"
                for episode_index, _, _ in self.index
            ],
            dtype=bool,
        )
        self.sample_weights = np.ones(len(self.index), np.float64)
        if replay_flags.any() and (~replay_flags).any():
            self.sample_weights[replay_flags] = 0.5 / replay_flags.sum()
            self.sample_weights[~replay_flags] = 0.5 / (~replay_flags).sum()
        else:
            self.sample_weights.fill(1.0 / len(self.index))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_index, start, actual_steps = self.index[index]
        episode = self.episodes[episode_index]
        stop = start + actual_steps
        current = slice(start, stop)
        following = slice(start + 1, stop + 1)

        def tensor(value: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.asarray(value, np.float32))

        def pad_rows(
            value: np.ndarray,
            *,
            fill: str = "edge",
        ) -> np.ndarray:
            rows = np.asarray(value, np.float32)
            if rows.shape[0] != actual_steps:
                raise ValueError(
                    f"window has {rows.shape[0]} rows; expected {actual_steps}"
                )
            if actual_steps == self.sequence_length:
                return rows
            padding_shape = (
                self.sequence_length - actual_steps,
                *rows.shape[1:],
            )
            if fill == "edge":
                padding = np.repeat(rows[-1:], padding_shape[0], axis=0)
            elif fill == "zero":
                padding = np.zeros(padding_shape, np.float32)
            else:
                raise ValueError(f"unknown padding mode {fill}")
            return np.concatenate([rows, padding], axis=0)

        temporal_valid = np.zeros((self.sequence_length, 1), np.float32)
        temporal_valid[:actual_steps] = 1.0
        current_network = pad_rows(episode.network[current])
        current_prediction = pad_rows(episode.prediction[current])
        current_streaming = pad_rows(episode.streaming[current])
        current_action = pad_rows(episode.action[current])
        following_network = pad_rows(episode.network[following])
        following_prediction = pad_rows(episode.prediction[following])
        following_streaming_raw = pad_rows(episode.streaming[following])

        target_streaming = self.normalizers.streaming.transform(
            following_streaming_raw
        )
        if (
            self.use_analytical_baseline
            and "analytical_streaming" in episode.targets
        ):
            analytical_raw = np.asarray(
                episode.targets["analytical_streaming"][current],
                np.float32,
            )
            if analytical_raw.shape != episode.streaming[current].shape:
                raise ValueError(
                    "analytical_streaming must have shape [time, streaming_dim]"
                )
            analytical_raw = pad_rows(analytical_raw)
            analytical_available = temporal_valid.copy()
        else:
            # A causal identity baseline keeps the residual target well defined.
            # It is not reported as an analytical baseline.
            analytical_raw = current_streaming.copy()
            analytical_available = np.zeros((self.sequence_length, 1), np.float32)
        analytical_streaming = self.normalizers.streaming.transform(analytical_raw)
        analytical_inflight_source = episode.targets.get(
            "analytical_inflight",
            np.zeros((episode.length, 4), np.float32),
        )
        initial_analytical_inflight = np.asarray(
            analytical_inflight_source[start],
            np.float32,
        )
        if initial_analytical_inflight.shape != (4,):
            raise ValueError("analytical_inflight must have shape [time, 4]")

        def scalar_target(name: str) -> tuple[np.ndarray, np.ndarray]:
            value = episode.targets.get(name)
            if value is None:
                return (
                    np.zeros((self.sequence_length, 1), np.float32),
                    np.zeros((self.sequence_length, 1), np.float32),
                )
            selected = np.asarray(value[current], np.float32)
            if selected.ndim == 1:
                selected = selected[:, None]
            if selected.shape != (actual_steps, 1):
                raise ValueError(
                    f"{name} must have shape [{actual_steps}, 1]"
                )
            return (
                pad_rows(selected, fill="zero"),
                temporal_valid.copy(),
            )

        reward, reward_available = scalar_target("reward")
        rebuffer, rebuffer_available = scalar_target("chunk_rebuffer_s")
        reward = self.normalizers.reward.transform(reward)
        rebuffer = self.normalizers.rebuffer.transform(rebuffer)
        prediction_risk = pad_rows(
            np.asarray(
                episode.targets["overestimation"][following],
                np.float32,
            ),
            fill="zero",
        )
        streaming_risk_source = episode.targets.get(
            "severe_streaming",
            episode.targets["high_risk"],
        )
        streaming_risk = pad_rows(
            np.asarray(streaming_risk_source[following], np.float32),
            fill="zero",
        )
        streaming_valid_source = episode.targets.get(
            "streaming_valid",
            np.ones((episode.length, 1), np.float32),
        )
        streaming_valid = pad_rows(
            np.asarray(
                streaming_valid_source[following],
                np.float32,
            ),
            fill="zero",
        )
        streaming_valid *= temporal_valid
        terminal_source = episode.targets.get(
            "terminal_within_interval",
            None,
        )
        continuation_source = episode.targets.get(
            "episode_active",
            np.ones((episode.length, 1), np.float32),
        )
        if terminal_source is None:
            continuation = pad_rows(
                np.asarray(
                    continuation_source[following],
                    np.float32,
                ),
                fill="zero",
            )
        else:
            continuation = pad_rows(
                1.0
                - np.asarray(
                    terminal_source[current],
                    np.float32,
                ),
                fill="zero",
            )
        return {
            "network": tensor(self.normalizers.network.transform(current_network)),
            "prediction": tensor(self.normalizers.prediction.transform(current_prediction)),
            "streaming": tensor(self.normalizers.streaming.transform(current_streaming)),
            "action": tensor(self.normalizers.action.transform(current_action)),
            "config": tensor(self.normalizers.config.transform(episode.config)),
            "context": tensor(self.normalizers.context.transform(episode.context)),
            "target_network": tensor(self.normalizers.network.transform(following_network)),
            "target_prediction": tensor(self.normalizers.prediction.transform(following_prediction)),
            "target_streaming": tensor(target_streaming),
            "target_streaming_valid": tensor(streaming_valid),
            "target_temporal_valid": tensor(temporal_valid),
            "analytical_streaming": tensor(analytical_streaming),
            "initial_analytical_inflight": tensor(
                initial_analytical_inflight
            ),
            "target_streaming_residual": tensor(
                target_streaming - analytical_streaming
            ),
            "analytical_baseline_available": tensor(analytical_available),
            "target_reward": tensor(reward),
            "reward_available": tensor(reward_available),
            "target_rebuffer": tensor(rebuffer),
            "rebuffer_available": tensor(rebuffer_available),
            "target_continuation": tensor(continuation),
            "target_prediction_risk": tensor(prediction_risk),
            "target_streaming_risk": tensor(streaming_risk),
            "target_events": tensor(
                np.concatenate([prediction_risk, streaming_risk], axis=-1)
            ),
            "network_weight": torch.tensor(self.network_weights[episode.episode_id], dtype=torch.float32),
        }
