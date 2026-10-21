from __future__ import annotations

import json
import re
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
else:
    try:
        import pandas as pd
    except ModuleNotFoundError:
        pd = None  # type: ignore[assignment]

from starworld.config import DataConfig
from starworld.data.leakage import require_causal_features, require_causal_provenance
from starworld.data.replay_logs import align_chunks_to_times, align_interval_actions, parse_chunk_log
from starworld.data.sources import load_source_tables
from starworld.data.split import assert_no_group_leakage, grouped_source_run_split
from starworld.interfaces.streaming import (
    apply_streaming_auxiliary,
    StreamingState,
    streaming_auxiliary_vector,
    transition_streaming_interval,
)
from starworld.schemas import (
    ACTION_FEATURES,
    CONFIG_FEATURES,
    CONTEXT_FEATURES,
    NETWORK_FEATURES,
    PREDICTION_FEATURES,
    STREAMING_FEATURES,
    Episode,
)

DEFAULT_BITRATE_CAP_MBPS = 120.0
DEFAULT_CHUNK_DURATION_S = 4.0
DEFAULT_PLANNED_CHUNKS = 48

MODEL_INPUT_PROVENANCE: dict[str, dict[str, Any]] = {
    "observed_window_mean_mbps": {
        "availability": "lagged_completed_measurement",
        "sources": ["history_mean_mbps", "future_mean_mbps"],
        "transform": "shift_completed_window_by_one_step",
    },
    "observed_window_p05_mbps": {
        "availability": "lagged_completed_measurement",
        "sources": ["history_p05_mbps", "future_p05_mbps"],
        "transform": "shift_completed_window_by_one_step",
    },
    "observed_window_min_mbps": {
        "availability": "lagged_completed_measurement",
        "sources": ["history_p05_mbps", "future_min_mbps"],
        "transform": "shift_completed_window_by_one_step",
    },
    "history_mean_mbps": {
        "availability": "causal_measurement_history",
        "sources": ["history_mean_mbps"],
    },
    "history_p05_mbps": {
        "availability": "causal_measurement_history",
        "sources": ["history_p05_mbps"],
    },
    "history_p25_mbps": {
        "availability": "causal_measurement_history",
        "sources": ["history_p25_mbps"],
    },
    "history_std_mbps": {
        "availability": "causal_measurement_history",
        "sources": ["history_std_mbps"],
    },
    "window_progress": {
        "availability": "explicit_pre_episode_config",
        "sources": ["window_start_s", "segment_duration_s"],
        "transform": "causal_clock_over_declared_segment_duration",
    },
    "point_prediction_mbps": {
        "availability": "causal_prediction_output",
        "sources": ["point_prediction_mbps"],
    },
    "safe_capacity_mbps": {
        "availability": "causal_prediction_output",
        "sources": ["heuristic_safe_capacity_mbps"],
    },
    "prediction_spread_mbps": {
        "availability": "causal_prediction_output",
        "sources": ["point_prediction_mbps", "heuristic_safe_capacity_mbps"],
    },
    "prediction_horizon_s": {
        "availability": "explicit_pre_episode_config",
        "sources": ["window_start_s", "window_end_s"],
    },
    "buffer_s": {
        "availability": "causal_replay_prefix",
        "sources": ["chunk_buffer_s", "chunk_completion_time_s"],
    },
    "last_bitrate_mbps": {
        "availability": "causal_replay_prefix",
        "sources": ["chunk_bitrate_kbps", "chunk_completion_time_s"],
    },
    "cumulative_rebuffer_s": {
        "availability": "causal_replay_prefix",
        "sources": ["chunk_rebuffer_s", "chunk_completion_time_s"],
        "transform": "prefix_sum_through_query_time",
    },
    "last_download_s": {
        "availability": "causal_replay_prefix",
        "sources": ["chunk_delay_ms", "chunk_completion_time_s"],
    },
    "playback_progress": {
        "availability": "causal_replay_prefix",
        "sources": ["completed_chunk_count", "protocol_planned_chunks"],
        "transform": "causal_completed_count_over_explicit_plan",
    },
    "bitrate_mbps": {
        "availability": "causal_interval_decisions",
        "sources": ["chunk_bitrate_kbps", "chunk_decision_time_s"],
    },
    "intervention": {
        "availability": "causal_interval_decisions",
        "sources": ["chunk_intervention", "chunk_decision_time_s"],
    },
    "decision_count": {
        "availability": "causal_interval_decisions",
        "sources": ["chunk_decision_time_s"],
    },
    "controller_bola": {
        "availability": "explicit_pre_episode_config",
        "sources": ["controller"],
    },
    "controller_robustmpc": {
        "availability": "explicit_pre_episode_config",
        "sources": ["controller"],
    },
    "controller_pensieve": {
        "availability": "explicit_pre_episode_config",
        "sources": ["controller"],
    },
    "controller_safesabr": {
        "availability": "explicit_pre_episode_config",
        "sources": ["controller"],
    },
    "uses_safe_capacity": {
        "availability": "explicit_pre_episode_config",
        "sources": ["controller_protocol"],
        "transform": "declared_controller_interface",
    },
    "bitrate_cap_mbps": {
        "availability": "explicit_pre_episode_config",
        "sources": ["benchmark_bitrate_ladder"],
        "transform": "declared_protocol_constant",
    },
    "chunk_duration_s": {
        "availability": "explicit_pre_episode_config",
        "sources": ["benchmark_video_manifest"],
        "transform": "declared_protocol_constant",
    },
    "region_us": {
        "availability": "exogenous_context",
        "sources": ["region"],
    },
    "region_osn": {
        "availability": "exogenous_context",
        "sources": ["region"],
    },
    "region_vic": {
        "availability": "exogenous_context",
        "sources": ["region"],
    },
}


def _numeric(frame: pd.DataFrame, name: str, default: float = 0.0) -> np.ndarray:
    if name not in frame:
        return np.full(len(frame), default, dtype=np.float32)
    return pd.to_numeric(frame[name], errors="coerce").fillna(default).to_numpy(np.float32)


def lagged_observation(
    frame: pd.DataFrame,
    initial_history_column: str,
    completed_window_column: str,
) -> np.ndarray:
    """Expose a completed measurement window only at the following timestamp."""
    initial = _numeric(frame, initial_history_column)[:1]
    completed = _numeric(frame, completed_window_column)
    return np.concatenate([initial, completed[:-1]])


def canonical_trace_key(value: str) -> str:
    """Normalize legacy archive/member spellings only for cross-table joins."""
    return re.sub(r"_+", "_", re.sub(r"[^a-zA-Z0-9]+", "_", str(value))).strip("_").lower()


def _unique_key_map(values: list[str], label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    collisions: dict[str, list[str]] = {}
    for value in values:
        key = canonical_trace_key(value)
        if key in result and result[key] != value:
            collisions.setdefault(key, [result[key]]).append(value)
        result[key] = value
    if collisions:
        examples = dict(list(collisions.items())[:3])
        raise ValueError(f"ambiguous canonical {label} keys: {examples}")
    return result


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _config_vector(controller: str) -> np.ndarray:
    """Return benchmark configuration declared before replay begins."""
    name = controller.lower()
    if name not in {
        "bola",
        "robustmpc",
        "pensieve",
        "safesabr",
        "networkonly",
    }:
        raise ValueError(f"unsupported controller configuration: {controller}")
    return np.asarray(
        [
            name == "bola",
            name == "robustmpc",
            name == "pensieve",
            name == "safesabr",
            name == "safesabr",
            DEFAULT_BITRATE_CAP_MBPS,
            DEFAULT_CHUNK_DURATION_S,
        ],
        dtype=np.float32,
    )


def _context_vector(region: str) -> np.ndarray:
    region_name = str(region).lower()
    return np.asarray(
        [
            region_name == "us",
            region_name == "osn",
            region_name == "vic",
        ],
        dtype=np.float32,
    )


def _streaming_state_from_row(row: np.ndarray) -> StreamingState:
    return StreamingState(
        buffer_s=float(row[0]),
        last_bitrate_mbps=float(row[1]),
        cumulative_rebuffer_s=float(row[2]),
        last_download_s=float(row[3]),
        playback_progress=float(row[4]),
    )


def _streaming_row(state: StreamingState) -> np.ndarray:
    return np.asarray(
        [
            state.buffer_s,
            state.last_bitrate_mbps,
            state.cumulative_rebuffer_s,
            state.last_download_s,
            state.playback_progress,
        ],
        dtype=np.float32,
    )


class CrossLayerExperienceConstructor:
    """Build source-traceable, action-branched cross-layer episodes."""

    def __init__(self, config: DataConfig):
        self.config = config
        self.input_features = (
            list(NETWORK_FEATURES)
            + list(PREDICTION_FEATURES)
            + list(STREAMING_FEATURES)
            + list(ACTION_FEATURES)
            + list(CONFIG_FEATURES)
            + list(CONTEXT_FEATURES)
        )
        require_causal_features(self.input_features)
        self.provenance_audit = require_causal_provenance(
            self.input_features,
            MODEL_INPUT_PROVENANCE,
        )

    def build(self) -> tuple[list[Episode], dict[str, Any]]:
        if pd is None:
            raise ModuleNotFoundError(
                "pandas is required to construct cross-layer episodes from CSV files; "
                "load_episode_archive remains available without pandas"
            )
        segments, prediction, abr = load_source_tables(
            self.config.segments_csv,
            self.config.prediction_csv,
            self.config.abr_outcomes_csv,
        )
        segment_by_trace = segments.drop_duplicates("trace_id").set_index("trace_id")
        prediction_groups_raw = {
            trace_id: group.sort_values("window_start_s").reset_index(drop=True)
            for trace_id, group in prediction.groupby("trace_id", sort=False)
        }
        segment_keys = _unique_key_map(segment_by_trace.index.astype(str).tolist(), "segment")
        prediction_keys = _unique_key_map(list(prediction_groups_raw), "prediction")
        episodes: list[Episode] = []
        excluded: list[dict[str, str]] = []
        for _, branch in abr.iterrows():
            trace_id = str(branch["trace_id"])
            key = canonical_trace_key(trace_id)
            segment_id = segment_keys.get(key)
            prediction_id = prediction_keys.get(key)
            if segment_id is None or prediction_id is None:
                excluded.append({"trace_id": trace_id, "reason": "missing_prediction_or_segment"})
                continue
            log_path = Path(str(branch["log_path"]))
            if not log_path.is_absolute():
                log_path = Path(self.config.abr_outcomes_csv).parent / log_path
            branch = branch.copy()
            branch["log_path"] = str(log_path)
            if not log_path.exists():
                excluded.append({"trace_id": trace_id, "reason": "missing_chunk_log"})
                continue
            try:
                episode = self._build_branch(
                    segment_by_trace.loc[segment_id],
                    prediction_groups_raw[prediction_id],
                    branch,
                )
                if episode.length < self.config.minimum_replay_steps:
                    excluded.append({"trace_id": trace_id, "reason": "too_few_aligned_steps"})
                    continue
                episode.validate()
                episodes.append(episode)
            except (OSError, ValueError) as error:
                excluded.append({"trace_id": trace_id, "reason": f"parse_error:{error}"})
            if self.config.maximum_episodes and len(episodes) >= self.config.maximum_episodes:
                break
        if self.config.include_network_only:
            for prediction_id, windows in prediction_groups_raw.items():
                segment_id = segment_keys.get(canonical_trace_key(prediction_id))
                if segment_id is None:
                    excluded.append(
                        {
                            "trace_id": str(prediction_id),
                            "reason": "network_only_missing_segment",
                        }
                    )
                    continue
                episode = self._build_network_only(
                    segment_by_trace.loc[segment_id],
                    windows,
                )
                if episode.length >= self.config.minimum_steps:
                    episode.validate()
                    episodes.append(episode)
        if not episodes:
            raise RuntimeError("no cross-layer episodes could be constructed")

        replay_actions = [episode.action for episode in episodes if episode.controller != "NetworkOnly"]
        combined_actions = np.concatenate(replay_actions) if replay_actions else np.zeros((0, 3))
        split = grouped_source_run_split(
            episodes,
            train_fraction=self.config.train_fraction,
            validation_fraction=self.config.validation_fraction,
            seed=self.config.split_seed,
        )
        assert_no_group_leakage(episodes, split)
        audit = {
            "episode_count": len(episodes),
            "network_trace_count": len({episode.trace_id for episode in episodes}),
            "source_run_count": len({episode.source_run_id for episode in episodes}),
            "controller_counts": pd.Series(
                [episode.controller for episode in episodes]
            ).value_counts().to_dict(),
            "replay_branch_count": int(
                sum(episode.controller != "NetworkOnly" for episode in episodes)
            ),
            "network_only_episode_count": int(
                sum(episode.controller == "NetworkOnly" for episode in episodes)
            ),
            "region_trace_counts": (
                pd.DataFrame(
                    {
                        "trace_id": [episode.trace_id for episode in episodes],
                        "region": [episode.region for episode in episodes],
                    }
                )
                .drop_duplicates()
                ["region"]
                .value_counts()
                .to_dict()
            ),
            "action_coverage": {
                "intervals": int(
                    sum(
                        episode.length
                        for episode in episodes
                        if episode.controller != "NetworkOnly"
                    )
                ),
                "intervals_with_decision": int(
                    sum(
                        (episode.action[:, 2] > 0).sum()
                        for episode in episodes
                        if episode.controller != "NetworkOnly"
                    )
                ),
                "mean_decisions_per_interval": float(combined_actions[:, 2].mean()) if len(combined_actions) else 0.0,
                "unique_bitrates_mbps": sorted(np.unique(combined_actions[:, 0]).tolist()),
            },
            "split_counts": {
                "train": len(split.train_ids),
                "validation": len(split.validation_ids),
                "test": len(split.test_ids),
            },
            "excluded_count": len(excluded),
            "excluded_reason_counts": pd.Series(
                [row["reason"].split(":", 1)[0] for row in excluded],
                dtype="object",
            ).value_counts().to_dict(),
            "excluded_examples": excluded[:20],
            "split_by_source_run": split.split_by_source_run,
            "leakage_audit": self.provenance_audit,
            "input_hashes": {
                "segments_csv": _sha256(self.config.segments_csv),
                "prediction_csv": _sha256(self.config.prediction_csv),
                "abr_outcomes_csv": _sha256(self.config.abr_outcomes_csv),
            },
            "feature_schema": {
                "network": list(NETWORK_FEATURES),
                "prediction": list(PREDICTION_FEATURES),
                "streaming": list(STREAMING_FEATURES),
                "action": list(ACTION_FEATURES),
                "config": list(CONFIG_FEATURES),
                "context": list(CONTEXT_FEATURES),
            },
        }
        return episodes, audit

    def _build_network_only(
        self,
        segment: pd.Series,
        windows: pd.DataFrame,
    ) -> Episode:
        windows = windows.sort_values("window_start_s").reset_index(drop=True)
        time_s = _numeric(windows, "window_start_s")
        duration = max(
            float(segment.get("duration_s", time_s.max() + 10.0)),
            1.0,
        )
        network = np.column_stack(
            [
                lagged_observation(
                    windows,
                    "history_mean_mbps",
                    "future_mean_mbps",
                ),
                lagged_observation(
                    windows,
                    "history_p05_mbps",
                    "future_p05_mbps",
                ),
                lagged_observation(
                    windows,
                    "history_p05_mbps",
                    "future_min_mbps",
                ),
                _numeric(windows, "history_mean_mbps"),
                _numeric(windows, "history_p05_mbps"),
                _numeric(windows, "history_p25_mbps"),
                _numeric(windows, "history_std_mbps"),
                time_s / duration,
            ]
        ).astype(np.float32)
        point = _numeric(windows, "point_prediction_mbps")
        safe = _numeric(windows, "heuristic_safe_capacity_mbps", 0.0)
        prediction = np.column_stack(
            [
                point,
                safe,
                np.maximum(point - safe, 0.0),
                _numeric(windows, "window_end_s") - time_s,
            ]
        ).astype(np.float32)
        length = len(windows)
        labels = str(
            segment.get(
                "paper_profile_labels",
                windows.iloc[0].get("paper_profile_labels", ""),
            )
        )
        branch_id = "NetworkOnly:measurement_prediction"
        zeros = np.zeros
        return Episode(
            episode_id=f"{segment.name}:{branch_id}",
            source_run_id=str(segment["source_run_id"]),
            trace_id=str(segment.name),
            branch_id=branch_id,
            region=str(segment["region"]),
            controller="NetworkOnly",
            time_s=time_s,
            network=network,
            prediction=prediction,
            streaming=zeros((length, len(STREAMING_FEATURES)), np.float32),
            action=zeros((length, len(ACTION_FEATURES)), np.float32),
            config=_config_vector("networkonly"),
            context=_context_vector(str(segment["region"])),
            targets={
                "future_network": np.column_stack(
                    [
                        _numeric(windows, "future_mean_mbps"),
                        _numeric(windows, "future_p05_mbps"),
                        _numeric(windows, "future_min_mbps"),
                    ]
                ).astype(np.float32),
                "overestimation": _numeric(
                    windows,
                    "overestimation_label",
                )[:, None],
                "high_risk": _numeric(
                    windows,
                    "high_risk_label",
                )[:, None],
                "severe_streaming": zeros((length, 1), np.float32),
                "streaming_valid": zeros((length, 1), np.float32),
                "analytical_inflight": zeros((length, 4), np.float32),
                "episode_active": np.ones((length, 1), np.float32),
                "terminal_within_interval": zeros((length, 1), np.float32),
            },
            metadata={
                "segment_path": str(segment.get("path", "")),
                "profile_labels": labels,
                "branch_type": "network_prediction_only",
                "streaming_supervision": False,
                "measurement_semantics": (
                    "network observation at t uses the measurement window "
                    "completed before t; no streaming branch is synthesized"
                ),
                "input_provenance_reference": "audit.json:leakage_audit",
            },
        )

    def _build_branch(
        self,
        segment: pd.Series,
        windows: pd.DataFrame,
        branch: pd.Series,
    ) -> Episode:
        log = parse_chunk_log(branch["log_path"])
        all_time_s = _numeric(windows, "window_start_s")
        # A replay branch has no active streaming state after its final chunk.
        # Removing those rows avoids training on an indefinitely frozen terminal
        # state while preserving pre-first-completion rows with an explicit mask.
        causal_rows = all_time_s <= log.terminal_time_s
        windows = windows.loc[causal_rows].reset_index(drop=True)
        if windows.empty:
            raise ValueError("replay terminates before the first prediction window")
        time_s = _numeric(windows, "window_start_s")
        duration = max(float(segment.get("duration_s", time_s.max() + 10.0)), 1.0)
        labels = str(segment.get("paper_profile_labels", windows.iloc[0].get("paper_profile_labels", "")))

        network = np.column_stack(
            [
                lagged_observation(windows, "history_mean_mbps", "future_mean_mbps"),
                lagged_observation(windows, "history_p05_mbps", "future_p05_mbps"),
                lagged_observation(windows, "history_p05_mbps", "future_min_mbps"),
                _numeric(windows, "history_mean_mbps"),
                _numeric(windows, "history_p05_mbps"),
                _numeric(windows, "history_p25_mbps"),
                _numeric(windows, "history_std_mbps"),
                time_s / duration,
            ]
        ).astype(np.float32)
        point = _numeric(windows, "point_prediction_mbps")
        safe = _numeric(windows, "heuristic_safe_capacity_mbps", 0.0)
        prediction = np.column_stack(
            [
                point,
                safe,
                np.maximum(point - safe, 0.0),
                _numeric(windows, "window_end_s") - time_s,
            ]
        ).astype(np.float32)
        aligned = align_chunks_to_times(
            log,
            time_s,
            planned_chunks=DEFAULT_PLANNED_CHUNKS,
        )
        streaming = np.column_stack([aligned[name] for name in STREAMING_FEATURES]).astype(np.float32)
        interval_action = align_interval_actions(
            log,
            time_s,
            _numeric(windows, "window_end_s"),
        )
        action = np.column_stack([interval_action[name] for name in ACTION_FEATURES]).astype(np.float32)
        config = _config_vector(str(branch["controller"]))
        context = _context_vector(str(segment["region"]))
        analytical_streaming = np.zeros_like(streaming)
        analytical_inflight = np.zeros((len(streaming), 4), np.float32)
        interval_throughput = _numeric(windows, "future_mean_mbps")
        analytical_state: StreamingState | None = None
        for index in range(len(streaming)):
            interval_state = _streaming_state_from_row(streaming[index])
            if analytical_state is not None:
                interval_state = apply_streaming_auxiliary(
                    interval_state,
                    streaming_auxiliary_vector(analytical_state),
                )
            analytical_inflight[index] = streaming_auxiliary_vector(
                interval_state
            )
            transition = transition_streaming_interval(
                interval_state,
                throughput_mbps=float(interval_throughput[index]),
                bitrate_mbps=float(action[index, 0]),
                decision_count=int(round(float(action[index, 2]))),
                chunk_duration_s=DEFAULT_CHUNK_DURATION_S,
                total_chunks=DEFAULT_PLANNED_CHUNKS,
            )
            analytical_state = transition.state
            analytical_streaming[index] = _streaming_row(
                transition.state
            )
        targets = {
            "future_network": np.column_stack(
                [
                    _numeric(windows, "future_mean_mbps"),
                    _numeric(windows, "future_p05_mbps"),
                    _numeric(windows, "future_min_mbps"),
                ]
            ).astype(np.float32),
            "overestimation": _numeric(windows, "overestimation_label")[:, None],
            "high_risk": _numeric(windows, "high_risk_label")[:, None],
            "chunk_rebuffer_s": interval_action["chunk_rebuffer_s"][:, None].astype(np.float32),
            "reward": interval_action["reward"][:, None].astype(np.float32),
            "analytical_streaming": analytical_streaming,
            "analytical_inflight": analytical_inflight,
            "severe_streaming": (
                streaming[:, 2] >= 10.0
            )[:, None].astype(np.float32),
            "streaming_valid": aligned["streaming_valid"][:, None].astype(np.float32),
            "episode_active": aligned["episode_active"][:, None].astype(np.float32),
            "terminal_within_interval": (
                (_numeric(windows, "window_end_s") >= log.terminal_time_s)
                & (time_s < log.terminal_time_s)
            )[:, None].astype(np.float32),
        }
        branch_id = f"{branch['controller']}:{Path(str(branch['log_path'])).name}"
        return Episode(
            episode_id=f"{segment.name}:{branch_id}",
            source_run_id=str(segment["source_run_id"]),
            trace_id=str(segment.name),
            branch_id=branch_id,
            region=str(segment["region"]),
            controller=str(branch["controller"]),
            time_s=time_s,
            network=network,
            prediction=prediction,
            streaming=streaming,
            action=action,
            config=config,
            context=context,
            targets=targets,
            metadata={
                "segment_path": str(segment.get("path", "")),
                "replay_log": str(branch["log_path"]),
                "profile_labels": labels,
                "trace_clock": {
                    "raw_origin_s": log.trace_origin_s,
                    "local_first_completion_s": float(log.time_s[0]),
                    "local_terminal_s": log.terminal_time_s,
                    "normalization": "first_raw_timestamp_minus_first_download_delay",
                },
                "pre_episode_config": {
                    "source": "declared_benchmark_protocol",
                    "planned_chunks": DEFAULT_PLANNED_CHUNKS,
                    "chunk_duration_s": DEFAULT_CHUNK_DURATION_S,
                    "bitrate_cap_mbps": DEFAULT_BITRATE_CAP_MBPS,
                    "uses_safe_capacity": str(branch["controller"]).lower() == "safesabr",
                },
                "alignment": (
                    "trace-local prediction-window state with causal completed-chunk "
                    "state, half-open interval decisions, explicit availability masks, "
                    "and post-termination rows removed"
                ),
                "measurement_semantics": (
                    "network observation at t uses the measurement window completed "
                    "before t; source future fields are shifted by one row and remain "
                    "targets at their original row"
                ),
                "input_provenance_reference": "audit.json:leakage_audit",
            },
        )

    def write(self, output_dir: str | Path | None = None) -> Path:
        episodes, audit = self.build()
        output = Path(output_dir or self.config.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {}
        index: list[dict[str, Any]] = []
        for number, episode in enumerate(episodes):
            prefix = f"e{number:06d}"
            for name in ("time_s", "network", "prediction", "streaming", "action", "config", "context"):
                payload[f"{prefix}_{name}"] = getattr(episode, name)
            for name, value in episode.targets.items():
                payload[f"{prefix}_target_{name}"] = value
            index.append(
                {
                    "prefix": prefix,
                    "episode_id": episode.episode_id,
                    "source_run_id": episode.source_run_id,
                    "trace_id": episode.trace_id,
                    "branch_id": episode.branch_id,
                    "region": episode.region,
                    "controller": episode.controller,
                    "metadata": episode.metadata,
                    "target_names": sorted(episode.targets),
                }
            )
        np.savez_compressed(output / "episodes.npz", **payload)
        (output / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
        (output / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        return output


def load_episode_archive(path: str | Path) -> tuple[list[Episode], dict[str, Any]]:
    root = Path(path)
    arrays = np.load(root / "episodes.npz")
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    episodes: list[Episode] = []
    for row in index:
        prefix = row["prefix"]
        targets = {
            name: arrays[f"{prefix}_target_{name}"]
            for name in row["target_names"]
        }
        episode = Episode(
            episode_id=row["episode_id"],
            source_run_id=row["source_run_id"],
            trace_id=row["trace_id"],
            branch_id=row["branch_id"],
            region=row["region"],
            controller=row["controller"],
            time_s=arrays[f"{prefix}_time_s"],
            network=arrays[f"{prefix}_network"],
            prediction=arrays[f"{prefix}_prediction"],
            streaming=arrays[f"{prefix}_streaming"],
            action=arrays[f"{prefix}_action"],
            config=arrays[f"{prefix}_config"],
            context=arrays[f"{prefix}_context"],
            targets=targets,
            metadata=row["metadata"],
        )
        episode.validate()
        episodes.append(episode)
    audit_path = root / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
    return episodes, audit
