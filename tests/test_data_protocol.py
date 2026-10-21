from __future__ import annotations

import unittest

import pandas as pd

from starworld.data.episodes import (
    DEFAULT_BITRATE_CAP_MBPS,
    DEFAULT_CHUNK_DURATION_S,
    MODEL_INPUT_PROVENANCE,
    _config_vector,
    lagged_observation,
)
from starworld.data.leakage import (
    audit_feature_names,
    audit_feature_provenance,
    require_causal_features,
    require_causal_provenance,
)
from starworld.data.split import (
    action_combination_holdout_split,
    assert_no_group_leakage,
    chronological_holdout_split,
    grouped_source_run_split,
    long_horizon_stress_split,
    region_trace_balanced_source_run_split,
    region_holdout_split,
)
from starworld.data.synthetic import make_synthetic_episodes


class DataProtocolTests(unittest.TestCase):
    def test_source_runs_do_not_cross_splits(self) -> None:
        episodes = make_synthetic_episodes()
        manifest = grouped_source_run_split(episodes, seed=3)
        assert_no_group_leakage(episodes, manifest)
        all_ids = set(manifest.train_ids + manifest.validation_ids + manifest.test_ids)
        self.assertEqual(all_ids, {episode.episode_id for episode in episodes})

    def test_region_trace_balanced_split_preserves_regions(self) -> None:
        episodes = make_synthetic_episodes(count=30, steps=20)
        manifest = region_trace_balanced_source_run_split(
            episodes,
            seed=3,
        )
        assert_no_group_leakage(episodes, manifest)
        by_id = {episode.episode_id: episode for episode in episodes}
        regions = {
            split: {
                by_id[episode_id].region
                for episode_id in getattr(manifest, f"{split}_ids")
            }
            for split in ("train", "validation", "test")
        }
        expected = {episode.region for episode in episodes}
        self.assertTrue(all(value == expected for value in regions.values()))
        split_by_run = manifest.split_by_source_run
        runs_by_region = {}
        for episode in episodes:
            runs_by_region.setdefault(episode.region, set()).add(
                episode.source_run_id
            )
        for region, runs in runs_by_region.items():
            counts = {
                split: sum(
                    split_by_run[run] == split
                    for run in runs
                )
                for split in ("train", "validation", "test")
            }
            self.assertEqual(sum(counts.values()), len(runs))
            self.assertTrue(all(value >= 1 for value in counts.values()))
            self.assertLessEqual(
                abs(counts["train"] - 0.6 * len(runs)),
                1.0,
            )
            self.assertLessEqual(
                abs(counts["validation"] - 0.2 * len(runs)),
                1.0,
            )

    def test_future_fields_are_rejected_as_inputs(self) -> None:
        report = audit_feature_names(["history_mean_mbps", "future_mean_mbps"])
        self.assertFalse(report["passed"])
        with self.assertRaises(ValueError):
            require_causal_features(["buffer_s", "high_risk_label"])
        with self.assertRaises(ValueError):
            require_causal_features(["paper_profile_labels"])

    def test_generalization_splits_preserve_source_run_groups(self) -> None:
        episodes = make_synthetic_episodes()
        for manifest in (
            region_holdout_split(episodes, "VIC"),
            chronological_holdout_split(episodes),
        ):
            assert_no_group_leakage(episodes, manifest)
            self.assertTrue(manifest.train_ids)
            self.assertTrue(manifest.validation_ids)
            self.assertTrue(manifest.test_ids)

    def test_action_holdout_excludes_controller_from_training(self) -> None:
        episodes = make_synthetic_episodes(count=18, steps=20)
        for index, episode in enumerate(episodes):
            episode.controller = "BOLA" if index % 2 == 0 else "Pensieve"
        manifest = action_combination_holdout_split(
            episodes,
            "Pensieve",
            seed=5,
        )
        assert_no_group_leakage(episodes, manifest)
        by_id = {episode.episode_id: episode for episode in episodes}
        self.assertTrue(manifest.test_ids)
        self.assertTrue(
            all(
                by_id[episode_id].controller != "Pensieve"
                for episode_id in manifest.train_ids + manifest.validation_ids
            )
        )
        self.assertTrue(
            all(
                by_id[episode_id].controller == "Pensieve"
                for episode_id in manifest.test_ids
            )
        )

    def test_long_horizon_split_keeps_only_long_test_episodes(self) -> None:
        episodes = make_synthetic_episodes(count=18, steps=20)
        for episode in episodes[::3]:
            episode.time_s = episode.time_s[:8]
            episode.network = episode.network[:8]
            episode.prediction = episode.prediction[:8]
            episode.streaming = episode.streaming[:8]
            episode.action = episode.action[:8]
            episode.targets = {
                name: value[:8]
                if getattr(value, "ndim", 0) > 0
                and value.shape[0] == 20
                else value
                for name, value in episode.targets.items()
            }
        manifest = long_horizon_stress_split(
            episodes,
            minimum_test_steps=12,
            seed=4,
        )
        assert_no_group_leakage(episodes, manifest)
        by_id = {episode.episode_id: episode for episode in episodes}
        self.assertTrue(manifest.test_ids)
        self.assertTrue(
            all(
                by_id[episode_id].length - 1 >= 12
                for episode_id in manifest.test_ids
            )
        )

    def test_completed_measurement_is_visible_only_on_next_step(self) -> None:
        frame = pd.DataFrame(
            {
                "history_mean_mbps": [10.0, 20.0, 30.0],
                "future_mean_mbps": [100.0, 200.0, 300.0],
            }
        )
        values = lagged_observation(
            frame,
            "history_mean_mbps",
            "future_mean_mbps",
        )
        self.assertEqual(values.tolist(), [10.0, 100.0, 200.0])

    def test_model_input_provenance_is_complete_and_causal(self) -> None:
        report = audit_feature_provenance(MODEL_INPUT_PROVENANCE, MODEL_INPUT_PROVENANCE)
        self.assertTrue(report["passed"], report["violations"])
        self.assertFalse(report["missing_features"])

    def test_provenance_rejects_full_replay_aggregates(self) -> None:
        provenance = {
            "bitrate_cap_mbps": {
                "availability": "explicit_pre_episode_config",
                "sources": ["chunk_bitrate_kbps"],
                "transform": "global_max_over_full_replay",
            }
        }
        report = audit_feature_provenance(["bitrate_cap_mbps"], provenance)
        self.assertFalse(report["passed"])
        with self.assertRaises(ValueError):
            require_causal_provenance(["bitrate_cap_mbps"], provenance)

    def test_future_named_source_requires_completed_window_lag(self) -> None:
        invalid = {
            "observed_window_mean_mbps": {
                "availability": "causal_measurement_history",
                "sources": ["future_mean_mbps"],
            }
        }
        self.assertFalse(
            audit_feature_provenance(["observed_window_mean_mbps"], invalid)["passed"]
        )

    def test_replay_configuration_is_declared_before_episode(self) -> None:
        for controller in ("BOLA", "RobustMPC", "Pensieve", "SafeSABR"):
            config = _config_vector(controller)
            self.assertEqual(float(config[5]), DEFAULT_BITRATE_CAP_MBPS)
            self.assertEqual(float(config[6]), DEFAULT_CHUNK_DURATION_S)
            self.assertEqual(float(config[4]), float(controller == "SafeSABR"))
