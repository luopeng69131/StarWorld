import csv
from datetime import datetime, timedelta
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import torch

from starworld.api import StarWorld
from starworld.config import load_config, ModelConfig, TrainingConfig
from starworld.data.episodes import CrossLayerExperienceConstructor, load_episode_archive
from starworld.data.prepare import causal_windows, parse_throughput, prepare_starnet
from starworld.data.synthetic import make_synthetic_episodes
from starworld.interfaces.policies import ThroughputFractionPolicy
from starworld.interfaces.streaming import (StreamingState, apply_streaming_auxiliary,
                                           streaming_auxiliary_vector, transition_streaming_interval)
from starworld.interfaces.torch_streaming import playback_interval
from starworld.models.world_model import StarWorldModel
from starworld.rollout.engine import HybridRolloutEngine
from starworld.interfaces.predictors import SafeCapacityPredictor
from starworld.rollout.trust import TrustGate


def write_logs(root, count=3):
    for run in range(count):
        path = root / "data_osn" / f"run_{run}" / "network_measurements.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        start = datetime(2025, 1, run + 1)
        path.write_text("\n".join(f"{start + timedelta(seconds=i)} TP: {50 + run + i % 11}"
                                  for i in range(120)), encoding="utf-8")


class PublicWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_measurement_only_pipeline_and_portable_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_logs(root / "raw")
            report = prepare_starnet(root / "raw", root / "tables", segment_s=120, history_s=20)
            self.assertEqual(report["segments"], 3)
            config = load_config(root / "tables/config.json")
            output = CrossLayerExperienceConstructor(config.data).write(root / "dataset")
            episodes, audit = load_episode_archive(output)
            self.assertEqual(len(episodes), 3)
            self.assertEqual(audit["replay_branch_count"], 0)
            self.assertEqual(audit["action_coverage"]["unique_bitrates_mbps"], [])
            self.assertTrue(all(ep.controller == "NetworkOnly" for ep in episodes))

    def test_relative_replay_logs_align_with_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_logs(root / "raw")
            prepare_starnet(root / "raw", root / "tables", segment_s=120, history_s=20)
            with (root / "tables/segments.csv").open() as handle:
                traces = list(csv.DictReader(handle))
            log = root / "tables/replay.log"
            log.write_text("\n".join(f"{2 * (i + 1)} 8000 12.0 0.0 4000000 2000 4.0"
                                      for i in range(48)), encoding="utf-8")
            with (root / "tables/replay_index.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["trace_id", "controller", "log_path"])
                writer.writeheader()
                writer.writerows({"trace_id": row["trace_id"], "controller": "BOLA", "log_path": "replay.log"}
                                 for row in traces)
            episodes, audit = CrossLayerExperienceConstructor(load_config(root / "tables/config.json").data).build()
            self.assertEqual(audit["replay_branch_count"], 3)
            self.assertEqual(audit["excluded_count"], 0)
            self.assertEqual(len(episodes), 6)

    def test_zip_and_extracted_duplicate_are_processed_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_logs(root / "raw", count=1)
            source = root / "raw/data_osn/run_0/network_measurements.log"
            with zipfile.ZipFile(root / "raw/data_osn.zip", "w") as archive:
                archive.write(source, "run_0/network_measurements.log")
            report = prepare_starnet(root / "raw", root / "tables", segment_s=120, history_s=20)
            self.assertEqual(len(report["runs"]), 1)

    def test_history_features_do_not_read_future_targets(self):
        first = np.full(120, 30.0)
        second = first.copy()
        second[60:70] = 1e5
        left, right = causal_windows(first, "t", "r"), causal_windows(second, "t", "r")
        for name in ("history_mean_mbps", "history_p05_mbps", "point_prediction_mbps"):
            self.assertEqual(left[0][name], right[0][name])
        self.assertNotEqual(left[0]["future_mean_mbps"], right[0]["future_mean_mbps"])

    def test_gaps_and_zero_capacity(self):
        text = "2025-01-01 00:00:00 TP: 0\n2025-01-01 00:00:02 TP: 2\n2025-01-01 00:00:10 TP: 5"
        blocks = parse_throughput(text)
        np.testing.assert_allclose(blocks[0][1], [0, 1, 2])
        self.assertEqual(len(blocks), 2)

    def test_differentiable_playback_matches_scalar_equations(self):
        rng = np.random.default_rng(8)
        states = rng.uniform(0, 10, (40, 5)).astype(np.float32)
        states[:, 4] /= 10
        auxiliary = rng.uniform(0, 3, (40, 4)).astype(np.float32)
        auxiliary[::2] = 0
        throughput = rng.uniform(.1, 200, 40).astype(np.float32)
        actions = np.column_stack((rng.uniform(1, 120, 40), np.zeros(40), rng.integers(0, 8, 40))).astype(np.float32)
        actual, aux = playback_interval(*map(torch.tensor, (states, auxiliary, throughput, actions)))
        for i in range(40):
            state = StreamingState(*map(float, states[i]))
            apply_streaming_auxiliary(state, auxiliary[i])
            expected = transition_streaming_interval(state, throughput[i], actions[i, 0], int(actions[i, 2])).state
            np.testing.assert_allclose(actual[i].numpy(), [expected.buffer_s, expected.last_bitrate_mbps,
                                       expected.cumulative_rebuffer_s, expected.last_download_s,
                                       expected.playback_progress], rtol=3e-5, atol=5e-5)
            np.testing.assert_allclose(aux[i].numpy(), streaming_auxiliary_vector(expected), rtol=3e-5, atol=5e-5)

    def test_shared_network_futures_and_physical_playback(self):
        episode = make_synthetic_episodes(count=1, steps=12)[0]
        model = StarWorldModel(ModelConfig(hidden_dim=16, latent_dim=4, dropout=0))
        engine = HybridRolloutEngine(model, SafeCapacityPredictor(), ThroughputFractionPolicy(.6))
        query = StarWorld.query(episode, 4, 3, 4, 12)
        first = engine.rollout(**query)
        engine.policy = ThroughputFractionPolicy(.95)
        second = engine.rollout(**query)
        np.testing.assert_array_equal(first.network, second.network)
        self.assertTrue((np.diff(first.streaming[:, :, 2], axis=1) >= 0).all())
        self.assertTrue((first.streaming >= 0).all())

    def test_horizon_boundary_and_invalid_uncertainty(self):
        gate = TrustGate()
        gate.calibrate(np.array([1, 1, 2, 2]), np.array([.1, .2, .1, .2]), np.ones(4))
        self.assertEqual(gate.decide(3, .01, .01).reason, "horizon_exceeds_calibration")
        with self.assertRaises(ValueError):
            gate.decide(1, float("nan"), .1)

    def test_fit_refine_reload_and_validation_calibration(self):
        episodes = make_synthetic_episodes(count=8, steps=12)
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model"
            world = StarWorld.fit(episodes, model_path, model_config=ModelConfig(hidden_dim=16, latent_dim=4),
                                  training_config=TrainingConfig(epochs=1, sequence_length=6, rollout_horizon=2),
                                  members=1, refinement_epochs=1)
            self.assertEqual(world.manifest["members"], ["member_00/refined.pt"])
            stats = world.calibrate(episodes, horizon=2, samples=3, max_queries=2)
            self.assertGreater(stats["validation_queries"], 0)
            loaded = StarWorld.load(model_path)
            self.assertIsNotNone(loaded.gate)
            results = loaded.imagine(loaded.split(episodes, "test")[0], prefix=3, horizon=2, samples=3)
            self.assertTrue(all(np.isfinite(v.network).all() for v in results.values()))
            with self.assertRaises(FileExistsError):
                StarWorld.fit(episodes, model_path)


if __name__ == "__main__":
    unittest.main()
