from __future__ import annotations

import unittest
import copy

import numpy as np
import torch

from starworld.config import ModelConfig
from starworld.data.synthetic import make_synthetic_episodes
from starworld.interfaces.policies import ThroughputFractionPolicy
from starworld.interfaces.predictors import SafeCapacityPredictor
from starworld.models.world_model import StarWorldModel
from starworld.rollout.engine import HybridRolloutEngine
from starworld.rollout.trust import TrustGate
from starworld.training.dataset import SequenceDataset, fit_normalizers
from starworld.training.losses import multi_step_rollout_loss, world_model_loss


class TrainingAndRolloutTests(unittest.TestCase):
    def test_dataset_and_loss_are_runnable(self) -> None:
        episodes = make_synthetic_episodes(count=8, steps=20)
        for episode in episodes:
            episode.targets["analytical_streaming"] = episode.streaming.copy()
        normalizers = fit_normalizers(episodes)
        dataset = SequenceDataset(episodes, sequence_length=8, normalizers=normalizers)
        item = dataset[0]
        batch = {name: value.unsqueeze(0) for name, value in item.items()}
        model = StarWorldModel(ModelConfig(hidden_dim=32, latent_dim=12))
        output = model(
            batch["network"],
            batch["prediction"],
            batch["streaming"],
            batch["action"],
            batch["config"],
            batch["context"],
            sample=True,
            posterior_network=batch["target_network"],
            analytical_streaming=batch["analytical_streaming"],
        )
        loss, metrics = world_model_loss(output, batch)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertIn("reward_mae", metrics)
        self.assertIn("rebuffer_mae", metrics)
        self.assertIn("prediction_risk_bce", metrics)
        self.assertIn("streaming_risk_bce", metrics)
        self.assertEqual(float(item["analytical_baseline_available"].min()), 1.0)
        torch.testing.assert_close(
            item["target_streaming_residual"],
            item["target_streaming"] - item["analytical_streaming"],
        )

    def test_network_only_episode_does_not_distort_streaming_normalizer(self) -> None:
        episodes = make_synthetic_episodes(count=4, steps=20)
        replay_normalizers = fit_normalizers(episodes)
        network_only = copy.deepcopy(episodes[0])
        network_only.episode_id = "network-only"
        network_only.controller = "NetworkOnly"
        network_only.streaming = np.zeros_like(network_only.streaming)
        network_only.action = np.zeros_like(network_only.action)
        network_only.targets["streaming_valid"] = np.zeros(
            (network_only.length, 1),
            np.float32,
        )
        combined = fit_normalizers([*episodes, network_only])
        np.testing.assert_allclose(
            replay_normalizers.streaming.mean,
            combined.streaming.mean,
        )
        np.testing.assert_allclose(
            replay_normalizers.streaming.scale,
            combined.streaming.scale,
        )
        dataset = SequenceDataset(
            [episodes[0], network_only],
            sequence_length=8,
            normalizers=combined,
        )
        self.assertEqual(dataset.network_weights[network_only.episode_id], 1.0)
        self.assertEqual(dataset.network_weights[episodes[0].episode_id], 0.0)

    def test_short_replay_episode_is_padded_not_discarded(self) -> None:
        episodes = make_synthetic_episodes(count=3, steps=8)
        normalizers = fit_normalizers(episodes)
        dataset = SequenceDataset(
            episodes,
            sequence_length=12,
            normalizers=normalizers,
        )
        self.assertEqual(len(dataset), 3)
        item = dataset[0]
        self.assertEqual(item["network"].shape[0], 12)
        self.assertEqual(float(item["target_temporal_valid"].sum()), 7.0)
        self.assertEqual(
            float(item["target_streaming_valid"].sum()),
            7.0,
        )

    def test_interior_window_restores_analytical_inflight_state(self) -> None:
        episodes = make_synthetic_episodes(count=2, steps=20)
        episode = episodes[0]
        auxiliary = np.zeros((episode.length, 4), np.float32)
        auxiliary[4] = [9.75, 8.0, 2.0, 1.5]
        episode.targets["analytical_inflight"] = auxiliary
        normalizers = fit_normalizers(episodes)
        dataset = SequenceDataset(
            episodes,
            sequence_length=6,
            normalizers=normalizers,
            stride=4,
        )
        interior_index = next(
            index
            for index, value in enumerate(dataset.index)
            if value[0] == 0 and value[1] == 4
        )
        np.testing.assert_allclose(
            dataset[interior_index]["initial_analytical_inflight"].numpy(),
            auxiliary[4],
        )

    def test_hybrid_rollout_shapes(self) -> None:
        episode = make_synthetic_episodes(count=1, steps=20)[0]
        model = StarWorldModel(ModelConfig(hidden_dim=32, latent_dim=12))
        engine = HybridRolloutEngine(
            model,
            SafeCapacityPredictor(),
            ThroughputFractionPolicy(),
            residual_weight=0.0,
        )
        result = engine.rollout(
            network_prefix=episode.network[:6],
            prediction_state=episode.prediction[5],
            streaming_state=episode.streaming[5],
            config=episode.config,
            context=episode.context,
            horizon=4,
            samples=3,
        )
        self.assertEqual(result.network.shape, (3, 4, 8))
        self.assertEqual(result.streaming.shape, (3, 4, 5))
        self.assertEqual(result.actions.shape, (3, 4, 3))
        self.assertEqual(result.prediction_diagnostic.shape, (3, 4, 4))
        self.assertEqual(result.prediction_risk_probability.shape, (3, 4))

    def test_rollout_uses_initial_analytical_inflight_state(self) -> None:
        episode = make_synthetic_episodes(count=1, steps=20)[0]
        model = StarWorldModel(
            ModelConfig(hidden_dim=32, latent_dim=12, dropout=0.0)
        )
        engine = HybridRolloutEngine(
            model,
            SafeCapacityPredictor(),
            ThroughputFractionPolicy(),
        )
        common = {
            "network_prefix": episode.network[:6],
            "prediction_state": episode.prediction[5],
            "streaming_state": episode.streaming[5],
            "config": episode.config,
            "context": episode.context,
            "horizon": 2,
            "samples": 1,
            "seed": 9,
        }
        fresh = engine.rollout(**common)
        resumed = engine.rollout(
            **common,
            streaming_auxiliary_state=np.asarray(
                [0.001, 8.0, 2.0, 1.0],
                np.float32,
            ),
        )
        self.assertFalse(
            np.allclose(fresh.streaming, resumed.streaming)
        )

    def test_multi_step_loss_recomputes_analytical_baseline(self) -> None:
        episodes = make_synthetic_episodes(count=2, steps=12)
        for episode in episodes:
            episode.targets["analytical_streaming"] = episode.streaming.copy()
        normalizers = fit_normalizers(episodes)
        dataset = SequenceDataset(
            episodes,
            sequence_length=6,
            normalizers=normalizers,
        )
        item = dataset[0]
        batch = {name: value.unsqueeze(0) for name, value in item.items()}
        model = StarWorldModel(
            ModelConfig(hidden_dim=24, latent_dim=8, dropout=0.0)
        ).eval()
        first = multi_step_rollout_loss(
            model,
            batch,
            horizon=3,
            normalizers=normalizers,
            use_analytical_baseline=True,
        )
        altered = dict(batch)
        altered["analytical_streaming"] = (
            batch["analytical_streaming"] + 1000.0
        )
        second = multi_step_rollout_loss(
            model,
            altered,
            horizon=3,
            normalizers=normalizers,
            use_analytical_baseline=True,
        )
        torch.testing.assert_close(first, second)

    def test_trust_gate_abstains_on_high_uncertainty(self) -> None:
        gate = TrustGate(target_coverage=0.75)
        gate.calibrate(
            horizon=np.asarray([1, 1, 1, 1, 3, 3, 3, 3]),
            uncertainty=np.asarray([0.1, 0.2, 0.3, 0.4, 0.2, 0.4, 0.6, 0.8]),
            error=np.ones(8),
        )
        self.assertTrue(gate.decide(1, 0.05, 0.05).accepted)
        self.assertFalse(gate.decide(3, 2.0, 2.0).accepted)

    def test_trust_calibration_is_selective_for_validation_error(self) -> None:
        uncertainty = np.asarray([0.05, 0.10, 0.15, 0.20, 0.8, 1.0, 1.2, 1.4])
        error = np.asarray([0.1, 0.12, 0.15, 0.18, 2.0, 2.5, 3.0, 4.0])
        gate = TrustGate(target_coverage=0.5)
        gate.calibrate(
            horizon=np.ones(8, dtype=int),
            uncertainty=uncertainty,
            error=error,
        )
        accepted = uncertainty <= gate.thresholds[1]
        self.assertLess(error[accepted].mean(), error.mean())
        self.assertTrue(gate.decide(1, 0.05, 0.0).accepted)
        self.assertFalse(gate.decide(1, 1.0, 0.0).accepted)
        self.assertGreaterEqual(gate.error_models[1][1], 0.0)

    def test_trust_gate_uses_training_support_for_ood(self) -> None:
        rng = np.random.default_rng(3)
        support = rng.normal(0.0, 0.2, size=(200, 3))
        gate = TrustGate(target_coverage=0.8)
        gate.fit_support(support)
        gate.calibrate(
            horizon=np.ones(20, dtype=int),
            uncertainty=np.linspace(0.05, 0.5, 20),
            error=np.linspace(0.1, 0.8, 20),
        )
        self.assertTrue(
            gate.decide(1, 0.05, 0.0, state=np.zeros(3)).accepted
        )
        decision = gate.decide(1, 0.05, 0.0, state=np.full(3, 20.0))
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "out_of_distribution")

    def test_normalizers_fit_joint_action_query_support(self) -> None:
        episodes = make_synthetic_episodes(count=8, steps=20)
        normalizers = fit_normalizers(episodes)
        expected = (
            episodes[0].network.shape[1]
            + episodes[0].config.size
            + episodes[0].action.shape[1]
        )
        self.assertEqual(normalizers.query_support_center.shape, (expected,))
        self.assertEqual(
            normalizers.query_support_precision.shape,
            (expected, expected),
        )
        gate = TrustGate(target_coverage=0.8)
        gate.set_support(
            normalizers.query_support_center,
            normalizers.query_support_precision,
            normalizers.query_support_radius,
        )
        gate.calibrate(
            horizon=np.ones(20, dtype=int),
            uncertainty=np.linspace(0.05, 0.5, 20),
            error=np.linspace(0.1, 0.8, 20),
        )
        outlier = normalizers.query_support_center.copy()
        outlier[-1] += 1000.0
        decision = gate.decide(1, 0.05, 0.0, state=outlier)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "out_of_distribution")
