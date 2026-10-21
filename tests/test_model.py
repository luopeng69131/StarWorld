from __future__ import annotations

import unittest

import torch

from starworld.config import ModelConfig
from starworld.models.world_model import StarWorldModel


def _inputs(config: ModelConfig, batch: int = 3, steps: int = 5) -> dict[str, torch.Tensor]:
    return {
        "network": torch.randn(batch, steps, config.network_dim),
        "prediction": torch.randn(batch, steps, config.prediction_dim),
        "streaming": torch.randn(batch, steps, config.streaming_dim),
        "action": torch.randn(batch, steps, config.action_dim),
        "config": torch.randn(batch, config.config_dim),
        "context": torch.randn(batch, config.context_dim),
    }


class ModelTests(unittest.TestCase):
    def test_model_shapes_and_finite_outputs(self) -> None:
        config = ModelConfig(hidden_dim=32, latent_dim=12)
        model = StarWorldModel(config).eval()
        inputs = _inputs(config)
        output = model(**inputs, sample=False)
        self.assertEqual(output.network_mean.shape, inputs["network"].shape)
        self.assertEqual(output.prediction_mean.shape, inputs["prediction"].shape)
        self.assertEqual(output.streaming_mean.shape, inputs["streaming"].shape)
        self.assertEqual(output.event_logits.shape, (3, 5, 2))
        self.assertEqual(output.prediction_risk_logits.shape, (3, 5, 1))
        self.assertEqual(output.streaming_risk_logits.shape, (3, 5, 1))
        self.assertEqual(output.reward_mean.shape, (3, 5, 1))
        self.assertEqual(output.rebuffer_mean.shape, (3, 5, 1))
        self.assertEqual(output.prior_mean.shape, (3, 5, config.latent_dim))
        self.assertTrue(torch.isfinite(output.network_log_std).all())

    def test_application_action_cannot_change_network_transition(self) -> None:
        config = ModelConfig(hidden_dim=32, latent_dim=12)
        model = StarWorldModel(config).eval()
        inputs = _inputs(config)
        first = model(**inputs, sample=False).network_mean
        inputs["action"] = inputs["action"] + 1000.0
        inputs["config"] = inputs["config"] * -50.0
        second = model(**inputs, sample=False).network_mean
        self.assertTrue(torch.equal(first, second))

    def test_streaming_inputs_cannot_influence_prediction_layer(self) -> None:
        config = ModelConfig(hidden_dim=32, latent_dim=12)
        model = StarWorldModel(config).eval()
        inputs = _inputs(config)
        first = model(**inputs, sample=False)
        changed = {name: value.clone() for name, value in inputs.items()}
        changed["streaming"] += 500.0
        changed["action"] -= 700.0
        changed["config"][:, list(config.streaming_config_indices)] += 900.0
        second = model(**changed, sample=False)
        self.assertTrue(torch.equal(first.network_mean, second.network_mean))
        self.assertTrue(torch.equal(first.prediction_mean, second.prediction_mean))
        self.assertTrue(
            torch.equal(first.prediction_risk_logits, second.prediction_risk_logits)
        )
        self.assertFalse(
            torch.equal(first.streaming_risk_logits, second.streaming_risk_logits)
        )

    def test_predictor_config_is_isolated_from_network_transition(self) -> None:
        config = ModelConfig(hidden_dim=32, latent_dim=12)
        model = StarWorldModel(config).eval()
        inputs = _inputs(config)
        first = model(**inputs, sample=False)
        inputs["config"][:, list(config.predictor_config_indices)] += 100.0
        second = model(**inputs, sample=False)
        self.assertTrue(torch.equal(first.network_mean, second.network_mean))
        self.assertFalse(torch.equal(first.prediction_mean, second.prediction_mean))

    def test_streaming_output_is_explicit_baseline_plus_residual(self) -> None:
        config = ModelConfig(hidden_dim=32, latent_dim=12)
        model = StarWorldModel(config).eval()
        inputs = _inputs(config, batch=2, steps=4)
        analytical = torch.randn(2, 4, config.streaming_dim)
        output = model(
            **inputs,
            sample=False,
            analytical_streaming=analytical,
        )
        torch.testing.assert_close(
            output.streaming_mean,
            analytical + output.streaming_residual_mean,
        )
        torch.testing.assert_close(
            output.event_logits,
            torch.cat(
                [output.prediction_risk_logits, output.streaming_risk_logits],
                dim=-1,
            ),
        )

    def test_observation_posterior_is_distinct_from_prior(self) -> None:
        config = ModelConfig(hidden_dim=32, latent_dim=12)
        model = StarWorldModel(config).eval()
        inputs = _inputs(config, batch=2, steps=4)
        output = model(
            **inputs,
            sample=False,
            posterior_network=torch.randn_like(inputs["network"]),
        )
        self.assertFalse(torch.equal(output.prior_mean, output.posterior_mean))
