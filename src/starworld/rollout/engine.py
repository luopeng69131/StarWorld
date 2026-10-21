from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from starworld.interfaces.policies import StreamingPolicy
from starworld.interfaces.predictors import PredictionInterface
from starworld.interfaces.streaming import (
    apply_streaming_auxiliary,
    StreamingState,
    transition_streaming_interval,
)
from starworld.models.world_model import StarWorldModel
from starworld.rollout.trust import TrustGate
from starworld.training.dataset import NormalizerBundle


@dataclass
class RolloutResult:
    network: np.ndarray
    capacity: np.ndarray
    streaming: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    severe_probability: np.ndarray
    aleatoric_uncertainty: np.ndarray
    prediction_diagnostic: np.ndarray | None = None
    prediction_risk_probability: np.ndarray | None = None
    trust_accepted: np.ndarray | None = None
    trust_confidence: np.ndarray | None = None
    trust_calibration_horizon: np.ndarray | None = None
    trust_reason: np.ndarray | None = None

    def uncertainty(self) -> np.ndarray:
        """Return the same horizon-wise spread used for gate calibration."""
        spread = self.network.std(axis=0).mean(axis=-1)
        return np.maximum(spread, self.aleatoric_uncertainty.mean(axis=0))

    def summary(self) -> dict[str, float]:
        final_rebuffer = self.streaming[:, -1, 2]
        result = {
            "mean_final_rebuffer_s": float(final_rebuffer.mean()),
            "p95_final_rebuffer_s": float(np.quantile(final_rebuffer, 0.95)),
            "mean_bitrate_mbps": float(self.actions[..., 0].mean()),
            "mean_reward": float(self.rewards.mean()),
            "severe_session_probability": float((final_rebuffer > 10.0).mean()),
        }
        if self.trust_accepted is not None:
            result["trust_acceptance_rate"] = float(self.trust_accepted.mean())
        return result


def _state_from_vector(value: np.ndarray) -> StreamingState:
    return StreamingState(
        buffer_s=float(value[0]),
        last_bitrate_mbps=float(value[1]),
        cumulative_rebuffer_s=float(value[2]),
        last_download_s=float(value[3]),
        playback_progress=float(value[4]),
    )


def _state_vector(value: StreamingState) -> np.ndarray:
    return np.asarray(
        [
            value.buffer_s,
            value.last_bitrate_mbps,
            value.cumulative_rebuffer_s,
            value.last_download_s,
            value.playback_progress,
        ],
        dtype=np.float32,
    )


class HybridRolloutEngine:
    """Sample network futures, then execute predictor and streaming interfaces."""

    def __init__(
        self,
        model: StarWorldModel,
        predictor: PredictionInterface,
        policy: StreamingPolicy,
        chunk_duration_s: float = 4.0,
        model_step_s: float = 10.0,
        device: str = "cpu",
        normalizers: NormalizerBundle | None = None,
        analytical_transition: bool = True,
        trust_gate: TrustGate | None = None,
        residual_weight: float | None = None,
        ensemble_models: list[StarWorldModel] | None = None,
    ):
        self.model = model.to(device).eval()
        self.models = [
            member.to(device).eval()
            for member in (ensemble_models or [model])
        ]
        self.predictor = predictor
        self.policy = policy
        self.chunk_duration_s = chunk_duration_s
        self.model_step_s = model_step_s
        self.device = torch.device(device)
        self.normalizers = normalizers
        self.analytical_transition = analytical_transition
        self.trust_gate = trust_gate
        self.residual_weight = 0.5 if residual_weight is None else float(residual_weight)

    def _normalize(self, name: str, value: np.ndarray) -> np.ndarray:
        if self.normalizers is None:
            return np.asarray(value, np.float32)
        return getattr(self.normalizers, name).transform(np.asarray(value, np.float32))

    def _inverse(self, name: str, value: np.ndarray) -> np.ndarray:
        if self.normalizers is None:
            return np.asarray(value, np.float32)
        return getattr(self.normalizers, name).inverse(np.asarray(value, np.float32))

    def _inverse_delta(self, name: str, value: np.ndarray) -> np.ndarray:
        if self.normalizers is None:
            return np.asarray(value, np.float32)
        return (
            np.asarray(value, np.float32)
            * getattr(self.normalizers, name).scale
        )

    @torch.no_grad()
    def rollout(
        self,
        network_prefix: np.ndarray,
        prediction_state: np.ndarray,
        streaming_state: np.ndarray,
        config: np.ndarray,
        context: np.ndarray,
        horizon: int,
        samples: int = 32,
        seed: int = 0,
        streaming_auxiliary_state: np.ndarray | None = None,
        recorded_actions: np.ndarray | None = None,
    ) -> RolloutResult:
        if horizon < 1 or samples < 1:
            raise ValueError("horizon and samples must be positive")
        if recorded_actions is not None:
            recorded_actions = np.asarray(recorded_actions, dtype=np.float32)
            if recorded_actions.shape != (horizon, self.model.config.action_dim):
                raise ValueError("recorded_actions must have shape [horizon, action_dim]")
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        network_history = np.asarray(network_prefix, dtype=np.float32)
        if network_history.ndim != 2:
            raise ValueError("network_prefix must have shape [history, network_dim]")
        initial_prediction = np.asarray(prediction_state, dtype=np.float32)
        if initial_prediction.shape != (self.model.config.prediction_dim,):
            raise ValueError(
                "prediction_state must match the model prediction dimension"
            )

        network_paths = np.zeros((samples, horizon, network_history.shape[1]), np.float32)
        capacities = np.zeros((samples, horizon, 2), np.float32)
        streaming_paths = np.zeros((samples, horizon, len(streaming_state)), np.float32)
        actions = np.zeros((samples, horizon, self.model.config.action_dim), np.float32)
        rewards = np.zeros((samples, horizon), np.float32)
        severe = np.zeros((samples, horizon), np.float32)
        aleatoric = np.zeros((samples, horizon), np.float32)
        prediction_diagnostics = np.zeros(
            (samples, horizon, self.model.config.prediction_dim),
            np.float32,
        )
        prediction_risk = np.zeros((samples, horizon), np.float32)
        trust_accepted = np.ones((samples, horizon), bool)
        trust_confidence = np.zeros((samples, horizon), np.float32)
        trust_calibration_horizon = np.zeros((samples, horizon), np.int32)
        trust_reason = np.empty((samples, horizon), dtype=object)
        trust_reason.fill("not_evaluated")
        config_tensor = torch.as_tensor(self._normalize("config", config), device=self.device).unsqueeze(0)
        context_tensor = torch.as_tensor(self._normalize("context", context), device=self.device).unsqueeze(0)

        for sample_index in range(samples):
            active_model = self.models[sample_index % len(self.models)]
            history = network_history.copy()
            stream = _state_from_vector(np.asarray(streaming_state, np.float32))
            if streaming_auxiliary_state is not None:
                stream = apply_streaming_auxiliary(
                    stream,
                    streaming_auxiliary_state,
                )
            hidden = active_model.initial_hidden(1, self.device)
            # Warm the recurrent network state using only observed network history.
            for observed in history[:-1]:
                observed_tensor = torch.as_tensor(
                    self._normalize("network", observed), device=self.device
                ).unsqueeze(0)
                _, _, hidden, _ = active_model.network_dynamics.step(
                    observed_tensor, context_tensor, hidden, sample=False
                )
            network = history[-1].copy()
            throughput_history = history[:, 0].astype(float).tolist()
            for step in range(horizon):
                capacity, uncertainty = self.predictor.predict(np.asarray(throughput_history))
                prediction = np.asarray(
                    [
                        capacity + max(uncertainty, 0.0),
                        max(capacity, 0.0),
                        max(uncertainty, 0.0),
                        self.model_step_s,
                    ],
                    np.float32,
                )
                bitrate = self.policy.action(stream, capacity, uncertainty)
                # The action is chosen from the causal capacity estimate. The
                # sampled next network state is used only after the decision
                # when the fixed-time streaming interval is executed.
                decision_throughput = max(float(capacity), 1e-3)
                estimated_download_s = max(
                    bitrate * self.chunk_duration_s / decision_throughput,
                    1e-3,
                )
                decision_count = int(
                    np.clip(
                        np.floor(self.model_step_s / estimated_download_s),
                        0,
                        8,
                    )
                )
                action = np.asarray(
                    [bitrate, 0.0, float(decision_count)],
                    np.float32,
                )
                if recorded_actions is not None:
                    action = recorded_actions[step]
                    bitrate = float(action[0])
                    decision_count = max(int(round(float(action[2]))), 0)
                output = active_model.step(
                    torch.as_tensor(self._normalize("network", network), device=self.device).unsqueeze(0),
                    torch.as_tensor(self._normalize("prediction", prediction), device=self.device).unsqueeze(0),
                    torch.as_tensor(
                        self._normalize("streaming", _state_vector(stream)), device=self.device
                    ).unsqueeze(0),
                    torch.as_tensor(self._normalize("action", action), device=self.device).unsqueeze(0),
                    config_tensor,
                    context_tensor,
                    hidden,
                    sample=True,
                    analytical_streaming=None,
                )
                hidden = output.hidden
                std = output.network_log_std.exp()
                network_tensor = output.network_mean + torch.randn_like(std) * std
                next_network = self._inverse("network", network_tensor.squeeze(0).cpu().numpy())
                next_network[:7] = np.maximum(next_network[:7], 0.0)
                next_network[0] = max(next_network[0], 1e-3)
                next_network[1] = min(next_network[1], next_network[0])
                next_network[2] = min(next_network[2], next_network[1])
                next_network[7] = np.clip(next_network[7], 0.0, 1.0)

                analytical = transition_streaming_interval(
                    stream,
                    throughput_mbps=float(next_network[0]),
                    bitrate_mbps=bitrate,
                    decision_count=decision_count,
                    chunk_duration_s=self.chunk_duration_s,
                    interval_duration_s=self.model_step_s,
                )
                analytical_vector = _state_vector(analytical.state)
                learned_residual = self._inverse_delta(
                    "streaming",
                    output.streaming_residual_mean.squeeze(0).cpu().numpy(),
                )
                if self.analytical_transition:
                    hybrid_state = analytical_vector + self.residual_weight * learned_residual
                else:
                    hybrid_state = _state_vector(stream) + self.residual_weight * learned_residual
                hybrid_state[:4] = np.maximum(hybrid_state[:4], 0.0)
                hybrid_state[2] = max(hybrid_state[2], stream.cumulative_rebuffer_s)
                hybrid_state[4] = np.clip(hybrid_state[4], 0.0, 1.0)
                stream = _state_from_vector(hybrid_state)
                if self.analytical_transition:
                    stream.inflight_remaining_megabits = (
                        analytical.state.inflight_remaining_megabits
                    )
                    stream.inflight_bitrate_mbps = (
                        analytical.state.inflight_bitrate_mbps
                    )
                    stream.inflight_elapsed_s = (
                        analytical.state.inflight_elapsed_s
                    )
                    stream.inflight_rebuffer_s = (
                        analytical.state.inflight_rebuffer_s
                    )
                streaming_risk_probability = float(
                    torch.sigmoid(output.streaming_risk_logits).squeeze().cpu()
                )
                prediction_diagnostic = self._inverse(
                    "prediction",
                    output.prediction_mean.squeeze(0).cpu().numpy(),
                )
                prediction_risk_probability = float(
                    torch.sigmoid(output.prediction_risk_logits).squeeze().cpu()
                )

                network_paths[sample_index, step] = next_network
                capacities[sample_index, step] = [capacity, uncertainty]
                prediction_diagnostics[sample_index, step] = prediction_diagnostic
                prediction_risk[sample_index, step] = prediction_risk_probability
                streaming_paths[sample_index, step] = hybrid_state
                actions[sample_index, step] = action
                reward_normalized = np.asarray(
                    [float(output.reward_mean.squeeze().cpu())],
                    np.float32,
                )
                rewards[sample_index, step] = float(
                    self._inverse("reward", reward_normalized)[0]
                )
                severe[sample_index, step] = streaming_risk_probability
                if self.normalizers is None:
                    aleatoric[sample_index, step] = float(std.mean().cpu())
                else:
                    network_scale = torch.as_tensor(
                        self.normalizers.network.scale, device=self.device, dtype=std.dtype
                    )
                    aleatoric[sample_index, step] = float((std * network_scale).mean().cpu())
                throughput_history.append(float(next_network[0]))
                network = next_network
        if self.trust_gate is not None:
            total_spread = network_paths.std(axis=0).mean(axis=-1)
            mean_aleatoric = aleatoric.mean(axis=0)
            epistemic = np.sqrt(
                np.maximum(total_spread**2 - mean_aleatoric**2, 0.0)
            )
            for step in range(horizon):
                for sample_index in range(samples):
                    support_state = network_paths[sample_index, step]
                    if (
                        self.trust_gate.support_center is not None
                        and self.trust_gate.support_center.size
                        == (
                            network_paths.shape[-1]
                            + np.asarray(config).size
                            + actions.shape[-1]
                        )
                    ):
                        support_state = np.concatenate(
                            [
                                support_state,
                                np.asarray(config, np.float32),
                                actions[sample_index, step],
                            ]
                        )
                    decision = self.trust_gate.decide(
                        horizon=step + 1,
                        epistemic=float(epistemic[step]),
                        aleatoric=float(mean_aleatoric[step]),
                        state=support_state,
                    )
                    trust_accepted[sample_index, step] = decision.accepted
                    trust_confidence[sample_index, step] = decision.confidence
                    trust_calibration_horizon[sample_index, step] = (
                        decision.supported_horizon
                    )
                    trust_reason[sample_index, step] = decision.reason
        return RolloutResult(
            network=network_paths,
            capacity=capacities,
            streaming=streaming_paths,
            actions=actions,
            rewards=rewards,
            severe_probability=severe,
            aleatoric_uncertainty=aleatoric,
            prediction_diagnostic=prediction_diagnostics,
            prediction_risk_probability=prediction_risk,
            trust_accepted=trust_accepted if self.trust_gate is not None else None,
            trust_confidence=(
                trust_confidence if self.trust_gate is not None else None
            ),
            trust_calibration_horizon=(
                trust_calibration_horizon
                if self.trust_gate is not None
                else None
            ),
            trust_reason=trust_reason if self.trust_gate is not None else None,
        )

    def compare_policies(
        self,
        policies: dict[str, StreamingPolicy],
        **rollout_kwargs: object,
    ) -> dict[str, dict[str, float]]:
        original = self.policy
        results: dict[str, dict[str, float]] = {}
        try:
            for name, policy in policies.items():
                self.policy = policy
                results[name] = self.rollout(**rollout_kwargs).summary()
        finally:
            self.policy = original
        return results
