from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from starworld.config import ModelConfig
from starworld.models.components import GaussianHead, MLP, sample_diagonal_gaussian


@dataclass
class WorldModelOutput:
    network_mean: torch.Tensor
    network_log_std: torch.Tensor
    prediction_mean: torch.Tensor
    prediction_log_std: torch.Tensor
    streaming_mean: torch.Tensor
    streaming_log_std: torch.Tensor
    streaming_residual_mean: torch.Tensor
    streaming_residual_log_std: torch.Tensor
    analytical_streaming_baseline: torch.Tensor
    prediction_risk_logits: torch.Tensor
    streaming_risk_logits: torch.Tensor
    reward_mean: torch.Tensor
    reward_log_std: torch.Tensor
    rebuffer_mean: torch.Tensor
    rebuffer_log_std: torch.Tensor
    continuation_logits: torch.Tensor
    prior_mean: torch.Tensor
    prior_log_std: torch.Tensor
    posterior_mean: torch.Tensor
    posterior_log_std: torch.Tensor
    hidden: torch.Tensor

    @property
    def event_logits(self) -> torch.Tensor:
        """Backward-compatible [prediction risk, streaming risk] view."""
        return torch.cat(
            [self.prediction_risk_logits, self.streaming_risk_logits],
            dim=-1,
        )


class StochasticNetworkDynamics(nn.Module):
    """Causal recurrent prior with an optional training-only observation posterior.

    The deterministic state is updated from the current network observation and
    context; a prior predicts the next latent; during training, the observed
    next network state may form a posterior used for reconstruction and KL
    regularization. Imagination uses only the prior.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = MLP(
            config.network_dim + config.context_dim,
            config.hidden_dim,
            config.hidden_dim,
            config.dropout,
        )
        self.recurrent = nn.GRUCell(config.hidden_dim, config.hidden_dim)
        self.prior = GaussianHead(config.hidden_dim, config.latent_dim)
        self.posterior_encoder = MLP(
            config.network_dim + config.context_dim,
            config.hidden_dim,
            config.hidden_dim,
            config.dropout,
        )
        self.posterior = GaussianHead(
            config.hidden_dim * 2,
            config.latent_dim,
        )
        self.decoder = GaussianHead(config.hidden_dim + config.latent_dim, config.network_dim)
        self.continuation_head = nn.Linear(config.hidden_dim + config.latent_dim, 1)

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.config.hidden_dim, device=device)

    def step_with_posterior(
        self,
        network: torch.Tensor,
        context: torch.Tensor,
        hidden: torch.Tensor,
        sample: bool,
        posterior_network: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        encoded = self.encoder(torch.cat([network, context], dim=-1))
        hidden = self.recurrent(encoded, hidden)
        prior_mean, prior_log_std = self.prior(hidden)
        if posterior_network is not None and self.config.posterior_uses_next_network:
            observed_next = self.posterior_encoder(
                torch.cat([posterior_network, context], dim=-1)
            )
            posterior_mean, posterior_log_std = self.posterior(
                torch.cat([hidden, observed_next], dim=-1)
            )
            latent = sample_diagonal_gaussian(
                posterior_mean,
                posterior_log_std,
                sample,
            )
        else:
            posterior_mean, posterior_log_std = prior_mean, prior_log_std
            latent = sample_diagonal_gaussian(prior_mean, prior_log_std, sample)
        decoded_mean, network_log_std = self.decoder(
            torch.cat([hidden, latent], dim=-1)
        )
        network_mean = (
            network + decoded_mean
            if self.config.network_residual_transition
            else decoded_mean
        )
        continuation_logits = self.continuation_head(torch.cat([hidden, latent], dim=-1))
        return (
            network_mean,
            network_log_std,
            hidden,
            latent,
            prior_mean,
            prior_log_std,
            posterior_mean,
            posterior_log_std,
            continuation_logits,
        )

    def step(
        self,
        network: torch.Tensor,
        context: torch.Tensor,
        hidden: torch.Tensor,
        sample: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Backward-compatible prior-only transition used by rollout warm-up."""
        (
            network_mean,
            network_log_std,
            hidden,
            latent,
            _,
            _,
            _,
            _,
            _,
        ) = self.step_with_posterior(
            network,
            context,
            hidden,
            sample,
            posterior_network=None,
        )
        return network_mean, network_log_std, hidden, latent


class StarWorldModel(nn.Module):
    """Factorized hybrid world model with explicit network-to-application direction."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self._validate_config_indices()
        self.network_dynamics = StochasticNetworkDynamics(config)
        prediction_config_dim = len(config.predictor_config_indices)
        streaming_config_dim = len(config.streaming_config_indices)
        if config.monolithic_cross_layer:
            joint_input = (
                config.hidden_dim
                + config.latent_dim
                + config.prediction_dim
                + config.streaming_dim
                + config.action_dim
                + config.config_dim
            )
            self.joint_cross_layer_backbone = MLP(
                joint_input,
                config.hidden_dim,
                config.hidden_dim,
                config.dropout,
            )
            self.prediction_backbone = None
            self.streaming_backbone = None
        else:
            prediction_input = (
                config.hidden_dim
                + config.latent_dim
                + config.prediction_dim
                + prediction_config_dim
            )
            streaming_input = (
                config.hidden_dim
                + config.latent_dim
                + config.prediction_dim
                + config.streaming_dim
                + config.action_dim
                + streaming_config_dim
            )
            self.joint_cross_layer_backbone = None
            self.prediction_backbone = MLP(
                prediction_input,
                config.hidden_dim,
                config.hidden_dim,
                config.dropout,
            )
            self.streaming_backbone = MLP(
                streaming_input,
                config.hidden_dim,
                config.hidden_dim,
                config.dropout,
            )
        self.prediction_head = GaussianHead(config.hidden_dim, config.prediction_dim)
        self.prediction_risk_head = nn.Linear(config.hidden_dim, 1)
        self.streaming_residual_head = GaussianHead(config.hidden_dim, config.streaming_dim)
        self.streaming_risk_head = nn.Linear(config.hidden_dim, 1)
        self.reward_head = GaussianHead(config.hidden_dim, 1)
        self.rebuffer_head = GaussianHead(config.hidden_dim, 1)

    def _validate_config_indices(self) -> None:
        for name, indices in (
            ("predictor_config_indices", self.config.predictor_config_indices),
            ("streaming_config_indices", self.config.streaming_config_indices),
        ):
            if len(set(indices)) != len(indices):
                raise ValueError(f"{name} contains duplicate indices")
            if any(index < 0 or index >= self.config.config_dim for index in indices):
                raise ValueError(f"{name} contains an index outside config_dim")
        overlap = set(self.config.predictor_config_indices).intersection(
            self.config.streaming_config_indices
        )
        if overlap:
            raise ValueError(
                "predictor and streaming configuration fields must be disjoint; "
                f"overlap={sorted(overlap)}"
            )

    @staticmethod
    def _select_config(
        config: torch.Tensor,
        indices: tuple[int, ...] | list[int],
    ) -> torch.Tensor:
        if not indices:
            return config[..., :0]
        return config[..., list(indices)]

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.network_dynamics.initial_hidden(batch_size, device)

    def step(
        self,
        network: torch.Tensor,
        prediction: torch.Tensor,
        streaming: torch.Tensor,
        action: torch.Tensor,
        config: torch.Tensor,
        context: torch.Tensor,
        hidden: torch.Tensor,
        sample: bool = True,
        posterior_network: torch.Tensor | None = None,
        analytical_streaming: torch.Tensor | None = None,
    ) -> WorldModelOutput:
        network_context = context if self.config.network_uses_context else torch.zeros_like(context)
        (
            network_mean,
            network_log_std,
            hidden,
            latent,
            prior_mean,
            prior_log_std,
            posterior_mean,
            posterior_log_std,
            continuation_logits,
        ) = self.network_dynamics.step_with_posterior(
            network,
            network_context,
            hidden,
            sample and self.config.stochastic_dynamics,
            posterior_network=posterior_network,
        )
        effective_config = config
        if self.config.config_controller_only:
            effective_config = torch.cat(
                [config[..., :4], torch.zeros_like(config[..., 4:])],
                dim=-1,
            )
        if self.config.monolithic_cross_layer:
            if self.joint_cross_layer_backbone is None:
                raise RuntimeError("monolithic cross-layer backbone is missing")
            action_features = (
                action
                if self.config.streaming_uses_action
                else torch.zeros_like(action)
            )
            joint_features = self.joint_cross_layer_backbone(
                torch.cat(
                    [
                        hidden,
                        latent,
                        prediction,
                        streaming,
                        action_features,
                        effective_config,
                    ],
                    dim=-1,
                )
            )
            prediction_features = joint_features
        else:
            if self.prediction_backbone is None:
                raise RuntimeError("factorized prediction backbone is missing")
            predictor_config = self._select_config(
                effective_config,
                self.config.predictor_config_indices,
            )
            if not self.config.prediction_uses_config:
                predictor_config = torch.zeros_like(predictor_config)
            prediction_features = self.prediction_backbone(
                torch.cat(
                    [hidden, latent, prediction, predictor_config],
                    dim=-1,
                )
            )
        prediction_delta, prediction_log_std = self.prediction_head(prediction_features)
        prediction_mean = prediction + prediction_delta
        prediction_risk_logits = self.prediction_risk_head(prediction_features)
        if self.config.monolithic_cross_layer:
            streaming_features = prediction_features
        else:
            if self.streaming_backbone is None:
                raise RuntimeError("factorized streaming backbone is missing")
            streaming_prediction = (
                prediction_mean
                if self.config.streaming_uses_prediction
                else torch.zeros_like(prediction_mean)
            )
            streaming_action = (
                action
                if self.config.streaming_uses_action
                else torch.zeros_like(action)
            )
            streaming_config = self._select_config(
                effective_config,
                self.config.streaming_config_indices,
            )
            streaming_features = self.streaming_backbone(
                torch.cat(
                    [
                        hidden,
                        latent,
                        streaming_prediction,
                        streaming,
                        streaming_action,
                        streaming_config,
                    ],
                    dim=-1,
                )
            )
        streaming_residual_mean, streaming_residual_log_std = self.streaming_residual_head(
            streaming_features
        )
        baseline = streaming if analytical_streaming is None else analytical_streaming
        streaming_mean = baseline + streaming_residual_mean
        reward_mean, reward_log_std = self.reward_head(streaming_features)
        rebuffer_mean, rebuffer_log_std = self.rebuffer_head(streaming_features)
        return WorldModelOutput(
            network_mean=network_mean,
            network_log_std=network_log_std,
            prediction_mean=prediction_mean,
            prediction_log_std=prediction_log_std,
            streaming_mean=streaming_mean,
            streaming_log_std=streaming_residual_log_std,
            streaming_residual_mean=streaming_residual_mean,
            streaming_residual_log_std=streaming_residual_log_std,
            analytical_streaming_baseline=baseline,
            prediction_risk_logits=prediction_risk_logits,
            streaming_risk_logits=self.streaming_risk_head(streaming_features),
            reward_mean=reward_mean,
            reward_log_std=reward_log_std,
            rebuffer_mean=rebuffer_mean,
            rebuffer_log_std=rebuffer_log_std,
            continuation_logits=continuation_logits,
            prior_mean=prior_mean,
            prior_log_std=prior_log_std,
            posterior_mean=posterior_mean,
            posterior_log_std=posterior_log_std,
            hidden=hidden,
        )

    def forward(
        self,
        network: torch.Tensor,
        prediction: torch.Tensor,
        streaming: torch.Tensor,
        action: torch.Tensor,
        config: torch.Tensor,
        context: torch.Tensor,
        sample: bool = True,
        posterior_network: torch.Tensor | None = None,
        analytical_streaming: torch.Tensor | None = None,
    ) -> WorldModelOutput:
        batch_size, steps, _ = network.shape
        hidden = self.initial_hidden(batch_size, network.device)
        outputs: list[WorldModelOutput] = []
        for step in range(steps):
            outputs.append(
                self.step(
                    network[:, step],
                    prediction[:, step],
                    streaming[:, step],
                    action[:, step],
                    config,
                    context,
                    hidden,
                    sample,
                    posterior_network=(
                        None
                        if posterior_network is None
                        else posterior_network[:, step]
                    ),
                    analytical_streaming=(
                        None
                        if analytical_streaming is None
                        else analytical_streaming[:, step]
                    ),
                )
            )
            hidden = outputs[-1].hidden
        return WorldModelOutput(
            network_mean=torch.stack([item.network_mean for item in outputs], dim=1),
            network_log_std=torch.stack([item.network_log_std for item in outputs], dim=1),
            prediction_mean=torch.stack([item.prediction_mean for item in outputs], dim=1),
            prediction_log_std=torch.stack([item.prediction_log_std for item in outputs], dim=1),
            streaming_mean=torch.stack([item.streaming_mean for item in outputs], dim=1),
            streaming_log_std=torch.stack([item.streaming_log_std for item in outputs], dim=1),
            streaming_residual_mean=torch.stack(
                [item.streaming_residual_mean for item in outputs], dim=1
            ),
            streaming_residual_log_std=torch.stack(
                [item.streaming_residual_log_std for item in outputs], dim=1
            ),
            analytical_streaming_baseline=torch.stack(
                [item.analytical_streaming_baseline for item in outputs], dim=1
            ),
            prediction_risk_logits=torch.stack(
                [item.prediction_risk_logits for item in outputs], dim=1
            ),
            streaming_risk_logits=torch.stack(
                [item.streaming_risk_logits for item in outputs], dim=1
            ),
            reward_mean=torch.stack([item.reward_mean for item in outputs], dim=1),
            reward_log_std=torch.stack([item.reward_log_std for item in outputs], dim=1),
            rebuffer_mean=torch.stack([item.rebuffer_mean for item in outputs], dim=1),
            rebuffer_log_std=torch.stack([item.rebuffer_log_std for item in outputs], dim=1),
            continuation_logits=torch.stack(
                [item.continuation_logits for item in outputs], dim=1
            ),
            prior_mean=torch.stack([item.prior_mean for item in outputs], dim=1),
            prior_log_std=torch.stack([item.prior_log_std for item in outputs], dim=1),
            posterior_mean=torch.stack([item.posterior_mean for item in outputs], dim=1),
            posterior_log_std=torch.stack(
                [item.posterior_log_std for item in outputs], dim=1
            ),
            hidden=hidden,
        )
