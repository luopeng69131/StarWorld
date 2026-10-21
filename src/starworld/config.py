from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DataConfig:
    segments_csv: str
    prediction_csv: str
    abr_outcomes_csv: str
    output_dir: str = "outputs/dataset"
    split_seed: int = 17
    train_fraction: float = 0.6
    validation_fraction: float = 0.2
    minimum_steps: int = 8
    minimum_replay_steps: int = 2
    maximum_episodes: int | None = None
    include_network_only: bool = True


@dataclass
class ModelConfig:
    network_dim: int = 8
    prediction_dim: int = 4
    streaming_dim: int = 5
    context_dim: int = 3
    action_dim: int = 3
    config_dim: int = 7
    hidden_dim: int = 96
    latent_dim: int = 32
    dropout: float = 0.1
    ensemble_size: int = 3
    network_uses_context: bool = True
    prediction_uses_config: bool = True
    streaming_uses_action: bool = True
    streaming_uses_prediction: bool = True
    stochastic_dynamics: bool = True
    monolithic_cross_layer: bool = False
    network_residual_transition: bool = False
    config_controller_only: bool = False
    # CONFIG_FEATURES index 4 is the predictor interface. Controller identity,
    # bitrate cap, and chunk duration are streaming-side configuration.
    predictor_config_indices: tuple[int, ...] = (4,)
    streaming_config_indices: tuple[int, ...] = (0, 1, 2, 3, 5, 6)
    posterior_uses_next_network: bool = True


@dataclass
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    sequence_length: int = 24
    rollout_horizon: int = 6
    gradient_clip: float = 5.0
    checkpoint_dir: str = "outputs/checkpoints"
    network_loss_weight: float = 1.0
    prediction_loss_weight: float = 0.5
    streaming_loss_weight: float = 1.0
    event_loss_weight: float = 0.5
    tail_loss_weight: float = 0.25
    reward_loss_weight: float = 0.25
    rebuffer_loss_weight: float = 0.5
    continuation_loss_weight: float = 0.05
    latent_kl_weight: float = 0.01
    latent_free_nats: float = 0.5
    use_analytical_streaming: bool = True
    balance_layer_sampling: bool = True


@dataclass
class EvaluationConfig:
    rollout_horizons: list[int] = field(default_factory=lambda: [1, 3, 6, 12])
    samples: int = 32
    risk_threshold_s: float = 10.0
    target_coverage: float = 0.8
    output_dir: str = "outputs/evaluation"


@dataclass
class StarWorldConfig:
    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    seed: int = 7
    device: str = "cpu"

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "StarWorldConfig":
        return cls(
            data=DataConfig(**values["data"]),
            model=ModelConfig(**values.get("model", {})),
            training=TrainingConfig(**values.get("training", {})),
            evaluation=EvaluationConfig(**values.get("evaluation", {})),
            seed=int(values.get("seed", 7)),
            device=str(values.get("device", "cpu")),
        )


def load_config(path: str | Path) -> StarWorldConfig:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        values = json.load(handle)
    for name in ("segments_csv", "prediction_csv", "abr_outcomes_csv"):
        source = Path(values["data"][name]).expanduser()
        values["data"][name] = str(source if source.is_absolute() else path.parent / source)
    return StarWorldConfig.from_dict(values)
