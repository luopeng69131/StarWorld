from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from starworld.config import TrainingConfig
from starworld.models.world_model import StarWorldModel
from starworld.training.dataset import NormalizerBundle, SequenceDataset
from starworld.training.losses import multi_step_rollout_loss, world_model_loss


@dataclass
class TrainingHistory:
    train: list[dict[str, float]] = field(default_factory=list)
    validation: list[dict[str, float]] = field(default_factory=list)


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        name: float(np.mean([row[name] for row in rows]))
        for name in rows[0]
    }


class Trainer:
    def __init__(
        self,
        model: StarWorldModel,
        config: TrainingConfig,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.config = config
        self.device = torch.device(device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    def _run_epoch(
        self,
        loader: DataLoader,
        training: bool,
        rollout_horizon: int,
        normalizers: NormalizerBundle,
    ) -> dict[str, float]:
        self.model.train(training)
        rows: list[dict[str, float]] = []
        for batch in loader:
            batch = {name: value.to(self.device) for name, value in batch.items()}
            with torch.set_grad_enabled(training):
                output = self.model(
                    batch["network"],
                    batch["prediction"],
                    batch["streaming"],
                    batch["action"],
                    batch["config"],
                    batch["context"],
                    sample=training,
                    posterior_network=batch["target_network"],
                    analytical_streaming=batch["analytical_streaming"],
                )
                loss, metrics = world_model_loss(
                    output,
                    batch,
                    {
                        "network": self.config.network_loss_weight,
                        "prediction": self.config.prediction_loss_weight,
                        "streaming": self.config.streaming_loss_weight,
                        "event": self.config.event_loss_weight,
                        "tail": self.config.tail_loss_weight,
                        "reward": self.config.reward_loss_weight,
                        "rebuffer": self.config.rebuffer_loss_weight,
                        "continuation": self.config.continuation_loss_weight,
                        "latent_kl": self.config.latent_kl_weight,
                        "latent_free_nats": self.config.latent_free_nats,
                    },
                )
                if rollout_horizon > 0:
                    rollout_loss = multi_step_rollout_loss(
                        self.model,
                        batch,
                        rollout_horizon,
                        normalizers=normalizers,
                        use_analytical_baseline=(
                            self.config.use_analytical_streaming
                        ),
                    )
                    loss = loss + 0.25 * rollout_loss
                    metrics["rollout_loss"] = float(rollout_loss.detach())
                else:
                    metrics["rollout_loss"] = 0.0
                metrics["loss"] = float(loss.detach())
            if training:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.optimizer.step()
            rows.append(metrics)
        return _mean_metrics(rows)

    def fit(
        self,
        train_dataset: SequenceDataset,
        validation_dataset: SequenceDataset,
        normalizers: NormalizerBundle,
        checkpoint_dir: str | Path | None = None,
    ) -> TrainingHistory:
        sampler = (
            WeightedRandomSampler(
                weights=torch.as_tensor(
                    train_dataset.sample_weights,
                    dtype=torch.double,
                ),
                num_samples=len(train_dataset),
                replacement=True,
            )
            if self.config.balance_layer_sampling
            else None
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=0,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        )
        output = Path(checkpoint_dir or self.config.checkpoint_dir)
        output.mkdir(parents=True, exist_ok=True)
        history = TrainingHistory()
        best_validation = float("inf")
        for epoch in range(1, self.config.epochs + 1):
            rollout_horizon = int(
                np.ceil(self.config.rollout_horizon * epoch / max(self.config.epochs, 1))
            )
            train_metrics = self._run_epoch(
                train_loader,
                training=True,
                rollout_horizon=rollout_horizon,
                normalizers=normalizers,
            )
            validation_metrics = self._run_epoch(
                validation_loader,
                training=False,
                rollout_horizon=rollout_horizon,
                normalizers=normalizers,
            )
            train_metrics["epoch"] = float(epoch)
            train_metrics["rollout_horizon"] = float(rollout_horizon)
            validation_metrics["epoch"] = float(epoch)
            validation_metrics["rollout_horizon"] = float(rollout_horizon)
            history.train.append(train_metrics)
            history.validation.append(validation_metrics)
            if validation_metrics["loss"] < best_validation:
                best_validation = validation_metrics["loss"]
                self.save(output / "best.pt", normalizers, epoch, best_validation)
        (output / "history.json").write_text(
            json.dumps({"train": history.train, "validation": history.validation}, indent=2),
            encoding="utf-8",
        )
        return history

    def save(
        self,
        path: str | Path,
        normalizers: NormalizerBundle,
        epoch: int,
        validation_loss: float,
    ) -> None:
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "model_config": vars(self.model.config),
                "normalizers": normalizers.to_dict(),
                "epoch": epoch,
                "validation_loss": validation_loss,
            },
            path,
        )


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[StarWorldModel, NormalizerBundle, dict[str, object]]:
    from starworld.config import ModelConfig

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = StarWorldModel(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    normalizers = NormalizerBundle.from_dict(checkpoint["normalizers"])
    return model, normalizers, checkpoint
