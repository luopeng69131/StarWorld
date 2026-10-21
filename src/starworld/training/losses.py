from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from starworld.interfaces.streaming import (
    apply_streaming_auxiliary,
    StreamingState,
    transition_streaming_interval,
)
from starworld.models.components import diagonal_gaussian_kl
from starworld.models.world_model import WorldModelOutput
from starworld.models.world_model import StarWorldModel
from starworld.training.dataset import NormalizerBundle


def gaussian_nll(
    target: torch.Tensor,
    mean: torch.Tensor,
    log_std: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    temporal_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = 0.5 * (((target - mean) / log_std.exp()) ** 2 + 2.0 * log_std + math.log(2.0 * math.pi))
    loss = loss.mean(dim=-1)
    if temporal_mask is not None:
        mask = temporal_mask[..., 0]
        loss = (loss * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
    else:
        loss = loss.mean(dim=-1)
    if sample_weight is not None:
        return (loss * sample_weight).sum() / sample_weight.sum().clamp_min(1e-8)
    return loss.mean()


def masked_gaussian_nll(
    target: torch.Tensor,
    mean: torch.Tensor,
    log_std: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    pointwise = 0.5 * (
        ((target - mean) / log_std.exp()) ** 2
        + 2.0 * log_std
        + math.log(2.0 * math.pi)
    )
    expanded_mask = mask.expand_as(pointwise)
    return (pointwise * expanded_mask).sum() / expanded_mask.sum().clamp_min(1.0)


def masked_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    pointwise = F.smooth_l1_loss(prediction, target, reduction="none")
    weighted = pointwise * mask
    return weighted.sum() / mask.sum().clamp_min(1.0)


def masked_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    return ((prediction - target).abs() * mask).sum() / mask.sum().clamp_min(1.0)


def branch_weighted_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor,
    temporal_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    pointwise = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
    ).mean(dim=-1)
    if temporal_mask is not None:
        mask = temporal_mask[..., 0]
        pointwise = (
            (pointwise * mask).sum(dim=-1)
            / mask.sum(dim=-1).clamp_min(1.0)
        )
    else:
        pointwise = pointwise.mean(dim=-1)
    return (
        pointwise * sample_weight
    ).sum() / sample_weight.sum().clamp_min(1e-8)


def branch_weighted_mean(
    value: torch.Tensor,
    sample_weight: torch.Tensor,
    temporal_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if temporal_mask is not None:
        mask = temporal_mask
        while mask.ndim > value.ndim and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        while mask.ndim < value.ndim:
            mask = mask.unsqueeze(-1)
        mask = mask.expand_as(value)
        per_sample = (
            (value * mask).reshape(value.shape[0], -1).sum(dim=-1)
            / mask.reshape(value.shape[0], -1).sum(dim=-1).clamp_min(1.0)
        )
    else:
        per_sample = value.reshape(value.shape[0], -1).mean(dim=-1)
    return (
        per_sample * sample_weight
    ).sum() / sample_weight.sum().clamp_min(1e-8)


def world_model_loss(
    output: WorldModelOutput,
    batch: dict[str, torch.Tensor],
    weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    weight = {
        "network": 1.0,
        "prediction": 0.5,
        "streaming": 1.0,
        "event": 0.5,
        "tail": 0.25,
        "reward": 0.25,
        "rebuffer": 0.5,
        "continuation": 0.05,
        "latent_kl": 0.01,
        "latent_free_nats": 0.5,
        **(weights or {}),
    }
    network = gaussian_nll(
        batch["target_network"],
        output.network_mean,
        output.network_log_std,
        batch["network_weight"],
        batch["target_temporal_valid"],
    )
    prediction = gaussian_nll(
        batch["target_prediction"],
        output.prediction_mean,
        output.prediction_log_std,
        batch["network_weight"],
        batch["target_temporal_valid"],
    )
    streaming_residual = masked_gaussian_nll(
        batch["target_streaming_residual"],
        output.streaming_residual_mean,
        output.streaming_residual_log_std,
        batch["target_streaming_valid"],
    )
    prediction_risk = branch_weighted_bce(
        output.prediction_risk_logits,
        batch["target_prediction_risk"],
        batch["network_weight"],
        batch["target_temporal_valid"],
    )
    streaming_risk_pointwise = F.binary_cross_entropy_with_logits(
        output.streaming_risk_logits,
        batch["target_streaming_risk"],
        reduction="none",
    )
    streaming_risk = (
        streaming_risk_pointwise * batch["target_streaming_valid"]
    ).sum() / batch["target_streaming_valid"].sum().clamp_min(1.0)
    event = 0.5 * (prediction_risk + streaming_risk)
    reward = masked_smooth_l1(
        output.reward_mean,
        batch["target_reward"],
        batch["reward_available"],
    )
    rebuffer = masked_smooth_l1(
        output.rebuffer_mean,
        batch["target_rebuffer"],
        batch["rebuffer_available"],
    )
    continuation_pointwise = F.binary_cross_entropy_with_logits(
        output.continuation_logits,
        batch["target_continuation"],
        reduction="none",
    )
    continuation = branch_weighted_mean(
        continuation_pointwise,
        batch["network_weight"],
        batch["target_temporal_valid"],
    )
    latent_kl_per_step = diagonal_gaussian_kl(
        output.posterior_mean,
        output.posterior_log_std,
        output.prior_mean,
        output.prior_log_std,
    )
    latent_kl_clamped = torch.clamp(
        latent_kl_per_step,
        min=float(weight["latent_free_nats"]),
    )
    latent_kl = branch_weighted_mean(
        latent_kl_clamped,
        batch["network_weight"],
        batch["target_temporal_valid"],
    )
    p05_target = batch["target_network"][..., 1]
    p05_prediction = output.network_mean[..., 1]
    valid_network = (
        (batch["target_temporal_valid"][..., 0] > 0.5)
        & (batch["network_weight"][:, None] > 0.0)
    )
    valid_p05 = p05_target.detach()[valid_network]
    low_tail_cutoff = (
        torch.quantile(valid_p05, 0.2)
        if valid_p05.numel()
        else torch.zeros((), device=p05_target.device)
    )
    tail_mask = (
        (p05_target <= low_tail_cutoff).float()
        * batch["target_temporal_valid"][..., 0]
    )
    branch_weight = batch["network_weight"][:, None]
    weighted_tail_mask = tail_mask * branch_weight
    tail = (
        torch.abs(p05_prediction - p05_target) * weighted_tail_mask
    ).sum() / weighted_tail_mask.sum().clamp_min(1.0)
    total = (
        weight["network"] * network
        + weight["prediction"] * prediction
        + weight["streaming"] * streaming_residual
        + weight["event"] * event
        + weight["tail"] * tail
        + weight["reward"] * reward
        + weight["rebuffer"] * rebuffer
        + weight["continuation"] * continuation
        + weight["latent_kl"] * latent_kl
    )
    streaming_dimension_names = (
        "buffer",
        "bitrate",
        "cumulative_rebuffer",
        "download_time",
        "playback_progress",
    )
    metrics = {
        "loss": float(total.detach()),
        "network_nll": float(network.detach()),
        "prediction_nll": float(prediction.detach()),
        "streaming_residual_nll": float(streaming_residual.detach()),
        "streaming_nll": float(streaming_residual.detach()),
        "event_bce": float(event.detach()),
        "prediction_risk_bce": float(prediction_risk.detach()),
        "streaming_risk_bce": float(streaming_risk.detach()),
        "tail_mae": float(tail.detach()),
        "reward_huber": float(reward.detach()),
        "reward_mae": float(
            masked_mae(
                output.reward_mean,
                batch["target_reward"],
                batch["reward_available"],
            ).detach()
        ),
        "rebuffer_huber": float(rebuffer.detach()),
        "rebuffer_mae": float(
            masked_mae(
                output.rebuffer_mean,
                batch["target_rebuffer"],
                batch["rebuffer_available"],
            ).detach()
        ),
        "continuation_bce": float(continuation.detach()),
        "latent_kl": float(latent_kl_per_step.mean().detach()),
        "analytical_baseline_fraction": float(
            batch["analytical_baseline_available"].mean().detach()
        ),
    }
    for index, name in enumerate(streaming_dimension_names):
        if index >= output.streaming_mean.shape[-1]:
            break
        mask = batch["target_streaming_valid"][..., 0]
        error = (
            output.streaming_mean[..., index]
            - batch["target_streaming"][..., index]
        ).abs()
        metrics[f"streaming_{name}_mae"] = float(
            ((error * mask).sum() / mask.sum().clamp_min(1.0)).detach()
        )
    return total, metrics


def multi_step_rollout_loss(
    model: StarWorldModel,
    batch: dict[str, torch.Tensor],
    horizon: int,
    normalizers: NormalizerBundle | None = None,
    use_analytical_baseline: bool = True,
) -> torch.Tensor:
    """Free-run latent states while retaining the recorded action sequence."""
    steps = min(horizon, batch["network"].shape[1])
    hidden = model.initial_hidden(batch["network"].shape[0], batch["network"].device)
    network = batch["network"][:, 0]
    prediction = batch["prediction"][:, 0]
    streaming = batch["streaming"][:, 0]
    initial_auxiliary = batch.get("initial_analytical_inflight")
    analytical_states: list[StreamingState] | None = (
        None
        if initial_auxiliary is None
        else [
            apply_streaming_auxiliary(StreamingState(), row)
            for row in initial_auxiliary.detach().cpu().numpy()
        ]
    )
    losses: list[torch.Tensor] = []
    for step in range(steps):
        output = model.step(
            network,
            prediction,
            streaming,
            batch["action"][:, step],
            batch["config"],
            batch["context"],
            hidden,
            sample=model.training,
            analytical_streaming=None,
        )
        hidden = output.hidden
        if use_analytical_baseline and normalizers is not None:
            raw_streaming = normalizers.streaming.inverse(
                streaming.detach().cpu().numpy()
            )
            raw_action = normalizers.action.inverse(
                batch["action"][:, step].detach().cpu().numpy()
            )
            raw_network = normalizers.network.inverse(
                output.network_mean.detach().cpu().numpy()
            )
            analytical_rows: list[list[float]] = []
            next_analytical_states: list[StreamingState] = []
            for row_index, (stream_row, action_row, network_row) in enumerate(
                zip(raw_streaming, raw_action, raw_network)
            ):
                state = StreamingState(
                    buffer_s=float(stream_row[0]),
                    last_bitrate_mbps=float(stream_row[1]),
                    cumulative_rebuffer_s=float(stream_row[2]),
                    last_download_s=float(stream_row[3]),
                    playback_progress=float(stream_row[4]),
                )
                if analytical_states is not None:
                    previous = analytical_states[row_index]
                    state.inflight_remaining_megabits = (
                        previous.inflight_remaining_megabits
                    )
                    state.inflight_bitrate_mbps = (
                        previous.inflight_bitrate_mbps
                    )
                    state.inflight_elapsed_s = previous.inflight_elapsed_s
                    state.inflight_rebuffer_s = previous.inflight_rebuffer_s
                next_state = transition_streaming_interval(
                    state,
                    throughput_mbps=max(float(network_row[0]), 1e-3),
                    bitrate_mbps=max(float(action_row[0]), 0.0),
                    decision_count=max(int(round(float(action_row[2]))), 0),
                    chunk_duration_s=4.0,
                ).state
                next_analytical_states.append(next_state)
                analytical_rows.append(
                    [
                        next_state.buffer_s,
                        next_state.last_bitrate_mbps,
                        next_state.cumulative_rebuffer_s,
                        next_state.last_download_s,
                        next_state.playback_progress,
                    ]
                )
            analytical_states = next_analytical_states
            analytical = torch.as_tensor(
                normalizers.streaming.transform(
                    np.asarray(analytical_rows, np.float32)
                ),
                device=streaming.device,
                dtype=streaming.dtype,
            )
            streaming_next = analytical + output.streaming_residual_mean
        else:
            streaming_next = streaming + output.streaming_residual_mean
        network_loss = gaussian_nll(
            batch["target_network"][:, step : step + 1],
            output.network_mean[:, None],
            output.network_log_std[:, None],
            batch["network_weight"],
            batch["target_temporal_valid"][:, step : step + 1],
        )
        prediction_loss = gaussian_nll(
            batch["target_prediction"][:, step : step + 1],
            output.prediction_mean[:, None],
            output.prediction_log_std[:, None],
            batch["network_weight"],
            batch["target_temporal_valid"][:, step : step + 1],
        )
        streaming_loss = masked_gaussian_nll(
            batch["target_streaming"][:, step : step + 1],
            streaming_next[:, None],
            output.streaming_log_std[:, None],
            batch["target_streaming_valid"][:, step : step + 1],
        )
        prediction_risk_loss = branch_weighted_bce(
            output.prediction_risk_logits[:, None],
            batch["target_prediction_risk"][:, step : step + 1],
            batch["network_weight"],
            batch["target_temporal_valid"][:, step : step + 1],
        )
        streaming_risk_pointwise = F.binary_cross_entropy_with_logits(
            output.streaming_risk_logits,
            batch["target_streaming_risk"][:, step],
            reduction="none",
        )
        streaming_risk_loss = (
            streaming_risk_pointwise * batch["target_streaming_valid"][:, step]
        ).sum() / batch["target_streaming_valid"][:, step].sum().clamp_min(1.0)
        reward_loss = masked_smooth_l1(
            output.reward_mean,
            batch["target_reward"][:, step],
            batch["reward_available"][:, step],
        )
        rebuffer_loss = masked_smooth_l1(
            output.rebuffer_mean,
            batch["target_rebuffer"][:, step],
            batch["rebuffer_available"][:, step],
        )
        losses.append(
            network_loss
            + 0.5 * prediction_loss
            + streaming_loss
            + 0.25 * (prediction_risk_loss + streaming_risk_loss)
            + 0.1 * reward_loss
            + 0.2 * rebuffer_loss
        )
        network = output.network_mean
        prediction = output.prediction_mean
        streaming = streaming_next
    return torch.stack(losses).mean()
