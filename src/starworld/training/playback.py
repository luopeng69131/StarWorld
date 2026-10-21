"""Multi-step refinement of the existing playback correction."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from starworld.interfaces.torch_streaming import playback_interval


def playback_loss(model, normalizers, batch, *, stochastic: bool, residual_scale=.5):
    device = batch["network"].device
    constants = {name: (torch.as_tensor(getattr(normalizers, name).mean, device=device),
                        torch.as_tensor(getattr(normalizers, name).scale, device=device))
                 for name in ("network", "streaming", "action")}

    def inverse(name, value):
        mean, scale = constants[name]
        return value * scale + mean

    mean, scale = constants["streaming"]
    network, prediction, streaming = (batch[n][:, 0] for n in ("network", "prediction", "streaming"))
    hidden = model.initial_hidden(len(network), device)
    auxiliary = batch["initial_analytical_inflight"]
    error_scale = torch.stack((scale[0].clamp_min(10), scale[2].clamp_min(2)))
    losses = []
    for step in range(min(batch["network"].shape[1], 12)):
        action = batch["action"][:, step]
        out = model.step(network, prediction, streaming, action, batch["config"],
                         batch["context"], hidden, sample=stochastic)
        hidden, network = out.hidden, out.network_mean
        if stochastic:
            network = network + torch.randn_like(network) * out.network_log_std.exp()
        previous = inverse("streaming", streaming)
        analytical, auxiliary = playback_interval(previous, auxiliary,
                                                  inverse("network", network)[:, 0],
                                                  inverse("action", action))
        corrected = analytical + residual_scale * out.streaming_residual_mean * scale
        corrected = torch.stack((corrected[:, 0].clamp_min(0), corrected[:, 1].clamp_min(0),
                                 torch.maximum(previous[:, 2], corrected[:, 2]),
                                 corrected[:, 3].clamp_min(0), corrected[:, 4].clamp(0, 1)), -1)
        streaming = (corrected - mean) / scale
        mask = batch["target_streaming_valid"][:, step, 0]
        truth = inverse("streaming", batch["target_streaming"][:, step])
        error = (corrected[:, [0, 2]] - truth[:, [0, 2]]).abs() / error_scale
        weight = 1.0 if step + 1 in (3, 6, 12) else .1
        losses.append(weight * (error.sum(-1) * mask).sum() / mask.sum().clamp_min(1))
        prediction = out.prediction_mean
    return torch.stack(losses).mean()


def refine_playback(model, normalizers, train_data, validation_data, *, epochs: int,
                    output: Path, learning_rate=3e-4, batch_size=128, device="cpu") -> dict:
    if epochs < 1:
        raise ValueError("refinement epochs must be positive")
    if model.streaming_backbone is None:
        raise ValueError("playback refinement uses the factorized streaming backbone")
    model = model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (model.streaming_backbone, model.streaming_residual_head.mean):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-4)
    loaders = {"train": DataLoader(train_data, batch_size=batch_size, shuffle=True),
               "validation": DataLoader(validation_data, batch_size=batch_size)}
    best, best_state, history = float("inf"), None, []
    for epoch in range(epochs + 1):
        row = {"epoch": epoch}
        for split, loader in loaders.items():
            training = split == "train" and epoch > 0
            if split == "train" and epoch == 0:
                continue
            model.eval()
            model.streaming_backbone.train(training)
            scores = []
            for batch in loader:
                if batch["target_streaming_valid"].sum() == 0:
                    continue
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.set_grad_enabled(training):
                    loss = playback_loss(model, normalizers, batch, stochastic=training)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, 2.)
                    optimizer.step()
                scores.append(float(loss.detach()))
            if not scores:
                raise ValueError(f"{split} split needs replay-supervised playback states")
            row[split] = float(np.mean(scores))
        history.append(row)
        if row["validation"] < best:
            best, best_state = row["validation"], deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "model_config": vars(model.config),
                "normalizers": normalizers.to_dict(), "validation_loss": best,
                "refinement": {"epochs": epochs, "learning_rate": learning_rate,
                               "residual_scale": .5, "selection": "validation multi-step playback loss"}}, output)
    return {"validation_loss": best, "history": history}
