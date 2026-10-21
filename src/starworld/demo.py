"""CPU example: procedural sequences -> training -> calibration -> imagination."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from starworld.api import StarWorld
from starworld.config import ModelConfig, TrainingConfig
from starworld.data.synthetic import make_synthetic_episodes
from starworld.interfaces.policies import ThroughputFractionPolicy


def save_results(results, output: Path, *, provenance="user-provided episodes") -> dict:
    output.mkdir(parents=True, exist_ok=True)
    reference_name = next(iter(results))
    reference = results[reference_name]
    summary = {"input": provenance, "steps": int(reference.network.shape[1]),
               "samples": int(reference.network.shape[0]), "state_interval_s": 10,
               "branches": {}, "paired_effects": {}}
    for name, result in results.items():
        row = result.summary()
        row["trust_decision"] = ("uncalibrated" if result.trust_accepted is None else
                                 "accepted" if result.trust_accepted.all() else "abstained")
        if result.trust_reason is not None:
            row["trust_reasons"] = sorted(set(result.trust_reason.ravel().tolist()))
        summary["branches"][name] = row
        np.savez_compressed(output / f"{name}.npz", network=result.network, streaming=result.streaming,
                            prediction=result.capacity, actions=result.actions,
                            trust_accepted=result.trust_accepted if result.trust_accepted is not None else np.zeros(0))
        if name != reference_name:
            difference = result.streaming[:, -1, 2] - reference.streaming[:, -1, 2]
            summary["paired_effects"][f"{name} - {reference_name}"] = {
                "mean_final_rebuffer_difference_s": float(difference.mean()),
                "p05_s": float(np.quantile(difference, .05)), "p95_s": float(np.quantile(difference, .95))}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def plot_results(results, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.8), layout="constrained")
    first = next(iter(results.values()))
    time = 10 * np.arange(1, first.network.shape[1] + 1)
    for path in first.network[:8, :, 0]:
        axes[0].plot(time, path, color="#6879B4", alpha=.45, lw=1)
    axes[0].set(title="Sampled network futures", ylabel="Throughput (Mbps)")
    for name, result in results.items():
        for ax, dim in zip(axes[1:], (0, 2)):
            values = result.streaming[..., dim]
            line, = ax.plot(time, values.mean(0), label=name, lw=1.8)
            ax.fill_between(time, np.quantile(values, .1, axis=0), np.quantile(values, .9, axis=0),
                            color=line.get_color(), alpha=.14, lw=0)
    axes[1].set(title="Imagined playback buffer", ylabel="Buffer (s)")
    axes[2].set(title="Imagined rebuffering", ylabel="Cumulative rebuffer (s)")
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xlabel("Future time (s)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.16)
    fig.savefig(output / "imagination.png", dpi=200, bbox_inches="tight", pad_inches=.04)
    plt.close(fig)


def run_demo(output, *, epochs=2, members=2, plot=False) -> dict:
    torch.set_num_threads(2)
    output = Path(output)
    episodes = make_synthetic_episodes(count=24, steps=20)
    training = TrainingConfig(epochs=epochs, batch_size=16, sequence_length=8, rollout_horizon=3)
    world = StarWorld.fit(episodes, output / "model", model_config=ModelConfig(hidden_dim=32, latent_dim=8),
                         training_config=training, members=members, refinement_epochs=1)
    world.calibrate(episodes, horizon=3, samples=6, max_queries=6)
    episode = world.split(episodes, "test")[0]
    results = world.imagine(episode, prefix=6, horizon=3, samples=12,
                            policies={"conservative": ThroughputFractionPolicy(.6),
                                      "responsive": ThroughputFractionPolicy(.95)})
    summary = save_results(results, output, provenance="procedurally generated synthetic sequences")
    if plot:
        plot_results(results, output)
    return summary
