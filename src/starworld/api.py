"""Training, calibration, and shared-future imagination through a small public API."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import random

import numpy as np
import torch

from starworld.config import ModelConfig, TrainingConfig
from starworld.data.split import grouped_source_run_split
from starworld.evaluation.calibration import load_trust_calibration, save_trust_calibration
from starworld.interfaces.policies import ThroughputFractionPolicy
from starworld.interfaces.predictors import SafeCapacityPredictor
from starworld.models.world_model import StarWorldModel
from starworld.rollout.engine import HybridRolloutEngine
from starworld.rollout.trust import TrustGate
from starworld.training.dataset import SequenceDataset, fit_normalizers
from starworld.training.playback import refine_playback
from starworld.training.trainer import Trainer, load_checkpoint


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class StarWorld:
    """A trained ensemble, its state normalization, and its calibrated Trust Gate."""

    def __init__(self, models, normalizers, manifest, *, gate=None, device="cpu", directory=None):
        self.models, self.normalizers, self.manifest = models, normalizers, manifest
        self.gate, self.device = gate, device
        self.directory = None if directory is None else Path(directory)

    @classmethod
    def fit(cls, episodes, output, *, model_config=None, training_config=None,
            members=2, device="cpu", seed=7, refinement_epochs=0,
            split_seed=17, train_fraction=.6, validation_fraction=.2):
        if members < 1:
            raise ValueError("members must be positive")
        output = Path(output)
        if (output / "training_manifest.json").exists():
            raise FileExistsError("model directory already contains a fitted ensemble; choose a new output directory")
        output.mkdir(parents=True, exist_ok=True)
        split = grouped_source_run_split(episodes, train_fraction=train_fraction,
                                         validation_fraction=validation_fraction, seed=split_seed)
        groups = {name: [ep for ep in episodes if split.split_by_source_run[ep.source_run_id] == name]
                  for name in ("train", "validation", "test")}
        normalizers = fit_normalizers(groups["train"])
        model_config, training_config = model_config or ModelConfig(), training_config or TrainingConfig()
        if training_config.epochs < 1:
            raise ValueError("epochs must be positive")
        datasets = {name: SequenceDataset(groups[name], training_config.sequence_length, normalizers,
                                         use_analytical_baseline=training_config.use_analytical_streaming)
                    for name in ("train", "validation")}
        checkpoint_paths = []
        for member in range(members):
            seed_everything(seed + member)
            folder = output / f"member_{member:02d}"
            Trainer(StarWorldModel(model_config), training_config, device).fit(
                datasets["train"], datasets["validation"], normalizers, folder)
            checkpoint = folder / "best.pt"
            if refinement_epochs:
                model, _, _ = load_checkpoint(checkpoint, device)
                checkpoint = folder / "refined.pt"
                history = refine_playback(model, normalizers, datasets["train"], datasets["validation"],
                                          epochs=refinement_epochs, output=checkpoint, device=device)
                (folder / "refinement.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            checkpoint_paths.append(str(checkpoint.relative_to(output)))
        manifest = {"split_by_source_run": split.split_by_source_run,
                    "train_ids": split.train_ids, "validation_ids": split.validation_ids,
                    "test_ids": split.test_ids, "members": checkpoint_paths,
                    "model_config": asdict(model_config), "training_config": asdict(training_config),
                    "refinement_epochs": refinement_epochs, "seed": seed, "split_seed": split_seed}
        (output / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return cls.load(output, device=device)

    @classmethod
    def load(cls, directory, *, device="cpu"):
        directory = Path(directory)
        manifest = json.loads((directory / "training_manifest.json").read_text(encoding="utf-8"))
        loaded = [load_checkpoint(directory / p, device) for p in manifest["members"]]
        calibration = directory / "trust_calibration.json"
        gate = load_trust_calibration(calibration) if calibration.exists() else None
        return cls([item[0] for item in loaded], loaded[0][1], manifest, gate=gate,
                   device=device, directory=directory)

    def split(self, episodes, name):
        if name not in ("train", "validation", "test"):
            raise ValueError("split must be train, validation, or test")
        return [ep for ep in episodes if self.manifest["split_by_source_run"].get(ep.source_run_id) == name]

    def engine(self, *, policy=None, predictor=None, use_gate=True):
        return HybridRolloutEngine(self.models[0], predictor or SafeCapacityPredictor(),
                                   policy or ThroughputFractionPolicy(), device=self.device,
                                   normalizers=self.normalizers, ensemble_models=self.models,
                                   trust_gate=self.gate if use_gate else None)

    @staticmethod
    def query(episode, prefix, horizon, samples, seed):
        if prefix < 1 or prefix > episode.length:
            raise ValueError("prefix must identify an observed state in the episode")
        index = prefix - 1
        auxiliary = episode.targets.get("analytical_inflight")
        return {"network_prefix": episode.network[:prefix],
                "prediction_state": episode.prediction[index],
                "streaming_state": episode.streaming[index],
                "config": episode.config, "context": episode.context,
                "horizon": horizon, "samples": samples, "seed": seed,
                "streaming_auxiliary_state": None if auxiliary is None else auxiliary[index]}

    def calibrate(self, episodes, *, horizon=6, samples=8, max_queries=24, target_coverage=.8):
        if horizon < 1 or samples < 2 or max_queries < 1:
            raise ValueError("use a positive horizon/query count and at least two samples")
        validation = self.split(episodes, "validation")
        records = []
        engine = self.engine(use_gate=False)
        for ep in validation:
            if ep.controller == "NetworkOnly" or ep.length <= horizon + 1:
                continue
            for index in range(0, ep.length - horizon, max(horizon, 1)):
                result = engine.rollout(**self.query(ep, index + 1, horizon, samples, 100 + len(records)),
                                        recorded_actions=ep.action[index:index + horizon])
                errors = np.abs(result.network.mean(0) - ep.network[index + 1:index + horizon + 1]).mean(-1)
                for h, (uncertainty, error) in enumerate(zip(result.uncertainty(), errors), 1):
                    records.append((h, float(uncertainty), float(error)))
                if len(records) >= max_queries * horizon:
                    break
            if len(records) >= max_queries * horizon:
                break
        if not records:
            raise ValueError("validation split needs replay episodes longer than the calibration horizon")
        gate = TrustGate(target_coverage)
        gate.set_support(self.normalizers.query_support_center, self.normalizers.query_support_precision,
                         self.normalizers.query_support_radius)
        records = np.asarray(records)
        gate.calibrate(records[:, 0].astype(int), records[:, 1], records[:, 2])
        self.gate = gate
        diagnostics = {"validation_queries": len(records) // horizon, "horizon_steps": horizon,
                       "samples_per_query": samples, "support_source": "training network/configuration/action states",
                       "error_target": "mean absolute network-state error", "uncertainty": "rollout path and predictive spread"}
        if self.directory is not None:
            save_trust_calibration(gate, diagnostics, self.directory / "trust_calibration.json")
        return diagnostics

    def imagine(self, episode, *, prefix=6, horizon=6, samples=16, seed=7, policies=None):
        policies = policies or {"balanced": ThroughputFractionPolicy(.85)}
        results = {name: self.engine(policy=policy).rollout(**self.query(episode, prefix, horizon, samples, seed))
                   for name, policy in policies.items()}
        anchor = next(iter(results.values())).network
        if any(not np.array_equal(result.network, anchor) for result in results.values()):
            raise RuntimeError("configuration branches must share the same sampled network futures")
        return results
