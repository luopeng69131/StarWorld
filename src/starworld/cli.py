"""Portable command-line entry points for the StarWorld workflow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from starworld.api import StarWorld
from starworld.config import load_config
from starworld.data.episodes import CrossLayerExperienceConstructor, load_episode_archive
from starworld.data.prepare import prepare_starnet
from starworld.demo import plot_results, run_demo, save_results
from starworld.interfaces.policies import ThroughputFractionPolicy


def main():
    parser = argparse.ArgumentParser(prog="starworld", description="Starlink cross-layer imagination")
    parser.add_argument("--threads", type=int, default=2)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run the synthetic CPU walkthrough")
    demo.add_argument("--output", type=Path, default=Path("outputs/demo"))
    demo.add_argument("--epochs", type=int, default=2)
    demo.add_argument("--members", type=int, default=2)
    demo.add_argument("--plot", action="store_true")
    data = commands.add_parser("prepare-data", help="Prepare StarNet logs and causal window tables")
    data.add_argument("--input-root", type=Path, required=True)
    data.add_argument("--output", type=Path, required=True)
    data.add_argument("--region", choices=("US", "OSN", "VIC"))
    data.add_argument("--segment-seconds", type=int, default=600)
    data.add_argument("--history-seconds", type=int, default=60)
    data.add_argument("--max-gap-seconds", type=int, default=2)
    data.add_argument("--max-runs", type=int)
    data.add_argument("--replay-index", type=Path)
    build = commands.add_parser("build-dataset", help="Align measurement, prediction, and playback records")
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    train = commands.add_parser("train", help="Fit a stochastic ensemble")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--epochs", type=int)
    train.add_argument("--members", type=int)
    train.add_argument("--refine-epochs", type=int, default=0)
    for name in ("calibrate-trust", "imagine"):
        sub = commands.add_parser(name)
        sub.add_argument("--model", type=Path, required=True)
        sub.add_argument("--dataset", type=Path, required=True)
        sub.add_argument("--device", default="cpu")
        sub.add_argument("--horizon", type=int, default=6, help="Number of 10-second steps")
        sub.add_argument("--samples", type=int, default=16)
        if name == "calibrate-trust":
            sub.add_argument("--max-queries", type=int, default=24)
            sub.add_argument("--target-coverage", type=float, default=.8)
        else:
            sub.add_argument("--prefix", type=int, default=6)
            sub.add_argument("--output", type=Path, required=True)
            sub.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    if args.command == "demo":
        report = run_demo(args.output, epochs=args.epochs, members=args.members, plot=args.plot)
    elif args.command == "prepare-data":
        report = prepare_starnet(args.input_root, args.output, region=args.region,
                                segment_s=args.segment_seconds, history_s=args.history_seconds,
                                max_gap_s=args.max_gap_seconds, max_runs=args.max_runs,
                                replay_index=args.replay_index)
    elif args.command == "build-dataset":
        directory = CrossLayerExperienceConstructor(load_config(args.config).data).write(args.output)
        report = {"dataset": str(directory), "audit": str(directory / "audit.json")}
    elif args.command == "train":
        config = load_config(args.config)
        if args.epochs is not None:
            config.training.epochs = args.epochs
        episodes, _ = load_episode_archive(args.dataset)
        world = StarWorld.fit(episodes, args.output, model_config=config.model,
                             training_config=config.training, members=args.members or config.model.ensemble_size,
                             device=config.device, seed=config.seed, refinement_epochs=args.refine_epochs,
                             split_seed=config.data.split_seed, train_fraction=config.data.train_fraction,
                             validation_fraction=config.data.validation_fraction)
        report = {"model": str(world.directory), "members": len(world.models)}
    else:
        world = StarWorld.load(args.model, device=args.device)
        episodes, _ = load_episode_archive(args.dataset)
        if args.command == "calibrate-trust":
            report = world.calibrate(episodes, horizon=args.horizon, samples=args.samples,
                                     max_queries=args.max_queries, target_coverage=args.target_coverage)
        else:
            candidates = [ep for ep in world.split(episodes, "test")
                          if ep.controller != "NetworkOnly" and ep.length >= args.prefix]
            if not candidates:
                raise ValueError("test split needs a replay episode covering the requested prefix")
            results = world.imagine(candidates[0], prefix=args.prefix, horizon=args.horizon, samples=args.samples,
                                    policies={"conservative": ThroughputFractionPolicy(.6),
                                              "responsive": ThroughputFractionPolicy(.95)})
            report = save_results(results, args.output)
            if args.plot:
                plot_results(results, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
