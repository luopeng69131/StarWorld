"""Run the CPU walkthrough after installing StarWorld."""
import argparse
import json

from starworld.demo import run_demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/demo")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--members", type=int, default=2)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_demo(args.output, epochs=args.epochs, members=args.members, plot=args.plot), indent=2))
