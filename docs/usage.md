# Training and Imagination

## Installation

```bash
python -m pip install -e '.[viz]'
```

The base package requires NumPy, pandas, and PyTorch. The `viz` extra adds Matplotlib for the example visualization.

## CPU Walkthrough

```bash
starworld demo --epochs 2 --members 2 --plot --output outputs/demo
```

This generates synthetic episodes at runtime and runs the full public workflow. Choose a fresh output directory for another training run, for example `--output outputs/demo-next`.

The default two branches use throughput-fraction policies with fractions 0.6 and 0.95. They demonstrate distinct bitrate decisions under shared futures. `summary.json` identifies the synthetic input provenance and includes the model's actual trust decisions. Sample arrays are saved for plotting or inspection.

## Train on Prepared Starlink Episodes

Prepare measurement tables and attach replay logs as described in [data.md](data.md), then run:

```bash
starworld build-dataset \
  --config outputs/tables/config.json --output outputs/dataset

starworld train \
  --config configs/starnet.json \
  --dataset outputs/dataset \
  --output outputs/model \
  --members 3 --epochs 20 --refine-epochs 20
```

`configs/starnet.json` points to the tables in `outputs/tables/` relative to the configuration file. It specifies model dimensions, training parameters, and the device. Set `device` to `cuda` to use a CUDA-enabled PyTorch installation.

Training fits normalization and joint-support statistics to the training partition and selects checkpoints using the validation partition. Playback refinement updates the existing playback backbone and residual-mean head. `training_manifest.json` records the selected checkpoint paths, configuration, and collection-run partition.

## Calibrate the Trust Gate

```bash
starworld calibrate-trust \
  --model outputs/model --dataset outputs/dataset \
  --horizon 6 --samples 16 --max-queries 24
```

The horizon is expressed in 10-second steps: `6` means 60 seconds. Calibration uses validation replay episodes and saves `outputs/model/trust_calibration.json`. It compares sampled network futures with observed validation continuations while following recorded actions. The support region comes from training states, configurations, and actions.

## Imagine Shared-Future Outcomes

```bash
starworld imagine \
  --model outputs/model --dataset outputs/dataset \
  --prefix 6 --horizon 6 --samples 16 \
  --plot --output outputs/query
```

The CLI selects a replay-covered test episode with a sufficiently long observed prefix. At each future step, the selected policy acts on the evolving imagined state. A calibrated model produces reliability decisions; a model awaiting calibration records the status `uncalibrated`.

The corresponding Python API is:

```python
from starworld.api import StarWorld
from starworld.data.episodes import load_episode_archive
from starworld.interfaces.policies import ThroughputFractionPolicy

episodes, _ = load_episode_archive("outputs/dataset")
world = StarWorld.load("outputs/model")
episode = next(ep for ep in world.split(episodes, "test")
               if ep.controller != "NetworkOnly" and ep.length >= 6)

results = world.imagine(
    episode, prefix=6, horizon=6, samples=16,
    policies={
        "conservative": ThroughputFractionPolicy(0.6),
        "responsive": ThroughputFractionPolicy(0.95),
    },
)
for name, result in results.items():
    print(name, result.summary())
```

Each `RolloutResult` contains `network`, `capacity`, `streaming`, `actions`, `rewards`, and trust arrays. Their first two dimensions are sample and future step. The `streaming` feature order is buffer, last bitrate, cumulative rebuffering, last download duration, and playback progress.

## Extend Predictors and Policies

Implement `PredictionInterface.predict(history)` to supply capacity and uncertainty in Mbps, or `StreamingPolicy.action(state, capacity, uncertainty)` to return the next bitrate in Mbps.

Pass custom instances to `world.engine(predictor=..., policy=...)` and call `rollout(**world.query(...))`. The engine retains the learned network transition and analytical playback process. Use the same sample seed when comparing configuration branches to share their network futures.

## Tests

```bash
python -m unittest discover -s tests -v
```
