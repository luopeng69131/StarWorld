# Method Walkthrough

StarWorld connects uncertain access evolution to prediction and streaming outcomes through two components: **Cross-Layer Imagination** and a **Trust Gate**. Offline learning and calibration prepare these components for queries.

## Query and Output

A query supplies an observed network history, the current prediction and playback states, configuration/context features, a rollout horizon, and a sample count. One step corresponds to 10 seconds.

The result contains sampled network paths, executable prediction outputs, bitrate actions, playback-state paths, outcome summaries, and reliability decisions. Configuration branches share the same sampled network futures, so their paired differences follow from their application behavior.

## Cross-Layer Imagination

1. **Sample network evolution.** A recurrent stochastic model encodes network observations and exogenous context. A prior supplies latent samples for imagination; a next-observation posterior supports training. Network dynamics are independent of application actions.
2. **Execute prediction and control.** A predictor reads causal throughput history. A streaming policy maps the current playback state and capacity estimate to a bitrate decision. The interfaces support custom implementations.
3. **Advance playback.** Fixed-time chunk equations account for downloading, buffer consumption, and rebuffering. In-flight downloads carry over between intervals. A learned mean correction adapts the analytical transition to replay behavior, followed by physical state projection.
4. **Compare branches.** Reusing the same sample seed gives candidate policies identical exogenous network futures. The API checks this invariant before returning a comparison.

Core paths:

```text
api.StarWorld.imagine
    -> rollout.engine.HybridRolloutEngine
        -> models.world_model.StochasticNetworkDynamics
        -> interfaces.predictors.PredictionInterface
        -> interfaces.policies.StreamingPolicy
        -> interfaces.streaming.transition_streaming_interval
        -> learned playback correction
        -> rollout.trust.TrustGate
```

## Learning

`training/trainer.py` fits network and cross-layer state objectives with multi-step supervision. Network distributions receive lower-tail-sensitive losses, while replay availability masks identify valid application supervision. Training normalization and support statistics are fitted to the training partition.

`training/playback.py` refines the existing playback backbone and residual-mean head against multi-step buffer and cumulative-rebuffer errors. Upstream components remain fixed during this stage. Its differentiable playback equations are implemented in `interfaces/torch_streaming.py` and tested against the scalar executable transition. Validation data select both initial and refined checkpoints.

Playback inference uses the conditional mean correction with a default scale of 0.5. Network futures remain stochastic. Buffer, bitrate, and download time remain nonnegative; playback progress stays in range; cumulative rebuffering is nondecreasing.

## Trust Gate

The gate evaluates uncertainty, joint data support, and requested horizon:

- **Uncertainty:** validation rollouts calibrate horizon-specific thresholds against realized network-state error. Calibration uses the same rollout sampling and spread calculation as imagination.
- **Support:** training episodes define a Mahalanobis support region over network state, configuration, and action features.
- **Horizon:** the longest calibrated horizon bounds the supported rollout length.

Calibration follows recorded validation actions step by step. At query time, executable policies choose their actions from imagined causal states. Gate decisions are recorded for every sampled step, and a branch-level result is accepted when its complete sampled rollout passes the checks. The summary retains the reasons for abstention.

## Data Flow

The shared `Episode` schema in `schemas.py` separates network, prediction, streaming, action, configuration, context, and target arrays. All segments and controller branches from a collection run share one data partition.

The [data guide](data.md) describes the source format and alignment rules. The [usage guide](usage.md) provides the runnable training and imagination commands.
