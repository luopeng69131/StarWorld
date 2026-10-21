# Maintenance Guide

## Project Identity

- Repository: `https://github.com/luopeng69131/StarWorld`.
- Package: `starworld`.
- Purpose: a compact, runnable method implementation with bilingual introduction, data preparation, and a CPU example.
- Core components: Cross-Layer Imagination, Trust Gate, and offline learning/calibration.

## Code Map

| Area | File or directory |
| --- | --- |
| Workflow and selected-model manifest | `src/starworld/api.py` |
| CLI | `src/starworld/cli.py` |
| Runtime synthetic example | `src/starworld/demo.py`, `examples/minimal_demo.py` |
| Observation/target schema | `src/starworld/schemas.py` |
| StarNet import and causal windows | `src/starworld/data/prepare.py` |
| Playback alignment and episode construction | `src/starworld/data/episodes.py`, `replay_logs.py` |
| Ensemble training and multi-step losses | `src/starworld/training/` |
| Differentiable playback refinement | `training/playback.py`, `interfaces/torch_streaming.py` |
| Query-time future propagation | `src/starworld/rollout/engine.py` |
| Validation-calibrated acceptance | `src/starworld/rollout/trust.py` |

The code is organized from the StarWorld research implementation. The public workflow makes paths configurable, exposes a compact API, uses conditional-mean playback corrections with physical projection, and calibrates uncertainty through the same engine used for imagination.

## Design Invariants

1. Network evolution uses observed network history and exogenous context; application actions affect downstream outcomes.
2. Future measurements remain targets until their observation window has completed.
3. All branches and windows from a collection run share one split.
4. Normalization and support statistics use training episodes; checkpoint selection and trust calibration use validation episodes.
5. Candidate policy branches reuse indexed network futures.
6. Cumulative rebuffering is nondecreasing, and in-flight downloads persist between intervals.
7. Trust decisions include uncertainty, joint support, and calibrated horizon.

## Verification

```bash
python -m pip install -e '.[viz]'
python -m unittest discover -s tests -v
starworld demo --epochs 1 --members 1 --plot --output outputs/maintenance-demo
```

Check Markdown links, matching English/Chinese README claims, image paths, runtime summaries, and the generated `imagination.png`. Use fresh model output directories for successive checks. Source data and runtime outputs belong in the ignored `data/` and `outputs/` directories.

## Repository Presentation

Keep the introduction focused on the background, the cross-layer imagination environment, its core components, and a short runnable example. Describe implemented capabilities directly. Keep data-source attribution visible in both READMEs and the data guide. Keep this work's publication-reference section reserved for a later update.

The repository is initially private. Visibility changes are coordinated with the owner at release time.
