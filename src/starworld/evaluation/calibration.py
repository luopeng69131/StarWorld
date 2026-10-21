"""Persist validation-calibrated Trust Gate state."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from starworld.rollout.trust import TrustGate


def save_trust_calibration(
    gate: TrustGate,
    diagnostics: dict[str, object],
    path: str | Path,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "target_coverage": gate.target_coverage,
        "thresholds": gate.thresholds,
        "error_limits": gate.error_limits,
        "error_models": {
            str(horizon): {"intercept": values[0], "slope": values[1]}
            for horizon, values in gate.error_models.items()
        },
        "calibrated_coverage": gate.calibrated_coverage,
        "ood_threshold": gate.ood_threshold,
        "support_center": (
            None if gate.support_center is None else gate.support_center.tolist()
        ),
        "support_precision": (
            None if gate.support_precision is None else gate.support_precision.tolist()
        ),
        "diagnostics": diagnostics,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_trust_calibration(path: str | Path) -> TrustGate:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    gate = TrustGate(float(payload["target_coverage"]))
    gate.thresholds = {
        int(key): float(value)
        for key, value in payload["thresholds"].items()
    }
    gate.error_limits = {
        int(key): float(value)
        for key, value in payload.get("error_limits", {}).items()
    }
    gate.error_models = {
        int(key): (
            float(value["intercept"]),
            float(value["slope"]),
        )
        for key, value in payload.get("error_models", {}).items()
    }
    gate.calibrated_coverage = {
        int(key): float(value)
        for key, value in payload.get("calibrated_coverage", {}).items()
    }
    center = payload.get("support_center")
    precision = payload.get("support_precision")
    radius = payload.get("ood_threshold")
    if center is not None and precision is not None and radius is not None:
        gate.set_support(
            np.asarray(center, dtype=float),
            np.asarray(precision, dtype=float),
            float(radius),
        )
    return gate
