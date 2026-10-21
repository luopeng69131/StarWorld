from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Mapping
from typing import Any


FORBIDDEN_INPUT_FIELDS = {
    "future_mean_mbps",
    "future_p05_mbps",
    "future_min_mbps",
    "drop_label",
    "low_capacity_label",
    "overestimation_label",
    "high_risk_label",
    "segment_abr_severe_session",
    "abr_severe_session",
    "is_replay_hard",
    "is_severe_rebuffer_proxy",
    "paper_profile_labels",
}

CAUSAL_AVAILABILITY = {
    "causal_measurement_history",
    "lagged_completed_measurement",
    "causal_prediction_output",
    "causal_replay_prefix",
    "causal_interval_decisions",
    "explicit_pre_episode_config",
    "exogenous_context",
}

FORBIDDEN_PROVENANCE = {
    "future_target",
    "full_replay_scan",
    "full_segment_summary",
    "post_episode_outcome",
    "paper_label",
}


def audit_feature_names(feature_names: Iterable[str]) -> dict[str, object]:
    names = set(feature_names)
    violations = sorted(names.intersection(FORBIDDEN_INPUT_FIELDS))
    return {
        "passed": not violations,
        "violations": violations,
        "checked_features": sorted(names),
    }


def audit_feature_provenance(
    feature_names: Iterable[str],
    provenance: Mapping[str, Mapping[str, Any]],
) -> dict[str, object]:
    """Audit when and how every model input becomes available.

    Name checks alone cannot detect a causal feature that was computed by
    scanning a complete replay.  Each feature therefore declares its
    availability class, source fields, and optional causal transform.
    Future-named measurement columns are permitted only when shifted until the
    source window has completed.
    """
    names = list(feature_names)
    missing = sorted(set(names).difference(provenance))
    unexpected = sorted(set(provenance).difference(names))
    violations: list[dict[str, str]] = []
    for name in names:
        record = provenance.get(name)
        if record is None:
            continue
        availability = str(record.get("availability", ""))
        transform = str(record.get("transform", ""))
        sources = [str(value) for value in record.get("sources", [])]
        if availability not in CAUSAL_AVAILABILITY:
            violations.append(
                {
                    "feature": name,
                    "reason": f"non-causal availability class: {availability or '<missing>'}",
                }
            )
        if availability in FORBIDDEN_PROVENANCE:
            violations.append({"feature": name, "reason": availability})
        forbidden_sources = sorted(set(sources).intersection(FORBIDDEN_INPUT_FIELDS))
        if forbidden_sources and not (
            availability == "lagged_completed_measurement"
            and transform == "shift_completed_window_by_one_step"
        ):
            violations.append(
                {
                    "feature": name,
                    "reason": f"future/outcome source without causal lag: {forbidden_sources}",
                }
            )
        if any(token in transform for token in ("full_replay", "global_max", "final_length")):
            violations.append(
                {
                    "feature": name,
                    "reason": f"full-trajectory transform is forbidden: {transform}",
                }
            )
    for name in missing:
        violations.append({"feature": name, "reason": "missing provenance record"})
    return {
        "passed": not violations and not unexpected,
        "violations": violations,
        "missing_features": missing,
        "unexpected_features": unexpected,
        "checked_features": names,
        "provenance": {name: dict(provenance[name]) for name in names if name in provenance},
    }


def require_causal_features(feature_names: Iterable[str]) -> None:
    report = audit_feature_names(feature_names)
    if not report["passed"]:
        raise ValueError(f"future or outcome leakage in model inputs: {report['violations']}")


def require_causal_provenance(
    feature_names: Iterable[str],
    provenance: Mapping[str, Mapping[str, Any]],
) -> dict[str, object]:
    report = audit_feature_provenance(feature_names, provenance)
    if not report["passed"]:
        raise ValueError(f"non-causal model-input provenance: {report['violations']}")
    return report
