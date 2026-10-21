from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrustDecision:
    accepted: bool
    confidence: float
    supported_horizon: int
    reason: str
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    predicted_error: float = float("nan")
    state_distance: float | None = None


class TrustGate:
    """Validation-calibrated selective gate for imagined rollouts."""

    def __init__(self, target_coverage: float = 0.8):
        if not 0 < target_coverage <= 1:
            raise ValueError("target coverage must lie in (0, 1]")
        self.target_coverage = target_coverage
        self.thresholds: dict[int, float] = {}
        self.error_limits: dict[int, float] = {}
        self.error_models: dict[int, tuple[float, float]] = {}
        self.calibrated_coverage: dict[int, float] = {}
        self.ood_threshold: float | None = None
        self.support_center: np.ndarray | None = None
        self.support_precision: np.ndarray | None = None

    def set_support(
        self,
        center: np.ndarray,
        precision: np.ndarray,
        radius: float,
    ) -> None:
        center = np.asarray(center, dtype=float)
        precision = np.asarray(precision, dtype=float)
        if center.ndim != 1 or precision.shape != (center.size, center.size):
            raise ValueError("support center/precision dimensions do not match")
        if not np.isfinite(center).all() or not np.isfinite(precision).all():
            raise ValueError("support statistics contain non-finite values")
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("support radius must be finite and positive")
        self.support_center = center
        self.support_precision = precision
        self.ood_threshold = float(radius)

    def fit_support(
        self,
        states: np.ndarray,
        radius_quantile: float = 0.99,
    ) -> None:
        """Fit a Mahalanobis support set from training states."""
        values = np.asarray(states, dtype=float)
        if values.ndim != 2 or values.shape[0] < 2:
            raise ValueError("support states must have shape [samples, dimensions]")
        if not 0 < radius_quantile < 1:
            raise ValueError("radius_quantile must lie in (0, 1)")
        center = values.mean(axis=0)
        covariance = np.atleast_2d(np.cov(values - center, rowvar=False))
        regularizer = max(float(np.trace(covariance)) / max(values.shape[1], 1), 1.0) * 1e-5
        precision = np.linalg.pinv(
            covariance + regularizer * np.eye(values.shape[1], dtype=float)
        )
        centered = values - center
        distances = np.sqrt(
            np.maximum(np.einsum("ni,ij,nj->n", centered, precision, centered), 0.0)
        )
        self.set_support(
            center,
            precision,
            max(float(np.quantile(distances, radius_quantile)), 1e-6),
        )

    def state_distance(self, state: np.ndarray) -> float | np.ndarray:
        if self.support_center is None or self.support_precision is None:
            raise RuntimeError("training-support statistics have not been fitted")
        values = np.asarray(state, dtype=float)
        centered = values - self.support_center
        if centered.ndim == 1:
            return float(
                np.sqrt(
                    max(
                        float(centered @ self.support_precision @ centered),
                        0.0,
                    )
                )
            )
        if centered.ndim != 2:
            raise ValueError("state must have shape [dimensions] or [samples, dimensions]")
        return np.sqrt(
            np.maximum(
                np.einsum(
                    "ni,ij,nj->n",
                    centered,
                    self.support_precision,
                    centered,
                ),
                0.0,
            )
        )

    def calibrate(
        self,
        horizon: np.ndarray,
        uncertainty: np.ndarray,
        error: np.ndarray,
        state_distance: np.ndarray | None = None,
        support_states: np.ndarray | None = None,
    ) -> None:
        horizon = np.asarray(horizon, dtype=int)
        uncertainty = np.asarray(uncertainty, dtype=float)
        error = np.asarray(error, dtype=float)
        if not (
            horizon.ndim == uncertainty.ndim == error.ndim == 1
            and horizon.size == uncertainty.size == error.size
        ):
            raise ValueError("horizon, uncertainty, and error must be aligned vectors")
        if not np.isfinite(uncertainty).all() or not np.isfinite(error).all():
            raise ValueError("calibration values contain non-finite values")
        if support_states is not None:
            self.fit_support(support_states)
        for value in sorted(set(horizon.tolist())):
            mask = horizon == value
            local_uncertainty = uncertainty[mask]
            local_error = error[mask]
            order = np.argsort(local_uncertainty, kind="stable")
            sorted_uncertainty = local_uncertainty[order]
            sorted_error = local_error[order]
            count = sorted_error.size
            maximum_count = max(1, int(np.floor(self.target_coverage * count)))
            minimum_count = max(1, int(np.floor(0.5 * self.target_coverage * count)))
            error_budget = float(
                np.quantile(local_error, max(self.target_coverage, 0.5))
            )
            selected_count = minimum_count
            selected_upper_bound = float("inf")
            for prefix_count in range(minimum_count, maximum_count + 1):
                accepted_error = sorted_error[:prefix_count]
                standard_error = float(accepted_error.std(ddof=0)) / np.sqrt(prefix_count)
                upper_bound = float(accepted_error.mean() + 1.96 * standard_error)
                if upper_bound <= error_budget:
                    selected_count = prefix_count
                    selected_upper_bound = upper_bound
            if not np.isfinite(selected_upper_bound):
                candidates: list[tuple[float, int]] = []
                for prefix_count in range(minimum_count, maximum_count + 1):
                    accepted_error = sorted_error[:prefix_count]
                    standard_error = float(accepted_error.std(ddof=0)) / np.sqrt(prefix_count)
                    candidates.append(
                        (
                            float(accepted_error.mean() + 1.96 * standard_error),
                            prefix_count,
                        )
                    )
                selected_upper_bound, selected_count = min(candidates)
                error_budget = max(error_budget, selected_upper_bound)
            threshold = float(sorted_uncertainty[selected_count - 1])
            variance = float(np.var(local_uncertainty))
            if variance > 1e-12:
                slope = max(
                    float(
                        np.cov(local_uncertainty, local_error, ddof=0)[0, 1]
                        / variance
                    ),
                    0.0,
                )
            else:
                slope = 0.0
            intercept = max(
                float(local_error.mean() - slope * local_uncertainty.mean()),
                0.0,
            )
            accepted_uncertainty = local_uncertainty <= threshold
            accepted_predicted_error = (
                intercept + slope * local_uncertainty[accepted_uncertainty]
            )
            if accepted_predicted_error.size:
                error_budget = max(
                    error_budget,
                    float(np.max(accepted_predicted_error)),
                )
            horizon_value = int(value)
            self.thresholds[horizon_value] = threshold
            self.error_limits[horizon_value] = error_budget
            self.error_models[horizon_value] = (intercept, slope)
            self.calibrated_coverage[horizon_value] = selected_count / count
        # Backward compatibility for callers that already computed distances.
        # A fitted training support set is preferred and is never overwritten.
        if state_distance is not None and self.support_center is None:
            distances = np.asarray(state_distance, dtype=float)
            if not np.isfinite(distances).all():
                raise ValueError("state distances contain non-finite values")
            self.ood_threshold = float(np.quantile(distances, 0.99))

    def decide(
        self,
        horizon: int,
        epistemic: float,
        aleatoric: float,
        state_distance: float | None = None,
        state: np.ndarray | None = None,
    ) -> TrustDecision:
        if not self.thresholds:
            raise RuntimeError("TrustGate must be calibrated on validation data")
        if horizon < 1 or not np.isfinite([epistemic, aleatoric]).all():
            raise ValueError("horizon must be positive and uncertainty must be finite")
        if epistemic < 0 or aleatoric < 0:
            raise ValueError("uncertainty must be nonnegative")
        known = sorted(self.thresholds)
        if horizon > known[-1]:
            return TrustDecision(False, 0.0, known[-1], "horizon_exceeds_calibration",
                                 epistemic, aleatoric)
        applicable = max((value for value in known if value <= horizon), default=known[0])
        total = float(np.hypot(epistemic, aleatoric))
        threshold = self.thresholds[applicable]
        if state is not None:
            if state_distance is not None:
                raise ValueError("provide state or state_distance, not both")
            state_distance = float(self.state_distance(state))
        intercept, slope = self.error_models.get(applicable, (0.0, 0.0))
        predicted_error = float(intercept + slope * total)
        error_limit = self.error_limits.get(applicable, float("inf"))
        if (
            state_distance is not None
            and self.ood_threshold is not None
            and state_distance > self.ood_threshold
        ):
            return TrustDecision(
                False,
                0.0,
                applicable,
                "out_of_distribution",
                epistemic,
                aleatoric,
                predicted_error,
                state_distance,
            )
        uncertainty_supported = total <= threshold
        error_supported = predicted_error <= error_limit
        accepted = uncertainty_supported and error_supported
        uncertainty_confidence = np.clip(
            1.0 - total / max(threshold, 1e-8),
            0.0,
            1.0,
        )
        error_confidence = np.clip(
            1.0 - predicted_error / max(error_limit, 1e-8),
            0.0,
            1.0,
        )
        confidence = float(min(uncertainty_confidence, error_confidence))
        reason = "supported"
        if not uncertainty_supported:
            reason = "uncertainty_above_validation_threshold"
        elif not error_supported:
            reason = "predicted_error_above_validation_limit"
        return TrustDecision(
            accepted=accepted,
            confidence=confidence,
            supported_horizon=applicable,
            reason=reason,
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            predicted_error=predicted_error,
            state_distance=state_distance,
        )
