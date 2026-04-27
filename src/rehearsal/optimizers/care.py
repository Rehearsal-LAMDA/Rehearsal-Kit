"""CARE rehearsal optimizers over fitted structural models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rehearsal.metrics.care import independent_normal_care_success, normal_cdf, normal_pdf


@dataclass(frozen=True)
class CAREOptimizationResult:
    """Decision-stage optimizer output for one CARE candidate set."""

    z: np.ndarray
    care_success: float
    objective_value: float
    solver_status: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


def optimize_care_independent_normal(
    base_mean: np.ndarray,
    effects: np.ndarray,
    variance: np.ndarray,
    intervals: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    max_iters: int = 200,
    tolerance: float = 1e-8,
) -> CAREOptimizationResult:
    """Optimize CARE success for independent Gaussian outcomes."""

    base_mean = np.asarray(base_mean, dtype=float).reshape(-1)
    effects = np.asarray(effects, dtype=float)
    variance = np.asarray(variance, dtype=float).reshape(-1)
    intervals = np.asarray(intervals, dtype=float)
    lower = np.asarray(lower, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)

    if effects.shape != (base_mean.size, lower.size):
        raise ValueError("effects must have shape (n_outcomes, n_alterations).")
    if variance.shape != base_mean.shape:
        raise ValueError("variance must match base_mean shape.")
    if intervals.shape != (base_mean.size, 2):
        raise ValueError("intervals must have shape (n_outcomes, 2).")
    if lower.shape != upper.shape:
        raise ValueError("lower and upper bounds must have the same shape.")

    if base_mean.size == 1:
        z, status, solver_meta = _solve_single_outcome_closed_form(
            float(base_mean[0]),
            effects.reshape(1, -1)[0],
            intervals[0],
            lower,
            upper,
        )
    else:
        z, status, solver_meta = _solve_independent_projected_gradient(
            base_mean,
            effects,
            variance,
            intervals,
            lower,
            upper,
            max_iters=max_iters,
            tolerance=tolerance,
        )

    mean = base_mean + effects @ z
    care_success = independent_normal_care_success(mean, variance, intervals)
    objective = -np.log(max(care_success, 1e-300))
    return CAREOptimizationResult(
        z=np.clip(z, lower, upper),
        care_success=float(np.clip(care_success, 0.0, 1.0)),
        objective_value=float(objective),
        solver_status=status,
        diagnostics=solver_meta,
    )


def _solve_single_outcome_closed_form(
    base_mean: float,
    effect: np.ndarray,
    interval: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    center = 0.5 * (float(interval[0]) + float(interval[1]))
    target = center - float(base_mean)
    z, target_dot = _bounded_linear_target(effect, target, lower, upper)
    return z, "closed_form_1d", {"target_mean": center, "target_dot": float(target_dot)}


def _bounded_linear_target(
    effect: np.ndarray,
    target: float,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, float]:
    effect = np.asarray(effect, dtype=float).reshape(-1)
    lower = np.asarray(lower, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)
    if effect.size == 0:
        return np.asarray([], dtype=float), 0.0
    if np.all(np.abs(effect) <= 1e-14):
        return np.clip(np.zeros_like(lower), lower, upper), 0.0
    z = np.where(effect >= 0.0, lower, upper).astype(float)
    z = np.where(np.abs(effect) <= 1e-14, np.clip(0.0, lower, upper), z)
    min_dot = float(effect @ z)
    max_z = np.where(effect >= 0.0, upper, lower).astype(float)
    max_dot = float(effect @ max_z)
    target_dot = float(np.clip(target, min_dot, max_dot))
    residual = target_dot - min_dot
    for idx, coef in enumerate(effect):
        if abs(coef) <= 1e-14 or residual <= 1e-12:
            continue
        if coef > 0:
            step = min(upper[idx] - z[idx], residual / coef)
            z[idx] += step
            residual -= coef * step
        else:
            step = min(z[idx] - lower[idx], residual / (-coef))
            z[idx] -= step
            residual -= (-coef) * step
    return np.clip(z, lower, upper), target_dot


def _solve_independent_projected_gradient(
    base_mean: np.ndarray,
    effects: np.ndarray,
    variance: np.ndarray,
    intervals: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    max_iters: int,
    tolerance: float,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    if effects.shape[1] == 0:
        return np.asarray([], dtype=float), "no_alterable_variables", {"iterations": 0}

    center = 0.5 * (intervals[:, 0] + intervals[:, 1])
    scale = 1.0 / np.sqrt(np.maximum(variance, 1e-12))
    weighted_effects = effects * scale[:, None]
    weighted_target = (center - base_mean) * scale
    center_aligned = np.linalg.pinv(weighted_effects) @ weighted_target
    initial_points = [
        np.clip(np.zeros_like(lower), lower, upper),
        np.clip(center_aligned, lower, upper),
        lower.copy(),
        upper.copy(),
        0.5 * (lower + upper),
    ]
    best_z = max(initial_points, key=lambda z: _log_care_success(base_mean + effects @ z, variance, intervals))
    best_value = _log_care_success(base_mean + effects @ best_z, variance, intervals)
    step = 1.0
    iterations = 0

    for iterations in range(1, max_iters + 1):
        mean = base_mean + effects @ best_z
        grad_mean = _log_care_success_mean_gradient(mean, variance, intervals)
        grad = effects.T @ grad_mean
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm < tolerance:
            break
        accepted = False
        local_step = step
        for _ in range(30):
            candidate = np.clip(best_z + local_step * grad, lower, upper)
            value = _log_care_success(base_mean + effects @ candidate, variance, intervals)
            if value >= best_value - 1e-15:
                accepted = True
                if np.linalg.norm(candidate - best_z) < tolerance:
                    best_z = candidate
                    best_value = value
                    break
                best_z = candidate
                best_value = value
                step = min(local_step * 1.5, 10.0)
                break
            local_step *= 0.5
        if not accepted:
            break

    return (
        np.clip(best_z, lower, upper),
        "projected_gradient_independent",
        {"iterations": int(iterations), "log_care_success": float(best_value)},
    )


def _log_care_success(mean: np.ndarray, variance: np.ndarray, intervals: np.ndarray) -> float:
    mean = np.asarray(mean, dtype=float).reshape(-1)
    std = np.sqrt(np.maximum(np.asarray(variance, dtype=float).reshape(-1), 1e-12))
    lower_z = (intervals[:, 0] - mean) / std
    upper_z = (intervals[:, 1] - mean) / std
    per_dim_success = np.clip(normal_cdf(upper_z) - normal_cdf(lower_z), 1e-300, 1.0)
    return float(np.sum(np.log(per_dim_success)))


def _log_care_success_mean_gradient(
    mean: np.ndarray,
    variance: np.ndarray,
    intervals: np.ndarray,
) -> np.ndarray:
    std = np.sqrt(np.maximum(np.asarray(variance, dtype=float).reshape(-1), 1e-12))
    lower_z = (intervals[:, 0] - mean) / std
    upper_z = (intervals[:, 1] - mean) / std
    per_dim_success = np.clip(normal_cdf(upper_z) - normal_cdf(lower_z), 1e-300, 1.0)
    return (normal_pdf(lower_z) - normal_pdf(upper_z)) / (std * per_dim_success)
