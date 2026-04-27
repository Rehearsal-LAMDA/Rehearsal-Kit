"""Bounded action optimizers for CME rehearsal."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rehearsal.metrics.cme import rbf_kernel


@dataclass(frozen=True)
class CMEOptimizationResult:
    """Decision-stage optimizer output for one CME candidate action set."""

    action: np.ndarray
    objective_value: float
    estimated_success_probability: float
    solver_status: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


def cme_action_kernel(
    action_history: np.ndarray,
    action: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    """Return ``k(A_i, action)`` for every historical action row."""

    history = _as_2d(action_history)
    point = np.asarray(action, dtype=float).reshape(1, -1)
    if history.shape[1] != point.shape[1]:
        raise ValueError("action dimension does not match action history.")
    return rbf_kernel(history, point, bandwidth).reshape(-1)


def cme_action_objective(
    action: np.ndarray,
    action_history: np.ndarray,
    omega: np.ndarray,
    bandwidth: float,
) -> float:
    """Legacy CME action objective ``sum_i omega_i k(A_i, a)``."""

    kernel = cme_action_kernel(action_history, action, bandwidth)
    return float(np.dot(np.asarray(omega, dtype=float).reshape(-1), kernel))


def optimize_action_projected_gradient(
    action_history: np.ndarray,
    omega: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    bandwidth: float,
    *,
    learning_rate: float = 0.1,
    max_steps: int = 100,
    num_restarts: int = 10,
    tolerance: float = 1e-6,
    positive_weight_threshold: float = 1e-9,
    rng: np.random.Generator | None = None,
) -> CMEOptimizationResult:
    """Maximize the CME action objective under box bounds by projected gradient."""

    history = _as_2d(action_history)
    weights = np.asarray(omega, dtype=float).reshape(-1)
    lo = np.asarray(lower, dtype=float).reshape(-1)
    hi = np.asarray(upper, dtype=float).reshape(-1)
    sigma = float(bandwidth)

    if history.shape[0] != weights.size:
        raise ValueError("omega length must match action history rows.")
    if history.shape[1] != lo.size or lo.shape != hi.shape:
        raise ValueError("bounds must match action dimension.")
    if np.any(lo > hi):
        raise ValueError("lower bounds must not exceed upper bounds.")
    if sigma <= 0.0 or not np.isfinite(sigma):
        raise ValueError("bandwidth must be finite and positive.")

    if rng is None:
        rng = np.random.default_rng()

    starts = _initial_points(
        history,
        weights,
        lo,
        hi,
        num_restarts=max(1, int(num_restarts)),
        positive_weight_threshold=positive_weight_threshold,
        rng=rng,
    )
    best_action = starts[0]
    best_value = -np.inf
    total_iterations = 0
    sigma_squared = max(sigma * sigma, 1e-12)

    for start in starts:
        current = np.clip(np.asarray(start, dtype=float).reshape(-1), lo, hi)
        for iteration in range(1, int(max_steps) + 1):
            total_iterations += 1
            grad = _objective_gradient(current, history, weights, sigma_squared)
            next_action = np.clip(current + float(learning_rate) * grad, lo, hi)
            if np.linalg.norm(next_action - current) < tolerance:
                current = next_action
                break
            current = next_action
        value = cme_action_objective(current, history, weights, sigma)
        if value > best_value:
            best_value = value
            best_action = current.copy()

    estimate = _normalized_success_estimate(best_action, history, weights, sigma)
    positive_count = int(np.sum(weights > positive_weight_threshold))
    return CMEOptimizationResult(
        action=np.clip(best_action, lo, hi),
        objective_value=float(best_value),
        estimated_success_probability=estimate,
        solver_status="projected_gradient",
        diagnostics={
            "iterations": int(total_iterations),
            "n_restarts": int(len(starts)),
            "positive_weight_count": positive_count,
            "bandwidth": sigma,
        },
    )


def _objective_gradient(
    action: np.ndarray,
    history: np.ndarray,
    weights: np.ndarray,
    sigma_squared: float,
) -> np.ndarray:
    diff = history - action.reshape(1, -1)
    squared = np.sum(diff * diff, axis=1)
    kernel = np.exp(-squared / (2.0 * sigma_squared))
    effective_weights = weights * kernel
    weighted_center = effective_weights @ history
    return (weighted_center - float(np.sum(effective_weights)) * action) / sigma_squared


def _initial_points(
    history: np.ndarray,
    weights: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    num_restarts: int,
    positive_weight_threshold: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    positive = np.flatnonzero(weights > positive_weight_threshold)
    starts: list[np.ndarray] = []
    if positive.size:
        ordered = positive[np.argsort(weights[positive])[::-1]]
        for idx in ordered[:num_restarts]:
            starts.append(np.clip(history[int(idx)], lower, upper))
    else:
        random_starts = rng.uniform(lower, upper, size=(num_restarts, lower.size))
        starts.extend(np.asarray(row, dtype=float) for row in random_starts)

    starts.append(np.clip(np.zeros_like(lower), lower, upper))
    starts.append(0.5 * (lower + upper))
    starts.append(lower.copy())
    starts.append(upper.copy())
    return _deduplicate_points(starts)


def _deduplicate_points(points: list[np.ndarray]) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for point in points:
        if not any(np.allclose(point, existing, atol=1e-12, rtol=0.0) for existing in unique):
            unique.append(point)
    return unique


def _normalized_success_estimate(
    action: np.ndarray,
    history: np.ndarray,
    weights: np.ndarray,
    bandwidth: float,
) -> float:
    kernel = cme_action_kernel(history, action, bandwidth)
    positive_mass = float(np.dot(np.maximum(weights, 0.0), kernel))
    total_mass = float(np.dot(np.abs(weights), kernel))
    if total_mass <= 1e-15:
        return 0.0
    return float(np.clip(positive_mass / total_mass, 0.0, 1.0))


def _as_2d(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        return matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        raise ValueError("values must be one- or two-dimensional.")
    return matrix
