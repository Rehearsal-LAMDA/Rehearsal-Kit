"""Bounded alteration optimizer used by Grad-Rh rehearsal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from rehearsal.core import AUFTask, DesiredRegion
from rehearsal.models.nonlinear import NonlinearStructuralModel


@dataclass(frozen=True)
class GradRhOptimizationResult:
    action: np.ndarray
    objective_value: float
    estimated_success_probability: float
    solver_status: str
    diagnostics: Mapping[str, Any]


def optimize_grad_rh_alterations(
    model: NonlinearStructuralModel,
    task: AUFTask,
    observation: Mapping[str, float],
    candidate: Sequence[str],
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    n_samples: int = 256,
    learning_rate: float = 0.05,
    epochs: int = 100,
    patience: int = 20,
    loss: str = "center_mae",
    num_restarts: int = 3,
    finite_difference_eps: float = 1e-4,
    rng: np.random.Generator | None = None,
) -> GradRhOptimizationResult:
    if rng is None:
        rng = np.random.default_rng()
    candidate = tuple(candidate)
    lower = np.asarray(lower, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)
    if lower.shape != upper.shape or lower.size != len(candidate):
        raise ValueError("Bounds must match candidate length.")
    if np.any(lower > upper):
        raise ValueError("Lower alteration bounds must not exceed upper bounds.")
    n_samples = int(n_samples)
    center, radius = desired_region_center_and_radius(task.desired_region)
    noise_bank = model.make_noise_bank(n_samples, rng=rng)

    def action_from_raw(raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=float).reshape(-1)
        return 0.5 * (np.tanh(raw) + 1.0) * (upper - lower) + lower

    def objective_for_action(action: np.ndarray) -> tuple[float, float]:
        alterations = {name: float(value) for name, value in zip(candidate, action)}
        outcomes = model.simulate_outcomes(
            task,
            observation,
            alterations,
            n_samples,
            rng=rng,
            noise_bank=noise_bank,
        )
        obj = _loss_value(outcomes, center, radius, loss)
        prob = float(np.mean(task.desired_region.contains(outcomes)))
        return obj, prob

    best: dict[str, Any] | None = None
    restarts = [np.zeros(len(candidate), dtype=float)]
    for _ in range(max(0, int(num_restarts) - 1)):
        restarts.append(rng.normal(scale=0.25, size=len(candidate)))

    for restart_idx, raw0 in enumerate(restarts):
        raw = np.asarray(raw0, dtype=float).copy()
        m = np.zeros_like(raw)
        v = np.zeros_like(raw)
        best_local: dict[str, Any] | None = None
        stale = 0
        for epoch in range(int(epochs)):
            action = action_from_raw(raw)
            obj, prob = objective_for_action(action)
            if best_local is None or obj < best_local["objective_value"]:
                best_local = {
                    "raw": raw.copy(),
                    "action": action.copy(),
                    "objective_value": float(obj),
                    "probability": float(prob),
                    "epoch": epoch,
                    "restart": restart_idx,
                }
                stale = 0
            else:
                stale += 1
            if best is None or _better(obj, prob, action, best):
                best = {
                    "raw": raw.copy(),
                    "action": action.copy(),
                    "objective_value": float(obj),
                    "probability": float(prob),
                    "epoch": epoch,
                    "restart": restart_idx,
                }
            if stale >= int(patience):
                break
            grad = np.zeros_like(raw)
            eps = float(finite_difference_eps)
            for j in range(raw.size):
                plus = raw.copy(); plus[j] += eps
                minus = raw.copy(); minus[j] -= eps
                plus_obj, _ = objective_for_action(action_from_raw(plus))
                minus_obj, _ = objective_for_action(action_from_raw(minus))
                grad[j] = (plus_obj - minus_obj) / (2.0 * eps)
            t = epoch + 1
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * (grad * grad)
            mhat = m / (1.0 - 0.9**t)
            vhat = v / (1.0 - 0.999**t)
            raw = raw - float(learning_rate) * mhat / (np.sqrt(vhat) + 1e-8)
            raw = np.clip(raw, -10.0, 10.0)
        if best_local is not None and (best is None or _better(best_local["objective_value"], best_local["probability"], best_local["action"], best)):
            best = best_local

    if best is None:
        raise RuntimeError("Grad-Rh optimizer failed to evaluate any candidate action.")
    return GradRhOptimizationResult(
        action=np.asarray(best["action"], dtype=float),
        objective_value=float(best["objective_value"]),
        estimated_success_probability=float(np.clip(best["probability"], 0.0, 1.0)),
        solver_status="finite_difference_adam",
        diagnostics={
            "loss": loss,
            "n_samples": int(n_samples),
            "epochs": int(epochs),
            "patience": int(patience),
            "num_restarts": int(num_restarts),
            "best_epoch": int(best["epoch"]),
            "best_restart": int(best["restart"]),
            "center": center.tolist(),
            "radius": float(radius),
        },
    )


def desired_region_center_and_radius(region: DesiredRegion) -> tuple[np.ndarray, float]:
    interval = _axis_aligned_intervals(region)
    if interval is not None:
        lower, upper = interval[:, 0], interval[:, 1]
        center = 0.5 * (lower + upper)
        radius = float(max(0.0, 0.5 * np.min(upper - lower)))
        return center, radius
    matrix = region.matrix
    vector = region.vector
    center = np.linalg.pinv(matrix) @ (0.5 * vector)
    slack = vector - matrix @ center
    norms = np.sqrt(np.sum(matrix * matrix, axis=1))
    valid = norms > 1e-12
    radius = float(max(0.0, np.min(slack[valid] / norms[valid]))) if np.any(valid) else 0.0
    return np.asarray(center, dtype=float).reshape(-1), radius


def _axis_aligned_intervals(region: DesiredRegion) -> np.ndarray | None:
    matrix = region.matrix
    vector = region.vector
    dim = matrix.shape[1]
    lower = np.full(dim, -np.inf)
    upper = np.full(dim, np.inf)
    for row, bound in zip(matrix, vector):
        nz = np.flatnonzero(np.abs(row) > 1e-12)
        if len(nz) != 1:
            return None
        idx = int(nz[0])
        coef = float(row[idx])
        if coef > 0:
            upper[idx] = min(upper[idx], float(bound) / coef)
        else:
            lower[idx] = max(lower[idx], float(bound) / coef)
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        return None
    return np.column_stack([lower, upper])


def _loss_value(outcomes: np.ndarray, center: np.ndarray, radius: float, loss: str) -> float:
    y = np.asarray(outcomes, dtype=float)
    center = np.asarray(center, dtype=float).reshape(1, -1)
    diff = y - center
    dist = np.sqrt(np.sum(diff * diff, axis=1))
    if loss == "center_mae":
        return float(np.mean(np.sum(np.abs(diff), axis=1)))
    if loss == "ball_huber":
        r = float(max(radius, 1e-12))
        return float(np.mean(np.where(dist <= r, dist * dist, 2.0 * dist * r - r * r)))
    if loss == "ball_insensitive":
        return float(np.mean(np.maximum(dist - float(radius), 0.0)))
    return float(np.mean(np.sum(diff * diff, axis=1)))


def _better(obj: float, prob: float, action: np.ndarray, best: Mapping[str, Any]) -> bool:
    if prob > float(best["probability"]) + 1e-12:
        return True
    if abs(prob - float(best["probability"])) <= 1e-12 and obj < float(best["objective_value"]) - 1e-12:
        return True
    if abs(prob - float(best["probability"])) <= 1e-12 and abs(obj - float(best["objective_value"])) <= 1e-12:
        return float(np.sum(np.abs(action))) < float(np.sum(np.abs(best["action"])))
    return False
