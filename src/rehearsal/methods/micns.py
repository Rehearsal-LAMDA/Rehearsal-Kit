"""NeurIPS 2024 MICNS rehearsal adapter."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from rehearsal.core import AUFTask, DecisionResult, candidate_alteration_sets, coerce_data_matrix
from rehearsal.core.regions import desired_region_intervals_under_independence
from rehearsal.metrics.probability import independent_normal_success_probability
from rehearsal.models import LinearGaussianSRM


@dataclass(frozen=True)
class _CandidateEvaluation:
    candidate: tuple[str, ...]
    alterations: dict[str, float]
    cost: float
    estimated_success_probability: float
    objective_value: float
    mean: np.ndarray
    variance: np.ndarray
    solver_status: str
    diagnostics: dict[str, Any]


class MICNSRehearsal:
    """Time-varying linear SRM rehearsal with cost-aware candidate selection."""

    def __init__(
        self,
        *,
        seed: int | None = None,
        candidate_alteration_sets: Sequence[Sequence[str]] | None = None,
        time_bandwidth_fraction: float = 0.25,
        ridge: float = 1e-4,
        confidence_level: float = 0.7,
        cost_penalty: float = 0.1,
        baseline_alterations: Mapping[str, float] | None = None,
    ) -> None:
        self.seed = seed
        self.candidate_alteration_sets = (
            tuple(tuple(candidate) for candidate in candidate_alteration_sets)
            if candidate_alteration_sets is not None
            else None
        )
        self.time_bandwidth_fraction = float(time_bandwidth_fraction)
        self.ridge = float(ridge)
        self.confidence_level = float(confidence_level)
        self.cost_penalty = float(cost_penalty)
        self.baseline_alterations = {str(name): float(value) for name, value in dict(baseline_alterations or {}).items()}

        self.config_: dict[str, Any] = {}
        self.columns_: tuple[str, ...] | None = None
        self.column_index_: dict[str, int] = {}
        self.data_matrix_: np.ndarray | None = None
        self.model_: LinearGaussianSRM | None = None
        self.fit_diagnostics_: dict[str, Any] = {}
        self.last_decision_: DecisionResult | None = None
        self.last_context_: dict[str, Any] | None = None

    def fit(
        self,
        data: Mapping[str, Sequence[float]] | np.ndarray,
        task: AUFTask,
        config: Mapping[str, Any] | None = None,
    ) -> "MICNSRehearsal":
        config_dict = dict(config or {})
        self.config_ = config_dict
        matrix, columns = coerce_data_matrix(data, task, config_dict)
        self.data_matrix_ = matrix
        self.columns_ = columns
        self.column_index_ = {name: idx for idx, name in enumerate(columns)}

        bandwidth_fraction = float(config_dict.get("time_bandwidth_fraction", self.time_bandwidth_fraction))
        ridge = float(config_dict.get("ridge", self.ridge))
        min_variance = float(config_dict.get("min_variance", 1e-6))
        latest_theta, theta_trajectory, residuals = _fit_time_varying_theta(
            matrix,
            columns,
            task.parents,
            bandwidth_fraction=bandwidth_fraction,
            ridge=ridge,
        )
        covariance = np.cov(residuals, rowvar=False, bias=True)
        covariance = np.atleast_2d(np.asarray(covariance, dtype=float))
        covariance = covariance + min_variance * np.eye(len(columns))
        self.model_ = LinearGaussianSRM(columns, latest_theta, covariance)
        self.fit_diagnostics_ = {
            "n_training_samples": int(matrix.shape[0]),
            "columns": columns,
            "time_bandwidth_fraction": bandwidth_fraction,
            "ridge": ridge,
            "confidence_level": float(config_dict.get("confidence_level", self.confidence_level)),
            "cost_penalty": float(config_dict.get("cost_penalty", self.cost_penalty)),
            "time_varying_theta_latest": latest_theta,
            "time_varying_theta_trajectory": theta_trajectory,
        }
        return self

    def suggest(self, observation: Mapping[str, float], task: AUFTask) -> DecisionResult:
        if self.model_ is None:
            raise RuntimeError("MICNSRehearsal.fit must be called before suggest.")
        start = time.perf_counter()
        intervals = desired_region_intervals_under_independence(task.desired_region, task.outcomes)
        candidates = candidate_alteration_sets(task, self.candidate_alteration_sets)
        confidence_level = float(self.config_.get("confidence_level", self.confidence_level))
        cost_penalty = float(self.config_.get("cost_penalty", self.cost_penalty))

        evaluations: list[_CandidateEvaluation] = []
        best: _CandidateEvaluation | None = None
        for candidate in candidates:
            evaluation = self._evaluate_candidate(observation, task, candidate, intervals, confidence_level, cost_penalty)
            evaluations.append(evaluation)
            if best is None or evaluation.objective_value > best.objective_value + 1e-15:
                best = evaluation
            elif (
                best is not None
                and abs(evaluation.objective_value - best.objective_value) <= 1e-15
                and evaluation.cost < best.cost
            ):
                best = evaluation

        if best is None:
            raise ValueError("No candidate alteration set is available.")

        runtime = time.perf_counter() - start
        diagnostics = {
            "selected_candidate": best.candidate,
            "solver_status": best.solver_status,
            "objective_value": best.objective_value,
            "estimated_success_probability_raw": best.estimated_success_probability,
            "n_candidates": len(evaluations),
            "candidate_diagnostics": [
                {
                    "candidate": evaluation.candidate,
                    "alterations": evaluation.alterations,
                    "cost": evaluation.cost,
                    "estimated_success_probability": evaluation.estimated_success_probability,
                    "objective_value": evaluation.objective_value,
                    "solver_status": evaluation.solver_status,
                    **evaluation.diagnostics,
                }
                for evaluation in evaluations
            ],
            **self.fit_diagnostics_,
            **best.diagnostics,
        }
        result = DecisionResult(
            alterations=best.alterations,
            estimated_success_probability=float(np.clip(best.estimated_success_probability, 0.0, 1.0)),
            cost=float(best.cost),
            diagnostics=diagnostics,
            runtime_seconds=runtime,
        )
        self.last_decision_ = result
        self.last_context_ = {
            "mean": best.mean.copy(),
            "variance": best.variance.copy(),
            "candidate": best.candidate,
            "observation": dict(observation),
        }
        return result

    def evaluate(self, task: AUFTask, n_samples: int) -> Mapping[str, Any]:
        if self.last_context_ is None or self.last_decision_ is None:
            raise RuntimeError("suggest must be called before evaluate.")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")

        rng = np.random.default_rng(self.seed)
        mean = np.asarray(self.last_context_["mean"], dtype=float).reshape(-1)
        variance = np.asarray(self.last_context_["variance"], dtype=float).reshape(-1)
        samples = rng.normal(mean, np.sqrt(np.maximum(variance, 1e-12)), size=(int(n_samples), mean.size))
        success = np.asarray(task.desired_region.contains(samples), dtype=bool)
        return {
            "estimated_success_probability": float(self.last_decision_.estimated_success_probability),
            "empirical_success_rate": float(np.mean(success)),
            "n_samples": int(n_samples),
            "selected_candidate": list(self.last_context_["candidate"]),
            "alterations": dict(self.last_decision_.alterations),
        }

    def _evaluate_candidate(
        self,
        observation: Mapping[str, float],
        task: AUFTask,
        candidate: tuple[str, ...],
        intervals: np.ndarray,
        confidence_level: float,
        cost_penalty: float,
    ) -> _CandidateEvaluation:
        assert self.model_ is not None
        base_mean, covariance_y = self.model_.outcome_moments(task, observation, candidate)
        lower, upper = task.alteration_domain.arrays_for(candidate)
        effect_a, effect_b, effect_c = self.model_.effect_matrices(task, candidate)
        x = np.asarray([float(observation[name]) for name in task.observed], dtype=float).reshape(-1)
        base_mean = (effect_a @ x).reshape(-1)
        variance = np.maximum(np.diag(effect_c @ self.model_.covariance @ effect_c.T), 1e-8)
        robust_intervals = _robust_intervals(intervals, variance, confidence_level)
        baseline = np.asarray([self.baseline_alterations.get(name, 0.0) for name in candidate], dtype=float)
        z, solver_status, solver_meta = _solve_min_cost_target(
            effect_b,
            base_mean,
            robust_intervals,
            lower,
            upper,
            baseline,
            np.asarray([task.alteration_domain.costs.get(name, 1.0) for name in candidate], dtype=float),
        )
        mean = base_mean + effect_b @ z
        alterations = {name: float(value) for name, value in zip(candidate, z)}
        cost = task.alteration_domain.cost(alterations)
        estimated_success = independent_normal_success_probability(mean, variance, intervals)
        objective = float(estimated_success - cost_penalty * cost / max(1.0, len(candidate)))
        return _CandidateEvaluation(
            candidate=candidate,
            alterations=alterations,
            cost=float(cost),
            estimated_success_probability=float(estimated_success),
            objective_value=objective,
            mean=np.asarray(mean, dtype=float),
            variance=np.asarray(variance, dtype=float),
            solver_status=solver_status,
            diagnostics={
                "base_mean": base_mean.tolist(),
                "variance": variance.tolist(),
                "robust_intervals": robust_intervals.tolist(),
                "target_intervals": intervals.tolist(),
                "baseline_alterations": baseline.tolist(),
                **solver_meta,
            },
        )


def _fit_time_varying_theta(
    matrix: np.ndarray,
    columns: Sequence[str],
    parents: Mapping[str, Sequence[str]],
    *,
    bandwidth_fraction: float,
    ridge: float,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, list[float]]], np.ndarray]:
    n_samples, n_variables = matrix.shape
    times = np.arange(n_samples, dtype=float)
    center = float(n_samples - 1)
    bandwidth = max(1.0, float(bandwidth_fraction) * max(1, n_samples - 1))
    weights = np.exp(-0.5 * ((times - center) / bandwidth) ** 2)

    column_index = {name: idx for idx, name in enumerate(columns)}
    theta_latest: dict[str, dict[str, float]] = {name: {} for name in columns}
    theta_trajectory: dict[str, dict[str, list[float]]] = {name: {} for name in columns}
    residuals = np.zeros_like(matrix, dtype=float)

    checkpoints = np.linspace(0, n_samples - 1, num=min(5, n_samples), dtype=int)
    for child in columns:
        child_idx = column_index[child]
        y = matrix[:, child_idx]
        parent_names = tuple(name for name in parents.get(child, ()) if name in column_index)
        if not parent_names:
            residuals[:, child_idx] = y
            continue
        x = matrix[:, [column_index[name] for name in parent_names]]
        beta_latest = _weighted_least_squares(x, y, weights, ridge)
        prediction = x @ beta_latest
        residuals[:, child_idx] = y - prediction
        for parent_name, beta in zip(parent_names, beta_latest):
            theta_latest.setdefault(parent_name, {})[child] = float(beta)
        for checkpoint in checkpoints:
            local_weights = np.exp(-0.5 * ((times - float(checkpoint)) / bandwidth) ** 2)
            beta_checkpoint = _weighted_least_squares(x, y, local_weights, ridge)
            for parent_name, beta in zip(parent_names, beta_checkpoint):
                theta_trajectory.setdefault(parent_name, {}).setdefault(child, []).append(float(beta))
    return theta_latest, theta_trajectory, residuals


def _weighted_least_squares(x: np.ndarray, y: np.ndarray, weights: np.ndarray, ridge: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    weights = np.maximum(weights, 1e-12)
    sqrt_weights = np.sqrt(weights).reshape(-1, 1)
    xw = x * sqrt_weights
    yw = y * sqrt_weights
    system = xw.T @ xw + float(ridge) * np.eye(x.shape[1])
    rhs = xw.T @ yw
    return (np.linalg.pinv(system) @ rhs).reshape(-1)


def _robust_intervals(intervals: np.ndarray, variance: np.ndarray, confidence_level: float) -> np.ndarray:
    radius_scale = _normal_quantile(0.5 + 0.5 * float(np.clip(confidence_level, 1e-6, 1.0 - 1e-6)))
    radius = radius_scale * np.sqrt(np.maximum(np.asarray(variance, dtype=float), 1e-12))
    robust = np.column_stack([intervals[:, 0] + radius, intervals[:, 1] - radius])
    invalid = robust[:, 0] > robust[:, 1]
    robust[invalid, :] = intervals[invalid, :]
    return robust


def _solve_min_cost_target(
    effects: np.ndarray,
    base_mean: np.ndarray,
    intervals: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    baseline: np.ndarray,
    costs: np.ndarray,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    effects = np.asarray(effects, dtype=float)
    base_mean = np.asarray(base_mean, dtype=float).reshape(-1)
    intervals = np.asarray(intervals, dtype=float)
    lower = np.asarray(lower, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)
    baseline = np.asarray(baseline, dtype=float).reshape(-1)
    costs = np.maximum(np.asarray(costs, dtype=float).reshape(-1), 1e-8)

    target = np.clip(base_mean, intervals[:, 0], intervals[:, 1])
    if np.all((base_mean >= intervals[:, 0]) & (base_mean <= intervals[:, 1])):
        target = base_mean.copy()
    else:
        target = 0.5 * (intervals[:, 0] + intervals[:, 1])

    centered_target = target - base_mean
    if effects.shape[1] == 0:
        return np.asarray([], dtype=float), "no_alterable_variables", {"target_mean": target.tolist()}

    weight_inv = np.diag(1.0 / costs)
    gram = effects @ weight_inv @ effects.T
    if gram.size == 1 and abs(gram.item()) <= 1e-12:
        z = np.clip(baseline, lower, upper)
        return z, "degenerate_effects", {"target_mean": target.tolist()}
    delta = weight_inv @ effects.T @ np.linalg.pinv(gram) @ centered_target.reshape(-1, 1)
    z = np.clip(baseline + delta.reshape(-1), lower, upper)
    mean = base_mean + effects @ z
    if not np.all((mean >= intervals[:, 0]) & (mean <= intervals[:, 1])):
        # Fall back to interval endpoints if clipping broke feasibility.
        choices = [z, np.clip(lower, lower, upper), np.clip(upper, lower, upper), np.clip(baseline, lower, upper)]
        target_points = [intervals[:, 0], 0.5 * (intervals[:, 0] + intervals[:, 1]), intervals[:, 1]]
        for point in target_points:
            delta = weight_inv @ effects.T @ np.linalg.pinv(gram) @ (point - base_mean).reshape(-1, 1)
            choices.append(np.clip(baseline + delta.reshape(-1), lower, upper))
        z = max(
            choices,
            key=lambda choice: independent_normal_success_probability(
                base_mean + effects @ choice,
                np.ones_like(base_mean),
                intervals,
            ),
        )
        solver_status = "clipped_weighted_least_squares"
    else:
        solver_status = "weighted_least_squares"
    return z, solver_status, {"target_mean": target.tolist(), "predicted_mean": mean.tolist()}


def _normal_quantile(probability: float) -> float:
    # Acklam-style approximation is overkill here; a short binary search on erf is enough.
    target = float(np.clip(probability, 1e-8, 1.0 - 1e-8))
    lo, hi = -8.0, 8.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        cdf = 0.5 * (1.0 + math.erf(mid / np.sqrt(2.0)))
        if cdf < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


NeurIPS2024MICNSRehearsal = MICNSRehearsal
