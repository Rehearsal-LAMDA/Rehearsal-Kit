"""NeurIPS 2023 QWZ23 rehearsal adapter."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from rehearsal.core import AUFTask, DecisionResult, candidate_alteration_sets, coerce_data_matrix
from rehearsal.core.regions import desired_region_intervals_under_independence
from rehearsal.models import LinearGaussianSRM


@dataclass(frozen=True)
class _GraphHypothesis:
    model: LinearGaussianSRM
    weight: float
    parent_map: dict[str, tuple[str, ...]]
    diagnostics: dict[str, Any]


class QWZ23Rehearsal:
    """Graph-uncertain linear SEM rehearsal with interval and information-gain search."""

    def __init__(
        self,
        *,
        seed: int | None = None,
        candidate_alteration_sets: Sequence[Sequence[str]] | None = None,
        n_graph_samples: int = 8,
        interval_grid_size: int = 41,
        bootstrap_fraction: float = 0.8,
        edge_keep_threshold: float = 0.15,
        success_threshold: float = 0.7,
        bound_delta: float = 0.05,
        fit_ridge: float = 1e-4,
    ) -> None:
        self.seed = seed
        self.candidate_alteration_sets = (
            tuple(tuple(candidate) for candidate in candidate_alteration_sets)
            if candidate_alteration_sets is not None
            else None
        )
        self.n_graph_samples = int(n_graph_samples)
        self.interval_grid_size = int(interval_grid_size)
        self.bootstrap_fraction = float(bootstrap_fraction)
        self.edge_keep_threshold = float(edge_keep_threshold)
        self.success_threshold = float(success_threshold)
        self.bound_delta = float(bound_delta)
        self.fit_ridge = float(fit_ridge)

        self.config_: dict[str, Any] = {}
        self.columns_: tuple[str, ...] | None = None
        self.data_matrix_: np.ndarray | None = None
        self.graph_hypotheses_: tuple[_GraphHypothesis, ...] = ()
        self.fit_diagnostics_: dict[str, Any] = {}
        self.last_decision_: DecisionResult | None = None
        self.last_context_: dict[str, Any] | None = None

    def fit(
        self,
        data: Mapping[str, Sequence[float]] | np.ndarray,
        task: AUFTask,
        config: Mapping[str, Any] | None = None,
    ) -> "QWZ23Rehearsal":
        config_dict = dict(config or {})
        self.config_ = config_dict
        matrix, columns = coerce_data_matrix(data, task, config_dict)
        self.data_matrix_ = matrix
        self.columns_ = columns

        graph_hypotheses = _bootstrap_graph_hypotheses(
            matrix,
            columns,
            task.parents,
            n_graph_samples=int(config_dict.get("n_graph_samples", self.n_graph_samples)),
            bootstrap_fraction=float(config_dict.get("bootstrap_fraction", self.bootstrap_fraction)),
            edge_keep_threshold=float(config_dict.get("edge_keep_threshold", self.edge_keep_threshold)),
            fit_ridge=float(config_dict.get("fit_ridge", self.fit_ridge)),
            seed=self.seed,
        )
        self.graph_hypotheses_ = tuple(graph_hypotheses)
        self.fit_diagnostics_ = {
            "n_training_samples": int(matrix.shape[0]),
            "columns": columns,
            "n_graph_hypotheses": len(graph_hypotheses),
            "graph_uncertainty": [
                {
                    "weight": hypothesis.weight,
                    "parents": hypothesis.parent_map,
                    **hypothesis.diagnostics,
                }
                for hypothesis in graph_hypotheses
            ],
        }
        return self

    def suggest(self, observation: Mapping[str, float], task: AUFTask) -> DecisionResult:
        if not self.graph_hypotheses_:
            raise RuntimeError("QWZ23Rehearsal.fit must be called before suggest.")

        start = time.perf_counter()
        candidates = candidate_alteration_sets(task, self.candidate_alteration_sets)
        intervals = desired_region_intervals_under_independence(task.desired_region, task.outcomes)
        success_threshold = float(self.config_.get("success_threshold", self.success_threshold))
        bound_delta = float(self.config_.get("bound_delta", self.bound_delta))
        grid_size = int(self.config_.get("interval_grid_size", self.interval_grid_size))
        rng = np.random.default_rng(self.seed)

        best: dict[str, Any] | None = None
        candidate_diagnostics = []
        for candidate in candidates:
            lower, upper = task.alteration_domain.arrays_for(candidate)
            if len(candidate) != 1:
                raise ValueError("QWZ23Rehearsal currently supports single-variable candidate sets only.")
            grid = np.linspace(lower[0], upper[0], grid_size)
            success_by_value = []
            info_gain_by_value = []
            bounds_by_value = []
            for value in grid:
                per_graph_probs = np.asarray(
                    [
                        _graph_success_probability(hypothesis.model, task, observation, candidate, np.array([value]), intervals)
                        for hypothesis in self.graph_hypotheses_
                    ],
                    dtype=float,
                )
                weights = np.asarray([hypothesis.weight for hypothesis in self.graph_hypotheses_], dtype=float)
                weights = weights / np.sum(weights)
                expected_success = float(np.dot(weights, per_graph_probs))
                info_gain = _binary_information_gain(per_graph_probs, weights)
                p_hat, p_low, p_high = _success_prob_bound_from_graphs(per_graph_probs, weights, delta=bound_delta)
                success_by_value.append(expected_success)
                info_gain_by_value.append(info_gain)
                bounds_by_value.append((p_hat, p_low, p_high))

            feasible_intervals = _find_feasible_intervals(grid, success_by_value, success_threshold)
            active_intervals = feasible_intervals if feasible_intervals else [(float(lower[0]), float(upper[0]))]
            candidate_value, info_gain = _select_value_by_information_gain(grid, info_gain_by_value, active_intervals)
            closest_idx = int(np.argmin(np.abs(grid - candidate_value)))
            p_hat, p_low, p_high = bounds_by_value[closest_idx]
            expected_success = float(success_by_value[closest_idx])
            cost = task.alteration_domain.cost({candidate[0]: float(candidate_value)})
            diagnostics = {
                "candidate": candidate,
                "value_grid": grid.tolist(),
                "success_grid": [float(value) for value in success_by_value],
                "info_gain_grid": [float(value) for value in info_gain_by_value],
                "feasible_intervals": feasible_intervals,
                "selected_interval": next((interval for interval in active_intervals if interval[0] <= candidate_value <= interval[1]), None),
                "success_probability_bound": {
                    "p_hat": float(p_hat),
                    "p_low": float(p_low),
                    "p_high": float(p_high),
                    "delta": float(bound_delta),
                },
                "selected_information_gain": float(info_gain),
                "had_feasible_interval": bool(feasible_intervals),
            }
            candidate_diagnostics.append(diagnostics)

            if best is None or info_gain > best["info_gain"] + 1e-15:
                best = {
                    "candidate": candidate,
                    "value": float(candidate_value),
                    "info_gain": float(info_gain),
                    "expected_success": expected_success,
                    "p_hat": float(p_hat),
                    "p_low": float(p_low),
                    "p_high": float(p_high),
                    "cost": float(cost),
                    "diagnostics": diagnostics,
                }
            elif (
                best is not None
                and abs(info_gain - best["info_gain"]) <= 1e-15
                and p_low > best["p_low"] + 1e-15
            ):
                best = {
                    "candidate": candidate,
                    "value": float(candidate_value),
                    "info_gain": float(info_gain),
                    "expected_success": expected_success,
                    "p_hat": float(p_hat),
                    "p_low": float(p_low),
                    "p_high": float(p_high),
                    "cost": float(cost),
                    "diagnostics": diagnostics,
                }

        if best is None:
            raise ValueError("No candidate alteration set is available.")

        runtime = time.perf_counter() - start
        selected_candidate = tuple(best["candidate"])
        alterations = {selected_candidate[0]: float(best["value"])}
        result = DecisionResult(
            alterations=alterations,
            estimated_success_probability=float(np.clip(best["expected_success"], 0.0, 1.0)),
            cost=float(best["cost"]),
            diagnostics={
                "selected_candidate": selected_candidate,
                "information_gain": float(best["info_gain"]),
                "success_probability_bound": {
                    "p_hat": float(best["p_hat"]),
                    "p_low": float(best["p_low"]),
                    "p_high": float(best["p_high"]),
                    "delta": float(bound_delta),
                },
                "n_candidates": len(candidates),
                "candidate_diagnostics": candidate_diagnostics,
                **self.fit_diagnostics_,
                **best["diagnostics"],
            },
            runtime_seconds=runtime,
        )
        self.last_decision_ = result
        self.last_context_ = {
            "candidate": selected_candidate,
            "value": float(best["value"]),
            "bound": (float(best["p_hat"]), float(best["p_low"]), float(best["p_high"])),
            "observation": dict(observation),
            "rng_seed": int(rng.integers(0, 2**31 - 1)),
        }
        return result

    def evaluate(self, task: AUFTask, n_samples: int) -> Mapping[str, Any]:
        if self.last_decision_ is None or self.last_context_ is None:
            raise RuntimeError("suggest must be called before evaluate.")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")

        candidate = tuple(self.last_context_["candidate"])
        value = float(self.last_context_["value"])
        intervals = desired_region_intervals_under_independence(task.desired_region, task.outcomes)
        weights = np.asarray([hypothesis.weight for hypothesis in self.graph_hypotheses_], dtype=float)
        weights = weights / np.sum(weights)
        probabilities = np.asarray(
            [
                _graph_success_probability(
                    hypothesis.model,
                    task,
                    dict(self.last_context_["observation"]),
                    candidate,
                    np.array([value]),
                    intervals,
                )
                for hypothesis in self.graph_hypotheses_
            ],
            dtype=float,
        )
        p_hat, p_low, p_high = _success_prob_bound_from_graphs(
            probabilities,
            weights,
            delta=float(self.config_.get("bound_delta", self.bound_delta)),
        )
        return {
            "estimated_success_probability": float(self.last_decision_.estimated_success_probability),
            "success_probability_bound": {
                "p_hat": float(p_hat),
                "p_low": float(p_low),
                "p_high": float(p_high),
            },
            "graph_ensemble_size": len(self.graph_hypotheses_),
            "n_samples": int(n_samples),
            "alterations": dict(self.last_decision_.alterations),
        }


def _bootstrap_graph_hypotheses(
    matrix: np.ndarray,
    columns: Sequence[str],
    parents: Mapping[str, Sequence[str]],
    *,
    n_graph_samples: int,
    bootstrap_fraction: float,
    edge_keep_threshold: float,
    fit_ridge: float,
    seed: int | None,
) -> list[_GraphHypothesis]:
    rng = np.random.default_rng(seed)
    n_samples = matrix.shape[0]
    bootstrap_n = max(2, int(math.ceil(n_samples * bootstrap_fraction)))
    column_index = {name: idx for idx, name in enumerate(columns)}
    hypotheses: list[_GraphHypothesis] = []

    for graph_idx in range(max(1, n_graph_samples)):
        sample_index = rng.choice(n_samples, size=bootstrap_n, replace=True)
        sample = matrix[sample_index]
        theta: dict[str, dict[str, float]] = {name: {} for name in columns}
        parent_map: dict[str, tuple[str, ...]] = {}
        residuals = np.zeros_like(sample, dtype=float)
        kept_edges = 0

        for child in columns:
            y = sample[:, column_index[child]]
            parent_names = tuple(name for name in parents.get(child, ()) if name in column_index)
            if not parent_names:
                residuals[:, column_index[child]] = y
                parent_map[child] = ()
                continue
            x = sample[:, [column_index[name] for name in parent_names]]
            beta = _ridge_fit(x, y, fit_ridge)
            scale = np.max(np.abs(beta)) if beta.size else 0.0
            keep = np.abs(beta) >= max(1e-10, edge_keep_threshold * max(scale, 1e-10))
            chosen_parents = []
            prediction = np.zeros_like(y, dtype=float)
            for idx, parent_name in enumerate(parent_names):
                if keep[idx]:
                    theta.setdefault(parent_name, {})[child] = float(beta[idx])
                    chosen_parents.append(parent_name)
                    prediction += float(beta[idx]) * x[:, idx]
                    kept_edges += 1
            parent_map[child] = tuple(chosen_parents)
            residuals[:, column_index[child]] = y - prediction

        covariance = np.cov(residuals, rowvar=False, bias=True)
        covariance = np.atleast_2d(np.asarray(covariance, dtype=float))
        covariance = covariance + 1e-6 * np.eye(len(columns))
        model = LinearGaussianSRM(tuple(columns), theta, covariance)
        hypotheses.append(
            _GraphHypothesis(
                model=model,
                weight=1.0,
                parent_map=parent_map,
                diagnostics={
                    "graph_index": graph_idx,
                    "bootstrap_n": bootstrap_n,
                    "kept_edges": kept_edges,
                },
            )
        )

    total_weight = float(len(hypotheses))
    return [
        _GraphHypothesis(
            model=hypothesis.model,
            weight=1.0 / total_weight,
            parent_map=hypothesis.parent_map,
            diagnostics=dict(hypothesis.diagnostics),
        )
        for hypothesis in hypotheses
    ]


def _ridge_fit(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    system = x.T @ x + float(ridge) * np.eye(x.shape[1])
    rhs = x.T @ y
    return (np.linalg.pinv(system) @ rhs).reshape(-1)


def _graph_success_probability(
    model: LinearGaussianSRM,
    task: AUFTask,
    observation: Mapping[str, float],
    candidate: Sequence[str],
    z: np.ndarray,
    intervals: np.ndarray,
) -> float:
    mean, covariance = model.outcome_moments(task, observation, candidate, z)
    variance = np.maximum(np.diag(covariance), 1e-8)
    lower = intervals[:, 0]
    upper = intervals[:, 1]
    std = np.sqrt(variance)
    upper_prob = 0.5 * (1.0 + np.vectorize(math.erf)((upper - mean) / (std * np.sqrt(2.0))))
    lower_prob = 0.5 * (1.0 + np.vectorize(math.erf)((lower - mean) / (std * np.sqrt(2.0))))
    return float(np.clip(np.prod(np.maximum(upper_prob - lower_prob, 0.0)), 0.0, 1.0))


def _find_feasible_intervals(grid: np.ndarray, success_grid: Sequence[float], threshold: float) -> list[tuple[float, float]]:
    success = np.asarray(success_grid, dtype=float)
    valid = success >= float(threshold)
    intervals: list[tuple[float, float]] = []
    start: float | None = None
    prev = None
    for value, keep in zip(grid, valid):
        if keep and start is None:
            start = float(value)
        if not keep and start is not None and prev is not None:
            intervals.append((start, float(prev)))
            start = None
        prev = float(value)
    if start is not None and prev is not None:
        intervals.append((start, float(prev)))
    return intervals


def _select_value_by_information_gain(
    grid: np.ndarray,
    info_gain_grid: Sequence[float],
    intervals: Sequence[tuple[float, float]],
) -> tuple[float, float]:
    best_value = float(grid[0])
    best_gain = float("-inf")
    for value, gain in zip(grid, info_gain_grid):
        if any(lower <= float(value) <= upper for lower, upper in intervals) and gain > best_gain:
            best_gain = float(gain)
            best_value = float(value)
    return best_value, best_gain


def _binary_information_gain(probabilities: np.ndarray, weights: np.ndarray) -> float:
    p = float(np.dot(weights, probabilities))
    predictive_entropy = _binary_entropy(p)
    expected_entropy = float(np.dot(weights, [_binary_entropy(probability) for probability in probabilities]))
    return float(max(0.0, predictive_entropy - expected_entropy))


def _binary_entropy(probability: float) -> float:
    p = float(np.clip(probability, 1e-12, 1.0 - 1e-12))
    return float(-(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)))


def _success_prob_bound_from_graphs(probabilities: np.ndarray, weights: np.ndarray, *, delta: float) -> tuple[float, float, float]:
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    p_hat = float(np.clip(np.dot(weights, probabilities), 0.0, 1.0))
    variance = float(np.dot(weights, (probabilities - p_hat) ** 2))
    effective_n = max(1.0, 1.0 / float(np.sum(weights * weights)))
    radius = math.sqrt(2.0 * variance * math.log(2.0 / max(delta, 1e-12)) / effective_n) + 7.0 * math.log(2.0 / max(delta, 1e-12)) / (3.0 * max(effective_n - 1.0, 1.0))
    return p_hat, float(max(0.0, p_hat - radius)), float(min(1.0, p_hat + radius))


NeurIPS2023QWZ23Rehearsal = QWZ23Rehearsal
