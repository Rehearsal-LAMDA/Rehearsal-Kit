"""NeurIPS 2023 QWZ23 rehearsal adapter."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from rehearsal.core import AUFTask, DecisionResult, candidate_alteration_sets, coerce_data_matrix
from rehearsal.models import LinearGaussianSRM


@dataclass(frozen=True)
class _GraphHypothesis:
    model: LinearGaussianSRM
    weight: float
    parent_map: dict[str, tuple[str, ...]]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class _AffineOutcomeSamples:
    bias: np.ndarray
    z_coef: np.ndarray
    weights: np.ndarray


class QWZ23Rehearsal:
    """Graph-uncertain SEM rehearsal with sampled multivariate maximization.

    The decision stage follows the MultivariateBaseline idea from
    ``previous_works/04-AAAI 2025/code/baseline_multi.py``: sample structural
    noise, express each sampled outcome as an affine function of the proposed
    alteration values, then maximize the sampled probability of satisfying
    ``M y <= d`` under the alteration bounds.
    """

    def __init__(
        self,
        *,
        seed: int | None = None,
        candidate_alteration_sets: Sequence[Sequence[str]] | None = None,
        n_graph_samples: int = 8,
        n_optimization_samples: int = 128,
        optimization_solver: str = "auto",
        milp_time_limit: float = 10.0,
        random_search_size: int = 2048,
        big_m: float = 1e6,
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
        self.n_optimization_samples = int(n_optimization_samples)
        self.optimization_solver = str(optimization_solver)
        self.milp_time_limit = float(milp_time_limit)
        self.random_search_size = int(random_search_size)
        self.big_m = float(big_m)
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
        n_optimization_samples = int(self.config_.get("n_optimization_samples", self.n_optimization_samples))
        optimization_solver = str(self.config_.get("optimization_solver", self.optimization_solver))
        milp_time_limit = float(self.config_.get("milp_time_limit", self.milp_time_limit))
        random_search_size = int(self.config_.get("random_search_size", self.random_search_size))
        big_m = float(self.config_.get("big_m", self.big_m))
        bound_delta = float(self.config_.get("bound_delta", self.bound_delta))
        rng = np.random.default_rng(self.seed)

        best: dict[str, Any] | None = None
        candidate_diagnostics = []
        for candidate in candidates:
            candidate = tuple(candidate)
            lower, upper = task.alteration_domain.arrays_for(candidate)
            sample_seed = int(rng.integers(0, 2**31 - 1))
            samples = _sample_affine_outcome_models(
                self.graph_hypotheses_,
                task,
                observation,
                candidate,
                n_samples=n_optimization_samples,
                seed=sample_seed,
            )
            solution = _maximize_sampled_success(
                samples,
                task.desired_region.matrix,
                task.desired_region.vector,
                lower,
                upper,
                solver=optimization_solver,
                random_search_size=random_search_size,
                big_m=big_m,
                time_limit=milp_time_limit,
                seed=sample_seed + 17,
            )
            alterations = {name: float(value) for name, value in zip(candidate, solution["values"])}
            cost = task.alteration_domain.cost(alterations)
            success_indicators = _sample_success_indicators(
                samples,
                task.desired_region.matrix,
                task.desired_region.vector,
                solution["values"],
            )
            p_hat, p_low, p_high = _success_prob_bound_from_samples(
                success_indicators,
                samples.weights,
                delta=bound_delta,
            )
            diagnostics = {
                "candidate": candidate,
                "alterations": alterations,
                "estimated_sampled_success_probability": float(solution["estimated_success_probability"]),
                "success_probability_bound": {
                    "p_hat": float(p_hat),
                    "p_low": float(p_low),
                    "p_high": float(p_high),
                    "delta": float(bound_delta),
                },
                "optimization_solver": solution["solver"],
                "optimization_status": solution["status"],
                "n_optimization_samples_per_graph": int(n_optimization_samples),
                "n_weighted_optimization_samples": int(samples.bias.shape[0]),
                "random_search_size": int(random_search_size),
                "big_m": float(big_m),
                **solution["diagnostics"],
            }
            candidate_diagnostics.append(diagnostics)

            expected_success = float(solution["estimated_success_probability"])
            if best is None or expected_success > best["expected_success"] + 1e-15:
                best = {
                    "candidate": candidate,
                    "alterations": alterations,
                    "expected_success": expected_success,
                    "p_hat": float(p_hat),
                    "p_low": float(p_low),
                    "p_high": float(p_high),
                    "cost": float(cost),
                    "diagnostics": diagnostics,
                    "sample_seed": sample_seed,
                }
            elif (
                best is not None
                and abs(expected_success - best["expected_success"]) <= 1e-15
                and cost < best["cost"] - 1e-15
            ):
                best = {
                    "candidate": candidate,
                    "alterations": alterations,
                    "expected_success": expected_success,
                    "p_hat": float(p_hat),
                    "p_low": float(p_low),
                    "p_high": float(p_high),
                    "cost": float(cost),
                    "diagnostics": diagnostics,
                    "sample_seed": sample_seed,
                }

        if best is None:
            raise ValueError("No candidate alteration set is available.")

        runtime = time.perf_counter() - start
        selected_candidate = tuple(best["candidate"])
        result = DecisionResult(
            alterations=dict(best["alterations"]),
            estimated_success_probability=float(np.clip(best["expected_success"], 0.0, 1.0)),
            cost=float(best["cost"]),
            diagnostics={
                "selected_candidate": selected_candidate,
                "selected_solver": best["diagnostics"]["optimization_solver"],
                "selected_status": best["diagnostics"]["optimization_status"],
                "sampled_success_maximization": True,
                "success_probability_bound": {
                    "p_hat": float(best["p_hat"]),
                    "p_low": float(best["p_low"]),
                    "p_high": float(best["p_high"]),
                    "delta": float(bound_delta),
                },
                "n_candidates": len(candidates),
                "candidate_diagnostics": candidate_diagnostics,
                "legacy_interval_grid_size_ignored": int(self.interval_grid_size),
                "legacy_success_threshold_ignored": float(self.success_threshold),
                **self.fit_diagnostics_,
                **best["diagnostics"],
            },
            runtime_seconds=runtime,
        )
        self.last_decision_ = result
        self.last_context_ = {
            "candidate": selected_candidate,
            "observation": dict(observation),
            "sample_seed": int(best["sample_seed"]),
        }
        return result

    def evaluate(self, task: AUFTask, n_samples: int) -> Mapping[str, Any]:
        if self.last_decision_ is None or self.last_context_ is None:
            raise RuntimeError("suggest must be called before evaluate.")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")

        candidate = tuple(self.last_context_["candidate"])
        samples = _sample_affine_outcome_models(
            self.graph_hypotheses_,
            task,
            dict(self.last_context_["observation"]),
            candidate,
            n_samples=int(n_samples),
            seed=int(self.last_context_["sample_seed"]) + 1009,
        )
        z = np.asarray([self.last_decision_.alterations[name] for name in candidate], dtype=float)
        success_indicators = _sample_success_indicators(
            samples,
            task.desired_region.matrix,
            task.desired_region.vector,
            z,
        )
        probability = float(np.dot(samples.weights, success_indicators.astype(float)))
        p_hat, p_low, p_high = _success_prob_bound_from_samples(
            success_indicators,
            samples.weights,
            delta=float(self.config_.get("bound_delta", self.bound_delta)),
        )
        return {
            "estimated_success_probability": probability,
            "success_probability_bound": {
                "p_hat": float(p_hat),
                "p_low": float(p_low),
                "p_high": float(p_high),
            },
            "graph_ensemble_size": len(self.graph_hypotheses_),
            "n_samples": int(n_samples),
            "samples_per_graph": int(n_samples),
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


def _sample_affine_outcome_models(
    hypotheses: Sequence[_GraphHypothesis],
    task: AUFTask,
    observation: Mapping[str, float],
    candidate: Sequence[str],
    *,
    n_samples: int,
    seed: int,
) -> _AffineOutcomeSamples:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    rng = np.random.default_rng(seed)
    x = np.asarray([float(observation[name]) for name in task.observed], dtype=float).reshape(-1)
    bias_blocks = []
    z_coef_blocks = []
    weight_blocks = []
    for hypothesis in hypotheses:
        mat_a, mat_b, mat_c = hypothesis.model.effect_matrices(task, candidate)
        noise = rng.multivariate_normal(
            np.zeros(len(hypothesis.model.variable_order)),
            hypothesis.model.covariance,
            size=int(n_samples),
        )
        bias = (mat_a @ x).reshape(1, -1) + noise @ mat_c.T
        z_coef = np.broadcast_to(mat_b, (int(n_samples), *mat_b.shape)).copy()
        weights = np.full(int(n_samples), float(hypothesis.weight) / float(n_samples), dtype=float)
        bias_blocks.append(bias)
        z_coef_blocks.append(z_coef)
        weight_blocks.append(weights)

    weights = np.concatenate(weight_blocks)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        raise ValueError("Graph hypothesis weights must sum to a positive value.")
    return _AffineOutcomeSamples(
        bias=np.vstack(bias_blocks),
        z_coef=np.vstack(z_coef_blocks),
        weights=weights / weight_sum,
    )


def _maximize_sampled_success(
    samples: _AffineOutcomeSamples,
    region_matrix: np.ndarray,
    region_vector: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    solver: str,
    random_search_size: int,
    big_m: float,
    time_limit: float,
    seed: int,
) -> dict[str, Any]:
    solver_name = str(solver).lower()
    if solver_name not in {"auto", "milp", "random-search", "random_search"}:
        raise ValueError("optimization_solver must be 'auto', 'milp', or 'random_search'.")

    if solver_name in {"auto", "milp"}:
        try:
            return _maximize_sampled_success_milp(
                samples,
                region_matrix,
                region_vector,
                lower,
                upper,
                big_m=big_m,
                time_limit=time_limit,
            )
        except Exception as exc:
            if solver_name == "milp":
                raise
            fallback = _maximize_sampled_success_random_search(
                samples,
                region_matrix,
                region_vector,
                lower,
                upper,
                random_search_size=random_search_size,
                seed=seed,
            )
            fallback["diagnostics"]["milp_fallback_reason"] = f"{type(exc).__name__}: {exc}"
            return fallback

    return _maximize_sampled_success_random_search(
        samples,
        region_matrix,
        region_vector,
        lower,
        upper,
        random_search_size=random_search_size,
        seed=seed,
    )


def _maximize_sampled_success_milp(
    samples: _AffineOutcomeSamples,
    region_matrix: np.ndarray,
    region_vector: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    big_m: float,
    time_limit: float,
) -> dict[str, Any]:
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
    except ImportError as exc:
        raise ImportError("QWZ23 MILP optimization requires scipy; install rehearsal[qwz23].") from exc

    bias = np.asarray(samples.bias, dtype=float)
    z_coef = np.asarray(samples.z_coef, dtype=float)
    matrix = np.asarray(region_matrix, dtype=float)
    vector = np.asarray(region_vector, dtype=float).reshape(-1)
    lower = np.asarray(lower, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)
    n_samples = bias.shape[0]
    n_region_rows = matrix.shape[0]
    n_alter = lower.size

    constraint_matrix = np.zeros((n_samples * n_region_rows, n_samples + n_alter), dtype=float)
    constraint_upper = np.zeros(n_samples * n_region_rows, dtype=float)
    for sample_idx in range(n_samples):
        rows = slice(sample_idx * n_region_rows, (sample_idx + 1) * n_region_rows)
        constraint_matrix[rows, sample_idx] = float(big_m)
        constraint_matrix[rows, n_samples:] = matrix @ z_coef[sample_idx]
        constraint_upper[rows] = vector - matrix @ bias[sample_idx] + float(big_m)

    objective = np.zeros(n_samples + n_alter, dtype=float)
    objective[:n_samples] = -np.asarray(samples.weights, dtype=float)
    integrality = np.ones(n_samples + n_alter, dtype=float)
    integrality[n_samples:] = 0.0
    bounds_lower = np.zeros(n_samples + n_alter, dtype=float)
    bounds_upper = np.ones(n_samples + n_alter, dtype=float)
    bounds_lower[n_samples:] = lower
    bounds_upper[n_samples:] = upper

    options: dict[str, float] = {}
    if time_limit > 0:
        options["time_limit"] = float(time_limit)
    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(bounds_lower, bounds_upper),
        constraints=LinearConstraint(constraint_matrix, ub=constraint_upper),
        options=options,
    )
    if result.x is None or not np.all(np.isfinite(result.x[n_samples:])):
        raise RuntimeError(f"scipy.optimize.milp did not return finite alteration values: status={result.status}")

    values = np.asarray(result.x[n_samples:], dtype=float)
    values = np.clip(values, lower, upper)
    estimated_success = _sample_success_probability(samples, matrix, vector, values)
    return {
        "values": values,
        "estimated_success_probability": estimated_success,
        "solver": "milp",
        "status": str(result.message),
        "diagnostics": {
            "milp_success": bool(result.success),
            "milp_status": int(result.status),
            "milp_objective": float(result.fun) if result.fun is not None else None,
            "milp_time_limit": float(time_limit),
        },
    }


def _maximize_sampled_success_random_search(
    samples: _AffineOutcomeSamples,
    region_matrix: np.ndarray,
    region_vector: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    random_search_size: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    lower = np.asarray(lower, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)
    n_alter = lower.size
    n_random = max(0, int(random_search_size))
    candidates = []
    candidates.append(np.zeros(n_alter, dtype=float))
    candidates.append((lower + upper) / 2.0)
    candidates.append(lower.copy())
    candidates.append(upper.copy())
    if n_random:
        candidates.extend(rng.uniform(lower, upper, size=(n_random, n_alter)))

    best_values = np.clip(candidates[0], lower, upper)
    best_probability = -1.0
    for values in candidates:
        values = np.clip(np.asarray(values, dtype=float), lower, upper)
        probability = _sample_success_probability(samples, region_matrix, region_vector, values)
        if probability > best_probability + 1e-15:
            best_probability = probability
            best_values = values

    return {
        "values": best_values,
        "estimated_success_probability": float(max(0.0, best_probability)),
        "solver": "random_search",
        "status": "success",
        "diagnostics": {
            "random_search_trials": len(candidates),
        },
    }


def _sample_success_probability(
    samples: _AffineOutcomeSamples,
    region_matrix: np.ndarray,
    region_vector: np.ndarray,
    values: np.ndarray,
) -> float:
    success = _sample_success_indicators(samples, region_matrix, region_vector, values)
    return float(np.clip(np.dot(samples.weights, success.astype(float)), 0.0, 1.0))


def _sample_success_indicators(
    samples: _AffineOutcomeSamples,
    region_matrix: np.ndarray,
    region_vector: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    outcomes = samples.bias + np.einsum("nyk,k->ny", samples.z_coef, values)
    matrix = np.asarray(region_matrix, dtype=float)
    vector = np.asarray(region_vector, dtype=float)
    return np.all(outcomes @ matrix.T <= vector + 1e-12, axis=1)


def _success_prob_bound_from_samples(
    success_indicators: np.ndarray,
    weights: np.ndarray,
    *,
    delta: float,
) -> tuple[float, float, float]:
    indicators = np.asarray(success_indicators, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    weights = weights / np.sum(weights)
    p_hat = float(np.clip(np.dot(weights, indicators), 0.0, 1.0))
    variance = float(np.dot(weights, (indicators - p_hat) ** 2))
    effective_n = max(1.0, 1.0 / float(np.sum(weights * weights)))
    radius = math.sqrt(2.0 * variance * math.log(2.0 / max(delta, 1e-12)) / effective_n)
    radius += 7.0 * math.log(2.0 / max(delta, 1e-12)) / (3.0 * max(effective_n - 1.0, 1.0))
    return p_hat, float(max(0.0, p_hat - radius)), float(min(1.0, p_hat + radius))


NeurIPS2023QWZ23Rehearsal = QWZ23Rehearsal
