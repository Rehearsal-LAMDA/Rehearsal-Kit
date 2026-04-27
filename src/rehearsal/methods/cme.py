"""Unpublished conditional-mean-embedding rehearsal adapter."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from rehearsal.core import AUFTask, DecisionResult
from rehearsal.core.data import candidate_alteration_sets, coerce_data_matrix
from rehearsal.metrics.cme import desired_region_surrogate_weights, median_or_mean_bandwidth, rbf_kernel
from rehearsal.optimizers.cme import cme_action_kernel, optimize_action_projected_gradient


@dataclass(frozen=True)
class _CandidateState:
    candidate: tuple[str, ...]
    environment_variables: tuple[str, ...]
    action_history: np.ndarray
    environment_kernel: np.ndarray | None
    action_bandwidth: float
    environment_bandwidth: float | None
    alpha: np.ndarray
    alpha_solver_status: str


class CMERehearsal:
    """Kernel conditional-mean-embedding adapter for non-parametric rehearsal."""

    def __init__(
        self,
        *,
        seed: int | None = None,
        candidate_alteration_sets: Sequence[Sequence[str]] | None = None,
        bandwidths: str | float | Mapping[str, float | str] = "auto",
        bandwidth_method: str = "legacy_mean",
        eta_surrogate: float | None = 10.0,
        lambda_surrogate: float | None = None,
        krr_lambda_alpha: float = 1e-1,
        krr_lambda_gamma: float = 1e-3,
        pgd_lr: float = 0.1,
        pgd_steps: int = 100,
        num_restarts: int = 10,
        tolerance: float = 1e-6,
        positive_weight_threshold: float = 1e-9,
    ) -> None:
        self.seed = seed
        self.candidate_alteration_sets = (
            tuple(tuple(candidate) for candidate in candidate_alteration_sets)
            if candidate_alteration_sets is not None
            else None
        )
        self.bandwidths = bandwidths
        self.bandwidth_method = bandwidth_method
        self.eta_surrogate = eta_surrogate if lambda_surrogate is None else lambda_surrogate
        self.krr_lambda_alpha = float(krr_lambda_alpha)
        self.krr_lambda_gamma = float(krr_lambda_gamma)
        self.pgd_lr = float(pgd_lr)
        self.pgd_steps = int(pgd_steps)
        self.num_restarts = int(num_restarts)
        self.tolerance = float(tolerance)
        self.positive_weight_threshold = float(positive_weight_threshold)

        self.config_: dict[str, Any] = {}
        self.columns_: tuple[str, ...] | None = None
        self.column_index_: dict[str, int] = {}
        self.X_hist_: np.ndarray | None = None
        self.Y_hist_: np.ndarray | None = None
        self.W_vec_: np.ndarray | None = None
        self.hard_success_: np.ndarray | None = None
        self.K_xx_: np.ndarray | None = None
        self.x_bandwidth_: float | None = None
        self.candidate_states_: tuple[_CandidateState, ...] = ()
        self.fit_diagnostics_: dict[str, Any] = {}
        self.last_decision_: DecisionResult | None = None
        self.last_context_: dict[str, Any] | None = None

    def fit(
        self,
        data: Mapping[str, Sequence[float]] | np.ndarray,
        task: AUFTask,
        config: Mapping[str, Any] | None = None,
    ) -> "CMERehearsal":
        config_dict = dict(config or {})
        self.config_ = config_dict
        columns = _default_columns(data, task, config_dict)
        matrix, resolved_columns = coerce_data_matrix(data, task, config_dict, columns=columns)
        self._data_matrix = matrix
        self.columns_ = resolved_columns
        self.column_index_ = {name: idx for idx, name in enumerate(resolved_columns)}

        self.X_hist_ = self._matrix_for(task.observed)
        self.Y_hist_ = self._matrix_for(task.outcomes)
        eta_surrogate = config_dict.get("eta_surrogate", config_dict.get("lambda_surrogate", self.eta_surrogate))
        self.W_vec_ = desired_region_surrogate_weights(
            self.Y_hist_,
            task.desired_region,
            eta_surrogate=eta_surrogate,
        )
        self.hard_success_ = np.asarray(task.desired_region.contains(self.Y_hist_), dtype=float)

        self.x_bandwidth_ = self._resolve_bandwidth("x", self.X_hist_)
        self.K_xx_ = rbf_kernel(self.X_hist_, self.X_hist_, self.x_bandwidth_)

        candidates = candidate_alteration_sets(task, self.candidate_alteration_sets)
        environment_variables = self._infer_environment_variables(task)
        states = []
        for candidate in candidates:
            action_history = self._matrix_for(candidate)
            action_bandwidth = self._resolve_bandwidth("a", action_history)
            action_kernel = rbf_kernel(action_history, action_history, action_bandwidth)

            candidate_environment = self._environment_for_candidate(task, candidate, environment_variables)
            if candidate_environment:
                environment_history = self._matrix_for(candidate_environment)
                environment_bandwidth = self._resolve_bandwidth("u", environment_history)
                environment_kernel = rbf_kernel(environment_history, environment_history, environment_bandwidth)
            else:
                environment_bandwidth = None
                environment_kernel = None

            combined_kernel = self.K_xx_ * action_kernel
            if environment_kernel is not None:
                combined_kernel = combined_kernel * environment_kernel
            alpha, alpha_status = _regularized_solve(
                combined_kernel,
                self.W_vec_,
                self.krr_lambda_alpha,
            )
            states.append(
                _CandidateState(
                    candidate=tuple(candidate),
                    environment_variables=candidate_environment,
                    action_history=action_history,
                    environment_kernel=environment_kernel,
                    action_bandwidth=action_bandwidth,
                    environment_bandwidth=environment_bandwidth,
                    alpha=alpha,
                    alpha_solver_status=alpha_status,
                )
            )

        self.candidate_states_ = tuple(states)
        self.fit_diagnostics_ = {
            "n_training_samples": int(matrix.shape[0]),
            "columns": resolved_columns,
            "environment_variables": environment_variables,
            "x_bandwidth": self.x_bandwidth_,
            "eta_surrogate": eta_surrogate,
            "krr_lambda_alpha": self.krr_lambda_alpha,
            "krr_lambda_gamma": self.krr_lambda_gamma,
        }
        return self

    def suggest(self, observation: Mapping[str, float], task: AUFTask) -> DecisionResult:
        if self.X_hist_ is None or self.K_xx_ is None or self.W_vec_ is None or self.x_bandwidth_ is None:
            raise RuntimeError("CMERehearsal.fit must be called before suggest.")
        if not self.candidate_states_:
            raise ValueError("No candidate alteration set is available.")

        start = time.perf_counter()
        x_query = np.asarray([float(observation[name]) for name in task.observed], dtype=float).reshape(1, -1)
        k_x_query = rbf_kernel(self.X_hist_, x_query, self.x_bandwidth_).reshape(-1)
        gamma, gamma_status = _regularized_solve(self.K_xx_, k_x_query, self.krr_lambda_gamma)

        best: dict[str, Any] | None = None
        candidate_diagnostics = []
        for idx, state in enumerate(self.candidate_states_):
            if state.environment_kernel is None:
                environment_influence = np.ones_like(gamma)
            else:
                environment_influence = state.environment_kernel @ gamma

            omega = state.alpha * k_x_query * environment_influence
            lower, upper = task.alteration_domain.arrays_for(state.candidate)
            rng = _candidate_rng(self.seed, idx)
            optimized = optimize_action_projected_gradient(
                state.action_history,
                omega,
                lower,
                upper,
                state.action_bandwidth,
                learning_rate=self.pgd_lr,
                max_steps=self.pgd_steps,
                num_restarts=self.num_restarts,
                tolerance=self.tolerance,
                positive_weight_threshold=self.positive_weight_threshold,
                rng=rng,
            )
            alterations = {name: float(value) for name, value in zip(state.candidate, optimized.action)}
            cost = task.alteration_domain.cost(alterations)
            action_kernel = cme_action_kernel(state.action_history, optimized.action, state.action_bandwidth)
            empirical_probability = self._weighted_hard_success_probability(k_x_query, environment_influence, action_kernel)
            diag = {
                "candidate": state.candidate,
                "environment_variables": state.environment_variables,
                "objective_value": optimized.objective_value,
                "estimated_success_probability": optimized.estimated_success_probability,
                "empirical_hard_success_rate": empirical_probability,
                "solver_status": optimized.solver_status,
                "action_bandwidth": state.action_bandwidth,
                "environment_bandwidth": state.environment_bandwidth,
                "alpha_solver_status": state.alpha_solver_status,
                **optimized.diagnostics,
            }
            candidate_diagnostics.append(diag)
            if _is_better_candidate(optimized.objective_value, optimized.estimated_success_probability, cost, best):
                best = {
                    "state": state,
                    "optimized": optimized,
                    "alterations": alterations,
                    "cost": cost,
                    "omega": omega,
                    "environment_influence": environment_influence,
                    "action_kernel": action_kernel,
                    "empirical_probability": empirical_probability,
                }

        if best is None:
            raise ValueError("No candidate alteration set is available.")

        runtime = time.perf_counter() - start
        state = best["state"]
        optimized = best["optimized"]
        diagnostics = {
            "selected_candidate": state.candidate,
            "objective_value": optimized.objective_value,
            "solver_status": optimized.solver_status,
            "n_candidates": len(self.candidate_states_),
            "n_training_samples": int(self.X_hist_.shape[0]),
            "x_bandwidth": self.x_bandwidth_,
            "action_bandwidth": state.action_bandwidth,
            "environment_bandwidth": state.environment_bandwidth,
            "gamma_solver_status": gamma_status,
            "candidate_diagnostics": candidate_diagnostics,
            **self.fit_diagnostics_,
            **optimized.diagnostics,
        }
        estimate = float(
            np.clip(
                0.5 * optimized.estimated_success_probability + 0.5 * best["empirical_probability"],
                0.0,
                1.0,
            )
        )
        result = DecisionResult(
            alterations=best["alterations"],
            estimated_success_probability=estimate,
            cost=float(best["cost"]),
            diagnostics=diagnostics,
            runtime_seconds=runtime,
        )
        self.last_decision_ = result
        self.last_context_ = {
            "k_x_query": k_x_query,
            "environment_influence": best["environment_influence"],
            "action_kernel": best["action_kernel"],
            "omega": best["omega"],
        }
        return result

    def evaluate(self, task: AUFTask, n_samples: int) -> Mapping[str, Any]:
        if self.last_decision_ is None or self.last_context_ is None:
            raise RuntimeError("suggest must be called before evaluate.")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")

        context = self.last_context_
        empirical_hard = self._weighted_hard_success_probability(
            context["k_x_query"],
            context["environment_influence"],
            context["action_kernel"],
        )
        empirical_smooth = self._weighted_smooth_success_probability(
            context["k_x_query"],
            context["environment_influence"],
            context["action_kernel"],
        )
        return {
            "estimated_success_probability": float(self.last_decision_.estimated_success_probability),
            "empirical_hard_success_rate": empirical_hard,
            "empirical_surrogate_success_rate": empirical_smooth,
            "n_samples": int(n_samples),
            "alterations": dict(self.last_decision_.alterations),
        }

    def _matrix_for(self, variables: Sequence[str]) -> np.ndarray:
        if self.columns_ is None:
            raise RuntimeError("fit must be called before accessing matrices.")
        if not variables:
            row_count = 0 if self.X_hist_ is None else self.X_hist_.shape[0]
            return np.zeros((row_count, 0), dtype=float)
        missing = [name for name in variables if name not in self.column_index_]
        if missing:
            raise ValueError(f"Training data is missing variables: {missing}.")
        matrix = np.asarray(
            [self._data_column(self.column_index_[name]) for name in variables],
            dtype=float,
        ).T
        return matrix

    def _data_column(self, index: int) -> np.ndarray:
        if self.columns_ is None:
            raise RuntimeError("fit must be called before accessing data columns.")
        if not hasattr(self, "_data_matrix"):
            raise RuntimeError("fit must be called before accessing data columns.")
        return self._data_matrix[:, index]

    def _infer_environment_variables(self, task: AUFTask) -> tuple[str, ...]:
        explicit = task.metadata.get("cme_environment_variables")
        if explicit is not None:
            return tuple(str(name) for name in explicit)
        if self.columns_ is None:
            return ()
        excluded = set(task.observed) | set(task.alterable) | set(task.outcomes)
        ordered = tuple(task.variable_order or self.columns_)
        return tuple(name for name in ordered if name in self.column_index_ and name not in excluded)

    def _environment_for_candidate(
        self,
        task: AUFTask,
        candidate: Sequence[str],
        environment_variables: Sequence[str],
    ) -> tuple[str, ...]:
        if not environment_variables:
            return ()
        ordered = tuple(task.variable_order or self.columns_ or ())
        index = {name: idx for idx, name in enumerate(ordered)}
        candidate_indices = [index[name] for name in candidate if name in index]
        if not candidate_indices:
            return tuple(name for name in environment_variables if name in self.column_index_)
        first_action_index = min(candidate_indices)
        return tuple(
            name
            for name in environment_variables
            if name in self.column_index_ and index.get(name, first_action_index + 1) < first_action_index
        )

    def _resolve_bandwidth(self, kind: str, values: np.ndarray) -> float:
        source = self.bandwidths
        if isinstance(source, Mapping):
            value = source.get(kind)
            if value is None or value == "auto":
                return median_or_mean_bandwidth(values, method=self.bandwidth_method)
            return _positive_float(value, f"bandwidths[{kind!r}]")
        if source == "auto":
            return median_or_mean_bandwidth(values, method=self.bandwidth_method)
        return _positive_float(source, "bandwidths")

    def _weighted_hard_success_probability(
        self,
        k_x_query: np.ndarray,
        environment_influence: np.ndarray,
        action_kernel: np.ndarray,
    ) -> float:
        if self.hard_success_ is None:
            return 0.0
        weights = _nonnegative_history_weights(k_x_query, environment_influence, action_kernel)
        if float(np.sum(weights)) <= 1e-15:
            return float(np.mean(self.hard_success_))
        return float(np.clip(np.dot(weights, self.hard_success_) / np.sum(weights), 0.0, 1.0))

    def _weighted_smooth_success_probability(
        self,
        k_x_query: np.ndarray,
        environment_influence: np.ndarray,
        action_kernel: np.ndarray,
    ) -> float:
        if self.W_vec_ is None:
            return 0.0
        weights = _nonnegative_history_weights(k_x_query, environment_influence, action_kernel)
        if float(np.sum(weights)) <= 1e-15:
            return float(np.mean(self.W_vec_))
        return float(np.clip(np.dot(weights, self.W_vec_) / np.sum(weights), 0.0, 1.0))


def _default_columns(
    data: Mapping[str, Sequence[float]] | np.ndarray,
    task: AUFTask,
    config: Mapping[str, Any],
) -> tuple[str, ...]:
    if isinstance(data, Mapping):
        if task.variable_order is not None:
            ordered = list(task.variable_order)
            ordered.extend(name for name in data if name not in ordered)
            return tuple(ordered)
        return tuple(data)
    columns = config.get("columns", task.variable_order)
    if columns is None:
        raise ValueError("Array data requires config['columns'] or task.variable_order.")
    return tuple(columns)


def _regularized_solve(kernel: np.ndarray, rhs: np.ndarray, regularization: float) -> tuple[np.ndarray, str]:
    matrix = np.asarray(kernel, dtype=float)
    vector = np.asarray(rhs, dtype=float).reshape(-1)
    if matrix.shape != (vector.size, vector.size):
        raise ValueError("kernel must be square and match rhs length.")
    reg = float(regularization)
    if reg < 0.0 or not np.isfinite(reg):
        raise ValueError("regularization must be finite and non-negative.")
    system = matrix + vector.size * reg * np.eye(vector.size)
    try:
        return np.linalg.solve(system, vector), "solve"
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(system, vector, rcond=None)[0], "lstsq"


def _candidate_rng(seed: int | None, candidate_index: int) -> np.random.Generator:
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(int(seed) + 104729 * (candidate_index + 1))


def _is_better_candidate(
    objective_value: float,
    estimated_success_probability: float,
    cost: float,
    best: Mapping[str, Any] | None,
) -> bool:
    if best is None:
        return True
    current = best["optimized"]
    if objective_value > current.objective_value + 1e-15:
        return True
    if abs(objective_value - current.objective_value) <= 1e-15:
        if estimated_success_probability > current.estimated_success_probability + 1e-15:
            return True
        if abs(estimated_success_probability - current.estimated_success_probability) <= 1e-15:
            return cost < float(best["cost"])
    return False


def _nonnegative_history_weights(
    k_x_query: np.ndarray,
    environment_influence: np.ndarray,
    action_kernel: np.ndarray,
) -> np.ndarray:
    weights = (
        np.asarray(k_x_query, dtype=float).reshape(-1)
        * np.maximum(np.asarray(environment_influence, dtype=float).reshape(-1), 0.0)
        * np.asarray(action_kernel, dtype=float).reshape(-1)
    )
    if float(np.sum(weights)) <= 1e-15:
        weights = np.asarray(k_x_query, dtype=float).reshape(-1) * np.asarray(action_kernel, dtype=float).reshape(-1)
    return np.maximum(weights, 0.0)


def _positive_float(value: float | str, label: str) -> float:
    number = float(value)
    if number <= 0.0 or not np.isfinite(number):
        raise ValueError(f"{label} must be finite and positive.")
    return number


UnpublishedCMERehearsal = CMERehearsal
