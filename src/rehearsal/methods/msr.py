"""IJCAI 2025 MSR adapter.

Multi-Step Rehearsal, MSR.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, MutableMapping, Sequence

import numpy as np

from rehearsal.core import AUFTask, DecisionResult
from rehearsal.core.data import candidate_alteration_sets
from rehearsal.datasets.sem import RehearsalDatasetSpec, simulate_sem
from rehearsal.models.nonlinear import NonlinearSRMLearner, NonlinearStructuralModel
from rehearsal.optimizers.grad_rh import desired_region_center_and_radius

MissingObservationSampler = Callable[
    [Mapping[str, float], Sequence[str], np.random.Generator, int],
    Mapping[str, float],
]


@dataclass
class _LinearGaussianState:
    variable_order: tuple[str, ...]
    lambda_matrix: np.ndarray
    eta: np.ndarray
    omega: np.ndarray
    min_variance: float = 1e-8

    @property
    def index(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(self.variable_order)}

    def copy(self) -> "_LinearGaussianState":
        return copy.deepcopy(self)

    def mean_covariance(self) -> tuple[np.ndarray, np.ndarray]:
        identity = np.eye(len(self.variable_order), dtype=float)
        try:
            structural_inverse = np.linalg.inv(identity - self.lambda_matrix)
        except np.linalg.LinAlgError:
            structural_inverse = np.linalg.pinv(identity - self.lambda_matrix)
        mean = structural_inverse @ self.eta
        covariance = structural_inverse @ self.omega @ structural_inverse.T
        return mean.reshape(-1), _stabilize_covariance(covariance, self.min_variance)

    def alter(self, variable: str, value: float) -> None:
        idx = self.index[variable]
        self.lambda_matrix[idx, :] = 0.0
        self.eta[idx] = float(value)
        self.omega[idx, :] = 0.0
        self.omega[:, idx] = 0.0

    def observe(self, variable: str, value: float) -> None:
        idx = self.index[variable]
        identity = np.eye(len(self.variable_order), dtype=float)
        try:
            structural_inverse = np.linalg.inv(identity - self.lambda_matrix)
        except np.linalg.LinAlgError:
            structural_inverse = np.linalg.pinv(identity - self.lambda_matrix)
        mean = structural_inverse @ self.eta
        covariance = structural_inverse @ self.omega @ structural_inverse.T
        variance = float(covariance[idx, idx])
        if variance <= self.min_variance:
            self.eta[idx] = float(value)
            return
        gain_left = self.omega @ structural_inverse.T[:, idx]
        gain_right = structural_inverse[(idx,), :] @ self.omega
        self.eta = self.eta + gain_left / variance * (float(value) - float(mean[idx]))
        self.omega = self.omega - gain_left.reshape(-1, 1) @ gain_right / variance
        self.omega = _stabilize_covariance(self.omega, self.min_variance, preserve_zeros=True)

    def outcome_moments(self, task: AUFTask) -> tuple[np.ndarray, np.ndarray]:
        mean, covariance = self.mean_covariance()
        indices = [self.index[name] for name in task.outcomes]
        return mean[indices], covariance[np.ix_(indices, indices)]

    def sample_outcomes(
        self,
        task: AUFTask,
        standard_normals: np.ndarray,
    ) -> np.ndarray:
        mean, covariance = self.outcome_moments(task)
        normals = np.asarray(standard_normals, dtype=float)
        if normals.ndim != 2 or normals.shape[1] != len(task.outcomes):
            raise ValueError("standard_normals must have shape (n_samples, n_outcomes).")
        transform = _covariance_square_root(covariance, self.min_variance)
        return normals @ transform.T + mean.reshape(1, -1)


@dataclass(frozen=True)
class _MSROptimizationResult:
    action: np.ndarray
    objective_value: float
    estimated_success_probability: float
    solver_status: str
    diagnostics: Mapping[str, Any]


class MSRRehearsal:
    """Multi-Step Rehearsal with closed-form Linear-Gaussian updates.

    Variables listed in ``stage_observables`` but absent from the ``observation``
    mapping passed to :meth:`suggest` are imputed when ``impute_missing_observations``
    is true. By default imputation uses the fitted linear-Gaussian predictive mean.
    Alternatively, pass ``dataset_spec`` to draw missing coordinates jointly from
    :func:`rehearsal.datasets.sem.simulate_sem`, or pass ``missing_observation_sampler``
    for custom fills (which takes precedence over ``dataset_spec``).
    """

    def __init__(
        self,
        *,
        seed: int | None = None,
        predictor_type: str = "linear",
        srm_type: str | None = None,
        feature_degree: int = 1,
        stages: Sequence[Sequence[str]] | None = None,
        stage_observables: Sequence[Sequence[str]] | None = None,
        candidate_alteration_sets: Sequence[Sequence[str]] | None = None,
        n_mc_samples: int = 256,
        learning_rate: float = 0.05,
        epochs: int = 80,
        patience: int = 20,
        loss: str = "center_mae",
        num_restarts: int = 3,
        finite_difference_eps: float = 1e-4,
        impute_missing_observations: bool = True,
        lookahead: bool = True,
        dataset_spec: RehearsalDatasetSpec | None = None,
        missing_observation_sampler: MissingObservationSampler | None = None,
    ) -> None:
        self.seed = seed
        self.predictor_type = str(srm_type or predictor_type)
        self.feature_degree = int(feature_degree)
        self.stages = tuple(tuple(stage) for stage in stages) if stages is not None else None
        self.stage_observables = (
            tuple(tuple(stage) for stage in stage_observables)
            if stage_observables is not None
            else None
        )
        self.candidate_alteration_sets = (
            tuple(tuple(candidate) for candidate in candidate_alteration_sets)
            if candidate_alteration_sets is not None
            else None
        )
        self.n_mc_samples = int(n_mc_samples)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.loss = str(loss)
        self.num_restarts = int(num_restarts)
        self.finite_difference_eps = float(finite_difference_eps)
        self.impute_missing_observations = bool(impute_missing_observations)
        self.lookahead = bool(lookahead)
        self.dataset_spec = dataset_spec
        self.missing_observation_sampler = missing_observation_sampler
        self.model_: NonlinearStructuralModel | None = None
        self.base_state_: _LinearGaussianState | None = None
        self.last_state_: _LinearGaussianState | None = None
        self.fit_diagnostics_: dict[str, Any] = {}
        self.last_decision_: DecisionResult | None = None
        self.last_observation_: dict[str, float] | None = None

    def fit(
        self,
        data: Any,
        task: AUFTask,
        config: Mapping[str, Any] | None = None,
    ) -> "MSRRehearsal":
        start = time.perf_counter()
        config_dict = dict(config or {})
        predictor_type = str(config_dict.get("predictor_type", config_dict.get("srm_type", self.predictor_type)))
        feature_degree = int(config_dict.get("feature_degree", self.feature_degree))
        if predictor_type != "linear" or feature_degree != 1:
            raise ValueError("MSR closed-form updates require predictor_type='linear' and feature_degree=1.")
        learner = NonlinearSRMLearner(
            predictor_type=predictor_type,
            feature_degree=feature_degree,
            ridge=float(config_dict.get("ridge", 1e-6)),
            min_variance=float(config_dict.get("min_variance", 1e-8)),
        )
        result = learner.fit(data, task, config_dict)
        self.model_ = result.model  # type: ignore[assignment]
        self.base_state_ = _state_from_nonlinear_model(
            self.model_,
            min_variance=float(config_dict.get("min_variance", 1e-8)),
        )
        self.fit_diagnostics_ = {
            **dict(result.diagnostics),
            "runtime_seconds": time.perf_counter() - start,
        }
        return self

    def suggest(self, observation: Mapping[str, float], task: AUFTask) -> DecisionResult:
        if self.model_ is None or self.base_state_ is None:
            raise RuntimeError("MSRRehearsal.fit must be called before suggest.")
        start = time.perf_counter()
        state = self.base_state_.copy()
        stages = _resolve_stages(task, self.stages)
        stage_observables = _resolve_stage_observables(task, stages, self.stage_observables)
        candidates = candidate_alteration_sets(task, self.candidate_alteration_sets)
        conditioning: dict[str, float] = {name: float(value) for name, value in observation.items()}
        initial_provided = frozenset(conditioning.keys())
        truth_impute = self.dataset_spec is not None or self.missing_observation_sampler is not None
        alterations: dict[str, float] = {}
        stage_diagnostics = []
        imputed_observations: dict[str, float] = {}

        for stage_idx, stage in enumerate(stages):
            observables = stage_observables[stage_idx]
            rng_truth = _stage_rng(self.seed, stage_idx, 90_000)
            truth_fill_labels: dict[str, str] = {}
            if self.impute_missing_observations and truth_impute:
                truth_fill_labels = _resolve_truth_missing_for_observables(
                    conditioning,
                    observables,
                    dataset_spec=self.dataset_spec,
                    missing_sampler=self.missing_observation_sampler,
                    stage_idx=stage_idx,
                    rng=rng_truth,
                )

            stage_set = set(stage)
            stage_candidates = tuple(candidate for candidate in candidates if set(candidate).issubset(stage_set))
            before_probability = _estimate_success_probability(
                state,
                task,
                self.n_mc_samples,
                rng=_stage_rng(self.seed, stage_idx, 0),
            )
            selected: dict[str, Any] | None = None
            candidate_diagnostics = []
            for candidate_idx, candidate in enumerate(stage_candidates):
                lower, upper = task.alteration_domain.arrays_for(candidate)
                optimized = _optimize_stage_alterations(
                    state,
                    task,
                    candidate,
                    lower=lower,
                    upper=upper,
                    n_samples=self.n_mc_samples,
                    learning_rate=self.learning_rate,
                    epochs=self.epochs,
                    patience=self.patience,
                    loss=self.loss,
                    num_restarts=self.num_restarts,
                    finite_difference_eps=self.finite_difference_eps,
                    rng=_stage_rng(self.seed, stage_idx, candidate_idx + 1),
                )
                candidate_alterations = {name: float(value) for name, value in zip(candidate, optimized.action)}
                candidate_cost = task.alteration_domain.cost(candidate_alterations)
                diag = {
                    "candidate": tuple(candidate),
                    "estimated_success_probability": optimized.estimated_success_probability,
                    "objective_value": optimized.objective_value,
                    "solver_status": optimized.solver_status,
                    "cost": candidate_cost,
                    **dict(optimized.diagnostics),
                }
                candidate_diagnostics.append(diag)
                if selected is None or _better_stage_candidate(
                    optimized.estimated_success_probability,
                    optimized.objective_value,
                    candidate_cost,
                    selected,
                ):
                    selected = {
                        "candidate": tuple(candidate),
                        "alterations": candidate_alterations,
                        "cost": candidate_cost,
                        "optimized": optimized,
                    }

            accepted = False
            lookahead_diagnostics = {}
            if selected is not None:
                optimized = selected["optimized"]
                if float(optimized.estimated_success_probability) > before_probability + 1e-12:
                    accepted = True
                    if self.lookahead:
                        baseline_state = state.copy()
                        selected_state = state.copy()
                        for variable, value in selected["alterations"].items():
                            selected_state.alter(variable, value)
                        if truth_impute:
                            _observe_values_from_conditioning(
                                baseline_state,
                                observables,
                                conditioning,
                            )
                            _observe_values_from_conditioning(
                                selected_state,
                                observables,
                                conditioning,
                            )
                        else:
                            _apply_predictive_stage_observations(
                                baseline_state,
                                observables,
                                dict(conditioning),
                                initial_provided=initial_provided,
                                impute_missing=self.impute_missing_observations,
                            )
                            _apply_predictive_stage_observations(
                                selected_state,
                                observables,
                                dict(conditioning),
                                initial_provided=initial_provided,
                                impute_missing=self.impute_missing_observations,
                            )
                        baseline_rollout = _rollout_future_success_probability(
                            baseline_state,
                            task,
                            stages,
                            stage_observables,
                            candidates,
                            dict(conditioning),
                            initial_provided=initial_provided,
                            dataset_spec=self.dataset_spec,
                            missing_sampler=self.missing_observation_sampler,
                            start_stage=stage_idx + 1,
                            n_samples=self.n_mc_samples,
                            learning_rate=self.learning_rate,
                            epochs=self.epochs,
                            patience=self.patience,
                            loss=self.loss,
                            num_restarts=self.num_restarts,
                            finite_difference_eps=self.finite_difference_eps,
                            impute_missing=self.impute_missing_observations,
                            seed=self.seed,
                        )
                        selected_rollout = _rollout_future_success_probability(
                            selected_state,
                            task,
                            stages,
                            stage_observables,
                            candidates,
                            dict(conditioning),
                            initial_provided=initial_provided,
                            dataset_spec=self.dataset_spec,
                            missing_sampler=self.missing_observation_sampler,
                            start_stage=stage_idx + 1,
                            n_samples=self.n_mc_samples,
                            learning_rate=self.learning_rate,
                            epochs=self.epochs,
                            patience=self.patience,
                            loss=self.loss,
                            num_restarts=self.num_restarts,
                            finite_difference_eps=self.finite_difference_eps,
                            impute_missing=self.impute_missing_observations,
                            seed=None if self.seed is None else int(self.seed) + 65_537,
                        )
                        lookahead_diagnostics = {
                            "baseline_rollout_probability": baseline_rollout,
                            "selected_rollout_probability": selected_rollout,
                        }
                        accepted = selected_rollout >= baseline_rollout - 1e-12
                if accepted:
                    alterations.update(selected["alterations"])
                    for variable, value in selected["alterations"].items():
                        state.alter(variable, value)

            if truth_impute:
                _observe_values_from_conditioning(state, observables, conditioning)
                observed_values = _truth_stage_observation_metadata(
                    observables,
                    conditioning,
                    initial_provided=initial_provided,
                    truth_fill_labels=truth_fill_labels,
                )
            else:
                observed_values = _apply_predictive_stage_observations(
                    state,
                    observables,
                    conditioning,
                    initial_provided=initial_provided,
                    impute_missing=self.impute_missing_observations,
                )

            for variable, payload in observed_values.items():
                if payload["source"] in ("predictive_mean", "ground_truth_sample", "sampler"):
                    imputed_observations[variable] = float(payload["value"])

            after_probability = _estimate_success_probability(
                state,
                task,
                self.n_mc_samples,
                rng=_stage_rng(self.seed, stage_idx, 10_000),
            )
            stage_diagnostics.append(
                {
                    "stage_index": stage_idx,
                    "variables": tuple(stage),
                    "candidate_diagnostics": candidate_diagnostics,
                    "selected_candidate": None if selected is None else selected["candidate"],
                    "accepted": accepted,
                    "probability_before": before_probability,
                    "probability_after": after_probability,
                    "observations": observed_values,
                    **lookahead_diagnostics,
                }
            )

        final_probability = _estimate_success_probability(
            state,
            task,
            self.n_mc_samples,
            rng=_stage_rng(self.seed, len(stages), 20_000),
        )
        runtime = time.perf_counter() - start
        diagnostics = {
            "method_family": "IJCAI 2025 MSR",
            "method_name": "Multi-Step Rehearsal",
            "stages": stages,
            "stage_diagnostics": stage_diagnostics,
            "imputed_observations": imputed_observations,
            "lookahead": self.lookahead,
            "truth_observation_imputation": truth_impute,
            **self.fit_diagnostics_,
        }
        result = DecisionResult(
            alterations=alterations,
            estimated_success_probability=float(np.clip(final_probability, 0.0, 1.0)),
            cost=float(task.alteration_domain.cost(alterations)),
            diagnostics=diagnostics,
            runtime_seconds=runtime,
        )
        self.last_state_ = state
        self.last_decision_ = result
        self.last_observation_ = conditioning
        return result

    def evaluate(self, task: AUFTask, n_samples: int) -> Mapping[str, Any]:
        if self.last_state_ is None or self.last_decision_ is None:
            raise RuntimeError("fit and suggest must be called before evaluate.")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        rng = np.random.default_rng(None if self.seed is None else int(self.seed) + 104729)
        probability = _estimate_success_probability(self.last_state_, task, int(n_samples), rng=rng)
        return {
            "estimated_success_probability": float(probability),
            "n_samples": int(n_samples),
            "alterations": dict(self.last_decision_.alterations),
        }


IJCAI2025MSRRehearsal = MSRRehearsal


def _state_from_nonlinear_model(
    model: NonlinearStructuralModel,
    *,
    min_variance: float,
) -> _LinearGaussianState:
    variables = tuple(model.variable_order)
    index = {name: idx for idx, name in enumerate(variables)}
    n_variables = len(variables)
    lambda_matrix = np.zeros((n_variables, n_variables), dtype=float)
    eta = np.zeros(n_variables, dtype=float)
    residual_columns = []

    for child in variables:
        predictor = model.predictors.get(child)
        if predictor is None:
            residual_columns.append(np.zeros(0, dtype=float))
            continue
        coefficients = np.asarray(predictor.coefficients, dtype=float).reshape(-1)
        expected_size = 1 + len(predictor.parents)
        if coefficients.size < expected_size:
            raise ValueError(f"Linear predictor for {child!r} has too few coefficients.")
        eta[index[child]] = float(coefficients[0])
        for parent_idx, parent in enumerate(predictor.parents):
            lambda_matrix[index[child], index[parent]] = float(coefficients[parent_idx + 1])
        residual_columns.append(np.asarray(predictor.residuals, dtype=float).reshape(-1))

    residual_length = max((column.size for column in residual_columns), default=0)
    residual_matrix = np.zeros((residual_length, n_variables), dtype=float)
    for variable_idx, residuals in enumerate(residual_columns):
        if residuals.size == residual_length:
            residual_matrix[:, variable_idx] = residuals
        elif residuals.size:
            raise ValueError("All predictor residual vectors must have the same length.")
    omega = np.cov(residual_matrix, rowvar=False, bias=True)
    omega = np.atleast_2d(np.asarray(omega, dtype=float))
    omega = _stabilize_covariance(omega, min_variance)
    return _LinearGaussianState(variables, lambda_matrix, eta, omega, min_variance=min_variance)


def _resolve_stages(task: AUFTask, override: Sequence[Sequence[str]] | None) -> tuple[tuple[str, ...], ...]:
    variables = set(task.all_variables())
    if override is not None:
        stages = tuple(tuple(stage) for stage in override)
    else:
        stages = _fallback_stages(task)
    for stage in stages:
        missing = set(stage) - variables
        if missing:
            raise ValueError(f"MSR stage contains variables missing from task: {sorted(missing)}.")
    return stages


def _fallback_stages(task: AUFTask) -> tuple[tuple[str, ...], ...]:
    stages = []
    seen: set[str] = set()
    for raw_stage in (task.observed, task.alterable, task.outcomes):
        stage = tuple(name for name in raw_stage if name not in seen)
        if stage:
            stages.append(stage)
            seen.update(stage)
    return tuple(stages)


def _resolve_stage_observables(
    task: AUFTask,
    stages: Sequence[Sequence[str]],
    override: Sequence[Sequence[str]] | None,
) -> tuple[tuple[str, ...], ...]:
    if override is not None:
        if len(override) != len(stages):
            raise ValueError("stage_observables must have the same length as stages.")
        stage_observables = tuple(tuple(values) for values in override)
    else:
        observed = set(task.observed)
        stage_observables_list = []
        for stage in stages:
            current = [name for name in stage if name in observed]
            stage_observables_list.append(tuple(dict.fromkeys(current)))
        stage_observables = tuple(stage_observables_list)
    variables = set(task.all_variables())
    for values in stage_observables:
        missing = set(values) - variables
        if missing:
            raise ValueError(f"MSR stage observables are missing from task: {sorted(missing)}.")
    return stage_observables


def _sample_missing_from_true_sem(
    spec: RehearsalDatasetSpec,
    conditioning: Mapping[str, float],
    missing: Sequence[str],
    rng: np.random.Generator,
) -> dict[str, float]:
    """Joint sample of missing coordinates from the dataset SEM given conditioning values."""

    if not missing:
        return {}
    sim = simulate_sem(spec, 1, rng=rng, observation=dict(conditioning))
    return {name: float(sim[name][0]) for name in missing}


def _resolve_truth_missing_for_observables(
    conditioning: MutableMapping[str, float],
    observables: Sequence[str],
    *,
    dataset_spec: RehearsalDatasetSpec | None,
    missing_sampler: MissingObservationSampler | None,
    stage_idx: int,
    rng: np.random.Generator,
) -> dict[str, str]:
    """Fill missing observables from ``simulate_sem`` or a user sampler; returns var -> source tag."""

    missing = [v for v in observables if v not in conditioning]
    if not missing:
        return {}
    if missing_sampler is not None:
        fills = dict(missing_sampler(dict(conditioning), tuple(missing), rng, int(stage_idx)))
        source_tag = "sampler"
    elif dataset_spec is not None:
        fills = _sample_missing_from_true_sem(dataset_spec, conditioning, missing, rng)
        source_tag = "ground_truth_sample"
    else:
        return {}
    out: dict[str, str] = {}
    for name in missing:
        if name not in fills:
            raise ValueError(
                f"Missing-observation fill must provide a value for {name!r}; "
                f"got keys {sorted(fills)}."
            )
        conditioning[name] = float(fills[name])
        out[name] = source_tag
    return out


def _observe_values_from_conditioning(
    state: _LinearGaussianState,
    observables: Sequence[str],
    conditioning: Mapping[str, float],
) -> None:
    for variable in observables:
        if variable in conditioning:
            state.observe(variable, float(conditioning[variable]))


def _apply_predictive_stage_observations(
    state: _LinearGaussianState,
    observables: Sequence[str],
    conditioning: MutableMapping[str, float],
    *,
    initial_provided: frozenset[str],
    impute_missing: bool,
) -> dict[str, dict[str, float | str]]:
    """Impute missing coordinates from the fitted LG state's mean; updates ``conditioning`` in place."""

    observed_values: dict[str, dict[str, float | str]] = {}
    for variable in observables:
        if variable in conditioning:
            value = float(conditioning[variable])
            if variable in initial_provided:
                source = "provided"
            else:
                source = "conditioning"
        elif impute_missing:
            mean, _ = state.mean_covariance()
            value = float(mean[state.index[variable]])
            conditioning[variable] = value
            source = "predictive_mean"
        else:
            continue
        state.observe(variable, value)
        observed_values[variable] = {"value": value, "source": source}
    return observed_values


def _truth_stage_observation_metadata(
    observables: Sequence[str],
    conditioning: Mapping[str, float],
    *,
    initial_provided: frozenset[str],
    truth_fill_labels: Mapping[str, str],
) -> dict[str, dict[str, float | str]]:
    observed_values: dict[str, dict[str, float | str]] = {}
    for variable in observables:
        if variable not in conditioning:
            continue
        value = float(conditioning[variable])
        if variable in initial_provided:
            source: str = "provided"
        elif variable in truth_fill_labels:
            source = str(truth_fill_labels[variable])
        else:
            source = "conditioning"
        observed_values[variable] = {"value": value, "source": source}
    return observed_values


def _rollout_future_success_probability(
    state: _LinearGaussianState,
    task: AUFTask,
    stages: Sequence[Sequence[str]],
    stage_observables: Sequence[Sequence[str]],
    candidates: Sequence[Sequence[str]],
    conditioning: MutableMapping[str, float],
    *,
    initial_provided: frozenset[str],
    dataset_spec: RehearsalDatasetSpec | None,
    missing_sampler: MissingObservationSampler | None,
    start_stage: int,
    n_samples: int,
    learning_rate: float,
    epochs: int,
    patience: int,
    loss: str,
    num_restarts: int,
    finite_difference_eps: float,
    impute_missing: bool,
    seed: int | None,
) -> float:
    trial_state = state.copy()
    truth_impute = dataset_spec is not None or missing_sampler is not None
    for stage_idx in range(int(start_stage), len(stages)):
        stage_set = set(stages[stage_idx])
        stage_candidates = tuple(candidate for candidate in candidates if set(candidate).issubset(stage_set))
        before_probability = _estimate_success_probability(
            trial_state,
            task,
            int(n_samples),
            rng=_stage_rng(seed, stage_idx, 30_000),
        )
        selected: dict[str, Any] | None = None
        for candidate_idx, candidate in enumerate(stage_candidates):
            lower, upper = task.alteration_domain.arrays_for(candidate)
            optimized = _optimize_stage_alterations(
                trial_state,
                task,
                candidate,
                lower=lower,
                upper=upper,
                n_samples=int(n_samples),
                learning_rate=learning_rate,
                epochs=epochs,
                patience=patience,
                loss=loss,
                num_restarts=num_restarts,
                finite_difference_eps=finite_difference_eps,
                rng=_stage_rng(seed, stage_idx, 40_000 + candidate_idx),
            )
            candidate_alterations = {name: float(value) for name, value in zip(candidate, optimized.action)}
            candidate_cost = task.alteration_domain.cost(candidate_alterations)
            if selected is None or _better_stage_candidate(
                optimized.estimated_success_probability,
                optimized.objective_value,
                candidate_cost,
                selected,
            ):
                selected = {
                    "candidate": tuple(candidate),
                    "alterations": candidate_alterations,
                    "cost": candidate_cost,
                    "optimized": optimized,
                }
        if selected is not None:
            optimized = selected["optimized"]
            if float(optimized.estimated_success_probability) > before_probability + 1e-12:
                for variable, value in selected["alterations"].items():
                    trial_state.alter(variable, value)
        observables = stage_observables[stage_idx]
        rng_imp = _stage_rng(seed, stage_idx, 95_000)
        if impute_missing and truth_impute:
            _resolve_truth_missing_for_observables(
                conditioning,
                observables,
                dataset_spec=dataset_spec,
                missing_sampler=missing_sampler,
                stage_idx=stage_idx,
                rng=rng_imp,
            )
        if truth_impute:
            _observe_values_from_conditioning(trial_state, observables, conditioning)
        else:
            _apply_predictive_stage_observations(
                trial_state,
                observables,
                conditioning,
                initial_provided=initial_provided,
                impute_missing=impute_missing,
            )
    return _estimate_success_probability(
        trial_state,
        task,
        int(n_samples),
        rng=_stage_rng(seed, len(stages), 50_000),
    )


def _optimize_stage_alterations(
    state: _LinearGaussianState,
    task: AUFTask,
    candidate: Sequence[str],
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    n_samples: int,
    learning_rate: float,
    epochs: int,
    patience: int,
    loss: str,
    num_restarts: int,
    finite_difference_eps: float,
    rng: np.random.Generator,
) -> _MSROptimizationResult:
    candidate = tuple(candidate)
    lower = np.asarray(lower, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)
    if lower.shape != upper.shape or lower.size != len(candidate):
        raise ValueError("Bounds must match candidate length.")
    if np.any(lower > upper):
        raise ValueError("Lower alteration bounds must not exceed upper bounds.")
    center, radius = desired_region_center_and_radius(task.desired_region)
    standard_normals = rng.normal(size=(int(n_samples), len(task.outcomes)))

    def action_from_raw(raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=float).reshape(-1)
        return 0.5 * (np.tanh(raw) + 1.0) * (upper - lower) + lower

    def objective_for_action(action: np.ndarray) -> tuple[float, float]:
        trial_state = state.copy()
        for name, value in zip(candidate, action):
            trial_state.alter(name, float(value))
        outcomes = trial_state.sample_outcomes(task, standard_normals)
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
            if best is None or _better_optimizer_point(obj, prob, action, best):
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
                plus = raw.copy()
                plus[j] += eps
                minus = raw.copy()
                minus[j] -= eps
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
        if best_local is not None and (best is None or _better_optimizer_point(
            best_local["objective_value"],
            best_local["probability"],
            best_local["action"],
            best,
        )):
            best = best_local

    if best is None:
        raise RuntimeError("MSR optimizer failed to evaluate any candidate action.")
    return _MSROptimizationResult(
        action=np.asarray(best["action"], dtype=float),
        objective_value=float(best["objective_value"]),
        estimated_success_probability=float(np.clip(best["probability"], 0.0, 1.0)),
        solver_status="linear_gaussian_finite_difference_adam",
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


def _estimate_success_probability(
    state: _LinearGaussianState,
    task: AUFTask,
    n_samples: int,
    *,
    rng: np.random.Generator,
) -> float:
    standard_normals = rng.normal(size=(int(n_samples), len(task.outcomes)))
    outcomes = state.sample_outcomes(task, standard_normals)
    return float(np.mean(task.desired_region.contains(outcomes)))


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


def _better_stage_candidate(probability: float, objective: float, cost: float, best: Mapping[str, Any]) -> bool:
    best_opt = best["optimized"]
    if probability > float(best_opt.estimated_success_probability) + 1e-12:
        return True
    if abs(probability - float(best_opt.estimated_success_probability)) <= 1e-12:
        if objective < float(best_opt.objective_value) - 1e-12:
            return True
        if abs(objective - float(best_opt.objective_value)) <= 1e-12 and cost < float(best["cost"]):
            return True
    return False


def _better_optimizer_point(obj: float, prob: float, action: np.ndarray, best: Mapping[str, Any]) -> bool:
    if prob > float(best["probability"]) + 1e-12:
        return True
    if abs(prob - float(best["probability"])) <= 1e-12 and obj < float(best["objective_value"]) - 1e-12:
        return True
    if abs(prob - float(best["probability"])) <= 1e-12 and abs(obj - float(best["objective_value"])) <= 1e-12:
        return float(np.sum(np.abs(action))) < float(np.sum(np.abs(best["action"])))
    return False


def _stage_rng(seed: int | None, stage_index: int, offset: int) -> np.random.Generator:
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(int(seed) + 3571 * (stage_index + 1) + int(offset))


def _covariance_square_root(covariance: np.ndarray, min_variance: float) -> np.ndarray:
    stable = _stabilize_covariance(covariance, min_variance)
    try:
        return np.linalg.cholesky(stable)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(stable)
        eigvals = np.clip(eigvals, min_variance, None)
        return eigvecs @ np.diag(np.sqrt(eigvals))


def _stabilize_covariance(
    covariance: np.ndarray,
    min_variance: float,
    *,
    preserve_zeros: bool = False,
) -> np.ndarray:
    matrix = np.asarray(covariance, dtype=float)
    matrix = 0.5 * (matrix + matrix.T)
    eigvals = np.linalg.eigvalsh(matrix)
    min_eig = float(np.min(eigvals)) if eigvals.size else 0.0
    if min_eig < -float(min_variance):
        matrix = matrix + (abs(min_eig) + float(min_variance)) * np.eye(matrix.shape[0])
    if not preserve_zeros:
        matrix = matrix + float(min_variance) * np.eye(matrix.shape[0])
    return 0.5 * (matrix + matrix.T)
