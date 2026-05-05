"""NeurIPS 2025 MUR rehearsal adapter."""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import numpy as np

from rehearsal.core import AUFTask, DecisionResult
from rehearsal.core.data import candidate_alteration_sets as normalize_candidate_sets
from rehearsal.core.data import coerce_data_matrix
from rehearsal.metrics.mur import infer_mur_region_center
from rehearsal.models import StructuralLearner, StructuralLearningResult
from rehearsal.models.time_series import LinearTimeSeriesSRM, LinearTimeSeriesSRMLearner
from rehearsal.optimizers.mur import rollout_mur_policy, select_mur_action


class MURRehearsal:
    """GMuR/FarMuR adapter backed by a linear additive time-series SRM."""

    def __init__(
        self,
        *,
        variant: str = "gmur",
        horizon: int | None = 0,
        window_length: int | None = None,
        candidate_alteration_sets: Sequence[Sequence[str]] | None = None,
        n_mc_samples: int = 256,
        bounded_solver: str = "projected-gradient",
        learning_rate: float | None = None,
        max_iters: int = 200,
        tolerance: float = 1e-8,
        num_restarts: int = 4,
        previous_state_prefix: str = "prev__",
        seed: int | None = None,
        region_center: Sequence[float] | Mapping[str, float] | None = None,
        structural_learner: StructuralLearner | None = None,
    ) -> None:
        self.variant = _normalize_variant(variant)
        if window_length is not None:
            horizon = int(window_length) - 1
        self.horizon = int(0 if horizon is None else horizon)
        if self.horizon < 0:
            raise ValueError("horizon must be non-negative.")
        solver_name = str(bounded_solver).replace("_", "-").lower()
        if solver_name not in {"projected-gradient", "box-qp", "projected-gradient-box-qp"}:
            raise ValueError("bounded_solver must be 'projected-gradient' or an equivalent box-QP option.")
        self.bounded_solver = solver_name
        self.candidate_alteration_sets = (
            tuple(tuple(candidate) for candidate in candidate_alteration_sets)
            if candidate_alteration_sets is not None
            else None
        )
        self.n_mc_samples = int(n_mc_samples)
        self.learning_rate = None if learning_rate is None else float(learning_rate)
        self.max_iters = int(max_iters)
        self.tolerance = float(tolerance)
        self.num_restarts = int(num_restarts)
        self.previous_state_prefix = str(previous_state_prefix)
        self.seed = seed
        self.region_center = region_center
        self.structural_learner = structural_learner or LinearTimeSeriesSRMLearner()
        self.rng_ = np.random.default_rng(seed)
        self.config_: dict[str, Any] = {}
        self.structural_result_: StructuralLearningResult | None = None
        self.model_: LinearTimeSeriesSRM | None = None
        self.candidates_: tuple[tuple[str, ...], ...] = ()
        self.region_center_: np.ndarray | None = None
        self.region_center_source_: str | None = None
        self.training_matrix_: np.ndarray | None = None
        self.training_columns_: tuple[str, ...] = ()
        self.last_training_state_: np.ndarray | None = None
        self.fit_diagnostics_: dict[str, Any] = {}
        self.last_decision_: DecisionResult | None = None
        self.last_context_: dict[str, Any] | None = None

    def fit(
        self,
        data: Mapping[str, Sequence[float]] | np.ndarray | None,
        task: AUFTask,
        config: Mapping[str, Any] | None = None,
    ) -> "MURRehearsal":
        if task.variable_order is None:
            raise ValueError("MUR requires task.variable_order to define the time-series matrix order.")
        config_dict = dict(config or {})
        if self.region_center is not None and "region_center" not in config_dict:
            config_dict["region_center"] = self.region_center
        self.config_ = config_dict
        self._validate_task_order(task)

        region_center, center_source = infer_mur_region_center(task, config_dict)
        if self.region_center is not None and center_source == "fit_config.region_center":
            center_source = "constructor.region_center"
        self.region_center_ = region_center
        self.region_center_source_ = center_source

        structural_result = self.structural_learner.fit(data, task, config_dict)
        model = structural_result.model
        if not isinstance(model, LinearTimeSeriesSRM):
            raise TypeError("MURRehearsal requires LinearTimeSeriesSRM from LinearTimeSeriesSRMLearner.")
        self.structural_result_ = structural_result
        self.model_ = model

        candidate_override = config_dict.get("candidate_alteration_sets", self.candidate_alteration_sets)
        self.candidates_ = normalize_candidate_sets(task, candidate_override)
        self.training_matrix_, self.training_columns_ = _coerce_optional_training_matrix(data, task, config_dict)
        self.last_training_state_ = (
            self.training_matrix_[-1].copy()
            if self.training_matrix_ is not None and self.training_matrix_.shape[0] > 0
            else np.zeros(model.n_variables, dtype=float)
        )

        natural_radius = float(model.spectral_radius_natural())
        altered_radius = float(model.spectral_radius_altered(tuple(task.observed) + tuple(task.alterable)))
        allow_unstable = bool(config_dict.get("allow_unstable", False))
        max_radius = max(natural_radius, altered_radius)
        if not allow_unstable and max_radius >= 1.0:
            raise ValueError(
                "MUR time-series SRM is non-stationary under the configured process "
                f"(spectral radius {max_radius:.6g} >= 1). Set fit_config['allow_unstable']=True to override."
            )

        self.fit_diagnostics_ = {
            "method_family": "NeurIPS 2025 MUR",
            "variant": self.variant,
            "horizon": self.horizon,
            "n_training_samples": int(0 if self.training_matrix_ is None else self.training_matrix_.shape[0]),
            "stationarity_spectral_radius": natural_radius,
            "stationarity_spectral_radius_altered": altered_radius,
            "allow_unstable": allow_unstable,
            "region_center": region_center.tolist(),
            "region_center_source": center_source,
            "candidate_sets": self.candidates_,
            "structural_learning": dict(structural_result.diagnostics),
        }
        return self

    def suggest(self, observation: Mapping[str, float], task: AUFTask) -> DecisionResult:
        if self.model_ is None or self.region_center_ is None:
            raise RuntimeError("MURRehearsal.fit must be called before suggest.")
        start = time.perf_counter()
        x_t = np.asarray([float(observation[name]) for name in task.observed], dtype=float)
        previous_state, previous_state_source = self._previous_state_from_observation(observation, task)
        remaining_horizon = 0 if self.variant == "gmur" else self.horizon
        selection = select_mur_action(
            self.model_,
            task,
            self.candidates_,
            x_t,
            previous_state,
            self.region_center_,
            remaining_horizon=remaining_horizon,
            total_horizon=self.horizon if self.variant == "farmur" else None,
            rng=self.rng_,
            learning_rate=self.learning_rate,
            max_iters=self.max_iters,
            tolerance=self.tolerance,
            num_restarts=self.num_restarts,
            n_probability_samples=max(self.n_mc_samples, 0),
        )
        runtime = time.perf_counter() - start
        diagnostics = {
            "method_family": "NeurIPS 2025 MUR",
            "variant": self.variant,
            "horizon": self.horizon,
            "remaining_horizon_for_solver": remaining_horizon,
            "selected_candidate": selection.candidate,
            "objective_value": float(selection.objective_value),
            "solver_status": selection.solver_status,
            "n_candidates": len(self.candidates_),
            "n_training_samples": self.fit_diagnostics_.get("n_training_samples", 0),
            "previous_state_source": previous_state_source,
            "region_center": self.region_center_.tolist(),
            "region_center_source": self.region_center_source_,
            "stationarity_spectral_radius": self.fit_diagnostics_.get("stationarity_spectral_radius"),
            "stationarity_spectral_radius_altered": self.fit_diagnostics_.get("stationarity_spectral_radius_altered"),
            "candidate_diagnostics": selection.candidate_diagnostics,
            "estimated_mur_success": float(selection.estimated_success_probability),
            "structural_learning": self.fit_diagnostics_.get("structural_learning", {}),
            **selection.diagnostics,
        }
        result = DecisionResult(
            alterations=selection.alterations,
            estimated_success_probability=float(np.clip(selection.estimated_success_probability, 0.0, 1.0)),
            cost=float(selection.cost),
            diagnostics=diagnostics,
            runtime_seconds=runtime,
        )
        self.last_decision_ = result
        self.last_context_ = {
            "x_t": x_t,
            "v_prev": previous_state,
            "center": self.region_center_.copy(),
            "candidate": selection.candidate,
            "mean": selection.mean,
            "F": selection.F,
            "previous_state_source": previous_state_source,
        }
        return result

    def evaluate(self, task: AUFTask, n_samples: int) -> Mapping[str, Any]:
        if self.model_ is None or self.last_context_ is None or self.last_decision_ is None:
            raise RuntimeError("suggest must be called before evaluate.")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        eval_seed = None if self.seed is None else int(self.seed) + 7919
        eval_rng = np.random.default_rng(eval_seed)
        evaluation = rollout_mur_policy(
            self.model_,
            task,
            variant=self.variant,
            horizon=self.horizon,
            x_t=self.last_context_["x_t"],
            v_prev=self.last_context_["v_prev"],
            center=self.last_context_["center"],
            candidates=self.candidates_,
            rng=eval_rng,
            n_samples=int(n_samples),
            learning_rate=self.learning_rate,
            max_iters=self.max_iters,
            tolerance=self.tolerance,
            num_restarts=self.num_restarts,
        )
        return {
            **evaluation,
            "alterations": dict(self.last_decision_.alterations),
        }

    def _validate_task_order(self, task: AUFTask) -> None:
        assert task.variable_order is not None
        order = tuple(task.variable_order)
        missing = set(task.observed) | set(task.alterable) | set(task.outcomes)
        missing -= set(order)
        if missing:
            raise ValueError(f"MUR task.variable_order is missing task variables: {sorted(missing)}.")
        if len(set(order)) != len(order):
            raise ValueError("MUR task.variable_order contains duplicates.")

    def _previous_state_from_observation(
        self,
        observation: Mapping[str, float],
        task: AUFTask,
    ) -> tuple[np.ndarray, str]:
        assert self.model_ is not None
        prefixed = [f"{self.previous_state_prefix}{name}" for name in self.model_.variable_order]
        if all(name in observation for name in prefixed):
            return (
                np.asarray([float(observation[name]) for name in prefixed], dtype=float),
                "observation_previous_state_prefix",
            )
        if "mur_initial_state" in task.metadata:
            return _coerce_state(task.metadata["mur_initial_state"], self.model_.variable_order), "task.metadata.mur_initial_state"
        if self.last_training_state_ is not None:
            return self.last_training_state_.copy(), "fit_data_last_row"
        return np.zeros(self.model_.n_variables, dtype=float), "zeros"


def _normalize_variant(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "gmur": "gmur",
        "greedy": "gmur",
        "greedy_mur": "gmur",
        "farmur": "farmur",
        "far_mur": "farmur",
        "far_sighted": "farmur",
        "farsighted": "farmur",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError("variant must be one of 'gmur' or 'farmur'.") from exc


def _coerce_state(value: Any, variable_order: Sequence[str]) -> np.ndarray:
    if isinstance(value, Mapping):
        missing = [name for name in variable_order if name not in value]
        if missing:
            raise ValueError(f"State mapping is missing variables: {missing}.")
        state = np.asarray([float(value[name]) for name in variable_order], dtype=float)
    else:
        state = np.asarray(value, dtype=float).reshape(-1)
    if state.shape != (len(tuple(variable_order)),):
        raise ValueError("State dimension must match task.variable_order.")
    if not np.all(np.isfinite(state)):
        raise ValueError("State values must be finite.")
    return state


def _coerce_optional_training_matrix(
    data: Mapping[str, Sequence[float]] | np.ndarray | None,
    task: AUFTask,
    config: Mapping[str, Any],
) -> tuple[np.ndarray | None, tuple[str, ...]]:
    if data is None:
        return None, tuple(task.variable_order or ())
    matrix, columns = coerce_data_matrix(data, task, config, columns=task.variable_order)
    return matrix, tuple(columns)


NeurIPS2025MURRehearsal = MURRehearsal
