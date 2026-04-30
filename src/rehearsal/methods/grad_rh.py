"""AAAI 2025 Grad-Rh adapter for the unified Rehearsal API."""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import numpy as np

from rehearsal.core import AUFTask, DecisionResult
from rehearsal.core.data import candidate_alteration_sets
from rehearsal.models.nonlinear import NonlinearSRMLearner, NonlinearStructuralModel
from rehearsal.optimizers.grad_rh import optimize_grad_rh_alterations


class GradRhRehearsal:
    """Nonlinear/multivariate rehearsal with bounded alteration optimization."""

    def __init__(
        self,
        *,
        seed: int | None = None,
        predictor_type: str = "linear",
        srm_type: str | None = None,
        feature_degree: int = 1,
        candidate_alteration_sets: Sequence[Sequence[str]] | None = None,
        n_mc_samples: int = 256,
        learning_rate: float = 0.05,
        epochs: int = 80,
        patience: int = 20,
        loss: str = "center_mae",
        num_restarts: int = 3,
    ) -> None:
        self.seed = seed
        self.predictor_type = str(srm_type or predictor_type)
        self.feature_degree = int(feature_degree)
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
        self.model_: NonlinearStructuralModel | None = None
        self.fit_diagnostics_: dict[str, Any] = {}
        self.last_decision_: DecisionResult | None = None
        self.last_observation_: dict[str, float] | None = None

    def fit(
        self,
        data: Any,
        task: AUFTask,
        config: Mapping[str, Any] | None = None,
    ) -> "GradRhRehearsal":
        start = time.perf_counter()
        config_dict = dict(config or {})
        learner = NonlinearSRMLearner(
            predictor_type=str(config_dict.get("predictor_type", config_dict.get("srm_type", self.predictor_type))),
            feature_degree=int(config_dict.get("feature_degree", self.feature_degree)),
            ridge=float(config_dict.get("ridge", 1e-6)),
            min_variance=float(config_dict.get("min_variance", 1e-8)),
        )
        result = learner.fit(data, task, config_dict)
        self.model_ = result.model  # type: ignore[assignment]
        self.fit_diagnostics_ = {**dict(result.diagnostics), "runtime_seconds": time.perf_counter() - start}
        return self

    def suggest(self, observation: Mapping[str, float], task: AUFTask) -> DecisionResult:
        if self.model_ is None:
            raise RuntimeError("GradRhRehearsal.fit must be called before suggest.")
        start = time.perf_counter()
        candidates = candidate_alteration_sets(task, self.candidate_alteration_sets)
        best: dict[str, Any] | None = None
        candidate_diagnostics = []
        for idx, candidate in enumerate(candidates):
            lower, upper = task.alteration_domain.arrays_for(candidate)
            rng = _candidate_rng(self.seed, idx)
            optimized = optimize_grad_rh_alterations(
                self.model_,
                task,
                observation,
                candidate,
                lower=lower,
                upper=upper,
                n_samples=self.n_mc_samples,
                learning_rate=self.learning_rate,
                epochs=self.epochs,
                patience=self.patience,
                loss=self.loss,
                num_restarts=self.num_restarts,
                rng=rng,
            )
            alterations = {name: float(value) for name, value in zip(candidate, optimized.action)}
            cost = task.alteration_domain.cost(alterations)
            diag = {
                "candidate": tuple(candidate),
                "estimated_success_probability": optimized.estimated_success_probability,
                "objective_value": optimized.objective_value,
                "solver_status": optimized.solver_status,
                "cost": cost,
                **dict(optimized.diagnostics),
            }
            candidate_diagnostics.append(diag)
            if best is None or _better_candidate(optimized.estimated_success_probability, optimized.objective_value, cost, best):
                best = {
                    "candidate": tuple(candidate),
                    "alterations": alterations,
                    "cost": cost,
                    "optimized": optimized,
                }
        if best is None:
            raise ValueError("No candidate alteration set is available.")
        runtime = time.perf_counter() - start
        optimized = best["optimized"]
        diagnostics = {
            "method_family": "AAAI 2025 Grad-Rh",
            "selected_candidate": best["candidate"],
            "n_candidates": len(candidates),
            "objective_value": optimized.objective_value,
            "solver_status": optimized.solver_status,
            "candidate_diagnostics": candidate_diagnostics,
            **self.fit_diagnostics_,
            **dict(optimized.diagnostics),
        }
        result = DecisionResult(
            alterations=best["alterations"],
            estimated_success_probability=float(optimized.estimated_success_probability),
            cost=float(best["cost"]),
            diagnostics=diagnostics,
            runtime_seconds=runtime,
        )
        self.last_decision_ = result
        self.last_observation_ = {name: float(value) for name, value in observation.items()}
        return result

    def evaluate(self, task: AUFTask, n_samples: int) -> Mapping[str, Any]:
        if self.model_ is None or self.last_decision_ is None or self.last_observation_ is None:
            raise RuntimeError("fit and suggest must be called before evaluate.")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        rng = np.random.default_rng(None if self.seed is None else int(self.seed) + 94111)
        outcomes = self.model_.simulate_outcomes(
            task,
            self.last_observation_,
            self.last_decision_.alterations,
            int(n_samples),
            rng=rng,
        )
        return {
            "estimated_success_probability": float(np.mean(task.desired_region.contains(outcomes))),
            "n_samples": int(n_samples),
            "alterations": dict(self.last_decision_.alterations),
        }


AAAI2025GradRhRehearsal = GradRhRehearsal


def _candidate_rng(seed: int | None, candidate_index: int) -> np.random.Generator:
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(int(seed) + 7919 * (candidate_index + 1))


def _better_candidate(probability: float, objective: float, cost: float, best: Mapping[str, Any]) -> bool:
    best_opt = best["optimized"]
    if probability > float(best_opt.estimated_success_probability) + 1e-12:
        return True
    if abs(probability - float(best_opt.estimated_success_probability)) <= 1e-12:
        if objective < float(best_opt.objective_value) - 1e-12:
            return True
        if abs(objective - float(best_opt.objective_value)) <= 1e-12 and cost < float(best["cost"]):
            return True
    return False
