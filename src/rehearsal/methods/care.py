"""ICML 2025 CARE rehearsal adapter."""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import numpy as np

from rehearsal.core import AUFTask, DecisionResult
from rehearsal.core.data import candidate_alteration_sets
from rehearsal.core.regions import circular_region_inner_care, desired_region_intervals_under_independence
from rehearsal.metrics.care import independent_normal_care_success
from rehearsal.models import LinearGaussianSRM, LinearGaussianSRMLearner, StructuralLearner, StructuralLearningResult
from rehearsal.optimizers import optimize_care_independent_normal


class CARERehearsal:
    """CARE adapter for linear Gaussian AUF tasks."""

    def __init__(
        self,
        *,
        seed: int | None = None,
        candidate_alteration_sets: Sequence[Sequence[str]] | None = None,
        max_iters: int = 200,
        tolerance: float = 1e-8,
        structural_learner: StructuralLearner | None = None,
    ) -> None:
        self.seed = seed
        self.candidate_alteration_sets = (
            tuple(tuple(candidate) for candidate in candidate_alteration_sets)
            if candidate_alteration_sets is not None
            else None
        )
        self.max_iters = int(max_iters)
        self.tolerance = float(tolerance)
        self.structural_learner = structural_learner or LinearGaussianSRMLearner()
        self.structural_result_: StructuralLearningResult | None = None
        self.model_: LinearGaussianSRM | None = None
        self.config_: dict[str, Any] = {}
        self.last_decision_: DecisionResult | None = None
        self.last_context_: dict[str, Any] | None = None

    def fit(
        self,
        data: Mapping[str, Sequence[float]] | np.ndarray,
        task: AUFTask,
        config: Mapping[str, Any] | None = None,
    ) -> "CARERehearsal":
        config_dict = dict(config or {})
        self.config_ = config_dict
        desired_region_intervals_under_independence(task.desired_region, task.outcomes)

        structural_result = self.structural_learner.fit(data, task, config_dict)
        model = structural_result.model
        if not hasattr(model, "effect_matrices") or not hasattr(model, "covariance"):
            raise TypeError(
                "CARERehearsal requires a structural model with "
                "effect_matrices(task, candidate) and covariance."
            )
        self.structural_result_ = structural_result
        self.model_ = model  # type: ignore[assignment]
        return self

    def suggest(self, observation: Mapping[str, float], task: AUFTask) -> DecisionResult:
        if self.model_ is None:
            raise RuntimeError("CARERehearsal.fit must be called before suggest.")

        start = time.perf_counter()
        intervals = desired_region_intervals_under_independence(task.desired_region, task.outcomes)
        x = np.asarray([float(observation[name]) for name in task.observed], dtype=float).reshape(-1, 1)
        candidates = candidate_alteration_sets(task, self.candidate_alteration_sets)

        best: dict[str, Any] | None = None
        candidate_diagnostics = []
        min_variance = float(self.config_.get("min_variance", 1e-8))
        for candidate in candidates:
            lower, upper = task.alteration_domain.arrays_for(candidate)
            mat_a, mat_b, mat_c = self.model_.effect_matrices(task, candidate)
            base_mean = (mat_a @ x).reshape(-1)
            cov_y = mat_c @ self.model_.covariance @ mat_c.T
            variance = np.maximum(np.diag(cov_y), min_variance)

            optimized = optimize_care_independent_normal(
                base_mean,
                mat_b,
                variance,
                intervals,
                lower,
                upper,
                max_iters=self.max_iters,
                tolerance=self.tolerance,
            )

            mean = base_mean + mat_b @ optimized.z
            alterations = {name: float(value) for name, value in zip(candidate, optimized.z)}
            cost = task.alteration_domain.cost(alterations)
            diag = {
                "candidate": tuple(candidate),
                "estimated_care_success": optimized.care_success,
                "objective_value": optimized.objective_value,
                "solver_status": optimized.solver_status,
                **optimized.diagnostics,
            }
            candidate_diagnostics.append(diag)
            if best is None or optimized.care_success > best["care_success"] + 1e-15:
                best = {
                    "candidate": tuple(candidate),
                    "z": optimized.z,
                    "mean": mean,
                    "variance": variance,
                    "care_success": optimized.care_success,
                    "objective": optimized.objective_value,
                    "cost": cost,
                    "status": optimized.solver_status,
                    "solver_meta": optimized.diagnostics,
                    "alterations": alterations,
                }
            elif best is not None and abs(optimized.care_success - best["care_success"]) <= 1e-15 and cost < best["cost"]:
                best.update(
                    {
                        "candidate": tuple(candidate),
                        "z": optimized.z,
                        "mean": mean,
                        "variance": variance,
                        "care_success": optimized.care_success,
                        "objective": optimized.objective_value,
                        "cost": cost,
                        "status": optimized.solver_status,
                        "solver_meta": optimized.diagnostics,
                        "alterations": alterations,
                    }
                )

        if best is None:
            raise ValueError("No candidate alteration set is available.")

        runtime = time.perf_counter() - start
        structural_diagnostics = dict(self.structural_result_.diagnostics) if self.structural_result_ else {}
        diagnostics = {
            "selected_candidate": best["candidate"],
            "objective_value": best["objective"],
            "estimated_care_success": best["care_success"],
            "solver_status": best["status"],
            "n_candidates": len(candidates),
            "desired_intervals": intervals.tolist(),
            "outcome_variances": best["variance"].tolist(),
            "assumed_independent_outcomes": len(task.outcomes) > 1,
            "candidate_diagnostics": candidate_diagnostics,
            "structural_learning": structural_diagnostics,
            **best["solver_meta"],
        }
        result = DecisionResult(
            alterations=best["alterations"],
            estimated_success_probability=float(np.clip(best["care_success"], 0.0, 1.0)),
            cost=float(best["cost"]),
            diagnostics=diagnostics,
            runtime_seconds=runtime,
        )
        self.last_decision_ = result
        self.last_context_ = {
            "candidate": best["candidate"],
            "mean": np.asarray(best["mean"], dtype=float),
            "variance": np.asarray(best["variance"], dtype=float),
        }
        return result

    def evaluate(self, task: AUFTask, n_samples: int) -> Mapping[str, Any]:
        if self.last_context_ is None or self.last_decision_ is None:
            raise RuntimeError("suggest must be called before evaluate.")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        intervals = desired_region_intervals_under_independence(task.desired_region, task.outcomes)
        rng = np.random.default_rng(self.seed)
        mean = self.last_context_["mean"]
        std = np.sqrt(np.maximum(self.last_context_["variance"], 1e-12))
        samples = rng.normal(mean, std, size=(int(n_samples), len(mean)))
        success = np.asarray(task.desired_region.contains(samples), dtype=bool)
        return {
            "estimated_success_probability": float(np.mean(success)),
            "analytic_care_success": independent_normal_care_success(
                mean,
                self.last_context_["variance"],
                intervals,
            ),
            "n_samples": int(n_samples),
            "alterations": dict(self.last_decision_.alterations),
        }


ICML2025CARERehearsal = CARERehearsal
