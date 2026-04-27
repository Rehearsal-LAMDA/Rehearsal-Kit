"""Shared linear Gaussian SRM/SEM model.

This module is intentionally paper-neutral. It is suitable as the common
structural model for the ICML 2025 CARE method and for later
NeurIPS 2023/2024 ports that learn or update linear SRM parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from rehearsal.core import AUFTask
from rehearsal.core.data import coerce_data_matrix
from rehearsal.models.base import StructuralLearningResult

Theta = Mapping[str, Mapping[str, float]]


@dataclass(frozen=True)
class LinearGaussianSRM:
    """A linear structural rehearsal model with Gaussian residuals.

    ``theta[parent][child]`` stores the linear coefficient from parent to child.
    ``covariance`` is the residual covariance matrix in ``variable_order``.
    """

    variable_order: tuple[str, ...]
    theta: dict[str, dict[str, float]]
    covariance: np.ndarray

    def __post_init__(self) -> None:
        covariance = np.asarray(self.covariance, dtype=float)
        if covariance.shape != (len(self.variable_order), len(self.variable_order)):
            raise ValueError("covariance shape must match variable_order.")
        if not np.all(np.isfinite(covariance)):
            raise ValueError("covariance must be finite.")

    @property
    def sigma(self) -> np.ndarray:
        """Backward-compatible alias used by early ICML adapter code."""

        return self.covariance

    @property
    def parents(self) -> dict[str, tuple[str, ...]]:
        return parents_from_theta(self.theta, self.variable_order)

    def effect_matrices(
        self,
        task: AUFTask,
        candidate: Sequence[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``A, B, C`` such that outcome mean is ``A x + B z``.

        ``C`` maps residual noise to outcomes after cutting incoming edges into
        the candidate alteration variables.
        """

        candidate = tuple(candidate)
        graph: dict[str, dict[str, float]] = {name: {} for name in self.variable_order}
        for parent, children in self.theta.items():
            for child, coef in children.items():
                if child in candidate:
                    continue
                graph.setdefault(parent, {})[child] = float(coef)

        observed = task.observed
        outcomes = task.outcomes
        variable_index = {name: idx for idx, name in enumerate(self.variable_order)}
        mat_a = np.zeros((len(outcomes), len(observed)), dtype=float)
        mat_b = np.zeros((len(outcomes), len(candidate)), dtype=float)
        mat_c = np.zeros((len(outcomes), len(self.variable_order)), dtype=float)

        for y_idx, outcome in enumerate(outcomes):
            mat_c[y_idx, variable_index[outcome]] = 1.0
            effects = total_path_effects(graph, outcome)
            for node, effect in effects.items():
                if abs(effect) <= 0.0:
                    continue
                if node in candidate:
                    mat_b[y_idx, candidate.index(node)] = effect
                elif node in observed:
                    mat_a[y_idx, observed.index(node)] = effect
                elif node in variable_index:
                    mat_c[y_idx, variable_index[node]] = effect
        return mat_a, mat_b, mat_c

    def outcome_moments(
        self,
        task: AUFTask,
        observation: Mapping[str, float],
        candidate: Sequence[str],
        alteration_values: Sequence[float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return Gaussian outcome mean and covariance for a candidate decision."""

        mat_a, mat_b, mat_c = self.effect_matrices(task, candidate)
        x = np.asarray([float(observation[name]) for name in task.observed], dtype=float).reshape(-1)
        mean = mat_a @ x
        if alteration_values is not None:
            z = np.asarray(alteration_values, dtype=float).reshape(-1)
            mean = mean + mat_b @ z
        covariance_y = mat_c @ self.covariance @ mat_c.T
        return mean.reshape(-1), covariance_y

    def simulate(
        self,
        n_samples: int,
        *,
        rng: np.random.Generator,
        observation: Mapping[str, float] | None = None,
        alterations: Mapping[str, float] | None = None,
    ) -> dict[str, np.ndarray]:
        """Sample from the structural model under optional observations/actions."""

        variables = tuple(self.variable_order)
        index = {name: idx for idx, name in enumerate(variables)}
        observed = dict(observation or {})
        altered = dict(alterations or {})
        noise = rng.multivariate_normal(np.zeros(len(variables)), self.covariance, size=int(n_samples))
        values: dict[str, np.ndarray] = {}
        parents = self.parents
        for name in variables:
            if name in observed:
                values[name] = np.full(n_samples, float(observed[name]))
                continue
            if name in altered:
                values[name] = np.full(n_samples, float(altered[name]))
                continue
            current = noise[:, index[name]].copy()
            for parent in parents.get(name, ()):
                current += float(self.theta[parent][name]) * values[parent]
            values[name] = current
        return values


class LinearGaussianSRMLearner:
    """Least-squares structural-parameter learner for a known variable graph."""

    def __init__(self, *, min_variance: float = 1e-8) -> None:
        self.min_variance = float(min_variance)

    def fit(
        self,
        data: Mapping[str, Sequence[float]] | np.ndarray,
        task: AUFTask,
        config: Mapping[str, Any] | None = None,
    ) -> StructuralLearningResult:
        config_dict = dict(config or {})
        min_variance = float(config_dict.get("min_variance", self.min_variance))
        matrix, columns = coerce_data_matrix(data, task, config_dict)
        if matrix.shape[0] < 2:
            raise ValueError("At least two samples are required to fit a linear Gaussian SRM.")

        column_index = {name: idx for idx, name in enumerate(columns)}
        parent_map = {name: tuple(task.parents.get(name, ())) for name in columns}
        theta: dict[str, dict[str, float]] = {name: {} for name in columns}
        residuals = np.zeros_like(matrix, dtype=float)

        for child in columns:
            y = matrix[:, column_index[child]].reshape(-1, 1)
            parents = parent_map.get(child, ())
            if not parents:
                residuals[:, column_index[child]] = y.ravel()
                continue
            missing = [name for name in parents if name not in column_index]
            if missing:
                raise ValueError(f"Parents for {child!r} are missing from data: {missing}.")
            x = matrix[:, [column_index[name] for name in parents]]
            beta = np.linalg.pinv(x.T @ x) @ x.T @ y
            prediction = x @ beta
            residuals[:, column_index[child]] = (y - prediction).ravel()
            for idx, parent in enumerate(parents):
                theta.setdefault(parent, {})[child] = float(beta[idx, 0])

        covariance = np.cov(residuals, rowvar=False, bias=True)
        covariance = np.atleast_2d(np.asarray(covariance, dtype=float))
        covariance = covariance + min_variance * np.eye(len(columns))
        model = LinearGaussianSRM(tuple(columns), theta, covariance)
        diagnostics = {
            "learner": type(self).__name__,
            "n_samples": int(matrix.shape[0]),
            "n_variables": int(matrix.shape[1]),
            "min_variance": min_variance,
        }
        return StructuralLearningResult(model=model, diagnostics=diagnostics)


def total_path_effects(graph: Mapping[str, Mapping[str, float]], end_node: str) -> dict[str, float]:
    """Return total directed path effects from every ancestor to ``end_node``."""

    nodes = set(graph)
    for children in graph.values():
        nodes.update(children)

    def dfs(node: str, active: set[str]) -> float:
        if node == end_node:
            return 1.0
        if node in active:
            raise ValueError("Linear SRM graph contains a directed cycle.")
        active.add(node)
        total = 0.0
        for child, coef in graph.get(node, {}).items():
            total += float(coef) * dfs(child, active)
        active.remove(node)
        return total

    effects: dict[str, float] = {}
    for node in nodes:
        if node == end_node:
            continue
        effect = dfs(node, set())
        if abs(effect) > 0.0:
            effects[node] = effect
    return effects


def parents_from_theta(theta: Theta, variable_order: Sequence[str]) -> dict[str, tuple[str, ...]]:
    """Build a child-to-parents map from ``theta[parent][child]`` coefficients."""

    parents: dict[str, list[str]] = {name: [] for name in variable_order}
    for parent, children in theta.items():
        for child in children:
            parents.setdefault(child, []).append(parent)
    return {name: tuple(parents.get(name, ())) for name in variable_order}
