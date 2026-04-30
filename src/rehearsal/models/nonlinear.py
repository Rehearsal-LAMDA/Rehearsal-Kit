"""Nonlinear structural models for Grad-Rh rehearsal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from rehearsal.core import AUFTask
from rehearsal.core.data import coerce_data_matrix
from rehearsal.models.base import StructuralLearningResult


@dataclass(frozen=True)
class ConditionalPredictor:
    """Conditional sampler for one structural equation."""

    child: str
    parents: tuple[str, ...]
    coefficients: np.ndarray
    residuals: np.ndarray
    feature_degree: int = 1
    predictor_type: str = "linear"
    min_variance: float = 1e-8

    def predict(self, parent_values: np.ndarray) -> np.ndarray:
        x = np.asarray(parent_values, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        features = polynomial_features(x, degree=self.feature_degree)
        return features @ self.coefficients

    def draw_noise(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        residuals = np.asarray(self.residuals, dtype=float).reshape(-1)
        if residuals.size == 0:
            return rng.normal(scale=np.sqrt(self.min_variance), size=int(n_samples))
        if self.predictor_type in {"flow", "flow-style", "flow_style", "residual_bootstrap"}:
            indices = rng.integers(0, residuals.size, size=int(n_samples))
            jitter = rng.normal(scale=np.sqrt(self.min_variance), size=int(n_samples))
            return residuals[indices] + jitter
        std = float(np.sqrt(max(np.var(residuals), self.min_variance)))
        return rng.normal(loc=0.0, scale=std, size=int(n_samples))


@dataclass(frozen=True)
class NonlinearStructuralModel:

    variable_order: tuple[str, ...]
    predictors: Mapping[str, ConditionalPredictor]
    training_matrix: np.ndarray | None = None
    columns: tuple[str, ...] | None = None

    @property
    def parents(self) -> dict[str, tuple[str, ...]]:
        return {name: tuple(predictor.parents) for name, predictor in self.predictors.items()}

    def make_noise_bank(
        self,
        n_samples: int,
        *,
        rng: np.random.Generator,
        variables: Sequence[str] | None = None,
    ) -> dict[str, np.ndarray]:
        selected = tuple(variables or self.variable_order)
        bank: dict[str, np.ndarray] = {}
        for name in selected:
            predictor = self.predictors.get(name)
            if predictor is None:
                bank[name] = np.zeros(int(n_samples), dtype=float)
            else:
                bank[name] = predictor.draw_noise(int(n_samples), rng)
        return bank

    def simulate(
        self,
        n_samples: int,
        *,
        rng: np.random.Generator,
        observation: Mapping[str, float] | None = None,
        alterations: Mapping[str, float] | None = None,
        noise_bank: Mapping[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        observed = dict(observation or {})
        altered = dict(alterations or {})
        bank = dict(noise_bank or self.make_noise_bank(n_samples, rng=rng))
        values: dict[str, np.ndarray] = {}
        for name in self.variable_order:
            if name in observed:
                values[name] = np.full(int(n_samples), float(observed[name]))
                continue
            if name in altered:
                values[name] = np.full(int(n_samples), float(altered[name]))
                continue
            predictor = self.predictors.get(name)
            if predictor is None:
                values[name] = np.asarray(bank.get(name, np.zeros(int(n_samples))), dtype=float).reshape(-1)
                continue
            if predictor.parents:
                parent_matrix = np.column_stack([values[parent] for parent in predictor.parents])
            else:
                parent_matrix = np.zeros((int(n_samples), 0), dtype=float)
            noise = np.asarray(bank.get(name, predictor.draw_noise(int(n_samples), rng)), dtype=float).reshape(-1)
            values[name] = predictor.predict(parent_matrix).reshape(-1) + noise
        return values

    def simulate_outcomes(
        self,
        task: AUFTask,
        observation: Mapping[str, float],
        alterations: Mapping[str, float],
        n_samples: int,
        *,
        rng: np.random.Generator,
        noise_bank: Mapping[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        samples = self.simulate(
            n_samples,
            rng=rng,
            observation=observation,
            alterations=alterations,
            noise_bank=noise_bank,
        )
        return np.column_stack([samples[name] for name in task.outcomes])


class NonlinearSRMLearner:
    """Fit conditional structural equations from the task graph."""

    def __init__(
        self,
        *,
        predictor_type: str = "linear",
        feature_degree: int = 1,
        ridge: float = 1e-6,
        min_variance: float = 1e-8,
    ) -> None:
        self.predictor_type = str(predictor_type)
        self.feature_degree = int(feature_degree)
        self.ridge = float(ridge)
        self.min_variance = float(min_variance)

    def fit(
        self,
        data: Mapping[str, Sequence[float]] | np.ndarray,
        task: AUFTask,
        config: Mapping[str, Any] | None = None,
    ) -> StructuralLearningResult:
        config_dict = dict(config or {})
        predictor_type = str(config_dict.get("predictor_type", config_dict.get("srm_type", self.predictor_type)))
        feature_degree = int(config_dict.get("feature_degree", self.feature_degree))
        ridge = float(config_dict.get("ridge", self.ridge))
        min_variance = float(config_dict.get("min_variance", self.min_variance))
        matrix, columns = coerce_data_matrix(data, task, config_dict)
        if matrix.shape[0] < 2:
            raise ValueError("At least two samples are required to fit nonlinear SRM predictors.")
        column_index = {name: idx for idx, name in enumerate(columns)}
        ordered = tuple(task.variable_order or columns)
        predictors: dict[str, ConditionalPredictor] = {}
        for child in ordered:
            if child not in column_index:
                continue
            parents = tuple(parent for parent in task.parents.get(child, ()) if parent in column_index)
            predictors[child] = fit_conditional_predictor(
                matrix,
                columns,
                child,
                parents,
                predictor_type=predictor_type,
                feature_degree=feature_degree,
                ridge=ridge,
                min_variance=min_variance,
            )
        model = NonlinearStructuralModel(ordered, predictors, training_matrix=matrix, columns=columns)
        return StructuralLearningResult(
            model=model,
            diagnostics={
                "learner": type(self).__name__,
                "predictor_type": predictor_type,
                "feature_degree": feature_degree,
                "n_samples": int(matrix.shape[0]),
                "n_variables": int(matrix.shape[1]),
            },
        )


def fit_nonlinear_structural_model(
    data: Mapping[str, Sequence[float]] | np.ndarray,
    task: AUFTask,
    config: Mapping[str, Any] | None = None,
) -> StructuralLearningResult:
    return NonlinearSRMLearner().fit(data, task, config)


def fit_conditional_predictor(
    matrix: np.ndarray,
    columns: Sequence[str],
    child: str,
    parents: Sequence[str],
    *,
    predictor_type: str = "linear",
    feature_degree: int = 1,
    ridge: float = 1e-6,
    min_variance: float = 1e-8,
) -> ConditionalPredictor:
    column_index = {name: idx for idx, name in enumerate(columns)}
    y = np.asarray(matrix[:, column_index[child]], dtype=float).reshape(-1, 1)
    if parents:
        x = matrix[:, [column_index[name] for name in parents]]
    else:
        x = np.zeros((matrix.shape[0], 0), dtype=float)
    features = polynomial_features(x, degree=int(feature_degree))
    system = features.T @ features + float(ridge) * np.eye(features.shape[1])
    rhs = features.T @ y
    try:
        coefficients = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(system, rhs, rcond=None)[0]
    residuals = (y - features @ coefficients).reshape(-1)
    return ConditionalPredictor(
        child=str(child),
        parents=tuple(parents),
        coefficients=np.asarray(coefficients, dtype=float),
        residuals=np.asarray(residuals, dtype=float),
        feature_degree=int(feature_degree),
        predictor_type=str(predictor_type),
        min_variance=float(min_variance),
    )


def polynomial_features(x: np.ndarray, *, degree: int = 1) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    if values.ndim != 2:
        raise ValueError("x must be a two-dimensional matrix.")
    n_samples, n_features = values.shape
    cols = [np.ones((n_samples, 1), dtype=float)]
    if n_features:
        cols.append(values)
        if int(degree) >= 2:
            cols.append(values * values)
            if n_features > 1:
                interactions = []
                for i in range(n_features):
                    for j in range(i + 1, n_features):
                        interactions.append((values[:, i] * values[:, j]).reshape(-1, 1))
                if interactions:
                    cols.append(np.hstack(interactions))
    return np.hstack(cols)


def topological_order(variables: Sequence[str], parents: Mapping[str, Sequence[str]]) -> tuple[str, ...]:

    ordered = list(variables)
    position = {name: idx for idx, name in enumerate(ordered)}
    remaining = set(ordered)
    result: list[str] = []
    while remaining:
        ready = [
            name
            for name in remaining
            if all(parent not in remaining for parent in parents.get(name, ()) if parent in position)
        ]
        if not ready:
            raise ValueError("Structural graph contains a directed cycle.")
        ready.sort(key=position.get)
        result.extend(ready)
        remaining.difference_update(ready)
    return tuple(result)
