"""Small public contracts shared by method adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class DesiredRegion:
    """A desired outcome region expressed as ``M y <= d``."""

    M: Sequence[Sequence[float]] | Sequence[float]
    d: Sequence[float]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        matrix = self.matrix
        vector = np.asarray(self.d, dtype=float).reshape(-1)
        if matrix.shape[0] != vector.shape[0]:
            raise ValueError("DesiredRegion.M and DesiredRegion.d row counts must match.")
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(vector)):
            raise ValueError("DesiredRegion.M and DesiredRegion.d must be finite.")

    @property
    def matrix(self) -> np.ndarray:
        matrix = np.asarray(self.M, dtype=float)
        if matrix.ndim == 0:
            return matrix.reshape(1, 1)
        if matrix.ndim == 1:
            return matrix.reshape(-1, 1)
        if matrix.ndim == 2:
            return matrix
        raise ValueError("DesiredRegion.M must be a scalar, vector, or matrix.")

    @property
    def vector(self) -> np.ndarray:
        return np.asarray(self.d, dtype=float).reshape(-1)

    @classmethod
    def from_intervals(
        cls,
        intervals: Mapping[str, tuple[float, float]] | Sequence[tuple[float, float]],
        variables: Sequence[str] | None = None,
    ) -> "DesiredRegion":
        """Build an axis-aligned interval region for the supplied outcomes."""

        if isinstance(intervals, Mapping):
            if variables is None:
                variables = tuple(intervals)
            ordered = [intervals[name] for name in variables]
        else:
            ordered = list(intervals)
            if variables is None:
                variables = tuple(f"y{i}" for i in range(len(ordered)))

        dim = len(ordered)
        matrix = np.zeros((2 * dim, dim), dtype=float)
        vector = np.zeros(2 * dim, dtype=float)
        for idx, (lower, upper) in enumerate(ordered):
            if lower > upper:
                raise ValueError(f"Interval lower bound exceeds upper bound for dimension {idx}.")
            matrix[idx, idx] = 1.0
            vector[idx] = float(upper)
            matrix[idx + dim, idx] = -1.0
            vector[idx + dim] = -float(lower)
        return cls(matrix, vector, metadata={"variables": tuple(variables), "kind": "intervals"})

    def contains(self, y: Sequence[float] | np.ndarray, *, atol: float = 1e-12) -> np.ndarray:
        """Return whether samples satisfy ``M y <= d``.

        ``y`` may be a single vector of shape ``(d,)`` or a batch with shape
        ``(n, d)``. The return value is a scalar boolean array for one sample or
        a boolean vector for a batch.
        """

        values = np.asarray(y, dtype=float)
        if values.ndim == 1:
            return np.all(self.matrix @ values <= self.vector + atol)
        if values.ndim != 2:
            raise ValueError("y must be a vector or a two-dimensional sample matrix.")
        return np.all(values @ self.matrix.T <= self.vector + atol, axis=1)


@dataclass(frozen=True)
class AlterationDomain:
    """Per-variable alteration bounds and optional absolute-value costs."""

    bounds: Mapping[str, tuple[float, float]]
    costs: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, (lower, upper) in self.bounds.items():
            if not np.isfinite(lower) or not np.isfinite(upper):
                raise ValueError(f"Alteration bounds for {name!r} must be finite.")
            if lower > upper:
                raise ValueError(f"Alteration lower bound exceeds upper bound for {name!r}.")
        for name, cost in self.costs.items():
            if cost < 0 or not np.isfinite(cost):
                raise ValueError(f"Alteration cost for {name!r} must be finite and non-negative.")

    def arrays_for(self, variables: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        lower = []
        upper = []
        for name in variables:
            if name not in self.bounds:
                raise KeyError(f"No alteration bounds configured for {name!r}.")
            lo, hi = self.bounds[name]
            lower.append(float(lo))
            upper.append(float(hi))
        return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)

    def clip(self, alterations: Mapping[str, float]) -> dict[str, float]:
        clipped: dict[str, float] = {}
        for name, value in alterations.items():
            if name not in self.bounds:
                raise KeyError(f"No alteration bounds configured for {name!r}.")
            lo, hi = self.bounds[name]
            clipped[name] = float(np.clip(value, lo, hi))
        return clipped

    def cost(self, alterations: Mapping[str, float]) -> float:
        total = 0.0
        for name, value in alterations.items():
            total += float(self.costs.get(name, 1.0)) * abs(float(value))
        return total


@dataclass(frozen=True)
class AUFTask:
    """A normalized Avoiding Undesired Future task definition."""

    observed_variables: Sequence[str]
    alterable_variables: Sequence[str]
    outcome_variables: Sequence[str]
    desired_region: DesiredRegion
    alteration_domain: AlterationDomain
    parents: Mapping[str, Sequence[str]] = field(default_factory=dict)
    candidate_alteration_sets: Sequence[Sequence[str]] | None = None
    variable_order: Sequence[str] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        observed = tuple(self.observed_variables)
        alterable = tuple(self.alterable_variables)
        outcomes = tuple(self.outcome_variables)
        if len(set(observed)) != len(observed):
            raise ValueError("observed_variables contains duplicates.")
        if len(set(alterable)) != len(alterable):
            raise ValueError("alterable_variables contains duplicates.")
        if len(set(outcomes)) != len(outcomes):
            raise ValueError("outcome_variables contains duplicates.")
        if self.desired_region.matrix.shape[1] != len(outcomes):
            raise ValueError("DesiredRegion dimension must match outcome_variables.")
        missing_bounds = set(alterable) - set(self.alteration_domain.bounds)
        if missing_bounds:
            missing = ", ".join(sorted(missing_bounds))
            raise ValueError(f"Missing alteration bounds for: {missing}.")

    @property
    def observed(self) -> tuple[str, ...]:
        return tuple(self.observed_variables)

    @property
    def alterable(self) -> tuple[str, ...]:
        return tuple(self.alterable_variables)

    @property
    def outcomes(self) -> tuple[str, ...]:
        return tuple(self.outcome_variables)

    def all_variables(self) -> tuple[str, ...]:
        if self.variable_order is not None:
            return tuple(self.variable_order)
        ordered = list(self.observed) + list(self.alterable) + list(self.outcomes)
        for child, parents in self.parents.items():
            for name in (child, *parents):
                if name not in ordered:
                    ordered.append(name)
        return tuple(ordered)


@dataclass(frozen=True)
class DecisionResult:
    alterations: Mapping[str, float]
    estimated_success_probability: float
    cost: float
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    runtime_seconds: float = 0.0

    def __post_init__(self) -> None:
        probability = float(self.estimated_success_probability)
        if probability < 0.0 or probability > 1.0:
            raise ValueError("estimated_success_probability must be in [0, 1].")
        if self.cost < 0.0:
            raise ValueError("cost must be non-negative.")
        if self.runtime_seconds < 0.0:
            raise ValueError("runtime_seconds must be non-negative.")


@dataclass(frozen=True)
class ExperimentResult:
    metrics: Mapping[str, float]
    seed: int | None = None
    config: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, str] = field(default_factory=dict)


class RehearsalMethod(Protocol):
    def fit(self, data: Any, task: AUFTask, config: Mapping[str, Any] | None = None) -> "RehearsalMethod":
        ...

    def suggest(self, observation: Mapping[str, float], task: AUFTask) -> DecisionResult:
        ...

    def evaluate(self, task: AUFTask, n_samples: int) -> Mapping[str, Any]:
        ...
