"""Base structural-learning contracts.

Most rehearsal-learning papers in this repository have two phases:

1. learn a structural model or its parameters from data, and
2. optimize rehearsal decisions against that fitted structure.

The protocols here keep those phases separable while still allowing method
adapters to expose the public ``RehearsalMethod`` API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from rehearsal.core import AUFTask


class StructuralModel(Protocol):
    """A fitted causal/structural model usable by rehearsal optimizers."""

    variable_order: tuple[str, ...]


@dataclass(frozen=True)
class StructuralLearningResult:
    """Output of a structural-learning phase."""

    model: StructuralModel
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


class StructuralLearner(Protocol):
    """Learns a structural model from observational or interventional data."""

    def fit(
        self,
        data: Any,
        task: AUFTask,
        config: Mapping[str, Any] | None = None,
    ) -> StructuralLearningResult:
        ...
