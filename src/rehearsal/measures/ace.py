"""Average Causal Effect (ACE) and Conditional ACE (CACE) measures.

ACE measures how much altering a single variable can shift the
probability of the target landing in the desired outcome bins, in the
absence of any other context. CACE adds prior ``do`` / ``observe``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from rehearsal.core import AUFTask
from rehearsal.measures.discretizer import UniformBinDiscretizer
from rehearsal.measures.mep import MEPCalculator
from rehearsal.models.nonlinear import NonlinearStructuralModel


@dataclass(frozen=True)
class ACEResult:
    """Result record returned by :func:`compute_ace` / :func:`compute_cace`."""

    variable: str
    ace: float
    kind: str  # "ace" or "cace"
    per_value_probs: dict[int, float]
    best_value: int
    worst_value: int
    do_conditions: dict[str, int] = field(default_factory=dict)
    ob_conditions: dict[str, int] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "ace": float(self.ace),
            "kind": self.kind,
            "per_value_probs": {int(k): float(v) for k, v in self.per_value_probs.items()},
            "best_value": int(self.best_value),
            "worst_value": int(self.worst_value),
            "do_conditions": dict(self.do_conditions),
            "ob_conditions": dict(self.ob_conditions),
            "diagnostics": dict(self.diagnostics),
        }


def compute_ace(
    model: NonlinearStructuralModel,
    task: AUFTask,
    variable: str,
    *,
    ordering: Sequence[str],
    target: str,
    desired_bins: Sequence[int],
    discretizer: UniformBinDiscretizer,
    num_samples: int = 1000,
    rng: np.random.Generator | None = None,
    calculator: MEPCalculator | None = None,
) -> ACEResult:
    """Unconditional ACE: ``max_v P(Y∈desired | do(var=v)) - min_v ...``."""

    return _ace_core(
        model,
        task,
        variable,
        ordering=ordering,
        target=target,
        desired_bins=desired_bins,
        discretizer=discretizer,
        do_conditions=None,
        ob_conditions=None,
        num_samples=num_samples,
        rng=rng,
        calculator=calculator,
        kind="ace",
    )


def compute_cace(
    model: NonlinearStructuralModel,
    task: AUFTask,
    variable: str,
    *,
    ordering: Sequence[str],
    target: str,
    desired_bins: Sequence[int],
    discretizer: UniformBinDiscretizer,
    do_conditions: Mapping[str, int] | None = None,
    ob_conditions: Mapping[str, int] | None = None,
    num_samples: int = 1000,
    rng: np.random.Generator | None = None,
    calculator: MEPCalculator | None = None,
) -> ACEResult:
    """Conditional ACE under prior ``do`` / ``observe`` context.

    """

    return _ace_core(
        model,
        task,
        variable,
        ordering=ordering,
        target=target,
        desired_bins=desired_bins,
        discretizer=discretizer,
        do_conditions=do_conditions,
        ob_conditions=ob_conditions,
        num_samples=num_samples,
        rng=rng,
        calculator=calculator,
        kind="cace",
    )


def _ace_core(
    model: NonlinearStructuralModel,
    task: AUFTask,
    variable: str,
    *,
    ordering: Sequence[str],
    target: str,
    desired_bins: Sequence[int],
    discretizer: UniformBinDiscretizer,
    do_conditions: Mapping[str, int] | None,
    ob_conditions: Mapping[str, int] | None,
    num_samples: int,
    rng: np.random.Generator | None,
    calculator: MEPCalculator | None,
    kind: str,
) -> ACEResult:
    if discretizer is None:
        raise ValueError("ACE/CACE require a UniformBinDiscretizer (continuous outputs must be discretized).")
    _ = task

    calc = calculator or MEPCalculator(
        model,
        ordering=ordering,
        target_node=target,
        desired_bins=desired_bins,
        discretizer=discretizer,
        alterable=task.alterable,
        num_samples=num_samples,
        rng=rng,
    )

    do_dict: dict[str, int] = {str(k): int(v) for k, v in (do_conditions or {}).items()}
    ob_dict: dict[str, int] = {str(k): int(v) for k, v in (ob_conditions or {}).items()}
    if variable in do_dict or variable in ob_dict:
        raise ValueError(
            f"variable {variable!r} must not appear in the conditioning context "
            "(it is the variable being measured)."
        )

    per_value: dict[int, float] = {}
    for value in discretizer.get_bins(variable):
        candidate_do = dict(do_dict)
        candidate_do[variable] = int(value)
        per_value[int(value)] = float(calc.gauge_auf_prob(candidate_do, ob_dict))

    if not per_value:
        raise RuntimeError(f"No bins available for variable {variable!r}.")

    best_value = max(per_value, key=per_value.get)
    worst_value = min(per_value, key=per_value.get)
    ace_value = float(per_value[best_value] - per_value[worst_value])

    return ACEResult(
        variable=str(variable),
        ace=ace_value,
        kind=str(kind),
        per_value_probs=per_value,
        best_value=int(best_value),
        worst_value=int(worst_value),
        do_conditions=do_dict,
        ob_conditions=ob_dict,
        diagnostics={
            "ordering": tuple(ordering),
            "target": str(target),
            "desired_bins": tuple(int(b) for b in desired_bins),
            "num_samples": int(calc.num_samples),
        },
    )
