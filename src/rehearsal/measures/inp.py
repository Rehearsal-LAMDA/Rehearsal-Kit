"""Influence Power (INP) measure implementation.

INP for a node ``v`` along an ordering, given a target node and its desired
outcome bins, is::

    INP(v) = MEP(do(v)) - MEP(observe(v))

where MEP is the Maximum Expected Probability of the target landing in the
desired bin set, taken recursively over the remaining nodes between ``v``
and the target. Conditional INP allows pre-fixed ``do`` or ``observe`` on
upstream nodes.

The recursion respects ``task.alterable``: at intermediate nodes that are
*not* alterable, only the observation branch is considered (since ``do(v)``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from rehearsal.core import AUFTask, DesiredRegion
from rehearsal.measures.discretizer import UniformBinDiscretizer
from rehearsal.measures.mep import MEPCalculator
from rehearsal.models.nonlinear import NonlinearStructuralModel


@dataclass(frozen=True)
class INPResult:
    """Result record for a single ``compute_inp`` call."""

    variable: str
    inp: float
    mep_do: float
    mep_ob: float
    best_do_bin: int
    best_do_value: float
    do_conditions: dict[str, int] = field(default_factory=dict)
    ob_conditions: dict[str, int] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "inp": float(self.inp),
            "mep_do": float(self.mep_do),
            "mep_ob": float(self.mep_ob),
            "best_do_bin": int(self.best_do_bin),
            "best_do_value": float(self.best_do_value),
            "do_conditions": dict(self.do_conditions),
            "ob_conditions": dict(self.ob_conditions),
            "diagnostics": dict(self.diagnostics),
        }


def compute_inp(
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
) -> INPResult:
    """Compute INP for a single ``variable`` under the supplied ordering.

    ``calculator`` may be passed to share caches across multiple
    ``compute_inp`` calls within the same conditioning context.
    """

    if discretizer is None:
        raise ValueError("compute_inp requires a UniformBinDiscretizer (continuous outputs must be discretized).")
    _ = task  # Reserved for future task-aware validation; explicitly unused for now.

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
    inp_value, best_value, mep_do, mep_ob = calc.compute_inp(variable, do_conditions, ob_conditions)
    best_do_value = discretizer.get_continuous_value(variable, best_value)
    return INPResult(
        variable=str(variable),
        inp=float(inp_value),
        mep_do=float(mep_do),
        mep_ob=float(mep_ob),
        best_do_bin=int(best_value),
        best_do_value=float(best_do_value),
        do_conditions={k: int(v) for k, v in (do_conditions or {}).items()},
        ob_conditions={k: int(v) for k, v in (ob_conditions or {}).items()},
        diagnostics={
            "ordering": tuple(ordering),
            "target": str(target),
            "desired_bins": tuple(int(b) for b in desired_bins),
            "num_samples": int(calc.num_samples),
        },
    )


def compute_inp_for_variables(
    model: NonlinearStructuralModel,
    task: AUFTask,
    variables: Sequence[str],
    *,
    ordering: Sequence[str],
    target: str,
    desired_bins: Sequence[int],
    discretizer: UniformBinDiscretizer,
    do_conditions: Mapping[str, int] | None = None,
    ob_conditions: Mapping[str, int] | None = None,
    num_samples: int = 1000,
    rng: np.random.Generator | None = None,
) -> dict[str, INPResult]:
    """Compute INP for each variable in ``variables`` against a shared MEP cache."""

    calc = MEPCalculator(
        model,
        ordering=ordering,
        target_node=target,
        desired_bins=desired_bins,
        discretizer=discretizer,
        alterable=task.alterable,
        num_samples=num_samples,
        rng=rng,
    )
    out: dict[str, INPResult] = {}
    for name in variables:
        out[str(name)] = compute_inp(
            model,
            task,
            name,
            ordering=ordering,
            target=target,
            desired_bins=desired_bins,
            discretizer=discretizer,
            do_conditions=do_conditions,
            ob_conditions=ob_conditions,
            num_samples=num_samples,
            rng=rng,
            calculator=calc,
        )
    return out


def desired_bins_from_region(
    target: str,
    desired_region: DesiredRegion,
    discretizer: UniformBinDiscretizer,
    outcome_variables: Sequence[str] | None = None,
) -> tuple[int, ...]:
    """Find the set of bins of ``target`` whose center satisfies the desired region.

    Used by examples and tests that already have a continuous
    :class:`DesiredRegion` and need to convert it into the discrete bin set
    consumed by :class:`MEPCalculator`. Only single-outcome regions are
    supported here; for multi-outcome desired regions, callers should provide
    ``desired_bins`` directly.
    """

    outcomes = tuple(outcome_variables) if outcome_variables is not None else (target,)
    if outcomes != (target,):
        raise ValueError(
            "desired_bins_from_region currently only supports single-outcome targets; "
            "pass desired_bins explicitly for multi-outcome regions."
        )

    centers = discretizer.get_bin_centers(target)
    matrix = desired_region.matrix
    vector = desired_region.vector
    if matrix.shape[1] != 1:
        raise ValueError(
            "desired_bins_from_region: desired region must have a single outcome dimension."
        )

    accepted: list[int] = []
    for bin_idx, center in enumerate(centers):
        y = np.array([float(center)], dtype=float)
        if bool(np.all(matrix @ y <= vector + 1e-12)):
            accepted.append(int(bin_idx))

    if not accepted:
        # Fall back to the bin whose center is closest to the region's midpoint
        # so downstream MEP calls always have a non-empty desired set.
        if matrix.shape == (2, 1):
            upper = float(vector[0]) / float(matrix[0, 0]) if matrix[0, 0] != 0 else None
            lower = -float(vector[1]) / float(matrix[1, 0]) if matrix[1, 0] != 0 else None
            if lower is not None and upper is not None:
                midpoint = 0.5 * (lower + upper)
                accepted.append(int(np.argmin(np.abs(centers - midpoint))))
        if not accepted:
            accepted.append(int(np.argmin(np.abs(centers))))

    return tuple(accepted)
