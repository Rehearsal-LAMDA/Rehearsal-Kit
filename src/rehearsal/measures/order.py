"""Partial-order utilities for INP.

When the user supplies a partial order rather than a total order, the
prompt's section 4.2 requires us to enumerate all compatible linear
extensions, evaluate the Maximum Expected Probability (MEP) for each, and
pick the extension whose MEP at the chosen ``start_node`` is largest.

The partial order is represented exactly like ``AUFTask.parents``:
``{var: tuple_of_must_precede_this}``. ``enumerate_linear_extensions`` yields
all topological sorts that respect those predecessor constraints and
``select_best_total_order_by_mep`` evaluates each one through a fresh
:class:`MEPCalculator`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np

from rehearsal.core import AUFTask
from rehearsal.measures.discretizer import UniformBinDiscretizer
from rehearsal.measures.mep import MEPCalculator
from rehearsal.models.nonlinear import NonlinearStructuralModel


@dataclass(frozen=True)
class OrderSelectionResult:
    """Result returned by :func:`select_best_total_order_by_mep`."""

    best_order: tuple[str, ...]
    best_mep: float
    best_start_node: str
    candidates: tuple[tuple[tuple[str, ...], float], ...]
    truncated: bool = False
    diagnostics: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "best_order": list(self.best_order),
            "best_mep": float(self.best_mep),
            "best_start_node": self.best_start_node,
            "candidates": [
                {"order": list(order), "mep": float(mep)} for order, mep in self.candidates
            ],
            "truncated": bool(self.truncated),
            "diagnostics": dict(self.diagnostics),
        }


def enumerate_linear_extensions(
    variables: Sequence[str],
    predecessor_map: Mapping[str, Iterable[str]] | None = None,
    *,
    max_extensions: int | None = None,
) -> list[tuple[str, ...]]:
    """Return all linear extensions of the supplied partial order.

    Parameters
    ----------
    variables:
        The variables to order. Duplicates are rejected.
    predecessor_map:
        Mapping ``{var: tuple_of_must_precede_this}``. A variable that is
        absent from the mapping is treated as having no predecessor
        constraints. Predecessors that are not in ``variables`` are silently
        ignored (mirroring how ``AUFTask.parents`` may reference variables
        outside the current scope).
    max_extensions:
        Optional cap on the number of extensions returned. The enumerator
        stops early once this many distinct extensions have been generated.
    """

    items = list(dict.fromkeys(str(v) for v in variables))
    if len(items) != len(variables):
        raise ValueError("enumerate_linear_extensions: variables must be unique.")
    name_set = set(items)

    raw_pred: dict[str, set[str]] = {}
    if predecessor_map:
        for child, parents in predecessor_map.items():
            child_str = str(child)
            if child_str not in name_set:
                continue
            raw_pred[child_str] = {str(p) for p in parents if str(p) in name_set and str(p) != child_str}
    for name in items:
        raw_pred.setdefault(name, set())

    _detect_cycle(items, raw_pred)

    cap = int(max_extensions) if max_extensions is not None else None
    results: list[tuple[str, ...]] = []

    def helper(remaining: set[str], current: list[str]) -> bool:
        if cap is not None and len(results) >= cap:
            return False  # signal to stop
        if not remaining:
            results.append(tuple(current))
            return cap is None or len(results) < cap
        ready = sorted(name for name in remaining if not (raw_pred[name] & remaining))
        if not ready:
            raise ValueError("Partial order has no linear extension (graph is cyclic).")
        for choice in ready:
            current.append(choice)
            keep_going = helper(remaining - {choice}, current)
            current.pop()
            if not keep_going:
                return False
        return True

    helper(set(items), [])
    return results


def select_best_total_order_by_mep(
    model: NonlinearStructuralModel,
    task: AUFTask,
    *,
    target: str,
    desired_bins: Sequence[int],
    discretizer: UniformBinDiscretizer,
    predecessor_map: Mapping[str, Iterable[str]] | None = None,
    variables: Sequence[str] | None = None,
    start_node: str | None = None,
    do_conditions: Mapping[str, int] | None = None,
    ob_conditions: Mapping[str, int] | None = None,
    num_samples: int = 1000,
    rng: np.random.Generator | None = None,
    max_extensions: int | None = None,
) -> OrderSelectionResult:
    """Pick the linear extension that maximises MEP at ``start_node``.

    ``predecessor_map`` defaults to ``task.parents`` when not supplied, so a
    user who provides only the task itself gets the same behaviour as
    "partial order = the task's parent DAG".
    """

    pred_map: Mapping[str, Iterable[str]] = predecessor_map if predecessor_map is not None else task.parents
    var_list = tuple(variables) if variables is not None else tuple(task.all_variables())
    if target not in var_list:
        raise ValueError(f"target {target!r} must appear in the variables to be ordered.")

    extensions = enumerate_linear_extensions(var_list, pred_map, max_extensions=max_extensions)
    if not extensions:
        raise ValueError("No linear extensions exist for the supplied partial order.")
    truncated = max_extensions is not None and len(extensions) >= int(max_extensions)

    candidates: list[tuple[tuple[str, ...], float]] = []
    best_order: tuple[str, ...] | None = None
    best_mep = -np.inf
    best_start: str | None = None

    skipped = 0
    for order in extensions:
        target_idx = order.index(target)
        if start_node is None:
            if target_idx == 0:
                skipped += 1
                continue
            chosen_start = order[0]
        else:
            if start_node not in order:
                skipped += 1
                continue
            if order.index(start_node) >= target_idx:
                # MEPCalculator's recursion only walks from start_node up to (but
                # not including) target, so any extension that places start_node
                # at-or-after target is structurally invalid for INP/MEP.
                skipped += 1
                continue
            chosen_start = start_node
        calc = MEPCalculator(
            model,
            ordering=order,
            target_node=target,
            desired_bins=desired_bins,
            discretizer=discretizer,
            alterable=task.alterable,
            num_samples=num_samples,
            rng=rng,
        )
        mep_value = calc.compute_mep(chosen_start, do_conditions, ob_conditions)
        candidates.append((order, float(mep_value)))
        if mep_value > best_mep:
            best_mep = float(mep_value)
            best_order = order
            best_start = chosen_start

    if best_order is None:
        raise ValueError("None of the enumerated extensions placed start_node before the target.")

    return OrderSelectionResult(
        best_order=tuple(best_order),
        best_mep=float(best_mep),
        best_start_node=str(best_start),
        candidates=tuple(candidates),
        truncated=bool(truncated),
        diagnostics={
            "n_candidates": len(candidates),
            "n_skipped": int(skipped),
            "max_extensions": max_extensions,
            "target": str(target),
            "desired_bins": tuple(int(b) for b in desired_bins),
        },
    )


def _detect_cycle(variables: Sequence[str], predecessor_map: Mapping[str, set[str]]) -> None:
    color: dict[str, int] = {name: 0 for name in variables}

    def dfs(node: str) -> None:
        color[node] = 1
        for pred in predecessor_map.get(node, ()):
            if color[pred] == 1:
                raise ValueError(
                    f"Partial order contains a cycle involving {node!r} and {pred!r}."
                )
            if color[pred] == 0:
                dfs(pred)
        color[node] = 2

    for name in variables:
        if color[name] == 0:
            dfs(name)


def _iter_unique(values: Iterable[str]) -> Iterator[str]:
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        yield text
