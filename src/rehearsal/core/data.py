"""Shared task/data helpers for rehearsal-learning adapters."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from rehearsal.core.types import AUFTask


def coerce_data_matrix(
    data: Mapping[str, Sequence[float]] | np.ndarray,
    task: AUFTask,
    config: Mapping[str, Any] | None = None,
    *,
    columns: Sequence[str] | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return a finite sample matrix and column names for a task.

    Paper-specific scripts historically mixed dictionaries, raw arrays, and
    hard-coded column orders. Framework code should normalize those variants at
    the boundary and keep downstream model learners column-order explicit.
    """

    config = dict(config or {})
    if isinstance(data, Mapping):
        resolved_columns = tuple(columns or task.all_variables())
        missing = [name for name in resolved_columns if name not in data]
        if missing:
            raise ValueError(f"Data mapping is missing variables: {missing}.")
        matrix = np.column_stack([np.asarray(data[name], dtype=float).reshape(-1) for name in resolved_columns])
    else:
        matrix = np.asarray(data, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("Array data must be two-dimensional.")
        columns_value = columns or config.get("columns", task.variable_order)
        if columns_value is None:
            raise ValueError("Array data requires config['columns'], task.variable_order, or columns=.")
        resolved_columns = tuple(columns_value)
        if matrix.shape[1] != len(resolved_columns):
            raise ValueError("Array column count does not match supplied column names.")

    if matrix.shape[0] == 0:
        raise ValueError("Training data must contain at least one sample.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Training data must be finite.")
    return matrix, tuple(resolved_columns)


def candidate_alteration_sets(
    task: AUFTask,
    override: Sequence[Sequence[str]] | None = None,
) -> tuple[tuple[str, ...], ...]:
    """Normalize and validate candidate rehearsal sets for a task."""

    raw = override if override is not None else task.candidate_alteration_sets
    if raw is None:
        raw = (task.alterable,)
    candidates = tuple(tuple(candidate) for candidate in raw)
    if not candidates:
        raise ValueError("At least one candidate alteration set is required.")
    alterable = set(task.alterable)
    for candidate in candidates:
        unknown = set(candidate) - alterable
        if unknown:
            raise ValueError(f"Candidate alteration set contains non-alterable variables: {sorted(unknown)}.")
    return candidates
