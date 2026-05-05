"""Small diagnostics and region helpers for MUR."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from rehearsal.core import AUFTask
from rehearsal.core.regions import desired_region_intervals_under_independence


def infer_mur_region_center(
    task: AUFTask,
    config: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, str]:
    """Infer or read the centrally symmetric desired-region center for MUR."""

    config_dict = dict(config or {})
    try:
        intervals = desired_region_intervals_under_independence(task.desired_region, task.outcomes)
    except ValueError:
        intervals = None
    if intervals is not None:
        return np.mean(intervals, axis=1).astype(float), "desired_region_intervals"

    metadata = dict(task.desired_region.metadata)
    if "center" in metadata:
        return _coerce_center(metadata["center"], task.outcomes), "desired_region.metadata.center"
    if "mur_region_center" in task.metadata:
        return _coerce_center(task.metadata["mur_region_center"], task.outcomes), "task.metadata.mur_region_center"
    if "region_center" in config_dict:
        return _coerce_center(config_dict["region_center"], task.outcomes), "fit_config.region_center"
    raise ValueError(
        "MUR requires a centrally symmetric desired-region center. "
        "Use interval DesiredRegion bounds, desired_region.metadata['center'], "
        "task.metadata['mur_region_center'], or fit_config['region_center']."
    )


def _coerce_center(value: Any, outcome_variables: Sequence[str]) -> np.ndarray:
    if isinstance(value, Mapping):
        missing = [name for name in outcome_variables if name not in value]
        if missing:
            raise ValueError(f"Region center mapping is missing outcomes: {missing}.")
        center = np.asarray([float(value[name]) for name in outcome_variables], dtype=float)
    else:
        center = np.asarray(value, dtype=float).reshape(-1)
    if center.shape != (len(tuple(outcome_variables)),):
        raise ValueError("Region center dimension must match outcome_variables.")
    if not np.all(np.isfinite(center)):
        raise ValueError("Region center must be finite.")
    return center
