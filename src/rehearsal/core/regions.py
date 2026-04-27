"""Desired-region helpers shared across rehearsal methods."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from rehearsal.core.types import DesiredRegion


def desired_region_intervals_under_independence(
    desired_region: DesiredRegion,
    outcome_variables: Sequence[str],
    *,
    atol: float = 1e-10,
) -> np.ndarray:
    """Return interval bounds when ``M y <= d`` is CARE for independent outcomes.

    With independent outcome dimensions, the CARE condition reduces to an
    axis-aligned box: each outcome dimension must have a finite lower and upper
    bound, and no constraint row may mix multiple outcome dimensions.
    """

    matrix = desired_region.matrix
    vector = desired_region.vector
    y_dim = len(outcome_variables)
    if matrix.shape[1] != y_dim:
        raise ValueError("Desired region dimension does not match outcome variables.")

    lower = np.full(y_dim, -np.inf, dtype=float)
    upper = np.full(y_dim, np.inf, dtype=float)
    for row, bound in zip(matrix, vector):
        nonzero = np.flatnonzero(np.abs(row) > atol)
        if nonzero.size == 0:
            if bound < -atol:
                raise ValueError("Desired region is empty due to an impossible constraint.")
            continue
        if nonzero.size != 1:
            names = ", ".join(outcome_variables)
            raise ValueError(
                "Desired region does not satisfy CARE under independent multidimensional "
                f"Y={names}: each constraint must involve exactly one outcome dimension."
            )
        dim = int(nonzero[0])
        coef = float(row[dim])
        threshold = float(bound) / coef
        if coef > 0:
            upper[dim] = min(upper[dim], threshold)
        else:
            lower[dim] = max(lower[dim], threshold)

    missing_bounds = []
    for idx, name in enumerate(outcome_variables):
        if not np.isfinite(lower[idx]) or not np.isfinite(upper[idx]):
            missing_bounds.append(name)
        elif lower[idx] > upper[idx] + atol:
            raise ValueError(f"Desired interval for outcome {name!r} is empty.")
    if missing_bounds:
        missing = ", ".join(missing_bounds)
        raise ValueError(
            "Desired region does not satisfy CARE under independent outcomes: "
            f"missing finite interval bounds for {missing}."
        )

    return np.column_stack([lower, upper])


def circular_region_inner_care(
    center: Sequence[float],
    radius: float,
    covariance: Sequence[Sequence[float]] | None = None,
) -> DesiredRegion:
    """Construct the ICML 2025 inner CARE embedding for a circular region."""

    center_arr = np.asarray(center, dtype=float).reshape(-1)
    if center_arr.size == 0:
        raise ValueError("center must contain at least one dimension.")
    if radius <= 0 or not np.isfinite(radius):
        raise ValueError("radius must be finite and positive.")

    dim = center_arr.size
    if covariance is None:
        covariance_arr = np.eye(dim)
    else:
        covariance_arr = np.asarray(covariance, dtype=float)
    if covariance_arr.shape != (dim, dim):
        raise ValueError("covariance shape must match center dimension.")

    eigenvalues, eigenvectors = np.linalg.eigh(covariance_arr)
    if np.any(eigenvalues <= 0):
        raise ValueError("covariance must be positive definite.")
    q_matrix = eigenvectors.T
    lambda_matrix = np.diag(1.0 / np.sqrt(eigenvalues))
    signed_identity = np.vstack([np.eye(dim), -np.eye(dim)])
    unsigned_identity = np.vstack([np.eye(dim), np.eye(dim)])
    matrix = signed_identity @ lambda_matrix @ q_matrix
    vector = (
        unsigned_identity @ lambda_matrix @ np.ones(dim) * (radius / np.sqrt(dim))
        + matrix @ center_arr
    )
    return DesiredRegion(
        matrix,
        vector,
        metadata={"kind": "circular_inner_care", "center": tuple(center_arr), "radius": float(radius)},
    )
