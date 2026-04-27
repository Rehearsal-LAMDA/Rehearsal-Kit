"""Metric helpers for conditional-mean-embedding rehearsal."""

from __future__ import annotations

import numpy as np

from rehearsal.core import DesiredRegion
from rehearsal.metrics.probability import normal_cdf


def rbf_kernel(
    x_left: np.ndarray,
    x_right: np.ndarray,
    bandwidth: float = 1.0,
) -> np.ndarray:
    """Return an RBF kernel matrix without requiring scipy."""

    left = _as_2d(x_left)
    right = _as_2d(x_right)
    sigma = float(bandwidth)
    if sigma <= 0.0 or not np.isfinite(sigma):
        raise ValueError("bandwidth must be finite and positive.")
    if left.shape[1] != right.shape[1]:
        raise ValueError("RBF inputs must have the same feature dimension.")

    left_norm = np.sum(left * left, axis=1, keepdims=True)
    right_norm = np.sum(right * right, axis=1, keepdims=True).T
    squared = np.maximum(left_norm + right_norm - 2.0 * left @ right.T, 0.0)
    return np.exp(-squared / (2.0 * sigma * sigma))


def median_or_mean_bandwidth(
    values: np.ndarray,
    *,
    method: str = "legacy_mean",
    min_bandwidth: float = 1e-6,
) -> float:
    """Compute a deterministic pairwise-distance bandwidth.

    ``legacy_mean`` matches the unpublished solver's ``2 * mean(pdist(X))``
    heuristic. ``median`` is useful for small non-linear fixtures.
    """

    matrix = _as_2d(values)
    if matrix.shape[0] < 2:
        return 1.0

    row, col = np.triu_indices(matrix.shape[0], k=1)
    distances = np.linalg.norm(matrix[row] - matrix[col], axis=1)
    if distances.size == 0:
        return 1.0

    if method == "legacy_mean":
        bandwidth = 2.0 * float(np.mean(distances))
    elif method == "mean":
        bandwidth = float(np.mean(distances))
    elif method == "median":
        bandwidth = float(np.median(distances))
    else:
        raise ValueError("method must be 'legacy_mean', 'mean', or 'median'.")
    if not np.isfinite(bandwidth) or bandwidth <= min_bandwidth:
        return 1.0
    return bandwidth


def desired_region_surrogate_weights(
    outcomes: np.ndarray,
    desired_region: DesiredRegion,
    *,
    eta_surrogate: float | None = 10.0,
) -> np.ndarray:
    """Return smooth CME target weights for ``M y <= d``.

    With positive ``eta_surrogate`` this uses the legacy probit-style smooth
    indicator. With ``None`` or non-positive values it returns the hard desired
    region indicator.
    """

    y_values = _as_2d(outcomes)
    if y_values.shape[1] != desired_region.matrix.shape[1]:
        raise ValueError("Outcome dimension does not match desired region.")

    if eta_surrogate is None or float(eta_surrogate) <= 0.0:
        return np.asarray(desired_region.contains(y_values), dtype=float)

    constraints = desired_region.vector.reshape(1, -1) - y_values @ desired_region.matrix.T
    probit_values = normal_cdf(float(eta_surrogate) * constraints)
    weights = np.prod(probit_values, axis=1)
    return np.clip(weights, 0.0, 1.0)


def _as_2d(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        return matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        raise ValueError("values must be one- or two-dimensional.")
    return matrix
