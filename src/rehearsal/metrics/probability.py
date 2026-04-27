"""Probability utilities used by lightweight method adapters."""

from __future__ import annotations

import math

import numpy as np


def normal_cdf(x: np.ndarray | float) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    erf = np.vectorize(math.erf, otypes=[float])
    return 0.5 * (1.0 + erf(values / math.sqrt(2.0)))


def normal_pdf(x: np.ndarray | float) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    return np.exp(-0.5 * values * values) / math.sqrt(2.0 * math.pi)


def independent_normal_success_probability(
    mean: np.ndarray,
    variance: np.ndarray,
    intervals: np.ndarray,
    *,
    min_std: float = 1e-8,
) -> float:
    """Compute ``P(lower_i <= Y_i <= upper_i for all i)`` for independent normals."""

    mean = np.asarray(mean, dtype=float).reshape(-1)
    variance = np.asarray(variance, dtype=float).reshape(-1)
    intervals = np.asarray(intervals, dtype=float)
    if intervals.shape != (mean.size, 2):
        raise ValueError("intervals must have shape (n_outcomes, 2).")
    std = np.sqrt(np.maximum(variance, min_std * min_std))
    lower_z = (intervals[:, 0] - mean) / std
    upper_z = (intervals[:, 1] - mean) / std
    per_dim = normal_cdf(upper_z) - normal_cdf(lower_z)
    per_dim = np.clip(per_dim, 0.0, 1.0)
    return float(np.clip(np.prod(per_dim), 0.0, 1.0))
