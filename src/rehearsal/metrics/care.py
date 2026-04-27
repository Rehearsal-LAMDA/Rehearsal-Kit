"""CARE metric helpers."""

from __future__ import annotations

import numpy as np

from rehearsal.metrics.probability import independent_normal_success_probability, normal_cdf, normal_pdf


def independent_normal_care_success(
    mean: np.ndarray,
    variance: np.ndarray,
    intervals: np.ndarray,
    *,
    min_std: float = 1e-8,
) -> float:
    """Compute independent Gaussian CARE success over interval outcomes."""

    return independent_normal_success_probability(mean, variance, intervals, min_std=min_std)
