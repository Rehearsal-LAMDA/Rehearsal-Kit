"""Backward-compatible aliases for the ICML 2025 CARE reproduction code.

The Bermuda and Manage datasets are canonicalized in ``rehearsal.datasets.bermuda``
and ``rehearsal.datasets.manage``. This module stays as a compatibility layer
for old experiment scripts that imported the dataset factories from the CARE
paper-specific namespace.
"""

from __future__ import annotations

from pathlib import Path

from rehearsal.datasets.bermuda import bermuda, load_bermuda_standardized_data
from rehearsal.datasets.manage import manage
from rehearsal.datasets.sem import (
    RehearsalDatasetSpec,
    estimate_true_auf_success_rate,
    generate_observational_data,
    sample_observation,
)

ICML2025DatasetSpec = RehearsalDatasetSpec


def manage_icml2025() -> RehearsalDatasetSpec:
    """Return the generic Manage dataset used by ICML 2025 CARE."""

    return manage()


def bermuda_icml2025(
    data_path: str | Path | None = None,
    *,
    covariance_profile: str = "paper",
) -> RehearsalDatasetSpec:
    """Return the generic Bermuda dataset used by ICML 2025 CARE."""

    return bermuda(data_path, covariance_profile=covariance_profile)


__all__ = [
    "ICML2025DatasetSpec",
    "bermuda_icml2025",
    "estimate_true_auf_success_rate",
    "generate_observational_data",
    "load_bermuda_standardized_data",
    "manage_icml2025",
    "sample_observation",
]
