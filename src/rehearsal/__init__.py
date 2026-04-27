"""Unified interfaces for rehearsal-learning method adapters."""

from rehearsal.core import (
    AUFTask,
    AlterationDomain,
    candidate_alteration_sets,
    circular_region_inner_care,
    coerce_data_matrix,
    DecisionResult,
    DesiredRegion,
    desired_region_intervals_under_independence,
    ExperimentResult,
    RehearsalMethod,
)
from rehearsal.methods import CARERehearsal, ICML2025CARERehearsal
from rehearsal.models import LinearGaussianSRM, LinearGaussianSRMLearner

__all__ = [
    "AUFTask",
    "AlterationDomain",
    "candidate_alteration_sets",
    "circular_region_inner_care",
    "coerce_data_matrix",
    "DecisionResult",
    "DesiredRegion",
    "desired_region_intervals_under_independence",
    "ExperimentResult",
    "CARERehearsal",
    "ICML2025CARERehearsal",
    "LinearGaussianSRM",
    "LinearGaussianSRMLearner",
    "RehearsalMethod",
]
