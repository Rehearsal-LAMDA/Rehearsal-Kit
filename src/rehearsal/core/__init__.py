"""Core task and result contracts for Rehearsal."""

from rehearsal.core.data import candidate_alteration_sets, coerce_data_matrix
from rehearsal.core.regions import circular_region_inner_care, desired_region_intervals_under_independence
from rehearsal.core.types import (
    AUFTask,
    AlterationDomain,
    DecisionResult,
    DesiredRegion,
    ExperimentResult,
    RehearsalMethod,
)

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
    "RehearsalMethod",
]
