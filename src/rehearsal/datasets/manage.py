"""Synthetic Manage dataset used by Rehearsal experiments."""

from __future__ import annotations

import numpy as np

from rehearsal.core import AUFTask, AlterationDomain, circular_region_inner_care
from rehearsal.datasets.sem import RehearsalDatasetSpec, theta_from_children
from rehearsal.models import parents_from_theta


def manage() -> RehearsalDatasetSpec:
    """Return the synthetic Manage SEM dataset."""

    nodes = (
        "competitor_feature",
        "economic_index",
        "competitor_raw_cost",
        "raw_cost",
        "self_pricing",
        "competitor_pricing",
        "total_profit",
        "custom_number",
    )
    theta = theta_from_children(
        {
            "competitor_feature": {"competitor_raw_cost": 10.0},
            "economic_index": {"raw_cost": 10.0},
            "self_pricing": {"total_profit": 0.9, "custom_number": -0.9},
            "raw_cost": {
                "competitor_pricing": 0.5,
                "self_pricing": 2.0,
                "total_profit": -1.0,
                "custom_number": 1.6,
            },
            "competitor_raw_cost": {"competitor_pricing": 1.3, "self_pricing": 0.4},
        }
    )
    covariance = np.array(
        [
            [0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.06, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.03, 0.016, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.016, 0.06, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.06, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.12],
        ],
        dtype=float,
    )
    center = np.array([1.2, 1.5], dtype=float)
    radius = 1.0
    desired_region = circular_region_inner_care(
        center,
        radius,
        covariance=np.diag([0.04, 0.12]),
    )
    task = AUFTask(
        observed_variables=("competitor_feature", "economic_index"),
        alterable_variables=("raw_cost", "self_pricing"),
        outcome_variables=("total_profit", "custom_number"),
        desired_region=desired_region,
        alteration_domain=AlterationDomain({"raw_cost": (-6.0, 6.0), "self_pricing": (-6.0, 6.0)}),
        parents=parents_from_theta(theta, nodes),
        candidate_alteration_sets=(("raw_cost", "self_pricing"),),
        variable_order=nodes,
        metadata={
            "dataset": "Synthetic Manage",
            "reference_paper": "ICML 2025 CARE",
            "true_region": {"kind": "circle", "center": tuple(center), "radius": radius},
        },
    )

    def true_region(y: np.ndarray) -> np.ndarray:
        values = np.asarray(y, dtype=float)
        return np.linalg.norm(values - center.reshape(1, -1), axis=1) <= radius

    return RehearsalDatasetSpec(
        name="manage",
        task=task,
        theta=theta,
        covariance=covariance,
        true_region_contains=true_region,
        paper_claim={
            "ours_care_success_percent": 99.01,
            "ours_100_round_success": 98.93,
            "ours_avg_time_ms": 5.86,
        },
        default_n_data=100,
        metadata={
            "source": "previous_works/05-ICML 2025/code/main_syn.py",
            "reference_paper": "ICML 2025 CARE",
        },
    )
