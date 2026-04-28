"""Shared SEM-backed dataset utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from rehearsal.core import AUFTask
from rehearsal.models import LinearGaussianSRM

Theta = Mapping[str, Mapping[str, float]]
RegionPredicate = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class RehearsalDatasetSpec:
    """Runnable SEM-backed dataset specification for Rehearsal examples."""

    name: str
    task: AUFTask
    theta: Theta
    covariance: np.ndarray
    true_region_contains: RegionPredicate
    paper_claim: Mapping[str, float]
    default_n_data: int
    metadata: Mapping[str, object]


def generate_observational_data(
    spec: RehearsalDatasetSpec,
    n_samples: int,
    *,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Generate observational samples from a linear Gaussian SEM dataset."""

    rng = np.random.default_rng(seed)
    return simulate_sem(spec, n_samples, rng=rng)


def sample_observation(spec: RehearsalDatasetSpec, rng: np.random.Generator) -> dict[str, float]:
    """Sample the observed stage from the dataset's exogenous noise model."""

    noise = rng.multivariate_normal(np.zeros(len(spec.task.variable_order)), spec.covariance)
    index = {name: idx for idx, name in enumerate(spec.task.variable_order)}
    return {name: float(noise[index[name]]) for name in spec.task.observed}


def estimate_true_auf_success_rate(
    spec: RehearsalDatasetSpec,
    observation: Mapping[str, float],
    alterations: Mapping[str, float],
    n_samples: int,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> float:
    """Monte Carlo AUF success rate under the dataset's true region."""

    if rng is None:
        rng = np.random.default_rng(seed)
    simulated = simulate_sem(spec, n_samples, rng=rng, observation=observation, alterations=alterations)
    outcomes = np.column_stack([simulated[name] for name in spec.task.outcomes])
    return float(np.mean(spec.true_region_contains(outcomes)))


def simulate_sem(
    spec: RehearsalDatasetSpec,
    n_samples: int,
    *,
    rng: np.random.Generator,
    observation: Mapping[str, float] | None = None,
    alterations: Mapping[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Simulate samples from a dataset's linear Gaussian SEM."""

    model = LinearGaussianSRM(tuple(spec.task.variable_order), dict(spec.theta), spec.covariance)
    return model.simulate(
        n_samples,
        rng=rng,
        observation=observation,
        alterations=alterations,
    )


def theta_from_children(values: Mapping[str, Mapping[str, float]]) -> dict[str, dict[str, float]]:
    """Return a mutable theta mapping from a nested parent-to-child mapping."""

    return {parent: dict(children) for parent, children in values.items()}
