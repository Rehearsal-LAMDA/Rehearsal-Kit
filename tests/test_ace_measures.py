"""Tests for the ACE / CACE measures (``rehearsal.measures.ace``)."""

from __future__ import annotations

import numpy as np
import pytest

from rehearsal.core import AUFTask, AlterationDomain, DesiredRegion
from rehearsal.measures import (
    ACEResult,
    UniformBinDiscretizer,
    compute_ace,
    compute_cace,
    desired_bins_from_region,
)
from rehearsal.models import OrderBasedStructuralLearner


def _toy_chain_with_irrelevant(seed: int = 0, n: int = 600):
    """Return a chain x -> z -> y plus a totally disconnected variable ``w``.

    ``w`` has no causal effect on ``y``, so ACE(w) must be (numerically)
    zero. ``z`` has a strong positive effect on ``y``.
    """

    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = 0.7 * x + rng.normal(scale=0.2, size=n)
    y = 1.1 * z + rng.normal(scale=0.15, size=n)
    w = rng.normal(size=n)  # disconnected from x, z, y
    task = AUFTask(
        observed_variables=("x",),
        alterable_variables=("z", "w"),
        outcome_variables=("y",),
        desired_region=DesiredRegion.from_intervals({"y": (0.5, 2.5)}, variables=("y",)),
        alteration_domain=AlterationDomain({"z": (-2.0, 2.0), "w": (-2.0, 2.0)}),
        candidate_alteration_sets=(("z",),),
        variable_order=("x", "z", "w", "y"),
        parents={"z": ("x",), "y": ("z",), "w": ()},
    )
    return {"x": x, "z": z, "w": w, "y": y}, task


def _fit_model(seed: int = 0):
    data, task = _toy_chain_with_irrelevant(seed)
    learner = OrderBasedStructuralLearner(max_parents=3)
    fit = learner.fit(data, task)
    return fit.model, task


def test_compute_ace_returns_high_value_for_strong_cause():
    model, task = _fit_model(seed=1)
    discr = UniformBinDiscretizer(task.all_variables(), n_bins=3, bin_range=(-3.0, 3.0))
    desired = desired_bins_from_region("y", task.desired_region, discr)

    ace_z = compute_ace(
        model, task, "z",
        ordering=task.variable_order, target="y",
        desired_bins=desired, discretizer=discr,
        num_samples=400, rng=np.random.default_rng(11),
    )
    assert isinstance(ace_z, ACEResult)
    assert ace_z.kind == "ace"
    assert 0.0 <= ace_z.ace <= 1.0 + 1e-9
    # z is the only causal driver of y in our toy chain; ACE should be substantial.
    assert ace_z.ace > 0.2


def test_compute_ace_is_zero_for_disconnected_variable():
    model, task = _fit_model(seed=2)
    discr = UniformBinDiscretizer(task.all_variables(), n_bins=3, bin_range=(-3.0, 3.0))
    desired = desired_bins_from_region("y", task.desired_region, discr)

    ace_w = compute_ace(
        model, task, "w",
        ordering=task.variable_order, target="y",
        desired_bins=desired, discretizer=discr,
        num_samples=600, rng=np.random.default_rng(22),
    )
    # ``w`` is independent of y; the ACE should be near zero (small MC noise tolerated).
    assert ace_w.ace <= 0.08


def test_compute_cace_supports_do_and_observe_context():
    model, task = _fit_model(seed=3)
    discr = UniformBinDiscretizer(task.all_variables(), n_bins=3, bin_range=(-3.0, 3.0))
    desired = desired_bins_from_region("y", task.desired_region, discr)

    cace_z = compute_cace(
        model, task, "z",
        ordering=task.variable_order, target="y",
        desired_bins=desired, discretizer=discr,
        ob_conditions={"x": discr.discretize("x", -1.0)},
        num_samples=400, rng=np.random.default_rng(33),
    )
    assert cace_z.kind == "cace"
    assert cace_z.ob_conditions == {"x": int(discr.discretize("x", -1.0))}
    assert 0.0 <= cace_z.ace <= 1.0 + 1e-9


def test_compute_ace_rejects_variable_in_context():
    model, task = _fit_model(seed=4)
    discr = UniformBinDiscretizer(task.all_variables(), n_bins=3, bin_range=(-3.0, 3.0))
    desired = desired_bins_from_region("y", task.desired_region, discr)

    with pytest.raises(ValueError):
        compute_cace(
            model, task, "z",
            ordering=task.variable_order, target="y",
            desired_bins=desired, discretizer=discr,
            do_conditions={"z": 1},  # the variable being measured cannot also be in context
            num_samples=100, rng=np.random.default_rng(0),
        )
