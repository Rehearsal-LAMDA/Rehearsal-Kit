"""Tests for the INP measure (``rehearsal.measures.inp``).

The tests use a tiny linear chain x -> u -> z -> y so the conditional
distributions are well understood. With z having a strong positive effect
on y, INP(z) should be clearly positive and the recursion's "best do"
should pick the high-z bin to push y into the desired region.
"""

from __future__ import annotations

import numpy as np
import pytest

from rehearsal.core import AUFTask, AlterationDomain, DesiredRegion
from rehearsal.measures import (
    INPResult,
    MEPCalculator,
    UniformBinDiscretizer,
    compute_inp,
    compute_inp_for_variables,
    desired_bins_from_region,
    enumerate_linear_extensions,
    select_best_total_order_by_mep,
)
from rehearsal.models import OrderBasedStructuralLearner


def _toy_chain(seed: int = 0, n: int = 600):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    u = 0.6 * x + rng.normal(scale=0.25, size=n)
    z = -0.3 * x + 0.7 * u + rng.normal(scale=0.2, size=n)
    y = 0.4 * x + 1.0 * z + rng.normal(scale=0.15, size=n)
    task = AUFTask(
        observed_variables=("x",),
        alterable_variables=("z",),
        outcome_variables=("y",),
        desired_region=DesiredRegion.from_intervals({"y": (0.5, 2.5)}, variables=("y",)),
        alteration_domain=AlterationDomain({"z": (-2.0, 2.0)}),
        candidate_alteration_sets=(("z",),),
        variable_order=("x", "u", "z", "y"),
        parents={"u": ("x",), "z": ("x", "u"), "y": ("x", "z")},
    )
    return {"x": x, "u": u, "z": z, "y": y}, task


def _fit_model(seed: int = 0):
    data, task = _toy_chain(seed)
    learner = OrderBasedStructuralLearner(max_parents=3)
    fit = learner.fit(data, task)
    return fit.model, task


def test_compute_inp_picks_high_z_bin_to_increase_y():
    model, task = _fit_model(seed=0)
    discr = UniformBinDiscretizer(task.all_variables(), n_bins=3, bin_range=(-3.0, 3.0))
    desired = desired_bins_from_region("y", task.desired_region, discr)

    result = compute_inp(
        model,
        task,
        "z",
        ordering=task.variable_order,
        target="y",
        desired_bins=desired,
        discretizer=discr,
        num_samples=400,
        rng=np.random.default_rng(11),
    )

    assert isinstance(result, INPResult)
    assert result.variable == "z"
    assert 0.0 <= result.mep_do <= 1.0 + 1e-9
    assert 0.0 <= result.mep_ob <= 1.0 + 1e-9
    # z has a strong positive effect on y; pushing z to the upper bin should
    # be the best alteration. Use a bias-free check tolerant to small MC noise.
    assert result.best_do_value > 0.5
    assert result.inp > 0.05


def test_conditional_inp_responds_to_upstream_observation():
    model, task = _fit_model(seed=1)
    discr = UniformBinDiscretizer(task.all_variables(), n_bins=3, bin_range=(-3.0, 3.0))
    desired = desired_bins_from_region("y", task.desired_region, discr)

    base = compute_inp(
        model, task, "z",
        ordering=task.variable_order, target="y",
        desired_bins=desired, discretizer=discr,
        num_samples=400, rng=np.random.default_rng(21),
    )
    conditioned = compute_inp(
        model, task, "z",
        ordering=task.variable_order, target="y",
        desired_bins=desired, discretizer=discr,
        ob_conditions={"x": discr.discretize("x", -1.0)},
        num_samples=400, rng=np.random.default_rng(22),
    )

    assert base.do_conditions == {} and base.ob_conditions == {}
    assert conditioned.ob_conditions == {"x": int(discr.discretize("x", -1.0))}
    # Conditioning should not crash and should still return a valid probability.
    assert 0.0 <= conditioned.mep_do <= 1.0 + 1e-9
    assert 0.0 <= conditioned.mep_ob <= 1.0 + 1e-9


def test_compute_inp_for_variables_shares_calculator():
    model, task = _fit_model(seed=2)
    discr = UniformBinDiscretizer(task.all_variables(), n_bins=3, bin_range=(-3.0, 3.0))
    desired = desired_bins_from_region("y", task.desired_region, discr)

    out = compute_inp_for_variables(
        model,
        task,
        ["u", "z"],
        ordering=task.variable_order,
        target="y",
        desired_bins=desired,
        discretizer=discr,
        num_samples=200,
        rng=np.random.default_rng(33),
    )

    assert set(out) == {"u", "z"}
    for record in out.values():
        assert isinstance(record, INPResult)
        # Every recursion ends with a valid probability in [0, 1].
        assert 0.0 <= record.mep_do <= 1.0 + 1e-9


def test_mep_calculator_rejects_target_as_start_node():
    model, task = _fit_model(seed=3)
    discr = UniformBinDiscretizer(task.all_variables(), n_bins=3, bin_range=(-3.0, 3.0))
    desired = desired_bins_from_region("y", task.desired_region, discr)
    calc = MEPCalculator(
        model,
        ordering=task.variable_order,
        target_node="y",
        desired_bins=desired,
        discretizer=discr,
        num_samples=100,
        rng=np.random.default_rng(0),
    )
    with pytest.raises(ValueError):
        calc.compute_inp("y")


def test_enumerate_linear_extensions_respects_partial_order():
    extensions = enumerate_linear_extensions(("a", "b", "c"), {"b": ("a",)})
    # Three permutations satisfy "a before b": (a,b,c), (a,c,b), (c,a,b)
    assert sorted(extensions) == sorted([("a", "b", "c"), ("a", "c", "b"), ("c", "a", "b")])

    extensions_full = enumerate_linear_extensions(("a", "b", "c"), {})
    assert len(extensions_full) == 6  # 3! = 6 with no constraints

    extensions_chain = enumerate_linear_extensions(("a", "b", "c", "d"), {"b": ("a",), "c": ("b",), "d": ("c",)})
    assert extensions_chain == [("a", "b", "c", "d")]


def test_enumerate_linear_extensions_detects_cycle():
    with pytest.raises(ValueError):
        enumerate_linear_extensions(("a", "b"), {"a": ("b",), "b": ("a",)})


def test_enumerate_linear_extensions_respects_max_extensions():
    capped = enumerate_linear_extensions(("a", "b", "c", "d"), {}, max_extensions=5)
    assert len(capped) == 5


def test_select_best_total_order_by_mep_returns_a_compatible_extension():
    model, task = _fit_model(seed=4)
    discr = UniformBinDiscretizer(task.all_variables(), n_bins=2, bin_range=(-3.0, 3.0))
    desired = desired_bins_from_region("y", task.desired_region, discr)

    selection = select_best_total_order_by_mep(
        model,
        task,
        target="y",
        desired_bins=desired,
        discretizer=discr,
        # Force x before u, but allow u and z to swap.
        predecessor_map={"u": ("x",), "y": ("z",)},
        variables=("x", "u", "z", "y"),
        start_node="u",
        num_samples=120,
        rng=np.random.default_rng(7),
        max_extensions=8,
    )

    # The best order must end at y and respect the constraints.
    assert selection.best_order[-1] == "y"
    assert selection.best_order.index("x") < selection.best_order.index("u")
    assert selection.best_order.index("z") < selection.best_order.index("y")
    assert 0.0 <= selection.best_mep <= 1.0 + 1e-9
    assert len(selection.candidates) >= 1
