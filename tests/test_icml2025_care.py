import numpy as np
import pytest

from rehearsal.core import AUFTask, AlterationDomain, DesiredRegion, DecisionResult
from rehearsal.methods import (
    CARERehearsal,
    circular_region_inner_care,
    desired_region_intervals_under_independence,
)


def _single_outcome_fixture(seed=7):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=240)
    z = 0.5 * x + rng.normal(scale=0.2, size=x.size)
    y = 0.3 * x + 1.4 * z + rng.normal(scale=0.15, size=x.size)
    task = AUFTask(
        observed_variables=("x",),
        alterable_variables=("z",),
        outcome_variables=("y",),
        desired_region=DesiredRegion.from_intervals({"y": (0.8, 1.2)}, variables=("y",)),
        alteration_domain=AlterationDomain({"z": (-2.0, 2.0)}),
        parents={"z": ("x",), "y": ("x", "z")},
        candidate_alteration_sets=(("z",),),
        variable_order=("x", "z", "y"),
    )
    return {"x": x, "z": z, "y": y}, task


def test_bounded_alteration_output_and_care_success_range():
    data, task = _single_outcome_fixture()
    method = CARERehearsal(seed=11).fit(data, task)

    result = method.suggest({"x": 0.25}, task)

    assert isinstance(result, DecisionResult)
    assert -2.0 <= result.alterations["z"] <= 2.0
    assert 0.0 <= result.estimated_success_probability <= 1.0
    assert 0.0 <= result.diagnostics["estimated_care_success"] <= 1.0
    assert result.cost >= 0.0


def test_scalar_outcome_accepts_vector_m_matrix():
    data, base_task = _single_outcome_fixture()
    task = AUFTask(
        observed_variables=base_task.observed_variables,
        alterable_variables=base_task.alterable_variables,
        outcome_variables=base_task.outcome_variables,
        desired_region=DesiredRegion(M=[1.0, -1.0], d=[1.2, -0.8]),
        alteration_domain=base_task.alteration_domain,
        parents=base_task.parents,
        candidate_alteration_sets=base_task.candidate_alteration_sets,
        variable_order=base_task.variable_order,
    )

    result = CARERehearsal(seed=11).fit(data, task).suggest({"x": 0.25}, task)

    assert -2.0 <= result.alterations["z"] <= 2.0
    assert 0.0 <= result.estimated_success_probability <= 1.0


def test_deterministic_under_fixed_seed():
    data, task = _single_outcome_fixture()
    first = CARERehearsal(seed=3).fit(data, task).suggest({"x": -0.1}, task)
    second = CARERehearsal(seed=3).fit(data, task).suggest({"x": -0.1}, task)

    assert first.alterations == pytest.approx(second.alterations)
    assert first.estimated_success_probability == pytest.approx(second.estimated_success_probability)


def test_works_on_tiny_synthetic_auf_task_and_evaluate():
    data, task = _single_outcome_fixture(seed=9)
    method = CARERehearsal(seed=19).fit(data, task)

    result = method.suggest({"x": 0.0}, task)
    evaluation = method.evaluate(task, n_samples=50)

    assert result.diagnostics["solver_status"] == "closed_form_1d"
    assert 0.0 <= evaluation["estimated_success_probability"] <= 1.0
    assert evaluation["n_samples"] == 50


def test_circular_region_inner_care_helper_for_identity_covariance():
    region = circular_region_inner_care(center=(0.0, 0.0), radius=np.sqrt(2.0), covariance=np.eye(2))

    intervals = desired_region_intervals_under_independence(region, ("y1", "y2"))

    assert intervals == pytest.approx(np.array([[-1.0, 1.0], [-1.0, 1.0]]))


def test_solver_diagnostics_fields_exist():
    data, task = _single_outcome_fixture()
    result = CARERehearsal(seed=5).fit(data, task).suggest({"x": 0.1}, task)

    for field in ("selected_candidate", "objective_value", "solver_status", "n_candidates"):
        assert field in result.diagnostics
    assert result.diagnostics["selected_candidate"] == ("z",)
    assert result.diagnostics["n_candidates"] == 1


def test_multidimensional_independent_interval_region_is_supported():
    rng = np.random.default_rng(4)
    x = rng.normal(size=260)
    z = 0.2 * x + rng.normal(scale=0.2, size=x.size)
    y1 = 0.5 * x + 1.0 * z + rng.normal(scale=0.15, size=x.size)
    y2 = -0.3 * x - 0.8 * z + rng.normal(scale=0.15, size=x.size)
    task = AUFTask(
        observed_variables=("x",),
        alterable_variables=("z",),
        outcome_variables=("y1", "y2"),
        desired_region=DesiredRegion.from_intervals(
            {"y1": (-0.25, 0.25), "y2": (-0.25, 0.25)},
            variables=("y1", "y2"),
        ),
        alteration_domain=AlterationDomain({"z": (-1.5, 1.5)}),
        parents={"z": ("x",), "y1": ("x", "z"), "y2": ("x", "z")},
        candidate_alteration_sets=(("z",),),
        variable_order=("x", "z", "y1", "y2"),
    )

    result = CARERehearsal(seed=1, max_iters=60).fit(
        {"x": x, "z": z, "y1": y1, "y2": y2},
        task,
    ).suggest({"x": 0.2}, task)

    assert -1.5 <= result.alterations["z"] <= 1.5
    assert result.diagnostics["assumed_independent_outcomes"] is True
    assert result.diagnostics["solver_status"] == "projected_gradient_independent"


def test_multidimensional_non_interval_region_raises_care_error():
    rng = np.random.default_rng(5)
    data = {
        "x": rng.normal(size=40),
        "z": rng.normal(size=40),
        "y1": rng.normal(size=40),
        "y2": rng.normal(size=40),
    }
    invalid_region = DesiredRegion(
        M=[[1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
        d=[1.0, 1.0, 1.0],
    )
    task = AUFTask(
        observed_variables=("x",),
        alterable_variables=("z",),
        outcome_variables=("y1", "y2"),
        desired_region=invalid_region,
        alteration_domain=AlterationDomain({"z": (-1.0, 1.0)}),
        parents={"y1": ("z",), "y2": ("z",)},
        variable_order=("x", "z", "y1", "y2"),
    )

    with pytest.raises(ValueError, match="does not satisfy CARE"):
        CARERehearsal().fit(data, task)
