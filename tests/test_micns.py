import numpy as np
import pytest

from rehearsal.core import AUFTask, AlterationDomain, DecisionResult, DesiredRegion
from rehearsal.experiments.run import run_experiment_configs
from rehearsal.methods import MICNSRehearsal
from rehearsal.methods.registry import available_methods, create_method


def _fixture(seed=0):
    rng = np.random.default_rng(seed)
    n = 250
    x = rng.normal(size=n)
    z = 0.5 * x + rng.normal(scale=0.25, size=n)
    y = 0.4 * x + 1.2 * z + rng.normal(scale=0.15, size=n)
    task = AUFTask(
        observed_variables=("x",),
        alterable_variables=("z",),
        outcome_variables=("y",),
        desired_region=DesiredRegion.from_intervals({"y": (0.6, 1.1)}, variables=("y",)),
        alteration_domain=AlterationDomain({"z": (-1.5, 1.5)}, costs={"z": 2.0}),
        parents={"z": ("x",), "y": ("x", "z")},
        candidate_alteration_sets=(("z",),),
        variable_order=("x", "z", "y"),
    )
    return {"x": x, "z": z, "y": y}, task


def test_micns_fit_suggest_evaluate_returns_decision_result():
    data, task = _fixture(seed=3)
    method = MICNSRehearsal(seed=11).fit(data, task)

    result = method.suggest({"x": 0.3}, task)
    evaluation = method.evaluate(task, n_samples=25)

    assert isinstance(result, DecisionResult)
    assert -1.5 <= result.alterations["z"] <= 1.5
    assert 0.0 <= result.estimated_success_probability <= 1.0
    assert result.diagnostics["selected_candidate"] == ("z",)
    assert result.diagnostics["solver_status"] in {"weighted_least_squares", "clipped_weighted_least_squares"}
    assert 0.0 <= evaluation["empirical_success_rate"] <= 1.0
    assert evaluation["n_samples"] == 25


def test_micns_is_deterministic_for_fixed_seed():
    data, task = _fixture(seed=5)
    params = {"seed": 17, "time_bandwidth_fraction": 0.3}

    first = MICNSRehearsal(**params).fit(data, task).suggest({"x": -0.1}, task)
    second = MICNSRehearsal(**params).fit(data, task).suggest({"x": -0.1}, task)

    assert first.alterations == pytest.approx(second.alterations)
    assert first.estimated_success_probability == pytest.approx(second.estimated_success_probability)


def test_micns_registry_and_bermuda_runner_work():
    method = create_method("micns", {"seed": 9})

    assert "micns" in available_methods()
    assert isinstance(method, MICNSRehearsal)

    result = run_experiment_configs(
        "examples/micns/bermuda_example.py",
        seeds=(1,),
        method_name="micns",
        method_params={"time_bandwidth_fraction": 0.3},
        eval_samples=5,
        params={"n_data": 30},
    )

    assert result["name"] == "micns_bermuda"
    assert result["method"] == "micns"
    assert result["n_runs"] == 1
    assert result["runs"][0]["evaluation"]["eval_samples"] == 5
    assert 0.0 <= result["runs"][0]["decision"]["estimated_success_probability"] <= 1.0
