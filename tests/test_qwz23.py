import numpy as np

from rehearsal.core import AUFTask, AlterationDomain, DecisionResult, DesiredRegion
from rehearsal.experiments.run import run_experiment_configs
from rehearsal.methods import QWZ23Rehearsal
from rehearsal.methods.registry import available_methods, create_method


def _fixture(seed=0):
    rng = np.random.default_rng(seed)
    n = 180
    x = rng.normal(size=n)
    z = 0.6 * x + rng.normal(scale=0.3, size=n)
    y = 0.2 * x + 1.4 * z + rng.normal(scale=0.2, size=n)
    task = AUFTask(
        observed_variables=("x",),
        alterable_variables=("z",),
        outcome_variables=("y",),
        desired_region=DesiredRegion.from_intervals({"y": (0.7, 1.0)}, variables=("y",)),
        alteration_domain=AlterationDomain({"z": (-1.2, 1.2)}),
        parents={"z": ("x",), "y": ("x", "z")},
        candidate_alteration_sets=(("z",),),
        variable_order=("x", "z", "y"),
    )
    return {"x": x, "z": z, "y": y}, task


def test_qwz23_fit_suggest_evaluate_exposes_required_components():
    data, task = _fixture(seed=2)
    method = QWZ23Rehearsal(seed=13, n_graph_samples=4, interval_grid_size=15).fit(data, task)

    result = method.suggest({"x": 0.15}, task)
    evaluation = method.evaluate(task, n_samples=20)

    assert isinstance(result, DecisionResult)
    assert -1.2 <= result.alterations["z"] <= 1.2
    assert 0.0 <= result.estimated_success_probability <= 1.0
    assert result.diagnostics["selected_candidate"] == ("z",)
    assert "graph_uncertainty" in result.diagnostics
    assert "feasible_intervals" in result.diagnostics
    assert "success_probability_bound" in result.diagnostics
    assert "information_gain" in result.diagnostics
    assert evaluation["graph_ensemble_size"] == 4
    assert evaluation["n_samples"] == 20


def test_qwz23_registry_and_bermuda_runner_work():
    method = create_method("qwz23", {"seed": 7})

    assert "qwz23" in available_methods()
    assert isinstance(method, QWZ23Rehearsal)

    result = run_experiment_configs(
        "examples/qwz23/bermuda_example.py",
        seeds=(1,),
        method_name="qwz23",
        method_params={"n_graph_samples": 4, "interval_grid_size": 15},
        eval_samples=5,
        params={"n_data": 30},
    )

    assert result["name"] == "qwz23_bermuda"
    assert result["method"] == "qwz23"
    assert result["n_runs"] == 1
    assert result["runs"][0]["evaluation"]["eval_samples"] == 5
    assert 0.0 <= result["runs"][0]["decision"]["estimated_success_probability"] <= 1.0
