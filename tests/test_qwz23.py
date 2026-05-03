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
    method = QWZ23Rehearsal(seed=13, n_graph_samples=4, n_optimization_samples=32, milp_time_limit=2).fit(data, task)

    result = method.suggest({"x": 0.15}, task)
    evaluation = method.evaluate(task, n_samples=20)

    assert isinstance(result, DecisionResult)
    assert -1.2 <= result.alterations["z"] <= 1.2
    assert 0.0 <= result.estimated_success_probability <= 1.0
    assert result.diagnostics["selected_candidate"] == ("z",)
    assert "graph_uncertainty" in result.diagnostics
    assert "success_probability_bound" in result.diagnostics
    assert result.diagnostics["sampled_success_maximization"] is True
    assert result.diagnostics["optimization_solver"] in {"milp", "random_search"}
    assert result.diagnostics["n_optimization_samples_per_graph"] == 32
    assert evaluation["graph_ensemble_size"] == 4
    assert evaluation["n_samples"] == 20


def test_qwz23_sampled_optimizer_supports_multivariate_alterations():
    rng = np.random.default_rng(4)
    n = 160
    x = rng.normal(size=n)
    z1 = 0.4 * x + rng.normal(scale=0.25, size=n)
    z2 = -0.3 * x + rng.normal(scale=0.25, size=n)
    y = 0.2 * x + 0.9 * z1 + 0.8 * z2 + rng.normal(scale=0.15, size=n)
    task = AUFTask(
        observed_variables=("x",),
        alterable_variables=("z1", "z2"),
        outcome_variables=("y",),
        desired_region=DesiredRegion.from_intervals({"y": (0.9, 1.1)}, variables=("y",)),
        alteration_domain=AlterationDomain({"z1": (-1.2, 1.2), "z2": (-1.2, 1.2)}),
        parents={"z1": ("x",), "z2": ("x",), "y": ("x", "z1", "z2")},
        candidate_alteration_sets=(("z1", "z2"),),
        variable_order=("x", "z1", "z2", "y"),
    )
    method = QWZ23Rehearsal(
        seed=5,
        n_graph_samples=3,
        n_optimization_samples=24,
        milp_time_limit=2,
        random_search_size=128,
    ).fit({"x": x, "z1": z1, "z2": z2, "y": y}, task)

    result = method.suggest({"x": 0.0}, task)

    assert result.diagnostics["selected_candidate"] == ("z1", "z2")
    assert set(result.alterations) == {"z1", "z2"}
    assert all(-1.2 <= value <= 1.2 for value in result.alterations.values())
    assert result.estimated_success_probability > 0.0


def test_qwz23_registry_and_bermuda_runner_work():
    method = create_method("qwz23", {"seed": 7})

    assert "qwz23" in available_methods()
    assert isinstance(method, QWZ23Rehearsal)

    result = run_experiment_configs(
        "examples/qwz23/bermuda_example.py",
        seeds=(1,),
        method_name="qwz23",
        method_params={"n_graph_samples": 4, "n_optimization_samples": 24, "milp_time_limit": 2},
        eval_samples=5,
        params={"n_data": 30},
    )

    assert result["name"] == "qwz23_bermuda"
    assert result["method"] == "qwz23"
    assert result["n_runs"] == 1
    assert result["runs"][0]["evaluation"]["eval_samples"] == 5
    assert 0.0 <= result["runs"][0]["decision"]["estimated_success_probability"] <= 1.0
