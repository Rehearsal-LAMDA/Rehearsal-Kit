import numpy as np

from rehearsal.core import AUFTask, AlterationDomain, DesiredRegion, DecisionResult
from rehearsal.experiments.run import run_experiment_configs
from rehearsal.methods import OLEMRhRehearsal
from rehearsal.methods.registry import available_methods, create_method
from rehearsal.models import OrderBasedStructuralLearner, full_dag_from_order, learn_olem_order_indices


def _fixture(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=180)
    u = 0.8 * x + rng.normal(scale=0.25, size=x.size)
    z = -0.2 * x + 0.6 * u + rng.normal(scale=0.2, size=x.size)
    y = 0.4 * x + 1.1 * z + rng.normal(scale=0.06, size=x.size)
    task = AUFTask(
        observed_variables=("x",),
        alterable_variables=("z",),
        outcome_variables=("y",),
        desired_region=DesiredRegion.from_intervals({"y": (0.6, 1.0)}, variables=("y",)),
        alteration_domain=AlterationDomain({"z": (-1.5, 1.5)}),
        candidate_alteration_sets=(("z",),),
        variable_order=("x", "u", "z", "y"),
    )
    return {"x": x, "u": u, "z": z, "y": y}, task


def test_olem_order_learning_returns_permutation_and_parents():
    data, task = _fixture(seed=1)
    matrix = np.column_stack([data[name] for name in task.variable_order])
    order_indices, scores = learn_olem_order_indices(matrix)
    learned = OrderBasedStructuralLearner(max_parents=2).learn_order(data, task)

    assert sorted(order_indices) == list(range(4))
    assert set(scores).issubset(set(range(4)))
    assert set(learned.order) == set(task.variable_order)
    assert all(len(parents) <= 2 for parents in learned.parents.values())
    assert full_dag_from_order(("a", "b", "c"))["c"] == ("a", "b")


def test_olem_rh_fit_suggest_evaluate_and_bounds():
    data, task = _fixture(seed=2)
    method = OLEMRhRehearsal(seed=6, max_parents=2, n_mc_samples=64, epochs=20, patience=8, num_restarts=2)

    result = method.fit(data, task).suggest({"x": 0.1}, task)
    evaluation = method.evaluate(task, n_samples=20)

    assert isinstance(result, DecisionResult)
    assert -1.5 <= result.alterations["z"] <= 1.5
    assert 0.0 <= result.estimated_success_probability <= 1.0
    assert "order" in result.diagnostics
    assert 0.0 <= evaluation["estimated_success_probability"] <= 1.0


def test_olem_rh_registry():
    data, task = _fixture(seed=3)
    method = create_method("olem-rh", {"seed": 3, "n_mc_samples": 32, "epochs": 10, "num_restarts": 1})
    result = method.fit(data, task).suggest({"x": 0.0}, task)

    assert "olem-rh" in available_methods()
    assert 0.0 <= result.estimated_success_probability <= 1.0


def test_olem_rh_bermuda_example_runs_through_seeded_batch_runner():
    result = run_experiment_configs(
        "examples/olem_rh/bermuda_example.py",
        seeds=(1,),
        method_name="olem-rh",
        method_params={"n_mc_samples": 32, "epochs": 8, "num_restarts": 1, "max_parents": 3},
        eval_samples=5,
        params={"n_data": 30},
    )

    assert result["name"] == "olem_rh_bermuda"
    assert result["method"] == "olem-rh"
    assert result["n_runs"] == 1
    assert 0.0 <= result["runs"][0]["decision"]["estimated_success_probability"] <= 1.0
    assert result["runs"][0]["evaluation"]["eval_samples"] == 5
