import numpy as np
import pytest

from rehearsal.core import AUFTask, AlterationDomain, DesiredRegion, DecisionResult
from rehearsal.experiments.run import run_experiment_configs
from rehearsal.methods import GradRhRehearsal
from rehearsal.methods.registry import available_methods, create_method
from rehearsal.models import NonlinearSRMLearner, NonlinearStructuralModel
from rehearsal.optimizers import desired_region_center_and_radius


def _fixture(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=160)
    z = 0.4 * x + rng.normal(scale=0.2, size=x.size)
    y = 0.5 + 0.3 * x + 1.2 * z + 0.2 * z * z + rng.normal(scale=0.05, size=x.size)
    task = AUFTask(
        observed_variables=("x",),
        alterable_variables=("z",),
        outcome_variables=("y",),
        desired_region=DesiredRegion.from_intervals({"y": (0.9, 1.2)}, variables=("y",)),
        alteration_domain=AlterationDomain({"z": (-1.5, 1.5)}),
        parents={"z": ("x",), "y": ("x", "z")},
        candidate_alteration_sets=(("z",),),
        variable_order=("x", "z", "y"),
    )
    return {"x": x, "z": z, "y": y}, task


def test_nonlinear_learner_exposes_linear_and_flow_style_predictors():
    data, task = _fixture(seed=1)
    linear = NonlinearSRMLearner(predictor_type="linear", feature_degree=2).fit(data, task).model
    flow = NonlinearSRMLearner(predictor_type="flow", feature_degree=2).fit(data, task).model

    assert isinstance(linear, NonlinearStructuralModel)
    assert linear.predictors["y"].parents == ("x", "z")
    assert flow.predictors["y"].predictor_type == "flow"


def test_grad_rh_fit_suggest_evaluate_and_bounds():
    data, task = _fixture(seed=2)
    method = GradRhRehearsal(seed=5, feature_degree=2, n_mc_samples=64, epochs=20, patience=8, num_restarts=2)

    result = method.fit(data, task).suggest({"x": 0.1}, task)
    evaluation = method.evaluate(task, n_samples=20)

    assert isinstance(result, DecisionResult)
    assert -1.5 <= result.alterations["z"] <= 1.5
    assert 0.0 <= result.estimated_success_probability <= 1.0
    assert result.diagnostics["selected_candidate"] == ("z",)
    assert 0.0 <= evaluation["estimated_success_probability"] <= 1.0


def test_grad_rh_registry_and_region_center():
    data, task = _fixture(seed=3)
    center, radius = desired_region_center_and_radius(task.desired_region)
    method = create_method("grad-rh", {"seed": 3, "n_mc_samples": 32, "epochs": 10, "num_restarts": 1})
    result = method.fit(data, task).suggest({"x": 0.0}, task)

    assert "grad-rh" in available_methods()
    assert center.tolist() == pytest.approx([1.05])
    assert radius == pytest.approx(0.15)
    assert 0.0 <= result.estimated_success_probability <= 1.0


def test_grad_rh_bermuda_example_runs_through_seeded_batch_runner():
    result = run_experiment_configs(
        "examples/grad_rh/bermuda_example.py",
        seeds=(1,),
        method_name="grad-rh",
        method_params={"n_mc_samples": 32, "epochs": 8, "num_restarts": 1},
        eval_samples=5,
        params={"n_data": 30},
    )

    assert result["name"] == "grad_rh_bermuda"
    assert result["method"] == "grad-rh"
    assert result["n_runs"] == 1
    assert 0.0 <= result["runs"][0]["decision"]["estimated_success_probability"] <= 1.0
    assert result["runs"][0]["evaluation"]["eval_samples"] == 5
