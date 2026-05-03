import numpy as np
import pytest

from rehearsal.core import AUFTask, AlterationDomain, DesiredRegion, DecisionResult
from rehearsal.experiments.run import run_experiment_configs
from rehearsal.methods import CMERehearsal
from rehearsal.methods.registry import available_methods, create_method
from rehearsal.metrics.cme import desired_region_surrogate_weights, rbf_kernel
from rehearsal.optimizers.cme import cme_action_objective, optimize_action_projected_gradient


def _cme_fixture(seed=0, *, include_metadata=True, multiple_candidates=False):
    rng = np.random.default_rng(seed)
    n_samples = 70
    x = rng.uniform(-1.0, 1.0, size=n_samples)
    u = rng.normal(scale=0.4, size=n_samples)
    a = 0.5 * x + 0.3 * u + rng.normal(scale=0.15, size=n_samples)
    b = -0.2 * x + rng.normal(scale=0.2, size=n_samples)
    y = 1.0 - (a - (0.35 * x + 0.2 * u)) ** 2 - 0.25 * (b + 0.1) ** 2
    y = y + rng.normal(scale=0.03, size=n_samples)
    data = {"x": x, "u": u, "a": a, "b": b, "y": y}
    alterable = ("a", "b") if multiple_candidates else ("a",)
    bounds = {"a": (-1.2, 1.2)}
    candidates = (("a",),)
    if multiple_candidates:
        bounds["b"] = (-1.0, 1.0)
        candidates = (("a",), ("b",))
    metadata = {"cme_environment_variables": ("u",)} if include_metadata else {}
    task = AUFTask(
        observed_variables=("x",),
        alterable_variables=alterable,
        outcome_variables=("y",),
        desired_region=DesiredRegion.from_intervals({"y": (0.85, 1.1)}, variables=("y",)),
        alteration_domain=AlterationDomain(bounds),
        candidate_alteration_sets=candidates,
        variable_order=("x", "u", "a", "b", "y"),
        metadata=metadata,
    )
    return data, task


def test_cme_fit_suggest_evaluate_on_tiny_synthetic_task():
    data, task = _cme_fixture(seed=1)
    method = CMERehearsal(
        seed=11,
        krr_lambda_alpha=0.05,
        krr_lambda_gamma=0.05,
        pgd_steps=25,
        num_restarts=5,
    ).fit(data, task)

    result = method.suggest({"x": 0.2}, task)
    evaluation = method.evaluate(task, n_samples=10)

    assert isinstance(result, DecisionResult)
    assert -1.2 <= result.alterations["a"] <= 1.2
    assert 0.0 <= result.estimated_success_probability <= 1.0
    assert result.cost >= 0.0
    assert 0.0 <= evaluation["empirical_hard_success_rate"] <= 1.0
    assert 0.0 <= evaluation["empirical_surrogate_success_rate"] <= 1.0
    assert evaluation["n_samples"] == 10


def test_cme_is_deterministic_under_fixed_seed():
    data, task = _cme_fixture(seed=2)
    params = {
        "seed": 7,
        "krr_lambda_alpha": 0.05,
        "krr_lambda_gamma": 0.05,
        "pgd_steps": 20,
        "num_restarts": 4,
    }

    first = CMERehearsal(**params).fit(data, task).suggest({"x": -0.1}, task)
    second = CMERehearsal(**params).fit(data, task).suggest({"x": -0.1}, task)

    assert first.alterations == pytest.approx(second.alterations)
    assert first.estimated_success_probability == pytest.approx(second.estimated_success_probability)
    assert first.diagnostics["objective_value"] == pytest.approx(second.diagnostics["objective_value"])


def test_cme_surrogate_weights_reward_desired_region():
    region = DesiredRegion.from_intervals({"y": (0.0, 1.0)}, variables=("y",))
    outcomes = np.array([[0.5], [0.8], [-1.0], [2.0]])

    weights = desired_region_surrogate_weights(outcomes, region, eta_surrogate=8.0)
    hard = desired_region_surrogate_weights(outcomes, region, eta_surrogate=0.0)

    assert weights[0] > weights[2]
    assert weights[1] > weights[3]
    assert hard.tolist() == [1.0, 1.0, 0.0, 0.0]


def test_cme_rbf_kernel_shape_symmetry_and_diagonal():
    values = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])

    kernel = rbf_kernel(values, values, bandwidth=1.5)
    cross = rbf_kernel(values[:2], values[1:], bandwidth=1.5)

    assert kernel.shape == (3, 3)
    assert cross.shape == (2, 2)
    assert kernel == pytest.approx(kernel.T)
    assert np.diag(kernel) == pytest.approx(np.ones(3))


def test_cme_pgd_optimizer_respects_bounds_and_returns_finite_objective():
    action_history = np.linspace(-1.0, 1.0, 21).reshape(-1, 1)
    omega = np.exp(-((action_history.reshape(-1) - 0.35) ** 2) / 0.08)
    lower = np.array([-0.5])
    upper = np.array([0.8])

    result = optimize_action_projected_gradient(
        action_history,
        omega,
        lower,
        upper,
        bandwidth=0.25,
        learning_rate=0.2,
        max_steps=40,
        num_restarts=4,
        rng=np.random.default_rng(3),
    )

    assert -0.5 <= result.action[0] <= 0.8
    assert np.isfinite(result.objective_value)
    assert result.objective_value >= cme_action_objective(np.array([-0.5]), action_history, omega, 0.25)
    assert 0.0 <= result.estimated_success_probability <= 1.0


def test_cme_environment_metadata_and_fallback_both_work():
    data, task_with_metadata = _cme_fixture(seed=3, include_metadata=True)
    result_with_metadata = CMERehearsal(seed=3, pgd_steps=15, num_restarts=3).fit(
        data,
        task_with_metadata,
    ).suggest({"x": 0.0}, task_with_metadata)

    _, task_without_metadata = _cme_fixture(seed=3, include_metadata=False)
    result_without_metadata = CMERehearsal(seed=3, pgd_steps=15, num_restarts=3).fit(
        data,
        task_without_metadata,
    ).suggest({"x": 0.0}, task_without_metadata)

    assert result_with_metadata.diagnostics["candidate_diagnostics"][0]["environment_variables"] == ("u",)
    assert result_without_metadata.diagnostics["candidate_diagnostics"][0]["environment_variables"] == ("u",)
    assert -1.2 <= result_without_metadata.alterations["a"] <= 1.2


def test_cme_multiple_candidates_diagnostics_and_registry():
    data, task = _cme_fixture(seed=4, multiple_candidates=True)
    method = create_method(
        "cme-rh",
        {"seed": 4, "pgd_steps": 15, "num_restarts": 3, "krr_lambda_alpha": 0.05},
    )

    result = method.fit(data, task).suggest({"x": 0.3}, task)

    assert "cme-rh" in available_methods()
    assert result.diagnostics["selected_candidate"] in (("a",), ("b",))
    assert result.diagnostics["n_candidates"] == 2
    for field in ("selected_candidate", "objective_value", "solver_status", "n_training_samples"):
        assert field in result.diagnostics


def test_cme_example_runs_through_seeded_batch_runner():
    result = run_experiment_configs(
        "examples/cme/cme_toy_experiment.py",
        seeds=(1,),
        method_name="cme-rh",
        method_params={"pgd_steps": 15, "num_restarts": 3},
        eval_samples=8,
        params={"n_samples": 50},
    )

    assert result["name"] == "cme_toy"
    assert result["method"] == "cme-rh"
    assert result["seeds"] == [1]
    assert result["n_runs"] == 1
    assert 0.0 <= result["runs"][0]["decision"]["estimated_success_probability"] <= 1.0
    assert result["runs"][0]["evaluation"]["eval_samples"] == 8


def test_cme_bermuda_example_runs_through_seeded_batch_runner():
    result = run_experiment_configs(
        "examples/cme/cme_bermuda_example.py",
        seeds=(1,),
        method_name="cme-rh",
        method_params={"pgd_steps": 5, "num_restarts": 2},
        eval_samples=5,
        params={"n_data": 30},
    )

    assert result["name"] == "cme_bermuda"
    assert result["method"] == "cme-rh"
    assert result["seeds"] == [1]
    assert result["n_runs"] == 1
    assert 0.0 <= result["runs"][0]["decision"]["estimated_success_probability"] <= 1.0
    assert result["runs"][0]["evaluation"]["eval_samples"] == 5
