from pathlib import Path

import numpy as np

from rehearsal.core import AUFTask, AlterationDomain, DecisionResult, DesiredRegion
from rehearsal.experiments.run import run_experiment_configs
from rehearsal.methods import MSRRehearsal
from rehearsal.methods.registry import available_methods, create_method


def _fixture(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=180)
    z = 0.4 * x + rng.normal(scale=0.2, size=x.size)
    y = 0.5 + 0.3 * x + 1.2 * z + rng.normal(scale=0.05, size=x.size)
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


def test_msr_fit_suggest_evaluate_and_stage_diagnostics():
    data, task = _fixture(seed=2)
    method = MSRRehearsal(
        seed=5,
        stages=(("x",), ("z",), ("y",)),
        stage_observables=(("x",), (), ()),
        n_mc_samples=64,
        epochs=20,
        patience=8,
        num_restarts=2,
    )

    result = method.fit(data, task).suggest({"x": 0.1}, task)
    evaluation = method.evaluate(task, n_samples=20)

    assert isinstance(result, DecisionResult)
    assert set(result.alterations).issubset({"z"})
    for value in result.alterations.values():
        assert -1.5 <= value <= 1.5
    assert 0.0 <= result.estimated_success_probability <= 1.0
    assert result.diagnostics["method_name"] == "Multi-Step Rehearsal"
    assert len(result.diagnostics["stage_diagnostics"]) == 3
    assert 0.0 <= evaluation["estimated_success_probability"] <= 1.0


def test_msr_missing_observation_sampler():
    data, task = _fixture(seed=2)

    def sampler(conditioning, missing, rng, stage_idx):
        return {name: -0.25 for name in missing}

    method = MSRRehearsal(
        seed=7,
        stages=(("x", "z"), ("z",), ("y",)),
        stage_observables=(("x", "z"), (), ()),
        missing_observation_sampler=sampler,
        lookahead=False,
        n_mc_samples=48,
        epochs=16,
        patience=8,
        num_restarts=2,
    )

    result = method.fit(data, task).suggest({"x": 0.1}, task)
    z_obs = result.diagnostics["stage_diagnostics"][0]["observations"]["z"]
    assert z_obs["source"] == "sampler"
    assert z_obs["value"] == -0.25
    assert result.diagnostics["truth_observation_imputation"] is True


def test_msr_registry():
    data, task = _fixture(seed=3)
    method = create_method(
        "msr",
        {
            "seed": 3,
            "stages": (("x",), ("z",), ("y",)),
            "stage_observables": (("x",), (), ()),
            "n_mc_samples": 32,
            "epochs": 10,
            "num_restarts": 1,
        },
    )
    result = method.fit(data, task).suggest({"x": 0.0}, task)

    assert "msr" in available_methods()
    assert 0.0 <= result.estimated_success_probability <= 1.0


def test_msr_bermuda_example_runs_through_seeded_batch_runner():
    example_path = Path(__file__).resolve().parents[1] / "examples" / "msr" / "bermuda_example.py"
    result = run_experiment_configs(
        example_path,
        seeds=(1,),
        method_name="msr",
        method_params={"n_mc_samples": 32, "epochs": 8, "num_restarts": 1},
        eval_samples=5,
        params={"n_data": 30},
    )

    assert result["name"] == "msr_bermuda"
    assert result["method"] == "msr"
    assert result["n_runs"] == 1
    assert 0.0 <= result["runs"][0]["decision"]["estimated_success_probability"] <= 1.0
    assert result["runs"][0]["evaluation"]["eval_samples"] == 5
