import json

import numpy as np
import pytest

from rehearsal.core import AUFTask, AlterationDomain, DecisionResult, DesiredRegion
from rehearsal.experiments.run import run_experiment_configs
from rehearsal.methods import MURRehearsal
from rehearsal.methods.registry import create_method
from rehearsal.models import LinearTimeSeriesSRM
from rehearsal.optimizers import compute_mur_matrices, select_mur_action


VARIABLES = ("x", "z", "y")
A_SIMPLE = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [3.0, 2.0, 0.0],
    ],
    dtype=float,
)
B_SIMPLE = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.5],
    ],
    dtype=float,
)
COV_SIMPLE = 1e-3 * np.eye(3)


def _task(region=None, bounds=(-2.0, 2.0), candidates=(("z",),), metadata=None):
    return AUFTask(
        observed_variables=("x",),
        alterable_variables=("z",),
        outcome_variables=("y",),
        desired_region=region or DesiredRegion.from_intervals({"y": (0.9, 1.1)}, variables=("y",)),
        alteration_domain=AlterationDomain({"z": bounds}),
        parents={"y": ("x", "z")},
        candidate_alteration_sets=candidates,
        variable_order=VARIABLES,
        metadata=metadata or {"mur_lagged_parents": {"y": ("y",)}},
    )


def _data():
    return {
        "x": np.linspace(-0.2, 0.2, 8),
        "z": np.linspace(0.1, -0.1, 8),
        "y": np.linspace(0.0, 0.3, 8),
    }


def _fit_config(**extra):
    return {"mur_A": A_SIMPLE, "mur_B": B_SIMPLE, "mur_noise_covariance": COV_SIMPLE, **extra}


def _observation(prev=True):
    obs = {"x": 0.1}
    if prev:
        obs.update({"prev__x": 0.0, "prev__z": 0.0, "prev__y": 0.2})
    return obs


def test_tiny_linear_time_series_task_fit_suggest_evaluate():
    method = MURRehearsal(seed=3, n_mc_samples=20, max_iters=30).fit(_data(), _task(), _fit_config())

    result = method.suggest(_observation(), _task())
    evaluation = method.evaluate(_task(), n_samples=20)

    assert isinstance(result, DecisionResult)
    assert -2.0 <= result.alterations["z"] <= 2.0
    assert 0.0 <= result.estimated_success_probability <= 1.0
    assert 0.0 <= evaluation["estimated_success_probability"] <= 1.0
    assert evaluation["n_samples"] == 20


def test_registry_creates_mur_method():
    method = create_method("mur", {"variant": "gmur", "horizon": 0})

    assert isinstance(method, MURRehearsal)


def test_gmur_and_farmur_variants_run():
    for variant in ("gmur", "farmur"):
        method = MURRehearsal(variant=variant, horizon=1, seed=5, n_mc_samples=10).fit(
            _data(),
            _task(),
            _fit_config(),
        )
        result = method.suggest(_observation(), _task())
        assert result.diagnostics["variant"] == variant
        assert 0.0 <= result.estimated_success_probability <= 1.0


def test_invalid_variant_raises():
    with pytest.raises(ValueError, match="variant"):
        MURRehearsal(variant="not-a-method")


def test_horizon_zero_farmur_matches_gmur_current_action():
    gmur = MURRehearsal(variant="gmur", horizon=0, seed=7, n_mc_samples=20).fit(_data(), _task(), _fit_config())
    farmur = MURRehearsal(variant="farmur", horizon=0, seed=7, n_mc_samples=20).fit(_data(), _task(), _fit_config())

    first = gmur.suggest(_observation(), _task())
    second = farmur.suggest(_observation(), _task())

    assert first.alterations == pytest.approx(second.alterations)
    assert first.diagnostics["H_shape"] == second.diagnostics["H_shape"]


def test_output_respects_alteration_domain_bounds():
    task = _task(region=DesiredRegion.from_intervals({"y": (100.0, 101.0)}, variables=("y",)), bounds=(-0.1, 0.1))
    result = MURRehearsal(seed=9, n_mc_samples=5, max_iters=40).fit(_data(), task, _fit_config()).suggest(
        _observation(),
        task,
    )

    assert -0.1 <= result.alterations["z"] <= 0.1


def test_fixed_seed_is_deterministic():
    first = MURRehearsal(seed=11, n_mc_samples=25, num_restarts=3).fit(_data(), _task(), _fit_config()).suggest(
        _observation(),
        _task(),
    )
    second = MURRehearsal(seed=11, n_mc_samples=25, num_restarts=3).fit(_data(), _task(), _fit_config()).suggest(
        _observation(),
        _task(),
    )

    assert first.alterations == pytest.approx(second.alterations)
    assert first.estimated_success_probability == pytest.approx(second.estimated_success_probability)


def test_desired_region_center_from_intervals():
    result = MURRehearsal(seed=1, n_mc_samples=5).fit(_data(), _task(), _fit_config()).suggest(_observation(), _task())

    assert result.diagnostics["region_center"] == pytest.approx([1.0])
    assert result.diagnostics["region_center_source"] == "desired_region_intervals"


def test_desired_region_center_from_metadata_and_config():
    metadata_region = DesiredRegion(M=[[1.0]], d=[2.0], metadata={"center": (0.25,)})
    metadata_task = _task(region=metadata_region)
    metadata_result = MURRehearsal(seed=1, n_mc_samples=5).fit(_data(), metadata_task, _fit_config()).suggest(
        _observation(),
        metadata_task,
    )
    assert metadata_result.diagnostics["region_center"] == pytest.approx([0.25])
    assert metadata_result.diagnostics["region_center_source"] == "desired_region.metadata.center"

    config_region = DesiredRegion(M=[[1.0]], d=[2.0])
    config_task = _task(region=config_region)
    config_result = MURRehearsal(seed=1, n_mc_samples=5).fit(
        _data(),
        config_task,
        _fit_config(region_center=[0.4]),
    ).suggest(_observation(), config_task)
    assert config_result.diagnostics["region_center"] == pytest.approx([0.4])
    assert config_result.diagnostics["region_center_source"] == "fit_config.region_center"


def test_missing_non_interval_center_raises():
    task = _task(region=DesiredRegion(M=[[1.0]], d=[2.0]))

    with pytest.raises(ValueError, match="center"):
        MURRehearsal().fit(_data(), task, _fit_config())


def test_mur_matrices_match_hand_computed_toy_case():
    model = LinearTimeSeriesSRM(VARIABLES, A_SIMPLE, B_SIMPLE, COV_SIMPLE)
    bundle = compute_mur_matrices(model, _task(), ("z",), remaining_horizon=0)

    assert bundle.M == pytest.approx(np.array([[3.0]]))
    assert bundle.H == pytest.approx(np.array([[2.0]]))
    assert bundle.N == pytest.approx(np.array([[0.0, 0.0, 0.5]]))
    assert bundle.H.shape == (1, 1)


def test_reverse_block_order_executes_current_action_last_block():
    model = LinearTimeSeriesSRM(VARIABLES, A_SIMPLE, B_SIMPLE, COV_SIMPLE)
    task = _task(region=DesiredRegion.from_intervals({"y": (1.0, 1.0)}, variables=("y",)))
    selection = select_mur_action(
        model,
        task,
        (("z",),),
        x_t=[0.0],
        v_prev=[0.0, 0.0, 0.0],
        center=[1.0],
        remaining_horizon=1,
        rng=np.random.default_rng(13),
        n_probability_samples=0,
    )

    assert selection.bundle.block_order == "reverse_chronological_current_last"
    assert selection.current_action == pytest.approx(selection.z_sequence[-1:])
    assert selection.z_sequence[0] != pytest.approx(selection.current_action[0])


def test_rank_deficient_h_does_not_crash_and_reports_fallback():
    A = np.zeros((3, 3), dtype=float)
    B = np.zeros((3, 3), dtype=float)
    task = _task(region=DesiredRegion.from_intervals({"y": (1.0, 1.2)}, variables=("y",)))
    result = MURRehearsal(seed=17, n_mc_samples=5).fit(
        _data(),
        task,
        {"mur_A": A, "mur_B": B, "mur_noise_covariance": COV_SIMPLE},
    ).suggest(_observation(), task)

    assert result.diagnostics["H_rank"] == 0
    assert result.diagnostics["used_pinv"] is True
    assert result.diagnostics["solver_status"] in {"unbounded_within_bounds", "projected_gradient_box_qp"}


def test_previous_state_prefix_and_fallback_diagnostics():
    prefixed = MURRehearsal(seed=19, n_mc_samples=5).fit(_data(), _task(), _fit_config()).suggest(
        _observation(prev=True),
        _task(),
    )
    fallback = MURRehearsal(seed=19, n_mc_samples=5).fit(_data(), _task(), _fit_config()).suggest(
        _observation(prev=False),
        _task(),
    )

    assert prefixed.diagnostics["previous_state_source"] == "observation_previous_state_prefix"
    assert fallback.diagnostics["previous_state_source"] == "fit_data_last_row"


def test_multiple_candidates_return_legal_selected_candidate():
    variables = ("x", "z1", "z2", "y")
    A = np.zeros((4, 4), dtype=float)
    A[3, 1] = 0.5
    A[3, 2] = 1.0
    B = np.zeros((4, 4), dtype=float)
    task = AUFTask(
        observed_variables=("x",),
        alterable_variables=("z1", "z2"),
        outcome_variables=("y",),
        desired_region=DesiredRegion.from_intervals({"y": (0.9, 1.1)}, variables=("y",)),
        alteration_domain=AlterationDomain({"z1": (-2.0, 2.0), "z2": (-2.0, 2.0)}),
        parents={"y": ("z1", "z2")},
        candidate_alteration_sets=(("z1",), ("z2",)),
        variable_order=variables,
    )
    data = {name: np.zeros(6) for name in variables}
    obs = {"x": 0.0, "prev__x": 0.0, "prev__z1": 0.0, "prev__z2": 0.0, "prev__y": 0.0}

    result = MURRehearsal(seed=23, n_mc_samples=5).fit(
        data,
        task,
        {"mur_A": A, "mur_B": B, "mur_noise_covariance": 1e-3 * np.eye(4)},
    ).suggest(obs, task)

    assert tuple(result.diagnostics["selected_candidate"]) in task.candidate_alteration_sets
    assert set(result.alterations).issubset(set(task.alterable))


def test_diagnostics_are_json_friendly():
    result = MURRehearsal(seed=29, n_mc_samples=5).fit(_data(), _task(), _fit_config()).suggest(_observation(), _task())

    json.dumps(dict(result.diagnostics))


def test_mur_toy_runner_smoke_for_both_variants():
    gmur = run_experiment_configs(
        "examples/mur/mur_toy_experiment.py",
        seeds=(1,),
        method_name="mur",
        method_params={"variant": "gmur", "horizon": 0, "n_mc_samples": 10, "max_iters": 10},
        eval_samples=5,
        params={"n_samples": 30},
    )
    farmur = run_experiment_configs(
        "examples/mur/mur_toy_experiment.py",
        seeds=(1,),
        method_name="mur",
        method_params={"variant": "farmur", "horizon": 1, "n_mc_samples": 10, "max_iters": 10},
        eval_samples=5,
        params={"n_samples": 30},
    )

    assert gmur["name"] == "mur_toy"
    assert farmur["name"] == "mur_toy"
    assert "true_auf_success_rate" in gmur["runs"][0]["evaluation"]
    assert "true_auf_success_rate" in farmur["runs"][0]["evaluation"]


def test_mur_bermuda_runner_smoke_and_horizon_zero_degeneracy():
    gmur = run_experiment_configs(
        "examples/mur/bermuda_example.py",
        seeds=(1,),
        method_name="mur",
        method_params={"variant": "gmur", "horizon": 0, "n_mc_samples": 10, "max_iters": 10},
        eval_samples=5,
        params={"n_data": 20},
    )
    farmur = run_experiment_configs(
        "examples/mur/bermuda_example.py",
        seeds=(1,),
        method_name="mur",
        method_params={"variant": "farmur", "horizon": 0, "n_mc_samples": 10, "max_iters": 10},
        eval_samples=5,
        params={"n_data": 20},
    )

    gmur_run = gmur["runs"][0]
    farmur_run = farmur["runs"][0]
    assert gmur["name"] == "mur_bermuda"
    assert gmur_run["metadata"]["lagged_matrix_profile"] == "zero"
    assert gmur_run["metadata"]["alterable_variables"] == ["DIC", "TA", "Omega", "Chla", "Nutrients_PC1"]
    assert gmur_run["decision"]["diagnostics"]["stationarity_spectral_radius"] == 0.0
    assert gmur_run["evaluation"]["eval_samples"] == 5
    assert 0.0 <= gmur_run["evaluation"]["true_auf_success_rate"] <= 1.0
    assert gmur_run["decision"]["alterations"] == pytest.approx(farmur_run["decision"]["alterations"])
    assert gmur_run["evaluation"]["true_auf_success_rate"] == pytest.approx(
        farmur_run["evaluation"]["true_auf_success_rate"]
    )
