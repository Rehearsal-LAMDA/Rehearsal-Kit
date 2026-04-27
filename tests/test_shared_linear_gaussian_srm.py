import numpy as np
import pytest

from rehearsal.core import AUFTask, AlterationDomain, DesiredRegion
from rehearsal.models import LinearGaussianSRM, LinearGaussianSRMLearner, parents_from_theta, total_path_effects


def _task():
    return AUFTask(
        observed_variables=("x",),
        alterable_variables=("z",),
        outcome_variables=("y",),
        desired_region=DesiredRegion.from_intervals({"y": (-0.25, 0.25)}, variables=("y",)),
        alteration_domain=AlterationDomain({"z": (-1.0, 1.0)}),
        parents={"z": ("x",), "y": ("x", "z")},
        candidate_alteration_sets=(("z",),),
        variable_order=("x", "z", "y"),
    )


def test_total_path_effects_are_paper_neutral_model_utility():
    graph = {"x": {"z": 0.5, "y": 0.25}, "z": {"y": 2.0}, "y": {}}

    effects = total_path_effects(graph, "y")

    assert effects["z"] == pytest.approx(2.0)
    assert effects["x"] == pytest.approx(1.25)


def test_linear_gaussian_learner_fits_known_srm_coefficients():
    rng = np.random.default_rng(12)
    x = rng.normal(size=2000)
    z = 0.7 * x + rng.normal(scale=0.6, size=x.size)
    y = -0.2 * x + 1.5 * z + rng.normal(scale=0.03, size=x.size)

    result = LinearGaussianSRMLearner().fit({"x": x, "z": z, "y": y}, _task())
    model = result.model

    assert model.theta["x"]["z"] == pytest.approx(0.7, abs=0.02)
    assert model.theta["z"]["y"] == pytest.approx(1.5, abs=0.05)
    assert result.diagnostics["n_samples"] == 2000


def test_linear_gaussian_srm_effect_matrices_support_rehearsal_optimizers():
    model = LinearGaussianSRM(
        ("x", "z", "y"),
        {"x": {"z": 0.5, "y": 0.25}, "z": {"y": 2.0}, "y": {}},
        np.eye(3),
    )

    mat_a, mat_b, mat_c = model.effect_matrices(_task(), ("z",))

    assert mat_a.shape == (1, 1)
    assert mat_b.shape == (1, 1)
    assert mat_c.shape == (1, 3)
    assert mat_a[0, 0] == pytest.approx(0.25)
    assert mat_b[0, 0] == pytest.approx(2.0)


def test_linear_gaussian_srm_simulate_uses_shared_parent_map():
    theta = {"x": {"z": 1.0}, "z": {"y": 1.0}}
    model = LinearGaussianSRM(("x", "z", "y"), theta, 1e-8 * np.eye(3))
    rng = np.random.default_rng(3)

    samples = model.simulate(5, rng=rng, observation={"x": 2.0}, alterations={"z": -1.0})

    assert parents_from_theta(theta, ("x", "z", "y"))["y"] == ("z",)
    assert np.allclose(samples["x"], 2.0)
    assert np.allclose(samples["z"], -1.0)
    assert samples["y"].shape == (5,)
