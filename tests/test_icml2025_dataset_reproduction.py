from rehearsal.datasets.icml2025 import bermuda_icml2025, generate_observational_data, manage_icml2025
from rehearsal.experiments.icml2025_reproduce import run_reproduction


def test_manage_and_bermuda_dataset_factories_generate_training_data():
    for spec in (manage_icml2025(), bermuda_icml2025()):
        data = generate_observational_data(spec, 8, seed=0)

        assert set(spec.task.variable_order) == set(data)
        assert all(values.shape == (8,) for values in data.values())
        assert spec.task.candidate_alteration_sets


def test_reproduction_runner_smoke_for_manage():
    results = run_reproduction("manage", runs=1, val_samples=20, rounds=3, seed=10, max_iters=5)
    summary = results["manage"]["summary"]

    assert summary["ours_care_success_percent"]["mean"] >= 0.0
    assert summary["ours_care_success_percent"]["mean"] <= 100.0
    assert summary["ours_100_round_success"]["mean"] >= 0.0


def test_bermuda_paper_covariance_reaches_claim_scale():
    results = run_reproduction("bermuda", runs=2, val_samples=200, rounds=5, seed=20, max_iters=5)
    summary = results["bermuda"]["summary"]

    assert summary["ours_care_success_percent"]["mean"] > 70.0
    assert summary["paper_claim"]["ours_care_success_percent"] == 82.76
