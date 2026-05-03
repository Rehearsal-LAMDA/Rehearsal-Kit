import json

import pytest

from rehearsal.experiments.run import main, run_experiment_configs


def test_seeded_runner_always_returns_batch_payload_for_one_seed():
    result = run_experiment_configs(
        "examples/care/care_toy_experiment.py",
        seeds=(3,),
        method_name="care",
        method_params={"max_iters": 5},
        eval_samples=8,
        params={"n_samples": 40},
    )

    assert result["name"] == "care_toy"
    assert result["method"] == "care"
    assert result["seeds"] == [3]
    assert result["n_runs"] == 1
    assert len(result["runs"]) == 1
    run = result["runs"][0]
    assert run["seed"] == 3
    assert "z" in run["decision"]["alterations"]
    assert 0.0 <= run["decision"]["estimated_success_probability"] <= 1.0
    assert "estimated_care_success" not in run["decision"]
    assert run["structural_learning"]["runtime_seconds"] >= 0.0
    assert run["evaluation"]["eval_samples"] == 8
    assert 0.0 <= run["evaluation"]["true_auf_success_rate"] <= 1.0
    assert result["summary"]["evaluation.true_auf_success_rate"]["std"] == 0.0
    assert result["summary"]["decision.runtime_seconds"]["mean"] >= 0.0
    assert result["summary"]["structural_learning.runtime_seconds"]["mean"] >= 0.0
    assert "evaluation.eval_samples" not in result["summary"]
    assert "decision.cost" not in result["summary"]
    assert "decision.estimated_care_success" not in result["summary"]


def test_rehearsal_run_cli_writes_batch_json(tmp_path, capsys):
    output = tmp_path / "result.json"

    exit_code = main(
        [
            "examples/care/care_toy_experiment.py",
            "--method",
            "care",
            "--method-params",
            "max_iters=5",
            "--params",
            "n_samples=40",
            "--seeds",
            "5",
            "--eval-samples",
            "6",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["method"] == "care"
    assert payload["seeds"] == [5]
    assert payload["n_runs"] == 1
    assert payload["runs"][0]["evaluation"]["eval_samples"] == 6
    assert payload["summary"]["evaluation.true_auf_success_rate"]["std"] == 0.0
    assert payload["summary"]["decision.runtime_seconds"]["mean"] >= 0.0
    assert payload["summary"]["structural_learning.runtime_seconds"]["mean"] >= 0.0
    assert "evaluation.eval_samples" not in payload["summary"]
    assert "decision.cost" not in payload["summary"]
    assert "decision.estimated_care_success" not in payload["summary"]
    captured = capsys.readouterr()
    assert f"wrote {output}" in captured.err
    assert "n_runs=1" in captured.err


def test_rehearsal_run_cli_rejects_singular_param_aliases():
    base_args = [
        "examples/care/care_toy_experiment.py",
        "--method",
        "care",
    ]

    for option in ("--param", "--method-param", "--fit-param"):
        with pytest.raises(SystemExit) as excinfo:
            main([*base_args, "--seeds", "1", option, "x=1"])
        assert excinfo.value.code == 2


def test_runner_requires_seed_list_and_rejects_seed_as_param():
    with pytest.raises(ValueError, match="seed list"):
        run_experiment_configs(
            "examples/care/care_toy_experiment.py",
            seeds=None,
            method_name="care",
            params={"n_samples": 40},
        )

    with pytest.raises(ValueError, match="runner seed list"):
        run_experiment_configs(
            "examples/care/care_toy_experiment.py",
            seeds=(1,),
            method_name="care",
            params={"seed": 1, "n_samples": 40},
        )

    with pytest.raises(ValueError, match="runner seed list"):
        run_experiment_configs(
            "examples/care/care_toy_experiment.py",
            seeds=(1,),
            method_name="care",
            method_params={"seed": 1},
            params={"n_samples": 40},
        )


def test_multi_seed_runner_summarizes_numeric_outputs():
    result = run_experiment_configs(
        "examples/care/care_toy_experiment.py",
        seeds=(1, 2, 3),
        method_name="care",
        method_params={"max_iters": 5},
        eval_samples=6,
        params={"n_samples": 40},
    )

    assert result["seeds"] == [1, 2, 3]
    assert result["n_runs"] == 3
    assert len(result["runs"]) == 3
    assert "evaluation.true_auf_success_rate" in result["summary"]
    assert "evaluation.no_action_true_auf_success_rate" in result["summary"]
    assert "decision.runtime_seconds" in result["summary"]
    assert "structural_learning.runtime_seconds" in result["summary"]
    assert set(result["summary"]) == {
        "decision.runtime_seconds",
        "evaluation.no_action_true_auf_success_rate",
        "evaluation.true_auf_success_rate",
        "structural_learning.runtime_seconds",
    }


def test_demo_config_generates_observation_per_seed():
    result = run_experiment_configs(
        "examples/care/care_toy_experiment.py",
        seeds=(1, 2),
        method_name="care",
        method_params={"max_iters": 5},
        eval_samples=6,
        params={"n_samples": 40},
    )

    observations = [run["observation"]["x"] for run in result["runs"]]
    assert observations[0] != observations[1]
    assert {run["metadata"]["n_samples"] for run in result["runs"]} == {40}


def test_bermuda_care_example_runs():
    result = run_experiment_configs(
        "examples/care/care_bermuda_example.py",
        seeds=(4,),
        method_name="care",
        method_params={"max_iters": 5},
        eval_samples=5,
        params={"n_data": 20},
    )

    assert result["name"] == "care_bermuda"
    assert result["method"] == "care"
    assert result["n_runs"] == 1
    assert result["runs"][0]["evaluation"]["eval_samples"] == 5
    assert 0.0 <= result["runs"][0]["evaluation"]["true_auf_success_rate"] <= 1.0


def test_manage_care_example_runs():
    result = run_experiment_configs(
        "examples/care/care_manage_example.py",
        seeds=(6,),
        method_name="care",
        method_params={"max_iters": 5},
        eval_samples=5,
        params={"n_data": 20},
    )

    assert result["name"] == "care_manage"
    assert result["method"] == "care"
    assert result["n_runs"] == 1
    assert result["runs"][0]["evaluation"]["eval_samples"] == 5
    assert 0.0 <= result["runs"][0]["evaluation"]["true_auf_success_rate"] <= 1.0
