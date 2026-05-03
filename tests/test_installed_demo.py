import json

from rehearsal.experiments.demo import main, run_demo


def test_installed_demo_runs_as_imported_api():
    result = run_demo(seed=3, n_samples=40, eval_samples=6, max_iters=5)

    assert result["name"] == "care_toy_installed_demo"
    assert result["method"] == "care"
    assert result["seeds"] == [3]
    assert result["n_runs"] == 1
    assert result["runs"][0]["evaluation"]["eval_samples"] == 6
    assert "evaluation.true_auf_success_rate" in result["summary"]
    assert "decision.runtime_seconds" in result["summary"]


def test_installed_demo_cli_writes_json(tmp_path, capsys):
    output = tmp_path / "demo.json"

    exit_code = main(
        [
            "--seed",
            "4",
            "--n-samples",
            "40",
            "--eval-samples",
            "6",
            "--max-iters",
            "5",
            "--output",
            str(output),
            "--compact",
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["name"] == "care_toy_installed_demo"
    assert payload["method"] == "care"
    assert payload["seeds"] == [4]
    assert payload["runs"][0]["evaluation"]["eval_samples"] == 6
    captured = capsys.readouterr()
    assert f"wrote {output}" in captured.err
