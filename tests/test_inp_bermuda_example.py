"""Smoke test: the Bermuda INP example runs end-to-end on a tiny budget."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_example_module():
    repo_root = Path(__file__).resolve().parents[1]
    example_path = repo_root / "examples" / "inp" / "bermuda_inp_example.py"
    spec = importlib.util.spec_from_file_location("bermuda_inp_example", example_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bermuda_inp_example_runs_end_to_end(tmp_path):
    module = _load_example_module()
    output = tmp_path / "inp_bermuda_measures_smoke.json"
    exit_code = module.main(
        [
            "--n-data",
            "60",
            "--num-samples",
            "60",
            "--n-bins",
            "2",
            "--start-node",
            "DIC",
            "--max-extensions",
            "3",
            "--output",
            str(output),
            "--quiet",
        ]
    )
    assert exit_code == 0
    assert output.is_file()

    payload = json.loads(output.read_text(encoding="utf-8"))

    # Config snapshot is correctly recorded.
    assert payload["config"]["n_bins"] == 2
    assert payload["config"]["start_node"] == "DIC"

    # Each demo block is present.
    demos = payload["demos"]
    assert set(demos) == {
        "A_total_order_inp",
        "B_partial_order_inp",
        "C_conditional_inp",
        "D_ace_and_cace",
    }

    # Demo A produces at least one INP record (DIC onwards) with a valid probability.
    inp_block = demos["A_total_order_inp"]["inp"]
    assert inp_block, "Demo A must produce at least one INP record"
    for record in inp_block.values():
        assert -1.0 <= record["inp"] <= 1.0
        assert 0.0 <= record["mep_do"] <= 1.0 + 1e-9
        assert 0.0 <= record["mep_ob"] <= 1.0 + 1e-9

    # Demo B's order selection produces a valid extension ending at the target.
    selection = demos["B_partial_order_inp"]["selection"]
    assert selection["best_order"][-1] == "NEC"
    assert 0.0 <= selection["best_mep"] <= 1.0 + 1e-9

    # Demo D evaluates ACE and CACE for every alterable variable in the task.
    ace = demos["D_ace_and_cace"]["ace"]
    cace = demos["D_ace_and_cace"]["cace"]
    assert set(ace) == set(cace)
    assert set(ace).issuperset({"DIC", "TA", "Omega", "Chla", "Nutrients_PC1"})
    for record in list(ace.values()) + list(cace.values()):
        assert 0.0 <= record["ace"] <= 1.0 + 1e-9
