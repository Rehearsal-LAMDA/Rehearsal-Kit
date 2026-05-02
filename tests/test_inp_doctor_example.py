"""Smoke + qualitative tests for the Doctor INP example.

The Doctor SEM is the canonical illustration that *non-ancestral* variables
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_example_module():
    repo_root = Path(__file__).resolve().parents[1]
    example_path = repo_root / "examples" / "inp" / "doctor_inp_example.py"
    spec = importlib.util.spec_from_file_location("doctor_inp_example", example_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_doctor_inp_example_runs_end_to_end(tmp_path):
    module = _load_example_module()
    output = tmp_path / "inp_doctor_measures_smoke.json"
    exit_code = module.main(
        [
            "--num-samples",
            "800",
            "--max-extensions",
            "6",
            "--output",
            str(output),
            "--quiet",
        ]
    )
    assert exit_code == 0
    assert output.is_file()

    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["config"]["n_bins"] == 2
    assert payload["config"]["start_node"] == "W"
    assert "DoctorOracleSEM" in payload["config"]["model"]
    assert payload["task"]["alterable_variables"] == ["W", "X", "Z1"]
    assert payload["task"]["outcome_variables"] == ["Y"]
    assert payload["task"]["total_order"] == ["U", "W", "X", "Z1", "Y"]

    demos = payload["demos"]
    assert set(demos) == {
        "A_total_order_inp",
        "B_partial_order_inp",
        "C_conditional_inp",
        "D_ace_and_cace",
    }

    inp_a = demos["A_total_order_inp"]["inp"]
    assert set(inp_a) == {"W", "X", "Z1"}
    for record in inp_a.values():
        assert -1.0 <= record["inp"] <= 1.0
        assert 0.0 <= record["mep_do"] <= 1.0 + 1e-9
        assert 0.0 <= record["mep_ob"] <= 1.0 + 1e-9

    selection = demos["B_partial_order_inp"]["selection"]
    assert selection["best_order"][-1] == "Y"
    assert selection["best_start_node"] == "W"
    assert 0.0 <= selection["best_mep"] <= 1.0 + 1e-9
    assert demos["B_partial_order_inp"]["ordered_variables"] == ["W", "X", "Z1", "Y"]

    cond = demos["C_conditional_inp"]["results"]
    assert {"unconditional", "do(U=0)", "ob(U=0)", "do(U=1)", "ob(U=1)"} <= set(cond)

    ace = demos["D_ace_and_cace"]["ace"]
    cace = demos["D_ace_and_cace"]["cace"]
    assert set(ace) == {"W", "X", "Z1"}
    assert set(cace) == {"W", "X", "Z1"}
    for record in list(ace.values()) + list(cace.values()):
        assert 0.0 <= record["ace"] <= 1.0 + 1e-9


def test_doctor_inp_demonstrates_non_ancestral_influence_power(tmp_path):

    module = _load_example_module()
    output = tmp_path / "inp_doctor_measures_qualitative.json"
    exit_code = module.main(
        [
            "--num-samples",
            "4000",
            "--max-extensions",
            "6",
            "--output",
            str(output),
            "--quiet",
        ]
    )
    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))

    inp_w = payload["demos"]["A_total_order_inp"]["inp"]["W"]
    ace_w = payload["demos"]["D_ace_and_cace"]["ace"]["W"]

    assert abs(inp_w["inp"] - 0.16) <= 0.1, (
        f"Expected INP(W,Y) around 0.16; got {inp_w['inp']:.4f}."
    )
    assert ace_w["ace"] <= 0.05, (
        f"Expected ACE(W,Y) ~ 0 (W is non-ancestral); got {ace_w['ace']:.4f}."
    )
    assert inp_w["inp"] > ace_w["ace"] + 0.05, (
        "INP(W,Y) should clearly exceed ACE(W,Y) on the Doctor task; "
        f"got INP={inp_w['inp']:.4f}, ACE={ace_w['ace']:.4f}."
    )

    cond = payload["demos"]["C_conditional_inp"]["results"]
    for label in ("do(U=0)", "ob(U=0)", "do(U=1)", "ob(U=1)"):
        assert abs(cond[label]["inp"]) <= 0.04, (
            f"Expected INP(W,Y|{label}) ~ 0, got {cond[label]['inp']:.4f}."
        )

    # Partial-order MEP search must prefer [W, X, Z1, Y] over [W, Z1, X, Y]
    # so observing X can inform the choice of Z1 (the very mechanism by
    # which W gains its non-trivial influence power).
    selection = payload["demos"]["B_partial_order_inp"]["selection"]
    assert selection["best_order"] == ["W", "X", "Z1", "Y"], (
        f"Expected best partial-order extension to be [W, X, Z1, Y]; "
        f"got {selection['best_order']}."
    )
