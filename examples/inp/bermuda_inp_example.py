#!/usr/bin/env python
"""Bermuda INP / ACE measure example.

Demonstrates the INP and ACE *measures* on the Bermuda dataset using a
rehearsal model fitted by ``OrderBasedStructuralLearner``. Run with:

    PYTHONPATH=src python examples/inp/bermuda_inp_example.py \
        --n-data 2000 --num-samples 1500 --n-bins 3 \
        --start-node TA \
        --output outputs/inp_bermuda_measures.json

The example showcases:

1. INP using the order discovered by the structural learner
   (the "learned order").
2. Partial-order INP: enumerate compatible linear extensions, pick the one
   with the largest MEP at the chosen ``start_node``, then compute INP
   under that order.
3. Conditional INP: pre-fix some upstream ``do`` and ``observe`` values.
4. ACE and CACE for the same alterable variables.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rehearsal.datasets import bermuda, generate_observational_data
from rehearsal.measures import (
    ACEResult,
    INPResult,
    MEPCalculator,
    OrderSelectionResult,
    UniformBinDiscretizer,
    compute_ace,
    compute_cace,
    compute_inp,
    compute_inp_for_variables,
    desired_bins_from_region,
    enumerate_linear_extensions,
    select_best_total_order_by_mep,
)
from rehearsal.models import OrderBasedStructuralLearner


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-data", type=int, default=2000, help="Observational sample size for fitting the rehearsal model.")
    parser.add_argument("--num-samples", type=int, default=1500, help="Monte Carlo samples per (do, ob) context inside the MEP recursion.")
    parser.add_argument("--n-bins", type=int, default=3, help="Bin count per alterable variable.")
    parser.add_argument(
        "--start-node",
        type=str,
        default="TA",
        help=(
            "Variable from which to start the INP recursion. Must precede the "
            "target in the learned order; with many downstream alterables, INP "
            "for upstream nodes is often 0 because downstream choices can compensate."
        ),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-extensions", type=int, default=8, help="Cap on enumerated linear extensions in Demo B.")
    parser.add_argument("--target-n-bins", type=int, default=None, help="Optional finer bin count for the target NEC (defaults to --n-bins).")
    parser.add_argument(
        "--alterable-bin-range",
        type=str,
        default="task",
        help=(
            "How to set the bin range for alterable variables: 'task' uses the task's "
            "alteration_domain bounds (default), or pass a comma-separated low,high pair "
            "to override (e.g. '-2,2')."
        ),
    )
    parser.add_argument("--output", type=str, default="outputs/inp_bermuda_measures.json")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def _resolve_bin_ranges(args: argparse.Namespace, task) -> dict[str, tuple[float, float]]:
    """Pick a bin range for every variable.

    Alterable variables default to the task's ``alteration_domain`` so the
    bin centers represent realistic alterations. Non-alterable variables
    (observed and intermediate) keep the wider standardized-data range.
    """

    if args.alterable_bin_range == "task":
        ranges: dict[str, tuple[float, float]] = {}
        for name in task.all_variables():
            if name in task.alterable:
                ranges[name] = task.alteration_domain.bounds[name]
            else:
                ranges[name] = (-3.0, 3.0)
        return ranges
    parts = args.alterable_bin_range.split(",")
    if len(parts) != 2:
        raise ValueError("--alterable-bin-range must be 'task' or a 'low,high' pair.")
    low, high = float(parts[0]), float(parts[1])
    ranges = {}
    for name in task.all_variables():
        ranges[name] = (low, high) if name in task.alterable else (-3.0, 3.0)
    return ranges


def build_experiment(args: argparse.Namespace) -> dict[str, Any]:
    spec = bermuda()
    task = spec.task
    target = task.outcomes[0]
    n_bins_map = {name: int(args.n_bins) for name in task.all_variables()}
    if args.target_n_bins is not None:
        n_bins_map[target] = int(args.target_n_bins)
    bin_ranges = _resolve_bin_ranges(args, task)
    discretizer = UniformBinDiscretizer(task.all_variables(), n_bins=n_bins_map, bin_range=bin_ranges)
    desired_bins = desired_bins_from_region(target, task.desired_region, discretizer, outcome_variables=task.outcomes)

    data = generate_observational_data(spec, int(args.n_data), seed=int(args.seed))
    learner = OrderBasedStructuralLearner(max_parents=4)
    fit = learner.fit(data, task)
    model = fit.model

    learned_order = tuple(fit.diagnostics.get("order", task.variable_order))

    return {
        "spec": spec,
        "task": task,
        "target": target,
        "discretizer": discretizer,
        "desired_bins": desired_bins,
        "model": model,
        "data": data,
        "learned_order": learned_order,
    }


def demo_total_order_inp(
    experiment: Mapping[str, Any],
    *,
    ordering: Sequence[str],
    start_node: str,
    num_samples: int,
    seed: int,
    label: str,
) -> dict[str, Any]:
    task = experiment["task"]
    target = experiment["target"]
    discretizer = experiment["discretizer"]
    desired_bins = experiment["desired_bins"]
    model = experiment["model"]

    candidates = [name for name in task.alterable if _between(name, ordering, start_node, target)]
    if start_node in candidates:
        # start_node itself is what compute_inp is evaluating; keep it.
        pass
    inp_results = compute_inp_for_variables(
        model,
        task,
        candidates,
        ordering=ordering,
        target=target,
        desired_bins=desired_bins,
        discretizer=discretizer,
        num_samples=num_samples,
        rng=np.random.default_rng(int(seed) + 1000),
    )
    return {
        "label": label,
        "ordering": list(ordering),
        "start_node": start_node,
        "evaluated_variables": list(candidates),
        "inp": {name: result.to_dict() for name, result in inp_results.items()},
    }


def demo_partial_order_inp(
    experiment: Mapping[str, Any],
    *,
    predecessor_map: Mapping[str, Sequence[str]],
    start_node: str,
    num_samples: int,
    seed: int,
    max_extensions: int | None,
) -> dict[str, Any]:
    task = experiment["task"]
    target = experiment["target"]
    discretizer = experiment["discretizer"]
    desired_bins = experiment["desired_bins"]
    model = experiment["model"]

    selection = select_best_total_order_by_mep(
        model,
        task,
        target=target,
        desired_bins=desired_bins,
        discretizer=discretizer,
        predecessor_map=predecessor_map,
        start_node=start_node,
        num_samples=num_samples,
        rng=np.random.default_rng(int(seed) + 2000),
        max_extensions=max_extensions,
    )
    best_inp = compute_inp_for_variables(
        model,
        task,
        [name for name in task.alterable if _between(name, selection.best_order, start_node, target)],
        ordering=selection.best_order,
        target=target,
        desired_bins=desired_bins,
        discretizer=discretizer,
        num_samples=num_samples,
        rng=np.random.default_rng(int(seed) + 2500),
    )
    return {
        "label": "Demo B: partial-order INP",
        "predecessor_map": {k: list(v) for k, v in predecessor_map.items()},
        "selection": selection.to_dict(),
        "inp_under_best_order": {name: result.to_dict() for name, result in best_inp.items()},
    }


def demo_conditional_inp(
    experiment: Mapping[str, Any],
    *,
    ordering: Sequence[str],
    variable: str,
    do_conditions: Mapping[str, int],
    ob_conditions: Mapping[str, int],
    num_samples: int,
    seed: int,
) -> dict[str, Any]:
    task = experiment["task"]
    target = experiment["target"]
    discretizer = experiment["discretizer"]
    desired_bins = experiment["desired_bins"]
    model = experiment["model"]

    rng = np.random.default_rng(int(seed) + 3000)
    unconditional = compute_inp(
        model, task, variable, ordering=ordering, target=target,
        desired_bins=desired_bins, discretizer=discretizer,
        num_samples=num_samples, rng=rng,
    )
    do_only = compute_inp(
        model, task, variable, ordering=ordering, target=target,
        desired_bins=desired_bins, discretizer=discretizer,
        do_conditions=do_conditions, num_samples=num_samples,
        rng=np.random.default_rng(int(seed) + 3100),
    )
    ob_only = compute_inp(
        model, task, variable, ordering=ordering, target=target,
        desired_bins=desired_bins, discretizer=discretizer,
        ob_conditions=ob_conditions, num_samples=num_samples,
        rng=np.random.default_rng(int(seed) + 3200),
    )
    combined = compute_inp(
        model, task, variable, ordering=ordering, target=target,
        desired_bins=desired_bins, discretizer=discretizer,
        do_conditions=do_conditions, ob_conditions=ob_conditions,
        num_samples=num_samples, rng=np.random.default_rng(int(seed) + 3300),
    )
    return {
        "label": "Demo C: conditional INP",
        "variable": variable,
        "ordering": list(ordering),
        "do_conditions": dict(do_conditions),
        "ob_conditions": dict(ob_conditions),
        "results": {
            "unconditional": unconditional.to_dict(),
            "with_do_only": do_only.to_dict(),
            "with_ob_only": ob_only.to_dict(),
            "with_do_and_ob": combined.to_dict(),
        },
    }


def demo_ace(
    experiment: Mapping[str, Any],
    *,
    ordering: Sequence[str],
    do_conditions: Mapping[str, int],
    ob_conditions: Mapping[str, int],
    num_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Compute ACE / CACE for every alterable variable.

    Unlike INP, ACE and CACE are well-defined for any alterable variable
    independently of the start_node, so we evaluate the whole alterable set
    rather than restricting to the variables downstream of start_node.
    """

    task = experiment["task"]
    target = experiment["target"]
    discretizer = experiment["discretizer"]
    desired_bins = experiment["desired_bins"]
    model = experiment["model"]

    # Share the calculator (and therefore its caches) across both ACE and CACE
    # passes; they only need ``gauge_auf_prob`` (no recursion), so this is cheap.
    calc = MEPCalculator(
        model,
        ordering=ordering,
        target_node=target,
        desired_bins=desired_bins,
        discretizer=discretizer,
        alterable=task.alterable,
        num_samples=num_samples,
        rng=np.random.default_rng(int(seed) + 4001),
    )

    candidates = list(task.alterable)
    ace_results: dict[str, dict[str, Any]] = {}
    cace_results: dict[str, dict[str, Any]] = {}

    for name in candidates:
        ace_results[name] = compute_ace(
            model, task, name,
            ordering=ordering, target=target, desired_bins=desired_bins, discretizer=discretizer,
            num_samples=num_samples, calculator=calc,
        ).to_dict()
        cace_results[name] = compute_cace(
            model, task, name,
            ordering=ordering, target=target, desired_bins=desired_bins, discretizer=discretizer,
            do_conditions={k: v for k, v in do_conditions.items() if k != name},
            ob_conditions={k: v for k, v in ob_conditions.items() if k != name},
            num_samples=num_samples, calculator=calc,
        ).to_dict()

    return {
        "label": "Demo D: ACE and CACE",
        "ordering": list(ordering),
        "evaluated_variables": list(candidates),
        "do_conditions_for_cace": dict(do_conditions),
        "ob_conditions_for_cace": dict(ob_conditions),
        "ace": ace_results,
        "cace": cace_results,
    }


def _between(name: str, ordering: Sequence[str], start_node: str, target: str) -> bool:
    if name not in ordering or start_node not in ordering or target not in ordering:
        return False
    start_idx = ordering.index(start_node)
    target_idx = ordering.index(target)
    name_idx = ordering.index(name)
    return start_idx <= name_idx < target_idx


def bermuda_partial_order(task) -> dict[str, tuple[str, ...]]:
    """Return a partial order capturing Bermuda's stage structure.

    The default follows a three-stage grouping (driver vars, then
    chemistry, then NEC) but leaves ordering flexibility within each stage so
    that ``select_best_total_order_by_mep`` has a non-trivial decision to
    make. Predecessors are subsets of the task's structural parents — a
    superset of the structural DAG would still be valid but would constrain
    the order more than necessary.
    """

    return {
        "Light": tuple(),
        "Temp": ("Light",),
        "Sal": ("Temp",),
        "Nutrients_PC1": tuple(),
        "DIC": ("Sal",),
        "TA": ("Sal",),
        "Chla": ("Light", "Nutrients_PC1", "Temp"),
        "Omega": ("DIC", "TA", "Sal", "Temp"),
        "pHsw": ("DIC", "TA", "Sal", "Temp"),
        "CO2": ("DIC", "TA", "Sal", "Temp"),
        "NEC": ("Omega", "pHsw", "CO2", "Chla", "Nutrients_PC1"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()

    experiment = build_experiment(args)
    task = experiment["task"]
    target = experiment["target"]
    learned_order = experiment["learned_order"]
    discretizer: UniformBinDiscretizer = experiment["discretizer"]

    if args.start_node not in learned_order:
        raise ValueError(
            f"--start-node {args.start_node!r} is not in the learned order {list(learned_order)}; "
            "pick a variable that the structural learner placed in its ordering."
        )
    if learned_order.index(args.start_node) >= learned_order.index(target):
        raise ValueError(
            f"--start-node {args.start_node!r} appears at-or-after the target "
            f"{target!r} in the learned order {list(learned_order)}; INP/MEP can only "
            "be evaluated for nodes that precede the target."
        )

    if not args.quiet:
        print(f"Bermuda task: target={target}, alterable={task.alterable}")
        print(f"Learned order:  {learned_order}")
        print(f"Desired bins for {target}: {experiment['desired_bins']}")

    demo_a = demo_total_order_inp(
        experiment,
        ordering=learned_order,
        start_node=args.start_node,
        num_samples=args.num_samples,
        seed=args.seed,
        label="Demo A: total-order INP (learned ordering)",
    )

    partial = bermuda_partial_order(task)
    demo_b = demo_partial_order_inp(
        experiment,
        predecessor_map=partial,
        start_node=args.start_node,
        num_samples=args.num_samples,
        seed=args.seed,
        max_extensions=args.max_extensions,
    )

    do_for_conditional: dict[str, int] = {}
    ob_for_conditional: dict[str, int] = {}
    start_idx = learned_order.index(args.start_node)
    # Pre-condition the conditional INP demo on a hypothetical environmental
    # observation (Temp) and a hypothetical upstream chemistry alteration
    # (pHsw). Both must precede ``start_node`` in the learned order so they
    # affect the recursion before it reaches the variable being measured.
    if (
        "Temp" in learned_order
        and learned_order.index("Temp") < start_idx
        and args.start_node != "Temp"
    ):
        ob_for_conditional["Temp"] = discretizer.discretize("Temp", 0.5)  # type: ignore[arg-type]
    if (
        "pHsw" in learned_order
        and learned_order.index("pHsw") < start_idx
        and args.start_node != "pHsw"
    ):
        do_for_conditional["pHsw"] = discretizer.discretize("pHsw", -0.5)  # type: ignore[arg-type]
    # Pick a downstream alterable for the conditional INP demo if start_node lies upstream.
    candidates_after_start = [name for name in task.alterable if _between(name, learned_order, args.start_node, target)]
    conditional_variable = candidates_after_start[0] if candidates_after_start else args.start_node
    if conditional_variable in do_for_conditional or conditional_variable in ob_for_conditional:
        # Avoid clashing with the variable being measured.
        conditional_variable = candidates_after_start[-1] if candidates_after_start else args.start_node

    demo_c = demo_conditional_inp(
        experiment,
        ordering=learned_order,
        variable=conditional_variable,
        do_conditions=do_for_conditional,
        ob_conditions=ob_for_conditional,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    demo_d = demo_ace(
        experiment,
        ordering=learned_order,
        do_conditions=do_for_conditional,
        ob_conditions=ob_for_conditional,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    elapsed = time.perf_counter() - started

    payload = {
        "config": {
            "n_data": int(args.n_data),
            "num_samples": int(args.num_samples),
            "n_bins": int(args.n_bins),
            "target_n_bins": int(args.target_n_bins) if args.target_n_bins is not None else int(args.n_bins),
            "start_node": args.start_node,
            "seed": int(args.seed),
            "max_extensions": int(args.max_extensions),
        },
        "task": {
            "observed_variables": list(task.observed),
            "alterable_variables": list(task.alterable),
            "outcome_variables": list(task.outcomes),
            "learned_order": list(learned_order),
            "desired_bins": list(experiment["desired_bins"]),
            "desired_region_metadata": _public_metadata(task.metadata),
        },
        "demos": {
            "A_total_order_inp": demo_a,
            "B_partial_order_inp": demo_b,
            "C_conditional_inp": demo_c,
            "D_ace_and_cace": demo_d,
        },
        "runtime_seconds": float(elapsed),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")

    if not args.quiet:
        print(f"\nWrote results to {output_path} (runtime {elapsed:.2f}s)")
        _summarise_inp(demo_a)
        _summarise_inp(demo_b, key="inp_under_best_order")
        _summarise_conditional(demo_c)
        _summarise_ace(demo_d)
    return 0


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _public_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return metadata suitable for example outputs."""

    return {str(key): value for key, value in metadata.items() if key != "raw_data_path"}


def _summarise_inp(demo: Mapping[str, Any], key: str = "inp") -> None:
    print(f"\n[{demo['label']}]")
    if "ordering" in demo:
        print(f"  ordering: {demo['ordering']}")
    items = demo[key]
    for name, record in items.items():
        print(f"  INP({name}) = {record['inp']:+.4f}  (mep_do={record['mep_do']:.4f}, mep_ob={record['mep_ob']:.4f}, best_value={record['best_do_value']:+.3f})")


def _summarise_conditional(demo: Mapping[str, Any]) -> None:
    print(f"\n[{demo['label']}] variable={demo['variable']}")
    for label, record in demo["results"].items():
        print(f"  {label}: INP={record['inp']:+.4f}  (mep_do={record['mep_do']:.4f}, mep_ob={record['mep_ob']:.4f})")


def _summarise_ace(demo: Mapping[str, Any]) -> None:
    print(f"\n[{demo['label']}]")
    for name in demo["evaluated_variables"]:
        ace_val = demo["ace"][name]["ace"]
        cace_val = demo["cace"][name]["ace"]
        print(f"  {name}: ACE={ace_val:.4f}  CACE={cace_val:.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
