#!/usr/bin/env python
"""Doctor INP / ACE measure example (non-ancestral influence).

The Doctor task illustrates that *non-ancestral* variables can have non-zero
influence power. It uses the following binary SEM:

    U  ~ Bern(0.5)
    W  ~ Bern(0.1)
    N_X ~ Bern(0.1)
    Z1 ~ Bern(0.1)
    N_Y ~ Bern(0.4)
    X  := min(U, W) * (1 - N_X)
    Y  := Z1 * (1 - U) + (1 - Z1) * N_Y

with desired outcome ``Y = 1`` and actionable variables ``{W, X, Z1}``.

Using the oracle SCM lets us isolate the INP measure from modelling bias.


The example showcases:

1. Demo A: INP under the order ``(U, W, X, Z1, Y)``.
2. Demo B: INP based on partial-order with enumerating compatible linear extensions,
   pick the one with the largest MEP at ``W``, then compute INP under it.
   The pedagogically expected winner is ``(W, X, Z1, Y)`` because observing
   X reveals information about U and lets the doctor pick Z1 optimally.
3. Demo C: conditional INP given upstream observations / alterations on
   U; the unconditional ``INP(W, Y) > 0`` collapses to ~0 once U is fixed.
4. Demo D: ACE / CACE for every actionable variable. ``ACE(W, Y) ~ 0``
   because W is not an ancestor of Y, even though ``INP(W, Y)`` is sizeable.

Run::

    PYTHONPATH=src python examples/inp/doctor_inp_example.py \
        --num-samples 4000 \
        --output outputs/inp_doctor_measures.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rehearsal.core import AUFTask, AlterationDomain, DesiredRegion
from rehearsal.measures import (
    MEPCalculator,
    UniformBinDiscretizer,
    compute_ace,
    compute_cace,
    compute_inp,
    compute_inp_for_variables,
    desired_bins_from_region,
    select_best_total_order_by_mep,
)


# ---------------------------------------------------------------------------
# Doctor SEM specification.
# ---------------------------------------------------------------------------

DOCTOR_VARIABLES: tuple[str, ...] = ("U", "W", "X", "Z1", "Y")
DOCTOR_EXPERT_ORDER: tuple[str, ...] = ("U", "W", "X", "Z1", "Y")
DOCTOR_PARENTS: dict[str, tuple[str, ...]] = {
    "U": (),
    "W": (),
    "X": ("U", "W"),
    "Z1": (),
    "Y": ("U", "Z1"),
}
DOCTOR_PROBS: dict[str, float] = {"U": 0.5, "W": 0.1, "X": 0.1, "Z1": 0.1, "Y": 0.4}
DOCTOR_TARGET: str = "Y"
DOCTOR_START_NODE: str = "W"
DOCTOR_ALTERABLE: tuple[str, ...] = ("W", "X", "Z1")


# ---------------------------------------------------------------------------
# Ground-truth Doctor SEM (duck-typed for MEPCalculator)
# ---------------------------------------------------------------------------


class DoctorOracleSEM:
    """Ground-truth Doctor structural model.
    """

    def __init__(self, probabilities: Mapping[str, float] | None = None) -> None:
        self.variable_order: tuple[str, ...] = DOCTOR_EXPERT_ORDER
        self.parents: dict[str, tuple[str, ...]] = {
            name: tuple(parents) for name, parents in DOCTOR_PARENTS.items()
        }
        if probabilities is None:
            self.probabilities: dict[str, float] = dict(DOCTOR_PROBS)
        else:
            missing = [name for name in DOCTOR_VARIABLES if name not in probabilities]
            if missing:
                raise ValueError(
                    f"DoctorOracleSEM: probabilities mapping is missing {missing!r}."
                )
            self.probabilities = {name: float(probabilities[name]) for name in DOCTOR_VARIABLES}

    def simulate(
        self,
        n_samples: int,
        *,
        rng: np.random.Generator,
        observation: Mapping[str, float] | None = None,
        alterations: Mapping[str, float] | None = None,
        noise_bank: Mapping[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        n = int(n_samples)
        if n <= 0:
            raise ValueError("DoctorOracleSEM.simulate: n_samples must be positive.")
        observed = dict(observation or {})
        altered = dict(alterations or {})
        overlap = set(observed) & set(altered)
        if overlap:
            raise ValueError(
                f"DoctorOracleSEM.simulate: variables overlap in observation and "
                f"alterations: {sorted(overlap)}."
            )
        bank = dict(noise_bank or {})
        p = self.probabilities

        def _noise(name: str, prob: float) -> np.ndarray:
            if name in bank:
                return np.asarray(bank[name], dtype=float).reshape(-1)
            return rng.binomial(1, prob, size=n).astype(float)

        N_U = _noise("U", p["U"])
        N_W = _noise("W", p["W"])
        N_X = _noise("X", p["X"])
        N_Z1 = _noise("Z1", p["Z1"])
        N_Y = _noise("Y", p["Y"])

        def _resolve(name: str, computed: np.ndarray) -> np.ndarray:
            if name in observed:
                return np.full(n, float(observed[name]))
            if name in altered:
                return np.full(n, float(altered[name]))
            return computed

        U = _resolve("U", N_U)
        W = _resolve("W", N_W)
        X = _resolve("X", np.minimum(U, W) * (1.0 - N_X))
        Z1 = _resolve("Z1", N_Z1)
        Y = _resolve("Y", Z1 * (1.0 - U) + (1.0 - Z1) * N_Y)

        return {"U": U, "W": W, "X": X, "Z1": Z1, "Y": Y}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4000,
        help=(
            "Monte Carlo samples per (do, ob) context inside the MEP recursion. "
            "Bumping this past 4000 brings the empirical INP within ~0.005 of "
            "the analytic 0.162; it costs little because no fitting happens."
        ),
    )
    parser.add_argument(
        "--start-node",
        type=str,
        default=DOCTOR_START_NODE,
        help="Variable that begins the INP recursion (defaults to W).",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--max-extensions",
        type=int,
        default=24,
        help="Cap on enumerated linear extensions in Demo B.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/inp_doctor_measures.json",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Task / SEM construction
# ---------------------------------------------------------------------------


def build_doctor_task() -> AUFTask:
    """Return the Doctor AUFTask with binary alteration domains."""

    return AUFTask(
        observed_variables=(),
        alterable_variables=DOCTOR_ALTERABLE,
        outcome_variables=(DOCTOR_TARGET,),
        # Y in {0, 1}; the desired outcome is Y = 1.
        desired_region=DesiredRegion.from_intervals({"Y": (0.5, 1.5)}),
        alteration_domain=AlterationDomain(
            {name: (0.0, 1.0) for name in DOCTOR_ALTERABLE}
        ),
        parents={k: tuple(v) for k, v in DOCTOR_PARENTS.items()},
        candidate_alteration_sets=tuple((name,) for name in DOCTOR_ALTERABLE),
        variable_order=DOCTOR_EXPERT_ORDER,
        metadata={
            "dataset": "Doctor (ICLR 2026 INP example)",
            "probabilities": dict(DOCTOR_PROBS),
            "structural_equations": {
                "U": "Bern(0.5)",
                "W": "Bern(0.1)",
                "X": "min(U, W) * (1 - N_X), N_X ~ Bern(0.1)",
                "Z1": "Bern(0.1)",
                "Y": "Z1 * (1 - U) + (1 - Z1) * N_Y, N_Y ~ Bern(0.4)",
            },
        },
    )


def doctor_partial_order() -> dict[str, tuple[str, ...]]:
    """Partial order over the rehearsal-relevant variables ``(W, X, Z1, Y)``.

    U is the unobservable allergy gene and is *not* actionable, so we
    deliberately keep it out of the rehearsal sequence: any extension that
    placed U after W would let the MEP recursion ``do(U)`` and inflate the
    score with a physically meaningless alteration. The remaining
    constraints reflect the SEM (``X`` follows ``W``) plus the rehearsal
    convention that the target must be last (``Y`` follows ``X`` and ``Z1``).
    """

    return {
        "W": (),
        "X": ("W",),
        "Z1": (),
        "Y": ("X", "Z1"),
    }


def doctor_partial_order_variables() -> tuple[str, ...]:
    """Variable set passed to :func:`select_best_total_order_by_mep`."""

    return ("W", "X", "Z1", "Y")


def build_experiment(args: argparse.Namespace) -> dict[str, Any]:
    task = build_doctor_task()
    target = task.outcomes[0]

    discretizer = UniformBinDiscretizer(
        task.all_variables(),
        n_bins=2,
        bin_range=(-0.5, 1.5),
    )
    desired_bins = desired_bins_from_region(
        target, task.desired_region, discretizer, outcome_variables=task.outcomes
    )

    # No learner -- plug the ground-truth Doctor SCM straight into MEPCalculator.
    model = DoctorOracleSEM()
    expert_order = DOCTOR_EXPERT_ORDER

    return {
        "task": task,
        "target": target,
        "discretizer": discretizer,
        "desired_bins": desired_bins,
        "model": model,
        "expert_order": expert_order,
    }


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------


def _between(name: str, ordering: Sequence[str], start_node: str, target: str) -> bool:
    if name not in ordering or start_node not in ordering or target not in ordering:
        return False
    start_idx = ordering.index(start_node)
    target_idx = ordering.index(target)
    name_idx = ordering.index(name)
    return start_idx <= name_idx < target_idx


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

    candidates = [
        name for name in task.alterable if _between(name, ordering, start_node, target)
    ]
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
    variables: Sequence[str],
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
        variables=variables,
        start_node=start_node,
        num_samples=num_samples,
        rng=np.random.default_rng(int(seed) + 2000),
        max_extensions=max_extensions,
    )
    candidates = [
        name
        for name in task.alterable
        if _between(name, selection.best_order, start_node, target)
    ]
    best_inp = compute_inp_for_variables(
        model,
        task,
        candidates,
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
        "ordered_variables": list(variables),
        "selection": selection.to_dict(),
        "evaluated_variables": list(candidates),
        "inp_under_best_order": {
            name: result.to_dict() for name, result in best_inp.items()
        },
    }


def demo_conditional_inp(
    experiment: Mapping[str, Any],
    *,
    ordering: Sequence[str],
    variable: str,
    upstream_variable: str,
    upstream_bins: Sequence[int],
    num_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Compute INP(variable, target) under various conditions on ``upstream_variable``.

    For the Doctor task, ``INP(W, Y | U)`` collapses to 0 once U is fixed.
    """

    task = experiment["task"]
    target = experiment["target"]
    discretizer = experiment["discretizer"]
    desired_bins = experiment["desired_bins"]
    model = experiment["model"]

    rng = np.random.default_rng(int(seed) + 3000)
    unconditional = compute_inp(
        model,
        task,
        variable,
        ordering=ordering,
        target=target,
        desired_bins=desired_bins,
        discretizer=discretizer,
        num_samples=num_samples,
        rng=rng,
    )

    conditioned: dict[str, dict[str, Any]] = {}
    for offset, bin_idx in enumerate(upstream_bins, start=1):
        bin_idx = int(bin_idx)
        do_label = f"do({upstream_variable}={bin_idx})"
        ob_label = f"ob({upstream_variable}={bin_idx})"
        conditioned[do_label] = compute_inp(
            model,
            task,
            variable,
            ordering=ordering,
            target=target,
            desired_bins=desired_bins,
            discretizer=discretizer,
            do_conditions={upstream_variable: bin_idx},
            num_samples=num_samples,
            rng=np.random.default_rng(int(seed) + 3100 + offset),
        ).to_dict()
        conditioned[ob_label] = compute_inp(
            model,
            task,
            variable,
            ordering=ordering,
            target=target,
            desired_bins=desired_bins,
            discretizer=discretizer,
            ob_conditions={upstream_variable: bin_idx},
            num_samples=num_samples,
            rng=np.random.default_rng(int(seed) + 3200 + offset),
        ).to_dict()

    return {
        "label": "Demo C: conditional INP",
        "variable": variable,
        "ordering": list(ordering),
        "upstream_variable": upstream_variable,
        "upstream_bins": [int(b) for b in upstream_bins],
        "results": {
            "unconditional": unconditional.to_dict(),
            **conditioned,
        },
    }


def demo_ace(
    experiment: Mapping[str, Any],
    *,
    ordering: Sequence[str],
    upstream_variable: str,
    upstream_bin: int,
    num_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Compute ACE / CACE for every alterable variable."""

    task = experiment["task"]
    target = experiment["target"]
    discretizer = experiment["discretizer"]
    desired_bins = experiment["desired_bins"]
    model = experiment["model"]

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

    ob_for_cace = {upstream_variable: int(upstream_bin)}
    candidates = list(task.alterable)
    ace_results: dict[str, dict[str, Any]] = {}
    cace_results: dict[str, dict[str, Any]] = {}
    for name in candidates:
        ace_results[name] = compute_ace(
            model,
            task,
            name,
            ordering=ordering,
            target=target,
            desired_bins=desired_bins,
            discretizer=discretizer,
            num_samples=num_samples,
            calculator=calc,
        ).to_dict()
        cace_results[name] = compute_cace(
            model,
            task,
            name,
            ordering=ordering,
            target=target,
            desired_bins=desired_bins,
            discretizer=discretizer,
            ob_conditions={k: v for k, v in ob_for_cace.items() if k != name},
            num_samples=num_samples,
            calculator=calc,
        ).to_dict()

    return {
        "label": "Demo D: ACE and CACE",
        "ordering": list(ordering),
        "evaluated_variables": list(candidates),
        "ob_conditions_for_cace": dict(ob_for_cace),
        "ace": ace_results,
        "cace": cace_results,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()

    experiment = build_experiment(args)
    task = experiment["task"]
    target = experiment["target"]
    expert_order = experiment["expert_order"]

    if not args.quiet:
        print(f"Doctor task (oracle SEM): target={target}, alterable={task.alterable}")
        print(f"Total order:    {expert_order}")
        print(f"Desired bins for {target}: {experiment['desired_bins']}")

    demo_a = demo_total_order_inp(
        experiment,
        ordering=expert_order,
        start_node=args.start_node,
        num_samples=args.num_samples,
        seed=args.seed,
        label="Demo A: total-order INP (expert ordering)",
    )

    demo_b = demo_partial_order_inp(
        experiment,
        predecessor_map=doctor_partial_order(),
        variables=doctor_partial_order_variables(),
        start_node=args.start_node,
        num_samples=args.num_samples,
        seed=args.seed,
        max_extensions=args.max_extensions,
    )

    demo_c = demo_conditional_inp(
        experiment,
        ordering=expert_order,
        variable=args.start_node,
        upstream_variable="U",
        upstream_bins=(0, 1),
        num_samples=args.num_samples,
        seed=args.seed,
    )

    demo_d = demo_ace(
        experiment,
        ordering=expert_order,
        upstream_variable="U",
        upstream_bin=1,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    elapsed = time.perf_counter() - started

    payload = {
        "config": {
            "num_samples": int(args.num_samples),
            "n_bins": 2,
            "model": "DoctorOracleSEM (ground-truth structural equations, no learning)",
            "start_node": args.start_node,
            "seed": int(args.seed),
            "max_extensions": int(args.max_extensions),
        },
        "task": {
            "observed_variables": list(task.observed),
            "alterable_variables": list(task.alterable),
            "outcome_variables": list(task.outcomes),
            "total_order": list(expert_order),
            "desired_bins": list(experiment["desired_bins"]),
            "structural_equations": dict(task.metadata.get("structural_equations", {})),
            "probabilities": dict(task.metadata.get("probabilities", {})),
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
    output_path.write_text(
        json.dumps(payload, indent=2, default=_json_default), encoding="utf-8"
    )

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


def _summarise_inp(demo: Mapping[str, Any], key: str = "inp") -> None:
    print(f"\n[{demo['label']}]")
    if "ordering" in demo:
        print(f"  ordering: {demo['ordering']}")
    items = demo[key]
    if not items:
        print("  (no eligible variables between start_node and target)")
        return
    for name, record in items.items():
        print(
            f"  INP({name}) = {record['inp']:+.4f}  "
            f"(mep_do={record['mep_do']:.4f}, mep_ob={record['mep_ob']:.4f}, "
            f"best_value={record['best_do_value']:+.3f})"
        )


def _summarise_conditional(demo: Mapping[str, Any]) -> None:
    print(f"\n[{demo['label']}] variable={demo['variable']}")
    for label, record in demo["results"].items():
        print(
            f"  {label}: INP={record['inp']:+.4f}  "
            f"(mep_do={record['mep_do']:.4f}, mep_ob={record['mep_ob']:.4f})"
        )


def _summarise_ace(demo: Mapping[str, Any]) -> None:
    print(f"\n[{demo['label']}]")
    for name in demo["evaluated_variables"]:
        ace_val = demo["ace"][name]["ace"]
        cace_val = demo["cace"][name]["ace"]
        print(f"  {name}: ACE={ace_val:.4f}  CACE={cace_val:.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
