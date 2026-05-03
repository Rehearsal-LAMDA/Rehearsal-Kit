"""Self-contained installed-package demo for Rehearsal."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rehearsal.core import AUFTask, AlterationDomain, DesiredRegion
from rehearsal.experiments.run import run_experiment_configs


def build_experiment(params: Mapping[str, Any], seed: int) -> dict[str, Any]:
    """Build a tiny seeded CARE experiment that ships inside the package."""

    seed = int(seed)
    n_samples = int(params.get("n_samples", 80))
    default_eval_samples = 20
    rng = np.random.default_rng(seed)

    x = rng.normal(size=n_samples)
    z = 0.5 * x + rng.normal(scale=0.2, size=n_samples)
    y = 0.3 * x + 1.4 * z + rng.normal(scale=0.15, size=n_samples)
    observed_x = float(params["observed_x"]) if "observed_x" in params else float(rng.normal())

    task = AUFTask(
        observed_variables=("x",),
        alterable_variables=("z",),
        outcome_variables=("y",),
        desired_region=DesiredRegion.from_intervals({"y": (0.8, 1.2)}, variables=("y",)),
        alteration_domain=AlterationDomain({"z": (-2.0, 2.0)}),
        parents={"z": ("x",), "y": ("x", "z")},
        candidate_alteration_sets=(("z",),),
        variable_order=("x", "z", "y"),
    )

    def evaluate(method: Any, task: AUFTask, decision: Any, experiment: Mapping[str, Any], n_samples: int) -> dict[str, Any]:
        samples = int(n_samples or default_eval_samples)
        eval_rng = np.random.default_rng(seed + 1009)
        obs_x = float(experiment["observation"]["x"])
        z_value = float(decision.alterations["z"])
        y_after = 0.3 * obs_x + 1.4 * z_value + eval_rng.normal(scale=0.15, size=samples)
        natural_z = 0.5 * obs_x + eval_rng.normal(scale=0.2, size=samples)
        y_no_action = 0.3 * obs_x + 1.4 * natural_z + eval_rng.normal(scale=0.15, size=samples)
        return {
            "true_auf_success_rate": float(np.mean((0.8 <= y_after) & (y_after <= 1.2))),
            "no_action_true_auf_success_rate": float(np.mean((0.8 <= y_no_action) & (y_no_action <= 1.2))),
            "eval_samples": samples,
        }

    return {
        "name": "care_toy_installed_demo",
        "task": task,
        "data": {"x": x, "z": z, "y": y},
        "observation": {"x": observed_x},
        "method_params": {"max_iters": 20},
        "default_eval_samples": default_eval_samples,
        "evaluate": evaluate,
        "metadata": {
            "source": "rehearsal.experiments.demo",
            "data_source": "seeded synthetic linear Gaussian SEM",
            "n_samples": n_samples,
            "observation_policy": "sampled from the same seeded SEM unless observed_x is provided",
        },
    }


def run_demo(
    *,
    seed: int = 3,
    n_samples: int = 80,
    eval_samples: int = 20,
    max_iters: int = 20,
    observed_x: float | None = None,
) -> dict[str, Any]:
    """Run the installed-package demo and return the standard batch payload."""

    params: dict[str, Any] = {"n_samples": int(n_samples)}
    if observed_x is not None:
        params["observed_x"] = float(observed_x)
    return run_experiment_configs(
        Path(__file__),
        seeds=(int(seed),),
        method_name="care",
        method_params={"max_iters": int(max_iters)},
        eval_samples=int(eval_samples),
        params=params,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the installed-package demo from the command line."""

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=80)
    parser.add_argument("--eval-samples", type=int, default=20)
    parser.add_argument("--max-iters", type=int, default=20)
    parser.add_argument("--observed-x", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--compact", action="store_true", help="Write compact JSON.")
    args = parser.parse_args(argv)

    result = run_demo(
        seed=args.seed,
        n_samples=args.n_samples,
        eval_samples=args.eval_samples,
        max_iters=args.max_iters,
        observed_x=args.observed_x,
    )
    indent = None if args.compact else 2
    payload = json.dumps(result, indent=indent, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {args.output} (n_runs={result['n_runs']}, method={result['method']})", file=sys.stderr)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
