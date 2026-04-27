#!/usr/bin/env python
"""Tiny seeded CARE experiment config for ``rehearsal-run``.

Example:

PYTHONPATH=src python -m rehearsal.experiments.run examples/care/care_toy_experiment.py \
  --method icml2025-care --seeds 1,2,3 --params n_samples=100 --eval-samples 50
"""

import numpy as np

from rehearsal.core import AUFTask, AlterationDomain, DesiredRegion


def build_experiment(params, seed):
    seed = int(seed)
    n_samples = int(params.get("n_samples", 240))
    default_eval_samples = 50
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

    def evaluate(method, task, decision, experiment, n_samples):
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
        "name": "care_toy",
        "task": task,
        "data": {"x": x, "z": z, "y": y},
        "observation": {"x": observed_x},
        "method_params": {"max_iters": 50},
        "default_eval_samples": default_eval_samples,
        "evaluate": evaluate,
        "metadata": {
            "source": "examples/care/care_toy_experiment.py",
            "data_source": "seeded synthetic linear Gaussian SEM",
            "n_samples": n_samples,
            "observation_policy": "sampled from the same seeded SEM unless observed_x is provided",
        },
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from rehearsal.experiments.run import main

    raise SystemExit(main([str(Path(__file__)), *sys.argv[1:]]))
