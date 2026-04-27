#!/usr/bin/env python
"""Tiny seeded CME experiment config for ``rehearsal-run``."""

import numpy as np

from rehearsal.core import AUFTask, AlterationDomain, DesiredRegion


def build_experiment(params, seed):
    seed = int(seed)
    n_samples = int(params.get("n_samples", 120))
    default_eval_samples = 50
    rng = np.random.default_rng(seed)

    x = rng.uniform(-1.0, 1.0, size=n_samples)
    u = rng.normal(scale=0.5, size=n_samples)
    a = 0.7 * x + 0.4 * u + rng.normal(scale=0.2, size=n_samples)
    y = 1.0 - (a - (0.5 * x + 0.2 * u)) ** 2 + rng.normal(scale=0.05, size=n_samples)
    observed_x = float(params["observed_x"]) if "observed_x" in params else float(rng.uniform(-1.0, 1.0))

    task = AUFTask(
        observed_variables=("x",),
        alterable_variables=("a",),
        outcome_variables=("y",),
        desired_region=DesiredRegion.from_intervals({"y": (0.85, 1.15)}, variables=("y",)),
        alteration_domain=AlterationDomain({"a": (-1.5, 1.5)}),
        candidate_alteration_sets=(("a",),),
        variable_order=("x", "u", "a", "y"),
        metadata={"cme_environment_variables": ("u",)},
    )

    def evaluate(method, task, decision, experiment, n_samples):
        samples = int(n_samples or default_eval_samples)
        eval_rng = np.random.default_rng(seed + 1009)
        obs_x = float(experiment["observation"]["x"])
        action = float(decision.alterations["a"])
        u_eval = eval_rng.normal(scale=0.5, size=samples)
        y_after = 1.0 - (action - (0.5 * obs_x + 0.2 * u_eval)) ** 2 + eval_rng.normal(scale=0.05, size=samples)
        natural_a = 0.7 * obs_x + 0.4 * u_eval + eval_rng.normal(scale=0.2, size=samples)
        y_no_action = 1.0 - (natural_a - (0.5 * obs_x + 0.2 * u_eval)) ** 2 + eval_rng.normal(scale=0.05, size=samples)
        return {
            "true_auf_success_rate": float(np.mean((0.85 <= y_after) & (y_after <= 1.15))),
            "no_action_true_auf_success_rate": float(np.mean((0.85 <= y_no_action) & (y_no_action <= 1.15))),
            "eval_samples": samples,
        }

    return {
        "name": "cme_toy",
        "task": task,
        "data": {"x": x, "u": u, "a": a, "y": y},
        "observation": {"x": observed_x},
        "method_params": {
            "eta_surrogate": 10.0,
            "krr_lambda_alpha": 0.05,
            "krr_lambda_gamma": 0.05,
            "pgd_steps": 60,
            "num_restarts": 8,
        },
        "default_eval_samples": default_eval_samples,
        "evaluate": evaluate,
        "metadata": {
            "source": "examples/cme/cme_toy_experiment.py",
            "data_source": "seeded synthetic non-parametric CME fixture",
            "n_samples": n_samples,
        },
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from rehearsal.experiments.run import main

    raise SystemExit(main([str(Path(__file__)), *sys.argv[1:]]))
