#!/usr/bin/env python
"""Bermuda CME experiment config for ``rehearsal-run``."""

from dataclasses import replace

import numpy as np

from rehearsal.datasets.icml2025 import (
    bermuda_icml2025,
    estimate_true_auf_success_rate,
    generate_observational_data,
    sample_observation,
)


def build_experiment(params, seed):
    seed = int(seed)
    n_data = int(params.get("n_data", 200))
    default_eval_samples = 50
    covariance_profile = str(params.get("covariance_profile", "paper"))

    spec = bermuda_icml2025(covariance_profile=covariance_profile)
    task = replace(
        spec.task,
        metadata={**dict(spec.task.metadata), "method_family": "unpublished CME"},
    )
    observation_rng = np.random.default_rng(seed + 1009)
    data = generate_observational_data(spec, n_data, seed=seed)
    observation = sample_observation(spec, observation_rng)

    def evaluate(method, task, decision, experiment, n_samples):
        samples = int(n_samples or default_eval_samples)
        return {
            "true_auf_success_rate": estimate_true_auf_success_rate(
                spec,
                experiment["observation"],
                decision.alterations,
                samples,
                seed=seed + 2003,
            ),
            "no_action_true_auf_success_rate": estimate_true_auf_success_rate(
                spec,
                experiment["observation"],
                {},
                samples,
                seed=seed + 3001,
            ),
            "eval_samples": samples,
        }

    return {
        "name": "cme_bermuda",
        "task": task,
        "data": data,
        "observation": observation,
        "method_params": {
            "eta_surrogate": 10.0,
            "krr_lambda_alpha": 0.095,
            "krr_lambda_gamma": 10.0,
            "pgd_lr": 0.05,
            "pgd_steps": 100,
            "num_restarts": 10,
        },
        "default_eval_samples": default_eval_samples,
        "evaluate": evaluate,
        "metadata": {
            "dataset": spec.name,
            "source": "examples/cme/cme_bermuda_example.py",
            "data_source": "seeded ICML 2025 Bermuda SEM used as CME training data",
            "covariance_profile": covariance_profile,
            "n_data": n_data,
            "observation_policy": "sampled from the migrated SEM for each run seed",
        },
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from rehearsal.experiments.run import main

    raise SystemExit(main([str(Path(__file__)), *sys.argv[1:]]))
