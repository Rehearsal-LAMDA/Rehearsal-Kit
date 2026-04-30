#!/usr/bin/env python
"""Bermuda OLEM-Rh experiment config for ``rehearsal-run``."""

import numpy as np

from rehearsal.datasets import (
    bermuda,
    estimate_true_auf_success_rate,
    generate_observational_data,
    sample_observation,
)


def build_experiment(params, seed):
    seed = int(seed)
    n_data = int(params.get("n_data", 200))
    default_eval_samples = 50
    covariance_profile = str(params.get("covariance_profile", "paper"))

    spec = bermuda(covariance_profile=covariance_profile)
    observation_rng = np.random.default_rng(seed + 1809)
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
                seed=seed + 2803,
            ),
            "no_action_true_auf_success_rate": estimate_true_auf_success_rate(
                spec,
                experiment["observation"],
                {},
                samples,
                seed=seed + 3801,
            ),
            "eval_samples": samples,
        }

    return {
        "name": "olem_rh_bermuda",
        "task": spec.task,
        "data": data,
        "observation": observation,
        "method_params": {
            "entropy_estimator": "gaussian",
            "max_parents": None,
            "predictor_type": "linear",
            "feature_degree": 1,
            "n_mc_samples": 96,
            "learning_rate": 0.05,
            "epochs": 30,
            "patience": 10,
            "num_restarts": 3,
            "loss": "center_mae",
        },
        "default_eval_samples": default_eval_samples,
        "evaluate": evaluate,
        "metadata": {
            "dataset": spec.name,
            "source": "examples/olem_rh/bermuda_example.py",
            "reference": "OLEM-Rh/code/olem.py",
            "data_source": "seeded Bermuda SEM used as OLEM-Rh training data",
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
