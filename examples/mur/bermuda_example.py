#!/usr/bin/env python
"""Bermuda MUR experiment config for ``rehearsal-run``.

The README reference uses the same static Bermuda benchmark as the other
method examples.  MUR learns its instantaneous matrix from the same seeded
training data and uses an empty lagged-parent map, so ``B = 0`` and there is no
inter-round influence in this comparison.  Custom MUR configs can still pass a
nonzero lagged matrix through ``fit_config['mur_B']``.

Example:

PYTHONPATH=src python -m rehearsal.experiments.run examples/mur/bermuda_example.py \
  --method mur --seeds 3 --method-params variant=gmur,horizon=0 \
  --params n_data=2000 --eval-samples 1000 --output outputs/mur_bermuda_seed3.json --compact
"""

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
    default_eval_samples = 100
    covariance_profile = str(params.get("covariance_profile", "paper"))
    dataset_spec = bermuda(covariance_profile=covariance_profile)
    task = dataset_spec.task
    data = generate_observational_data(dataset_spec, n_data, seed=seed)
    observation_rng = np.random.default_rng(seed + 1009)
    observation = sample_observation(dataset_spec, observation_rng)
    observation.update({f"prev__{name}": 0.0 for name in task.variable_order})

    def evaluate(method, task, decision, experiment, n_samples):
        samples = int(n_samples or default_eval_samples)
        return {
            "true_auf_success_rate": estimate_true_auf_success_rate(
                dataset_spec,
                experiment["observation"],
                decision.alterations,
                samples,
                seed=seed + 2003,
            ),
            "no_action_true_auf_success_rate": estimate_true_auf_success_rate(
                dataset_spec,
                experiment["observation"],
                {},
                samples,
                seed=seed + 3001,
            ),
            "eval_samples": samples,
            "lagged_matrix_profile": "zero",
        }

    return {
        "name": "mur_bermuda",
        "task": task,
        "data": data,
        "observation": observation,
        "method_params": {"n_mc_samples": 500, "max_iters": 100, "num_restarts": 2},
        "fit_config": {"mur_lagged_parents": {}},
        "default_eval_samples": default_eval_samples,
        "evaluate": evaluate,
        "metadata": {
            "dataset": dataset_spec.name,
            "source": "examples/mur/bermuda_example.py",
            "data_source": "seeded Bermuda linear Gaussian SEM migration",
            "covariance_profile": covariance_profile,
            "n_data": n_data,
            "inter_round_influence": "disabled",
            "lagged_matrix_profile": "zero",
            "package_horizon_for_reference": 0,
            "alterable_variables": task.alterable,
        },
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from rehearsal.experiments.run import main

    raise SystemExit(main([str(Path(__file__)), *sys.argv[1:]]))
