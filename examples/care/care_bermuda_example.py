#!/usr/bin/env python
"""Bermuda CARE experiment config for ``rehearsal-run``.

Example:

PYTHONPATH=src python -m rehearsal.experiments.run examples/care/care_bermuda_example.py \
  --method icml2025-care --seeds 1,2,3 --params n_data=100 --eval-samples 500
"""

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
            "paper_claim": dict(spec.paper_claim),
        }

    return {
        "name": "care_bermuda",
        "task": spec.task,
        "data": data,
        "observation": observation,
        "method_params": {"max_iters": 50},
        "default_eval_samples": default_eval_samples,
        "evaluate": evaluate,
        "metadata": {
            "dataset": spec.name,
            "source": "examples/care/care_bermuda_example.py",
            "data_source": "seeded ICML 2025 Bermuda linear Gaussian SEM migration",
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
