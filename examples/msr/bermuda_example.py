#!/usr/bin/env python
"""Bermuda MSR experiment config for ``rehearsal-run``.

The core ``MSRRehearsal`` adapter does not ship dataset-specific stage layouts; this file
holds the Bermuda paper multi-step ``stages`` and ``stage_observables`` and passes them
via ``method_params``.
"""

import numpy as np

from rehearsal.datasets import (
    bermuda,
    estimate_true_auf_success_rate,
    generate_observational_data,
    sample_observation,
)


BERMUDA_MSR_STAGES = (
    ("Light", "Temp", "Sal"),
    ("DIC", "TA", "Nutrients_PC1", "CO2"),
    ("Omega", "Chla", "pHsw"),
    ("NEC",),
)

BERMUDA_MSR_STAGE_OBSERVABLES = (
    ("Light", "Temp", "Sal"),
    ("CO2",),
    (),
    (),
)


def build_experiment(params, seed):
    seed = int(seed)
    n_data = int(params.get("n_data", 200))
    default_eval_samples = 50
    covariance_profile = str(params.get("covariance_profile", "paper"))

    spec = bermuda(covariance_profile=covariance_profile)
    observation_rng = np.random.default_rng(seed + 1709)
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
                seed=seed + 2703,
            ),
            "no_action_true_auf_success_rate": estimate_true_auf_success_rate(
                spec,
                experiment["observation"],
                {},
                samples,
                seed=seed + 3701,
            ),
            "eval_samples": samples,
        }

    return {
        "name": "msr_bermuda",
        "method_name": "msr",
        "task": spec.task,
        "data": data,
        "observation": observation,
        "method_params": {
            "predictor_type": "linear",
            "feature_degree": 1,
            "stages": BERMUDA_MSR_STAGES,
            "stage_observables": BERMUDA_MSR_STAGE_OBSERVABLES,
            "n_mc_samples": 256,
            "learning_rate": 0.05,
            "epochs": 60,
            "patience": 15,
            "num_restarts": 3,
            "loss": "center_mae",
            "dataset_spec": spec,
        },
        "default_eval_samples": default_eval_samples,
        "evaluate": evaluate,
        "metadata": {
            "dataset": spec.name,
            "source": "examples/msr/bermuda_example.py",
            "method": "MSR / Multi-Step Rehearsal",
            "data_source": "seeded Bermuda SEM used as MSR training data",
            "covariance_profile": covariance_profile,
            "n_data": n_data,
            "stages": BERMUDA_MSR_STAGES,
            "stage_observables": BERMUDA_MSR_STAGE_OBSERVABLES,
            "observation_policy": "sampled from the migrated SEM for each run seed",
            "missing_observations": "filled via simulate_sem(spec, ...) when absent from the observation dict",
        },
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from rehearsal.experiments.run import main

    raise SystemExit(main([str(Path(__file__)), *sys.argv[1:]]))
