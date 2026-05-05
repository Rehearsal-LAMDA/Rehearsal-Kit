#!/usr/bin/env python
"""Tiny seeded MUR experiment config for ``rehearsal-run``.

Example:

PYTHONPATH=src python -m rehearsal.experiments.run examples/mur/mur_toy_experiment.py \
  --method mur --seeds 3 --method-params variant=farmur,horizon=2 --eval-samples 100
"""

import numpy as np

from rehearsal.core import AUFTask, AlterationDomain, DesiredRegion
from rehearsal.core.data import candidate_alteration_sets
from rehearsal.metrics.mur import infer_mur_region_center
from rehearsal.models import LinearTimeSeriesSRM
from rehearsal.optimizers import rollout_mur_policy


VARIABLES = ("x", "z", "y")
A_TRUE = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.2, 0.0, 0.0],
        [0.4, 1.1, 0.0],
    ],
    dtype=float,
)
B_TRUE = np.array(
    [
        [0.45, 0.0, 0.0],
        [0.0, 0.10, 0.0],
        [0.0, -0.35, 0.20],
    ],
    dtype=float,
)
NOISE_COVARIANCE = np.diag([0.04, 0.03, 0.05])


def build_experiment(params, seed):
    seed = int(seed)
    n_samples = int(params.get("n_samples", 240))
    default_eval_samples = 80
    rng = np.random.default_rng(seed)
    model = _true_model()
    data = _simulate_series(model, n_samples, rng)
    previous_state = np.asarray([data[name][-1] for name in VARIABLES], dtype=float)
    current_natural = model.sample_next(previous_state, rng=rng)
    observed_x = float(params.get("observed_x", current_natural[0]))

    task = AUFTask(
        observed_variables=("x",),
        alterable_variables=("z",),
        outcome_variables=("y",),
        desired_region=DesiredRegion.from_intervals({"y": (0.85, 1.15)}, variables=("y",)),
        alteration_domain=AlterationDomain({"z": (-2.0, 2.0)}),
        parents={"z": ("x",), "y": ("x", "z")},
        candidate_alteration_sets=(("z",),),
        variable_order=VARIABLES,
        metadata={"mur_lagged_parents": {"x": ("x",), "z": ("z",), "y": ("z", "y")}},
    )
    observation = {"x": observed_x}
    observation.update({f"prev__{name}": float(value) for name, value in zip(VARIABLES, previous_state)})

    def evaluate(method, task, decision, experiment, n_samples):
        samples = int(n_samples or default_eval_samples)
        eval_rng = np.random.default_rng(seed + 2003)
        center, _ = infer_mur_region_center(task, experiment["fit_config"])
        x_t = np.asarray([experiment["observation"][name] for name in task.observed], dtype=float)
        v_prev = _previous_state_from_observation(experiment["observation"], task)
        rollout = rollout_mur_policy(
            model,
            task,
            variant=method.variant,
            horizon=method.horizon,
            x_t=x_t,
            v_prev=v_prev,
            center=center,
            candidates=candidate_alteration_sets(task),
            rng=eval_rng,
            n_samples=samples,
            learning_rate=method.learning_rate,
            max_iters=method.max_iters,
            tolerance=method.tolerance,
            num_restarts=method.num_restarts,
        )
        return {
            "true_auf_success_rate": rollout["aggregate_success_rate"],
            "no_action_true_auf_success_rate": _estimate_no_action_success(
                model,
                task,
                experiment["observation"],
                method.horizon,
                samples,
                seed + 3001,
            ),
            "eval_samples": samples,
        }

    return {
        "name": "mur_toy",
        "task": task,
        "data": data,
        "observation": observation,
        "method_params": {"n_mc_samples": 80, "max_iters": 80, "num_restarts": 2},
        "fit_config": {"mur_A": A_TRUE, "mur_B": B_TRUE, "mur_noise_covariance": NOISE_COVARIANCE},
        "default_eval_samples": default_eval_samples,
        "evaluate": evaluate,
        "metadata": {
            "source": "examples/mur/mur_toy_experiment.py",
            "data_source": "seeded linear additive time-series SRM",
            "n_samples": n_samples,
            "observation_policy": "current x sampled from the same time-series SRM unless observed_x is provided",
        },
    }


def _true_model():
    return LinearTimeSeriesSRM(VARIABLES, A_TRUE, B_TRUE, NOISE_COVARIANCE)


def _simulate_series(model, n_samples, rng):
    previous = np.zeros(model.n_variables, dtype=float)
    rows = []
    for _ in range(int(n_samples)):
        previous = model.sample_next(previous, rng=rng)
        rows.append(previous.copy())
    matrix = np.vstack(rows)
    return {name: matrix[:, idx] for idx, name in enumerate(VARIABLES)}


def _previous_state_from_observation(observation, task):
    return np.asarray([float(observation[f"prev__{name}"]) for name in task.variable_order], dtype=float)


def _estimate_no_action_success(model, task, observation, horizon, n_samples, seed):
    rng = np.random.default_rng(seed)
    previous_initial = _previous_state_from_observation(observation, task)
    x_initial = {name: float(observation[name]) for name in task.observed}
    E_y = model.selection_matrix(task.outcomes)
    successes = []
    for _ in range(int(n_samples)):
        previous = previous_initial.copy()
        outcomes = []
        for step in range(int(horizon) + 1):
            noise = rng.multivariate_normal(np.zeros(model.n_variables), model.covariance)
            fixed = x_initial if step == 0 else None
            current = model.solve_next_state(previous, noise, fixed_values=fixed)
            outcomes.append((E_y.T @ current).reshape(-1))
            previous = current
        aggregate_y = np.mean(np.vstack(outcomes), axis=0)
        successes.append(bool(task.desired_region.contains(aggregate_y)))
    return float(np.mean(successes))


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from rehearsal.experiments.run import main

    raise SystemExit(main([str(Path(__file__)), *sys.argv[1:]]))
