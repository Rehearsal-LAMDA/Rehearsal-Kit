"""Reproduce the ICML 2025 CARE dataset claims."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rehearsal.datasets import (
    RehearsalDatasetSpec,
    bermuda,
    estimate_true_auf_success_rate,
    generate_observational_data,
    manage,
    sample_observation,
)
from rehearsal.methods import CARERehearsal


def run_reproduction(
    dataset: str = "all",
    *,
    runs: int = 20,
    val_samples: int = 1000,
    rounds: int = 100,
    seed: int = 202405,
    n_data: int | None = None,
    max_iters: int = 200,
    bermuda_covariance_profile: str = "paper",
) -> dict[str, Any]:
    specs = _select_specs(dataset, bermuda_covariance_profile)
    results = {}
    for spec in specs:
        results[spec.name] = run_dataset(
            spec,
            runs=runs,
            val_samples=val_samples,
            rounds=rounds,
            seed=seed,
            n_data=n_data if n_data is not None else spec.default_n_data,
            max_iters=max_iters,
        )
    return results


def run_dataset(
    spec: RehearsalDatasetSpec,
    *,
    runs: int,
    val_samples: int,
    rounds: int,
    seed: int,
    n_data: int,
    max_iters: int,
) -> dict[str, Any]:
    seed_rng = np.random.default_rng(seed)
    run_seeds = seed_rng.choice(np.arange(max(runs * 50, runs + 1)), size=runs, replace=False)
    rows = []
    for run_seed in run_seeds:
        rows.append(
            _run_once(
                spec,
                run_seed=int(run_seed),
                n_data=n_data,
                val_samples=val_samples,
                rounds=rounds,
                max_iters=max_iters,
            )
        )

    summary = {
        "dataset": spec.name,
        "runs": int(runs),
        "n_data": int(n_data),
        "val_samples": int(val_samples),
        "rounds": int(rounds),
        "paper_claim": dict(spec.paper_claim),
        "ours_care_success_percent": _mean_std([row["ours_care_success"] * 100.0 for row in rows]),
        "ours_100_round_success": _mean_std([row["ours_round_success"] for row in rows]),
        "ours_avg_time_ms": _mean_std([row["ours_runtime_ms"] for row in rows]),
        "no_action_care_success_percent": _mean_std([row["no_action_care_success"] * 100.0 for row in rows]),
        "no_action_100_round_success": _mean_std([row["no_action_round_success"] for row in rows]),
        "random_care_success_percent": _mean_std([row["random_care_success"] * 100.0 for row in rows]),
        "random_100_round_success": _mean_std([row["random_round_success"] for row in rows]),
    }
    return {"summary": summary, "runs": rows}


def _run_once(
    spec: RehearsalDatasetSpec,
    *,
    run_seed: int,
    n_data: int,
    val_samples: int,
    rounds: int,
    max_iters: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(run_seed)
    train = generate_observational_data(spec, n_data, seed=run_seed)
    method = CARERehearsal(seed=run_seed, max_iters=max_iters).fit(train, spec.task)
    observation = sample_observation(spec, rng)

    result = method.suggest(observation, spec.task)
    ours_care_success = estimate_true_auf_success_rate(
        spec,
        observation,
        result.alterations,
        val_samples,
        rng=rng,
    )
    no_action_care_success = estimate_true_auf_success_rate(spec, observation, {}, val_samples, rng=rng)
    random_alterations = _random_alterations(spec, rng)
    random_care_success = estimate_true_auf_success_rate(spec, observation, random_alterations, val_samples, rng=rng)

    ours_success = 0
    no_action_success = 0
    random_success = 0
    round_runtime = 0.0
    for _ in range(rounds):
        obs = sample_observation(spec, rng)
        no_action_success += int(estimate_true_auf_success_rate(spec, obs, {}, 1, rng=rng) > 0.0)
        rand_alt = _random_alterations(spec, rng)
        random_success += int(estimate_true_auf_success_rate(spec, obs, rand_alt, 1, rng=rng) > 0.0)
        start = time.perf_counter()
        round_result = method.suggest(obs, spec.task)
        round_runtime += time.perf_counter() - start
        ours_success += int(
            estimate_true_auf_success_rate(spec, obs, round_result.alterations, 1, rng=rng) > 0.0
        )

    return {
        "seed": int(run_seed),
        "ours_care_success": float(ours_care_success),
        "ours_round_success": int(ours_success),
        "ours_runtime_ms": float((result.runtime_seconds + round_runtime / max(rounds, 1)) * 1000.0),
        "selected_candidate": tuple(result.diagnostics["selected_candidate"]),
        "estimated_inner_care_success": float(result.estimated_success_probability),
        "no_action_care_success": float(no_action_care_success),
        "no_action_round_success": int(no_action_success),
        "random_care_success": float(random_care_success),
        "random_round_success": int(random_success),
    }


def _random_alterations(spec: RehearsalDatasetSpec, rng: np.random.Generator) -> dict[str, float]:
    candidate_sets = tuple(tuple(candidate) for candidate in spec.task.candidate_alteration_sets)
    candidate = candidate_sets[int(rng.integers(0, len(candidate_sets)))]
    lower, upper = spec.task.alteration_domain.arrays_for(candidate)
    values = rng.uniform(lower, upper)
    return {name: float(value) for name, value in zip(candidate, values)}


def _select_specs(dataset: str, bermuda_covariance_profile: str) -> Sequence[RehearsalDatasetSpec]:
    if dataset == "all":
        return (manage(), bermuda(covariance_profile=bermuda_covariance_profile))
    if dataset == "manage":
        return (manage(),)
    if dataset == "bermuda":
        return (bermuda(covariance_profile=bermuda_covariance_profile),)
    raise ValueError(f"Unknown dataset: {dataset}")


def _mean_std(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}


def _print_summary(results: Mapping[str, Any]) -> None:
    for name, payload in results.items():
        summary = payload["summary"]
        print(f"\n{name}")
        print(f"  runs={summary['runs']} n_data={summary['n_data']} val_samples={summary['val_samples']} rounds={summary['rounds']}")
        _print_metric(
            "ours CARE Success (%)",
            summary["ours_care_success_percent"],
            summary["paper_claim"]["ours_care_success_percent"],
        )
        _print_metric("ours 100-Rds Succ. Freq.", summary["ours_100_round_success"], summary["paper_claim"]["ours_100_round_success"])
        _print_metric("ours Avg. Time (ms)", summary["ours_avg_time_ms"], summary["paper_claim"]["ours_avg_time_ms"])
        _print_metric("no action CARE Success (%)", summary["no_action_care_success_percent"], None)
        _print_metric("random CARE Success (%)", summary["random_care_success_percent"], None)


def _print_metric(label: str, metric: Mapping[str, float], paper_value: float | None) -> None:
    suffix = "" if paper_value is None else f" | paper ours {paper_value:.2f}"
    print(f"  {label}: {metric['mean']:.2f} +- {metric['std']:.2f}{suffix}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("all", "manage", "bermuda"), default="all")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--val-samples", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=202405)
    parser.add_argument("--n-data", type=int, default=None)
    parser.add_argument("--max-iters", type=int, default=200)
    parser.add_argument("--bermuda-covariance-profile", choices=("paper", "legacy_isotropic"), default="paper")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    results = run_reproduction(
        args.dataset,
        runs=args.runs,
        val_samples=args.val_samples,
        rounds=args.rounds,
        seed=args.seed,
        n_data=args.n_data,
        max_iters=args.max_iters,
        bermuda_covariance_profile=args.bermuda_covariance_profile,
    )
    _print_summary(results)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
