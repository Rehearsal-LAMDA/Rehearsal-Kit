"""Run a Rehearsal experiment from a Python config file."""

from __future__ import annotations

import argparse
import inspect
import importlib.util
import json
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rehearsal.core import AUFTask, DecisionResult, RehearsalMethod
from rehearsal.methods.registry import available_methods, create_method

DEFAULT_METHOD_NAME = "icml2025-care"
SUMMARY_EVALUATION_METRICS = (
    "true_auf_success_rate",
    "no_action_true_auf_success_rate",
)
SUMMARY_DECISION_METRICS = (
    "runtime_seconds",
)
SUMMARY_STRUCTURAL_LEARNING_METRICS = (
    "runtime_seconds",
)


def run_experiment_configs(
    config_path: str | Path,
    *,
    seeds: Sequence[int],
    method_name: str | None = None,
    method_params: Mapping[str, Any] | None = None,
    fit_config: Mapping[str, Any] | None = None,
    eval_samples: int | None = None,
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a seed list and summarize numeric outputs.

    The experiment runner intentionally has no single-run mode. Even
    ``seeds=(1,)`` returns a batch payload with ``runs`` and ``summary`` so that
    downstream analysis always consumes one output shape.
    """

    seed_tuple = _normalize_seeds(seeds)
    base_params = dict(params or {})
    base_method_params = dict(method_params or {})
    _reject_seed_override(base_params, "--params")
    _reject_seed_override(base_method_params, "--method-params")

    runs = [
        _run_seeded_experiment(
            config_path,
            seed=int(seed),
            method_name=method_name,
            method_params=base_method_params,
            fit_config=fit_config,
            eval_samples=eval_samples,
            params=base_params,
        )
        for seed in seed_tuple
    ]

    first = runs[0]
    return {
        "name": first["name"],
        "method": first["method"],
        "seeds": [int(seed) for seed in seed_tuple],
        "n_runs": len(runs),
        "runs": runs,
        "summary": _summarize_runs(runs),
    }


def _run_seeded_experiment(
    config_path: str | Path,
    *,
    seed: int,
    method_name: str | None,
    method_params: Mapping[str, Any] | None,
    fit_config: Mapping[str, Any] | None,
    eval_samples: int | None,
    params: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Load one seeded experiment config, run fit/suggest/evaluate, and return JSON-ready results."""

    experiment = _load_experiment(config_path, params or {}, seed=seed)
    task = _required(experiment, "task")
    if not isinstance(task, AUFTask):
        raise TypeError("Experiment config field 'task' must be an AUFTask.")
    data = _required(experiment, "data")
    observation = dict(_required(experiment, "observation"))

    method = _resolve_method(experiment, method_name, method_params, seed=seed)
    resolved_fit_config = {**dict(experiment.get("fit_config", {})), **dict(fit_config or {})}
    fit_start = time.perf_counter()
    method.fit(data, task, resolved_fit_config)
    structural_learning_runtime_seconds = time.perf_counter() - fit_start
    decision = method.suggest(observation, task)

    resolved_eval_samples = eval_samples
    if resolved_eval_samples is None:
        resolved_eval_samples = experiment.get(
            "default_eval_samples",
            experiment.get("eval_samples", experiment.get("evaluate_samples", 0)),
        )
    if callable(experiment.get("evaluate")):
        evaluation = experiment["evaluate"](
            method=method,
            task=task,
            decision=decision,
            experiment=experiment,
            n_samples=int(resolved_eval_samples or 0),
        )
    elif resolved_eval_samples:
        evaluation = method.evaluate(task, int(resolved_eval_samples))
    else:
        evaluation = {}

    return {
        "name": experiment.get("name", Path(config_path).stem),
        "method": _method_label(method, method_name, experiment),
        "seed": int(seed),
        "metadata": _jsonable(experiment.get("metadata", {})),
        "observation": _jsonable(observation),
        "structural_learning": {
            "runtime_seconds": float(structural_learning_runtime_seconds),
        },
        "decision": _decision_to_dict(decision),
        "evaluation": _jsonable(evaluation),
    }


def _load_experiment(config_path: str | Path, params: Mapping[str, Any], *, seed: int) -> Mapping[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(f"rehearsal_user_experiment_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load experiment config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for name in ("build_experiment", "create_experiment", "get_experiment"):
        builder = getattr(module, name, None)
        if callable(builder):
            return dict(_call_experiment_builder(builder, params, seed))
    if hasattr(module, "EXPERIMENT"):
        experiment = getattr(module, "EXPERIMENT")
        if callable(experiment):
            experiment = _call_experiment_builder(experiment, params, seed)
        return dict(experiment)
    raise AttributeError(
        "Experiment config must define build_experiment(params), create_experiment(params), "
        "get_experiment(params), or EXPERIMENT."
    )


def _call_experiment_builder(builder: Any, params: Mapping[str, Any], seed: int) -> Mapping[str, Any]:
    """Call config factories with the seeded-batch contract.

    New configs should define ``build_experiment(params, seed)`` or
    ``build_experiment(params, *, seed)``. A legacy one-argument factory still
    works, but receives ``seed`` inside params only as a migration fallback.
    """

    signature = inspect.signature(builder)
    parameters = signature.parameters
    if _accepts_keyword_seed(parameters):
        return builder(dict(params), seed=int(seed))
    if _accepts_positional_seed(parameters):
        return builder(dict(params), int(seed))

    legacy_params = {**dict(params), "seed": int(seed)}
    return builder(legacy_params)


def _accepts_keyword_seed(parameters: Mapping[str, inspect.Parameter]) -> bool:
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return True
    parameter = parameters.get("seed")
    return parameter is not None and parameter.kind in {
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    }


def _accepts_positional_seed(parameters: Mapping[str, inspect.Parameter]) -> bool:
    positional = {
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }
    if any(parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in parameters.values()):
        return True
    return sum(1 for parameter in parameters.values() if parameter.kind in positional) >= 2


def _resolve_method(
    experiment: Mapping[str, Any],
    method_name: str | None,
    method_params: Mapping[str, Any] | None,
    *,
    seed: int,
) -> RehearsalMethod:
    if experiment.get("method") is not None and method_name is None:
        return experiment["method"]
    resolved_name = method_name or experiment.get("method_name", DEFAULT_METHOD_NAME)
    experiment_method_params = dict(experiment.get("method_params", {}))
    cli_method_params = dict(method_params or {})
    _reject_seed_override(experiment_method_params, "experiment method_params")
    _reject_seed_override(cli_method_params, "--method-params")
    resolved_params = {**experiment_method_params, **cli_method_params, "seed": int(seed)}
    return create_method(str(resolved_name), resolved_params)


def _required(experiment: Mapping[str, Any], key: str) -> Any:
    if key not in experiment:
        raise KeyError(f"Experiment config is missing required field {key!r}.")
    return experiment[key]


def _method_label(method: RehearsalMethod, method_name: str | None, experiment: Mapping[str, Any]) -> str:
    if method_name is not None:
        return method_name
    if experiment.get("method_name") is not None:
        return str(experiment["method_name"])
    if experiment.get("method") is not None:
        return type(method).__name__
    return DEFAULT_METHOD_NAME


def _decision_to_dict(decision: DecisionResult) -> dict[str, Any]:
    return {
        "alterations": _jsonable(dict(decision.alterations)),
        "estimated_success_probability": float(decision.estimated_success_probability),
        "cost": float(decision.cost),
        "diagnostics": _jsonable(dict(decision.diagnostics)),
        "runtime_seconds": float(decision.runtime_seconds),
    }


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _summarize_runs(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    paths: dict[str, list[float]] = {}
    for run in runs:
        structural_learning = run.get("structural_learning", {})
        if isinstance(structural_learning, Mapping):
            for metric in SUMMARY_STRUCTURAL_LEARNING_METRICS:
                value = structural_learning.get(metric)
                if isinstance(value, (bool, np.bool_)):
                    continue
                if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
                    paths.setdefault(f"structural_learning.{metric}", []).append(float(value))
        decision = run.get("decision", {})
        if isinstance(decision, Mapping):
            for metric in SUMMARY_DECISION_METRICS:
                value = decision.get(metric)
                if isinstance(value, (bool, np.bool_)):
                    continue
                if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
                    paths.setdefault(f"decision.{metric}", []).append(float(value))
        evaluation = run.get("evaluation", {})
        if not isinstance(evaluation, Mapping):
            continue
        for metric in SUMMARY_EVALUATION_METRICS:
            value = evaluation.get(metric)
            if isinstance(value, (bool, np.bool_)):
                continue
            if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
                paths.setdefault(f"evaluation.{metric}", []).append(float(value))
    summary = {}
    for path, values in sorted(paths.items()):
        if len(values) != len(runs):
            continue
        arr = np.asarray(values, dtype=float)
        summary[path] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
    return summary


def _parse_key_value(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("Expected KEY=VALUE.")
    key, value = raw.split("=", 1)
    if not key:
        raise argparse.ArgumentTypeError("Expected non-empty KEY.")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = value
    return key, parsed


def _merge_pairs(pairs: Sequence[tuple[str, Any]] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key, value in pairs or ():
        merged[key] = value
    return merged


def _normalize_seeds(seeds: Sequence[int] | None) -> tuple[int, ...]:
    if seeds is None:
        raise ValueError("A seed list is required. Use --seeds 1 for a one-seed batch.")
    normalized = tuple(int(seed) for seed in seeds)
    if not normalized:
        raise ValueError("A seed list must include at least one integer.")
    return normalized


def _reject_seed_override(values: Mapping[str, Any], source: str) -> None:
    if "seed" in values:
        raise ValueError(f"Do not pass seed through {source}; use the runner seed list instead.")


def _parse_seeds(raw: str | None) -> tuple[int, ...] | None:
    if raw is None or raw == "":
        return None
    seeds = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        seeds.append(int(part))
    if not seeds:
        raise argparse.ArgumentTypeError("--seeds must include at least one integer.")
    return tuple(seeds)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("config", type=Path, help="Python file defining the experiment.")
    parser.add_argument("--method", choices=available_methods(), default=None)
    parser.add_argument("--method-params", action="append", type=_parse_key_value, default=[])
    parser.add_argument("--fit-params", action="append", type=_parse_key_value, default=[])
    parser.add_argument("--params", action="append", type=_parse_key_value, default=[])
    parser.add_argument("--seeds", type=_parse_seeds, required=True, help="Comma-separated seeds, e.g. 1 or 1,2,3.")
    parser.add_argument("--eval-samples", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--compact", action="store_true", help="Write compact JSON.")
    args = parser.parse_args(argv)

    try:
        result = run_experiment_configs(
            args.config,
            seeds=args.seeds,
            method_name=args.method,
            method_params=_merge_pairs(args.method_params),
            fit_config=_merge_pairs(args.fit_params),
            eval_samples=args.eval_samples,
            params=_merge_pairs(args.params),
        )
    except ValueError as exc:
        parser.error(str(exc))
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
