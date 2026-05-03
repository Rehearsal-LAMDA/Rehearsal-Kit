# Rehearsal Architecture

The package is organized around a small set of stable contracts.

## Task Layer

`rehearsal.core.AUFTask` defines the common AUF problem:

- observed variables `X`
- alterable variables `Z`
- outcome variables `Y`
- desired region `M y <= d`
- alteration bounds and optional costs
- optional sequential stages

All method adapters should consume this task object instead of bespoke script
arguments.

## Method Layer

Every method implements:

```python
fit(data, task, config=None)
suggest(observation, task)
evaluate(task, n_samples)
```

`suggest` returns a `DecisionResult`, regardless of whether the method comes
from graph uncertainty, online SRM, nonlinear differentiable optimization,
sequential AUF, order-based rehearsal, or influence-power evaluation.

## Model Layer

Most rehearsal-learning methods in this repository, except the IJCAI 2025
sequential workflow and ICLR 2026 influence-power analysis, should be organized
as two phases:

1. structural learning: learn a graph, order, or structural parameters from
   observational/interventional data;
2. rehearsal optimization: choose alterations against the fitted structure.

`rehearsal.models` owns the structural-learning side. For example,
`LinearGaussianSRM` and `LinearGaussianSRMLearner` are paper-neutral components
that can be reused by the ICML 2025, NeurIPS 2023, and NeurIPS 2024 adapters.
Heavy dependencies such as torch, flow libraries, or QP solvers should be
imported only inside the modules that need them.

## Rehearsal Layer

`rehearsal.optimizers` owns reusable decision-stage optimizers over fitted structural
models. `rehearsal.methods` should stay thin: it wires a structural learner to a
rehearsal optimizer and returns framework result objects.

## Experiment Configs

`rehearsal.experiments.run` executes Python config files as seeded batches.
Configs should expose `build_experiment(params, seed)`, where `params` contains
CLI experiment parameters and `seed` comes only from `--seeds`. The config
builds the task, generated training data, desired region, and per-seed
observation for that run.

The runner intentionally has no separate single-seed output shape. Passing
`--seeds 3` still returns `runs` plus aggregate `summary` with mean, standard
deviation, min, and max. This keeps paper experiments and follow-up analysis on
one contract.

## Experiment Layer

Experiment runners should create `ExperimentResult` objects with metrics,
seeds, config snapshots, and artifact paths. Plotting and paper-specific
postprocessing should stay outside core method APIs.
