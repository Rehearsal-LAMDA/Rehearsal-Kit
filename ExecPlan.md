# Rehearsal Open-Source Package Integration ExecPlan

## Purpose

Convert the heterogeneous rehearsal-learning research code in `previous_works/`
into a unified, maintainable Python package. The package should make future
research code share one framework for tasks, data, SRM/SEM models, methods,
metrics, experiments, and results instead of maintaining one-off scripts for
each paper.

This plan is self-contained because `.agent/PLANS.md` is not present in this
repository.

## Scope

Port in this round:

- NeurIPS 2023 rehearsal learning under graph uncertainty.
- NeurIPS 2024 online/time-varying linear SRM rehearsal.
- AAAI 2025 nonlinear and multivariate rehearsal.
- ICML 2025 CARE/circular-region rehearsal.
- IJCAI 2025 sequential-decision AUF.
- ICLR 2026 influence-power measurement.
- Unpublished order-based rehearsal.

Reference only in this round:

- FCS 2022.
- NeurIPS 2025.
- Unpublished non-parametric rehearsal via conditional mean embeddings: expose
  extension interfaces only, no full implementation yet.

## Target Architecture

Use a `src/rehearsal/` package layout:

- `rehearsal.core`: canonical variable schema, graph/order representation, AUF task
  specification, desired-region specification, alteration domains,
  stage/sequential metadata, seed/device helpers, result serialization.
- `rehearsal.models`: SRM/SEM estimator and sampler interfaces, linear Gaussian SEM,
  nonlinear/flow hooks, order-based sampler hooks, and conditional mean
  embedding interfaces.
- `rehearsal.optimizers`: reusable decision-stage optimizers over fitted structural
  models. For most non-IJCAI/ICLR methods, keep the structural-learning phase
  in `rehearsal.models` and the rehearsal decision optimization phase here.
- `rehearsal.methods`: method adapters with one common API:
  `fit(data, task, config)`, `suggest(observation, task)`, and
  `evaluate(task, n_samples)`. Adapters should be thin orchestration layers
  that bind a structural learner to a rehearsal optimizer.
- `rehearsal.datasets`: synthetic fixtures and reusable loaders/factories for
  Bermuda/SEM `.mat`, traffic, and market data.
- `rehearsal.experiments`: config-driven entrypoints, standardized output
  directories, result serialization, and smoke-run helpers.
- `rehearsal.metrics`: success probability, AUF probability, probability bounds,
  cost, runtime, regret/online success rate, and influence power.

Public interfaces that all paper ports must honor:

- `AUFTask`: observed variables `X`, alterable variables `Z`, outcome variables
  `Y`, desired region `M y <= d`, alteration ranges, optional costs, optional
  sequential stages.
- `RehearsalMethod`: protocol for algorithm adapters.
- `DecisionResult`: selected alterations, estimated success probability, cost,
  diagnostics, runtime.
- `ExperimentResult`: per-run metrics, seed, config snapshot, artifact paths.

Legacy code in `previous_works/` is read-only source material. New code should
extract algorithms and testable behavior while avoiding script-level globals and
plotting-heavy experiment code in package APIs.

## Work Packages

### Framework Agent

Ownership:

- `src/rehearsal/core/`
- `src/rehearsal/models/base.py`
- `src/rehearsal/models/nonparametric.py`
- `src/rehearsal/datasets/`
- `src/rehearsal/experiments/`
- `src/rehearsal/metrics/`
- package docs and smoke tests

Tasks:

- Scaffold the package and public API.
- Define dataclasses/protocols for tasks, desired regions, alteration bounds,
  sequential stages, method results, experiment results, and samplers.
- Add graph/order utilities that can support DAG, partial-order, and learned
  order methods.
- Add seed and device helpers.
- Add JSON-friendly result serialization.
- Add a conditional-mean-embedding/non-parametric extension interface only.
- Provide toy synthetic fixtures and contract tests for method adapters.
- Keep dependency additions out of runtime metadata unless explicitly approved.

Acceptance criteria:

- A developer can import `rehearsal.AUFTask`, `rehearsal.DesiredRegion`,
  `rehearsal.DecisionResult`, and `rehearsal.RehearsalMethod`.
- A tiny synthetic AUF task runs through at least one adapter and returns a
  bounded `DecisionResult`.
- `pytest` passes on CPU without historical data downloads.

### Agent A: IJCAI 2025 + ICLR 2026

Ownership:

- `src/rehearsal/methods/sequential.py`
- `src/rehearsal/metrics/influence.py`
- tests for sequential and influence-power behavior

Tasks:

- Port IJCAI sequential AUF decision workflow with stage-wise observation
  updates and retrospective inference.
- Port ICLR influence-power measurement as an estimator API using Monte Carlo
  evaluation under maximum expected utility.
- Reuse `AUFTask`, `DecisionResult`, and sequential-stage metadata.

Acceptance criteria:

- Single-stage sequential AUF matches the non-sequential adapter on a toy task.
- Two-stage toy task improves or preserves estimated success probability after
  updated observations.
- Influence-power ranking is correct on a controlled linear SEM.

### Agent B: AAAI 2025 + Order-Based

Ownership:

- `src/rehearsal/methods/nonlinear.py`
- `src/rehearsal/methods/order_based.py`
- `src/rehearsal/models/nonlinear.py`
- `src/rehearsal/models/order.py`
- tests for nonlinear/multivariate and order-based behavior

Tasks:

- Port AAAI 2025 nonlinear/multivariate rehearsal from
  `previous_works/04-AAAI 2025/code`.
- Expose linear and flow predictors through package model interfaces.
- Implement differentiable alteration optimization with multivariate alteration
  ranges and loss selection.
- Implement order-based rehearsal from the unpublished paper: order-learning
  interface, order-based sampler, and differentiable AUF optimization objective.
- Keep heavy dependencies optional; do not add production dependencies without
  confirmation.

Acceptance criteria:

- Alteration range normalization handles scalar, per-variable, and fixed-value
  ranges.
- Differentiable optimizer returns correct shape and respects bounds.
- Learned-order sampler respects known topological order on a toy task.
- A reduced synthetic task reproduces expected qualitative behavior.

### Agent C: NeurIPS 2023 + NeurIPS 2024 + ICML 2025

Ownership:

- `src/rehearsal/methods/graph_uncertainty.py`
- `src/rehearsal/methods/online.py`
- `src/rehearsal/methods/care.py`
- shared solver tests

Tasks:

- Port NeurIPS 2023 graph uncertainty, SEM learning, interval finding,
  success-probability bounds, and information-gain intervention selection from
  `previous_works/02-NeurIPS 2023/code`.
- Port NeurIPS 2024 online/time-varying linear SRM solver and cost-aware
  candidate rehearsal selection from `previous_works/03-NeurIPS 2024/code`.
- Port ICML 2025 CARE and circular-region solver from
  `previous_works/05-ICML 2025/code`.
- Normalize all solver outputs to `DecisionResult` and experiment outputs to
  `ExperimentResult`.

Acceptance criteria:

- Graph weights normalize to one under enumerated and sampled graph sets.
- Interval intersection handles empty, singleton, and overlapping intervals.
- QP/projection constraints respect alteration bounds.
- Online updates change estimator state over time and preserve deterministic
  behavior under a fixed seed.
- ICML CARE solver returns bounded values and finite CARE success estimates on a toy
  task.

## Milestones

1. Framework skeleton.
   - Package imports work.
   - Core dataclass validation tests pass.
   - A toy method contract test passes.

2. Linear/SRM method ports.
   - NeurIPS 2023, NeurIPS 2024, and ICML 2025 algorithms are callable through
     the common API.
   - Reduced CPU smoke experiments are reproducible.

3. Nonlinear/order/sequential ports.
   - AAAI nonlinear/multivariate, order-based, and IJCAI sequential workflows
     are callable through the common API.
   - Heavy dependencies are isolated behind optional imports.

4. Influence and experiment layer.
   - ICLR influence-power estimator works on controlled SEMs.
   - Config-driven experiment entrypoints write standardized
     `ExperimentResult` artifacts.

5. Documentation and compatibility.
   - Architecture docs explain how to add a new method adapter.
   - Previous paper mappings document what legacy scripts were ported and what
     remains reference-only.

## Verification

Run after Python changes:

```bash
pytest
```

Run after JavaScript changes only:

```bash
npm test
```

Do not run formatters or large generated-output commands unless they are part
of a specific implementation phase.

## Risks And Controls

- Heavy optional dependencies: isolate imports inside the modules that need
  them, provide clear errors, and keep core tests CPU-only.
- Legacy script coupling: move plotting, global constants, and hard-coded paths
  into experiment configs rather than package APIs.
- Method API drift: add contract tests that every adapter must pass.
- Reproducibility drift: route random state through framework seed utilities and
  include config snapshots in `ExperimentResult`.
- Scope creep: keep FCS 2022, NeurIPS 2025, and non-parametric full
  implementation out of this round.
