# OnGoing

This document lists the remaining paper-porting work for the Rehearsal package.

The current framework must be treated as fixed. All workers must implement their methods inside the existing package structure and must not redesign or rewrite the framework.

## Global Rules For All Workers

- Strictly follow the existing framework in `src/rehearsal/`.
- Do not modify the framework contract unless a tiny compatibility change is unavoidable.
- Do not create a separate script-only implementation outside the package.
- Do not modify anything under `previous_works/`; those files are read-only references.
- Every method adapter must use the existing `fit(data, task, config=None)`, `suggest(observation, task)`, and `evaluate(task, n_samples)` interface.
- `suggest(...)` must return `rehearsal.core.DecisionResult`.
- Register each new method in `rehearsal.methods.registry` with one stable method name.
- Metric-only work must not be forced into the method registry; expose it through `rehearsal.metrics` and add a reproducible example script instead.
- Add focused tests under `tests/`.
- Add at least one Bermuda experiment config and run it through the README-style command:
  `PYTHONPATH=src python -m rehearsal.experiments.run ... --seeds ... --output ... --compact`.
- For metric-only work, replace the runner command with a Bermuda metric script that writes JSON under `outputs/`.
- Method-runner Bermuda outputs must include `runs` and `summary`, following the current README output contract.
- Metric-only Bermuda outputs must use a clear JSON schema with metric values, ranks when applicable, input configuration, and evaluation sample count.
- Do not add new production dependencies without confirmation.
- Keep heavy or optional dependencies isolated behind optional imports.
- After Python changes, run:
  `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests`

## To wangt: NIPS 2023 qwz23 + NIPS 2024 micns

### Scope

Implement the remaining NIPS 2023 (`qwz23`) and NIPS 2024 (`micns`) methods in the unified package.

Reference code:

- `previous_works/02-NeurIPS 2023/code/`
- `previous_works/03-NeurIPS 2024/code/`

Suggested package targets:

- `src/rehearsal/methods/qwz23.py`
- `src/rehearsal/methods/micns.py`
- shared utilities in `src/rehearsal/models/linear_gaussian.py`, `src/rehearsal/optimizers/`, and `src/rehearsal/metrics/` only when needed
- examples under `examples/`
- tests under `tests/`

Required outcomes:

- Port NIPS 2023 graph uncertainty, SEM learning, interval finding, success-probability bounds, and information-gain intervention selection as `qwz23`.
- Port NIPS 2024 time-varying linear SRM solver and cost-aware candidate rehearsal selection as `micns`.
- Normalize all outputs to `DecisionResult`.
- Add registry names exactly `qwz23` and `micns`.
- Add Bermuda configs that run through the existing experiment runner.
- Produce at least one Bermuda JSON result for each method under `outputs/`.

Minimum Bermuda checks:

```bash
PYTHONPATH=src python -m rehearsal.experiments.run examples/qwz23/bermuda_example.py \
  --method qwz23 \
  --seeds 1 \
  --params n_data=2000 \
  --eval-samples 1000 \
  --output outputs/qwz23_bermuda_seed1.json \
  --compact
```

```bash
PYTHONPATH=src python -m rehearsal.experiments.run examples/micns/bermuda_example.py \
  --method micns \
  --seeds 1 \
  --params n_data=2000 \
  --eval-samples 1000 \
  --output outputs/micns_bermuda_seed1.json \
  --compact
```

## To taoyx: AAAI 2025 grad-rh + unpublished order-based olem

### Scope

Implement AAAI 2025 (`grad-rh`) nonlinear/multivariate rehearsal and the unpublished order-based method (`olem`).

Reference code and papers:

- `previous_works/04-AAAI 2025/code/`
- `previous_works/unpublished/Order-Based Rehearsal Learning.pdf`
- Existing unpublished CME implementation is already present; do not reimplement CME.

Suggested package targets:

- `src/rehearsal/methods/grad_rh.py`
- `src/rehearsal/methods/olem.py`
- `src/rehearsal/models/nonlinear.py`
- `src/rehearsal/models/order.py`
- `src/rehearsal/optimizers/`
- examples under `examples/`
- tests under `tests/`

Required outcomes:

- Port AAAI 2025 nonlinear and multivariate alteration workflow as `grad-rh`.
- Expose linear and flow-style predictors through package model interfaces.
- Implement differentiable alteration optimization with bounds.
- Implement order-based sampler / order-learning interface and AUF optimization objective as `olem`.
- Add registry names exactly `grad-rh` and `olem`.
- Add Bermuda configs that run through the existing experiment runner.
- Produce at least one Bermuda JSON result for each method under `outputs/`.

Minimum Bermuda checks:

```bash
PYTHONPATH=src python -m rehearsal.experiments.run examples/grad_rh/bermuda_example.py \
  --method grad-rh \
  --seeds 1 \
  --params n_data=2000 \
  --eval-samples 1000 \
  --output outputs/grad_rh_bermuda_seed1.json \
  --compact
```

```bash
PYTHONPATH=src python -m rehearsal.experiments.run examples/olem/bermuda_example.py \
  --method olem \
  --seeds 1 \
  --params n_data=2000 \
  --eval-samples 1000 \
  --output outputs/olem_bermuda_seed1.json \
  --compact
```

## To taol: IJCAI 2025 aufsd + ICLR 2026 inp

### Scope

Implement IJCAI 2025 (`aufsd`) sequential-decision AUF and ICLR 2026 influence-power metric (`inp`).

Reference papers/code:

- `previous_works/06-IJCAI 2025/`
- `previous_works/08-ICLR 2026/`

Suggested package targets:

- `src/rehearsal/methods/aufsd.py`
- `src/rehearsal/metrics/inp.py`
- `src/rehearsal/optimizers/`
- examples under `examples/`
- tests under `tests/`

Required outcomes:

- Port IJCAI sequential AUF decision workflow with stage-wise observation updates and retrospective inference as `aufsd`.
- Port ICLR influence-power measurement as a metric API named `inp`.
- Reuse existing `AUFTask`, `DecisionResult`, and sequential-stage metadata for `aufsd`.
- `inp` may consume `AUFTask`/Bermuda task metadata, but it must return influence-power metric records rather than `DecisionResult`.
- Add registry name exactly `aufsd` for the IJCAI method.
- Do not register `inp` as a method and do not call it with `--method inp`.
- Add a Bermuda config for `aufsd` that runs through the existing experiment runner.
- Add a Bermuda influence-power script for `inp` that computes the influence power of every alterable `Z` variable.
- Produce at least one Bermuda JSON result for `aufsd` and one Bermuda influence-power JSON result for `inp` under `outputs/`.

Minimum Bermuda checks:

```bash
PYTHONPATH=src python -m rehearsal.experiments.run examples/aufsd/bermuda_example.py \
  --method aufsd \
  --seeds 1 \
  --params n_data=2000 \
  --eval-samples 1000 \
  --output outputs/aufsd_bermuda_seed1.json \
  --compact
```

Minimum Bermuda influence-power check:

```bash
PYTHONPATH=src python examples/inp/bermuda_influence_power.py \
  --n-data 2000 \
  --eval-samples 1000 \
  --output outputs/inp_bermuda_influence_power.json
```

The `inp` output JSON must contain one entry per Bermuda alterable `Z` variable with at least:

- variable name
- influence power value
- rank
- evaluation sample count

## Completion Criteria

A task is complete only when:

- The method or metric is implemented inside the unified package.
- Method adapters are registered in `rehearsal.methods.registry`; metric-only APIs such as `inp` are not registered as methods.
- The method or metric has focused unit tests.
- A Bermuda method example runs with the README-style command, or a metric-only Bermuda script writes the required JSON output.
- Method output JSON follows the existing batch runner contract; metric-only output JSON follows its documented metric schema.
- The full test suite passes.
- The worker reports implemented files, remaining TODOs, Bermuda command or metric script used, output path, and pytest result.
