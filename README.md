# Rehearsal

The FCS22 paper, proposed by Professor Zhi-Hua Zhou from Nanjing University,
introduced rehearsal learning as an AUF decision task over observational
structural data: learn influence relations from context, then select feasible
alterations so future outcomes land in a desired region rather than an
undesired future; see the reference PDF
[here](https://www.lamda.nju.edu.cn/publication/fcs22_rehearsal.pdf).

The `rehearsal` package provides a unified interface for methods migrated from
`previous_works/`: shared task contracts, structural-model interfaces, method
adapters, optimizers, metrics, influence measures, datasets, and seeded
experiment runners for comparing rehearsal methods under one CLI shape.

Historical code remains in `previous_works/` as read-only reference material.
See `ExecPlan.md` for the staged porting plan.

## Implemented Method Provenance

This table covers the currently registered `rehearsal-run --method` adapters
and standalone measure demos. Values of the form `--method ...` are stable
method-registry names; InP is a measure API with standalone example CLIs rather
than a `RehearsalMethod` adapter. Unpublished methods are listed as `arXiv
2026`.

| Registry / entry point | Implementation | Year / venue | Paper | Example config |
| --- | --- | --- | --- | --- |
| `qwz23` | `QWZ23Rehearsal` | 2023 NeurIPS | Rehearsal Learning for Avoiding Undesired Future | `examples/qwz23/bermuda_example.py` |
| `micns` | `MICNSRehearsal` | 2024 NeurIPS | Avoiding Undesired Future with Minimal Cost in Non-Stationary Environments | `examples/micns/bermuda_example.py` |
| `grad-rh` | `GradRhRehearsal` | 2025 AAAI | Gradient-Based Nonlinear Rehearsal Learning with Multivariate Alterations | `examples/grad_rh/bermuda_example.py` |
| `care` | `ICML2025CARERehearsal` | 2025 ICML | Enabling Optimal Decisions in Rehearsal Learning under CARE Condition | `examples/care/care_bermuda_example.py` |
| `msr` | `MSRRehearsal` | 2025 IJCAI | Avoiding Undesired Future with Sequential Decisions | `examples/msr/bermuda_example.py` |
| `cme-rh` | `CMERehearsal` | arXiv 2026 | Non-Parametric Rehearsal Learning via Conditional Mean Embeddings | `examples/cme/cme_bermuda_example.py` |
| `olem-rh` | `OLEMRhRehearsal` | arXiv 2026 | Order-Based Rehearsal Learning | `examples/olem_rh/bermuda_example.py` |
| InP measure demos | `compute_inp`, `compute_inp_for_variables` | 2026 ICLR | On Measuring Influence in Avoiding Undesired Future | `examples/inp/bermuda_inp_example.py` |

## Bermuda Example

Bermuda is a standardized continuous SEM for an AUF task: `Light`, `Temp`, and
`Sal` are observed context variables; `DIC`, `TA`, `Omega`, `Chla`, and
`Nutrients_PC1` are bounded alterable variables; `NEC` is the outcome; and
success means placing `NEC` in the desired interval with high probability under
the true simulator.

Each seeded method example samples a Bermuda context, learns a structural model
from `n_data=2000` observational samples, selects bounded alterations, and
evaluates the selected action with `eval_samples=1000`. The observed Bermuda
context is sampled inside each seeded experiment config. Do not pass observed
variables through `--params`; use `--seeds` to make sampled observations
reproducible.

### Rehearsal Learning Results

The tracked method outputs are single-seed Bermuda references with seed `3`.
The table reports the true AUF probability measured by each example's true
simulator.

| Method | Venue | Output | True AUF probability |
| --- | --- | --- | ---: |
| `qwz23` | 2023 NeurIPS | `outputs/qwz23_bermuda_seed3.json` | 0.833 |
| `micns` | 2024 NeurIPS | `outputs/micns_bermuda_seed3.json` | 0.837 |
| `grad-rh` | 2025 AAAI | `outputs/grad_rh_bermuda_seed3.json` | 0.827 |
| `care` | 2025 ICML | `outputs/care_bermuda_seed3.json` | 0.840 |
| `msr` | 2025 IJCAI | `outputs/msr_bermuda_seed3.json` | 0.830 |
| `cme-rh` | arXiv 2026 | `outputs/cme_bermuda_seed3.json` | 0.831 |
| `olem-rh` | arXiv 2026 | `outputs/olem_rh_bermuda_seed3.json` | 0.808 |

### Order Learning And Variable InP

The Bermuda measure example learns a variable order from continuous
observational data, discretizes every variable into `3` bins for the recursive
MEP / InP calculation, and evaluates InP for every alterable variable that lies
on the active path from the chosen `start_node` to `NEC`.

The tracked `outputs/inp_bermuda_measures.json` run uses `n_data=2000`,
`num_samples=1500`, `n_bins=3`, and `start_node=TA`. The learned order is
`Temp -> pHsw -> TA -> DIC -> CO2 -> Sal -> Light -> Omega -> NEC ->
Nutrients_PC1 -> Chla`; under that order, the evaluated variables have these
values:

| Variable | InP | MEP-alter | MEP-observe |
| --- | ---: | ---: | ---: |
| `DIC` | 0.469 | 0.650 | 0.181 |
| `TA` | 0.322 | 0.982 | 0.659 |
| `Omega` | 0.186 | 0.190 | 0.004 |

The partial-order demo selects a compatible Bermuda order with best MEP `0.980`
for start node `TA`; under that order, `DIC`, `TA`, and `Omega` have InP values
`0.473`, `0.346`, and `0.174`, respectively.

## Project Structure

- `src/rehearsal/`: installable Python package. It contains the shared task
  contracts, model interfaces, method adapters, measure APIs, optimizers,
  metrics, datasets, and experiment runners.
- `tests/`: focused regression and contract tests for the package.
- `examples/`: runnable method and measure examples used by the README commands.
- `docs/`: architecture notes and method-porting guidance.
- `previous_works/`: read-only historical code, paper sources, data, and PDFs
  used as reference material while migrating methods into the unified package.
- `ExecPlan.md`, `OnGoing.md`, `code_idea.md`: project planning and remaining
  porting tasks.
- `outputs/`: example experiment result JSON files generated by README-style
  commands and tracked as reproducible reference outputs.

Files intentionally kept out of Git include Python bytecode, pytest/cache
directories, OS metadata such as `.DS_Store`, local agent/editor state,
packaging/build artifacts, LaTeX auxiliary files, and local runtime artifacts.

The current implementation includes the measure APIs and method adapters listed
in the provenance table above while keeping shared model, task, optimizer, and
experiment-runner interfaces reusable across papers.

## Package Layout

- `rehearsal.core`: AUF task objects, desired regions, alteration domains,
  result contracts, and validation.
- `rehearsal.models`: structural-learning models. `LinearGaussianSRM` and
  `LinearGaussianSRMLearner` are shared components for CARE and future
  NeurIPS 2023 / NeurIPS 2024 adapters.
- `rehearsal.optimizers`: rehearsal-stage optimizers over fitted structural
  models.
- `rehearsal.methods`: thin method adapters exposing `fit`, `suggest`, and
  `evaluate`.
- `rehearsal.measures`: InP, MEP, ACE, CACE, and partial-order utilities for
  evaluating influence properties of fitted rehearsal models.
- `rehearsal.datasets`: reusable dataset and SEM factories, including generic
  Bermuda and Manage dataset modules shared across methods.
- `rehearsal.experiments`: command-line runners for seeded experiment batches.

## Installation And Packaged Demo

After the package is published, install the base package with:

```bash
python -m pip install rehearsal
```

The base install includes the NumPy-backed core APIs, method adapters,
experiment runner, and an installed toy demo. Bermuda `.mat` loading needs
SciPy, which is exposed as an optional extra:

```bash
python -m pip install "rehearsal[bermuda]"
```

QWZ23 uses sampled multivariate maximization. It runs with a NumPy random-search
fallback, and installs SciPy for the preferred MILP optimizer via:

```bash
python -m pip install "rehearsal[qwz23]"
```

For publishing from a local checkout, install the release tools extra:

```bash
python -m pip install "rehearsal[publish]"
```

The package ships a self-contained smoke demo that does not require the
repository's `examples/` directory:

```bash
rehearsal-demo \
  --seed 3 \
  --n-samples 40 \
  --eval-samples 6 \
  --max-iters 5 \
  --output outputs/care_demo_from_package.json \
  --compact
```

The same demo can be imported and run from Python:

```python
from rehearsal.experiments.demo import run_demo

result = run_demo(seed=3, n_samples=40, eval_samples=6, max_iters=5)
print(result["name"], result["method"], result["n_runs"])
print(result["runs"][0]["evaluation"])
```

## Runner Contract

The generic runner has one execution shape: a seeded batch. There is no separate
single-seed output mode. If you want one seed, pass a one-element seed list:

```text
--seeds 3
```

The output always contains `runs` and `summary`. With one seed, `summary` still
contains `mean`, `std`, `min`, and `max`; the standard deviation is `0.0`.

An experiment config should define:

```python
def build_experiment(params, seed):
    # seed is supplied only by rehearsal-run --seeds.
    # Build task, generated training data, and the observed individual here.
    return {
        "name": "my_experiment",
        "task": task,
        "data": data_for_this_seed,
        "observation": observation_for_this_seed,
        "method_params": {"pgd_steps": 60},
        "default_eval_samples": 500,
        "evaluate": evaluate_true_auf,
        "metadata": {"n_samples": n_samples},
    }
```

Do not pass `seed` through `--params`, `--method-params`, or
`method_params` returned by the config. The seed list is the single source of
run randomness. The runner passes each seed to the config and to the method
constructor.

`data` and `observation` are not meant to be global constants. In the provided
demos, `n_samples=100` means: for each seed, generate 100 training samples
inside `build_experiment(params, seed)`, then return that generated dictionary
as `data`. The observed individual is also sampled inside the same seeded
factory.

## CLI Parameters

Use these forms only:

| Argument | Meaning |
| --- | --- |
| `--seeds 3,4,5` | Required. The exact run seeds. `--seeds 3` is a one-seed batch. |
| `--method NAME` | Method registry name. Currently registered: `care`, `cme-rh`, `grad-rh`, `micns`, `msr`, `olem-rh`, `qwz23`. |
| `--params KEY=VALUE` | Experiment config parameters passed to `build_experiment(params, seed)`. |
| `--method-params KEY=VALUE` | Method constructor parameters. Do not put `seed` here. |
| `--fit-params KEY=VALUE` | Extra options passed to `method.fit(...)`; rarely needed. |
| `--eval-samples N` | True AUF Monte Carlo samples used by the config's evaluator. |
| `--output path.json` | Optional JSON output path. |
| `--compact` | Print compact JSON. |

When `--output` is provided, the full JSON payload is written to that file and
the runner prints a short completion line such as
`wrote outputs/cme_bermuda_seed3.json (n_runs=1, method=cme-rh)`.

The removed singular aliases `--param`, `--method-param`, and `--fit-param`
are intentionally rejected.

## Measure And Method CLI Examples

Run these commands from the repository root. They generate the tracked Bermuda
reference outputs under `outputs/`.

InP / ICLR 2026 measure example:

```bash
env PYTHONPATH=src python examples/inp/bermuda_inp_example.py \
  --n-data 2000 \
  --num-samples 1500 \
  --n-bins 3 \
  --start-node TA \
  --output outputs/inp_bermuda_measures.json \
  --quiet
```

QWZ23 / NeurIPS 2023:

```bash
env PYTHONPATH=src python -m rehearsal.experiments.run examples/qwz23/bermuda_example.py \
  --method qwz23 \
  --seeds 3 \
  --params n_data=2000 \
  --eval-samples 1000 \
  --output outputs/qwz23_bermuda_seed3.json \
  --compact
```

MICNS / NeurIPS 2024:

```bash
env PYTHONPATH=src python -m rehearsal.experiments.run examples/micns/bermuda_example.py \
  --method micns \
  --seeds 3 \
  --params n_data=2000 \
  --eval-samples 1000 \
  --output outputs/micns_bermuda_seed3.json \
  --compact
```

Grad-Rh / AAAI 2025:

```bash
env PYTHONPATH=src python -m rehearsal.experiments.run examples/grad_rh/bermuda_example.py \
  --method grad-rh \
  --seeds 3 \
  --params n_data=2000 \
  --eval-samples 1000 \
  --output outputs/grad_rh_bermuda_seed3.json \
  --compact
```

CARE / ICML 2025:

```bash
env PYTHONPATH=src python -m rehearsal.experiments.run examples/care/care_bermuda_example.py \
  --method care \
  --seeds 3 \
  --params n_data=2000 \
  --eval-samples 1000 \
  --output outputs/care_bermuda_seed3.json \
  --compact
```

MSR / IJCAI 2025, registered as `msr`:

```bash
env PYTHONPATH=src python -m rehearsal.experiments.run examples/msr/bermuda_example.py \
  --method msr \
  --seeds 3 \
  --params n_data=2000 \
  --eval-samples 1000 \
  --output outputs/msr_bermuda_seed3.json \
  --compact
```

CME / arXiv 2026:

```bash
env PYTHONPATH=src python -m rehearsal.experiments.run examples/cme/cme_bermuda_example.py \
  --method cme-rh \
  --seeds 3 \
  --params n_data=2000 \
  --eval-samples 1000 \
  --output outputs/cme_bermuda_seed3.json \
  --compact
```

OLEM-Rh / arXiv 2026:

```bash
env PYTHONPATH=src python -m rehearsal.experiments.run examples/olem_rh/bermuda_example.py \
  --method olem-rh \
  --seeds 3 \
  --params n_data=2000 \
  --eval-samples 1000 \
  --output outputs/olem_rh_bermuda_seed3.json \
  --compact
```

## Output Shape

A one-seed run still returns a batch:

```json
{
  "name": "cme_bermuda",
  "method": "cme-rh",
  "seeds": [3],
  "n_runs": 1,
  "runs": [
    {
      "seed": 3,
      "observation": {"Light": 0.01, "Temp": -0.02, "Sal": 0.03},
      "structural_learning": {
        "runtime_seconds": 0.004
      },
      "decision": {
        "alterations": {"DIC": 0.2, "TA": 0.1},
        "estimated_success_probability": 0.8,
        "cost": 0.3,
        "runtime_seconds": 0.001
      },
      "evaluation": {
        "true_auf_success_rate": 0.82,
        "no_action_true_auf_success_rate": 0.12,
        "eval_samples": 1000
      }
    }
  ],
  "summary": {
    "structural_learning.runtime_seconds": {
      "mean": 0.004,
      "std": 0.0,
      "min": 0.004,
      "max": 0.004
    },
    "decision.runtime_seconds": {
      "mean": 0.001,
      "std": 0.0,
      "min": 0.001,
      "max": 0.001
    },
    "evaluation.true_auf_success_rate": {
      "mean": 0.82,
      "std": 0.0,
      "min": 0.82,
      "max": 0.82
    },
    "evaluation.no_action_true_auf_success_rate": {
      "mean": 0.12,
      "std": 0.0,
      "min": 0.12,
      "max": 0.12
    }
  }
}
```

The batch `summary` intentionally includes only true AUF Monte Carlo success
metrics plus structural-learning and decision-stage runtimes. It does not
summarize decision cost, method-internal estimates, or `eval_samples`.

The provided examples report these per-run evaluation fields:

- `true_auf_success_rate`: success rate after the suggested alteration under
  the true data-generating process supplied by the experiment config.
- `no_action_true_auf_success_rate`: success rate for the same observation with
  no alteration under the same true data-generating process.
- `eval_samples`: the Monte Carlo count from `--eval-samples` or the example's
  `default_eval_samples`.
- `structural_learning.runtime_seconds`: per-run wall-clock time spent in
  `method.fit(...)` for that seed's training data.
- `decision.runtime_seconds`: per-run wall-clock time spent in the method's
  `suggest(...)` decision step. It does not include structural fitting or true
  AUF Monte Carlo evaluation time.

## Method Registry

`--method ...` is resolved by `rehearsal.methods.registry`. The registry lets
the runner instantiate methods by a stable CLI name without every experiment
config importing and constructing the adapter itself.

```python
"grad-rh": GradRhRehearsal
"care": ICML2025CARERehearsal
"micns": MICNSRehearsal
"msr": MSRRehearsal
"olem-rh": OLEMRhRehearsal
"qwz23": QWZ23Rehearsal
"cme-rh": CMERehearsal
```

There are no legacy method-name aliases beyond the stable registry names listed
above.

## Collaboration Guidelines

This repository is a research-code migration project. Keep changes small,
reviewable, and aligned with the shared `src/rehearsal/` package interfaces
rather than adding new one-off experiment scripts.

### Commit Prefixes

Use a short, bracketed prefix at the start of every commit subject:

| Prefix | Use for |
| --- | --- |
| `[ENH]` | New features, method adapters, experiment runners, or supported capabilities. |
| `[FIX]` | Bug fixes, numerical corrections, CLI contract fixes, or broken-test repairs. |
| `[DOC]` | README, architecture notes, method-porting notes, comments, or examples that do not change behavior. |
| `[TST]` | New or updated tests, fixtures, smoke checks, or regression coverage. |
| `[REF]` | Refactors that preserve behavior while improving structure or readability. |
| `[EXP]` | Reproducible experiment configs, result JSON files, or benchmark-output updates. |
| `[DATA]` | Dataset loaders, small tracked data fixtures, or metadata changes. |
| `[DEP]` | Dependency, packaging, or environment changes. Production dependencies require prior confirmation. |
| `[CHORE]` | Repository maintenance, formatting-only changes, or cleanup with no user-facing behavior change. |

Commit subjects should be imperative and specific, for example
`[ENH] Add CME Bermuda batch runner` or
`[FIX] Preserve one-seed batch summary shape`.

### Branches And Reviews

- Use branch names such as `enh/cme-runner`, `fix/seed-summary`,
  `doc/collaboration-guidelines`, or `exp/care-bermuda-smoke`.
- Keep each pull request focused on one method, runner contract, dataset, or
  documentation topic.
- For complex features or significant refactors, write or update an ExecPlan
  before implementation and keep the plan current as the work changes.
- Treat `previous_works/` as read-only historical reference material. Port
  behavior into `src/rehearsal/`, add focused tests in `tests/`, and document
  method-specific notes under `docs/`.

### Testing And Verification

- After Python package, example, or test changes, run:

  ```bash
  env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests
  ```

- After modifying JavaScript files, run `npm test`.
- When changing CLI behavior, include or update a regression test in
  `tests/test_experiment_runner.py`.
- When changing numerical methods, prefer deterministic toy tests with fixed
  seeds before adding larger experiment outputs.
- Keep tracked `outputs/` files reproducible from README-style commands and
  avoid committing local cache, temporary, or exploratory artifacts.

### Dependencies And Data

- Keep runtime dependencies minimal. Ask for confirmation before adding any new
  production dependency.
- Prefer optional imports for heavy research dependencies and keep CPU smoke
  tests runnable without historical data downloads.
- Prefer `pnpm` when installing JavaScript dependencies.
- Track only small, necessary data fixtures. Large generated artifacts should
  stay outside Git unless they are explicitly accepted as reproducibility
  references.

## Verification

Run the pytest command listed in `Testing And Verification`.
