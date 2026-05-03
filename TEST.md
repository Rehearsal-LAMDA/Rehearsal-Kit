# README Command Verification

Executed from `/home/duwb/Research/Rehearsal` on 2026-05-03.

All README `bash` command-line examples exited with code `0`. The README also
contains non-command `text`, `python`, and `json` snippets; those are not listed
here as executable command examples.

## Method Output Commands

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

## Method True AUF Results

All method outputs use the same seed and evaluation budget:
`--seeds 3`, `--params n_data=2000`, and `--eval-samples 1000`.

| Method | Output | True AUF probability |
| --- | --- | ---: |
| `qwz23` | `outputs/qwz23_bermuda_seed3.json` | 0.1680 |
| `micns` | `outputs/micns_bermuda_seed3.json` | 0.8370 |
| `grad-rh` | `outputs/grad_rh_bermuda_seed3.json` | 0.8270 |
| `care` | `outputs/care_bermuda_seed3.json` | 0.8400 |
| `cme-rh` | `outputs/cme_bermuda_seed3.json` | 0.8310 |
| `olem-rh` | `outputs/olem_rh_bermuda_seed3.json` | 0.8080 |

## INP Bermuda

Command:

```bash
env PYTHONPATH=src python examples/inp/bermuda_inp_example.py \
  --n-data 2000 \
  --num-samples 1500 \
  --n-bins 3 \
  --start-node TA \
  --output outputs/inp_bermuda_measures.json \
  --quiet
```

Demo A total-order INP results:

| Variable | INP |
| --- | ---: |
| `DIC` | 0.4691 |
| `TA` | 0.3225 |
| `Omega` | 0.1863 |

## Pytest

Command:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests
```

Effect:

```text
62 passed in 2.68s
```
