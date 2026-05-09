Experiments
===========

The generic runner executes Python experiment configs as seeded batches. A
config exposes ``build_experiment(params, seed)`` and returns the task, data,
observation, method parameters, evaluator, and metadata for each seed.

Basic Form
----------

.. code-block:: bash

   python -m rehearsal.experiments.run examples/care/care_bermuda_example.py \
     --method care \
     --seeds 3 \
     --params n_data=2000 \
     --eval-samples 1000 \
     --output outputs/care_bermuda_seed3.json \
     --compact

Measure And Method CLI Examples
-------------------------------

Run these commands from the repository root. They generate the tracked Bermuda
reference outputs under ``outputs/``.

InP / ICLR 2026 measure example:

.. code-block:: bash

   env PYTHONPATH=src python examples/inp/bermuda_inp_example.py \
     --n-data 2000 \
     --num-samples 1500 \
     --n-bins 3 \
     --start-node TA \
     --output outputs/inp_bermuda_measures.json \
     --quiet

QWZ23 / NeurIPS 2023:

.. code-block:: bash

   env PYTHONPATH=src python -m rehearsal.experiments.run examples/qwz23/bermuda_example.py \
     --method qwz23 \
     --seeds 3 \
     --params n_data=2000 \
     --eval-samples 1000 \
     --output outputs/qwz23_bermuda_seed3.json \
     --compact

CARE / ICML 2025:

.. code-block:: bash

   env PYTHONPATH=src python -m rehearsal.experiments.run examples/care/care_bermuda_example.py \
     --method care \
     --seeds 3 \
     --params n_data=2000 \
     --eval-samples 1000 \
     --output outputs/care_bermuda_seed3.json \
     --compact

OLEM-Rh / arXiv 2026:

.. code-block:: bash

   env PYTHONPATH=src python -m rehearsal.experiments.run examples/olem_rh/bermuda_example.py \
     --method olem-rh \
     --seeds 3 \
     --params n_data=2000 \
     --eval-samples 1000 \
     --output outputs/olem_rh_bermuda_seed3.json \
     --compact

Rehearsal Learning Results
--------------------------

The tracked method outputs are single-seed Bermuda references with seed ``3``.
The table reports the true AUF probability measured by each example's true
simulator.

.. list-table::
   :header-rows: 1
   :widths: 20 20 45 15

   * - Method
     - Venue
     - Output
     - True AUF probability
   * - ``qwz23``
     - 2023 NeurIPS
     - ``outputs/qwz23_bermuda_seed3.json``
     - 0.833
   * - ``micns``
     - 2024 NeurIPS
     - ``outputs/micns_bermuda_seed3.json``
     - 0.837
   * - ``grad-rh``
     - 2025 AAAI
     - ``outputs/grad_rh_bermuda_seed3.json``
     - 0.827
   * - ``care``
     - 2025 ICML
     - ``outputs/care_bermuda_seed3.json``
     - 0.840
   * - ``mur``
     - 2025 NeurIPS
     - ``outputs/mur_bermuda_seed3.json``
     - 0.840
   * - ``msr``
     - 2025 IJCAI
     - ``outputs/msr_bermuda_seed3.json``
     - 0.830
   * - ``cme-rh``
     - arXiv 2026
     - ``outputs/cme_bermuda_seed3.json``
     - 0.831
   * - ``olem-rh``
     - arXiv 2026
     - ``outputs/olem_rh_bermuda_seed3.json``
     - 0.808

Please note that these methods are evaluated under a unified Bermuda scenario
primarily for providing runnable side-by-side implementation references.
However, they do not share identical original settings or input requirements,
as they were proposed to address different variants of the AUF decision
problem. Therefore, these results are intended for functional demonstration
rather than a direct performance comparison. For instance, ``olem-rh`` utilizes
only observational data as input, whereas other methods may rely on additional
structural information.

Runner Contract
---------------

The runner has one execution shape: a seeded batch. If you want one seed, pass
a one-element seed list:

.. code-block:: text

   --seeds 3

The output always contains ``runs`` and ``summary``. With one seed,
``summary`` still contains ``mean``, ``std``, ``min``, and ``max``.

Common Arguments
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Argument
     - Meaning
   * - ``--seeds 3,4,5``
     - Required run seeds.
   * - ``--method NAME``
     - Method registry name, such as ``care`` or ``cme-rh``.
   * - ``--params KEY=VALUE``
     - Parameters passed to ``build_experiment(params, seed)``.
   * - ``--method-params KEY=VALUE``
     - Parameters passed to the method constructor.
   * - ``--eval-samples N``
     - Monte Carlo samples used by the config evaluator.
   * - ``--output path.json``
     - Optional JSON output path.
   * - ``--compact``
     - Print compact JSON.
