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
