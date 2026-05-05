Quickstart
==========

Installed Demo
--------------

The package ships a self-contained smoke demo that does not require the
repository's ``examples/`` directory:

.. code-block:: bash

   rehearsal-demo \
     --seed 3 \
     --n-samples 40 \
     --eval-samples 6 \
     --max-iters 5 \
     --output outputs/care_demo_from_package.json \
     --compact

Python API
----------

The same demo can be imported and run from Python:

.. code-block:: python

   from rehearsal.experiments.demo import run_demo

   result = run_demo(seed=3, n_samples=40, eval_samples=6, max_iters=5)
   print(result["name"], result["method"], result["n_runs"])
   print(result["runs"][0]["evaluation"])

Seeded Experiment Runner
------------------------

Run a Bermuda example through the generic seeded runner:

.. code-block:: bash

   python -m rehearsal.experiments.run examples/care/care_bermuda_example.py \
     --method care \
     --seeds 3 \
     --params n_data=2000 \
     --eval-samples 1000 \
     --output outputs/care_bermuda_seed3.json \
     --compact

The output always contains ``runs`` and ``summary``. With one seed, ``summary``
still contains ``mean``, ``std``, ``min``, and ``max`` values.
