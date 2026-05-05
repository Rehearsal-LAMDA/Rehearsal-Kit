Contributing
============

This repository is a research-code migration project. Keep changes small,
reviewable, and aligned with the shared ``src/rehearsal/`` interfaces instead
of adding one-off experiment scripts.

Testing
-------

After Python package, example, or test changes, run:

.. code-block:: bash

   env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests

When changing CLI behavior, include or update a regression test in
``tests/test_experiment_runner.py``. When changing numerical methods, prefer
deterministic toy tests with fixed seeds before adding larger experiment
outputs.

Dependencies
------------

Keep runtime dependencies minimal. Prefer optional imports for heavy research
dependencies and keep CPU smoke tests runnable without historical data
downloads.
