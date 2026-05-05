Architecture
============

The package is organized around a small set of stable contracts.

Task Layer
----------

``rehearsal.core.AUFTask`` defines the common AUF problem:

* observed variables ``X``
* alterable variables ``Z``
* outcome variables ``Y``
* desired region ``M y <= d``
* alteration bounds and optional costs
* optional sequential stages

All method adapters consume this task object instead of bespoke script
arguments.

Method Layer
------------

Every method implements:

.. code-block:: python

   fit(data, task, config=None)
   suggest(observation, task)
   evaluate(task, n_samples)

``suggest`` returns a ``DecisionResult`` regardless of whether the method comes
from graph uncertainty, online SRM, nonlinear differentiable optimization,
sequential AUF, order-based rehearsal, or influence-power evaluation.

Model And Experiment Layers
---------------------------

``rehearsal.models`` owns structural learning, while
``rehearsal.optimizers`` owns reusable decision-stage optimizers over fitted
structural models. ``rehearsal.methods`` should stay thin: it wires structural
learning to rehearsal optimization and returns framework result objects.

``rehearsal.experiments.run`` executes Python config files as seeded batches.
Configs expose ``build_experiment(params, seed)`` and return the task, data,
observation, method parameters, evaluator, and metadata for each seed.
