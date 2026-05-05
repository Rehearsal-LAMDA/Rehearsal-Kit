API Reference
=============

The package is organized into a small set of public modules:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Module
     - Purpose
   * - ``rehearsal.core``
     - AUF task objects, desired regions, alteration domains, result contracts,
       and validation.
   * - ``rehearsal.models``
     - Structural-learning models and shared model interfaces.
   * - ``rehearsal.optimizers``
     - Decision-stage optimizers over fitted structural models.
   * - ``rehearsal.methods``
     - Thin method adapters exposing ``fit``, ``suggest``, and ``evaluate``.
   * - ``rehearsal.measures``
     - InP, MEP, ACE, CACE, and partial-order influence utilities.
   * - ``rehearsal.datasets``
     - Dataset and SEM factories, including Bermuda and Manage examples.
   * - ``rehearsal.experiments``
     - Command-line runners for seeded experiment batches.

The stable command-line method names are documented in :doc:`methods`.
