Methods
=======

The current method registry exposes these stable ``rehearsal-run --method``
names:

.. list-table::
   :header-rows: 1
   :widths: 15 18 37 30

   * - Registry
     - Venue
     - Paper
     - Setting
   * - ``qwz23``
     - 2023 NeurIPS
     - Rehearsal Learning for Avoiding Undesired Future
     - Interaction-based, explicit graphs, linear
   * - ``micns``
     - 2024 NeurIPS
     - Avoiding Undesired Future with Minimal Cost in Non-Stationary Environments
     - Interaction-free, explicit graphs, linear, non-stationary
   * - ``grad-rh``
     - 2025 AAAI
     - Gradient-Based Nonlinear Rehearsal Learning with Multivariate Alterations
     - Interaction-free, explicit graphs, nonlinear
   * - ``care``
     - 2025 ICML
     - Enabling Optimal Decisions in Rehearsal Learning under CARE Condition
     - Interaction-free, explicit graphs, linear, optimal under CARE condition
   * - ``msr``
     - 2025 IJCAI
     - Avoiding Undesired Future with Sequential Decisions
     - Interaction-free, explicit graphs, multi-stage
   * - ``mur``
     - 2025 NeurIPS
     - Variance-Reduced Long-Term Rehearsal Learning with Quadratic Programming Reformulation
     - Interaction-free, explicit graphs, linear, long-term
   * - ``olem-rh``
     - arXiv 2026
     - Order-Based Rehearsal Learning
     - Interaction-free, no explicit graphs, order-based
   * - ``cme-rh``
     - arXiv 2026
     - Non-Parametric Rehearsal Learning via Conditional Mean Embeddings
     - Interaction-free, no explicit graphs, non-parametric

Standalone measure demos for the 2026 ICLR influence-measure work live under
``examples/inp/``.

These methods do not share exactly the same original setting. The unified
Bermuda outputs are functional side-by-side references, not a direct performance
ranking across papers with different assumptions.
