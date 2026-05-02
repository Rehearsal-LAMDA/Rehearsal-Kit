"""Measure APIs for rehearsal models.

Measures evaluate properties of fitted rehearsal models (such as how much
altering a variable can shift the probability of a desired outcome).
They are deliberately separate from ``rehearsal.metrics`` (which computes
scalar quality numbers for decisions) and from ``rehearsal.methods`` (which
selects rehearsal actions).

The two measures provided in this module are:

- ``compute_inp`` / ``compute_inp_for_variables``: the Influence Power
  measure. INP requires a rehearsal model, a total ordering, and a
  discretized view of every variable along the order.
- ``compute_ace`` / ``compute_cace``: Average Causal Effect on the desired
  outcome probability, in unconditional and conditional contexts.

For partial orders, ``enumerate_linear_extensions`` and
``select_best_total_order_by_mep`` pick the linear extension that maximises
the Maximum Expected Probability (MEP) at the chosen start node.
"""

from rehearsal.measures.ace import ACEResult, compute_ace, compute_cace
from rehearsal.measures.discretizer import UniformBinDiscretizer
from rehearsal.measures.inp import (
    INPResult,
    compute_inp,
    compute_inp_for_variables,
    desired_bins_from_region,
)
from rehearsal.measures.mep import MEPCalculator
from rehearsal.measures.order import (
    OrderSelectionResult,
    enumerate_linear_extensions,
    select_best_total_order_by_mep,
)

__all__ = [
    "ACEResult",
    "INPResult",
    "MEPCalculator",
    "OrderSelectionResult",
    "UniformBinDiscretizer",
    "compute_ace",
    "compute_cace",
    "compute_inp",
    "compute_inp_for_variables",
    "desired_bins_from_region",
    "enumerate_linear_extensions",
    "select_best_total_order_by_mep",
]
