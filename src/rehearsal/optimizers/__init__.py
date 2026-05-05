"""Decision-stage rehearsal optimizers."""

from rehearsal.optimizers.care import CAREOptimizationResult, optimize_care_independent_normal
from rehearsal.optimizers.cme import (
    CMEOptimizationResult,
    cme_action_kernel,
    cme_action_objective,
    optimize_action_projected_gradient,
)
from rehearsal.optimizers.grad_rh import (
    GradRhOptimizationResult,
    desired_region_center_and_radius,
    optimize_grad_rh_alterations,
)
from rehearsal.optimizers.mur import (
    MURActionSelection,
    MURMatrixBundle,
    MURQPResult,
    compute_mur_matrices,
    estimate_mur_success_probability,
    rollout_mur_policy,
    select_mur_action,
    solve_mur_box_qp,
)

__all__ = [
    "CAREOptimizationResult",
    "CMEOptimizationResult",
    "MURActionSelection",
    "MURMatrixBundle",
    "MURQPResult",
    "cme_action_kernel",
    "cme_action_objective",
    "compute_mur_matrices",
    "estimate_mur_success_probability",
    "optimize_action_projected_gradient",
    "optimize_care_independent_normal",
    "GradRhOptimizationResult",
    "desired_region_center_and_radius",
    "optimize_grad_rh_alterations",
    "rollout_mur_policy",
    "select_mur_action",
    "solve_mur_box_qp",
]
