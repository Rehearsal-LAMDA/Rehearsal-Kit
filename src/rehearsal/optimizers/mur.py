"""Matrix construction and bounded QP solvers for NeurIPS 2025 MUR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from rehearsal.core import AUFTask
from rehearsal.models.time_series import LinearTimeSeriesSRM


@dataclass(frozen=True)
class MURMatrixBundle:
    """Matrices in the MUR aggregation expression."""

    M: np.ndarray
    N: np.ndarray
    H: np.ndarray
    F: np.ndarray
    U: np.ndarray
    C: np.ndarray
    Gamma: np.ndarray
    U_tilde: np.ndarray
    C_tilde: np.ndarray
    Gamma_tilde: np.ndarray
    Xi: np.ndarray
    candidate: tuple[str, ...]
    remaining_horizon: int
    block_order: str
    current_block_start: int
    current_block_stop: int
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MURQPResult:
    """Bounded least-squares output for one MUR candidate."""

    z_sequence: np.ndarray
    current_action: np.ndarray
    objective_value: float
    solver_status: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MURActionSelection:
    """Selected current action and diagnostics across candidate sets."""

    candidate: tuple[str, ...]
    alterations: dict[str, float]
    z_sequence: np.ndarray
    current_action: np.ndarray
    objective_value: float
    estimated_success_probability: float
    cost: float
    solver_status: str
    bundle: MURMatrixBundle
    mean: np.ndarray
    F: np.ndarray
    candidate_diagnostics: tuple[dict[str, Any], ...]
    diagnostics: dict[str, Any] = field(default_factory=dict)


def compute_mur_matrices(
    model: LinearTimeSeriesSRM,
    task: AUFTask,
    candidate: Sequence[str],
    remaining_horizon: int,
) -> MURMatrixBundle:
    """Compute MUR ``M``, ``N``, ``H``, and ``F`` for one candidate set.

    The sequence order follows the paper and legacy solver:
    ``[z_{t+T}, z_{t+T-1}, ..., z_t]``.  The current action is therefore the
    last block of the solved vector.
    """

    candidate = tuple(candidate)
    horizon = int(remaining_horizon)
    if horizon < 0:
        raise ValueError("remaining_horizon must be non-negative.")
    if not candidate:
        raise ValueError("MUR requires a non-empty candidate alteration set.")

    identity = np.eye(model.n_variables)
    E_x = model.selection_matrix(task.observed)
    E_action = model.selection_matrix(candidate)
    E_y = model.selection_matrix(task.outcomes)
    P_future = identity - model.projector(candidate)
    P_now = identity - model.projector(tuple(task.observed) + candidate)

    future_operator = identity - P_future @ model.A
    now_operator = identity - P_now @ model.A
    U, future_u_diag = _solve_with_diagnostics(future_operator, E_action, "future_U")
    C, future_c_diag = _solve_with_diagnostics(future_operator, P_future, "future_C")
    U_tilde, now_u_diag = _solve_with_diagnostics(now_operator, E_action, "current_U_tilde")
    C_tilde, now_c_diag = _solve_with_diagnostics(now_operator, P_now, "current_C_tilde")
    Xi, now_x_diag = _solve_with_diagnostics(now_operator, E_x, "current_Xi")

    Gamma = C @ model.B
    Gamma_tilde = C_tilde @ model.B
    powers = _matrix_powers(Gamma, horizon)
    cumulative = _cumulative_sums(powers)
    S_T = cumulative[horizon]
    normalizer = float(horizon + 1)

    M = (E_y.T @ S_T @ Xi) / normalizer
    N = (E_y.T @ S_T @ Gamma_tilde) / normalizer
    future_h_blocks = [E_y.T @ cumulative[idx] @ U for idx in range(horizon)]
    current_h_block = E_y.T @ cumulative[horizon] @ U_tilde
    H = np.hstack([*future_h_blocks, current_h_block]) / normalizer

    future_f_blocks = [E_y.T @ cumulative[idx] @ C for idx in range(horizon)]
    current_f_block = E_y.T @ cumulative[horizon] @ C_tilde
    F = np.hstack([*future_f_blocks, current_f_block]) / normalizer

    candidate_size = len(candidate)
    current_start = horizon * candidate_size
    current_stop = current_start + candidate_size
    diagnostics = {
        "M_shape": tuple(M.shape),
        "N_shape": tuple(N.shape),
        "H_shape": tuple(H.shape),
        "F_shape": tuple(F.shape),
        "H_rank": int(np.linalg.matrix_rank(H)),
        "remaining_horizon": horizon,
        "block_order": "reverse_chronological_current_last",
        "candidate": candidate,
        "used_pinv": bool(
            future_u_diag["used_pinv"]
            or future_c_diag["used_pinv"]
            or now_u_diag["used_pinv"]
            or now_c_diag["used_pinv"]
            or now_x_diag["used_pinv"]
        ),
        "linear_solve_diagnostics": {
            **future_u_diag,
            **future_c_diag,
            **now_u_diag,
            **now_c_diag,
            **now_x_diag,
        },
    }
    return MURMatrixBundle(
        M=M,
        N=N,
        H=H,
        F=F,
        U=U,
        C=C,
        Gamma=Gamma,
        U_tilde=U_tilde,
        C_tilde=C_tilde,
        Gamma_tilde=Gamma_tilde,
        Xi=Xi,
        candidate=candidate,
        remaining_horizon=horizon,
        block_order="reverse_chronological_current_last",
        current_block_start=current_start,
        current_block_stop=current_stop,
        diagnostics=diagnostics,
    )


def solve_mur_box_qp(
    M: np.ndarray,
    N: np.ndarray,
    H: np.ndarray,
    x_t: Sequence[float],
    v_prev: Sequence[float],
    center: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    rng: np.random.Generator,
    learning_rate: float | None = None,
    max_iters: int = 200,
    tolerance: float = 1e-8,
    num_restarts: int = 4,
) -> MURQPResult:
    """Solve ``min_z ||H z - (center - M x_t - N v_prev)||^2`` with box bounds."""

    M = np.asarray(M, dtype=float)
    N = np.asarray(N, dtype=float)
    H = np.asarray(H, dtype=float)
    x = np.asarray(x_t, dtype=float).reshape(-1)
    previous = np.asarray(v_prev, dtype=float).reshape(-1)
    center_arr = np.asarray(center, dtype=float).reshape(-1)
    lower_block = np.asarray(lower, dtype=float).reshape(-1)
    upper_block = np.asarray(upper, dtype=float).reshape(-1)
    if lower_block.shape != upper_block.shape:
        raise ValueError("lower and upper must have the same shape.")
    if np.any(lower_block > upper_block):
        raise ValueError("lower bounds must not exceed upper bounds.")
    if M.shape[1] != x.size:
        raise ValueError("M and x_t shapes are incompatible.")
    if N.shape[1] != previous.size:
        raise ValueError("N and v_prev shapes are incompatible.")
    if M.shape[0] != center_arr.size or N.shape[0] != center_arr.size or H.shape[0] != center_arr.size:
        raise ValueError("M, N, H, and center outcome dimensions are incompatible.")
    if lower_block.size == 0:
        raise ValueError("MUR box-QP requires at least one alteration variable.")
    if H.shape[1] % lower_block.size != 0:
        raise ValueError("H column count must be a multiple of the candidate size.")

    n_blocks = H.shape[1] // lower_block.size
    lower_seq = np.tile(lower_block, n_blocks)
    upper_seq = np.tile(upper_block, n_blocks)
    b = center_arr - M @ x - N @ previous
    rank = int(np.linalg.matrix_rank(H))
    used_pinv = bool(rank < min(H.shape))
    if H.shape[1] == 0:
        unbounded = np.asarray([], dtype=float)
    else:
        unbounded = np.linalg.pinv(H) @ b
    unbounded = np.asarray(unbounded, dtype=float).reshape(-1)
    unbounded_objective = _least_squares_objective(H, unbounded, b)

    within_bounds = bool(np.all(unbounded >= lower_seq - tolerance) and np.all(unbounded <= upper_seq + tolerance))
    if within_bounds:
        z = np.clip(unbounded, lower_seq, upper_seq)
        status = "unbounded_within_bounds"
        iterations = 0
        used_unbounded = True
    else:
        z, iterations, status = _projected_gradient_box_qp(
            H,
            b,
            lower_seq,
            upper_seq,
            unbounded,
            rng=rng,
            learning_rate=learning_rate,
            max_iters=int(max_iters),
            tolerance=float(tolerance),
            num_restarts=int(num_restarts),
        )
        used_unbounded = False

    objective = _least_squares_objective(H, z, b)
    current_action = z[-lower_block.size :].copy()
    diagnostics = {
        "rank": rank,
        "H_rank": rank,
        "H_shape": tuple(H.shape),
        "b_norm": float(np.linalg.norm(b)),
        "used_pinv": used_pinv,
        "used_unbounded_solution": used_unbounded,
        "unbounded_objective_value": float(unbounded_objective),
        "n_iters": int(iterations),
        "n_blocks": int(n_blocks),
        "block_order": "reverse_chronological_current_last",
    }
    return MURQPResult(
        z_sequence=np.clip(z, lower_seq, upper_seq),
        current_action=np.clip(current_action, lower_block, upper_block),
        objective_value=float(objective),
        solver_status=status,
        diagnostics=diagnostics,
    )


def select_mur_action(
    model: LinearTimeSeriesSRM,
    task: AUFTask,
    candidates: Sequence[Sequence[str]],
    x_t: Sequence[float],
    v_prev: Sequence[float],
    center: Sequence[float],
    *,
    remaining_horizon: int,
    total_horizon: int | None = None,
    rng: np.random.Generator,
    learning_rate: float | None = None,
    max_iters: int = 200,
    tolerance: float = 1e-8,
    num_restarts: int = 4,
    n_probability_samples: int = 0,
) -> MURActionSelection:
    """Solve all candidate MUR QPs and return the selected current action."""

    x = np.asarray(x_t, dtype=float).reshape(-1)
    previous = np.asarray(v_prev, dtype=float).reshape(-1)
    center_arr = np.asarray(center, dtype=float).reshape(-1)
    reweight = 1.0
    if total_horizon is not None:
        reweight = float(remaining_horizon + 1) / float(total_horizon + 1)

    best: dict[str, Any] | None = None
    candidate_diagnostics: list[dict[str, Any]] = []
    for candidate_raw in candidates:
        candidate = tuple(candidate_raw)
        lower, upper = task.alteration_domain.arrays_for(candidate)
        bundle = compute_mur_matrices(model, task, candidate, remaining_horizon)
        M = reweight * bundle.M
        N = reweight * bundle.N
        H = reweight * bundle.H
        F = reweight * bundle.F
        qp = solve_mur_box_qp(
            M,
            N,
            H,
            x,
            previous,
            center_arr,
            lower,
            upper,
            rng=rng,
            learning_rate=learning_rate,
            max_iters=max_iters,
            tolerance=tolerance,
            num_restarts=num_restarts,
        )
        mean = M @ x + N @ previous + H @ qp.z_sequence
        probability = estimate_mur_success_probability(
            task,
            mean,
            F,
            model.covariance,
            n_probability_samples,
            rng=rng,
            objective_value=qp.objective_value,
        )
        alterations = {name: float(value) for name, value in zip(candidate, qp.current_action)}
        cost = float(task.alteration_domain.cost(alterations))
        diag = {
            "candidate": candidate,
            "objective_value": float(qp.objective_value),
            "estimated_success_probability": float(probability),
            "cost": cost,
            "solver_status": qp.solver_status,
            "reweight": float(reweight),
            **bundle.diagnostics,
            **qp.diagnostics,
        }
        candidate_diagnostics.append(diag)
        record = {
            "candidate": candidate,
            "alterations": alterations,
            "z_sequence": qp.z_sequence,
            "current_action": qp.current_action,
            "objective_value": float(qp.objective_value),
            "estimated_success_probability": float(probability),
            "cost": cost,
            "solver_status": qp.solver_status,
            "bundle": bundle,
            "mean": mean,
            "F": F,
            "diagnostics": {**bundle.diagnostics, **qp.diagnostics, "reweight": float(reweight)},
        }
        if _is_better_selection(record, best):
            best = record

    if best is None:
        raise ValueError("No candidate alteration set is available.")

    return MURActionSelection(
        candidate=best["candidate"],
        alterations=best["alterations"],
        z_sequence=best["z_sequence"],
        current_action=best["current_action"],
        objective_value=best["objective_value"],
        estimated_success_probability=best["estimated_success_probability"],
        cost=best["cost"],
        solver_status=best["solver_status"],
        bundle=best["bundle"],
        mean=best["mean"],
        F=best["F"],
        candidate_diagnostics=tuple(candidate_diagnostics),
        diagnostics=best["diagnostics"],
    )


def estimate_mur_success_probability(
    task: AUFTask,
    mean: Sequence[float],
    F: np.ndarray,
    noise_covariance: np.ndarray,
    n_samples: int,
    *,
    rng: np.random.Generator,
    objective_value: float | None = None,
) -> float:
    """Estimate aggregate desired-region probability from fitted noise."""

    mean_arr = np.asarray(mean, dtype=float).reshape(-1)
    samples = int(n_samples)
    if samples <= 0:
        if objective_value is None:
            return 0.0
        return float(np.clip(1.0 / (1.0 + max(float(objective_value), 0.0)), 0.0, 1.0))
    F = np.asarray(F, dtype=float)
    covariance = np.asarray(noise_covariance, dtype=float)
    if F.shape[1] % covariance.shape[0] != 0:
        raise ValueError("F column count must be a multiple of the noise covariance dimension.")
    n_blocks = F.shape[1] // covariance.shape[0]
    noise = rng.multivariate_normal(np.zeros(covariance.shape[0]), covariance, size=(samples, n_blocks))
    stacked_noise = noise.reshape(samples, n_blocks * covariance.shape[0])
    y_samples = mean_arr.reshape(1, -1) + stacked_noise @ F.T
    success = np.asarray(task.desired_region.contains(y_samples), dtype=bool)
    return float(np.clip(np.mean(success), 0.0, 1.0))


def rollout_mur_policy(
    model: LinearTimeSeriesSRM,
    task: AUFTask,
    *,
    variant: str,
    horizon: int,
    x_t: Sequence[float],
    v_prev: Sequence[float],
    center: Sequence[float],
    candidates: Sequence[Sequence[str]],
    rng: np.random.Generator,
    n_samples: int,
    learning_rate: float | None = None,
    max_iters: int = 200,
    tolerance: float = 1e-8,
    num_restarts: int = 4,
) -> dict[str, Any]:
    """Simulate repeated GMuR/FarMuR decisions under the fitted model."""

    if variant not in {"gmur", "farmur"}:
        raise ValueError("variant must be 'gmur' or 'farmur'.")
    total_horizon = int(horizon)
    if total_horizon < 0:
        raise ValueError("horizon must be non-negative.")
    samples = int(n_samples)
    if samples <= 0:
        raise ValueError("n_samples must be positive.")

    E_x = model.selection_matrix(task.observed)
    E_y = model.selection_matrix(task.outcomes)
    initial_x = np.asarray(x_t, dtype=float).reshape(-1)
    initial_prev = np.asarray(v_prev, dtype=float).reshape(-1)
    initial_center = np.asarray(center, dtype=float).reshape(-1)
    aggregate_success: list[bool] = []
    current_success: list[bool] = []

    for _ in range(samples):
        previous = initial_prev.copy()
        x_current = initial_x.copy()
        center_current = initial_center.copy()
        y_values = []
        first_y = None
        for step in range(total_horizon + 1):
            noise = rng.multivariate_normal(np.zeros(model.n_variables), model.covariance)
            if step > 0:
                natural_state = model.solve_next_state(previous, noise)
                x_current = (E_x.T @ natural_state).reshape(-1)

            remaining = 0 if variant == "gmur" else total_horizon - step
            selection = select_mur_action(
                model,
                task,
                candidates,
                x_current,
                previous,
                center_current,
                remaining_horizon=remaining,
                total_horizon=total_horizon if variant == "farmur" else None,
                rng=rng,
                learning_rate=learning_rate,
                max_iters=max_iters,
                tolerance=tolerance,
                num_restarts=num_restarts,
                n_probability_samples=0,
            )
            fixed_values = {
                **{name: float(value) for name, value in zip(task.observed, x_current)},
                **selection.alterations,
            }
            current_state = model.solve_next_state(previous, noise, fixed_values=fixed_values)
            y_current = (E_y.T @ current_state).reshape(-1)
            if first_y is None:
                first_y = y_current.copy()
            y_values.append(y_current)
            if variant == "farmur":
                center_current = center_current - y_current / float(total_horizon + 1)
            previous = current_state

        aggregate_y = np.mean(np.vstack(y_values), axis=0)
        aggregate_success.append(bool(task.desired_region.contains(aggregate_y)))
        current_success.append(bool(task.desired_region.contains(first_y)))

    return {
        "estimated_success_probability": float(np.mean(aggregate_success)),
        "aggregate_success_rate": float(np.mean(aggregate_success)),
        "current_round_success_rate": float(np.mean(current_success)),
        "variant": variant,
        "horizon": total_horizon,
        "n_samples": samples,
    }


def _projected_gradient_box_qp(
    H: np.ndarray,
    b: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    unbounded: np.ndarray,
    *,
    rng: np.random.Generator,
    learning_rate: float | None,
    max_iters: int,
    tolerance: float,
    num_restarts: int,
) -> tuple[np.ndarray, int, str]:
    midpoint = 0.5 * (lower + upper)
    initial_points = [
        np.clip(unbounded, lower, upper),
        midpoint,
        np.clip(np.zeros_like(lower), lower, upper),
        lower.copy(),
        upper.copy(),
    ]
    for _ in range(max(num_restarts, 0)):
        initial_points.append(rng.uniform(lower, upper))

    best = min(initial_points, key=lambda z: _least_squares_objective(H, z, b))
    best_value = _least_squares_objective(H, best, b)
    spectral_norm = float(np.linalg.norm(H, ord=2)) if H.size else 0.0
    base_step = float(learning_rate) if learning_rate is not None else 1.0 / max(2.0 * spectral_norm * spectral_norm, 1e-12)
    total_iterations = 0

    for start in initial_points:
        z = np.asarray(start, dtype=float).copy()
        value = _least_squares_objective(H, z, b)
        local_iterations = 0
        for local_iterations in range(1, max_iters + 1):
            gradient = 2.0 * H.T @ (H @ z - b)
            if float(np.linalg.norm(gradient)) < tolerance:
                break
            step = base_step
            accepted = False
            for _ in range(30):
                candidate = np.clip(z - step * gradient, lower, upper)
                candidate_value = _least_squares_objective(H, candidate, b)
                if candidate_value <= value + 1e-15:
                    accepted = True
                    if np.linalg.norm(candidate - z) < tolerance:
                        z = candidate
                        value = candidate_value
                        break
                    z = candidate
                    value = candidate_value
                    break
                step *= 0.5
            if not accepted:
                break
        total_iterations += int(local_iterations)
        if value < best_value - 1e-15:
            best = z.copy()
            best_value = value

    return np.clip(best, lower, upper), total_iterations, "projected_gradient_box_qp"


def _least_squares_objective(H: np.ndarray, z: np.ndarray, b: np.ndarray) -> float:
    residual = H @ z - b
    return float(residual @ residual)


def _is_better_selection(candidate: Mapping[str, Any], best: Mapping[str, Any] | None) -> bool:
    if best is None:
        return True
    objective = float(candidate["objective_value"])
    best_objective = float(best["objective_value"])
    if objective < best_objective - 1e-12:
        return True
    if objective > best_objective + 1e-12:
        return False
    probability = float(candidate["estimated_success_probability"])
    best_probability = float(best["estimated_success_probability"])
    if probability > best_probability + 1e-12:
        return True
    if probability < best_probability - 1e-12:
        return False
    return float(candidate["cost"]) < float(best["cost"]) - 1e-12


def _matrix_powers(matrix: np.ndarray, max_power: int) -> list[np.ndarray]:
    powers = [np.eye(matrix.shape[0])]
    for _ in range(max_power):
        powers.append(powers[-1] @ matrix)
    return powers


def _cumulative_sums(powers: Sequence[np.ndarray]) -> list[np.ndarray]:
    cumulative = []
    running = np.zeros_like(powers[0])
    for power in powers:
        running = running + power
        cumulative.append(running.copy())
    return cumulative


def _solve_with_diagnostics(matrix: np.ndarray, rhs: np.ndarray, label: str) -> tuple[np.ndarray, dict[str, Any]]:
    rank = int(np.linalg.matrix_rank(matrix))
    full_rank = rank == matrix.shape[0]
    used_pinv = not full_rank
    if full_rank:
        try:
            solution = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError:
            solution = np.linalg.pinv(matrix) @ rhs
            used_pinv = True
    else:
        solution = np.linalg.pinv(matrix) @ rhs
    diagnostics = {
        f"{label}_rank": rank,
        f"{label}_shape": tuple(matrix.shape),
        f"{label}_used_pinv": bool(used_pinv),
        "used_pinv": bool(used_pinv),
    }
    return solution, diagnostics
