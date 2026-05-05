"""Linear additive time-series SRM for NeurIPS 2025 MUR.

The model represents

``V_t = A V_t + B V_{t-1} + eps_t``

with rows indexed by children and columns indexed by parents.  This is
intentionally separate from :mod:`rehearsal.models.linear_gaussian`, whose
static path-effect helpers do not represent lagged influence matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from rehearsal.core import AUFTask
from rehearsal.core.data import coerce_data_matrix
from rehearsal.models.base import StructuralLearningResult

Theta = Mapping[str, Mapping[str, float]]


@dataclass(frozen=True)
class LinearTimeSeriesSRM:
    """Linear time-series structural rehearsal model."""

    variable_order: Sequence[str]
    instantaneous_matrix: Sequence[Sequence[float]]
    lagged_matrix: Sequence[Sequence[float]]
    noise_covariance: Sequence[Sequence[float]]

    def __post_init__(self) -> None:
        order = tuple(self.variable_order)
        if not order:
            raise ValueError("variable_order must contain at least one variable.")
        if len(set(order)) != len(order):
            raise ValueError("variable_order contains duplicates.")
        n_variables = len(order)
        instantaneous = _square_matrix(self.instantaneous_matrix, n_variables, "instantaneous_matrix")
        lagged = _square_matrix(self.lagged_matrix, n_variables, "lagged_matrix")
        covariance = _square_matrix(self.noise_covariance, n_variables, "noise_covariance")
        if not np.allclose(covariance, covariance.T, atol=1e-10):
            raise ValueError("noise_covariance must be symmetric.")
        object.__setattr__(self, "variable_order", order)
        object.__setattr__(self, "instantaneous_matrix", instantaneous)
        object.__setattr__(self, "lagged_matrix", lagged)
        object.__setattr__(self, "noise_covariance", covariance)

    @property
    def A(self) -> np.ndarray:
        """Instantaneous coefficient matrix."""

        return np.asarray(self.instantaneous_matrix, dtype=float)

    @property
    def B(self) -> np.ndarray:
        """Lagged coefficient matrix."""

        return np.asarray(self.lagged_matrix, dtype=float)

    @property
    def covariance(self) -> np.ndarray:
        """Noise covariance matrix."""

        return np.asarray(self.noise_covariance, dtype=float)

    @property
    def n_variables(self) -> int:
        return len(self.variable_order)

    @property
    def variable_index(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(self.variable_order)}

    def selection_matrix(self, variables: Sequence[str]) -> np.ndarray:
        """Return an ``n_variables x len(variables)`` column selector."""

        variables = tuple(variables)
        if len(set(variables)) != len(variables):
            raise ValueError("selection variables contain duplicates.")
        index = self.variable_index
        matrix = np.zeros((self.n_variables, len(variables)), dtype=float)
        for col, name in enumerate(variables):
            try:
                matrix[index[name], col] = 1.0
            except KeyError as exc:
                raise ValueError(f"Unknown variable {name!r} for time-series SRM.") from exc
        return matrix

    def projector(self, variables: Sequence[str]) -> np.ndarray:
        """Return the coordinate projector for ``variables``."""

        selector = self.selection_matrix(tuple(dict.fromkeys(variables)))
        return selector @ selector.T

    def natural_transition_matrix(self) -> np.ndarray:
        """Return ``(I - A)^-1 B`` using a pseudoinverse fallback."""

        return _solve_or_pinv(np.eye(self.n_variables) - self.A, self.B)

    def altered_transition_matrix(self, fixed_variables: Sequence[str]) -> np.ndarray:
        """Return the lagged transition when ``fixed_variables`` are externally set."""

        identity = np.eye(self.n_variables)
        free_projection = identity - self.projector(fixed_variables)
        return _solve_or_pinv(identity - free_projection @ self.A, free_projection @ self.B)

    def spectral_radius_natural(self) -> float:
        """Spectral radius of the natural time-series transition."""

        return _spectral_radius(self.natural_transition_matrix())

    def spectral_radius_altered(self, fixed_variables: Sequence[str]) -> float:
        """Spectral radius after fixing the supplied variables externally."""

        return _spectral_radius(self.altered_transition_matrix(fixed_variables))

    def solve_next_mean(
        self,
        previous_state: Sequence[float],
        *,
        fixed_values: Mapping[str, float] | None = None,
    ) -> np.ndarray:
        """Return ``E[V_t | V_{t-1}]`` under optional observed/action fixes."""

        return self.solve_next_state(previous_state, np.zeros(self.n_variables), fixed_values=fixed_values)

    def solve_next_state(
        self,
        previous_state: Sequence[float],
        noise: Sequence[float],
        *,
        fixed_values: Mapping[str, float] | None = None,
    ) -> np.ndarray:
        """Solve the current state under a noise draw and optional fixed variables."""

        previous = np.asarray(previous_state, dtype=float).reshape(-1)
        epsilon = np.asarray(noise, dtype=float).reshape(-1)
        if previous.shape != (self.n_variables,):
            raise ValueError("previous_state length must match variable_order.")
        if epsilon.shape != (self.n_variables,):
            raise ValueError("noise length must match variable_order.")

        fixed = dict(fixed_values or {})
        identity = np.eye(self.n_variables)
        fixed_vector = np.zeros(self.n_variables, dtype=float)
        if fixed:
            index = self.variable_index
            fixed_names = []
            for name, value in fixed.items():
                if name not in index:
                    raise ValueError(f"Unknown fixed variable {name!r}.")
                fixed_names.append(name)
                fixed_vector[index[name]] = float(value)
            free_projection = identity - self.projector(fixed_names)
        else:
            free_projection = identity
        rhs = fixed_vector + free_projection @ (self.B @ previous + epsilon)
        return _solve_or_pinv(identity - free_projection @ self.A, rhs).reshape(-1)

    def sample_next(
        self,
        previous_state: Sequence[float],
        *,
        rng: np.random.Generator | None = None,
        noise: Sequence[float] | None = None,
        fixed_values: Mapping[str, float] | None = None,
    ) -> np.ndarray:
        """Sample ``V_t`` from the model under optional fixed variables."""

        if noise is None:
            if rng is None:
                raise ValueError("rng is required when noise is not provided.")
            noise_arr = rng.multivariate_normal(np.zeros(self.n_variables), self.covariance)
        else:
            noise_arr = np.asarray(noise, dtype=float).reshape(-1)
        return self.solve_next_state(previous_state, noise_arr, fixed_values=fixed_values)


class LinearTimeSeriesSRMLearner:
    """Fit or construct a :class:`LinearTimeSeriesSRM` for MUR."""

    def __init__(self, *, min_variance: float = 1e-8) -> None:
        self.min_variance = float(min_variance)

    def fit(
        self,
        data: Mapping[str, Sequence[float]] | np.ndarray | None,
        task: AUFTask,
        config: Mapping[str, Any] | None = None,
    ) -> StructuralLearningResult:
        config_dict = dict(config or {})
        min_variance = float(config_dict.get("min_variance", self.min_variance))
        variable_order = _resolve_variable_order(task, config_dict)
        matrix = _optional_training_matrix(data, task, config_dict, variable_order)

        if "mur_A" in config_dict or "mur_B" in config_dict:
            if "mur_A" not in config_dict or "mur_B" not in config_dict:
                raise ValueError("Direct MUR construction requires both fit_config['mur_A'] and fit_config['mur_B'].")
            instantaneous = _square_matrix(config_dict["mur_A"], len(variable_order), "mur_A")
            lagged = _square_matrix(config_dict["mur_B"], len(variable_order), "mur_B")
            source = "direct_matrices"
        elif "mur_instantaneous_theta" in config_dict or "mur_lagged_theta" in config_dict:
            instantaneous = matrix_from_parent_child_theta(
                config_dict.get("mur_instantaneous_theta", {}),
                variable_order,
            )
            lagged = matrix_from_parent_child_theta(
                config_dict.get("mur_lagged_theta", {}),
                variable_order,
            )
            source = "theta_dicts"
        else:
            if matrix is None:
                raise ValueError("Time-series SRM estimation requires training data or explicit MUR matrices.")
            instantaneous, lagged, covariance, fit_diagnostics = _estimate_matrices_from_data(
                matrix,
                variable_order,
                task,
                config_dict,
                min_variance,
            )
            model = LinearTimeSeriesSRM(variable_order, instantaneous, lagged, covariance)
            diagnostics = {
                "learner": type(self).__name__,
                "source": "time_ordered_least_squares",
                "n_samples": int(matrix.shape[0]),
                "n_variables": len(variable_order),
                "min_variance": min_variance,
                **fit_diagnostics,
            }
            return StructuralLearningResult(model=model, diagnostics=diagnostics)

        if "mur_noise_covariance" in config_dict:
            covariance = _square_matrix(config_dict["mur_noise_covariance"], len(variable_order), "mur_noise_covariance")
        elif matrix is not None and matrix.shape[0] >= 2:
            covariance = _residual_covariance(matrix, instantaneous, lagged, min_variance)
        else:
            covariance = min_variance * np.eye(len(variable_order))

        model = LinearTimeSeriesSRM(variable_order, instantaneous, lagged, covariance)
        diagnostics = {
            "learner": type(self).__name__,
            "source": source,
            "n_samples": int(0 if matrix is None else matrix.shape[0]),
            "n_variables": len(variable_order),
            "min_variance": min_variance,
        }
        return StructuralLearningResult(model=model, diagnostics=diagnostics)


def matrix_from_parent_child_theta(theta: Theta | Mapping[str, Any], variable_order: Sequence[str]) -> np.ndarray:
    """Build a coefficient matrix from ``theta[parent][child]``."""

    order = tuple(variable_order)
    index = {name: idx for idx, name in enumerate(order)}
    matrix = np.zeros((len(order), len(order)), dtype=float)
    for parent, children in dict(theta or {}).items():
        if parent not in index:
            raise ValueError(f"Theta parent {parent!r} is not in variable_order.")
        if not isinstance(children, Mapping):
            raise ValueError("Theta values must be parent-to-child mappings.")
        for child, value in children.items():
            if child not in index:
                raise ValueError(f"Theta child {child!r} is not in variable_order.")
            matrix[index[child], index[parent]] = float(value)
    return matrix


def _estimate_matrices_from_data(
    matrix: np.ndarray,
    columns: Sequence[str],
    task: AUFTask,
    config: Mapping[str, Any],
    min_variance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if matrix.shape[0] < 2:
        raise ValueError("At least two time-ordered samples are required to fit a time-series SRM.")

    columns = tuple(columns)
    column_index = {name: idx for idx, name in enumerate(columns)}
    lagged_parent_map = dict(task.metadata.get("mur_lagged_parents", {}))
    lagged_parent_map.update(dict(config.get("mur_lagged_parents", {})))
    instantaneous = np.zeros((len(columns), len(columns)), dtype=float)
    lagged = np.zeros_like(instantaneous)
    residuals = np.zeros((matrix.shape[0] - 1, len(columns)), dtype=float)
    current = matrix[1:, :]
    previous = matrix[:-1, :]
    rank_deficient_children: list[str] = []

    for child in columns:
        child_idx = column_index[child]
        instantaneous_parents = tuple(task.parents.get(child, ()))
        lagged_parents = tuple(lagged_parent_map.get(child, ()))
        _validate_parent_names(child, instantaneous_parents, column_index, "instantaneous")
        _validate_parent_names(child, lagged_parents, column_index, "lagged")

        design_blocks = []
        if instantaneous_parents:
            design_blocks.append(current[:, [column_index[name] for name in instantaneous_parents]])
        if lagged_parents:
            design_blocks.append(previous[:, [column_index[name] for name in lagged_parents]])
        y = current[:, child_idx]
        if design_blocks:
            design = np.column_stack(design_blocks)
            rank = int(np.linalg.matrix_rank(design))
            if rank < design.shape[1]:
                rank_deficient_children.append(child)
            beta = np.linalg.pinv(design) @ y
            prediction = design @ beta
            cursor = 0
            for parent in instantaneous_parents:
                instantaneous[child_idx, column_index[parent]] = float(beta[cursor])
                cursor += 1
            for parent in lagged_parents:
                lagged[child_idx, column_index[parent]] = float(beta[cursor])
                cursor += 1
        else:
            prediction = np.zeros_like(y)
        residuals[:, child_idx] = y - prediction

    covariance = (residuals.T @ residuals) / max(residuals.shape[0], 1)
    covariance = covariance + min_variance * np.eye(len(columns))
    diagnostics = {
        "rank_deficient_children": tuple(rank_deficient_children),
        "n_regression_rows": int(matrix.shape[0] - 1),
        "n_lagged_parent_entries": int(sum(len(tuple(value)) for value in lagged_parent_map.values())),
    }
    return instantaneous, lagged, covariance, diagnostics


def _resolve_variable_order(task: AUFTask, config: Mapping[str, Any]) -> tuple[str, ...]:
    order = config.get("variable_order", task.variable_order)
    if order is None:
        order = task.all_variables()
    order_tuple = tuple(order)
    missing = set(task.observed) | set(task.alterable) | set(task.outcomes)
    missing -= set(order_tuple)
    if missing:
        raise ValueError(f"task variables are missing from variable_order: {sorted(missing)}.")
    return order_tuple


def _optional_training_matrix(
    data: Mapping[str, Sequence[float]] | np.ndarray | None,
    task: AUFTask,
    config: Mapping[str, Any],
    variable_order: Sequence[str],
) -> np.ndarray | None:
    if data is None:
        return None
    matrix, columns = coerce_data_matrix(data, task, config, columns=variable_order)
    if tuple(columns) != tuple(variable_order):
        raise ValueError("Training data columns must match variable_order.")
    return matrix


def _residual_covariance(
    matrix: np.ndarray,
    instantaneous: np.ndarray,
    lagged: np.ndarray,
    min_variance: float,
) -> np.ndarray:
    current = matrix[1:, :]
    previous = matrix[:-1, :]
    residuals = current - current @ instantaneous.T - previous @ lagged.T
    covariance = (residuals.T @ residuals) / max(residuals.shape[0], 1)
    return covariance + min_variance * np.eye(matrix.shape[1])


def _validate_parent_names(
    child: str,
    parents: Sequence[str],
    column_index: Mapping[str, int],
    kind: str,
) -> None:
    missing = [name for name in parents if name not in column_index]
    if missing:
        raise ValueError(f"{kind} parents for {child!r} are missing from data: {missing}.")


def _square_matrix(value: Sequence[Sequence[float]] | np.ndarray, n_variables: int, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (n_variables, n_variables):
        raise ValueError(f"{name} must have shape ({n_variables}, {n_variables}).")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite.")
    return matrix


def _solve_or_pinv(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    if np.linalg.matrix_rank(matrix) == matrix.shape[0]:
        try:
            return np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError:
            pass
    return np.linalg.pinv(matrix) @ rhs


def _spectral_radius(matrix: np.ndarray) -> float:
    if matrix.size == 0:
        return 0.0
    eigenvalues = np.linalg.eigvals(np.asarray(matrix, dtype=float))
    return float(np.max(np.abs(eigenvalues)))
