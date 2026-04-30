"""Order-based structural learning utilities for OLEM-Rh rehearsal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from math import log, pi, e, ceil
from scipy.special import digamma

from rehearsal.core import AUFTask
from rehearsal.core.data import coerce_data_matrix
from rehearsal.models.base import StructuralLearningResult
from rehearsal.models.nonlinear import NonlinearSRMLearner


@dataclass(frozen=True)
class OrderLearningResult:
    order: tuple[str, ...]
    parents: Mapping[str, tuple[str, ...]]
    scores: Mapping[str, float]
    diagnostics: Mapping[str, Any]


class OrderBasedStructuralLearner:
    """Learn an OLEM-style variable order and fit conditional samplers on it."""

    def __init__(
        self,
        *,
        max_parents: int | None = None,
        entropy_estimator: str = "gaussian",
        predictor_type: str = "linear",
        feature_degree: int = 1,
        parent_correlation_threshold: float = 1e-6,
    ) -> None:
        self.max_parents = max_parents
        self.entropy_estimator = str(entropy_estimator)
        self.predictor_type = str(predictor_type)
        self.feature_degree = int(feature_degree)
        self.parent_correlation_threshold = float(parent_correlation_threshold)

    def learn_order(
        self,
        data: Mapping[str, Sequence[float]] | np.ndarray,
        task: AUFTask,
        config: Mapping[str, Any] | None = None,
    ) -> OrderLearningResult:
        config_dict = dict(config or {})
        matrix, columns = coerce_data_matrix(data, task, config_dict)
        max_parents = config_dict.get("max_parents", self.max_parents)
        if max_parents is not None:
            max_parents = int(max_parents)
        entropy_estimator = str(config_dict.get("entropy_estimator", self.entropy_estimator))
        order_indices, scores = learn_olem_order_indices(matrix, entropy_estimator=entropy_estimator)
        order = tuple(columns[idx] for idx in order_indices)
        parents = parents_from_order(
            matrix,
            columns,
            order,
            max_parents=max_parents,
            correlation_threshold=float(
                config_dict.get("parent_correlation_threshold", self.parent_correlation_threshold)
            ),
        )
        return OrderLearningResult(
            order=order,
            parents=parents,
            scores={columns[idx]: float(value) for idx, value in scores.items()},
            diagnostics={
                "learner": type(self).__name__,
                "entropy_estimator": entropy_estimator,
                "max_parents": max_parents,
                "n_samples": int(matrix.shape[0]),
            },
        )

    def fit(
        self,
        data: Mapping[str, Sequence[float]] | np.ndarray,
        task: AUFTask,
        config: Mapping[str, Any] | None = None,
    ) -> StructuralLearningResult:
        config_dict = dict(config or {})
        order_result = self.learn_order(data, task, config_dict)
        learned_task = _task_with_learned_order(task, order_result.order, order_result.parents)
        learner = NonlinearSRMLearner(
            predictor_type=str(config_dict.get("predictor_type", self.predictor_type)),
            feature_degree=int(config_dict.get("feature_degree", self.feature_degree)),
            ridge=float(config_dict.get("ridge", 1e-6)),
            min_variance=float(config_dict.get("min_variance", 1e-8)),
        )
        result = learner.fit(data, learned_task, config_dict)
        diagnostics = {
            **result.diagnostics,
            **order_result.diagnostics,
            "order": order_result.order,
            "learned_parents": {name: tuple(value) for name, value in order_result.parents.items()},
            "order_scores": dict(order_result.scores),
        }
        return StructuralLearningResult(model=result.model, diagnostics=diagnostics)


def learn_order_based_structure(
    data: Mapping[str, Sequence[float]] | np.ndarray,
    task: AUFTask,
    config: Mapping[str, Any] | None = None,
) -> StructuralLearningResult:
    return OrderBasedStructuralLearner().fit(data, task, config)


def learn_olem_order_indices(matrix: np.ndarray, *, entropy_estimator: str = "gaussian") -> tuple[tuple[int, ...], dict[int, float]]:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("matrix must be two-dimensional.")
    remaining = list(range(values.shape[1]))
    sinks: list[int] = []
    scores: dict[int, float] = {}
    while remaining:
        candidate_scores: dict[int, float] = {}
        for idx in remaining:
            others = [j for j in remaining if j != idx]
            if entropy_estimator in ["gaussian"]:
                score = _gaussian_conditional_entropy(values[:, idx], values[:, others])
            else:
                score = -_entropy_estimate_kozachenko(values[:, others])
            candidate_scores[idx] = score
        sink = max(candidate_scores, key=lambda i: candidate_scores[i])
        scores[sink] = candidate_scores[sink]
        sinks.append(sink)
        remaining.remove(sink)
    return tuple(reversed(sinks)), scores


def parents_from_order(
    matrix: np.ndarray,
    columns: Sequence[str],
    order: Sequence[str],
    *,
    max_parents: int | None = None,
    correlation_threshold: float = 1e-6,
) -> dict[str, tuple[str, ...]]:
    values = np.asarray(matrix, dtype=float)
    column_index = {name: idx for idx, name in enumerate(columns)}
    parents: dict[str, tuple[str, ...]] = {}
    for pos, child in enumerate(order):
        previous = [name for name in order[:pos] if name in column_index]
        if max_parents is not None:
            child_values = values[:, column_index[child]]
            ranked = sorted(
                previous,
                key=lambda name: abs(_safe_corr(values[:, column_index[name]], child_values)),
                reverse=True,
            )
            previous = [
                name
                for name in ranked
                if abs(_safe_corr(values[:, column_index[name]], child_values)) >= correlation_threshold
            ][: int(max_parents)]
        parents[child] = tuple(previous)
    return parents


def full_dag_from_order(order: Sequence[str]) -> dict[str, tuple[str, ...]]:
    return {child: tuple(order[:idx]) for idx, child in enumerate(order)}


def _task_with_learned_order(
    task: AUFTask,
    order: Sequence[str],
    parents: Mapping[str, Sequence[str]],
) -> AUFTask:
    from dataclasses import replace

    return replace(task, variable_order=tuple(order), parents={name: tuple(value) for name, value in parents.items()})


def _entropy_estimate_kozachenko(data: np.ndarray) -> float:
    n_samples = data.shape[0]
    dim = data.shape[1]
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    elif isinstance(data, list):
        data = torch.tensor(data)
    k = 1
    log_V_d = (dim/2) * log(pi) - torch.lgamma(torch.tensor(1+dim/2)) # V_d = pi**(d/2) / torch.gamma(1+d/2)
    ret = 0
    for i in range(n_samples):
        distances = torch.norm(data - data[i], dim=1)
        k_nearest_dist, t = torch.topk(distances, k=k+1, largest=False)
        w = torch.Tensor(k)
        for j in range(k):
            rho = k_nearest_dist[j+1]
            if rho < 1e-3:
                rho = 1e-3
            log_xi = log_V_d + log(n_samples-1) + dim*log(rho) - digamma(j+1)
            w[j] = 1/k
            ret += w[j] * log_xi
    ret /= n_samples
    return ret


def _gaussian_conditional_entropy(y: np.ndarray, x: np.ndarray) -> float:
    yv = np.asarray(y, dtype=float).reshape(-1, 1)
    xv = np.asarray(x, dtype=float)
    design = np.column_stack([np.ones(xv.shape[0]), xv])
    beta = np.linalg.pinv(design.T @ design) @ design.T @ yv
    residual = (yv - design @ beta).reshape(-1)
    var = max(float(np.var(residual)), 1e-12)
    return 0.5 * log(2.0 * pi * e * var)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=float).reshape(-1)
    bv = np.asarray(b, dtype=float).reshape(-1)
    if np.std(av) <= 1e-12 or np.std(bv) <= 1e-12:
        return 0.0
    return float(np.corrcoef(av, bv)[0, 1])
