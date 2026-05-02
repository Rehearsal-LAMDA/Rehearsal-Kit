"""Direct-enumeration Maximum Expected Probability (MEP) calculator.

The calculator binds a structural model to a ``UniformBinDiscretizer`` and
recursively compares alter and observe branches along a supplied ordering.

For each pair ``(do_conditions, ob_conditions)``:

- continuous alterations are obtained by mapping each ``do`` bin to its
  bin center and passed to ``model.simulate(alterations=...)``;
- conditional sampling under observations is approximated by drawing
  unconditional samples and rejection-filtering by the observed bin index
  filtering).

Per-context simulations and per-(node, value) bin counts are cached.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from rehearsal.measures.discretizer import UniformBinDiscretizer
from rehearsal.models.nonlinear import NonlinearStructuralModel


_FrozenContext = tuple[tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]]


@dataclass(frozen=True)
class MEPDecisionLog:
    """Per-node decision diagnostics from a single MEP recursion.

    ``alterable`` indicates whether the node was eligible for ``do``; when it
    is ``False`` (a non-alterable variable) the recursion only evaluates the
    observation branch and ``mep_do`` / ``best_do_value`` are ``None``.
    """

    node: str
    do_conditions: dict[str, int]
    ob_conditions: dict[str, int]
    mep_do: float | None
    mep_ob: float
    chosen: str  # "do" or "ob"
    best_do_value: int | None
    alterable: bool = True


class MEPCalculator:
    """Compute MEP and INP via direct enumeration on a discretized model."""

    def __init__(
        self,
        model: NonlinearStructuralModel,
        *,
        ordering: Sequence[str],
        target_node: str,
        desired_bins: Sequence[int],
        discretizer: UniformBinDiscretizer,
        alterable: Sequence[str] | None = None,
        num_samples: int = 1000,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.model = model
        self.ordering: tuple[str, ...] = tuple(ordering)
        if target_node not in self.ordering:
            raise ValueError(f"target_node {target_node!r} is not part of the supplied ordering.")
        self.target_node = str(target_node)
        self.desired_bins: tuple[int, ...] = tuple(int(b) for b in desired_bins)
        if not self.desired_bins:
            raise ValueError("desired_bins must contain at least one bin index.")
        self.discretizer = discretizer
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        self.num_samples = int(num_samples)
        self.rng = rng if rng is not None else np.random.default_rng()

        for name in self.ordering:
            if not discretizer.has(name):
                raise KeyError(
                    f"Discretizer is missing variable {name!r} from the rehearsal ordering."
                )

        # ``alterable=None`` keeps the unrestricted behaviour (any upstream node
        # may be ``do``-ed in the recursion). When provided, the recursion at
        # intermediate nodes only branches on ``do`` for nodes in this set;
        # non-alterable nodes can only be observed.
        if alterable is None:
            self._alterable: frozenset[str] | None = None
        else:
            names = frozenset(str(n) for n in alterable)
            unknown = names - set(self.ordering)
            if unknown:
                raise ValueError(
                    "alterable contains variables not in the ordering: "
                    f"{sorted(unknown)}"
                )
            if self.target_node in names:
                raise ValueError(
                    f"target_node {self.target_node!r} cannot appear in the alterable set."
                )
            self._alterable = names

        self._sample_cache: dict[_FrozenContext, dict[str, np.ndarray]] = {}
        self._prob_cache: dict[tuple[str, _FrozenContext], dict[int, float]] = {}
        self._mep_memo: dict[tuple[_FrozenContext, str | None], float] = {}
        self._best_do_memo: dict[tuple[_FrozenContext, str], int] = {}
        self.decision_log: list[MEPDecisionLog] = []

    # ------------------------------------------------------------------ public API

    def compute_mep_do(
        self,
        start_node: str,
        do_conditions: Mapping[str, int] | None = None,
        ob_conditions: Mapping[str, int] | None = None,
    ) -> tuple[float, int]:
        """Return ``MEP(do(start_node))`` and the bin index that achieves it."""

        do_dict = self._normalise(do_conditions)
        ob_dict = self._normalise(ob_conditions)
        self._require_in_ordering(start_node)
        self._require_disjoint(do_dict, ob_dict)
        if start_node in do_dict or start_node in ob_dict:
            raise ValueError(
                f"start_node {start_node!r} must not appear in do/ob conditions."
            )
        return self._best_do_at(start_node, do_dict, ob_dict)

    def compute_mep_do_all(
        self,
        start_node: str,
        do_conditions: Mapping[str, int] | None = None,
        ob_conditions: Mapping[str, int] | None = None,
    ) -> dict[int, float]:
        """Return ``{bin: MEP given do(start_node=bin)}``."""

        do_dict = self._normalise(do_conditions)
        ob_dict = self._normalise(ob_conditions)
        self._require_in_ordering(start_node)
        self._require_disjoint(do_dict, ob_dict)
        if start_node in do_dict or start_node in ob_dict:
            raise ValueError(
                f"start_node {start_node!r} must not appear in do/ob conditions."
            )
        next_node = self._next_node(start_node)
        out: dict[int, float] = {}
        for value in self.discretizer.get_bins(start_node):
            new_do = dict(do_dict)
            new_do[start_node] = int(value)
            out[int(value)] = self._mep_recursive(new_do, ob_dict, next_node)
        return out

    def compute_mep_ob(
        self,
        start_node: str,
        do_conditions: Mapping[str, int] | None = None,
        ob_conditions: Mapping[str, int] | None = None,
    ) -> float:
        """Return ``MEP(observe(start_node))``: probability-weighted MEP over bins."""

        do_dict = self._normalise(do_conditions)
        ob_dict = self._normalise(ob_conditions)
        self._require_in_ordering(start_node)
        self._require_disjoint(do_dict, ob_dict)
        if start_node in do_dict or start_node in ob_dict:
            raise ValueError(
                f"start_node {start_node!r} must not appear in do/ob conditions."
            )
        return self._mep_ob_at(start_node, do_dict, ob_dict)

    def compute_mep(
        self,
        start_node: str,
        do_conditions: Mapping[str, int] | None = None,
        ob_conditions: Mapping[str, int] | None = None,
    ) -> float:
        """Return ``max(MEP_do, MEP_ob)`` at ``start_node``.

        This is the value used to compare candidate total orders for the
        partial-order extension search.
        """

        do_dict = self._normalise(do_conditions)
        ob_dict = self._normalise(ob_conditions)
        self._require_in_ordering(start_node)
        self._require_disjoint(do_dict, ob_dict)
        return self._mep_recursive(do_dict, ob_dict, start_node)

    def compute_inp(
        self,
        start_node: str,
        do_conditions: Mapping[str, int] | None = None,
        ob_conditions: Mapping[str, int] | None = None,
    ) -> tuple[float, int, float, float]:
        """Return ``(inp, best_do_bin, mep_do, mep_ob)`` for ``start_node``."""

        mep_do, best_value = self.compute_mep_do(start_node, do_conditions, ob_conditions)
        mep_ob = self.compute_mep_ob(start_node, do_conditions, ob_conditions)
        return mep_do - mep_ob, best_value, mep_do, mep_ob

    def gauge_auf_prob(
        self,
        do_conditions: Mapping[str, int] | None = None,
        ob_conditions: Mapping[str, int] | None = None,
    ) -> float:
        """Return ``P(target ∈ desired_bins | do, ob)``."""

        do_dict = self._normalise(do_conditions)
        ob_dict = self._normalise(ob_conditions)
        self._require_disjoint(do_dict, ob_dict)
        return self._gauge_auf_prob(do_dict, ob_dict)

    # ------------------------------------------------------------------ recursion

    def _mep_recursive(
        self,
        do_dict: dict[str, int],
        ob_dict: dict[str, int],
        node: str | None,
    ) -> float:
        ctx = self._freeze(do_dict, ob_dict)
        cache_key = (ctx, node)
        cached = self._mep_memo.get(cache_key)
        if cached is not None:
            return cached

        if node is None:
            result = self._gauge_auf_prob(do_dict, ob_dict)
        elif node in do_dict or node in ob_dict:
            result = self._mep_recursive(do_dict, ob_dict, self._next_node(node))
        elif self._alterable is not None and node not in self._alterable:
            # Non-alterable variable: only the observation branch is valid;
            # ``do(v)`` is not a physically realisable action for ``v``.
            mep_ob = self._mep_ob_at(node, do_dict, ob_dict)
            self.decision_log.append(
                MEPDecisionLog(
                    node=node,
                    do_conditions=dict(do_dict),
                    ob_conditions=dict(ob_dict),
                    mep_do=None,
                    mep_ob=float(mep_ob),
                    chosen="ob",
                    best_do_value=None,
                    alterable=False,
                )
            )
            result = mep_ob
        else:
            mep_do, best_value = self._best_do_at(node, do_dict, ob_dict)
            mep_ob = self._mep_ob_at(node, do_dict, ob_dict)
            chosen = "do" if mep_do >= mep_ob else "ob"
            self.decision_log.append(
                MEPDecisionLog(
                    node=node,
                    do_conditions=dict(do_dict),
                    ob_conditions=dict(ob_dict),
                    mep_do=float(mep_do),
                    mep_ob=float(mep_ob),
                    chosen=chosen,
                    best_do_value=int(best_value),
                    alterable=True,
                )
            )
            result = max(mep_do, mep_ob)

        self._mep_memo[cache_key] = float(result)
        return float(result)

    def _best_do_at(
        self,
        node: str,
        do_dict: dict[str, int],
        ob_dict: dict[str, int],
    ) -> tuple[float, int]:
        next_node = self._next_node(node)
        best_value: int | None = None
        best_mep = -np.inf
        for value in self.discretizer.get_bins(node):
            new_do = dict(do_dict)
            new_do[node] = int(value)
            mep = self._mep_recursive(new_do, ob_dict, next_node)
            if mep > best_mep:
                best_mep = mep
                best_value = int(value)
        if best_value is None:
            raise RuntimeError(f"No bins available for node {node!r}.")
        ctx = self._freeze(do_dict, ob_dict)
        self._best_do_memo[(ctx, node)] = best_value
        return float(best_mep), best_value

    def _mep_ob_at(
        self,
        node: str,
        do_dict: dict[str, int],
        ob_dict: dict[str, int],
    ) -> float:
        next_node = self._next_node(node)
        prob_dist = self._bin_distribution(node, do_dict, ob_dict)
        total = 0.0
        for value, prob in prob_dist.items():
            if prob <= 0.0:
                continue
            new_ob = dict(ob_dict)
            new_ob[node] = int(value)
            total += float(prob) * self._mep_recursive(do_dict, new_ob, next_node)
        return float(total)

    # ------------------------------------------------------------------ probability

    def _gauge_auf_prob(
        self,
        do_dict: dict[str, int],
        ob_dict: dict[str, int],
    ) -> float:
        prob_dist = self._bin_distribution(self.target_node, do_dict, ob_dict)
        desired = set(self.desired_bins)
        return float(sum(prob for value, prob in prob_dist.items() if value in desired))

    def _bin_distribution(
        self,
        node: str,
        do_dict: dict[str, int],
        ob_dict: dict[str, int],
    ) -> dict[int, float]:
        ctx = self._freeze(do_dict, ob_dict)
        cache_key = (node, ctx)
        cached = self._prob_cache.get(cache_key)
        if cached is not None:
            return cached
        samples = self._simulate(do_dict, ob_dict)
        column = np.asarray(samples[node], dtype=float).reshape(-1)
        if column.size == 0:
            uniform_prob = 1.0 / max(1, self.discretizer.n_bins(node))
            dist = {int(v): uniform_prob for v in self.discretizer.get_bins(node)}
        else:
            bins = self.discretizer.discretize(node, column)
            counts = np.bincount(np.asarray(bins, dtype=int), minlength=self.discretizer.n_bins(node))
            total = float(counts.sum())
            dist = {int(v): float(counts[v]) / total for v in self.discretizer.get_bins(node)}
        self._prob_cache[cache_key] = dist
        return dist

    def _simulate(
        self,
        do_dict: dict[str, int],
        ob_dict: dict[str, int],
    ) -> dict[str, np.ndarray]:
        ctx = self._freeze(do_dict, ob_dict)
        cached = self._sample_cache.get(ctx)
        if cached is not None:
            return cached

        alterations = {
            name: self.discretizer.get_continuous_value(name, value)
            for name, value in do_dict.items()
        }
        samples = self.model.simulate(
            self.num_samples,
            rng=self.rng,
            alterations=alterations or None,
        )
        if ob_dict:
            mask = np.ones(self.num_samples, dtype=bool)
            for name, bin_idx in ob_dict.items():
                values = np.asarray(samples[name], dtype=float).reshape(-1)
                sample_bins = self.discretizer.discretize(name, values)
                mask &= np.asarray(sample_bins, dtype=int) == int(bin_idx)
            if mask.any():
                samples = {key: np.asarray(value)[mask] for key, value in samples.items()}
            else:
                # Fall back to the unconditional samples; matches the reference's
                # behaviour when rejection sampling produces zero matches and avoids
                # propagating divide-by-zero errors deeper into the recursion.
                samples = {key: np.asarray(value) for key, value in samples.items()}

        self._sample_cache[ctx] = samples
        return samples

    # ------------------------------------------------------------------ helpers

    def _next_node(self, current: str) -> str | None:
        try:
            idx = self.ordering.index(current)
        except ValueError:
            return None
        for candidate in self.ordering[idx + 1:]:
            if candidate == self.target_node:
                return None
            return candidate
        return None

    def _require_in_ordering(self, name: str) -> None:
        if name not in self.ordering:
            raise ValueError(f"Node {name!r} is not in this calculator's ordering.")
        if name == self.target_node:
            raise ValueError(
                f"Node {name!r} is the target node; INP/MEP only apply to upstream nodes."
            )
        if self.ordering.index(name) >= self.ordering.index(self.target_node):
            raise ValueError(
                f"Node {name!r} appears after the target {self.target_node!r} in the ordering; "
                "INP/MEP can only be evaluated for nodes that precede the target."
            )

    @staticmethod
    def _normalise(conditions: Mapping[str, int] | None) -> dict[str, int]:
        if conditions is None:
            return {}
        out: dict[str, int] = {}
        for name, value in conditions.items():
            out[str(name)] = int(value)
        return out

    @staticmethod
    def _freeze(do_dict: dict[str, int], ob_dict: dict[str, int]) -> _FrozenContext:
        return (
            tuple(sorted((str(k), int(v)) for k, v in do_dict.items())),
            tuple(sorted((str(k), int(v)) for k, v in ob_dict.items())),
        )

    @staticmethod
    def _require_disjoint(do_dict: Mapping[str, Any], ob_dict: Mapping[str, Any]) -> None:
        overlap = set(do_dict).intersection(ob_dict)
        if overlap:
            raise ValueError(
                f"do_conditions and ob_conditions must be disjoint (overlap: {sorted(overlap)})."
            )
