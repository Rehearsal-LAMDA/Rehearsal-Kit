"""Uniform fixed-range discretizer used to discretize continuous outputs.

Each variable gets ``n_bins`` uniform bins over a fixed range, independent
of the data. The default ``n_bins`` is small because the MEP recursion
fan-out grows quickly with the number of bins.
"""

from __future__ import annotations

from typing import Mapping, Sequence, Tuple, Union

import numpy as np


_RangeLike = Union[Tuple[float, float], Sequence[float]]


class UniformBinDiscretizer:
    """Per-variable uniform bins over a fixed range.

    Parameters
    ----------
    variables:
        Names of the variables that this discretizer covers. Must be unique.
    n_bins:
        Default number of bins per variable. Either an integer used for every
        variable, or a mapping ``{name: n_bins}`` to override individual
        variables. Variables not present in the mapping fall back to the
        scalar default.
    bin_range:
        Default ``(low, high)`` tuple shared by all variables. May be
        overridden per variable via a mapping ``{name: (low, high)}``.
    """

    def __init__(
        self,
        variables: Sequence[str],
        *,
        n_bins: int | Mapping[str, int] = 3,
        bin_range: _RangeLike | Mapping[str, _RangeLike] = (-3.0, 3.0),
    ) -> None:
        names = tuple(str(name) for name in variables)
        if len(set(names)) != len(names):
            raise ValueError("UniformBinDiscretizer variables must be unique.")
        if not names:
            raise ValueError("UniformBinDiscretizer requires at least one variable.")

        self._variables: tuple[str, ...] = names
        self._n_bins: dict[str, int] = self._coerce_n_bins(names, n_bins)
        self._ranges: dict[str, tuple[float, float]] = self._coerce_ranges(names, bin_range)

        self._edges: dict[str, np.ndarray] = {}
        self._centers: dict[str, np.ndarray] = {}
        for name in names:
            low, high = self._ranges[name]
            edges = np.linspace(low, high, self._n_bins[name] + 1, dtype=float)
            self._edges[name] = edges
            self._centers[name] = 0.5 * (edges[:-1] + edges[1:])

    @property
    def variables(self) -> tuple[str, ...]:
        return self._variables

    def has(self, name: str) -> bool:
        return name in self._n_bins

    def n_bins(self, name: str) -> int:
        self._require(name)
        return self._n_bins[name]

    def bin_range(self, name: str) -> tuple[float, float]:
        self._require(name)
        return self._ranges[name]

    def get_bins(self, name: str) -> tuple[int, ...]:
        """Return the tuple of legal bin indices for ``name``."""

        self._require(name)
        return tuple(range(self._n_bins[name]))

    def get_bin_edges(self, name: str) -> np.ndarray:
        self._require(name)
        return self._edges[name].copy()

    def get_bin_centers(self, name: str) -> np.ndarray:
        self._require(name)
        return self._centers[name].copy()

    def get_continuous_value(self, name: str, bin_idx: int) -> float:
        self._require(name)
        centers = self._centers[name]
        idx = int(bin_idx)
        if idx < 0 or idx >= centers.size:
            raise ValueError(
                f"bin index {bin_idx} out of range [0, {centers.size - 1}] for variable {name!r}."
            )
        return float(centers[idx])

    def discretize(self, name: str, value: float | np.ndarray) -> int | np.ndarray:
        """Map a continuous value (or array) to its bin index.

        Values outside the configured range are clamped to the nearest bin so
        that downstream code never sees an out-of-range bin index.
        """

        self._require(name)
        edges = self._edges[name]
        n = self._n_bins[name]
        idx = np.digitize(np.asarray(value, dtype=float), edges) - 1
        idx = np.clip(idx, 0, n - 1)
        if np.ndim(value) == 0:
            return int(idx)
        return idx.astype(int)

    @staticmethod
    def _coerce_n_bins(
        names: Sequence[str],
        n_bins: int | Mapping[str, int],
    ) -> dict[str, int]:
        if isinstance(n_bins, Mapping):
            default = None
            resolved: dict[str, int] = {}
            for name in names:
                if name in n_bins:
                    resolved[name] = int(n_bins[name])
                elif default is not None:
                    resolved[name] = int(default)
                else:
                    raise ValueError(
                        f"UniformBinDiscretizer: n_bins mapping is missing variable {name!r} "
                        "and no scalar default was provided."
                    )
        else:
            scalar = int(n_bins)
            resolved = {name: scalar for name in names}
        for name, value in resolved.items():
            if value < 1:
                raise ValueError(f"n_bins for {name!r} must be >= 1, got {value}.")
        return resolved

    @staticmethod
    def _coerce_ranges(
        names: Sequence[str],
        bin_range: _RangeLike | Mapping[str, _RangeLike],
    ) -> dict[str, tuple[float, float]]:
        if isinstance(bin_range, Mapping):
            resolved: dict[str, tuple[float, float]] = {}
            for name in names:
                if name not in bin_range:
                    raise ValueError(
                        f"UniformBinDiscretizer: bin_range mapping is missing variable {name!r}."
                    )
                resolved[name] = UniformBinDiscretizer._coerce_single_range(name, bin_range[name])
        else:
            shared = UniformBinDiscretizer._coerce_single_range("<default>", bin_range)
            resolved = {name: shared for name in names}
        return resolved

    @staticmethod
    def _coerce_single_range(name: str, value: _RangeLike) -> tuple[float, float]:
        low, high = float(value[0]), float(value[1])
        if not (np.isfinite(low) and np.isfinite(high)):
            raise ValueError(f"bin_range for {name!r} must be finite, got ({low}, {high}).")
        if low >= high:
            raise ValueError(f"bin_range for {name!r} must have low < high, got ({low}, {high}).")
        return low, high

    def _require(self, name: str) -> None:
        if name not in self._n_bins:
            raise KeyError(f"Variable {name!r} is not registered with this discretizer.")

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        bins = {name: self._n_bins[name] for name in self._variables}
        return f"UniformBinDiscretizer(variables={self._variables!r}, n_bins={bins!r})"
