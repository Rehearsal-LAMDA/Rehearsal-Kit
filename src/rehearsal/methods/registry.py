"""Small method registry used by experiment config runners."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Mapping

from rehearsal.core import RehearsalMethod
from rehearsal.methods.care import ICML2025CARERehearsal
from rehearsal.methods.cme import CMERehearsal

MethodFactory = Callable[..., RehearsalMethod]

_METHODS: dict[str, MethodFactory] = {
    "icml2025-care": ICML2025CARERehearsal,
    "unpublished-cme": CMERehearsal,
}


def create_method(name: str, params: Mapping[str, Any] | None = None) -> RehearsalMethod:
    """Instantiate a registered rehearsal method."""

    try:
        factory = _METHODS[name]
    except KeyError as exc:
        available = ", ".join(sorted(_METHODS))
        raise ValueError(f"Unknown method {name!r}. Available methods: {available}.") from exc
    return factory(**dict(params or {}))


def register_method(name: str, factory: MethodFactory) -> None:
    """Register a method factory for local experiment scripts."""

    if not name:
        raise ValueError("method name must be non-empty.")
    _METHODS[name] = factory


def available_methods() -> tuple[str, ...]:
    return tuple(sorted(_METHODS))
