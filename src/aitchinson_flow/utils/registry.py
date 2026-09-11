from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar


T = TypeVar("T")


class Registry(Generic[T]):
    """Minimal decorator-style registry mapping string keys to builders of ``T``.

    Parameters
    ----------
    label:
        Human-readable label used inside error messages (e.g. ``"model"``,
        ``"lm"``, ``"benchmark task"``).
    """

    def __init__(self, label: str) -> None:
        self._label = label
        self._builders: dict[str, T] = {}

    @property
    def builders(self) -> dict[str, T]:
        """Return the underlying ``name -> builder`` dict (live reference)."""
        return self._builders

    def register(self, name: str) -> Callable[[T], T]:
        """Decorator that registers ``builder`` under ``name``.

        Raises ``ValueError`` if ``name`` is already taken.
        """

        def deco(builder: T) -> T:
            if name in self._builders:
                raise ValueError(f"Duplicate {self._label} key: {name!r}")
            self._builders[name] = builder
            return builder

        return deco

    def get(self, name: str) -> T:
        """Look up a builder, raising ``KeyError`` with the registered keys."""
        try:
            return self._builders[name]
        except KeyError as e:
            raise KeyError(
                f"Unknown {self._label} {name!r}. Registered: {sorted(self.builders)}"
            ) from e

    def keys(self) -> tuple[str, ...]:
        """Return all registered keys in sorted order."""
        return tuple(sorted(self._builders))
