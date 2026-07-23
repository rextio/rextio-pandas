"""Side-effect-free public annotation vocabulary for rextio-pandas.

These marker classes intentionally import neither pandas nor Rextio. Runtime
values remain ordinary pandas objects; the integrated Rextio analyzer resolves
the exact dotted annotation spellings to plugin types without executing the
annotations.
"""

from __future__ import annotations

from typing import Any


class SeriesF64:
    """An exact pandas Series with non-nullable NumPy ``float64`` storage."""


class SeriesI64:
    """An exact pandas Series with non-nullable NumPy ``int64`` storage."""


class SeriesBool:
    """An exact pandas Series with non-nullable NumPy ``bool`` storage."""


class DataFrameF64:
    """Research-only DataFrame marker; not a registered native plugin type."""

    def __class_getitem__(cls, schema: object) -> type[DataFrameF64]:
        """Keep ``DataFrameF64[Schema]`` importable without runtime dependencies."""
        del schema
        return cls


def __getattr__(name: str) -> Any:
    """Reject misspelled annotation names with the normal module error."""
    raise AttributeError(name)


__all__ = ["DataFrameF64", "SeriesBool", "SeriesF64", "SeriesI64"]
