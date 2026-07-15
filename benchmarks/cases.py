"""Deterministic Series.map product benchmark fixture and context baseline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

KERNEL_SOURCE = """
from rextio_pandas.types import SeriesF64


def series_udf(value: float) -> float:
    return value * 2.0 if value > 0.0 else -value


def series_map(values: SeriesF64) -> SeriesF64:
    return values.map(series_udf)
"""


@dataclass(frozen=True)
class BenchmarkCase:
    """One Series product cell and its semantically equivalent context lanes."""

    route: str
    function_name: str
    argument: pd.Series
    contexts: tuple[tuple[str, Callable[[], pd.Series]], ...]


def _series_vectorized(source: pd.Series) -> pd.Series:
    values = source.to_numpy(copy=False)
    output = np.where(values > 0.0, values * 2.0, -values)
    return pd.Series(output, index=source.index, name=source.name)


def make_case(route: str, size: int) -> BenchmarkCase:
    """Build the sole headline-authoritative Series.map case."""
    if route != "series.map":
        raise ValueError(f"unknown benchmark route: {route}")
    source = pd.Series(
        np.linspace(-1000.0, 1000.0, size, dtype=np.float64),
        name="values",
    )
    return BenchmarkCase(
        route=route,
        function_name="series_map",
        argument=source,
        contexts=(("vectorized_numpy_pandas", lambda: _series_vectorized(source)),),
    )


__all__ = ["BenchmarkCase", "KERNEL_SOURCE", "make_case"]
