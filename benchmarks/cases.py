"""Deterministic product-route benchmark fixtures and context baselines."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

KERNEL_SOURCE = """
from rextio_pandas.types import DataFrameF64, SeriesF64


class Pair:
    left: float
    right: float


def series_udf(value: float) -> float:
    return value * 2.0 if value > 0.0 else -value


def row_udf(row) -> float:
    return row["left"] if row["left"] >= 0.0 else -row["right"]


def series_map(values: SeriesF64) -> SeriesF64:
    return values.map(series_udf)


def dataframe_apply(frame: DataFrameF64[Pair]) -> SeriesF64:
    return frame.apply(row_udf, axis=1)
"""


@dataclass(frozen=True)
class BenchmarkCase:
    """One product route and its semantically equivalent context functions."""

    route: str
    function_name: str
    argument: pd.Series | pd.DataFrame
    contexts: tuple[tuple[str, Callable[[], pd.Series]], ...]


def _series_vectorized(source: pd.Series) -> pd.Series:
    values = source.to_numpy(copy=False)
    output = np.where(values > 0.0, values * 2.0, -values)
    return pd.Series(output, index=source.index, name=source.name)


def _dataframe_default(frame: pd.DataFrame) -> pd.Series:
    return frame.apply(
        lambda row: row["left"] if row["left"] >= 0.0 else -row["right"],
        axis=1,
    )


def _dataframe_raw(frame: pd.DataFrame) -> pd.Series:
    return frame.apply(
        lambda row: row[0] if row[0] >= 0.0 else -row[1],
        axis=1,
        raw=True,
    )


def _dataframe_vectorized(frame: pd.DataFrame) -> pd.Series:
    left = frame["left"].to_numpy(copy=False)
    right = frame["right"].to_numpy(copy=False)
    output = np.where(left >= 0.0, left, -right)
    return pd.Series(output, index=frame.index, name=None)


def make_case(route: str, size: int) -> BenchmarkCase:
    """Build one deterministic Series or homogeneous-f64 DataFrame case."""
    if route == "series.map":
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
    if route == "dataframe.apply":
        frame = pd.DataFrame(
            {
                "left": np.linspace(-1000.0, 1000.0, size, dtype=np.float64),
                "right": np.linspace(1000.0, -1000.0, size, dtype=np.float64),
            }
        )
        return BenchmarkCase(
            route=route,
            function_name="dataframe_apply",
            argument=frame,
            contexts=(
                ("pandas_apply_default", lambda: _dataframe_default(frame)),
                ("pandas_apply_raw_true", lambda: _dataframe_raw(frame)),
                ("vectorized_numpy_pandas", lambda: _dataframe_vectorized(frame)),
            ),
        )
    raise ValueError(f"unknown benchmark route: {route}")


__all__ = ["BenchmarkCase", "KERNEL_SOURCE", "make_case"]
