from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal


PANDAS_VERSION = "2.3.3"
NUMPY_VERSION = "2.3.5"


def test_pinned_characterization_versions() -> None:
    assert pd.__version__ == PANDAS_VERSION
    assert np.__version__ == NUMPY_VERSION


@pytest.mark.parametrize(
    ("dtype", "values", "scalar_type"),
    [
        ("int64", [-2, 0, 3], int),
        ("float64", [-2.5, 0.0, 3.5], float),
    ],
)
def test_series_map_mapper_receives_python_scalars(
    dtype: str,
    values: list[int] | list[float],
    scalar_type: type[int] | type[float],
) -> None:
    seen: list[type[object]] = []
    source = pd.Series(values, dtype=dtype, name="source")

    result = source.map(lambda value: seen.append(type(value)) or value)

    assert seen == [scalar_type] * len(source)
    assert_series_equal(result, source, check_exact=True)


@pytest.mark.parametrize("dtype", ["int64", "float64"])
def test_series_map_empty_preserves_contract_without_calling_mapper(dtype: str) -> None:
    calls = 0
    source = pd.Series([], dtype=dtype, name="empty")

    def mapper(value: object) -> object:
        nonlocal calls
        calls += 1
        return value

    result = source.map(mapper)

    assert calls == 0
    assert_series_equal(result, source, check_exact=True)
    assert type(result) is pd.Series
    assert type(result.index) is pd.RangeIndex
    assert result.index.equals(pd.RangeIndex(0))
    assert result.index.name is None
    assert result.attrs == {}
    assert result.flags.allows_duplicate_labels is True


@pytest.mark.parametrize("size", [1, 7])
def test_series_map_preserves_range_index_name_dtype_and_order(size: int) -> None:
    source = pd.Series(
        np.arange(size, dtype=np.float64),
        index=pd.RangeIndex(start=0, stop=size, step=1),
        name="values",
    )

    result = source.map(lambda value: value + 0.5)
    expected = pd.Series(
        np.arange(size, dtype=np.float64) + 0.5,
        index=source.index,
        name="values",
    )

    assert_series_equal(result, expected, check_exact=True)
    assert type(result) is pd.Series
    assert result.index is source.index
    assert result.attrs == {}
    assert result.flags.allows_duplicate_labels is True


def test_series_map_int64_limits_expose_python_widening() -> None:
    i64 = np.iinfo(np.int64)

    max_result = pd.Series([i64.max], dtype="int64").map(lambda value: value + 1)
    min_result = pd.Series([i64.min], dtype="int64").map(lambda value: value - 1)
    negated_min = pd.Series([i64.min], dtype="int64").map(lambda value: -value)

    assert max_result.dtype == np.dtype("uint64")
    assert max_result.tolist() == [2**63]
    assert min_result.dtype == np.dtype("object")
    assert min_result.tolist() == [-(2**63) - 1]
    assert negated_min.dtype == np.dtype("uint64")
    assert negated_min.tolist() == [2**63]


def test_series_map_float_special_values_and_signed_zero() -> None:
    source = pd.Series([-0.0, 0.0, math.nan, math.inf, -math.inf], dtype="float64")

    result = source.map(lambda value: -value)
    expected = pd.Series([0.0, -0.0, -math.nan, -math.inf, math.inf], dtype="float64")

    assert_series_equal(result, expected, check_exact=True)
    result_bits = result.to_numpy().view(np.uint64)
    expected_bits = expected.to_numpy().view(np.uint64)
    assert np.array_equal(result_bits, expected_bits)
    assert np.signbit(result.iloc[0]) == np.bool_(False)
    assert np.signbit(result.iloc[1]) == np.bool_(True)


@pytest.mark.parametrize(
    ("dtype", "mapper", "message"),
    [
        ("int64", lambda value: value / 0, "division by zero"),
        ("float64", lambda value: value / 0, "float division by zero"),
        ("int64", lambda value: value // 0, "integer division or modulo by zero"),
        ("float64", lambda value: value // 0, "float floor division by zero"),
        ("int64", lambda value: value % 0, "integer modulo by zero"),
        ("float64", lambda value: value % 0, "float modulo"),
    ],
)
def test_series_map_zero_division_raises_without_warning(
    dtype: str,
    mapper: object,
    message: str,
) -> None:
    source = pd.Series([0, 1, -1], dtype=dtype)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ZeroDivisionError, match=f"^{message}$"):
            source.map(mapper)  # type: ignore[arg-type]

    assert caught == []


def test_series_map_copies_strided_values_in_logical_order() -> None:
    backing = np.arange(12, dtype=np.float64)
    source = pd.Series(backing[::2], name="strided")
    assert source.to_numpy(copy=False).flags.c_contiguous is False

    result = source.map(lambda value: value)

    assert_series_equal(result, pd.Series(backing[::2], name="strided"), check_exact=True)
    assert result.to_numpy(copy=False).flags.c_contiguous is True
    backing[0] = 999.0
    assert result.iloc[0] == 0.0


def test_series_map_descriptor_can_be_shadowed_on_an_exact_instance() -> None:
    source = pd.Series([1.0], dtype="float64")
    descriptor = pd.Series.__dict__["map"]

    assert pd.Series.map is descriptor
    source.__dict__["map"] = "shadowed"

    assert source.map == "shadowed"
    result = descriptor(source, lambda value: value)
    assert_series_equal(result, pd.Series([1.0], dtype="float64"), check_exact=True)
