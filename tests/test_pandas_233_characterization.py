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
        ("bool", [True, False, True], bool),
    ],
)
def test_series_map_mapper_receives_python_scalars(
    dtype: str,
    values: list[int] | list[float] | list[bool],
    scalar_type: type[int] | type[float],
) -> None:
    seen: list[type[object]] = []
    source = pd.Series(values, dtype=dtype, name="source")

    result = source.map(lambda value: seen.append(type(value)) or value)

    assert seen == [scalar_type] * len(source)
    assert_series_equal(result, source, check_exact=True)


@pytest.mark.parametrize("dtype", ["int64", "float64", "bool"])
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


def test_series_map_na_action_none_does_not_reach_map_infer_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pandas._libs.lib as lib

    def unreachable(*args: object, **kwargs: object) -> object:
        raise AssertionError("map_infer_mask is unreachable for na_action=None")

    monkeypatch.setattr(lib, "map_infer_mask", unreachable)
    source = pd.Series([1.0, 2.0], dtype="float64")

    result = source.map(lambda value: value * 2.0, na_action=None)

    assert_series_equal(result, pd.Series([2.0, 4.0], dtype="float64"), check_exact=True)


def test_series_map_does_not_read_finalize_shadow_from_input_instance() -> None:
    source = pd.Series([1.0, 2.0], dtype="float64")
    calls = 0

    def input_only_shadow(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("input instance __finalize__ must not be called")

    source.__dict__["__finalize__"] = input_only_shadow
    descriptor = pd.Series.__dict__["map"]

    result = descriptor(source, lambda value: value * 2.0)

    assert calls == 0
    assert_series_equal(result, pd.Series([2.0, 4.0], dtype="float64"), check_exact=True)


@pytest.mark.parametrize(
    ("kind", "frame", "row_dtype", "scalar_type"),
    [
        (
            "float64",
            pd.DataFrame({"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])}),
            np.dtype("float64"),
            np.float64,
        ),
        (
            "int64",
            pd.DataFrame(
                {
                    "a": np.array([1, 2], dtype=np.int64),
                    "b": np.array([3, 4], dtype=np.int64),
                }
            ),
            np.dtype("int64"),
            np.int64,
        ),
        (
            "mixed",
            pd.DataFrame(
                {
                    "a": np.array([1, 2], dtype=np.int64),
                    "b": np.array([3.0, 4.0]),
                }
            ),
            np.dtype("float64"),
            np.float64,
        ),
    ],
)
def test_dataframe_apply_row_dtype_and_subscript_scalar_types(
    kind: str,
    frame: pd.DataFrame,
    row_dtype: np.dtype[object],
    scalar_type: type[object],
) -> None:
    seen: list[tuple[object, ...]] = []

    def observe(row: pd.Series) -> float | int:
        seen.append(
            (
                type(row),
                row.dtype,
                type(row["a"]),
                type(row.iloc[0]),
                row.name,
                tuple(row.index),
                row.index.name,
            )
        )
        return row["a"]

    result = frame.apply(observe, axis=1)

    assert kind in {"float64", "int64", "mixed"}
    assert seen == [
        (pd.Series, row_dtype, scalar_type, scalar_type, 0, ("a", "b"), None),
        (pd.Series, row_dtype, scalar_type, scalar_type, 1, ("a", "b"), None),
    ]
    assert type(result) is pd.Series
    assert result.index is frame.index
    assert result.name is None
    assert result.attrs == {}
    assert result.flags.allows_duplicate_labels is True
    if kind == "mixed":
        assert result.dtype == np.dtype("float64")


@pytest.mark.parametrize("size", [1, 7])
def test_dataframe_apply_float64_output_contract(size: int) -> None:
    frame = pd.DataFrame(
        {
            "left": np.linspace(-3.0, 3.0, size),
            "right": np.linspace(10.0, 20.0, size),
        }
    )

    result = frame.apply(
        lambda row: row["left"] if row["left"] >= 0.0 else -row["right"],
        axis=1,
    )
    expected = pd.Series(
        [left if left >= 0.0 else -right for left, right in frame.itertuples(index=False)],
        dtype="float64",
    )

    assert_series_equal(result, expected, check_exact=True)
    assert type(result.index) is pd.RangeIndex
    assert result.index.equals(pd.RangeIndex(size))


@pytest.mark.parametrize(
    ("shape", "frame", "expected_calls", "expected"),
    [
        (
            "0xN",
            pd.DataFrame(
                {
                    "a": pd.Series([], dtype="float64"),
                    "b": pd.Series([], dtype="float64"),
                }
            ),
            [(np.dtype("float64"), 2, None)],
            pd.Series([], dtype="float64"),
        ),
        (
            "Nx0",
            pd.DataFrame(index=pd.RangeIndex(2)),
            [(np.dtype("float64"), 0, None), (np.dtype("float64"), 0, None)],
            pd.Series([1.0, 1.0], dtype="float64"),
        ),
        (
            "0x0",
            pd.DataFrame(),
            [(np.dtype("float64"), 0, None)],
            pd.Series([], dtype="float64"),
        ),
    ],
)
def test_dataframe_apply_empty_shape_call_behavior(
    shape: str,
    frame: pd.DataFrame,
    expected_calls: list[tuple[np.dtype[object], int, None]],
    expected: pd.Series,
) -> None:
    calls: list[tuple[np.dtype[object], int, None]] = []

    def observe(row: pd.Series) -> float:
        calls.append((row.dtype, len(row), row.name))
        return 1.0

    result = frame.apply(observe, axis=1)

    assert shape in {"0xN", "Nx0", "0x0"}
    assert calls == expected_calls
    assert_series_equal(result, expected, check_exact=True)


def test_dataframe_apply_float_special_values_warnings_and_signed_zero() -> None:
    frame = pd.DataFrame(
        {
            "value": [-0.0, 0.0, math.nan, math.inf, -math.inf],
            "other": np.ones(5, dtype=np.float64),
        }
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = frame.apply(lambda row: -row["value"], axis=1)

    expected = pd.Series([0.0, -0.0, -math.nan, -math.inf, math.inf], dtype="float64")
    assert caught == []
    assert_series_equal(result, expected, check_exact=True)
    assert np.array_equal(result.to_numpy().view(np.uint64), expected.to_numpy().view(np.uint64))


@pytest.mark.parametrize(
    ("operator", "mapper", "message_variants"),
    [
        (
            "divide",
            lambda row: row["value"] / 0,
            (("invalid value encountered in scalar divide",) * 2,),
        ),
        (
            "floor_divide",
            lambda row: row["value"] // 0,
            (("invalid value encountered in scalar floor_divide",) * 2,),
        ),
        (
            "remainder",
            lambda row: row["value"] % 0,
            (
                (),
                ("invalid value encountered in scalar remainder",) * 4,
            ),
        ),
    ],
)
def test_dataframe_apply_numpy_float_zero_operation_warning_behavior(
    operator: str,
    mapper: object,
    message_variants: tuple[tuple[str, ...], ...],
) -> None:
    frame = pd.DataFrame(
        {"value": [-0.0, 0.0, math.nan, math.inf, -math.inf]},
        dtype="float64",
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = frame.apply(mapper, axis=1)  # type: ignore[arg-type]

    assert operator in {"divide", "floor_divide", "remainder"}
    assert result.dtype == np.dtype("float64")
    observed_messages = tuple(str(item.message) for item in caught)
    # NumPy 2.3.5 platform wheels differ here: macOS arm64 is silent for
    # scalar remainder while Linux x86_64 emits one warning for each of the
    # four invalid inputs. DataFrame.apply is a NO-GO characterization, so
    # retain both exact observations instead of turning wheel variance into a
    # blocking Series.map product failure.
    assert observed_messages in message_variants
    assert all(item.category is RuntimeWarning for item in caught)


def test_dataframe_apply_int64_overflow_is_numpy_scalar_semantics() -> None:
    limits = np.iinfo(np.int64)
    frame = pd.DataFrame({"value": np.array([limits.max, limits.min, 0], dtype=np.int64)})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = frame.apply(lambda row: row["value"] + np.int64(1), axis=1)

    expected = pd.Series([limits.min, limits.min + 1, 1], dtype="int64")
    assert_series_equal(result, expected, check_exact=True)
    assert [(item.category, str(item.message)) for item in caught] == [
        (RuntimeWarning, "overflow encountered in scalar add")
    ]


def test_dataframe_apply_fortran_storage_uses_logical_rows_and_columns() -> None:
    values = np.asfortranarray(np.arange(12, dtype=np.float64).reshape(4, 3))
    frame = pd.DataFrame(values, columns=["a", "b", "c"])
    assert frame.to_numpy(copy=False).flags.f_contiguous is True
    seen: list[tuple[float, float, float]] = []

    result = frame.apply(
        lambda row: seen.append((row["a"], row["b"], row["c"])) or row["a"],
        axis=1,
    )

    assert seen == [tuple(row) for row in values.tolist()]
    assert_series_equal(result, pd.Series(values[:, 0]), check_exact=True)


def test_dataframe_apply_descriptor_can_be_shadowed_on_an_exact_instance() -> None:
    frame = pd.DataFrame({"a": [1.0]}, dtype="float64")
    descriptor = pd.DataFrame.__dict__["apply"]

    assert pd.DataFrame.apply is descriptor
    frame.__dict__["apply"] = "shadowed"

    assert frame.apply == "shadowed"
    result = descriptor(frame, lambda row: row["a"], axis=1)
    assert_series_equal(result, pd.Series([1.0], dtype="float64"), check_exact=True)
