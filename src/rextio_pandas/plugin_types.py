"""Materialized pandas plugin types for the API 1.3 incubator."""

from __future__ import annotations

from rextio.plugins.api import BoundaryConversion, PluginType

from rextio_pandas.diagnostics import SERIES_F64, SERIES_I64


def _series_type(key: str, annotation: str, rust_type: str, extractor: str) -> PluginType:
    return PluginType(
        key=key,
        annotations=(f"rextio_pandas.types.{annotation}",),
        rust_type=rust_type,
        conversion=BoundaryConversion(
            param_rust="pyo3::Bound<'py, pyo3::PyAny>",
            param_expr=f"{extractor}(py, &{{param}})?",
            return_rust="pyo3::Bound<'py, pyo3::PyAny>",
            return_expr="__rxtpd_materialize_series(py, {value})?",
        ),
    )


PLUGIN_TYPES: tuple[PluginType, ...] = (
    _series_type(
        SERIES_F64,
        "SeriesF64",
        "RxtPandasSeriesF64",
        "__rxtpd_extract_series_f64",
    ),
    _series_type(
        SERIES_I64,
        "SeriesI64",
        "RxtPandasSeriesI64",
        "__rxtpd_extract_series_i64",
    ),
)

_BY_KEY = {plugin_type.key: plugin_type for plugin_type in PLUGIN_TYPES}


def plugin_types() -> tuple[PluginType, ...]:
    """Return the stable materialized Series vocabulary."""
    return PLUGIN_TYPES


def plugin_type(key: str) -> PluginType:
    """Return one registered type by key."""
    return _BY_KEY[key]


__all__ = ["PLUGIN_TYPES", "plugin_type", "plugin_types"]
