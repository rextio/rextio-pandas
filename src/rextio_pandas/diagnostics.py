"""Plugin keys, diagnostics, and stable runtime contract messages."""

from __future__ import annotations

from rextio.analyzer.diagnostics import Diagnostic
from rextio.plugins.api import ClaimSite, Rejected

SERIES_F64 = "rextio-pandas/series-f64"
SERIES_I64 = "rextio-pandas/series-i64"
# Research-only key used by characterization fixtures. It is deliberately not
# registered in the plugin type vocabulary and cannot reach claim/lower.
PROTOTYPE_FRAME_F64 = "rextio-pandas/prototype-frame-f64"

SERIES_TYPES = frozenset({SERIES_F64, SERIES_I64})

DIAGNOSTIC_SHAPE = "RXTP-PANDAS-001"
DIAGNOSTIC_SIGNATURE = "RXTP-PANDAS-002"
DIAGNOSTIC_BODY = "RXTP-PANDAS-003"

RUNTIME_ERRORS = {
    "version": (
        "rextio-pandas requires CPython 3.11, pandas==2.3.3, and numpy==2.3.5 "
        "for this public alpha release"
    ),
    "series_class": "rextio-pandas Series contract requires an exact pandas.Series",
    "series_empty": "rextio-pandas Series contract does not accept an empty Series",
    "series_index": (
        "rextio-pandas Series contract requires an unnamed RangeIndex(start=0, step=1)"
    ),
    "series_name": "rextio-pandas Series contract requires Series.name to be None or str",
    "series_attrs": "rextio-pandas Series contract requires empty attrs",
    "series_flags": (
        "rextio-pandas Series contract requires flags.allows_duplicate_labels is True"
    ),
    "series_method": (
        "rextio-pandas Series contract requires the pinned pandas.Series.map "
        "descriptor with no instance override"
    ),
    "series_f64": (
        "rextio-pandas SeriesF64 contract requires non-nullable NumPy-backed float64 storage"
    ),
    "series_i64": (
        "rextio-pandas SeriesI64 contract requires non-nullable NumPy-backed int64 storage"
    ),
    # Retained only by the unregistered DataFrame research prototype. No public
    # claim/lower path emits these errors.
    "frame_class": "rextio-pandas DataFrame contract requires an exact pandas.DataFrame",
    "frame_empty": (
        "rextio-pandas DataFrameF64 contract does not accept a DataFrame with zero rows"
    ),
    "frame_zero_columns": "rextio-pandas DataFrameF64 contract requires at least one column",
    "frame_index": (
        "rextio-pandas DataFrame contract requires an unnamed RangeIndex(start=0, step=1)"
    ),
    "frame_attrs": "rextio-pandas DataFrame contract requires empty attrs",
    "frame_flags": (
        "rextio-pandas DataFrame contract requires flags.allows_duplicate_labels is True"
    ),
    "frame_method": (
        "rextio-pandas DataFrame contract requires the pinned pandas.DataFrame.apply "
        "descriptor with no instance override"
    ),
    "frame_f64": (
        "rextio-pandas DataFrameF64 contract requires homogeneous non-nullable "
        "NumPy-backed float64 columns"
    ),
    "frame_schema": (
        "rextio-pandas DataFrameF64 contract requires unique ordered string columns "
        "exactly matching the declared schema and columns.name is None"
    ),
}


def reject(site: ClaimSite, code: str, message: str, suggestion: str) -> Rejected:
    """Build a location-neutral plugin rejection; core stamps the source site."""
    return Rejected(
        diagnostic=Diagnostic(
            code=code,
            severity="error",
            message=f"rextio-pandas cannot lower {site.target!r}: {message}",
            file_path="",
            line=0,
            column=0,
            suggestion=suggestion,
        )
    )


__all__ = [
    "DIAGNOSTIC_BODY",
    "DIAGNOSTIC_SHAPE",
    "DIAGNOSTIC_SIGNATURE",
    "PROTOTYPE_FRAME_F64",
    "RUNTIME_ERRORS",
    "SERIES_F64",
    "SERIES_I64",
    "SERIES_TYPES",
    "reject",
]
