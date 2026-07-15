"""Plugin keys, diagnostics, and stable runtime contract messages."""

from __future__ import annotations

from rextio.analyzer.diagnostics import Diagnostic
from rextio.plugins.api import ClaimSite, Rejected

SERIES_F64 = "rextio-pandas/series-f64"
SERIES_I64 = "rextio-pandas/series-i64"
FRAME_F64 = "rextio-pandas/frame-f64"

SERIES_TYPES = frozenset({SERIES_F64, SERIES_I64})

DIAGNOSTIC_SHAPE = "RXTP-PANDAS-001"
DIAGNOSTIC_SIGNATURE = "RXTP-PANDAS-002"
DIAGNOSTIC_BODY = "RXTP-PANDAS-003"

RUNTIME_ERRORS = {
    "version": ("rextio-pandas requires pandas==2.3.3 and numpy==2.3.5 for this private incubator"),
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
    "FRAME_F64",
    "RUNTIME_ERRORS",
    "SERIES_F64",
    "SERIES_I64",
    "SERIES_TYPES",
    "reject",
]
