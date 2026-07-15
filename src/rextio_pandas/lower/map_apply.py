"""Lower audited Series map claims into one native Rust loop."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweredExpr, LoweringContext

from rextio_pandas.claim.map_apply import (
    DATAFRAME_APPLY_RULE,
    SERIES_MAP_RULE,
    audit_frame_callable,
    audit_series_callable,
)
from rextio_pandas.diagnostics import FRAME_F64, SERIES_F64, SERIES_I64, SERIES_TYPES
from rextio_pandas.rust_snippets.map_apply import (
    boundary_helpers,
    dataframe_apply_helpers,
    series_map_helpers,
)


def lower(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr:
    """Revalidate frozen metadata and emit the deterministic map helper call."""
    if claimed.rule_id == DATAFRAME_APPLY_RULE:
        return _lower_dataframe_apply(claimed, ctx)
    return _lower_series_map(claimed, ctx)


def _lower_series_map(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr:
    """Lower one defensively revalidated Series map claim."""
    receiver = claimed.receiver
    if (
        claimed.rule_id != SERIES_MAP_RULE
        or receiver is None
        or receiver.arg_type not in SERIES_TYPES
        or len(claimed.callables) != 1
        or ctx.receiver is None
    ):
        raise ValueError("rextio-pandas received malformed Series.map lower metadata")
    meta = claimed.callables[0]
    audit = audit_series_callable(meta, receiver.arg_type)
    if not audit.accepted:
        raise ValueError(f"rextio-pandas refused changed callable metadata: {audit.reason}")
    result_key = SERIES_F64 if audit.result_type == "float" else SERIES_I64
    if claimed.result_type != result_key:
        raise ValueError(
            "rextio-pandas Series.map result type changed between claim and lower: "
            f"{claimed.result_type!r} != {result_key!r}"
        )
    helper_name, helpers = series_map_helpers(receiver.arg_type, result_key, meta)
    return LoweredExpr(
        rust=f"{helper_name}(py, &{ctx.receiver})?",
        helpers=(boundary_helpers(), *helpers),
    )


def _lower_dataframe_apply(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr:
    """Lower one homogeneous-f64, schema-bound row apply claim."""
    receiver = claimed.receiver
    if (
        receiver is None
        or receiver.arg_type != FRAME_F64
        or receiver.schema is None
        or len(claimed.callables) != 1
        or ctx.receiver is None
        or claimed.result_type != SERIES_F64
    ):
        raise ValueError("rextio-pandas received malformed DataFrame.apply lower metadata")
    meta = claimed.callables[0]
    audit = audit_frame_callable(meta, receiver.schema)
    if not audit.accepted:
        raise ValueError(f"rextio-pandas refused changed row metadata: {audit.reason}")
    helper_name, helpers = dataframe_apply_helpers(receiver.schema, meta)
    return LoweredExpr(
        rust=f"{helper_name}(py, &{ctx.receiver})?",
        helpers=(boundary_helpers(), *helpers),
    )


__all__ = ["lower"]
