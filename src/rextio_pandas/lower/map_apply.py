"""Lower audited Series map claims into one native Rust loop."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweredExpr, LoweringContext

from rextio_pandas.claim.map_apply import SERIES_MAP_RULE, audit_series_callable
from rextio_pandas.diagnostics import SERIES_BOOL, SERIES_F64, SERIES_I64, SERIES_TYPES
from rextio_pandas.rust_snippets.map_apply import (
    boundary_helpers,
    series_map_helpers,
)


def lower(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr:
    """Revalidate one Series.map claim and emit its deterministic helper call."""
    return _lower_series_map(claimed, ctx)


def _lower_series_map(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr:
    """Lower one defensively revalidated Series map claim."""
    if getattr(ctx, "backend", "pyo3") != "pyo3":
        raise ValueError(
            "rextio-pandas supports only PyO3 host-extension lowering; "
            "standalone artifact lowering is not declared"
        )
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
    result_type = audit.result_type
    if result_type not in {"float", "int", "bool"}:
        raise ValueError("rextio-pandas refused an unsupported audited scalar result type")
    result_key = {
        "float": SERIES_F64,
        "int": SERIES_I64,
        "bool": SERIES_BOOL,
    }[result_type]
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


__all__ = ["lower"]
