"""Fail-closed claim logic for the narrow ``Series.map`` route."""

from __future__ import annotations

from dataclasses import dataclass

from rextio.config.schema import RextioConfig
from rextio.plugins.api import (
    CallableBodyExpr,
    CallableMeta,
    Claimed,
    ClaimResult,
    ClaimSite,
    NotCovered,
    SchemaMeta,
)

from rextio_pandas.diagnostics import (
    DIAGNOSTIC_APPLY_BODY,
    DIAGNOSTIC_APPLY_SCHEMA,
    DIAGNOSTIC_APPLY_SHAPE,
    DIAGNOSTIC_BODY,
    DIAGNOSTIC_SHAPE,
    DIAGNOSTIC_SIGNATURE,
    FRAME_F64,
    SERIES_F64,
    SERIES_I64,
    SERIES_TYPES,
    reject,
)

SERIES_MAP_RULE = "rextio-pandas/series-map"
DATAFRAME_APPLY_RULE = "rextio-pandas/dataframe-apply-axis1"

_COMPARISONS = frozenset({"==", "!=", "<", "<=", ">", ">="})
_FLOAT_BINOPS = frozenset({"+", "-", "*"})


@dataclass(frozen=True)
class BodyAudit:
    """Result of reconstructing the type of one audited callable body."""

    accepted: bool
    result_type: str | None = None
    reason: str = ""


def _fail(reason: str) -> BodyAudit:
    return BodyAudit(False, reason=reason)


def _audit_expr(expr: CallableBodyExpr, input_type: str, param_name: str) -> BodyAudit:
    """Reconstruct and validate the deliberately narrower pandas UDF grammar."""
    if expr.kind == "param":
        if expr.param_index != 0 or expr.name != param_name or expr.result_type != input_type:
            return _fail("parameter metadata is inconsistent with the callable signature")
        return BodyAudit(True, input_type)

    if expr.kind == "literal":
        literal = expr.literal
        if literal is None or literal.kind not in {"int", "float"}:
            return _fail("only finite int/float literals are audited")
        literal_type = literal.kind
        if expr.result_type != literal_type:
            return _fail("literal result type is inconsistent")
        return BodyAudit(True, literal_type)

    children = tuple(_audit_expr(child, input_type, param_name) for child in expr.children)
    failed = next((child for child in children if not child.accepted), None)
    if failed is not None:
        return failed
    child_types = tuple(child.result_type for child in children)

    if expr.kind == "unary":
        if input_type != "float" or expr.op != "-" or child_types != ("float",):
            return _fail("only unary negation of float64 values is audited")
        if expr.result_type != "float":
            return _fail("unary result type is inconsistent")
        return BodyAudit(True, "float")

    if expr.kind == "binop":
        if (
            input_type != "float"
            or expr.op not in _FLOAT_BINOPS
            or child_types != ("float", "float")
        ):
            return _fail("only same-type float64 +, -, and * are audited")
        if expr.result_type != "float":
            return _fail("binary result type is inconsistent")
        return BodyAudit(True, "float")

    if expr.kind == "compare":
        if not expr.ops or any(op not in _COMPARISONS for op in expr.ops):
            return _fail("identity, membership, and unaudited comparisons are rejected")
        if len(set(child_types)) != 1 or child_types[0] != input_type:
            return _fail("comparisons must use only the input scalar type")
        if expr.result_type != "bool":
            return _fail("comparison result type is inconsistent")
        return BodyAudit(True, "bool")

    if expr.kind == "boolop":
        if expr.op not in {"and", "or"} or any(kind != "bool" for kind in child_types):
            return _fail("boolean composition requires boolean operands")
        if expr.result_type != "bool":
            return _fail("boolean result type is inconsistent")
        return BodyAudit(True, "bool")

    if expr.kind == "cond":
        if len(child_types) != 3 or child_types[0] != "bool":
            return _fail("conditional expressions require a boolean test")
        if child_types[1] != child_types[2] or child_types[1] not in {"int", "float"}:
            return _fail("conditional branches must have one identical numeric type")
        if expr.result_type != child_types[1]:
            return _fail("conditional result type is inconsistent")
        return BodyAudit(True, child_types[1])

    return _fail(f"callable node kind {expr.kind!r} is outside the audited pandas grammar")


def audit_series_callable(meta: CallableMeta, receiver_type: str) -> BodyAudit:
    """Validate signature and every body node without trusting a native symbol."""
    input_type = "float" if receiver_type == SERIES_F64 else "int"
    if meta.arg_index != 0 or meta.keyword:
        return _fail("the mapper must be the sole positional callable")
    if len(meta.params) != 1:
        return _fail("the mapper must have exactly one parameter")
    param = meta.params[0]
    if param.param_type != input_type:
        return _fail(f"the mapper parameter must be annotated {input_type}")
    if meta.return_type not in ({"float"} if input_type == "float" else {"int", "float"}):
        return _fail("the mapper return annotation is outside the supported scalar matrix")
    if meta.runtime_semantics:
        return _fail("runtime-semantics callables cannot run in the pandas native loop")
    if not meta.body.available or meta.body.expression is None:
        return _fail(meta.body.unavailable_reason or "no closed callable body is available")

    audit = _audit_expr(meta.body.expression, input_type, param.name)
    if not audit.accepted:
        return audit
    if audit.result_type != meta.return_type:
        return _fail("the audited body type does not match the return annotation")
    return audit


def _audit_row_expr(
    expr: CallableBodyExpr,
    param_name: str,
    schema: SchemaMeta,
) -> BodyAudit:
    """Validate the row-UDF grammar and reconstruct every node's scalar type."""
    if expr.kind == "param":
        if expr.param_index != 0 or expr.name != param_name or expr.result_type != schema.identity:
            return _fail("row parameter metadata is inconsistent")
        return BodyAudit(True, "row")

    if expr.kind == "literal":
        literal = expr.literal
        if literal is None or literal.kind != "float" or expr.result_type != "float":
            return _fail("row UDF literals must be finite float literals")
        return BodyAudit(True, "float")

    children = tuple(_audit_row_expr(child, param_name, schema) for child in expr.children)
    failed = next((child for child in children if not child.accepted), None)
    if failed is not None:
        return failed
    child_types = tuple(child.result_type for child in children)

    if expr.kind == "subscript":
        if child_types != ("row",) or schema.field_type(expr.name) != "float":
            return _fail(f"row subscript {expr.name!r} is not a declared float64 field")
        if expr.result_type != "float":
            return _fail("row subscript result type is inconsistent")
        return BodyAudit(True, "float")

    if expr.kind == "field":
        return _fail('attribute row access is rejected; use row["field"]')

    if expr.kind == "unary":
        if expr.op != "-" or child_types != ("float",) or expr.result_type != "float":
            return _fail("only unary negation of float64 row values is audited")
        return BodyAudit(True, "float")

    if expr.kind == "compare":
        if not expr.ops or any(op not in _COMPARISONS for op in expr.ops):
            return _fail("identity, membership, and unaudited row comparisons are rejected")
        if any(child_type != "float" for child_type in child_types):
            return _fail("row comparisons require only float64 operands")
        if expr.result_type != "bool":
            return _fail("row comparison result type is inconsistent")
        return BodyAudit(True, "bool")

    if expr.kind == "boolop":
        if expr.op not in {"and", "or"} or any(kind != "bool" for kind in child_types):
            return _fail("row boolean composition requires boolean operands")
        if expr.result_type != "bool":
            return _fail("row boolean result type is inconsistent")
        return BodyAudit(True, "bool")

    if expr.kind == "cond":
        if child_types != ("bool", "float", "float") or expr.result_type != "float":
            return _fail("row conditionals require a bool test and two float64 branches")
        return BodyAudit(True, "float")

    return _fail(f"row callable node kind {expr.kind!r} is outside the audited pandas grammar")


def audit_frame_callable(meta: CallableMeta, schema: SchemaMeta) -> BodyAudit:
    """Validate the exact homogeneous-f64 row signature, schema, and body."""
    if not schema.fields:
        return _fail("the declared DataFrame schema must contain at least one field")
    if any(field.field_type != "float" for field in schema.fields):
        return _fail("the declared DataFrame schema must contain only float fields")
    if meta.arg_index != 0 or meta.keyword or len(meta.params) != 1:
        return _fail("the row UDF must be one positional project function with one parameter")
    param = meta.params[0]
    if param.param_type != schema.identity:
        return _fail("the unannotated row parameter must be bound to the receiver schema")
    if meta.return_type != "float":
        return _fail("the row UDF return must be annotated float")
    if meta.runtime_semantics:
        return _fail("runtime-semantics row callables cannot run in the native loop")
    if not meta.body.available or meta.body.expression is None:
        return _fail(meta.body.unavailable_reason or "no closed row body is available")
    audit = _audit_row_expr(meta.body.expression, param.name, schema)
    if not audit.accepted:
        return audit
    if audit.result_type != "float":
        return _fail("the audited row body does not return float")
    return audit


def _claim_series_map(site: ClaimSite) -> ClaimResult:
    receiver = site.receiver
    if receiver is None or receiver.arg_type not in SERIES_TYPES:
        return NotCovered()
    if site.target.rpartition(".")[2] != "map":
        return NotCovered()
    if receiver.expr_kind != "name" or not receiver.is_safe:
        return reject(
            site,
            DIAGNOSTIC_SHAPE,
            "the receiver must be a plain local or parameter name",
            "Bind the exact annotated Series to a plain name before calling series.map(udf).",
        )
    if len(site.operand_types) != 1 or site.keywords or len(site.callables) != 1:
        return reject(
            site,
            DIAGNOSTIC_SHAPE,
            "only series.map(udf) with one positional project-function reference is supported",
            "Remove na_action, keyword callable forms, and every extra argument.",
        )
    meta = site.callables[0]
    audit = audit_series_callable(meta, receiver.arg_type)
    if not audit.accepted:
        code = (
            DIAGNOSTIC_SIGNATURE
            if "parameter" in audit.reason or "annotation" in audit.reason
            else DIAGNOSTIC_BODY
        )
        return reject(
            site,
            code,
            audit.reason,
            "Use one statically resolved scalar UDF whose complete body is inside the documented audited subset.",
        )
    result_key = SERIES_F64 if audit.result_type == "float" else SERIES_I64
    return Claimed(rule_id=SERIES_MAP_RULE, result_type=result_key)


def _claim_dataframe_apply(site: ClaimSite) -> ClaimResult:
    receiver = site.receiver
    if receiver is None or receiver.arg_type != FRAME_F64:
        return NotCovered()
    if site.target.rpartition(".")[2] != "apply":
        return NotCovered()
    if receiver.expr_kind != "name" or not receiver.is_safe:
        return reject(
            site,
            DIAGNOSTIC_APPLY_SHAPE,
            "the DataFrame receiver must be a plain local or parameter name",
            "Bind the exact annotated DataFrame to a plain name before apply.",
        )
    if len(site.operand_types) != 1 or len(site.callables) != 1:
        return reject(
            site,
            DIAGNOSTIC_APPLY_SHAPE,
            "only frame.apply(udf, axis=1) with one positional callable is supported",
            "Pass one bare row UDF followed only by the literal keyword axis=1.",
        )
    if len(site.keywords) != 1 or site.keywords[0].name != "axis":
        return reject(
            site,
            DIAGNOSTIC_APPLY_SHAPE,
            "axis=1 must be the sole keyword",
            "Write exactly frame.apply(udf, axis=1); remove raw/result_type/args/kwargs.",
        )
    axis = site.keywords[0].literal
    if not axis.is_literal:
        # Dynamic keyword expressions and **kwargs are core-owned pre-offer
        # failures. If a malformed synthetic site reaches us, do not mislabel it
        # with a plugin diagnostic.
        return NotCovered()
    if isinstance(axis.value, bool) or axis.value != 1:
        return reject(
            site,
            DIAGNOSTIC_APPLY_SHAPE,
            "axis must be the non-bool integer literal 1",
            'Use axis=1 exactly; axis=True and axis="columns" are rejected.',
        )
    schema = receiver.schema
    if schema is None:
        return reject(
            site,
            DIAGNOSTIC_APPLY_SCHEMA,
            "DataFrameF64 must be parameterized by a valid declared schema",
            "Annotate the receiver as DataFrameF64[Row] where Row contains only float fields.",
        )
    audit = audit_frame_callable(site.callables[0], schema)
    if not audit.accepted:
        code = (
            DIAGNOSTIC_APPLY_SCHEMA
            if "schema" in audit.reason or "parameter" in audit.reason or "return" in audit.reason
            else DIAGNOSTIC_APPLY_BODY
        )
        return reject(
            site,
            code,
            audit.reason,
            'Use literal row["field"] reads and only comparisons, boolean composition, unary negation, and a float conditional.',
        )
    return Claimed(rule_id=DATAFRAME_APPLY_RULE, result_type=SERIES_F64)


def claim(site: ClaimSite, config: RextioConfig) -> ClaimResult:
    """Claim only the exact Series map surface; everything else stays with core."""
    del config
    if site.kind != "call":
        return NotCovered()
    result = _claim_series_map(site)
    if not isinstance(result, NotCovered):
        return result
    return _claim_dataframe_apply(site)


__all__ = [
    "BodyAudit",
    "DATAFRAME_APPLY_RULE",
    "SERIES_MAP_RULE",
    "audit_frame_callable",
    "audit_series_callable",
    "claim",
]
