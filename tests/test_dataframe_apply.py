from __future__ import annotations

from pathlib import Path

import pytest

from rextio.analyzer.project_scanner import analyze_project
from rextio.config.schema import RextioConfig
from rextio.plugins.api import (
    CallableBody,
    CallableBodyExpr,
    CallableMeta,
    CallableParam,
    Claimed,
    ClaimLiteral,
    ClaimSite,
    KeywordArg,
    LoweringContext,
    NotCovered,
    ReceiverMeta,
    Rejected,
    ScalarLiteral,
    SchemaField,
    SchemaMeta,
)

from rextio_pandas.claim.map_apply import DATAFRAME_APPLY_RULE
from rextio_pandas.diagnostics import FRAME_F64, SERIES_F64
from rextio_pandas.plugin import RextioPandasPlugin
from rextio_pandas.rust_snippets.map_apply import (
    _AUTHORITY_FINGERPRINTS,
    _rust_string,
    boundary_helpers,
    compute_authority_fingerprint,
    dataframe_apply_helpers,
)

from conftest import pandas_registry

PLUGIN = RextioPandasPlugin()
CONFIG = RextioConfig()


def row_param() -> CallableBodyExpr:
    return CallableBodyExpr(kind="param", param_index=0, name="row", result_type="app.Row")


def field(name: str) -> CallableBodyExpr:
    return CallableBodyExpr(
        kind="subscript",
        name=name,
        children=(row_param(),),
        result_type="float",
    )


def float_literal(value: float) -> CallableBodyExpr:
    return CallableBodyExpr(
        kind="literal",
        literal=ScalarLiteral("float", value),
        result_type="float",
    )


def branch_body() -> CallableBodyExpr:
    test = CallableBodyExpr(
        kind="compare",
        ops=(">=",),
        children=(field("left"), float_literal(0.0)),
        result_type="bool",
    )
    negative = CallableBodyExpr(
        kind="unary",
        op="-",
        children=(field("right"),),
        result_type="float",
    )
    return CallableBodyExpr(
        kind="cond",
        children=(test, field("left"), negative),
        result_type="float",
    )


def row_meta(body: CallableBodyExpr) -> CallableMeta:
    return CallableMeta(
        arg_index=0,
        qualname="app.choose",
        params=(CallableParam("row", "app.Row"),),
        return_type="float",
        body=CallableBody(available=True, expression=body),
    )


def schema(*types: str) -> SchemaMeta:
    names = ("left", "right", "third")
    return SchemaMeta(
        "app.Row",
        tuple(SchemaField(names[index], field_type) for index, field_type in enumerate(types)),
    )


def axis(value: bool | int | str) -> KeywordArg:
    value_type = "bool" if isinstance(value, bool) else "int" if isinstance(value, int) else "str"
    return KeywordArg(
        name="axis",
        arg_type=value_type,
        literal=ClaimLiteral(is_literal=True, value=value),
    )


def apply_site(
    callable_meta: CallableMeta,
    *,
    declared_schema: SchemaMeta | None = None,
    keywords: tuple[KeywordArg, ...] = (axis(1),),
    operand_types: tuple[str | None, ...] = (None,),
) -> ClaimSite:
    return ClaimSite(
        kind="call",
        target="frame.apply",
        operand_types=operand_types,
        file_path="app.py",
        line=1,
        column=0,
        receiver=ReceiverMeta(
            arg_type=FRAME_F64,
            expr_kind="name",
            is_safe=True,
            schema=declared_schema if declared_schema is not None else schema("float", "float"),
        ),
        callables=(callable_meta,),
        keywords=keywords,
    )


def test_claims_exact_homogeneous_f64_axis1_row_udf() -> None:
    result = PLUGIN.claim(apply_site(row_meta(branch_body())), CONFIG)
    assert result == Claimed(rule_id=DATAFRAME_APPLY_RULE, result_type=SERIES_F64)


@pytest.mark.parametrize("axis_value", [0, -1, True, "columns"])
def test_rejects_every_axis_spelling_except_non_bool_integer_one(
    axis_value: bool | int | str,
) -> None:
    result = PLUGIN.claim(
        apply_site(row_meta(branch_body()), keywords=(axis(axis_value),)),
        CONFIG,
    )
    assert isinstance(result, Rejected)
    assert result.diagnostic.code == "RXTP-PANDAS-011"


def test_dynamic_axis_is_left_to_core_without_plugin_diagnostic() -> None:
    dynamic = KeywordArg(name="axis", arg_type="int", literal=ClaimLiteral())
    result = PLUGIN.claim(
        apply_site(row_meta(branch_body()), keywords=(dynamic,)),
        CONFIG,
    )
    assert result == NotCovered()


def test_rejects_omitted_positional_and_extra_apply_arguments() -> None:
    callable_meta = row_meta(branch_body())
    omitted = PLUGIN.claim(apply_site(callable_meta, keywords=()), CONFIG)
    positional = PLUGIN.claim(
        apply_site(callable_meta, keywords=(), operand_types=(None, "int")), CONFIG
    )
    raw = KeywordArg(
        name="raw",
        arg_type="bool",
        literal=ClaimLiteral(is_literal=True, value=False),
    )
    extra = PLUGIN.claim(
        apply_site(callable_meta, keywords=(axis(1), raw)),
        CONFIG,
    )
    for result in (omitted, positional, extra):
        assert isinstance(result, Rejected)
        assert result.diagnostic.code == "RXTP-PANDAS-011"


def test_rejects_missing_mixed_and_empty_schema() -> None:
    callable_meta = row_meta(branch_body())
    missing_site = apply_site(callable_meta)
    missing_site = ClaimSite(
        **{
            **missing_site.__dict__,
            "receiver": ReceiverMeta(
                arg_type=FRAME_F64,
                expr_kind="name",
                is_safe=True,
                schema=None,
            ),
        }
    )
    results = (
        PLUGIN.claim(missing_site, CONFIG),
        PLUGIN.claim(apply_site(callable_meta, declared_schema=schema("float", "int")), CONFIG),
        PLUGIN.claim(
            apply_site(callable_meta, declared_schema=SchemaMeta("app.Empty", ())), CONFIG
        ),
    )
    for result in results:
        assert isinstance(result, Rejected)
        assert result.diagnostic.code == "RXTP-PANDAS-012"


@pytest.mark.parametrize(
    "body",
    [
        CallableBodyExpr(
            kind="binop",
            op="+",
            children=(field("left"), field("right")),
            result_type="float",
        ),
        CallableBodyExpr(
            kind="call",
            target="abs",
            children=(field("left"),),
            result_type="float",
        ),
        CallableBodyExpr(
            kind="field",
            name="left",
            children=(row_param(),),
            result_type="float",
        ),
        CallableBodyExpr(
            kind="subscript",
            name="missing",
            children=(row_param(),),
            result_type="float",
        ),
    ],
)
def test_rejects_unaudited_row_bodies(body: CallableBodyExpr) -> None:
    result = PLUGIN.claim(apply_site(row_meta(body)), CONFIG)
    assert isinstance(result, Rejected)
    assert result.diagnostic.code == "RXTP-PANDAS-013"


def test_lower_is_schema_hashed_deterministic_and_pure() -> None:
    callable_meta = row_meta(branch_body())
    claimed = apply_site(callable_meta)
    claimed = ClaimSite(
        **{
            **claimed.__dict__,
            "rule_id": DATAFRAME_APPLY_RULE,
            "result_type": SERIES_F64,
        }
    )
    context = LoweringContext(
        operands=("choose",),
        receiver="frame",
        target_language="rust",
        fresh_name=lambda prefix: f"{prefix}_0",
    )

    first = PLUGIN.lower(claimed, context)
    second = PLUGIN.lower(claimed, context)

    assert first == second
    assert first.rust.startswith("__rxtpd_apply_frame_")
    source = "\n".join(first.helpers)
    assert 'const EXPECTED_COLUMNS: &[&str] = &["left", "right"]' in source
    assert "py.detach(|| __rxtpd_apply_values_" in source
    hot = source[source.index("fn __rxtpd_apply_values_") : source.index("fn __rxtpd_apply_frame_")]
    assert "for row in input.outer_iter()" in hot
    assert "row[0]" in hot and "row[1]" in hot
    assert "PyObject" not in hot
    assert "Python::attach" not in hot
    assert "Python::with_gil" not in hot
    assert ".call" not in hot
    assert "+" not in hot


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("left", '"left"'),
        ("값", r'"\u{ac12}"'),
        ('a"b\\c', r'"a\"b\\c"'),
        ("\n\t\r\x00", r'"\n\t\r\0"'),
        ("\x1f", r'"\u{1f}"'),
        ("\U0001f600", r'"\u{1f600}"'),
    ],
)
def test_rust_string_encoder_emits_valid_rust_literals(source: str, expected: str) -> None:
    encoded = _rust_string(source)
    assert encoded == expected
    # JSON's ``\uXXXX`` (no braces) is not a valid Rust escape and must not leak.
    assert "\\u" not in encoded or "\\u{" in encoded


def test_unicode_schema_field_lowers_to_braced_rust_unicode_escape() -> None:
    schema_meta = SchemaMeta("app.Row", (SchemaField("값", "float"),))
    body = CallableBodyExpr(
        kind="subscript",
        name="값",
        children=(
            CallableBodyExpr(kind="param", param_index=0, name="row", result_type="app.Row"),
        ),
        result_type="float",
    )
    meta = CallableMeta(
        arg_index=0,
        qualname="app.udf",
        params=(CallableParam("row", "app.Row"),),
        return_type="float",
        body=CallableBody(available=True, expression=body),
    )
    _, helpers = dataframe_apply_helpers(schema_meta, meta)
    source = "\n".join(helpers)
    assert r'EXPECTED_COLUMNS: &[&str] = &["\u{ac12}"]' in source
    # No bare JSON-style ``\uXXXX`` escapes survive into the Rust source.
    import re

    assert re.search(r"\\u(?!\{)", source) is None


def test_boundary_uses_immutable_semantic_fingerprint() -> None:
    source = boundary_helpers()
    # The frozen authority fingerprints are embedded as string constants and
    # compared against a freshly computed live digest at runtime.
    for constant, key in (
        ("__RXTPD_SERIES_MAP_FP", "series_map"),
        ("__RXTPD_SERIES_TO_NUMPY_FP", "series_to_numpy"),
        ("__RXTPD_FRAME_APPLY_FP", "frame_apply"),
        ("__RXTPD_FRAME_TO_NUMPY_FP", "frame_to_numpy"),
    ):
        assert f'const {constant}: &str = "{_AUTHORITY_FINGERPRINTS[key]}"' in source
        assert f"!= {constant}" in source
    assert "fn __rxtpd_method_fingerprint(" in source
    # Full semantic coverage, not just bytecode: defaults, kwdefaults, globals
    # provenance, closure, constants/names, exception table.
    assert 'descriptor.getattr("__defaults__")?' in source
    assert 'descriptor.getattr("__kwdefaults__")?' in source
    assert 'descriptor.getattr("__globals__")?.is(&module_dict)' in source
    assert 'descriptor.getattr("__closure__")?.is_none()' in source
    assert 'code.getattr("co_consts")?' in source
    assert 'code.getattr("co_exceptiontable")?' in source
    # Fingerprint is captured from a known-good authority, never regenerated from
    # a possibly-patched live descriptor at lowering time.
    assert "co_code.as_slice() != trusted_co_code" not in source
    # A hardcoded Python-minor gate documents the fail-closed version pin.
    assert "co_flags" in source
    for module_name in (
        "pandas.core.series",
        "pandas.core.base",
        "pandas.core.frame",
    ):
        assert f'"{module_name}"' in source


def test_authority_fingerprints_match_pinned_pandas() -> None:
    # The frozen constants must equal the live pinned pandas so drift is caught
    # loudly, not silently trusted.
    import pandas as pd

    live = {
        "series_map": (pd.Series.__dict__["map"], "pandas.core.series"),
        "series_to_numpy": (pd.Series.to_numpy, "pandas.core.base"),
        "frame_apply": (pd.DataFrame.__dict__["apply"], "pandas.core.frame"),
        "frame_to_numpy": (pd.DataFrame.__dict__["to_numpy"], "pandas.core.frame"),
    }
    for key, (function, module_name) in live.items():
        assert compute_authority_fingerprint(function, module_name) == _AUTHORITY_FINGERPRINTS[key]


def test_changed_defaults_or_globals_change_the_fingerprint() -> None:
    import types

    import pandas as pd

    original = pd.Series.__dict__["map"]
    base = compute_authority_fingerprint(original, "pandas.core.series")

    changed_defaults = types.FunctionType(
        original.__code__,
        original.__globals__,
        original.__name__,
        ("ignore",),  # na_action default flipped from None
        original.__closure__,
    )
    changed_defaults.__qualname__ = original.__qualname__
    changed_defaults.__module__ = original.__module__
    assert compute_authority_fingerprint(changed_defaults, "pandas.core.series") != base

    foreign_globals = types.FunctionType(
        original.__code__,
        dict(original.__globals__),  # a caller-supplied copy, not the module dict
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    with pytest.raises(TypeError):
        compute_authority_fingerprint(foreign_globals, "pandas.core.series")


def test_boundary_rejects_extension_dtypes_before_conversion() -> None:
    source = boundary_helpers()
    assert "fn __rxtpd_is_exact_numpy_dtype" in source
    assert "if !dtype.is_instance(&numpy_dtype_type)?" in source
    # The Series extraction gates dtype before ``to_numpy`` for both routes.
    assert '__rxtpd_series_parts(py, value, "float64"' in source
    assert '__rxtpd_series_parts(py, value, "int64"' in source
    assert "if !__rxtpd_is_exact_numpy_dtype(py, &dtype, expected_dtype)?" in source
    # The frame extraction checks every column dtype with the same instance gate.
    assert '__rxtpd_is_exact_numpy_dtype(py, &item, "float64")?' in source


def _write_module(root: Path, source: str) -> None:
    module = root / "src" / "app" / "kernels.py"
    module.parent.mkdir(parents=True)
    (module.parent / "__init__.py").write_text("", encoding="utf-8")
    module.write_text(source, encoding="utf-8")


def _function(analysis: object, name: str):
    for module in analysis.modules:  # type: ignore[attr-defined]
        for function in module.functions:
            if function.qualname == name:
                return function
    raise AssertionError(name)


def test_analyzer_routes_exact_schema_bound_apply(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        """
from rextio_pandas.types import DataFrameF64, SeriesF64

class Row:
    left: float
    right: float

def choose(row) -> float:
    return row["left"] if row["left"] >= 0.0 else -row["right"]

def run(frame: DataFrameF64[Row]) -> SeriesF64:
    return frame.apply(choose, axis=1)
""",
    )
    registry = pandas_registry()
    analysis = analyze_project(
        tmp_path,
        active_plugins=registry.active,
        plugin_registry=registry,
        plugin_config=CONFIG,
    )
    run = _function(analysis, "app.kernels.run")
    assert run.route == "native-plugin:rextio-pandas"
    assert run.accepted is True
    claim = run.plugin_claims[0]
    assert claim.rule_id == DATAFRAME_APPLY_RULE
    assert claim.receiver.schema.identity == "app.kernels.Row"
    assert [(item.name, item.field_type) for item in claim.receiver.schema.fields] == [
        ("left", "float"),
        ("right", "float"),
    ]
    assert claim.callables[0].accepts_native is False
    assert claim.callables[0].body.available is True


@pytest.mark.parametrize(
    ("function", "call", "expected_code"),
    [
        ("omitted", "frame.apply(choose)", "RXTP-PANDAS-011"),
        ("positional", "frame.apply(choose, 1)", "RXTP-PANDAS-011"),
        ("bool_axis", "frame.apply(choose, axis=True)", "RXTP-PANDAS-011"),
        ("string_axis", 'frame.apply(choose, axis="columns")', "RXTP-PANDAS-011"),
        ("raw", "frame.apply(choose, axis=1, raw=False)", "RXTP-PANDAS-011"),
        ("raw_true", "frame.apply(choose, axis=1, raw=True)", "RXTP-PANDAS-011"),
        (
            "keyword_callable",
            "frame.apply(func=choose, axis=1)",
            "RXTP-PANDAS-011",
        ),
        (
            "result_type",
            "frame.apply(choose, axis=1, result_type=None)",
            "RXTP-PANDAS-011",
        ),
    ],
)
def test_analyzer_rejects_static_apply_shapes_with_plugin_codes(
    tmp_path: Path,
    function: str,
    call: str,
    expected_code: str,
) -> None:
    _write_module(
        tmp_path,
        f"""
from rextio_pandas.types import DataFrameF64, SeriesF64

class Row:
    left: float

def choose(row) -> float:
    return row["left"]

def {function}(frame: DataFrameF64[Row]) -> SeriesF64:
    return {call}
""",
    )
    registry = pandas_registry()
    analysis = analyze_project(
        tmp_path,
        active_plugins=registry.active,
        plugin_registry=registry,
        plugin_config=CONFIG,
    )
    analyzed = _function(analysis, f"app.kernels.{function}")
    assert analyzed.route == "fallback-python"
    assert any(item.code == expected_code for item in analyzed.diagnostics)


def test_analyzer_rejects_mixed_schema_and_row_arithmetic(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        """
from rextio_pandas.types import DataFrameF64, SeriesF64

class Mixed:
    left: float
    count: int

class FloatRow:
    left: float
    right: float

def mixed_udf(row) -> float:
    return row["left"]

def add_udf(row) -> float:
    return row["left"] + row["right"]

def mixed(frame: DataFrameF64[Mixed]) -> SeriesF64:
    return frame.apply(mixed_udf, axis=1)

def arithmetic(frame: DataFrameF64[FloatRow]) -> SeriesF64:
    return frame.apply(add_udf, axis=1)
""",
    )
    registry = pandas_registry()
    analysis = analyze_project(
        tmp_path,
        active_plugins=registry.active,
        plugin_registry=registry,
        plugin_config=CONFIG,
    )
    mixed = _function(analysis, "app.kernels.mixed")
    arithmetic = _function(analysis, "app.kernels.arithmetic")
    assert any(item.code == "RXTP-PANDAS-012" for item in mixed.diagnostics)
    assert any(item.code == "RXTP-PANDAS-013" for item in arithmetic.diagnostics)
    assert mixed.route == arithmetic.route == "fallback-python"


def test_analyzer_dynamic_axis_remains_core_owned(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        """
from rextio_pandas.types import DataFrameF64, SeriesF64

class Row:
    value: float

def choose(row) -> float:
    return row["value"]

def dynamic(frame: DataFrameF64[Row], axis: int) -> SeriesF64:
    return frame.apply(choose, axis=axis)
""",
    )
    registry = pandas_registry()
    analysis = analyze_project(
        tmp_path,
        active_plugins=registry.active,
        plugin_registry=registry,
        plugin_config=CONFIG,
    )
    dynamic = _function(analysis, "app.kernels.dynamic")
    assert dynamic.route == "fallback-python"
    assert not any(item.code.startswith("RXTP-PANDAS-") for item in dynamic.diagnostics)
