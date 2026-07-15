from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from rextio.analyzer.project_scanner import analyze_project
from rextio.build.orchestrator import _plugin_lowering_inputs
from rextio.codegen.rust.generator import generate_rust_module
from rextio.config.schema import RextioConfig
from rextio.ir.lowering import lower_project
from rextio.ir.nodes import BlockIR, FunctionIR, ModuleIR
from rextio.plugins.api import (
    CallableBody,
    CallableBodyExpr,
    CallableMeta,
    CallableParam,
    Claimed,
    ClaimSite,
    LoweringContext,
    ReceiverMeta,
    Rejected,
    ScalarLiteral,
)

from rextio_pandas.claim.map_apply import SERIES_MAP_RULE
from rextio_pandas.diagnostics import SERIES_F64, SERIES_I64
from rextio_pandas.plugin import RextioPandasPlugin
from rextio_pandas.rust_snippets.map_apply import boundary_helpers

from conftest import pandas_registry

PLUGIN = RextioPandasPlugin()
CONFIG = RextioConfig()


def param(result_type: str) -> CallableBodyExpr:
    return CallableBodyExpr(kind="param", param_index=0, name="value", result_type=result_type)


def literal(kind: str, value: int | float) -> CallableBodyExpr:
    return CallableBodyExpr(
        kind="literal",
        literal=ScalarLiteral(kind, value),
        result_type=kind,
    )


def meta(body: CallableBodyExpr, input_type: str, return_type: str) -> CallableMeta:
    return CallableMeta(
        arg_index=0,
        qualname="app.udf",
        params=(CallableParam("value", input_type),),
        return_type=return_type,
        body=CallableBody(available=True, expression=body),
    )


def site(
    receiver_type: str,
    callable_meta: CallableMeta,
    *,
    target: str = "series.map",
    operand_types: tuple[str | None, ...] = (None,),
    keywords: tuple[object, ...] = (),
) -> ClaimSite:
    return ClaimSite(
        kind="call",
        target=target,
        operand_types=operand_types,
        file_path="app.py",
        line=1,
        column=0,
        receiver=ReceiverMeta(arg_type=receiver_type, expr_kind="name", is_safe=True),
        callables=(callable_meta,),
        keywords=keywords,  # type: ignore[arg-type]
    )


def branchy_f64_body() -> CallableBodyExpr:
    value = param("float")
    test = CallableBodyExpr(
        kind="compare",
        ops=(">",),
        children=(value, literal("float", 0.0)),
        result_type="bool",
    )
    positive = CallableBodyExpr(
        kind="binop",
        op="*",
        children=(value, literal("float", 2.0)),
        result_type="float",
    )
    negative = CallableBodyExpr(
        kind="unary",
        op="-",
        children=(value,),
        result_type="float",
    )
    return CallableBodyExpr(
        kind="cond",
        children=(test, positive, negative),
        result_type="float",
    )


def test_claims_branchy_f64_and_i64_identity() -> None:
    assert PLUGIN.claim(
        site(SERIES_F64, meta(branchy_f64_body(), "float", "float")), CONFIG
    ) == Claimed(
        rule_id=SERIES_MAP_RULE,
        result_type=SERIES_F64,
    )
    assert PLUGIN.claim(site(SERIES_I64, meta(param("int"), "int", "int")), CONFIG) == Claimed(
        rule_id=SERIES_MAP_RULE,
        result_type=SERIES_I64,
    )


def test_claims_full_domain_safe_i64_to_f64_conditional() -> None:
    value = param("int")
    test = CallableBodyExpr(
        kind="compare",
        ops=(">=",),
        children=(value, literal("int", 0)),
        result_type="bool",
    )
    body = CallableBodyExpr(
        kind="cond",
        children=(test, literal("float", 1.5), literal("float", 2.5)),
        result_type="float",
    )
    result = PLUGIN.claim(site(SERIES_I64, meta(body, "int", "float")), CONFIG)
    assert result == Claimed(rule_id=SERIES_MAP_RULE, result_type=SERIES_F64)


@pytest.mark.parametrize(
    "body",
    [
        CallableBodyExpr(
            kind="binop",
            op="/",
            children=(param("float"), literal("float", 2.0)),
            result_type="float",
        ),
        CallableBodyExpr(
            kind="unary",
            op="-",
            children=(param("int"),),
            result_type="int",
        ),
        CallableBodyExpr(
            kind="call",
            target="math.sqrt",
            children=(param("float"),),
            result_type="float",
        ),
        CallableBodyExpr(
            kind="cond",
            children=(
                CallableBodyExpr(
                    kind="compare",
                    ops=("is",),
                    children=(param("float"), literal("float", 0.0)),
                    result_type="bool",
                ),
                literal("float", 1.0),
                literal("float", 2.0),
            ),
            result_type="float",
        ),
    ],
)
def test_rejects_unaudited_body_even_if_callable_is_native(body: CallableBodyExpr) -> None:
    input_type = "int" if body.children and body.children[0].result_type == "int" else "float"
    receiver_type = SERIES_I64 if input_type == "int" else SERIES_F64
    callable_meta = meta(body, input_type, body.result_type or input_type)
    callable_meta = CallableMeta(
        **{
            **callable_meta.__dict__,
            "accepts_native": True,
            "native_symbol": "app_udf",
        }
    )

    result = PLUGIN.claim(site(receiver_type, callable_meta), CONFIG)

    assert isinstance(result, Rejected)
    assert result.diagnostic.code == "RXTP-PANDAS-003"


def test_rejects_wrong_signature_and_shape() -> None:
    callable_meta = meta(param("float"), "float", "float")
    wrong_signature = CallableMeta(
        arg_index=0,
        qualname="app.udf",
        params=(CallableParam("value", "int"),),
        return_type="float",
        body=callable_meta.body,
    )
    result = PLUGIN.claim(site(SERIES_F64, wrong_signature), CONFIG)
    assert isinstance(result, Rejected)
    assert result.diagnostic.code == "RXTP-PANDAS-002"

    shaped = site(SERIES_F64, callable_meta, operand_types=(None, "str"))
    result = PLUGIN.claim(shaped, CONFIG)
    assert isinstance(result, Rejected)
    assert result.diagnostic.code == "RXTP-PANDAS-001"


def test_lower_is_deterministic_and_contains_pure_detached_hot_loop() -> None:
    callable_meta = meta(branchy_f64_body(), "float", "float")
    claimed = site(SERIES_F64, callable_meta)
    claimed = ClaimSite(
        **{
            **claimed.__dict__,
            "rule_id": SERIES_MAP_RULE,
            "result_type": SERIES_F64,
        }
    )
    context = LoweringContext(
        operands=("udf",),
        receiver="series",
        target_language="rust",
        fresh_name=lambda prefix: f"{prefix}_0",
    )

    first = PLUGIN.lower(claimed, context)
    second = PLUGIN.lower(claimed, context)

    assert first == second
    assert first.rust.startswith("__rxtpd_map_series_") and first.rust.endswith("(py, &series)?")
    source = "\n".join(first.helpers)
    assert "py.detach(|| __rxtpd_map_values_" in source
    hot = source[source.index("fn __rxtpd_map_values_") : source.index("fn __rxtpd_map_series_")]
    assert "for &value in input.iter()" in hot
    assert "PyObject" not in hot
    assert "Python::attach" not in hot
    assert "Python::with_gil" not in hot
    assert ".call" not in hot
    assert "if " in hot


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


def _generated_source(analysis: object, registry: object) -> str:
    type_maps, providers, types_by_key = _plugin_lowering_inputs(
        SimpleNamespace(plugins=registry)  # type: ignore[arg-type]
    )
    assert type_maps is not None and providers is not None and types_by_key is not None
    module_ir = lower_project(analysis, plugin_types=type_maps)  # type: ignore[arg-type]
    return generate_rust_module(
        module_ir,
        plugin_providers=providers,
        plugin_types_by_key=types_by_key,
    )


def test_claimless_parameter_only_signature_emits_boundary_support(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        """
from rextio_pandas.types import SeriesF64

def inspect(series: SeriesF64) -> float:
    return 1.0
""",
    )
    registry = pandas_registry()
    analysis = analyze_project(
        tmp_path,
        active_plugins=registry.active,
        plugin_registry=registry,
        plugin_config=CONFIG,
    )
    inspect = _function(analysis, "app.kernels.inspect")
    assert inspect.accepted is True
    assert inspect.plugin_claims == []

    source = _generated_source(analysis, registry)
    assert source.count(boundary_helpers()) == 1
    assert "let series = __rxtpd_extract_series_f64(py, &series)?;" in source
    assert "fn __rxtpd_map_values_" not in source


def test_return_only_signature_source_collects_boundary_support() -> None:
    """Exercise return-position collection independently of a parameter."""
    registry = pandas_registry()
    _maps, _providers, types_by_key = _plugin_lowering_inputs(
        SimpleNamespace(plugins=registry)  # type: ignore[arg-type]
    )
    assert types_by_key is not None
    series = types_by_key[SERIES_F64]
    function = FunctionIR(
        name="make_series",
        qualname="app.kernels.make_series",
        module_name="app.kernels",
        params=[],
        return_type=series,
        body=BlockIR(statements=[]),
        plugin_lowered=True,
    )
    source = generate_rust_module(ModuleIR(functions=[function]))
    assert source.count(boundary_helpers()) == 1
    assert "fn app__kernels__make_series" in source


def test_analyzer_routes_exact_series_map_through_plugin(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        """
from rextio_pandas.types import SeriesF64

def branch(value: float) -> float:
    return value * 2.0 if value > 0.0 else -value

def run(series: SeriesF64) -> SeriesF64:
    return series.map(branch)
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
    assert len(run.plugin_claims) == 1
    claim = run.plugin_claims[0]
    assert claim.rule_id == SERIES_MAP_RULE
    assert claim.receiver.arg_type == SERIES_F64
    assert claim.callables[0].qualname == "app.kernels.branch"
    assert claim.callables[0].body.available is True

    # The signature and LoweredExpr both contribute the same boundary helper;
    # API 1.3 exact-text dedup emits it once alongside the map-specific helper.
    source = _generated_source(analysis, registry)
    assert source.count(boundary_helpers()) == 1
    assert source.count("fn __rxtpd_map_values_") == 1


def test_analyzer_rejects_keyword_mapper_and_na_action_with_plugin_code(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        """
from rextio_pandas.types import SeriesF64

def identity(value: float) -> float:
    return value

def keyword(series: SeriesF64) -> SeriesF64:
    return series.map(arg=identity)

def na_action(series: SeriesF64) -> SeriesF64:
    return series.map(identity, na_action="ignore")
""",
    )
    registry = pandas_registry()
    analysis = analyze_project(
        tmp_path,
        active_plugins=registry.active,
        plugin_registry=registry,
        plugin_config=CONFIG,
    )

    for qualname in ("app.kernels.keyword", "app.kernels.na_action"):
        function = _function(analysis, qualname)
        assert function.route == "fallback-python"
        assert any(d.code == "RXTP-PANDAS-001" for d in function.diagnostics)
