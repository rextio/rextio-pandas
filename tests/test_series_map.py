from __future__ import annotations

from pathlib import Path
from dataclasses import replace
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
    ClaimLiteral,
    ClaimSite,
    KeywordArg,
    LoweringContext,
    ReceiverMeta,
    Rejected,
    ScalarLiteral,
)

from rextio_pandas.claim.map_apply import SERIES_MAP_RULE
from rextio_pandas.diagnostics import SERIES_BOOL, SERIES_F64, SERIES_I64
from rextio_pandas.plugin import RextioPandasPlugin
from rextio_pandas.rust_snippets.map_apply import boundary_helpers

from conftest import pandas_registry

PLUGIN = RextioPandasPlugin()
CONFIG = RextioConfig()


def param(result_type: str) -> CallableBodyExpr:
    return CallableBodyExpr(kind="param", param_index=0, name="value", result_type=result_type)


def literal(kind: str, value: int | float | bool) -> CallableBodyExpr:
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
        operand_literals=(ClaimLiteral(is_literal=False),),
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


def predicate_f64_body() -> CallableBodyExpr:
    """Return a Core-representable bool body using its complete approved grammar."""
    value = param("float")
    positive = CallableBodyExpr(
        kind="compare",
        ops=(">",),
        children=(value, literal("float", 0.0)),
        result_type="bool",
    )
    small = CallableBodyExpr(
        kind="compare",
        ops=("<",),
        children=(value, literal("float", 10.0)),
        result_type="bool",
    )
    combined = CallableBodyExpr(
        kind="boolop",
        op="and",
        children=(positive, small),
        result_type="bool",
    )
    return CallableBodyExpr(
        kind="cond",
        children=(
            CallableBodyExpr(
                kind="unary",
                op="not",
                children=(combined,),
                result_type="bool",
            ),
            literal("bool", False),
            literal("bool", True),
        ),
        result_type="bool",
    )


def invert_bool_body() -> CallableBodyExpr:
    """Return the bounded SeriesBool mapper body."""
    return CallableBodyExpr(
        kind="unary",
        op="not",
        children=(param("bool"),),
        result_type="bool",
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
    ("receiver_type", "input_type"),
    [(SERIES_F64, "float"), (SERIES_I64, "int")],
)
def test_claims_numeric_predicates_as_exact_bool_series(
    receiver_type: str, input_type: str
) -> None:
    if input_type == "int":
        value = param("int")
        body = CallableBodyExpr(
            kind="boolop",
            op="or",
            children=(
                CallableBodyExpr(
                    kind="compare",
                    ops=(">=",),
                    children=(value, literal("int", 0)),
                    result_type="bool",
                ),
                CallableBodyExpr(
                    kind="unary",
                    op="not",
                    children=(literal("bool", False),),
                    result_type="bool",
                ),
            ),
            result_type="bool",
        )
    else:
        body = predicate_f64_body()

    assert PLUGIN.claim(site(receiver_type, meta(body, input_type, "bool")), CONFIG) == Claimed(
        rule_id=SERIES_MAP_RULE,
        result_type=SERIES_BOOL,
    )


def test_claims_bounded_series_bool_to_bool_map() -> None:
    assert PLUGIN.claim(
        site(SERIES_BOOL, meta(invert_bool_body(), "bool", "bool")),
        CONFIG,
    ) == Claimed(rule_id=SERIES_MAP_RULE, result_type=SERIES_BOOL)

    ordering = CallableBodyExpr(
        kind="compare",
        ops=("<",),
        children=(param("bool"), literal("bool", True)),
        result_type="bool",
    )
    wrong_return = PLUGIN.claim(
        site(SERIES_BOOL, meta(param("bool"), "bool", "int")),
        CONFIG,
    )
    assert isinstance(wrong_return, Rejected)
    assert wrong_return.diagnostic.code == "RXTP-PANDAS-002"

    rejected_ordering = PLUGIN.claim(
        site(SERIES_BOOL, meta(ordering, "bool", "bool")),
        CONFIG,
    )
    assert isinstance(rejected_ordering, Rejected)
    assert rejected_ordering.diagnostic.code == "RXTP-PANDAS-003"


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


def test_claims_only_exact_literal_none_na_action() -> None:
    callable_meta = meta(param("float"), "float", "float")
    exact_none = KeywordArg(
        name="na_action",
        arg_type="None",
        literal=ClaimLiteral(is_literal=True, value=None),
    )

    assert PLUGIN.claim(site(SERIES_F64, callable_meta, keywords=(exact_none,)), CONFIG) == Claimed(
        rule_id=SERIES_MAP_RULE,
        result_type=SERIES_F64,
    )

    near_misses = (
        KeywordArg(
            name="na_action",
            arg_type="str",
            literal=ClaimLiteral(is_literal=True, value="ignore"),
        ),
        KeywordArg(
            name="na_action",
            arg_type="None",
            literal=ClaimLiteral(is_literal=False),
        ),
        KeywordArg(
            name="na_action",
            arg_type="str",
            literal=ClaimLiteral(is_literal=True, value=None),
        ),
        KeywordArg(
            name="arg",
            arg_type="None",
            literal=ClaimLiteral(is_literal=True, value=None),
        ),
    )
    for keyword in near_misses:
        result = PLUGIN.claim(site(SERIES_F64, callable_meta, keywords=(keyword,)), CONFIG)
        assert isinstance(result, Rejected)
        assert result.diagnostic.code == "RXTP-PANDAS-001"

    duplicate = PLUGIN.claim(
        site(SERIES_F64, callable_meta, keywords=(exact_none, exact_none)),
        CONFIG,
    )
    assert isinstance(duplicate, Rejected)
    assert duplicate.diagnostic.code == "RXTP-PANDAS-001"


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


def test_lower_series_bool_map_uses_the_existing_pure_bool_loop() -> None:
    callable_meta = meta(invert_bool_body(), "bool", "bool")
    claimed = replace(
        site(SERIES_BOOL, callable_meta),
        rule_id=SERIES_MAP_RULE,
        result_type=SERIES_BOOL,
    )
    context = LoweringContext(
        operands=("invert",),
        receiver="flags",
        target_language="rust",
        fresh_name=lambda prefix: f"{prefix}_0",
    )

    lowered = PLUGIN.lower(claimed, context)

    assert lowered.rust.endswith("(py, &flags)?")
    source = "\n".join(lowered.helpers)
    assert "input: &numpy::ndarray::Array1<bool>" in source
    assert "numpy::ndarray::Array1<bool>" in source
    assert "output.push((!value));" in source


def test_lower_keeps_api_13_contexts_without_a_backend_field_on_pyo3() -> None:
    callable_meta = meta(branchy_f64_body(), "float", "float")
    claimed = ClaimSite(
        **{
            **site(SERIES_F64, callable_meta).__dict__,
            "rule_id": SERIES_MAP_RULE,
            "result_type": SERIES_F64,
        }
    )

    lowered = PLUGIN.lower(
        claimed,
        SimpleNamespace(receiver="series", operands=("udf",), target_language="rust"),
    )

    assert lowered.rust.endswith("(py, &series)?")


@pytest.mark.parametrize("backend", ["standalone-rust", "rust-crate", "host-executable"])
def test_lower_rejects_non_pyo3_backends(backend: str) -> None:
    callable_meta = meta(branchy_f64_body(), "float", "float")
    claimed = ClaimSite(
        **{
            **site(SERIES_F64, callable_meta).__dict__,
            "rule_id": SERIES_MAP_RULE,
            "result_type": SERIES_F64,
        }
    )

    with pytest.raises(ValueError, match="PyO3 host-extension lowering"):
        PLUGIN.lower(
            claimed,
            SimpleNamespace(
                receiver="series",
                operands=("udf",),
                target_language="rust",
                backend=backend,
            ),
        )


def test_lower_rejects_forged_series_map_site_before_helper_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callable_meta = meta(branchy_f64_body(), "float", "float")
    claimed = replace(
        site(SERIES_F64, callable_meta),
        rule_id=SERIES_MAP_RULE,
        result_type=SERIES_F64,
    )
    context = LoweringContext(
        operands=("udf",),
        receiver="series",
        target_language="rust",
        fresh_name=lambda prefix: f"{prefix}_0",
    )
    forged_receiver = SimpleNamespace(
        arg_type=SERIES_F64,
        expr_kind="name",
        is_safe=False,
        schema=None,
    )
    forged_sites = (
        replace(claimed, kind="binop"),
        replace(claimed, target="map"),
        replace(claimed, target="series.apply"),
        replace(claimed, receiver=ReceiverMeta(SERIES_F64, "call", False)),
        replace(claimed, receiver=forged_receiver),  # type: ignore[arg-type]
        replace(claimed, receiver=ReceiverMeta("float", "name", True)),
        replace(claimed, operand_types=()),
        replace(claimed, operand_types=("float",)),
        replace(claimed, operand_literals=()),
        replace(claimed, operand_literals=(ClaimLiteral(is_literal=True, value=None),)),
        replace(
            claimed,
            keywords=(
                KeywordArg(
                    name="na_action",
                    arg_type="str",
                    literal=ClaimLiteral(is_literal=True, value="ignore"),
                ),
            ),
        ),
    )

    def unreachable(*args: object, **kwargs: object) -> object:
        raise AssertionError("forged lower metadata reached Series.map helper generation")

    monkeypatch.setattr("rextio_pandas.lower.map_apply.series_map_helpers", unreachable)
    for forged in forged_sites:
        with pytest.raises(ValueError, match="malformed Series.map lower metadata"):
            PLUGIN.lower(forged, context)


@pytest.mark.parametrize(
    "context",
    (
        SimpleNamespace(receiver=None, operands=("udf",), target_language="rust"),
        SimpleNamespace(receiver="series", operands=(), target_language="rust"),
        SimpleNamespace(receiver="series", operands=("udf", "extra"), target_language="rust"),
        SimpleNamespace(receiver="series", operands=("udf",), target_language="python"),
    ),
)
def test_lower_rejects_forged_series_map_context_before_helper_generation(
    context: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callable_meta = meta(branchy_f64_body(), "float", "float")
    claimed = replace(
        site(SERIES_F64, callable_meta),
        rule_id=SERIES_MAP_RULE,
        result_type=SERIES_F64,
    )

    def unreachable(*args: object, **kwargs: object) -> object:
        raise AssertionError("forged lower context reached Series.map helper generation")

    monkeypatch.setattr("rextio_pandas.lower.map_apply.series_map_helpers", unreachable)
    with pytest.raises(ValueError, match="malformed Series.map lower metadata"):
        PLUGIN.lower(claimed, context)  # type: ignore[arg-type]


def test_lower_accepts_only_exact_literal_none_na_action() -> None:
    callable_meta = meta(branchy_f64_body(), "float", "float")
    exact_none = KeywordArg(
        name="na_action",
        arg_type="None",
        literal=ClaimLiteral(is_literal=True, value=None),
    )
    claimed = replace(
        site(SERIES_F64, callable_meta, keywords=(exact_none,)),
        rule_id=SERIES_MAP_RULE,
        result_type=SERIES_F64,
    )
    context = LoweringContext(
        operands=("udf",),
        receiver="series",
        target_language="rust",
        fresh_name=lambda prefix: f"{prefix}_0",
    )

    lowered = PLUGIN.lower(claimed, context)

    assert lowered.rust.startswith("__rxtpd_map_series_")
    assert lowered.rust.endswith("(py, &series)?")


def test_series_map_materializer_reuses_extraction_time_class_validation() -> None:
    source = boundary_helpers()
    materializer = source[
        source.index("fn __rxtpd_materialize_series") : source.index(
            "fn __rxtpd_materialize_frame_f64"
        )
    ]

    assert materializer.count("__rxtpd_pinned_series_class") == 1
    assert materializer.index("let Some(source) = source else") < materializer.index(
        "__rxtpd_pinned_series_class"
    )


def test_series_boundary_derives_length_from_guarded_exact_numpy_array() -> None:
    source = boundary_helpers()
    boundary = source[
        source.index("fn __rxtpd_series_parts") : source.index(
            "fn __rxtpd_extract_series_f64"
        )
    ]

    assert "value.len()" not in boundary
    assert boundary.index("__rxtpd_is_exact_numpy_dtype") < boundary.index(
        'let to_numpy = series_class.getattr("to_numpy")?'
    )
    assert boundary.index(".cast_into::<numpy::PyArray1<T>>()") < boundary.index(
        "let length = typed.readonly().as_array().len()"
    )
    assert boundary.index("if length == 0") < boundary.index("if stop != length as isize")


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
    """Exercise return-position collection independently at source generation.

    Core intentionally rejects an ordinary claimless materialized alias return
    and claimless calls of materialized plugin functions (RXT092). Runtime
    Series return materialization is therefore covered by the identity-map
    product regression; this IR probe isolates return-position helper
    collection without bypassing either core guard.
    """
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


def test_analyzer_composes_two_and_four_stage_maps_without_intermediate_materialization(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        """
from rextio_pandas.types import SeriesBool, SeriesF64

def scale(value: float) -> float:
    return value * 2.0

def shift(value: float) -> float:
    return value + 1.0

def predicate(value: float) -> bool:
    return (value > 0.0 and not value == 7.0) or False

def invert(value: bool) -> bool:
    return not value

def two_stage(series: SeriesF64) -> SeriesBool:
    scaled = series.map(scale)
    return scaled.map(predicate)

def four_stage(series: SeriesF64) -> SeriesBool:
    first = series.map(scale)
    second = first.map(shift)
    third = second.map(scale)
    return third.map(predicate)

def bool_stage(series: SeriesBool) -> SeriesBool:
    return series.map(invert)

def predicate_then_invert(series: SeriesF64) -> SeriesBool:
    flags = series.map(predicate)
    return flags.map(invert)
""",
    )
    registry = pandas_registry()
    analysis = analyze_project(
        tmp_path,
        active_plugins=registry.active,
        plugin_registry=registry,
        plugin_config=CONFIG,
    )

    for name, stages in (("app.kernels.two_stage", 2), ("app.kernels.four_stage", 4)):
        function = _function(analysis, name)
        assert function.route == "native-plugin:rextio-pandas"
        assert function.accepted is True
        assert len(function.plugin_claims) == stages
        assert function.plugin_claims[-1].result_type == SERIES_BOOL

    bool_stage = _function(analysis, "app.kernels.bool_stage")
    assert bool_stage.route == "native-plugin:rextio-pandas"
    assert bool_stage.accepted is True
    assert len(bool_stage.plugin_claims) == 1
    assert bool_stage.plugin_claims[0].receiver.arg_type == SERIES_BOOL
    assert bool_stage.plugin_claims[0].result_type == SERIES_BOOL

    predicate_then_invert = _function(analysis, "app.kernels.predicate_then_invert")
    assert predicate_then_invert.route == "native-plugin:rextio-pandas"
    assert predicate_then_invert.accepted is True
    assert len(predicate_then_invert.plugin_claims) == 2
    assert predicate_then_invert.plugin_claims[1].receiver.arg_type == SERIES_BOOL
    assert predicate_then_invert.plugin_claims[1].result_type == SERIES_BOOL

    source = _generated_source(analysis, registry)
    for name, stages in (("app__kernels__two_stage", 2), ("app__kernels__four_stage", 4)):
        function_source = source[source.index(f"fn {name}") :]
        next_function = function_source.find("\n#[pyfunction]", 1)
        if next_function != -1:
            function_source = function_source[:next_function]
        assert function_source.count("__rxtpd_extract_series_f64(py, &series)?") == 1
        assert function_source.count("__rxtpd_map_series_") == stages
        assert function_source.count("__rxtpd_materialize_series(py,") == 1

    bool_source = source[source.index("fn app__kernels__bool_stage") :]
    bool_next = bool_source.find("\n#[pyfunction]", 1)
    if bool_next != -1:
        bool_source = bool_source[:bool_next]
    assert bool_source.count("__rxtpd_extract_series_bool(py, &series)?") == 1
    assert bool_source.count("__rxtpd_map_series_") == 1
    assert bool_source.count("__rxtpd_materialize_series(py,") == 1

    chained_source = source[source.index("fn app__kernels__predicate_then_invert") :]
    chained_next = chained_source.find("\n#[pyfunction]", 1)
    if chained_next != -1:
        chained_source = chained_source[:chained_next]
    assert chained_source.count("__rxtpd_extract_series_f64(py, &series)?") == 1
    assert chained_source.count("__rxtpd_map_series_") == 2
    assert chained_source.count("__rxtpd_materialize_series(py,") == 1


def test_analyzer_accepts_only_literal_none_na_action_with_plugin_code(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        """
from rextio_pandas.types import SeriesF64

def identity(value: float) -> float:
    return value

def keyword(series: SeriesF64) -> SeriesF64:
    return series.map(arg=identity)

def explicit_none(series: SeriesF64) -> SeriesF64:
    return series.map(identity, na_action=None)

def ignore(series: SeriesF64) -> SeriesF64:
    return series.map(identity, na_action="ignore")

def dynamic(series: SeriesF64) -> SeriesF64:
    action = None
    return series.map(identity, na_action=action)
""",
    )
    registry = pandas_registry()
    analysis = analyze_project(
        tmp_path,
        active_plugins=registry.active,
        plugin_registry=registry,
        plugin_config=CONFIG,
    )

    explicit_none = _function(analysis, "app.kernels.explicit_none")
    assert explicit_none.route == "native-plugin:rextio-pandas"
    assert explicit_none.accepted is True
    assert len(explicit_none.plugin_claims) == 1
    keyword = explicit_none.plugin_claims[0].keywords[0]
    assert keyword.name == "na_action"
    assert keyword.arg_type == "None"
    assert keyword.literal.is_literal is True
    assert keyword.literal.value is None

    for qualname in ("app.kernels.keyword", "app.kernels.ignore"):
        function = _function(analysis, qualname)
        assert function.route == "fallback-python"
        assert any(d.code == "RXTP-PANDAS-001" for d in function.diagnostics)

    dynamic = _function(analysis, "app.kernels.dynamic")
    assert dynamic.route == "fallback-python"
    assert dynamic.plugin_claims == []
