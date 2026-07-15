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
    _APPLY_CLASS_AUTHORITIES,
    _APPLY_CLASS_DIGESTS,
    _APPLY_FUNCTION_AUTHORITIES,
    _APPLY_FUNCTION_DIGESTS,
    _AUTHORITY_CODE_DIGESTS,
    _AUTHORITY_DEFAULTS,
    _AUTHORITY_GLOBALS,
    _rust_string,
    boundary_helpers,
    compute_authority_class_digest,
    compute_authority_code_digest,
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


def test_boundary_uses_type_tagged_digest_not_repr() -> None:
    source = boundary_helpers()
    # The frozen authority code digests are embedded and compared to a digest the
    # generated Rust rebuilds from exact PyO3 type checks (no Python repr/hashlib).
    for constant, key in (
        ("__RXTPD_SERIES_MAP_DIGEST", "series_map"),
        ("__RXTPD_SERIES_TO_NUMPY_DIGEST", "series_to_numpy"),
        ("__RXTPD_FRAME_APPLY_DIGEST", "frame_apply"),
        ("__RXTPD_FRAME_TO_NUMPY_DIGEST", "frame_to_numpy"),
        ("__RXTPD_APPLY_FRAME_APPLY_DIGEST", "apply_frame_apply"),
    ):
        assert f'const {constant}: &str = "{_AUTHORITY_CODE_DIGESTS[key]}"' in source
        assert f"!= {constant}" in source
    # No user-controlled serialization reaches the authority check.
    assert ".repr()" not in source
    assert 'py.import("hashlib")' not in source
    assert "fn __rxtpd_method_fingerprint(" not in source
    # Type-tagged canonical encoder with exact type checks, hashed by the crate.
    assert "fn __rxtpd_encode_value(" in source
    assert "fn __rxtpd_code_digest(" in source
    assert "sha2::Sha256::new()" in source
    assert "is_exact_instance_of::<PyBool>()" in source
    assert "is_exact_instance_of::<PyInt>()" in source
    # Defaults validated structurally (never through the digest).
    assert "fn __rxtpd_validate_series_to_numpy(" in source
    assert "no_default" in source
    assert 'py.import("pandas._libs.lib")' in source


def test_import_target_is_validated_against_a_frozen_authority_not_metadata() -> None:
    # The DataFrame.apply import target ``pandas.core.apply.frame_apply`` must be
    # checked by a frozen-authority validator (exact code digest + structural
    # defaults/globals), not accepted on metadata alone.
    source = boundary_helpers()
    assert "fn __rxtpd_validate_apply_frame_apply(" in source
    # DataFrame.apply's import_from check delegates to that validator.
    assert 'let target = imported.getattr("frame_apply")' in source
    assert "__rxtpd_validate_apply_frame_apply(py, &target)?;" in source
    # The old metadata-only acceptance of the import target (compare __module__
    # and __qualname__ on ``target`` then trust it) must be gone.
    assert 'target.getattr("__module__")' not in source
    assert 'target.getattr("__qualname__")' not in source
    # The frozen code digest is what actually guards the imported function.
    validator = source[source.index("fn __rxtpd_validate_apply_frame_apply(") :]
    validator = validator[: validator.index("\n}\n") + 3]
    assert "!= __RXTPD_APPLY_FRAME_APPLY_DIGEST" in validator
    assert '__closure__")?.is_none()' in validator
    assert '__kwdefaults__")?.is_none()' in validator


def test_metadata_and_globals_matching_frame_apply_replacement_is_caught_by_digest() -> None:
    # The precise blocker: a replacement built with types.FunctionType that has
    # the exact function type, __module__, __qualname__ and the pinned
    # pandas.core.apply globals dict -- everything the previous weak import check
    # verified -- but a DIFFERENT code object. Only the frozen code digest
    # separates it from the canonical authority.
    import types

    import pandas.core.apply as apply_mod

    original = apply_mod.frame_apply

    def impostor(
        obj: object,
        func: object,
        axis: object = 0,
        raw: bool = False,
        result_type: object = None,
        by_row: str = "compat",
        engine: str = "python",
        engine_kwargs: object = None,
        args: object = None,
        kwargs: object = None,
    ) -> object:
        raise RuntimeError("impostor frame_apply executed")

    forged = types.FunctionType(
        impostor.__code__,
        apply_mod.__dict__,
        "frame_apply",
        original.__defaults__,
        original.__closure__,
    )
    forged.__module__ = "pandas.core.apply"
    forged.__qualname__ = "frame_apply"

    # Everything the previous metadata-only check verified matches exactly.
    assert type(forged) is type(original)
    assert forged.__module__ == original.__module__ == "pandas.core.apply"
    assert forged.__qualname__ == original.__qualname__ == "frame_apply"
    assert forged.__globals__ is apply_mod.__dict__
    assert forged.__closure__ is None and forged.__kwdefaults__ is None
    assert forged.__defaults__ == original.__defaults__
    # The frozen code-digest authority is what rejects the forgery.
    assert forged.__code__ is not original.__code__
    assert compute_authority_code_digest(forged) != _AUTHORITY_CODE_DIGESTS["apply_frame_apply"]
    assert compute_authority_code_digest(original) == _AUTHORITY_CODE_DIGESTS["apply_frame_apply"]


def _live_authority_functions() -> dict[str, object]:
    import pandas as pd
    import pandas.core.apply as apply_mod

    return {
        "series_map": pd.Series.__dict__["map"],
        "series_to_numpy": pd.Series.to_numpy,
        "frame_apply": pd.DataFrame.__dict__["apply"],
        "frame_to_numpy": pd.DataFrame.__dict__["to_numpy"],
        # DataFrame.apply imports this module-level function at call time; it is a
        # frozen authority in its own right, not trusted via the live binding.
        "apply_frame_apply": apply_mod.frame_apply,
    }


def test_authority_code_digests_match_pinned_pandas() -> None:
    # The frozen constants must equal the live pinned pandas so drift is caught
    # loudly, not silently trusted.
    live = _live_authority_functions()
    assert set(live) == set(_AUTHORITY_CODE_DIGESTS)
    for key, function in live.items():
        assert compute_authority_code_digest(function) == _AUTHORITY_CODE_DIGESTS[key]


def test_custom_repr_default_does_not_forge_the_code_digest() -> None:
    # The reproduced attack: a replacement map with the original code object and a
    # non-None default whose custom __repr__ returns "None". The digest covers
    # only the code object, so it is unchanged; the forgery is caught by the
    # separate structural default check, not the digest.
    import types

    import pandas as pd

    class FakeNone:
        def __repr__(self) -> str:
            return "None"

    original = pd.Series.__dict__["map"]
    forged = types.FunctionType(
        original.__code__,
        original.__globals__,
        original.__name__,
        (FakeNone(),),
        original.__closure__,
    )
    # Same code object -> identical code digest (repr plays no part).
    assert compute_authority_code_digest(forged) == _AUTHORITY_CODE_DIGESTS["series_map"]
    # But the runtime default spec requires the actual None singleton.
    assert _AUTHORITY_DEFAULTS["series_map"] == (("none",),)
    assert forged.__defaults__[0] is not None


def test_global_binding_specs_match_live_bytecode() -> None:
    # A drift that added an unchecked LOAD_GLOBAL must fail this test rather than
    # silently escape the runtime binding checks.
    import dis

    live = _live_authority_functions()
    assert set(live) == set(_AUTHORITY_GLOBALS)
    for key, function in live.items():
        load_globals = {
            ins.argval for ins in dis.get_instructions(function) if ins.opname == "LOAD_GLOBAL"
        }
        spec = _AUTHORITY_GLOBALS[key]
        covered = (
            set(spec["builtins"])
            | {name for name, _ in spec["modules"]}
            | {name for name, _, _ in spec["module_attrs"]}
            | {name for name, _ in spec.get("module_authorities", ())}
        )
        assert load_globals == covered, (key, load_globals, covered)
        imports = [
            ins.argval for ins in dis.get_instructions(function) if ins.opname == "IMPORT_NAME"
        ]
        assert imports == [mod for mod, _ in spec["import_from"]], key


def _live_apply_module_authorities() -> dict[str, object]:
    import pandas.core.apply as apply_mod

    live: dict[str, object] = {}
    for key, spec in _APPLY_CLASS_AUTHORITIES.items():
        live[key] = getattr(apply_mod, spec["attr"])
    for key, spec in _APPLY_FUNCTION_AUTHORITIES.items():
        live[key] = getattr(apply_mod, spec["attr"])
    return live


def test_apply_module_authorities_are_independently_frozen_not_self_compared() -> None:
    # The three ``pandas.core.apply`` members that ``frame_apply`` loads
    # (FrameColumnApply/FrameRowApply for axis dispatch, reconstruct_func) must be
    # validated against frozen authorities, not re-imported from the same mutable
    # module and compared to themselves.
    source = boundary_helpers()
    # Independent validators are generated and embedded.
    for fn in (
        "fn __rxtpd_validate_frame_column_apply(",
        "fn __rxtpd_validate_frame_row_apply(",
        "fn __rxtpd_validate_reconstruct_func(",
    ):
        assert fn in source
    # apply_frame_apply's globals check delegates to those validators on the live
    # bound member, never a self-comparison to a re-import of pandas.core.apply.
    for name, key in (
        ("FrameColumnApply", "frame_column_apply"),
        ("FrameRowApply", "frame_row_apply"),
        ("reconstruct_func", "reconstruct_func"),
    ):
        assert f'let bound = module_dict.get_item("{name}")' in source
        assert f"__rxtpd_validate_{key}(py, &bound)?;" in source
    # The old same-module self-comparison (import pandas.core.apply then getattr
    # the same attribute and compare identity) must be gone for all three.
    assert 'py.import("pandas.core.apply")?.getattr("FrameColumnApply")' not in source
    assert 'py.import("pandas.core.apply")?.getattr("FrameRowApply")' not in source
    assert 'py.import("pandas.core.apply")?.getattr("reconstruct_func")' not in source
    # The frozen authority digests actually guard the members.
    assert "const __RXTPD_RECONSTRUCT_FUNC_DIGEST: &str =" in source
    assert "const __RXTPD_FRAME_COLUMN_APPLY_CLASS_DIGEST: &str =" in source
    assert "const __RXTPD_FRAME_ROW_APPLY_CLASS_DIGEST: &str =" in source
    assert f'"{_APPLY_FUNCTION_DIGESTS["reconstruct_func"]}"' in source
    assert f'"{_APPLY_CLASS_DIGESTS["frame_column_apply"]}"' in source
    assert f'"{_APPLY_CLASS_DIGESTS["frame_row_apply"]}"' in source
    # The class authority folds in real executable behavior (method code digests
    # via the shared canonical encoder), the axis selector, and the MRO chain --
    # not a mere module/qualname check.
    column_validator = source[source.index("fn __rxtpd_validate_frame_column_apply(") :]
    column_validator = column_validator[: column_validator.index("\n}\n") + 3]
    assert "__rxtpd_code_digest(&code, error)?" in column_validator
    assert 'class_dict.get_item("axis")' in column_validator
    assert 'object.getattr("__mro__")' in column_validator
    assert "!= __RXTPD_FRAME_COLUMN_APPLY_CLASS_DIGEST" in column_validator


def test_apply_module_authority_digests_match_pinned_pandas() -> None:
    # Frozen class/function authority digests must equal the live pinned pandas so
    # a version bump fails loudly rather than silently trusting a self-comparison.
    live = _live_apply_module_authorities()
    assert set(live) == set(_APPLY_CLASS_AUTHORITIES) | set(_APPLY_FUNCTION_AUTHORITIES)
    for key, spec in _APPLY_CLASS_AUTHORITIES.items():
        assert compute_authority_class_digest(live[key], spec) == _APPLY_CLASS_DIGESTS[key]
    for key in _APPLY_FUNCTION_AUTHORITIES:
        assert compute_authority_code_digest(live[key]) == _APPLY_FUNCTION_DIGESTS[key]


def _class_digest_or_none(cls: object, spec: dict) -> str | None:
    # Mirrors the generated Rust: a missing/wrong-typed member fails the check
    # (``get_item`` error -> rejection) rather than yielding a valid digest.
    try:
        return compute_authority_class_digest(cls, spec)
    except (TypeError, KeyError):
        return None


def test_same_module_class_replacement_is_rejected_by_the_frozen_class_digest() -> None:
    # The reproduced defect at the authority layer. The old check compared the
    # module attribute back to a re-import of the same module, so ANY replacement
    # passed. The frozen structural digest instead distinguishes classes by axis
    # selector + method code, so a same-module replacement (here modelled by the
    # sibling apply class, whose ``axis`` and method bodies differ) is rejected.
    import types

    import pandas.core.apply as apply_mod

    for key, spec in _APPLY_CLASS_AUTHORITIES.items():
        original = getattr(apply_mod, spec["attr"])
        assert compute_authority_class_digest(original, spec) == _APPLY_CLASS_DIGESTS[key]

    column = _APPLY_CLASS_AUTHORITIES["frame_column_apply"]
    row = _APPLY_CLASS_AUTHORITIES["frame_row_apply"]
    # Cross-class replacement (FrameRowApply where FrameColumnApply is expected).
    assert (
        compute_authority_class_digest(apply_mod.FrameRowApply, column)
        != _APPLY_CLASS_DIGESTS["frame_column_apply"]
    )
    assert (
        compute_authority_class_digest(apply_mod.FrameColumnApply, row)
        != _APPLY_CLASS_DIGESTS["frame_row_apply"]
    )
    # A same-module subclass override (inherits the covered members, so they are
    # absent from its own ``__dict__``) is rejected, not silently accepted.
    tampered = types.new_class("FrameColumnApply", (apply_mod.FrameColumnApply,))
    tampered.__qualname__ = "FrameColumnApply"
    assert "axis" not in tampered.__dict__
    assert _class_digest_or_none(tampered, column) != _APPLY_CLASS_DIGESTS["frame_column_apply"]


def test_same_module_reconstruct_func_replacement_changes_the_code_digest() -> None:
    # Replacing ``pandas.core.apply.reconstruct_func`` with a same-module function
    # of matching metadata but a different code object changes the frozen code
    # digest that now guards it.
    import types

    import pandas.core.apply as apply_mod

    original = apply_mod.reconstruct_func
    assert compute_authority_code_digest(original) == _APPLY_FUNCTION_DIGESTS["reconstruct_func"]

    def impostor(func, **kwargs):  # pragma: no cover - never executed
        raise RuntimeError("impostor reconstruct_func executed")

    forged = types.FunctionType(
        impostor.__code__,
        apply_mod.__dict__,
        "reconstruct_func",
        original.__defaults__,
        original.__closure__,
    )
    forged.__module__ = "pandas.core.apply"
    forged.__qualname__ = "reconstruct_func"
    assert forged.__globals__ is apply_mod.__dict__
    assert compute_authority_code_digest(forged) != _APPLY_FUNCTION_DIGESTS["reconstruct_func"]


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
