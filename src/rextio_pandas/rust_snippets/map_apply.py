"""Rust helper generation for pandas boundaries and native map/apply loops."""

from __future__ import annotations

import hashlib
import json
from typing import TypedDict

from rextio.plugins.api import CallableBodyExpr, CallableMeta, SchemaMeta

from rextio_pandas.diagnostics import RUNTIME_ERRORS, SERIES_F64, SERIES_I64

_STRUCT = {
    SERIES_F64: "RxtPandasSeriesF64",
    SERIES_I64: "RxtPandasSeriesI64",
}
_RUST_SCALAR = {SERIES_F64: "f64", SERIES_I64: "i64"}

PINNED_PANDAS_VERSION = "2.3.3"
PINNED_NUMPY_VERSION = "2.3.5"
PINNED_PYTHON = (3, 11)


class _FrozenTypeShape(TypedDict):
    qualname: str
    flags_mask: int
    flags: int
    basicsize: int
    itemsize: int
    dictoffset: int
    weakrefoffset: int


class _CythonFunctionTypeAuthority(_FrozenTypeShape):
    module: str
    metatype: _FrozenTypeShape


# Frozen CPython-visible structure of Cython 3.1.4's callable type. The live
# module binding is still checked by exact identity, but these immutable type
# fields prevent a coordinated replacement from supplying a Python class and
# then validating an instance against that same mutable replacement.
_CYTHON_FUNCTION_TYPE_AUTHORITY: _CythonFunctionTypeAuthority = {
    "module": "_cython_3_1_4",
    "qualname": "cython_function_or_method",
    # Py_TPFLAGS_VALID_VERSION_TAG (bit 19) is a mutable CPython cache bit;
    # exclude it while freezing every other low 32-bit structural flag.
    "flags_mask": 4294443007,
    "flags": 155392,
    "basicsize": 176,
    "itemsize": 0,
    "dictoffset": 64,
    "weakrefoffset": 40,
    "metatype": {
        "qualname": "_common_types_metatype",
        "flags_mask": 4294443007,
        "flags": 2147507072,
        "basicsize": 904,
        "itemsize": 40,
        "dictoffset": 264,
        "weakrefoffset": 368,
    },
}

# The Rust-side SHA-256 of the canonical code encoding uses the ``sha2`` crate
# already pinned (``sha2 = "0.10"``) in the core-generated Cargo manifest, so it
# is not re-declared as a plugin crate dependency (core reserves core crates).

# Frozen known-good SHA-256 of the *canonical, type-tagged* encoding of each
# trusted pandas method's code object (CPython 3.11 / pandas 2.3.3 / numpy
# 2.3.5). Unlike the previous approach these are NOT computed through Python
# ``repr`` (which a replacement default with a custom ``__repr__`` could forge);
# the generated Rust rebuilds the identical byte sequence from exact PyO3 type
# checks and hashes it with the pinned ``sha2`` crate. The digest covers only the
# executable code semantics (bytecode, constants, names/varnames/freevars/
# cellvars, arg counts, flags, stack size, exception table). Function defaults,
# kwdefaults, closure, module/qualname/type, and the execution-relevant global
# bindings are validated separately and structurally. These constants are frozen
# and never regenerated from a live descriptor at lowering; a test asserts they
# still match the pinned pandas.
_AUTHORITY_CODE_DIGESTS = {
    "series_map": "4f755e3868aee6be8e0d29b39354354d8829da3c0a00c4c1bc48dab3e83f9c4d",
    "series_map_values": "b8265ba6e7641007bf34d4a1bc005a687748b75d9c75e525244ae8e475124be2",
    "series_algorithms_map_array": (
        "e3d53c86735480faa62792dc2b7c375f55462360e4981295dbeeed347820e691"
    ),
    "series_lib_map_infer": "12bb606c8f6f3fe73f566bd28a6c75483e86e72ae6ec9bd93922bedbc6218ace",
    "series_constructor_fget": ("06a862891fe5589f33a76a07ce2fdf79ae9e00d57b4345ec64ed0a2d81dfaa9f"),
    "series_ndframe_finalize": ("4f7eca0c8a608086cc5729aa28b4976774c655ca8d4a49d54c41c7ccaa04b0b1"),
    "series_to_numpy": "1d2a907fafe5072adc69dde5099e4a83b48102e68e9d35e0aafbec8d16e70050",
    "frame_apply": "82441419a1a610c5b2e81af0c6978974682fb3116e8f10de5b9313641fec00d1",
    "frame_to_numpy": "ff8ef88dbed7a3c530f339452059b2aae33e65e73c3f58ae60f01c4469bb5e71",
    # ``DataFrame.apply`` imports ``pandas.core.apply.frame_apply`` at call time.
    # That imported function is a trusted authority too, so its code object is
    # frozen exactly like the four bound methods above (never re-derived from the
    # live, mutable ``pandas.core.apply.frame_apply`` binding at lowering).
    "apply_frame_apply": "38fa716dab42aa69bddb5dfc11c58f9b7c5b64e8e2f97dd2a5f5630cbfb168c2",
}
# CPython optimization mode changes executable code only where the pinned
# function contains removable assertions. Keep each supported mode bound to a
# distinct frozen digest; never accept the optimized digest in normal mode.
_AUTHORITY_OPTIMIZED_CODE_DIGESTS = {
    "series_ndframe_finalize": {
        1: "a2e90d16b2fede73b47f546d8213c73ce16e8b94234192e9887bb3f3a4f85ab3",
    },
}
_AUTHORITY_META = {
    "series_map": {"module": "pandas.core.series", "qualname": "Series.map"},
    "series_map_values": {
        "module": "pandas.core.base",
        "qualname": "IndexOpsMixin._map_values",
    },
    "series_algorithms_map_array": {
        "module": "pandas.core.algorithms",
        "qualname": "map_array",
    },
    "series_lib_map_infer": {
        "module": "pandas._libs.lib",
        "qualname": "map_infer",
        "function_type_module": "_cython_3_1_4",
        "function_type_qualname": "cython_function_or_method",
    },
    "series_constructor_fget": {
        "module": "pandas.core.series",
        "qualname": "Series._constructor",
    },
    "series_ndframe_finalize": {
        "module": "pandas.core.generic",
        "qualname": "NDFrame.__finalize__",
    },
    "series_to_numpy": {"module": "pandas.core.base", "qualname": "IndexOpsMixin.to_numpy"},
    "frame_apply": {"module": "pandas.core.frame", "qualname": "DataFrame.apply"},
    "frame_to_numpy": {"module": "pandas.core.frame", "qualname": "DataFrame.to_numpy"},
    "apply_frame_apply": {"module": "pandas.core.apply", "qualname": "frame_apply"},
}
# Structural default specs, validated item-by-item with exact type/identity
# checks. ``no_default`` is the exact ``pandas._libs.lib.no_default`` singleton.
_AUTHORITY_DEFAULTS = {
    "series_map": (("none",),),
    "series_map_values": (("none",), ("bool", True)),
    "series_algorithms_map_array": (("none",), ("bool", True)),
    "series_lib_map_infer": (("bool", True), ("bool", False)),
    "series_constructor_fget": None,
    "series_ndframe_finalize": (("none",),),
    "series_to_numpy": (("none",), ("bool", False), ("no_default",)),
    "frame_apply": (
        ("int", 0),
        ("bool", False),
        ("none",),
        ("empty_tuple",),
        ("str", "compat"),
        ("str", "python"),
        ("none",),
    ),
    "frame_to_numpy": (("none",), ("bool", False), ("no_default",)),
    # frame_apply(obj, func, axis=0, raw=False, result_type=None, by_row="compat",
    #             engine="python", engine_kwargs=None, args=None, kwargs=None)
    "apply_frame_apply": (
        ("int", 0),
        ("bool", False),
        ("none",),
        ("str", "compat"),
        ("str", "python"),
        ("none",),
        ("none",),
        ("none",),
    ),
}
# Execution-relevant global bindings, derived from each pinned method's
# ``LOAD_GLOBAL``/``IMPORT_NAME`` set. Every binding must be identical to the
# canonical authority object resolved from its independently named module (or
# builtins). A test re-derives these from the live bytecode so drift cannot add
# an unchecked global.
_AUTHORITY_GLOBALS: dict[str, dict[str, tuple]] = {
    "series_map": {"builtins": (), "modules": (), "module_attrs": (), "import_from": ()},
    "series_map_values": {
        "builtins": ("isinstance",),
        "modules": (("algorithms", "pandas.core.algorithms"),),
        "module_attrs": (("ExtensionArray", "pandas.core.arrays.base", "ExtensionArray"),),
        "import_from": (),
    },
    "series_algorithms_map_array": {
        "builtins": ("ValueError", "isinstance", "dict", "hasattr", "len", "object"),
        "modules": (("np", "numpy"), ("lib", "pandas._libs.lib")),
        "module_attrs": (
            ("is_dict_like", "pandas.core.dtypes.inference", "is_dict_like"),
            ("ABCSeries", "pandas.core.dtypes.generic", "ABCSeries"),
            ("take_nd", "pandas.core.array_algos.take", "take_nd"),
            ("isna", "pandas.core.dtypes.missing", "isna"),
        ),
        # ``from pandas import Series`` is inside the dict-like-mapper branch.
        # The claimed mapper is a plain callable, so this import is bytecode
        # drift-accounted but deliberately outside the executed authority graph.
        "unreachable_import_from": (("pandas", "Series"),),
        "import_from": (),
    },
    "series_lib_map_infer": {
        "builtins": (),
        "modules": (),
        "module_attrs": (),
        "import_from": (),
    },
    "series_constructor_fget": {
        "builtins": (),
        "modules": (),
        "module_attrs": (("Series", "pandas", "Series"),),
        "import_from": (),
    },
    "series_ndframe_finalize": {
        "builtins": ("isinstance", "set", "str", "object", "getattr", "all"),
        "modules": (),
        "module_attrs": (
            ("NDFrame", "pandas.core.series", "NDFrame"),
            ("deepcopy", "copy", "deepcopy"),
        ),
        "import_from": (),
    },
    "series_to_numpy": {
        "builtins": ("isinstance", "next", "iter", "TypeError"),
        "modules": (("np", "numpy"), ("lib", "pandas._libs.lib")),
        "module_attrs": (
            ("ExtensionDtype", "pandas.core.dtypes.base", "ExtensionDtype"),
            ("can_hold_element", "pandas.core.dtypes.cast", "can_hold_element"),
            ("isna", "pandas.core.dtypes.missing", "isna"),
            ("using_copy_on_write", "pandas._config", "using_copy_on_write"),
        ),
        "import_from": (),
    },
    "frame_apply": {
        "builtins": (),
        "modules": (),
        "module_attrs": (),
        "import_from": (("pandas.core.apply", "frame_apply"),),
    },
    "frame_to_numpy": {
        "builtins": (),
        "modules": (("np", "numpy"),),
        "module_attrs": (),
        "import_from": (),
    },
    # The imported frame_apply function loads only these three canonical
    # pandas.core.apply members; the drift test re-derives them from live
    # bytecode so an added LOAD_GLOBAL cannot escape the identity checks. Unlike
    # ``module_attrs`` (whose canonical authority lives in an independently named
    # module, e.g. ``pandas.core.dtypes.base`` for ``series_to_numpy``), these
    # three members live in ``pandas.core.apply`` itself. Re-importing that same
    # module and comparing identity would be a self-comparison that any
    # module-attribute replacement trivially passes, so each is instead validated
    # against an independently *frozen* authority (see ``module_authorities``).
    "apply_frame_apply": {
        "builtins": (),
        "modules": (),
        "module_attrs": (),
        "module_authorities": (
            ("FrameColumnApply", "frame_column_apply"),
            ("FrameRowApply", "frame_row_apply"),
            ("reconstruct_func", "reconstruct_func"),
        ),
        "import_from": (),
    },
}
_NO_DEFAULT_MODULE = "pandas._libs.lib"
_NO_DEFAULT_ATTR = "no_default"

# Independently frozen authorities for the three ``pandas.core.apply`` members
# that ``frame_apply`` loads. ``reconstruct_func`` is a plain module-level
# function, validated by a full frozen code-digest function authority.
# ``FrameColumnApply``/``FrameRowApply`` are classes, validated by a frozen
# type-tagged structural digest over the live class identity, MRO chain, the
# ``axis`` selector, and the code of the property getters + plain methods that
# drove the private ``DataFrame.apply(axis=1)`` prototype (``axis`` is 1 for
# ``FrameColumnApply`` and 0 for ``FrameRowApply``). None of these is derived
# from the live mutable ``pandas.core.apply`` binding at lowering or runtime; the
# expected digests are frozen constants, and drift tests re-derive them from the
# pinned package so a version bump fails loudly rather than silently trusting.
_APPLY_CLASS_AUTHORITIES: dict[str, dict] = {
    "frame_column_apply": {
        "attr": "FrameColumnApply",
        "int_attrs": ("axis",),
        "property_methods": ("result_columns", "result_index", "series_generator"),
        "function_methods": ("wrap_results_for_axis",),
    },
    "frame_row_apply": {
        "attr": "FrameRowApply",
        "int_attrs": ("axis",),
        "property_methods": ("result_columns", "result_index", "series_generator"),
        "function_methods": ("wrap_results_for_axis",),
    },
}
_APPLY_FUNCTION_AUTHORITIES: dict[str, dict[str, str]] = {
    "reconstruct_func": {
        "attr": "reconstruct_func",
        "module": "pandas.core.apply",
        "qualname": "reconstruct_func",
    },
}
# Frozen known-good digests (CPython 3.11 / pandas 2.3.3). Class digests are the
# structural class-authority digest; the function digest is the canonical code
# digest. Captured from the pinned package and asserted by drift tests.
_APPLY_CLASS_DIGESTS = {
    "frame_column_apply": "e95b5a9a8d4beeabdcc25d3240bc73b49d2d420b90ac9522fc75655764782f7f",
    "frame_row_apply": "a3e58d688739cc726a601e62dd34f46cff28b1e57c731756096bd2421907826c",
}
_APPLY_FUNCTION_DIGESTS = {
    "reconstruct_func": "08c2f2b0acd3ae10826d7ef7cb654735fd8ace9ed4d9d09b78cf45e014eba27a",
}
_CLASS_DIGEST_CONST = {
    "frame_column_apply": "__RXTPD_FRAME_COLUMN_APPLY_CLASS_DIGEST",
    "frame_row_apply": "__RXTPD_FRAME_ROW_APPLY_CLASS_DIGEST",
}
_FUNCTION_DIGEST_CONST = {
    "reconstruct_func": "__RXTPD_RECONSTRUCT_FUNC_DIGEST",
}


def _encode_value(value: object, out: list[bytes]) -> None:
    """Type-tagged, length-delimited encoding of one allowed const value.

    Mirrors the generated Rust encoder exactly; accepts only the builtin const
    types present in the pinned code objects (including nested code objects from
    comprehensions/lambdas, encoded recursively) and rejects everything else.
    """
    import struct
    import types
    from typing import cast

    value_type = type(value)
    if value is None:
        out.append(b"N")
    elif value_type is bool:
        out.append(b"T" if value else b"F")
    elif value_type is int:
        out.append(b"i" + cast(int, value).to_bytes(8, "little", signed=True))
    elif value_type is str:
        encoded = cast(str, value).encode("utf-8")
        out.append(b"s" + struct.pack("<I", len(encoded)) + encoded)
    elif value_type is tuple:
        items = cast("tuple[object, ...]", value)
        out.append(b"t" + struct.pack("<I", len(items)))
        for item in items:
            _encode_value(item, out)
    elif value_type is types.CodeType:
        # Nested code objects (comprehensions, nested defs) are folded into the
        # digest recursively so a replacement cannot hide behavior inside them.
        out.append(b"c")
        _encode_code(cast("types.CodeType", value), out)
    else:
        raise TypeError(f"disallowed const type {value_type!r}")


def _encode_code(code: object, out: list[bytes]) -> None:
    """Type-tagged canonical encoding of one code object's executable fields."""
    import struct

    for name in (
        "co_argcount",
        "co_posonlyargcount",
        "co_kwonlyargcount",
        "co_nlocals",
        "co_flags",
        "co_stacksize",
    ):
        out.append(b"i" + int(getattr(code, name)).to_bytes(8, "little", signed=True))
    for name in ("co_code", "co_exceptiontable"):
        raw = getattr(code, name)
        out.append(b"b" + struct.pack("<I", len(raw)) + raw)
    for name in ("co_names", "co_varnames", "co_freevars", "co_cellvars"):
        tpl = getattr(code, name)
        out.append(b"t" + struct.pack("<I", len(tpl)))
        for item in tpl:
            if type(item) is not str:
                raise TypeError(f"{name} contains a non-str entry")
            encoded = item.encode("utf-8")
            out.append(b"s" + struct.pack("<I", len(encoded)) + encoded)
    for name in ("co_name", "co_qualname"):
        encoded = getattr(code, name).encode("utf-8")
        out.append(b"s" + struct.pack("<I", len(encoded)) + encoded)
    out.append(b"t" + struct.pack("<I", len(code.co_consts)))  # type: ignore[attr-defined]
    for const in code.co_consts:  # type: ignore[attr-defined]
        _encode_value(const, out)


def compute_authority_code_digest(function: object) -> str:
    """Python mirror of the Rust canonical code digest (for capture + tests)."""
    import types

    code = getattr(function, "__code__", None)
    if not isinstance(code, types.CodeType):
        raise TypeError("pinned pandas callable does not expose an exact code object")
    out: list[bytes] = []
    _encode_code(code, out)
    return hashlib.sha256(b"".join(out)).hexdigest()


def compute_authority_class_digest(cls: object, spec: dict) -> str:
    """Python mirror of the Rust frozen class-authority digest.

    Folds the live class identity, its full MRO chain, the named integer class
    attributes (the axis selector), and the code digests of the named property
    getters and plain-function methods into a single type-tagged SHA-256. Because
    every value is read from the *live* class, any replacement that changes the
    class name/module, its base hierarchy, the axis constant, or the body of any
    covered method yields a different digest than the frozen authority constant.
    """
    import struct
    import types

    if not isinstance(cls, type):
        raise TypeError("pinned pandas authority is not a class")

    def _push_str(text: str) -> None:
        encoded = text.encode("utf-8")
        out.append(b"s" + struct.pack("<I", len(encoded)) + encoded)

    out: list[bytes] = [b"C"]
    _push_str(cls.__module__)
    _push_str(cls.__qualname__)
    mro = cls.__mro__
    out.append(b"m" + struct.pack("<I", len(mro)))
    for base in mro:
        _push_str(base.__module__)
        _push_str(base.__qualname__)
    class_dict = cls.__dict__
    int_attrs = spec["int_attrs"]
    out.append(b"I" + struct.pack("<I", len(int_attrs)))
    for name in int_attrs:
        _push_str(name)
        value = class_dict[name]
        if type(value) is not int:
            raise TypeError(f"class attr {name} is not an exact int")
        out.append(b"i" + value.to_bytes(8, "little", signed=True))
    property_methods = spec["property_methods"]
    out.append(b"P" + struct.pack("<I", len(property_methods)))
    for name in property_methods:
        _push_str(name)
        member = class_dict[name]
        if type(member) is not property:
            raise TypeError(f"class member {name} is not an exact property")
        if member.fset is not None or member.fdel is not None:
            raise TypeError(f"property {name} is not read-only")
        if not isinstance(member.fget, types.FunctionType):
            raise TypeError(f"property {name} getter is not a plain function")
        _push_str(compute_authority_code_digest(member.fget))
    function_methods = spec["function_methods"]
    out.append(b"F" + struct.pack("<I", len(function_methods)))
    for name in function_methods:
        _push_str(name)
        member = class_dict[name]
        if not isinstance(member, types.FunctionType):
            raise TypeError(f"class member {name} is not a plain function")
        _push_str(compute_authority_code_digest(member))
    return hashlib.sha256(b"".join(out)).hexdigest()


_DIGEST_CONST = {
    "series_map": "__RXTPD_SERIES_MAP_DIGEST",
    "series_map_values": "__RXTPD_SERIES_MAP_VALUES_DIGEST",
    "series_algorithms_map_array": "__RXTPD_ALGORITHMS_MAP_ARRAY_DIGEST",
    "series_lib_map_infer": "__RXTPD_LIB_MAP_INFER_DIGEST",
    "series_constructor_fget": "__RXTPD_SERIES_CONSTRUCTOR_FGET_DIGEST",
    "series_ndframe_finalize": "__RXTPD_NDFRAME_FINALIZE_DIGEST",
    "series_to_numpy": "__RXTPD_SERIES_TO_NUMPY_DIGEST",
    "frame_apply": "__RXTPD_FRAME_APPLY_DIGEST",
    "frame_to_numpy": "__RXTPD_FRAME_TO_NUMPY_DIGEST",
    "apply_frame_apply": "__RXTPD_APPLY_FRAME_APPLY_DIGEST",
}
_OPTIMIZED_DIGEST_CONST = {
    ("series_ndframe_finalize", 1): "__RXTPD_NDFRAME_FINALIZE_OPTIMIZE_1_DIGEST",
}

# Fixed Rust helpers: type-tagged canonical encoding + pinned-crate SHA-256, all
# from exact PyO3 type checks. No Python ``repr``/``hashlib`` is involved, so a
# custom ``__repr__`` (or any caller-controlled serialization hook) can no longer
# forge executable identity.
_RUST_IDENTITY_FUNCTIONS = r"""fn __rxtpd_type_error(message: &'static str) -> pyo3::PyErr {
    pyo3::exceptions::PyTypeError::new_err(message)
}

fn __rxtpd_push_str(buffer: &mut Vec<u8>, text: &str) {
    let bytes = text.as_bytes();
    buffer.push(b's');
    buffer.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    buffer.extend_from_slice(bytes);
}

fn __rxtpd_encode_value(
    value: &pyo3::Bound<'_, pyo3::PyAny>,
    buffer: &mut Vec<u8>,
    error: &'static str,
) -> pyo3::PyResult<()> {
    use pyo3::types::{PyAnyMethods, PyBool, PyInt, PyString, PyTuple, PyTupleMethods};
    // Accept only the exact builtin constant types present in the pinned code
    // objects; reject any custom/unknown object before it can reach the hash.
    if value.is_none() {
        buffer.push(b'N');
        return Ok(());
    }
    if value.is_exact_instance_of::<PyBool>() {
        let flag: bool = value.extract().map_err(|_| __rxtpd_type_error(error))?;
        buffer.push(if flag { b'T' } else { b'F' });
        return Ok(());
    }
    if value.is_exact_instance_of::<PyInt>() {
        let number: i64 = value.extract().map_err(|_| __rxtpd_type_error(error))?;
        buffer.push(b'i');
        buffer.extend_from_slice(&number.to_le_bytes());
        return Ok(());
    }
    if value.is_exact_instance_of::<PyString>() {
        let text: String = value.extract().map_err(|_| __rxtpd_type_error(error))?;
        let bytes = text.as_bytes();
        buffer.push(b's');
        buffer.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        buffer.extend_from_slice(bytes);
        return Ok(());
    }
    if value.is_exact_instance_of::<PyTuple>() {
        let tuple = value
            .cast::<PyTuple>()
            .map_err(|_| __rxtpd_type_error(error))?;
        buffer.push(b't');
        buffer.extend_from_slice(&(tuple.len() as u32).to_le_bytes());
        for item in tuple.iter() {
            __rxtpd_encode_value(&item, buffer, error)?;
        }
        return Ok(());
    }
    // Nested code objects (comprehensions / nested defs stored as consts) are
    // folded in recursively so behavior cannot hide inside them.
    let code_type = value.py().import("types")?.getattr("CodeType")?;
    if value.is_exact_instance(&code_type) {
        buffer.push(b'c');
        __rxtpd_encode_code(value, buffer, error)?;
        return Ok(());
    }
    Err(__rxtpd_type_error(error))
}

fn __rxtpd_encode_code(
    code: &pyo3::Bound<'_, pyo3::PyAny>,
    buffer: &mut Vec<u8>,
    error: &'static str,
) -> pyo3::PyResult<()> {
    use pyo3::types::{PyAnyMethods, PyString, PyTuple, PyTupleMethods};
    for name in [
        "co_argcount",
        "co_posonlyargcount",
        "co_kwonlyargcount",
        "co_nlocals",
        "co_flags",
        "co_stacksize",
    ] {
        let value: i64 = code
            .getattr(name)?
            .extract()
            .map_err(|_| __rxtpd_type_error(error))?;
        buffer.push(b'i');
        buffer.extend_from_slice(&value.to_le_bytes());
    }
    for name in ["co_code", "co_exceptiontable"] {
        let raw: Vec<u8> = code
            .getattr(name)?
            .extract()
            .map_err(|_| __rxtpd_type_error(error))?;
        buffer.push(b'b');
        buffer.extend_from_slice(&(raw.len() as u32).to_le_bytes());
        buffer.extend_from_slice(&raw);
    }
    for name in ["co_names", "co_varnames", "co_freevars", "co_cellvars"] {
        let field = code.getattr(name)?;
        let tuple = field
            .cast::<PyTuple>()
            .map_err(|_| __rxtpd_type_error(error))?;
        buffer.push(b't');
        buffer.extend_from_slice(&(tuple.len() as u32).to_le_bytes());
        for item in tuple.iter() {
            if !item.is_exact_instance_of::<PyString>() {
                return Err(__rxtpd_type_error(error));
            }
            let text: String = item.extract().map_err(|_| __rxtpd_type_error(error))?;
            let bytes = text.as_bytes();
            buffer.push(b's');
            buffer.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            buffer.extend_from_slice(bytes);
        }
    }
    for name in ["co_name", "co_qualname"] {
        let text: String = code
            .getattr(name)?
            .extract()
            .map_err(|_| __rxtpd_type_error(error))?;
        let bytes = text.as_bytes();
        buffer.push(b's');
        buffer.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        buffer.extend_from_slice(bytes);
    }
    let consts_field = code.getattr("co_consts")?;
    let consts = consts_field
        .cast::<PyTuple>()
        .map_err(|_| __rxtpd_type_error(error))?;
    buffer.push(b't');
    buffer.extend_from_slice(&(consts.len() as u32).to_le_bytes());
    for item in consts.iter() {
        __rxtpd_encode_value(&item, buffer, error)?;
    }
    Ok(())
}

fn __rxtpd_sha256_hex(buffer: &[u8]) -> String {
    use sha2::Digest;
    let mut hasher = sha2::Sha256::new();
    hasher.update(buffer);
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(64);
    for byte in digest.iter() {
        hex.push_str(&format!("{:02x}", byte));
    }
    hex
}

fn __rxtpd_code_digest(
    code: &pyo3::Bound<'_, pyo3::PyAny>,
    error: &'static str,
) -> pyo3::PyResult<String> {
    let mut buffer: Vec<u8> = Vec::new();
    __rxtpd_encode_code(code, &mut buffer, error)?;
    Ok(__rxtpd_sha256_hex(&buffer))
}"""


def _default_item_rs(index: int, spec: tuple) -> str:
    idx = str(index)
    kind = spec[0]
    if kind == "no_default":
        return (
            "    {\n"
            "        let item = defaults.get_item(" + idx + ")?;\n"
            "        let no_default = py.import(" + _rust_string(_NO_DEFAULT_MODULE) + ")?"
            ".getattr(" + _rust_string(_NO_DEFAULT_ATTR) + ")?;\n"
            "        if !item.is(&no_default) {\n"
            "            return Err(__rxtpd_type_error(error));\n"
            "        }\n"
            "    }\n"
        )
    if kind == "none":
        cond = "!item.is_none()"
    elif kind == "bool":
        expected = "true" if spec[1] else "false"
        cond = (
            "!item.is_exact_instance_of::<pyo3::types::PyBool>() "
            "|| item.extract::<bool>()? != " + expected
        )
    elif kind == "int":
        cond = (
            "!item.is_exact_instance_of::<pyo3::types::PyInt>() "
            "|| item.extract::<i64>()? != " + str(int(spec[1]))
        )
    elif kind == "str":
        cond = (
            "!item.is_exact_instance_of::<pyo3::types::PyString>() "
            "|| item.extract::<String>()? != " + _rust_string(spec[1])
        )
    elif kind == "empty_tuple":
        cond = (
            "!item.is_exact_instance_of::<PyTuple>() "
            "|| item.cast::<PyTuple>().map_err(|_| __rxtpd_type_error(error))?.len() != 0"
        )
    else:  # pragma: no cover - guarded by the frozen specs
        raise ValueError(f"unknown default spec kind: {kind!r}")
    return (
        "    {\n"
        "        let item = defaults.get_item(" + idx + ")?;\n"
        "        if " + cond + " {\n"
        "            return Err(__rxtpd_type_error(error));\n"
        "        }\n"
        "    }\n"
    )


def _globals_rs(spec: dict) -> str:
    blocks: list[str] = []
    for name in spec["builtins"]:
        rn = _rust_string(name)
        blocks.append(
            "    if module_dict.contains(" + rn + ")? {\n"
            '        let builtins = py.import("builtins")?;\n'
            "        if !module_dict.get_item(" + rn + ")?.is(&builtins.getattr(" + rn + ")?) {\n"
            "            return Err(__rxtpd_type_error(error));\n"
            "        }\n"
            "    }\n"
        )
    for name, modname in spec["modules"]:
        blocks.append(
            "    {\n"
            "        let canonical = py.import(" + _rust_string(modname) + ")?;\n"
            "        let bound = module_dict.get_item(" + _rust_string(name) + ")"
            ".map_err(|_| __rxtpd_type_error(error))?;\n"
            "        if !bound.is(&canonical) {\n"
            "            return Err(__rxtpd_type_error(error));\n"
            "        }\n"
            "    }\n"
        )
    for name, defmod, attr in spec["module_attrs"]:
        blocks.append(
            "    {\n"
            "        let canonical = py.import(" + _rust_string(defmod) + ")?"
            ".getattr(" + _rust_string(attr) + ")?;\n"
            "        let bound = module_dict.get_item(" + _rust_string(name) + ")"
            ".map_err(|_| __rxtpd_type_error(error))?;\n"
            "        if !bound.is(&canonical) {\n"
            "            return Err(__rxtpd_type_error(error));\n"
            "        }\n"
            "    }\n"
        )
    for name, validator_key in spec.get("module_authorities", ()):
        # The bound object is the live ``module_dict`` member that the pinned
        # frame_apply loads. It is NOT compared back to a re-import of the same
        # mutable module (a self-comparison); it is validated against its own
        # independently frozen authority (class structural digest or function
        # code digest), so a same-module replacement is rejected.
        blocks.append(
            "    {\n"
            "        let bound = module_dict.get_item(" + _rust_string(name) + ")"
            ".map_err(|_| __rxtpd_type_error(error))?;\n"
            "        __rxtpd_validate_" + validator_key + "(py, &bound)?;\n"
            "    }\n"
        )
    for impmod, attr in spec["import_from"]:
        rm = _rust_string(impmod)
        ra = _rust_string(attr)
        # The imported target is validated against its own independently frozen
        # canonical authority (exact function type, module, qualname, code digest,
        # structural defaults/kwdefaults/closure, and its own global bindings) --
        # NOT merely by re-reading and trusting the same mutable module binding.
        # A replacement with matching metadata and the pinned globals dict but a
        # different code object fails the frozen code-digest check.
        validator_key = _import_target_validator_key(impmod, attr)
        blocks.append(
            "    {\n"
            "        let imported = py.import(" + rm + ")?;\n"
            "        let target = imported.getattr("
            + ra
            + ").map_err(|_| __rxtpd_type_error(error))?;\n"
            "        __rxtpd_validate_" + validator_key + "(py, &target)?;\n"
            "    }\n"
        )
    return "".join(blocks)


def _import_target_validator_key(impmod: str, attr: str) -> str:
    """Map an ``import_from`` target to its frozen-authority validator key.

    The mapping is derived from ``_AUTHORITY_META`` so an import target and the
    canonical function it must equal cannot drift apart: every import target must
    have a frozen authority (code digest + structural checks), never a weaker
    metadata-only acceptance.
    """
    for key, meta in _AUTHORITY_META.items():
        if meta["module"] == impmod and meta["qualname"] == attr:
            return key
    raise ValueError(f"no frozen authority validator for import target {impmod}.{attr}")


def _validator_rs(key: str, error: dict[str, str]) -> str:
    meta = _AUTHORITY_META[key]
    err = error["series_method"] if key.startswith("series") else error["frame_method"]
    digest_const = _DIGEST_CONST[key]
    module = _rust_string(meta["module"])
    qualname = _rust_string(meta["qualname"])
    defaults = _AUTHORITY_DEFAULTS[key]
    optimized_digests = _AUTHORITY_OPTIMIZED_CODE_DIGESTS.get(key)
    if optimized_digests is None:
        digest_check = (
            "    if __rxtpd_code_digest(&code, error)? != " + digest_const + " {\n"
            "        return Err(__rxtpd_type_error(error));\n    }\n"
        )
    else:
        if set(optimized_digests) != {1}:
            raise ValueError(f"unsupported optimization digest modes for {key}")
        optimize_1_const = _OPTIMIZED_DIGEST_CONST[(key, 1)]
        digest_check = (
            '    let optimize: i64 = py.import("sys")?.getattr("flags")?'
            '.getattr("optimize")?.extract()?;\n'
            "    let expected_digest = match optimize {\n"
            "        0 => " + digest_const + ",\n"
            "        1 => " + optimize_1_const + ",\n"
            "        _ => return Err(__rxtpd_type_error(error)),\n"
            "    };\n"
            "    if __rxtpd_code_digest(&code, error)? != expected_digest {\n"
            "        return Err(__rxtpd_type_error(error));\n    }\n"
        )
    if "function_type_module" in meta:
        type_authority = _CYTHON_FUNCTION_TYPE_AUTHORITY
        if (
            meta["function_type_module"] != type_authority["module"]
            or meta["function_type_qualname"] != type_authority["qualname"]
        ):
            raise ValueError(f"unknown frozen callable type authority for {key}")
        metatype_authority = type_authority["metatype"]
        type_check = (
            "    let function_type = py.import("
            + _rust_string(type_authority["module"])
            + ")?.getattr("
            + _rust_string(type_authority["qualname"])
            + ")?;\n"
            '    let builtins = py.import("builtins")?;\n'
            '    let exact_type = builtins.getattr("type")?;\n'
            '    let object_type = builtins.getattr("object")?;\n'
            "    let function_metatype = function_type.get_type();\n"
            "    if !function_metatype.is_exact_instance(&exact_type)\n"
            '        || function_metatype.getattr("__qualname__")?.extract::<String>()? != '
            + _rust_string(metatype_authority["qualname"])
            + '\n        || (function_metatype.getattr("__flags__")?.extract::<u64>()? & '
            + str(metatype_authority["flags_mask"])
            + ") != "
            + str(metatype_authority["flags"])
            + '\n        || function_metatype.getattr("__basicsize__")?.extract::<i64>()? != '
            + str(metatype_authority["basicsize"])
            + '\n        || function_metatype.getattr("__itemsize__")?.extract::<i64>()? != '
            + str(metatype_authority["itemsize"])
            + '\n        || function_metatype.getattr("__dictoffset__")?.extract::<i64>()? != '
            + str(metatype_authority["dictoffset"])
            + '\n        || function_metatype.getattr("__weakrefoffset__")?.extract::<i64>()? != '
            + str(metatype_authority["weakrefoffset"])
            + "\n    {\n"
            "        return Err(__rxtpd_type_error(error));\n    }\n"
            '    let function_metatype_mro = function_metatype.getattr("__mro__")?;\n'
            "    let function_metatype_mro = function_metatype_mro\n"
            "        .cast::<PyTuple>()\n"
            "        .map_err(|_| __rxtpd_type_error(error))?;\n"
            "    if function_metatype_mro.len() != 3\n"
            "        || !function_metatype_mro.get_item(0)?.is(&function_metatype)\n"
            "        || !function_metatype_mro.get_item(1)?.is(&exact_type)\n"
            "        || !function_metatype_mro.get_item(2)?.is(&object_type)\n"
            "    {\n"
            "        return Err(__rxtpd_type_error(error));\n    }\n"
            '    if function_type.getattr("__module__")?.extract::<String>()? != '
            + _rust_string(type_authority["module"])
            + '\n        || function_type.getattr("__qualname__")?.extract::<String>()? != '
            + _rust_string(type_authority["qualname"])
            + '\n        || (function_type.getattr("__flags__")?.extract::<u64>()? & '
            + str(type_authority["flags_mask"])
            + ") != "
            + str(type_authority["flags"])
            + '\n        || function_type.getattr("__basicsize__")?.extract::<i64>()? != '
            + str(type_authority["basicsize"])
            + '\n        || function_type.getattr("__itemsize__")?.extract::<i64>()? != '
            + str(type_authority["itemsize"])
            + '\n        || function_type.getattr("__dictoffset__")?.extract::<i64>()? != '
            + str(type_authority["dictoffset"])
            + '\n        || function_type.getattr("__weakrefoffset__")?.extract::<i64>()? != '
            + str(type_authority["weakrefoffset"])
            + "\n    {\n"
            "        return Err(__rxtpd_type_error(error));\n    }\n"
            '    let function_type_mro = function_type.getattr("__mro__")?;\n'
            "    let function_type_mro = function_type_mro\n"
            "        .cast::<PyTuple>()\n"
            "        .map_err(|_| __rxtpd_type_error(error))?;\n"
            "    if function_type_mro.len() != 2\n"
            "        || !function_type_mro.get_item(0)?.is(&function_type)\n"
            "        || !function_type_mro.get_item(1)?.is(&object_type)\n"
            "        || !descriptor.is_exact_instance(&function_type)\n"
            "    {\n"
            "        return Err(__rxtpd_type_error(error));\n    }\n"
        )
    else:
        type_check = (
            '    let types_module = py.import("types")?;\n'
            '    if !descriptor.is_exact_instance(&types_module.getattr("FunctionType")?) {\n'
            "        return Err(__rxtpd_type_error(error));\n    }\n"
        )
    parts = [
        "fn __rxtpd_validate_" + key + "(\n",
        "    py: pyo3::Python<'_>,\n",
        "    descriptor: &pyo3::Bound<'_, pyo3::PyAny>,\n",
        ") -> pyo3::PyResult<()> {\n",
        "    use pyo3::types::{PyAnyMethods, PyTuple, PyTupleMethods};\n",
        "    let error = " + err + ";\n",
        type_check,
        "    let module = py.import(" + module + ")?;\n",
        '    if descriptor.getattr("__module__")?.extract::<String>()? != ' + module + " {\n",
        "        return Err(__rxtpd_type_error(error));\n    }\n",
        '    if descriptor.getattr("__qualname__")?.extract::<String>()? != ' + qualname + " {\n",
        "        return Err(__rxtpd_type_error(error));\n    }\n",
        '    let module_dict = module.getattr("__dict__")?;\n',
        '    if !descriptor.getattr("__globals__")?.is(&module_dict) {\n',
        "        return Err(__rxtpd_type_error(error));\n    }\n",
        '    if !descriptor.getattr("__closure__")?.is_none() {\n',
        "        return Err(__rxtpd_type_error(error));\n    }\n",
        '    let code = descriptor.getattr("__code__")?;\n',
        '    if code.getattr("co_freevars")?.len()? != 0 {\n',
        "        return Err(__rxtpd_type_error(error));\n    }\n",
        digest_check,
        '    if !descriptor.getattr("__kwdefaults__")?.is_none() {\n',
        "        return Err(__rxtpd_type_error(error));\n    }\n",
    ]
    if defaults is None:
        parts.extend(
            [
                '    if !descriptor.getattr("__defaults__")?.is_none() {\n',
                "        return Err(__rxtpd_type_error(error));\n    }\n",
            ]
        )
    else:
        parts.extend(
            [
                '    let defaults = descriptor.getattr("__defaults__")?;\n',
                "    if !defaults.is_exact_instance_of::<PyTuple>() {\n",
                "        return Err(__rxtpd_type_error(error));\n    }\n",
                "    let defaults = defaults.cast::<PyTuple>()"
                ".map_err(|_| __rxtpd_type_error(error))?;\n",
                "    if defaults.len() != " + str(len(defaults)) + " {\n",
                "        return Err(__rxtpd_type_error(error));\n    }\n",
            ]
        )
        for index, spec in enumerate(defaults):
            parts.append(_default_item_rs(index, spec))
    parts.append(_globals_rs(_AUTHORITY_GLOBALS[key]))
    parts.append("    Ok(())\n}\n")
    return "".join(parts)


def _rust_method_validators(error: dict[str, str]) -> str:
    return "\n".join(
        _validator_rs(key, error)
        for key in (
            "series_map",
            "series_map_values",
            "series_algorithms_map_array",
            "series_lib_map_infer",
            "series_constructor_fget",
            "series_ndframe_finalize",
            "series_to_numpy",
            "frame_apply",
            "frame_to_numpy",
            "apply_frame_apply",
        )
    )


def _function_authority_validator_rs(key: str, error_expr: str) -> str:
    """Frozen code-digest function authority for a ``pandas.core.apply`` member.

    Validates executable identity (exact function type, module, qualname, empty
    closure/freevars, ``None`` kw/positional defaults, and the frozen code
    digest) of the actual bound object -- never a self-comparison to the live
    mutable module attribute.
    """
    spec = _APPLY_FUNCTION_AUTHORITIES[key]
    const = _FUNCTION_DIGEST_CONST[key]
    module = _rust_string(spec["module"])
    qualname = _rust_string(spec["qualname"])
    return (
        "fn __rxtpd_validate_" + key + "(\n"
        "    py: pyo3::Python<'_>,\n"
        "    object: &pyo3::Bound<'_, pyo3::PyAny>,\n"
        ") -> pyo3::PyResult<()> {\n"
        "    use pyo3::types::PyAnyMethods;\n"
        "    let error = " + error_expr + ";\n"
        '    let types_module = py.import("types")?;\n'
        '    if !object.is_exact_instance(&types_module.getattr("FunctionType")?) {\n'
        "        return Err(__rxtpd_type_error(error));\n    }\n"
        '    if object.getattr("__module__")?.extract::<String>().map_err(|_| '
        "__rxtpd_type_error(error))? != " + module + " {\n"
        "        return Err(__rxtpd_type_error(error));\n    }\n"
        '    if object.getattr("__qualname__")?.extract::<String>().map_err(|_| '
        "__rxtpd_type_error(error))? != " + qualname + " {\n"
        "        return Err(__rxtpd_type_error(error));\n    }\n"
        "    let module_dict = py.import(" + module + ')?.getattr("__dict__")?;\n'
        '    if !object.getattr("__globals__")?.is(&module_dict) {\n'
        "        return Err(__rxtpd_type_error(error));\n    }\n"
        '    if !object.getattr("__closure__")?.is_none() {\n'
        "        return Err(__rxtpd_type_error(error));\n    }\n"
        '    let code = object.getattr("__code__")?;\n'
        '    if code.getattr("co_freevars")?.len()? != 0 {\n'
        "        return Err(__rxtpd_type_error(error));\n    }\n"
        "    if __rxtpd_code_digest(&code, error)? != " + const + " {\n"
        "        return Err(__rxtpd_type_error(error));\n    }\n"
        '    if !object.getattr("__kwdefaults__")?.is_none() {\n'
        "        return Err(__rxtpd_type_error(error));\n    }\n"
        '    if !object.getattr("__defaults__")?.is_none() {\n'
        "        return Err(__rxtpd_type_error(error));\n    }\n"
        "    Ok(())\n}\n"
    )


def _class_authority_validator_rs(key: str, error_expr: str) -> str:
    """Frozen structural class authority for a ``pandas.core.apply`` class.

    Rebuilds -- from the *live* class -- a type-tagged digest over the class
    identity, full MRO chain, integer class attributes (the axis selector), and
    the code digests of the named read-only property getters and plain methods,
    then compares it to a frozen constant. A same-module class replacement that
    changes any of these is rejected; there is no comparison to the live module
    attribute.
    """
    spec = _APPLY_CLASS_AUTHORITIES[key]
    const = _CLASS_DIGEST_CONST[key]
    parts = [
        "fn __rxtpd_validate_" + key + "(\n",
        "    py: pyo3::Python<'_>,\n",
        "    object: &pyo3::Bound<'_, pyo3::PyAny>,\n",
        ") -> pyo3::PyResult<()> {\n",
        "    use pyo3::types::{PyAnyMethods, PyInt, PyType, PyTuple, PyTupleMethods};\n",
        "    let error = " + error_expr + ";\n",
        '    let types_module = py.import("types")?;\n',
        '    let function_type = types_module.getattr("FunctionType")?;\n',
        '    let property_type = py.import("builtins")?.getattr("property")?;\n',
        "    if object.cast::<PyType>().is_err() {\n",
        "        return Err(__rxtpd_type_error(error));\n    }\n",
        "    let mut buffer: Vec<u8> = Vec::new();\n",
        "    buffer.push(b'C');\n",
        '    __rxtpd_push_str(&mut buffer, &object.getattr("__module__")?'
        ".extract::<String>().map_err(|_| __rxtpd_type_error(error))?);\n",
        '    __rxtpd_push_str(&mut buffer, &object.getattr("__qualname__")?'
        ".extract::<String>().map_err(|_| __rxtpd_type_error(error))?);\n",
        '    let mro_field = object.getattr("__mro__")?;\n',
        "    let mro = mro_field.cast::<PyTuple>().map_err(|_| __rxtpd_type_error(error))?;\n",
        "    buffer.push(b'm');\n",
        "    buffer.extend_from_slice(&(mro.len() as u32).to_le_bytes());\n",
        "    for base in mro.iter() {\n",
        '        __rxtpd_push_str(&mut buffer, &base.getattr("__module__")?'
        ".extract::<String>().map_err(|_| __rxtpd_type_error(error))?);\n",
        '        __rxtpd_push_str(&mut buffer, &base.getattr("__qualname__")?'
        ".extract::<String>().map_err(|_| __rxtpd_type_error(error))?);\n",
        "    }\n",
        '    let class_dict = object.getattr("__dict__")?;\n',
        "    buffer.push(b'I');\n",
        "    buffer.extend_from_slice(&(" + str(len(spec["int_attrs"])) + "u32).to_le_bytes());\n",
    ]
    for name in spec["int_attrs"]:
        rn = _rust_string(name)
        parts.append(
            "    {\n"
            "        __rxtpd_push_str(&mut buffer, " + rn + ");\n"
            "        let value = class_dict.get_item(" + rn + ")"
            ".map_err(|_| __rxtpd_type_error(error))?;\n"
            "        if !value.is_exact_instance_of::<PyInt>() {\n"
            "            return Err(__rxtpd_type_error(error));\n        }\n"
            "        let number: i64 = value.extract().map_err(|_| __rxtpd_type_error(error))?;\n"
            "        buffer.push(b'i');\n"
            "        buffer.extend_from_slice(&number.to_le_bytes());\n"
            "    }\n"
        )
    parts.append(
        "    buffer.push(b'P');\n"
        "    buffer.extend_from_slice(&("
        + str(len(spec["property_methods"]))
        + "u32).to_le_bytes());\n"
    )
    for name in spec["property_methods"]:
        rn = _rust_string(name)
        parts.append(
            "    {\n"
            "        __rxtpd_push_str(&mut buffer, " + rn + ");\n"
            "        let member = class_dict.get_item(" + rn + ")"
            ".map_err(|_| __rxtpd_type_error(error))?;\n"
            "        if !member.is_exact_instance(&property_type) {\n"
            "            return Err(__rxtpd_type_error(error));\n        }\n"
            '        if !member.getattr("fset")?.is_none() '
            '|| !member.getattr("fdel")?.is_none() {\n'
            "            return Err(__rxtpd_type_error(error));\n        }\n"
            '        let fget = member.getattr("fget")?;\n'
            "        if !fget.is_exact_instance(&function_type) {\n"
            "            return Err(__rxtpd_type_error(error));\n        }\n"
            '        let code = fget.getattr("__code__")?;\n'
            "        __rxtpd_push_str(&mut buffer, &__rxtpd_code_digest(&code, error)?);\n"
            "    }\n"
        )
    parts.append(
        "    buffer.push(b'F');\n"
        "    buffer.extend_from_slice(&("
        + str(len(spec["function_methods"]))
        + "u32).to_le_bytes());\n"
    )
    for name in spec["function_methods"]:
        rn = _rust_string(name)
        parts.append(
            "    {\n"
            "        __rxtpd_push_str(&mut buffer, " + rn + ");\n"
            "        let member = class_dict.get_item(" + rn + ")"
            ".map_err(|_| __rxtpd_type_error(error))?;\n"
            "        if !member.is_exact_instance(&function_type) {\n"
            "            return Err(__rxtpd_type_error(error));\n        }\n"
            '        let code = member.getattr("__code__")?;\n'
            "        __rxtpd_push_str(&mut buffer, &__rxtpd_code_digest(&code, error)?);\n"
            "    }\n"
        )
    parts.append(
        "    if __rxtpd_sha256_hex(&buffer) != " + const + " {\n"
        "        return Err(__rxtpd_type_error(error));\n    }\n"
        "    Ok(())\n}\n"
    )
    return "".join(parts)


def _apply_module_authority_validators(error: dict[str, str]) -> str:
    frame_error = error["frame_method"]
    validators = [
        _function_authority_validator_rs(key, frame_error) for key in _APPLY_FUNCTION_AUTHORITIES
    ]
    validators += [
        _class_authority_validator_rs(key, frame_error) for key in _APPLY_CLASS_AUTHORITIES
    ]
    return "\n".join(validators)


_RUST_SIMPLE_ESCAPES = {
    '"': '\\"',
    "\\": "\\\\",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
    "\0": "\\0",
}


def _rust_string(value: str) -> str:
    r"""Encode ``value`` as a valid Rust ``&str`` literal.

    ``json.dumps`` emits JSON ``\uXXXX`` escapes, which are *not* valid Rust
    string escapes (Rust requires ``\u{...}``) and cannot represent non-BMP
    code points in a single escape. This encoder emits printable ASCII
    verbatim, the standard Rust simple escapes for quotes/backslashes/controls,
    and ``\u{hex}`` for every other code point, including full non-BMP scalars.
    """
    pieces = ['"']
    for char in value:
        simple = _RUST_SIMPLE_ESCAPES.get(char)
        if simple is not None:
            pieces.append(simple)
            continue
        code_point = ord(char)
        if 0x20 <= code_point <= 0x7E:
            pieces.append(char)
        else:
            pieces.append(f"\\u{{{code_point:x}}}")
    pieces.append('"')
    return "".join(pieces)


def boundary_helpers() -> str:
    """Return shared one-shot validation/extraction/materialization helpers."""
    error = {name: _rust_string(message) for name, message in RUNTIME_ERRORS.items()}
    series_map_digest = _rust_string(_AUTHORITY_CODE_DIGESTS["series_map"])
    series_map_values_digest = _rust_string(_AUTHORITY_CODE_DIGESTS["series_map_values"])
    algorithms_map_array_digest = _rust_string(
        _AUTHORITY_CODE_DIGESTS["series_algorithms_map_array"]
    )
    lib_map_infer_digest = _rust_string(_AUTHORITY_CODE_DIGESTS["series_lib_map_infer"])
    series_constructor_fget_digest = _rust_string(
        _AUTHORITY_CODE_DIGESTS["series_constructor_fget"]
    )
    ndframe_finalize_digest = _rust_string(_AUTHORITY_CODE_DIGESTS["series_ndframe_finalize"])
    ndframe_finalize_optimize_1_digest = _rust_string(
        _AUTHORITY_OPTIMIZED_CODE_DIGESTS["series_ndframe_finalize"][1]
    )
    series_to_numpy_digest = _rust_string(_AUTHORITY_CODE_DIGESTS["series_to_numpy"])
    frame_apply_digest = _rust_string(_AUTHORITY_CODE_DIGESTS["frame_apply"])
    frame_to_numpy_digest = _rust_string(_AUTHORITY_CODE_DIGESTS["frame_to_numpy"])
    apply_frame_apply_digest = _rust_string(_AUTHORITY_CODE_DIGESTS["apply_frame_apply"])
    reconstruct_func_digest = _rust_string(_APPLY_FUNCTION_DIGESTS["reconstruct_func"])
    frame_column_apply_class_digest = _rust_string(_APPLY_CLASS_DIGESTS["frame_column_apply"])
    frame_row_apply_class_digest = _rust_string(_APPLY_CLASS_DIGESTS["frame_row_apply"])
    identity_functions = _RUST_IDENTITY_FUNCTIONS
    method_validators = _rust_method_validators(error)
    module_authority_validators = _apply_module_authority_validators(error)
    version = _rust_string(PINNED_PANDAS_VERSION)
    numpy_version = _rust_string(PINNED_NUMPY_VERSION)
    py_major, py_minor = PINNED_PYTHON
    return f"""struct RxtPandasSeriesF64 {{
    values: numpy::ndarray::Array1<f64>,
    source: Option<pyo3::Py<pyo3::PyAny>>,
}}

struct RxtPandasSeriesI64 {{
    values: numpy::ndarray::Array1<i64>,
    source: Option<pyo3::Py<pyo3::PyAny>>,
}}

struct RxtPandasFrameF64 {{
    values: numpy::ndarray::Array2<f64>,
    columns: Vec<String>,
}}

// Frozen known-good SHA-256 of each trusted method's canonical, type-tagged
// code encoding (CPython 3.11 / pandas 2.3.3 / numpy 2.3.5). Immutable
// constants, never regenerated from a live descriptor.
const __RXTPD_SERIES_MAP_DIGEST: &str = {series_map_digest};
const __RXTPD_SERIES_MAP_VALUES_DIGEST: &str = {series_map_values_digest};
const __RXTPD_ALGORITHMS_MAP_ARRAY_DIGEST: &str = {algorithms_map_array_digest};
const __RXTPD_LIB_MAP_INFER_DIGEST: &str = {lib_map_infer_digest};
const __RXTPD_SERIES_CONSTRUCTOR_FGET_DIGEST: &str = {series_constructor_fget_digest};
const __RXTPD_NDFRAME_FINALIZE_DIGEST: &str = {ndframe_finalize_digest};
const __RXTPD_NDFRAME_FINALIZE_OPTIMIZE_1_DIGEST: &str = {ndframe_finalize_optimize_1_digest};
const __RXTPD_SERIES_TO_NUMPY_DIGEST: &str = {series_to_numpy_digest};
const __RXTPD_FRAME_APPLY_DIGEST: &str = {frame_apply_digest};
const __RXTPD_FRAME_TO_NUMPY_DIGEST: &str = {frame_to_numpy_digest};
const __RXTPD_APPLY_FRAME_APPLY_DIGEST: &str = {apply_frame_apply_digest};
// The three ``pandas.core.apply`` members ``frame_apply`` loads are validated
// against these independently frozen authorities -- a function code digest and
// two structural class digests -- never by self-comparison to the same module.
const __RXTPD_RECONSTRUCT_FUNC_DIGEST: &str = {reconstruct_func_digest};
const __RXTPD_FRAME_COLUMN_APPLY_CLASS_DIGEST: &str = {frame_column_apply_class_digest};
const __RXTPD_FRAME_ROW_APPLY_CLASS_DIGEST: &str = {frame_row_apply_class_digest};

{identity_functions}

{method_validators}

{module_authority_validators}

fn __rxtpd_is_exact_numpy_dtype(
    py: pyo3::Python<'_>,
    dtype: &pyo3::Bound<'_, pyo3::PyAny>,
    expected_name: &str,
) -> pyo3::PyResult<bool> {{
    use pyo3::types::PyAnyMethods;
    let numpy_module = py.import("numpy")?;
    let numpy_dtype_type = numpy_module.getattr("dtype")?;
    // Nullable Float64/Int64, numeric Categorical, Sparse, and Arrow-backed
    // dtypes are NOT instances of ``numpy.dtype``; rejecting non-instances
    // before conversion refuses every pandas extension storage even when
    // ``to_numpy()`` would coincidentally yield a numeric ndarray.
    if !dtype.is_instance(&numpy_dtype_type)? {{
        return Ok(false);
    }}
    let expected = numpy_dtype_type.call1((expected_name,))?;
    dtype.eq(&expected)
}}

fn __rxtpd_pinned_series_class<'py>(
    py: pyo3::Python<'py>,
) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {{
    use pyo3::types::PyAnyMethods;
    let pandas = py.import("pandas")?;
    let numpy_module = py.import("numpy")?;
    let pandas_version: String = pandas.getattr("__version__")?.extract()?;
    let numpy_version: String = numpy_module.getattr("__version__")?.extract()?;
    if pandas_version != {version} || numpy_version != {numpy_version} {{
        return Err(__rxtpd_type_error({error["version"]}));
    }}
    let version_info = py.import("sys")?.getattr("version_info")?;
    let py_major: i64 = version_info.getattr("major")?.extract()?;
    let py_minor: i64 = version_info.getattr("minor")?.extract()?;
    if py_major != {py_major} || py_minor != {py_minor} {{
        return Err(__rxtpd_type_error({error["version"]}));
    }}
    let series_class = pandas.getattr("Series")?;
    let class_dict = series_class.getattr("__dict__")?;
    let map_descriptor = class_dict
        .get_item("map")
        .map_err(|_| __rxtpd_type_error({error["series_method"]}))?;
    let map_attribute = series_class
        .getattr("map")
        .map_err(|_| __rxtpd_type_error({error["series_method"]}))?;
    let map_values_owner = py.import("pandas.core.base")?.getattr("IndexOpsMixin")?;
    let map_values_descriptor = map_values_owner
        .getattr("__dict__")?
        .get_item("_map_values")
        .map_err(|_| __rxtpd_type_error({error["series_method"]}))?;
    let map_values_attribute = series_class
        .getattr("_map_values")
        .map_err(|_| __rxtpd_type_error({error["series_method"]}))?;
    let algorithms_map_array = py
        .import("pandas.core.algorithms")?
        .getattr("map_array")
        .map_err(|_| __rxtpd_type_error({error["series_method"]}))?;
    let lib_map_infer = py
        .import("pandas._libs.lib")?
        .getattr("map_infer")
        .map_err(|_| __rxtpd_type_error({error["series_method"]}))?;
    let constructor_property = class_dict
        .get_item("_constructor")
        .map_err(|_| __rxtpd_type_error({error["series_method"]}))?;
    let property_type = py.import("builtins")?.getattr("property")?;
    if !constructor_property.is_exact_instance(&property_type)
        || !constructor_property.getattr("fset")?.is_none()
        || !constructor_property.getattr("fdel")?.is_none()
    {{
        return Err(__rxtpd_type_error({error["series_method"]}));
    }}
    let constructor_fget = constructor_property
        .getattr("fget")
        .map_err(|_| __rxtpd_type_error({error["series_method"]}))?;
    let finalize_owner = py.import("pandas.core.generic")?.getattr("NDFrame")?;
    let finalize_descriptor = finalize_owner
        .getattr("__dict__")?
        .get_item("__finalize__")
        .map_err(|_| __rxtpd_type_error({error["series_method"]}))?;
    let finalize_attribute = series_class
        .getattr("__finalize__")
        .map_err(|_| __rxtpd_type_error({error["series_method"]}))?;
    let to_numpy_attribute = series_class
        .getattr("to_numpy")
        .map_err(|_| __rxtpd_type_error({error["series_method"]}))?;
    if !map_descriptor.is(&map_attribute)
        || class_dict.contains("_map_values")?
        || !map_values_descriptor.is(&map_values_attribute)
        || class_dict.contains("__finalize__")?
        || !finalize_descriptor.is(&finalize_attribute)
    {{
        return Err(__rxtpd_type_error({error["series_method"]}));
    }}
    __rxtpd_validate_series_map(py, &map_descriptor)?;
    __rxtpd_validate_series_map_values(py, &map_values_descriptor)?;
    __rxtpd_validate_series_algorithms_map_array(py, &algorithms_map_array)?;
    __rxtpd_validate_series_lib_map_infer(py, &lib_map_infer)?;
    __rxtpd_validate_series_constructor_fget(py, &constructor_fget)?;
    __rxtpd_validate_series_ndframe_finalize(py, &finalize_descriptor)?;
    __rxtpd_validate_series_to_numpy(py, &to_numpy_attribute)?;
    Ok(series_class)
}}

fn __rxtpd_pinned_frame_class<'py>(
    py: pyo3::Python<'py>,
) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {{
    use pyo3::types::PyAnyMethods;
    let pandas = py.import("pandas")?;
    let numpy_module = py.import("numpy")?;
    let pandas_version: String = pandas.getattr("__version__")?.extract()?;
    let numpy_version: String = numpy_module.getattr("__version__")?.extract()?;
    if pandas_version != {version} || numpy_version != {numpy_version} {{
        return Err(__rxtpd_type_error({error["version"]}));
    }}
    let version_info = py.import("sys")?.getattr("version_info")?;
    let py_major: i64 = version_info.getattr("major")?.extract()?;
    let py_minor: i64 = version_info.getattr("minor")?.extract()?;
    if py_major != {py_major} || py_minor != {py_minor} {{
        return Err(__rxtpd_type_error({error["version"]}));
    }}
    let frame_class = pandas.getattr("DataFrame")?;
    let class_dict = frame_class.getattr("__dict__")?;
    let apply_descriptor = class_dict
        .get_item("apply")
        .map_err(|_| __rxtpd_type_error({error["frame_method"]}))?;
    let apply_attribute = frame_class
        .getattr("apply")
        .map_err(|_| __rxtpd_type_error({error["frame_method"]}))?;
    let to_numpy_descriptor = class_dict
        .get_item("to_numpy")
        .map_err(|_| __rxtpd_type_error({error["frame_method"]}))?;
    let to_numpy_attribute = frame_class
        .getattr("to_numpy")
        .map_err(|_| __rxtpd_type_error({error["frame_method"]}))?;
    if !apply_descriptor.is(&apply_attribute) || !to_numpy_descriptor.is(&to_numpy_attribute) {{
        return Err(__rxtpd_type_error({error["frame_method"]}));
    }}
    __rxtpd_validate_frame_apply(py, &apply_descriptor)?;
    __rxtpd_validate_frame_to_numpy(py, &to_numpy_descriptor)?;
    Ok(frame_class)
}}

fn __rxtpd_series_parts<'py>(
    py: pyo3::Python<'py>,
    value: &pyo3::Bound<'py, pyo3::PyAny>,
    expected_dtype: &str,
    dtype_error: &'static str,
) -> pyo3::PyResult<(pyo3::Py<pyo3::PyAny>, pyo3::Bound<'py, pyo3::PyAny>)> {{
    use pyo3::types::{{PyAnyMethods, PyDict, PyDictMethods, PyString}};
    let series_class = __rxtpd_pinned_series_class(py)?;
    let series_type = series_class.cast::<pyo3::types::PyType>()?;
    if !value.get_type().is(series_type) {{
        return Err(__rxtpd_type_error({error["series_class"]}));
    }}
    let instance_dict = value.getattr("__dict__")?;
    if instance_dict.contains("map")?
        || instance_dict.contains("_map_values")?
        || instance_dict.contains("to_numpy")?
    {{
        return Err(__rxtpd_type_error({error["series_method"]}));
    }}
    let length = value.len()?;
    if length == 0 {{
        return Err(__rxtpd_type_error({error["series_empty"]}));
    }}
    if value.getattr("attrs")?.len()? != 0 {{
        return Err(__rxtpd_type_error({error["series_attrs"]}));
    }}
    let allows_duplicates: bool = value
        .getattr("flags")?
        .getattr("allows_duplicate_labels")?
        .extract()?;
    if !allows_duplicates {{
        return Err(__rxtpd_type_error({error["series_flags"]}));
    }}
    let pandas = py.import("pandas")?;
    let range_class = pandas.getattr("RangeIndex")?;
    let index = value.getattr("index")?;
    let range_type = range_class.cast::<pyo3::types::PyType>()?;
    if !index.get_type().is(range_type) {{
        return Err(__rxtpd_type_error({error["series_index"]}));
    }}
    let start: isize = index.getattr("start")?.extract()?;
    let stop: isize = index.getattr("stop")?.extract()?;
    let step: isize = index.getattr("step")?.extract()?;
    if start != 0
        || stop != length as isize
        || step != 1
        || !index.getattr("name")?.is_none()
    {{
        return Err(__rxtpd_type_error({error["series_index"]}));
    }}
    let name = value.getattr("name")?;
    if !name.is_none() && !name.is_instance_of::<PyString>() {{
        return Err(__rxtpd_type_error({error["series_name"]}));
    }}
    let dtype = value.getattr("dtype")?;
    if !__rxtpd_is_exact_numpy_dtype(py, &dtype, expected_dtype)? {{
        return Err(__rxtpd_type_error(dtype_error));
    }}
    let to_numpy = series_class.getattr("to_numpy")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("copy", false)?;
    let array = to_numpy.call((value,), Some(&kwargs))?;
    Ok((value.clone().unbind(), array))
}}

fn __rxtpd_extract_series_f64<'py>(
    py: pyo3::Python<'py>,
    value: &pyo3::Bound<'py, pyo3::PyAny>,
) -> pyo3::PyResult<RxtPandasSeriesF64> {{
    use numpy::PyArrayMethods;
    let (source, array) = __rxtpd_series_parts(py, value, "float64", {error["series_f64"]})?;
    let typed = array
        .cast::<numpy::PyArray1<f64>>()
        .map_err(|_| __rxtpd_type_error({error["series_f64"]}))?;
    let values = typed.readonly().as_array().to_owned();
    Ok(RxtPandasSeriesF64 {{
        values,
        source: Some(source),
    }})
}}

fn __rxtpd_extract_series_i64<'py>(
    py: pyo3::Python<'py>,
    value: &pyo3::Bound<'py, pyo3::PyAny>,
) -> pyo3::PyResult<RxtPandasSeriesI64> {{
    use numpy::PyArrayMethods;
    let (source, array) = __rxtpd_series_parts(py, value, "int64", {error["series_i64"]})?;
    let typed = array
        .cast::<numpy::PyArray1<i64>>()
        .map_err(|_| __rxtpd_type_error({error["series_i64"]}))?;
    let values = typed.readonly().as_array().to_owned();
    Ok(RxtPandasSeriesI64 {{
        values,
        source: Some(source),
    }})
}}

fn __rxtpd_extract_frame_f64<'py>(
    py: pyo3::Python<'py>,
    value: &pyo3::Bound<'py, pyo3::PyAny>,
) -> pyo3::PyResult<RxtPandasFrameF64> {{
    use numpy::PyArrayMethods;
    use pyo3::types::{{PyAnyMethods, PyDict, PyDictMethods}};
    let frame_class = __rxtpd_pinned_frame_class(py)?;
    let frame_type = frame_class.cast::<pyo3::types::PyType>()?;
    if !value.get_type().is(frame_type) {{
        return Err(__rxtpd_type_error({error["frame_class"]}));
    }}
    let instance_dict = value.getattr("__dict__")?;
    if instance_dict.contains("apply")? || instance_dict.contains("to_numpy")? {{
        return Err(__rxtpd_type_error({error["frame_method"]}));
    }}
    let shape: (usize, usize) = value.getattr("shape")?.extract()?;
    if shape.0 == 0 {{
        return Err(__rxtpd_type_error({error["frame_empty"]}));
    }}
    if shape.1 == 0 {{
        return Err(__rxtpd_type_error({error["frame_zero_columns"]}));
    }}
    if value.getattr("attrs")?.len()? != 0 {{
        return Err(__rxtpd_type_error({error["frame_attrs"]}));
    }}
    let allows_duplicates: bool = value
        .getattr("flags")?
        .getattr("allows_duplicate_labels")?
        .extract()?;
    if !allows_duplicates {{
        return Err(__rxtpd_type_error({error["frame_flags"]}));
    }}
    let pandas = py.import("pandas")?;
    let range_class = pandas.getattr("RangeIndex")?;
    let index = value.getattr("index")?;
    let range_type = range_class.cast::<pyo3::types::PyType>()?;
    if !index.get_type().is(range_type) {{
        return Err(__rxtpd_type_error({error["frame_index"]}));
    }}
    let start: isize = index.getattr("start")?.extract()?;
    let stop: isize = index.getattr("stop")?.extract()?;
    let step: isize = index.getattr("step")?.extract()?;
    if start != 0
        || stop != shape.0 as isize
        || step != 1
        || !index.getattr("name")?.is_none()
    {{
        return Err(__rxtpd_type_error({error["frame_index"]}));
    }}
    let columns = value.getattr("columns")?;
    let columns_unique: bool = columns.getattr("is_unique")?.extract()?;
    if !columns_unique || !columns.getattr("name")?.is_none() {{
        return Err(__rxtpd_type_error({error["frame_schema"]}));
    }}
    let column_names: Vec<String> = columns
        .call_method0("tolist")?
        .extract()
        .map_err(|_| __rxtpd_type_error({error["frame_schema"]}))?;
    if column_names.len() != shape.1 {{
        return Err(__rxtpd_type_error({error["frame_schema"]}));
    }}
    // Require every column to be exact non-nullable NumPy float64 storage.
    // Iterating with the ``numpy.dtype`` instance check rejects nullable
    // Float64, numeric Categorical, Sparse, and Arrow-backed columns even if
    // their ``__eq__`` were to compare equal to ``float64``.
    let dtype_list = value.getattr("dtypes")?.call_method0("tolist")?;
    let dtype_count = dtype_list.len()?;
    for position in 0..dtype_count {{
        let item = dtype_list.get_item(position)?;
        if !__rxtpd_is_exact_numpy_dtype(py, &item, "float64")? {{
            return Err(__rxtpd_type_error({error["frame_f64"]}));
        }}
    }}
    let to_numpy = frame_class.getattr("to_numpy")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("copy", false)?;
    let array = to_numpy.call((value,), Some(&kwargs))?;
    let typed = array
        .cast::<numpy::PyArray2<f64>>()
        .map_err(|_| __rxtpd_type_error({error["frame_f64"]}))?;
    let values = typed.readonly().as_array().to_owned();
    Ok(RxtPandasFrameF64 {{
        values,
        columns: column_names,
    }})
}}

trait RxtPandasMaterializedSeries {{
    type Elem: numpy::Element;
    fn into_parts(
        self,
    ) -> (
        numpy::ndarray::Array1<Self::Elem>,
        Option<pyo3::Py<pyo3::PyAny>>,
    );
}}

impl RxtPandasMaterializedSeries for RxtPandasSeriesF64 {{
    type Elem = f64;
    fn into_parts(
        self,
    ) -> (numpy::ndarray::Array1<f64>, Option<pyo3::Py<pyo3::PyAny>>) {{
        (self.values, self.source)
    }}
}}

impl RxtPandasMaterializedSeries for RxtPandasSeriesI64 {{
    type Elem = i64;
    fn into_parts(
        self,
    ) -> (numpy::ndarray::Array1<i64>, Option<pyo3::Py<pyo3::PyAny>>) {{
        (self.values, self.source)
    }}
}}

fn __rxtpd_materialize_series<'py, T>(
    py: pyo3::Python<'py>,
    value: T,
) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>>
where
    T: RxtPandasMaterializedSeries,
{{
    use numpy::ToPyArray;
    use pyo3::types::{{PyAnyMethods, PyDict, PyDictMethods}};
    let (values, source) = value.into_parts();
    let array = values.to_pyarray(py);
    let Some(source) = source else {{
        let series_class = __rxtpd_pinned_series_class(py)?;
        return series_class.call1((array,));
    }};
    let source = source.bind(py);
    let constructor = source.getattr("_constructor")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("index", source.getattr("index")?)?;
    kwargs.set_item("copy", false)?;
    let result = constructor.call((array,), Some(&kwargs))?;
    let finalize_kwargs = PyDict::new(py);
    finalize_kwargs.set_item("method", "map")?;
    result
        .getattr("__finalize__")?
        .call((source,), Some(&finalize_kwargs))
}}

fn __rxtpd_materialize_frame_f64<'py>(
    py: pyo3::Python<'py>,
    value: RxtPandasFrameF64,
) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {{
    use numpy::ToPyArray;
    use pyo3::types::{{PyAnyMethods, PyDict, PyDictMethods}};
    let frame_class = __rxtpd_pinned_frame_class(py)?;
    let array = value.values.to_pyarray(py);
    let kwargs = PyDict::new(py);
    kwargs.set_item("columns", value.columns)?;
    frame_class.call((array,), Some(&kwargs))
}}"""


def _literal(expr: CallableBodyExpr) -> str:
    literal = expr.literal
    if literal is None:
        raise ValueError("missing callable literal")
    if literal.kind == "float":
        return f"{literal.value!r}_f64"
    if literal.kind == "int":
        value = int(literal.value)
        if value == -(2**63):
            return "i64::MIN"
        if value == 2**63 - 1:
            return "i64::MAX"
        return f"{value}_i64"
    raise ValueError(f"unsupported audited literal kind: {literal.kind}")


def _render_expr(expr: CallableBodyExpr) -> str:
    if expr.kind == "param":
        return "value"
    if expr.kind == "literal":
        return _literal(expr)
    if expr.kind == "unary":
        return f"(-{_render_expr(expr.children[0])})"
    if expr.kind == "binop":
        left, right = expr.children
        return f"({_render_expr(left)} {expr.op} {_render_expr(right)})"
    if expr.kind == "compare":
        rendered = []
        for index, op in enumerate(expr.ops):
            rendered.append(
                f"({_render_expr(expr.children[index])} {op} "
                f"{_render_expr(expr.children[index + 1])})"
            )
        return f"({' && '.join(rendered)})"
    if expr.kind == "boolop":
        op = "&&" if expr.op == "and" else "||"
        return f"({f' {op} '.join(_render_expr(child) for child in expr.children)})"
    if expr.kind == "cond":
        test, body, orelse = expr.children
        return (
            f"(if {_render_expr(test)} {{ {_render_expr(body)} }} "
            f"else {{ {_render_expr(orelse)} }})"
        )
    raise ValueError(f"unsupported audited body node at lower time: {expr.kind!r}")


def series_map_helpers(
    input_key: str,
    output_key: str,
    meta: CallableMeta,
) -> tuple[str, tuple[str, ...]]:
    """Return a deterministic pure hot loop plus its GIL-detaching wrapper."""
    expression = meta.body.expression
    if expression is None:
        raise ValueError("Series.map lowering requires an available body")
    identity = json.dumps(
        {
            "route": "series.map",
            "input": input_key,
            "output": output_key,
            "callable": meta.body.to_dict(),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    hot_name = f"__rxtpd_map_values_{suffix}"
    wrapper_name = f"__rxtpd_map_series_{suffix}"
    input_struct = _STRUCT[input_key]
    output_struct = _STRUCT[output_key]
    input_scalar = _RUST_SCALAR[input_key]
    output_scalar = _RUST_SCALAR[output_key]
    rust_expr = _render_expr(expression)
    hot = f"""fn {hot_name}(
    input: &numpy::ndarray::Array1<{input_scalar}>,
) -> numpy::ndarray::Array1<{output_scalar}> {{
    let mut output = Vec::with_capacity(input.len());
    for &value in input.iter() {{
        output.push({rust_expr});
    }}
    numpy::ndarray::Array1::from_vec(output)
}}"""
    wrapper = f"""fn {wrapper_name}<'py>(
    py: pyo3::Python<'py>,
    input: &{input_struct},
) -> pyo3::PyResult<{output_struct}> {{
    let source = input.source.as_ref().map(|source| source.clone_ref(py));
    let values = py.detach(|| {hot_name}(&input.values));
    Ok({output_struct} {{ values, source }})
}}"""
    return wrapper_name, (hot, wrapper)


def _render_row_expr(expr: CallableBodyExpr, field_indexes: dict[str, int]) -> str:
    """Render an already-audited row expression against one ndarray row view."""
    if expr.kind == "literal":
        return _literal(expr)
    if expr.kind == "subscript":
        return f"row[{field_indexes[expr.name]}]"
    if expr.kind == "unary":
        return f"(-{_render_row_expr(expr.children[0], field_indexes)})"
    if expr.kind == "compare":
        rendered = []
        for index, op in enumerate(expr.ops):
            rendered.append(
                f"({_render_row_expr(expr.children[index], field_indexes)} {op} "
                f"{_render_row_expr(expr.children[index + 1], field_indexes)})"
            )
        return f"({' && '.join(rendered)})"
    if expr.kind == "boolop":
        op = "&&" if expr.op == "and" else "||"
        return (
            f"({f' {op} '.join(_render_row_expr(child, field_indexes) for child in expr.children)})"
        )
    if expr.kind == "cond":
        test, body, orelse = expr.children
        return (
            f"(if {_render_row_expr(test, field_indexes)} "
            f"{{ {_render_row_expr(body, field_indexes)} }} "
            f"else {{ {_render_row_expr(orelse, field_indexes)} }})"
        )
    raise ValueError(f"unsupported audited row node at lower time: {expr.kind!r}")


def prototype_dataframe_apply_helpers(
    schema: SchemaMeta,
    meta: CallableMeta,
) -> tuple[str, tuple[str, ...]]:
    """Return research-only row-loop snippets for the NO-GO apply prototype.

    The plugin never dispatches this helper. It remains solely so the pinned
    semantic characterization and authority-bypass evidence stay reproducible.
    """
    expression = meta.body.expression
    if expression is None:
        raise ValueError("DataFrame.apply prototype generation requires an available body")
    identity = json.dumps(
        {
            "route": "dataframe.apply.axis1",
            "schema": schema.to_dict(),
            "callable": meta.body.to_dict(),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    hot_name = f"__rxtpd_apply_values_{suffix}"
    wrapper_name = f"__rxtpd_apply_frame_{suffix}"
    field_indexes = {field.name: index for index, field in enumerate(schema.fields)}
    rust_expr = _render_row_expr(expression, field_indexes)
    expected_columns = ", ".join(_rust_string(field.name) for field in schema.fields)
    hot = f"""fn {hot_name}(
    input: &numpy::ndarray::Array2<f64>,
) -> numpy::ndarray::Array1<f64> {{
    let mut output = Vec::with_capacity(input.nrows());
    for row in input.outer_iter() {{
        output.push({rust_expr});
    }}
    numpy::ndarray::Array1::from_vec(output)
}}"""
    wrapper = f"""fn {wrapper_name}<'py>(
    py: pyo3::Python<'py>,
    input: &RxtPandasFrameF64,
) -> pyo3::PyResult<RxtPandasSeriesF64> {{
    const EXPECTED_COLUMNS: &[&str] = &[{expected_columns}];
    if input.columns.len() != EXPECTED_COLUMNS.len()
        || !input
            .columns
            .iter()
            .zip(EXPECTED_COLUMNS.iter())
            .all(|(actual, expected)| actual == expected)
    {{
        return Err(__rxtpd_type_error({_rust_string(RUNTIME_ERRORS["frame_schema"])}));
    }}
    let values = py.detach(|| {hot_name}(&input.values));
    Ok(RxtPandasSeriesF64 {{
        values,
        source: None,
    }})
}}"""
    return wrapper_name, (hot, wrapper)


__all__ = [
    "boundary_helpers",
    "prototype_dataframe_apply_helpers",
    "series_map_helpers",
]
