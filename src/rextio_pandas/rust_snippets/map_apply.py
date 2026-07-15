"""Rust helper generation for pandas boundaries and native map/apply loops."""

from __future__ import annotations

import hashlib
import json

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

# Immutable known-good semantic fingerprints of the four trusted pandas methods,
# captured once from the pinned authority (CPython 3.11 / pandas 2.3.3 /
# numpy 2.3.5) by ``compute_authority_fingerprint`` below. These are frozen
# constants embedded into the generated Rust; they are NOT regenerated from
# whatever descriptor happens to exist at lowering time, so a pandas patched
# before lowering can never become the trusted value. The fingerprint covers the
# full executable code-object semantics (bytecode, constants, names, varnames,
# flags, arg counts, stack size, exception table), the function ``__defaults__``
# and ``__kwdefaults__``, and the module/qualname; the runtime guard additionally
# proves an empty closure and that ``__globals__`` is the pinned defining module's
# own dictionary. A test asserts these constants still equal the live pinned
# pandas so drift is caught loudly rather than silently trusted.
_AUTHORITY_FINGERPRINTS = {
    "series_map": "1b4d2b08156d20c13e03b240dd2b839e4f709ecefaae1aa68477a040ab3a3231",
    "series_to_numpy": "eb587a2abea2f9fd8ebec3a7d8802b4cacb397f6a98348cedef1c90bd728eb0d",
    "frame_apply": "c8503b0ab63075db08f7c6a8a810f3ba360fbe268743bde3bde6f29e44f7fff5",
    "frame_to_numpy": "78b63dedde7c364344a7da200954ede760cfbb2d0115b423ee1835db53b1fee3",
}
_AUTHORITY_MODULES = {
    "series_map": "pandas.core.series",
    "series_to_numpy": "pandas.core.base",
    "frame_apply": "pandas.core.frame",
    "frame_to_numpy": "pandas.core.frame",
}
# Fields hashed, in the exact order mirrored by the generated Rust guard.
_FINGERPRINT_REPR_FIELDS = (
    "__module__",
    "__qualname__",
    "__defaults__",
    "__kwdefaults__",
    "co_argcount",
    "co_posonlyargcount",
    "co_kwonlyargcount",
    "co_nlocals",
    "co_flags",
    "co_stacksize",
    "co_names",
    "co_varnames",
    "co_freevars",
    "co_cellvars",
    "co_name",
    "co_qualname",
    "co_consts",
)


def compute_authority_fingerprint(function: object, module_name: str) -> str:
    """Compute the semantic fingerprint of one pinned pandas method.

    This is the Python authority mirror of the generated Rust guard: identical
    field order and serialization. It validates the same invariants (genuine
    function type, empty closure, and ``__globals__`` identity with the pinned
    defining module) and returns the SHA-256 hex digest. It is used only to
    capture ``_AUTHORITY_FINGERPRINTS`` and to assert in tests that those frozen
    constants still match the pinned pandas; the runtime never regenerates the
    expected value.
    """
    import sys
    import types

    if not isinstance(function, types.FunctionType):
        raise TypeError("pinned pandas descriptor is not a plain function")
    module = sys.modules.get(module_name)
    if module is None or function.__globals__ is not module.__dict__:
        raise TypeError("pinned pandas descriptor has foreign globals provenance")
    if function.__closure__ is not None:
        raise TypeError("pinned pandas descriptor has a non-empty closure")
    code = function.__code__
    if code.co_freevars:
        raise TypeError("pinned pandas descriptor has free variables")

    digest = hashlib.sha256()
    for field in _FINGERPRINT_REPR_FIELDS:
        source = (
            function
            if field in {"__module__", "__qualname__", "__defaults__", "__kwdefaults__"}
            else code
        )
        digest.update(repr(getattr(source, field)).encode("utf-8"))
        digest.update(b"\x00")
    digest.update(code.co_code)
    digest.update(b"\x00")
    digest.update(code.co_exceptiontable)
    digest.update(b"\x00")
    return digest.hexdigest()


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
    series_map_fp = _rust_string(_AUTHORITY_FINGERPRINTS["series_map"])
    series_to_numpy_fp = _rust_string(_AUTHORITY_FINGERPRINTS["series_to_numpy"])
    frame_apply_fp = _rust_string(_AUTHORITY_FINGERPRINTS["frame_apply"])
    frame_to_numpy_fp = _rust_string(_AUTHORITY_FINGERPRINTS["frame_to_numpy"])
    series_map_module = _rust_string(_AUTHORITY_MODULES["series_map"])
    series_to_numpy_module = _rust_string(_AUTHORITY_MODULES["series_to_numpy"])
    frame_apply_module = _rust_string(_AUTHORITY_MODULES["frame_apply"])
    frame_to_numpy_module = _rust_string(_AUTHORITY_MODULES["frame_to_numpy"])
    version = _rust_string(PINNED_PANDAS_VERSION)
    numpy_version = _rust_string(PINNED_NUMPY_VERSION)
    py_major, py_minor = PINNED_PYTHON
    return f"""struct RxtPandasSeriesF64 {{
    values: numpy::ndarray::Array1<f64>,
    name: pyo3::Py<pyo3::PyAny>,
}}

struct RxtPandasSeriesI64 {{
    values: numpy::ndarray::Array1<i64>,
    name: pyo3::Py<pyo3::PyAny>,
}}

struct RxtPandasFrameF64 {{
    values: numpy::ndarray::Array2<f64>,
    columns: Vec<String>,
}}

// Frozen known-good semantic fingerprints of the four trusted pandas methods,
// captured once from CPython 3.11 / pandas 2.3.3 / numpy 2.3.5. These are
// immutable constants, never regenerated from the live descriptor.
const __RXTPD_SERIES_MAP_FP: &str = {series_map_fp};
const __RXTPD_SERIES_TO_NUMPY_FP: &str = {series_to_numpy_fp};
const __RXTPD_FRAME_APPLY_FP: &str = {frame_apply_fp};
const __RXTPD_FRAME_TO_NUMPY_FP: &str = {frame_to_numpy_fp};

fn __rxtpd_type_error(message: &'static str) -> pyo3::PyErr {{
    pyo3::exceptions::PyTypeError::new_err(message)
}}

fn __rxtpd_method_fingerprint(
    py: pyo3::Python<'_>,
    descriptor: &pyo3::Bound<'_, pyo3::PyAny>,
    module_name: &str,
    error: &'static str,
) -> pyo3::PyResult<String> {{
    use pyo3::types::{{PyAnyMethods, PyBytes}};
    // A genuine pinned descriptor is a plain Python function. Because ``co_code``
    // alone omits defaults/kwdefaults/constants/names/flags/exception-table and
    // globals provenance, this hashes the full executable code-object semantics
    // AND the function ``__defaults__``/``__kwdefaults__``/module/qualname, after
    // proving the type, an empty closure/freevars, and that ``__globals__`` is the
    // pinned defining module's own dictionary (not a caller-supplied copy). The
    // resulting digest is compared to a frozen authority constant, so changed
    // defaults, changed globals, forged constants, replacement, deletion, or a
    // malformed descriptor all surface the stable ``TypeError``.
    let types_module = py.import("types")?;
    let function_type = types_module.getattr("FunctionType")?;
    if !descriptor.is_instance(&function_type)? {{
        return Err(__rxtpd_type_error(error));
    }}
    let modules = py.import("sys")?.getattr("modules")?;
    let module = modules
        .get_item(module_name)
        .map_err(|_| __rxtpd_type_error(error))?;
    let module_dict = module.getattr("__dict__")?;
    if !descriptor.getattr("__globals__")?.is(&module_dict) {{
        return Err(__rxtpd_type_error(error));
    }}
    if !descriptor.getattr("__closure__")?.is_none() {{
        return Err(__rxtpd_type_error(error));
    }}
    let code = descriptor.getattr("__code__")?;
    if code.getattr("co_freevars")?.len()? != 0 {{
        return Err(__rxtpd_type_error(error));
    }}
    let hasher = py.import("hashlib")?.call_method0("sha256")?;
    let separator = PyBytes::new(py, b"\\x00");
    let repr_fields = [
        descriptor.getattr("__module__")?,
        descriptor.getattr("__qualname__")?,
        descriptor.getattr("__defaults__")?,
        descriptor.getattr("__kwdefaults__")?,
        code.getattr("co_argcount")?,
        code.getattr("co_posonlyargcount")?,
        code.getattr("co_kwonlyargcount")?,
        code.getattr("co_nlocals")?,
        code.getattr("co_flags")?,
        code.getattr("co_stacksize")?,
        code.getattr("co_names")?,
        code.getattr("co_varnames")?,
        code.getattr("co_freevars")?,
        code.getattr("co_cellvars")?,
        code.getattr("co_name")?,
        code.getattr("co_qualname")?,
        code.getattr("co_consts")?,
    ];
    for field in repr_fields {{
        let encoded = field.repr()?.call_method1("encode", ("utf-8",))?;
        hasher.call_method1("update", (encoded,))?;
        hasher.call_method1("update", (&separator,))?;
    }}
    let co_code = code.getattr("co_code")?;
    hasher.call_method1("update", (co_code,))?;
    hasher.call_method1("update", (&separator,))?;
    let co_exceptiontable = code.getattr("co_exceptiontable")?;
    hasher.call_method1("update", (co_exceptiontable,))?;
    hasher.call_method1("update", (&separator,))?;
    let hexdigest: String = hasher.call_method0("hexdigest")?.extract()?;
    Ok(hexdigest)
}}

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
    let to_numpy_attribute = series_class
        .getattr("to_numpy")
        .map_err(|_| __rxtpd_type_error({error["series_method"]}))?;
    if !map_descriptor.is(&map_attribute) {{
        return Err(__rxtpd_type_error({error["series_method"]}));
    }}
    if __rxtpd_method_fingerprint(py, &map_descriptor, {series_map_module}, {error["series_method"]})?
        != __RXTPD_SERIES_MAP_FP
    {{
        return Err(__rxtpd_type_error({error["series_method"]}));
    }}
    if __rxtpd_method_fingerprint(
        py,
        &to_numpy_attribute,
        {series_to_numpy_module},
        {error["series_method"]},
    )? != __RXTPD_SERIES_TO_NUMPY_FP
    {{
        return Err(__rxtpd_type_error({error["series_method"]}));
    }}
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
    if __rxtpd_method_fingerprint(py, &apply_descriptor, {frame_apply_module}, {error["frame_method"]})?
        != __RXTPD_FRAME_APPLY_FP
    {{
        return Err(__rxtpd_type_error({error["frame_method"]}));
    }}
    if __rxtpd_method_fingerprint(
        py,
        &to_numpy_descriptor,
        {frame_to_numpy_module},
        {error["frame_method"]},
    )? != __RXTPD_FRAME_TO_NUMPY_FP
    {{
        return Err(__rxtpd_type_error({error["frame_method"]}));
    }}
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
    if instance_dict.contains("map")? || instance_dict.contains("to_numpy")? {{
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
    Ok((name.unbind(), array))
}}

fn __rxtpd_extract_series_f64<'py>(
    py: pyo3::Python<'py>,
    value: &pyo3::Bound<'py, pyo3::PyAny>,
) -> pyo3::PyResult<RxtPandasSeriesF64> {{
    use numpy::PyArrayMethods;
    let (name, array) = __rxtpd_series_parts(py, value, "float64", {error["series_f64"]})?;
    let typed = array
        .cast::<numpy::PyArray1<f64>>()
        .map_err(|_| __rxtpd_type_error({error["series_f64"]}))?;
    let values = typed.readonly().as_array().to_owned();
    Ok(RxtPandasSeriesF64 {{ values, name }})
}}

fn __rxtpd_extract_series_i64<'py>(
    py: pyo3::Python<'py>,
    value: &pyo3::Bound<'py, pyo3::PyAny>,
) -> pyo3::PyResult<RxtPandasSeriesI64> {{
    use numpy::PyArrayMethods;
    let (name, array) = __rxtpd_series_parts(py, value, "int64", {error["series_i64"]})?;
    let typed = array
        .cast::<numpy::PyArray1<i64>>()
        .map_err(|_| __rxtpd_type_error({error["series_i64"]}))?;
    let values = typed.readonly().as_array().to_owned();
    Ok(RxtPandasSeriesI64 {{ values, name }})
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
    fn into_parts(self) -> (numpy::ndarray::Array1<Self::Elem>, pyo3::Py<pyo3::PyAny>);
}}

impl RxtPandasMaterializedSeries for RxtPandasSeriesF64 {{
    type Elem = f64;
    fn into_parts(self) -> (numpy::ndarray::Array1<f64>, pyo3::Py<pyo3::PyAny>) {{
        (self.values, self.name)
    }}
}}

impl RxtPandasMaterializedSeries for RxtPandasSeriesI64 {{
    type Elem = i64;
    fn into_parts(self) -> (numpy::ndarray::Array1<i64>, pyo3::Py<pyo3::PyAny>) {{
        (self.values, self.name)
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
    let (values, name) = value.into_parts();
    let series_class = __rxtpd_pinned_series_class(py)?;
    let array = values.to_pyarray(py);
    let kwargs = PyDict::new(py);
    kwargs.set_item("name", name.bind(py))?;
    series_class.call((array,), Some(&kwargs))
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
    let name = input.name.clone_ref(py);
    let values = py.detach(|| {hot_name}(&input.values));
    Ok({output_struct} {{ values, name }})
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


def dataframe_apply_helpers(
    schema: SchemaMeta,
    meta: CallableMeta,
) -> tuple[str, tuple[str, ...]]:
    """Return a schema-bound pure row loop and its GIL-detaching wrapper."""
    expression = meta.body.expression
    if expression is None:
        raise ValueError("DataFrame.apply lowering requires an available body")
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
        name: py.None(),
    }})
}}"""
    return wrapper_name, (hot, wrapper)


__all__ = ["boundary_helpers", "dataframe_apply_helpers", "series_map_helpers"]
