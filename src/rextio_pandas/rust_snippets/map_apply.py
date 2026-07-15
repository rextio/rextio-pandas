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


def _rust_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def boundary_helpers() -> str:
    """Return shared one-shot validation/extraction/materialization helpers."""
    error = {name: _rust_string(message) for name, message in RUNTIME_ERRORS.items()}
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

fn __rxtpd_type_error(message: &'static str) -> pyo3::PyErr {{
    pyo3::exceptions::PyTypeError::new_err(message)
}}

fn __rxtpd_expected_method(
    method: &pyo3::Bound<'_, pyo3::PyAny>,
    expected_module: &str,
    qualname: &str,
) -> pyo3::PyResult<bool> {{
    if method.cast::<pyo3::types::PyFunction>().is_err() {{
        return Ok(false);
    }}
    let module: String = method.getattr("__module__")?.extract()?;
    let actual_qualname: String = method.getattr("__qualname__")?.extract()?;
    Ok(module == expected_module && actual_qualname == qualname)
}}

fn __rxtpd_pinned_series_class<'py>(
    py: pyo3::Python<'py>,
) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {{
    use pyo3::types::PyAnyMethods;
    let pandas = py.import("pandas")?;
    let numpy_module = py.import("numpy")?;
    let pandas_version: String = pandas.getattr("__version__")?.extract()?;
    let numpy_version: String = numpy_module.getattr("__version__")?.extract()?;
    if pandas_version != "2.3.3" || numpy_version != "2.3.5" {{
        return Err(__rxtpd_type_error({error["version"]}));
    }}
    let series_class = pandas.getattr("Series")?;
    let class_dict = series_class.getattr("__dict__")?;
    let map_descriptor = class_dict.get_item("map")?;
    let map_attribute = series_class.getattr("map")?;
    let to_numpy_attribute = series_class.getattr("to_numpy")?;
    if !map_descriptor.is(&map_attribute)
        || !__rxtpd_expected_method(&map_descriptor, "pandas.core.series", "Series.map")?
        || !__rxtpd_expected_method(
            &to_numpy_attribute,
            "pandas.core.base",
            "IndexOpsMixin.to_numpy",
        )?
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
    if pandas_version != "2.3.3" || numpy_version != "2.3.5" {{
        return Err(__rxtpd_type_error({error["version"]}));
    }}
    let frame_class = pandas.getattr("DataFrame")?;
    let class_dict = frame_class.getattr("__dict__")?;
    let apply_descriptor = class_dict.get_item("apply")?;
    let apply_attribute = frame_class.getattr("apply")?;
    let to_numpy_descriptor = class_dict.get_item("to_numpy")?;
    let to_numpy_attribute = frame_class.getattr("to_numpy")?;
    if !apply_descriptor.is(&apply_attribute)
        || !to_numpy_descriptor.is(&to_numpy_attribute)
        || !__rxtpd_expected_method(
            &apply_descriptor,
            "pandas.core.frame",
            "DataFrame.apply",
        )?
        || !__rxtpd_expected_method(
            &to_numpy_descriptor,
            "pandas.core.frame",
            "DataFrame.to_numpy",
        )?
    {{
        return Err(__rxtpd_type_error({error["frame_method"]}));
    }}
    Ok(frame_class)
}}

fn __rxtpd_series_parts<'py>(
    py: pyo3::Python<'py>,
    value: &pyo3::Bound<'py, pyo3::PyAny>,
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
    let (name, array) = __rxtpd_series_parts(py, value)?;
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
    let (name, array) = __rxtpd_series_parts(py, value)?;
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
    let numpy_module = py.import("numpy")?;
    let expected_dtype = numpy_module.getattr("dtype")?.call1(("float64",))?;
    let all_f64: bool = value
        .getattr("dtypes")?
        .call_method1("eq", (expected_dtype,))?
        .call_method0("all")?
        .extract()?;
    if !all_f64 {{
        return Err(__rxtpd_type_error({error["frame_f64"]}));
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
