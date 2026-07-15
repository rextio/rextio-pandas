# rextio-pandas

`rextio-pandas` is a private, unpublished incubator for narrow pandas numeric
lowering. It requires Rextio plugin API 1.3 from exact core commit
`ac2b79d304f13abaaecaf7714f897574c3b6256f`; released `rextio==0.1.2`
implements API 1.2 and cannot load this plugin. Development and evidence are
pinned to `pandas==2.3.3` and `numpy==2.3.5`.

## Series.map surface

The implemented source form is exactly:

```python
from rextio_pandas.types import SeriesF64

def transform(value: float) -> float:
    return value * 2.0 if value > 0.0 else -value

def run(series: SeriesF64) -> SeriesF64:
    return series.map(transform)
```

`SeriesI64` is the corresponding non-nullable NumPy `int64` spelling. An
`int -> int` UDF is limited to full-domain-safe identity/literal/comparison/
boolean/conditional bodies; an `int -> float` conditional is also supported.
`SeriesF64` supports finite float literals, unary negation, same-type
comparisons, boolean composition, conditional expressions, and audited
`+`/`-`/`*`. A native symbol never bypasses this body audit.

The receiver must be a plain local or parameter name with the exact plugin
annotation. The mapper must be one positional bare project-function reference.
Keyword callable forms, `na_action`, lambdas, closures, calls, division,
floor/mod/power/matmul, bit/shift, identity/membership operators, unsupported
side effects, nullable/extension/object storage, subclasses, and noncanonical
indexes stay outside the native route.

At runtime, accepted inputs must be an exact, nonempty `pandas.Series` with
exact NumPy `float64` or `int64` storage, an unnamed
`RangeIndex(0, len, 1)`, empty `.attrs`,
`flags.allows_duplicate_labels is True`, and an unmodified public
`Series.map`/`Series.to_numpy` method identity. `Series.name` is `None` or
`str`. Contract misses raise stable `TypeError` messages; they do not silently
deopt. Strided arrays are copied by logical ndarray indexing into owned Rust
storage.

The boundary validates and extracts once, a deterministic helper runs the
complete UDF in one GIL-detached Rust loop with no Python callback, and pandas
materializes the result once. The successful result preserves exact Series
class, dtype, values (including NaN/Inf/signed zero), order, RangeIndex, name,
and default metadata.

## DataFrame.apply surface

The second implemented source form is exactly:

```python
from rextio_pandas.types import DataFrameF64, SeriesF64

class Row:
    left: float
    right: float

def choose(row) -> float:
    return row["left"] if row["left"] >= 0.0 else -row["right"]

def run(frame: DataFrameF64[Row]) -> SeriesF64:
    return frame.apply(choose, axis=1)
```

The schema is a nonempty simple class whose unique ordered fields are all
`float`. The row parameter is unannotated and may be read only as
`row["field"]`. The closed body allows finite float literals, unary negation,
same-type comparisons, boolean composition, and a float conditional. Every
binary operation, scalar call, attribute/dynamic/missing field, non-float
branch, mixed/int/object/nullable column, axis other than the non-bool integer
literal `1`, positional/omitted/dynamic axis, keyword callable, and extra
keyword (`raw`, `result_type`, `args`, or kwargs) remains outside the route.

At runtime the input must be an exact, nonempty `pandas.DataFrame` with at
least one homogeneous non-nullable NumPy `float64` column. Columns are unique
ordered strings exactly matching the schema and `columns.name is None`; the
same canonical index/default-metadata/method-identity restrictions as Series
map. C/F/strided storage is copied by logical ndarray indexing. Contract
misses raise the documented stable `TypeError`, never a silent deopt.

The output is an exact float64 Series with default name/metadata and the
canonical input RangeIndex. The row UDF runs in a GIL-detached Rust
`outer_iter()` loop without pandas indexing or Python calls. Mixed numeric rows
are deliberately NO-GO because pandas coerces them before the UDF; homogeneous
integer rows are NO-GO because NumPy-scalar warnings and overflow semantics
are observable.

All annotation markers import without pandas or Rextio. Both eager annotations
and `from __future__ import annotations` are supported by the static analyzer.

## Benchmarks

`python -m benchmarks.bench_product_routes` builds and measures the actual
generated wrapper. Native/fallback pairs include validation, copies,
conversion, result construction/destruction, raw samples, paired bootstrap
intervals, correctness digests, provenance, and a null-call floor. Default
apply, a semantically equivalent positional `apply(raw=True)` UDF, vectorized
NumPy/pandas, and cold/warm Numba are context-only lanes. DataFrame results are
not eligible for a headline speedup claim while mixed-row and general
NumPy-scalar semantics remain unresolved.

## Install for development

```bash
python -m pip install -e '.[dev]'
```

The project dependency uses a credential-free exact Git commit URL. This
package is marked `Private :: Do Not Upload` and must not be published while
the API 1.3 core remains an incubator.
