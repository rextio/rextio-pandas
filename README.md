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

`DataFrameF64[Schema]` is already a side-effect-free annotation marker, but
`DataFrame.apply` is not implemented in the Series increment.

## Install for development

```bash
python -m pip install -e '.[dev]'
```

The project dependency uses a credential-free exact Git commit URL. This
package is marked `Private :: Do Not Upload` and must not be published while
the API 1.3 core remains an incubator.
