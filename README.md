# rextio-pandas

`rextio-pandas` is a private, unpublished incubator for audited pandas
`Series.map` lowering. It requires Rextio plugin API 1.3 from exact core commit
`2bd1d1da0cf59e97d1659606bcb1ec12491e032c`, which lives only in the private
`rextio/rextio-core-next` repository; the released `rextio==0.1.2` package
shares that version number but implements API 1.2 and cannot load this plugin.
The dependency is therefore a credential-free exact-commit VCS pin
(`git+https://github.com/rextio/rextio-core-next.git@<commit>`), never a
`rextio>=…` range that could select the published wheel. Development and
evidence are pinned to `pandas==2.3.3` and `numpy==2.3.5`.

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

Both materialized Series types own the shared Rust boundary support through
plugin API 1.3 `PluginType.helpers`. Core therefore emits the extract/type/
materialize definitions for an accepted Series signature even when the
function contains no plugin claim, and exact-text dedup emits the same support
only once when a `Series.map` claim also contributes it.

A separate real-Cargo fixture contains only a parameter-only `SeriesF64`
function and no `Series.map` claim; it proves that type-owned extraction builds,
executes, and enforces the runtime boundary contract independently. Core
intentionally rejects a claimless materialized alias return and calls between
materialized plugin functions (the alias-divergence and RXT092 guards), so
return-only helper collection is verified at source-generation level. Runtime
Series return materialization is exercised honestly through the supported
identity `Series.map` product claim, not described as claimless lowering.

## DataFrame.apply prototype / NO-GO

`DataFrame.apply(axis=1)` is **not a supported native route**. The plugin does
not register `pandas.DataFrame.apply` as a covered symbol, does not register
`DataFrameF64` as a native plugin type, and its normal claim/lower dispatch can
never produce an apply claim. Check/build reports therefore retain apply code
as ordinary Python fallback and never expose a hidden native product route.
The currently shared `boundary_helpers()` text can still place unused prototype
frame definitions in a Series-generated crate; that is not a registered,
claimed, or lowered DataFrame apply hot loop/product route.

The repository retains clearly named private prototype helpers and pinned
characterization evidence for a homogeneous-float64 row loop. That experiment
cannot be promoted safely: the frozen `FrameColumnApply` authority digest
checks module/qualname, MRO names, `axis`, and selected property/method code,
but omits executable behavior such as `apply()`, `__new__`,
`__getattribute__`, forged base behavior, and the complete reachable global
graph. A same-module/same-qualname replacement can copy every digested member,
produce the same digest, add an unchecked `apply()`, and change ordinary pandas
results from `[11.0, 22.0]` to `[-999.0, -999.0]` while a compiled row loop
would remain unchanged. Extending another partial authority graph is not an
acceptable release gate.

The side-effect-free `DataFrameF64[Schema]` marker remains importable only so
the research characterization is reproducible; it is not in the plugin's
registered type vocabulary and conveys no native-support promise. Mixed-row
coercion and NumPy-scalar warning/overflow semantics remain additional NO-GO
constraints.

All annotation markers import without pandas or Rextio. The registered
`SeriesF64`/`SeriesI64` spellings support eager annotations and
`from __future__ import annotations`.

## Benchmarks

`python -m benchmarks.bench_product_routes` builds and measures the actual
generated **Series.map-only** wrapper. Native/fallback pairs include validation,
copies, conversion, result construction/destruction, raw samples, paired
bootstrap intervals, correctness digests, provenance, and a null-call floor.
Vectorized NumPy/pandas and cold/warm Numba remain context-only lanes.
DataFrame.apply has no product cell, context cell, break-even entry, or speedup
claim in the authoritative benchmark.

## Install for development

```bash
python -m pip install -e '.[dev]'
```

Resolving the install requires read access to the private
`rextio/rextio-core-next` repository at the pinned commit; the credential-free
URL means the fetching environment supplies its own git credentials (never a
token embedded in `pyproject.toml`). The dependency-resolution provenance can
be reproduced with `scripts/clean_env_proof.py`, which builds the wheel, installs
it into a throwaway environment while resolving the exact VCS commit (no
`--no-deps`), and asserts the installed core's `direct_url.json` URL/commit,
`PLUGIN_API_VERSION == "1.3"`, the imported `rextio`/`rextio_pandas` paths, and
that the selected `rextio.plugins` entry point is this wheel. This package is
marked `Private :: Do Not Upload` and must not be published while the API 1.3
core remains an incubator.
