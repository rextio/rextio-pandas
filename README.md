# rextio-pandas

`rextio-pandas` is a **public alpha** Rextio plugin that lowers an audited
pandas `Series.map` slice to native Rust. It requires **CPython 3.11 only**
(`requires-python = ">=3.11,<3.12"`) and declares Rextio provider API 1.3 via
the public package range `rextio>=0.1.3,<0.2`. It is compatible with Core API
1.3 and later 1.x minors (including Core 0.1.5 / API 1.4). Development and evidence are pinned to
`pandas==2.3.3` and `numpy==2.3.5`. Other CPython minors (including 3.12 and
3.13) are intentionally unsupported for this release.

| Status | Meaning |
| --- | --- |
| **GO** | Supported numeric `Series.map` product route (this release) |
| **NO-GO** | `DataFrame.apply(axis=1)` — ordinary Python fallback only |

This is an honest alpha: the supported surface is narrow, small inputs can be
slower than pure pandas, and `DataFrame.apply` is deliberately not a native
product route.

## Install

```bash
# Use a CPython 3.11 interpreter (3.12+ will be rejected by packaging metadata)
python3.11 -m pip install rextio-pandas
```

For development:

```bash
python3.11 -m pip install -e '.[dev]'
```

Requires CPython 3.11 and a working Rust toolchain (`cargo` / `rustc`) when
Rextio builds native extensions for accepted routes.

Repository: [github.com/rextio/rextio-pandas](https://github.com/rextio/rextio-pandas).
PyPI package: `rextio-pandas`.

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
`flags.allows_duplicate_labels is True`, and an unmodified reachable pandas
authority graph. That graph covers `Series.map`, inherited
`IndexOpsMixin._map_values`, `pandas.core.algorithms.map_array`, Cython
`pandas._libs.lib.map_infer`, `Series._constructor`, inherited
`NDFrame.__finalize__`, and `Series.to_numpy`. Plain Python members require the
exact non-mutable CPython `PyFunction` type plus their frozen executable and
global-binding authorities. Every builtin name those frozen authorities load is
validated independently against a structural/C-level rule (not by comparing two
lookups from the live mutable `builtins` mapping); pure-Python mutation of
`builtins.__dict__` before or after native-module import therefore fails closed
with the stable Series-method `TypeError`. Malicious native extensions capable
of fabricating CPython builtin objects remain out of scope. The Cython callable
requires its exact callable type and a frozen type/metatype structure, so
coordinated replacement of the live type anchor is rejected. `Series.name` is
`None` or `str`. Contract misses raise stable `TypeError` messages; they do not
silently deopt. Strided arrays are copied by logical ndarray indexing into
owned Rust storage.

Every public native call performs the complete authority validation once while
extracting the input. A deterministic helper then runs the complete UDF in one
GIL-detached Rust loop with no Python callback, and pandas materializes the
result once through the same envelope as ordinary `Series.map`:
`source._constructor(values, index=source.index, copy=False).__finalize__(source,
method="map")`. The normal source-carrying path does not repeat validation at
materialization. CPython `-O` (`optimize=1`) has a separately frozen
`NDFrame.__finalize__` digest; `-OO` is intentionally unsupported and fails
closed. The successful result preserves exact Series class, dtype, values
(including NaN/Inf/signed zero), order, RangeIndex, name, and default metadata.

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

The repository retains clearly named prototype helpers and pinned
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

The harness uses the **installed** `rextio` package by default (public range
`>=0.1.3,<0.2`, provider API 1.3 with compatible Core API 1.3+). For optional extra git provenance from a local
core checkout, set `REXTIO_CORE_ROOT` to that directory; no machine-local path
is hard-coded.

### Authoritative Series-only result (2026-07-16)

The retained full run is schema 4 at plugin commit
`35d651b1684c6a48a6222e19635df853840aed8e` and core commit
`2bd1d1da0cf59e97d1659606bcb1ec12491e032c`, with nine counterbalanced paired
samples per size, a 10 ms minimum calibration target, and seed `20260715`. All
six cells passed the fail-closed headline-eligibility gates and had identical
native/fallback correctness digests. The check/build evidence reports one
accepted native route and zero rejected routes under plugin API 1.3.

| Size | Native median (µs) | pandas fallback median (µs) | Paired native/fallback ratio (95% CI) | Result |
| ---: | ---: | ---: | ---: | --- |
| 1 | 93.938 | 9.610 | 9.802362 (9.589628–10.017574) | native 9.80× slower |
| 10 | 96.817 | 10.773 | 9.183213 (7.893036–9.481407) | native 9.18× slower |
| 100 | 92.650 | 15.694 | 5.914319 (5.702861–6.057157) | native 5.91× slower |
| 1,000 | 100.707 | 74.341 | 1.354658 (1.295011–1.412292) | native 1.35× slower |
| 10,000 | 109.263 | 666.875 | 0.162690 (0.157637–0.166345) | native 6.15× faster |
| 100,000 | 254.839 | 6,924.520 | 0.037236 (0.036711–0.037654) | native 26.86× faster |

The paired ratio is the median of the nine within-pair ratios, not the quotient
of the separately rounded lane medians.

The **sustained measured break-even is 10,000 elements**: it is the first
measured size whose paired 95% CI is wholly below 1.0 and remains so at every
larger measured size. This is not an interpolated claim about the interval
between 1,000 and 10,000, nor an extrapolation beyond 100,000. Complete
per-public-call correctness validation supersedes the earlier 1,000-element
threshold: native is still about 1.35× slower at 1,000, then about 6.15× faster
at 10,000 and 26.86× faster at 100,000. The small-input losses are part of the
result, not excluded from the headline evidence.
Vectorized NumPy/pandas and warm Numba were faster than the native route at
every measured size, but remain explicitly labeled context-only rather than
Rextio target claims. See [the benchmark evidence](benchmarks/results/latest.json)
and its retained `check.json`/`build.json` reports.

## Clean-environment dependency proof

```bash
python scripts/clean_env_proof.py
# optional offline/pre-release: python scripts/clean_env_proof.py --find-links /path/to/wheels
```

The script requires a CPython 3.11 host (other minors fail immediately with a
clear message). It builds this package's wheel, installs it into a throwaway
environment with full dependency resolution, and asserts `rextio` is in range,
compatible Core plugin API (major 1, minor at least 3), import paths, and the
`rextio.plugins` entry point.
