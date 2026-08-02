# rextio-pandas

<p align="center">
  <img src="https://raw.githubusercontent.com/rextio/rextio-pandas/main/assets/readme/rextio-icon.png" width="96" alt="Rextio icon">
</p>

<p align="center"><strong>Native Rust lowering for a deliberately narrow, audited `pandas.Series.map` surface.</strong></p>

<p align="center">
  English · <a href="https://github.com/rextio/rextio-pandas/blob/main/README.ko.md">한국어</a> · <a href="https://github.com/rextio/rextio-pandas/blob/main/README.zh-hans.md">简体中文</a> · <a href="https://github.com/rextio/rextio-pandas/blob/main/README.zh-hant.md">繁體中文</a> · <a href="https://github.com/rextio/rextio-pandas/blob/main/README.ja.md">日本語</a>
</p>

`rextio-pandas` compiles exact numeric/boolean `Series.map` calls into a GIL-detached Rust loop when the receiver, mapper, dtype, index, and pandas runtime authority all match the audited contract. Call shapes not claimed during analysis stay on ordinary pandas fallback; after a native route is selected, a runtime contract miss raises a stable `TypeError` rather than silently deoptimizing.

> **Public Alpha 0.1.2** (released 2026-07-26). **CPython 3.11 only**; packaging rejects 3.12+. Requires `rextio>=0.1.3,<0.2`, `pandas==2.3.3`, and `numpy==2.3.5`.
>
> **Visible boundary:** `DataFrame.apply(axis=1)`, `Series.where`, `Series.mask`, and broader DataFrame operations are **not native routes**. They stay on Python fallback.

## Measured proof

The retained historical benchmark measures one single-stage `SeriesF64 -> SeriesF64` mapper through the real generated wrapper:

| Elements | Native vs pandas fallback |
| ---: | ---: |
| 1 | 9.80× slower |
| 10 | 9.18× slower |
| 100 | 5.91× slower |
| 1,000 | 1.35× slower |
| 10,000 | 6.15× faster |
| 100,000 | 26.86× faster |

The sustained measured break-even is **10,000 elements**, limited to those measured sizes and that exact single-stage F64 case. It is not evidence for boolean lanes, multi-stage pipelines, other workloads, or unmeasured sizes. Vectorized NumPy/pandas and warm Numba were faster at every measured size and are context-only comparisons, not Rextio target claims.

## How it works

1. Rextio recognizes an exact plugin-annotated `Series.map` call and audits the project mapper's closed expression grammar.
2. Each public native call validates the complete reachable pandas method/runtime authority once while extracting the input.
3. One GIL-detached Rust loop runs without Python callbacks. A two- or four-stage supported pipeline keeps intermediates in Rust and materializes pandas only once at the end.

Contract misses do not silently deopt inside a selected native call: the boundary fails closed with stable `TypeError` messages. Calls not claimed at analysis time remain normal Python fallback.

## Quick start

```bash
python3.11 -m pip install "rextio-pandas==0.1.2"
```

```toml
# rextio.toml
[rust]
build_tool = "cargo"

[plugins]
enabled = ["rextio-pandas"]
```

```python
from rextio_pandas.types import SeriesF64

def absolute(value: float) -> float:
    return value if value >= 0.0 else -value

def normalize(series: SeriesF64) -> SeriesF64:
    return series.map(absolute)
```

```bash
rextio build .
```

The mapper must be one positional bare project-function reference. Omitted `na_action` and literal `na_action=None` are supported; `na_action="ignore"`, dynamic values, keyword mapper forms, and lambdas remain fallback.

## Supported `Series.map` surface

| Receiver | Supported result | Mapper grammar summary |
| --- | --- | --- |
| `SeriesF64` | `SeriesF64`, `SeriesBool`, `SeriesI64` | finite float literals, unary negation, same-type comparisons, boolean composition/conditionals, audited `+ - *`; int result only through literal-safe conditionals |
| `SeriesI64` | `SeriesI64`, `SeriesBool`, `SeriesF64` | full-domain-safe identity/literal/comparison/boolean/conditional bodies; float result only through a supported conditional |
| `SeriesBool` | `SeriesBool`, `SeriesI64`, `SeriesF64` | bool parameter/literals, `not`, `and`/`or`, equality/inequality, conditionals; numeric results select exact literals |

Integer literals must fit signed int64; float literals must be finite. There is no float-to-int coercion. Calls, division, floor/mod/power/matmul, bit/shift, identity/membership operators, side effects, closures, and unaudited bodies are outside the native grammar.

## Runtime boundary

Accepted input must be:

- an exact, nonempty `pandas.Series` with exact NumPy `float64`, `int64`, or `bool` storage;
- indexed by an unnamed `RangeIndex(0, len, 1)`;
- `attrs == {}`, `flags.allows_duplicate_labels is True`, and `name` equal to `None` or `str`;
- backed by the pinned, unmodified reachable pandas authority graph.

Nullable/extension/object/categorical/Arrow storage, subclasses, empty Series, arbitrary indexes/MultiIndex, and noncanonical metadata are unsupported. Strided arrays are copied by logical indexing into owned Rust storage. Results preserve exact Series class, dtype, values (including NaN/Inf/signed zero), order, RangeIndex, name, and default metadata.

CPython `-O` has a separately frozen supported authority digest. `-OO` is intentionally unsupported and fails closed. This provider does not advertise standalone artifact capability.

## Explicit fallback surface

- `DataFrame.apply(axis=1)` is not registered, claimed, lowered, or benchmarked as a product route. The importable research marker does not imply support.
- `Series.where` and `Series.mask` remain fallback because their alignment/casting authority graph and empty-Series semantics are not certified.
- Binary-Series operations, nullable data, arbitrary pandas call options, unsupported mappers, and every unlisted pandas/DataFrame operation remain fallback.

## Benchmark evidence

See the [benchmark method](benchmarks/README.md) and [authoritative schema-4 result](benchmarks/results/latest.json). The run includes validation, copies, conversion, result construction/destruction, paired samples, correctness digests, route/build evidence, provenance, and a null-call floor. Small-input losses are retained rather than hidden.

## Development

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest
.venv/bin/ruff check src tests benchmarks
.venv/bin/mypy src
```

See [CHANGELOG.md](CHANGELOG.md) for release history.

## License

MIT
