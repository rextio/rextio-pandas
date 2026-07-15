# Product-route benchmark

Run the actual generated plugin routes against the exact original pandas call
through the same generated wrapper:

```bash
python -m benchmarks.bench_product_routes
```

For a fast correctness/provenance gate:

```bash
python -m benchmarks.bench_product_routes --smoke
```

A fail-closed preflight (no Python `assert`, so it survives `python -O`) runs
before any measurement. It builds against exact core commit
`ac2b79d304f13abaaecaf7714f897574c3b6256f`, requires both check-report routes to
be `native-plugin:rextio-pandas`, and records the actual imported
`rextio.__file__`/`rextio_pandas.__file__`, the core/plugin git commit and dirty
state, the core and plugin `direct_url.json`, `PLUGIN_API_VERSION == "1.3"`, the
selected `rextio.plugins` entry point/import path, the `check.json` SHA-256 and
claimed native routes, the benchmark harness SHA-256, the Rust toolchain, and
the Python/pandas/NumPy versions and CPU/OS. Each measured wrapper call includes
validation, owned copies, Python/Rust conversion, pandas result materialization,
and destruction. Samples are calibrated to at least 10 ms and GC is disabled
only during samples.

For every nine-repeat cell the native-first/fallback-first order is genuinely
counterbalanced — the counts are forced to differ by at most one (5:4 or 4:5)
and the balanced orders are seeded-shuffled, never drawn independently per
repeat — and each schedule plus its balance summary is recorded. Paired
bootstrap intervals retain all raw timings.

`headline_eligible` is fail-closed: a cell qualifies only when it is the exact
Series route, its route/provenance is verified, native and fallback correctness
digests match, every required sample is present and positive, the schedule is
counterbalanced, both medians clear the null-call floor by a safety multiple,
and the bootstrap interval is narrow. Any failure records the blocking reasons
in `headline_ineligible_reasons` and keeps the cell out of headlines and
break-even.

`pandas_apply_default`, the semantically equivalent positional-array
`pandas_apply_raw_true`, vectorized NumPy/pandas, and cold/warm Numba are
context-only lanes — never Rextio target claims; Numba honestly reports itself
unavailable when it cannot be imported. Only the latest JSON is kept under
`benchmarks/results/latest.json` (now trackable so the director can preserve the
final raw evidence); it contains correctness digests, compile time, the full
provenance block above, the null-call floor, small-input losses, and measured
break-even cells.

DataFrame timings are deliberately ineligible for a headline speedup claim:
mixed-row coercion and general NumPy-scalar warning/overflow semantics remain
NO-GO. The benchmark reports the narrow homogeneous-f64 experiment without
extrapolating beyond measured sizes.
