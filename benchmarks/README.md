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

The harness builds against exact core commit
`ac2b79d304f13abaaecaf7714f897574c3b6256f`, asserts both check-report routes
are `native-plugin:rextio-pandas`, and includes validation, owned copies,
Python/Rust conversion, pandas result materialization, and destruction inside
the measured wrapper calls. Samples are calibrated to at least 10 ms, GC is
disabled only during samples, native/fallback order is counterbalanced, and
paired bootstrap intervals retain all raw timings and schedules.

`pandas_apply_default`, the semantically equivalent positional-array
`pandas_apply_raw_true`, vectorized NumPy/pandas, and cold/warm Numba are
context-only lanes. They are not Rextio target claims. Only the latest JSON is
kept under `benchmarks/results/latest.json`; it contains correctness digests,
compile time, package/core/plugin SHAs, Rust/Python/package versions, CPU/OS,
the null-call floor, small-input losses, and measured break-even cells.

DataFrame timings are deliberately ineligible for a headline speedup claim:
mixed-row coercion and general NumPy-scalar warning/overflow semantics remain
NO-GO. The benchmark reports the narrow homogeneous-f64 experiment without
extrapolating beyond measured sizes.
