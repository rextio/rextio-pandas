# Changelog

## 0.0.1 (unreleased private incubator)

### Series.map increment

- Pin the experiment to Rextio core
  `ac2b79d304f13abaaecaf7714f897574c3b6256f` (plugin API 1.3),
  pandas 2.3.3, NumPy 2.3.5, and rust-numpy 0.29.0.
- Add side-effect-free `SeriesF64`, `SeriesI64`, and future
  `DataFrameF64[Schema]` annotation spellings.
- Add fail-closed static claim/audit logic for one positional scalar project
  UDF passed to `series.map(udf)`.
- Add materialized pandas boundaries with exact runtime class, dtype, index,
  metadata, and method-identity validation.
- Lower supported f64/i64 bodies to deterministic, GIL-detached native loops;
  extraction and pandas result construction each occur once per call.
- Add pinned-version semantic characterization, analyzer/claim/lower tests,
  and real-Cargo product-route equivalence and rejection evidence.

### DataFrame.apply increment

- Characterize pandas 2.3.3 homogeneous/mixed row scalar types, empty-shape
  calls, NumPy-scalar warnings/overflow, signed zero, and F-order behavior
  before lowering.
- Add the exact `DataFrameF64[Schema]` plus
  `frame.apply(row_udf, axis=1)` homogeneous-float64 route.
- Validate exact DataFrame class/dtype/index/columns/schema/default metadata
  and method identity at the one-shot boundary with stable `TypeError`
  failures and no runtime deopt.
- Lower only literal subscripts, finite f64 literals, unary negation,
  comparisons, boolean composition, and conditional expressions into a
  deterministic GIL-detached Rust row loop; all binops/calls and mixed or
  integer frames remain fallback/NO-GO.
- Add real-Cargo product-route tests for one/multiple columns and schemas,
  C/F storage, NaN/Inf/signed-zero, large inputs, fresh native/fallback
  processes, and the full runtime rejection table.
- Add an honest actual-wrapper benchmark harness with >=10 ms calibration,
  counterbalanced pairs, paired bootstrap confidence intervals, raw samples,
  provenance, null-call floor, and default/raw/vectorized/Numba context lanes.
