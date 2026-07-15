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

