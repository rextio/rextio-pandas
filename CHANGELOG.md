# Changelog

## 0.0.1 (unreleased private incubator)

### Series.map increment

- Pin the experiment to Rextio core
  `ac2b79d304f13abaaecaf7714f897574c3b6256f` (plugin API 1.3) via a
  credential-free exact-commit VCS pin on the private
  `rextio/rextio-core-next` repository, plus pandas 2.3.3, NumPy 2.3.5, and
  rust-numpy 0.29.0.
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

### Independent-review follow-up

- Point the exact-commit VCS core dependency at the private
  `rextio/rextio-core-next` repository (where the integrated API 1.3 commit
  actually exists) and add `scripts/clean_env_proof.py`, which resolves the
  dependency in a throwaway environment and asserts the core's
  `direct_url.json` commit, plugin API 1.3, imported module paths, and the
  selected entry point.
- Reject every pandas extension dtype/storage (nullable `Float64`/`Int64`,
  numeric `Categorical`, `Sparse`, Arrow-backed) at the runtime boundary by
  requiring an exact `numpy.dtype` instance before conversion, instead of
  trusting whatever `to_numpy()` returns.
- Replace the forgeable `__module__`/`__qualname__` method checks with an
  immutable per-method `co_code` fingerprint captured from the pinned pandas
  2.3.3 descriptors, so `functools.wraps` replacements, deletion, malformed
  descriptors, and instance shadowing all fail with the stable `TypeError`.
- Emit Unicode schema field names as valid Rust `\u{...}` string escapes via a
  dedicated tested encoder rather than JSON `\uXXXX`.
- Strengthen the benchmark: track the final `benchmarks/results/latest.json`,
  gather fail-closed `python -O`-safe provenance (imported module files,
  git commit/dirty, direct URLs, API version, entry point, check-report digest,
  claimed routes, harness digest, toolchain/OS), force a genuinely
  counterbalanced 5:4/4:5 seeded schedule per cell, and make
  `headline_eligible` fail closed on correctness, route/provenance, schedule
  balance, near-floor timing, missing samples, or unstable intervals.

### Adversarial post-review follow-up

- Replace the `co_code`-only method check (bypassable via changed
  defaults/globals while diverging native vs fallback) with an immutable
  semantic fingerprint over the full code-object (bytecode, constants, names,
  varnames, flags, arg counts, stack size, exception table), `__defaults__`,
  `__kwdefaults__`, and module/qualname, plus runtime proof of an empty closure
  and that `__globals__` is the pinned defining module's own dict. The expected
  fingerprint is a frozen known-good CPython 3.11 / pandas 2.3.3 / numpy 2.3.5
  authority constant (a Python-minor gate fails closed off-pin), never
  regenerated from the live descriptor at lowering. Adds real-Cargo regressions
  for changed defaults, changed globals, forged constants, replacement,
  deletion, non-function, and `to_numpy` instance shadowing.
- Make the provenance tests mode-aware (editable proves the resolved checkout
  path + Git SHA; VCS proves the credential-free URL + commit; wheel proves the
  wheel/import/entry-point) instead of matching a directory-name substring.
- Make the benchmark preflight genuinely fail-closed: reject dirty core/plugin
  trees, a wrong installed core/plugin (per parsed `direct_url.json` mode), a
  foreign or non-loading entry point, off-pin pandas/NumPy, and generated/native
  import paths outside the freshly built project — all before any build/timing.
- Preserve and verify both `check.json` and `build.json` (native-build agrees
  with the checked routes), recording report paths/digests, the native artifact
  path/digest, and the imported timing module paths; copy raw reports into
  `benchmarks/results/evidence/` on a full run only.
- Remove the odd-repeat lane bias by assigning the extra first position from the
  per-cell seed (not always native); recompute counterbalance from the raw
  schedule in the eligibility gate.
- Isolate smoke output to an ignored temp location that can never overwrite the
  tracked `latest.json`; only a full run atomically replaces `latest.json` and
  its `evidence/` directory, with cleanup scoped to those harness-owned files.
- Report a *sustained* measured Series break-even (favorable at the size and
  every larger measured size, all eligible; never interpolated) and record the
  installed Numba version.
