# Changelog

## 0.0.1 (unreleased private incubator)

### API 1.3 finalization

- Advance every dependency, provenance, clean-environment, and benchmark gate
  to integrated core commit `2bd1d1da0cf59e97d1659606bcb1ec12491e032c`.
- Make `Series.map` the sole supported/GO route. Both registered materialized
  Series types now own the shared boundary support through API 1.3
  `PluginType.helpers`, so claimless parameter signatures and return-position
  source generation resolve their Rust types/extractors/materializers. A
  claimless-only real-Cargo fixture covers parameter extraction and rejection;
  return-only collection remains a source-generation test because core's
  alias-divergence/RXT092 guards intentionally forbid a claimless materialized
  round trip. Runtime return materialization is covered by an identity
  `Series.map` product claim. Exact text remains deduplicated when a map claim
  contributes the same helper.
- De-promote `DataFrame.apply(axis=1)` to a private prototype/NO-GO. Remove it
  from coverage, the registered type vocabulary, normal claim/lower dispatch,
  native rule records, product-route Cargo assertions, and every authoritative
  benchmark cell/headline/break-even. Check/build retains ordinary fallback.
  Shared boundary text may still emit unused prototype frame definitions, but
  no DataFrame apply hot loop/product route is registered, claimed, lowered, or
  benchmarked.
- Record the decisive authority bypass: a same-module/same-qualname
  `FrameColumnApply` can copy every member in the frozen class digest yet add an
  unchecked `apply()` that changes pandas results from `[11.0, 22.0]` to
  `[-999.0, -999.0]`. The partial class/base/global authority graph is therefore
  not a sound product gate and is not extended in this pass.

### Series.map increment

- Pin the experiment to Rextio core
  `2bd1d1da0cf59e97d1659606bcb1ec12491e032c` (plugin API 1.3) via a
  credential-free exact-commit VCS pin on the private
  `rextio/rextio-core-next` repository, plus pandas 2.3.3, NumPy 2.3.5, and
  rust-numpy 0.29.0.
- Add side-effect-free registered `SeriesF64`/`SeriesI64` spellings and a
  research-only, unregistered `DataFrameF64[Schema]` marker.
- Add fail-closed static claim/audit logic for one positional scalar project
  UDF passed to `series.map(udf)`.
- Add materialized pandas boundaries with exact runtime class, dtype, index,
  metadata, and method-identity validation.
- Lower supported f64/i64 bodies to deterministic, GIL-detached native loops;
  extraction and pandas result construction each occur once per call.
- Add pinned-version semantic characterization, analyzer/claim/lower tests,
  and real-Cargo product-route equivalence and rejection evidence.

### DataFrame.apply prototype history (now NO-GO)

- Characterize pandas 2.3.3 homogeneous/mixed row scalar types, empty-shape
  calls, NumPy-scalar warnings/overflow, signed zero, and F-order behavior
  before lowering.
- Prototype the exact `DataFrameF64[Schema]` plus
  `frame.apply(row_udf, axis=1)` homogeneous-float64 experiment.
- Validate exact DataFrame class/dtype/index/columns/schema/default metadata
  and method identity at the one-shot boundary with stable `TypeError`
  failures and no runtime deopt.
- Lower only literal subscripts, finite f64 literals, unary negation,
  comparisons, boolean composition, and conditional expressions into a
  deterministic GIL-detached Rust row loop; all binops/calls and mixed or
  integer frames remain fallback/NO-GO.
- Retain source/semantic characterization for one/multiple columns and schemas,
  C/F storage, NaN/Inf/signed-zero, and the runtime rejection table as private
  research evidence; real-Cargo product assertions now prove non-promotion.
- Keep the authoritative actual-wrapper benchmark harness Series-only, with
  >=10 ms calibration, counterbalanced pairs, paired bootstrap confidence
  intervals, raw samples, provenance, and a null-call floor.

The review entries below preserve the chronological prototype history. Their
references to DataFrame native/Cargo product regressions describe the earlier
experiment and are superseded by the finalization decision above; current
Cargo evidence asserts fallback/non-promotion, while authority logic remains
as unit/source characterization only.

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

### Second adversarial post-review follow-up

- Remove every user-controlled `repr` from the method authority check. The
  code-identity digest is now a type-tagged, length-delimited canonical encoding
  built in Rust from exact PyO3 type checks (only the builtin const types the
  pinned code objects actually contain, with recursive exact tuples) and hashed
  with the core-managed `sha2` crate — no Python `repr`/`hashlib`. Function
  defaults are validated structurally item-by-item (exact builtin type and value,
  `None`/`False` singletons with no bool/int interchange, exact strings/tuples,
  and the exact `pandas._libs.lib.no_default` object), separately from the digest.
  Fixes the custom-`__repr__` default collision; adds real-Cargo regressions for
  the custom-`__repr__` `None`, an `int` masquerading as `False`, and a forged
  sentinel.
- Validate every execution-relevant global binding of the trusted methods.
  The `LOAD_GLOBAL`/import set is derived from the pinned bytecode and hard-coded
  as an auditable binding spec; each binding must be identical to the canonical
  authority object from its independently named defining module (or builtins),
  numpy/lib must be the canonical module objects, and `DataFrame.apply`'s
  `pandas.core.apply.frame_apply` import target is validated. A test re-derives
  the spec from live bytecode so drift cannot add an unchecked global. Adds
  real-Cargo regressions mutating `pandas.core.base.np`, `isna`, `ExtensionDtype`,
  `pandas.core.frame.np`, and `pandas.core.apply.frame_apply` (restored in
  `finally`).
- Close the benchmark artifact binding gap: the timed `_rextio_native.__file__`
  must resolve exactly equal to `build.json`'s `installed_path`, and the timed
  Python wrapper must be the exact generated `kernels.py`. `native_artifact.sha256`
  is now the standard raw-bytes SHA-256 via a dedicated helper; the harness
  manifest digest is kept separately as `harness_manifest_sha256`.
- Validate every raw schedule pair (not only first positions): a pair must be
  exactly `[native, fallback]` or `[fallback, native]`; duplicate/unknown/missing/
  extra lanes set `is_counterbalanced` false and fail eligibility closed.
- Replace the editable plugin path `startswith` check with resolved
  `Path.is_relative_to(checkout / "src")`, with a sibling-prefix (`src-evil`)
  negative test.

### Third adversarial post-review follow-up

- Bind `DataFrame.apply`'s `pandas.core.apply.frame_apply` import target to its
  own independently frozen canonical authority instead of accepting it on
  metadata alone. The previous import check accepted any exact `FunctionType`
  whose `__module__`/`__qualname__` matched and whose `__globals__` reused
  `pandas.core.apply.__dict__`, so a `types.FunctionType` replacement with those
  matching attributes but an attacker-controlled code object was trusted. The
  imported function is now validated exactly like the four bound methods: frozen
  known-good code digest (type-tagged canonical encoding hashed with `sha2`; no
  Python `repr`/`hashlib`), exact function type/module/qualname, empty closure
  and zero freevars, `None` `__kwdefaults__`, structural `__defaults__`, and
  checks of its own execution-relevant global bindings (`FrameColumnApply`,
  `FrameRowApply`, `reconstruct_func`). The import-target
  validator key is derived from the frozen authority metadata so a target and
  its canonical authority cannot drift apart. Adds a real-Cargo regression that
  installs a metadata-and-globals-matching forged `frame_apply` and asserts the
  stable DataFrame `TypeError`, a unit test proving the forged code object
  changes the digest, and re-derives the new digest and global spec from the
  live pinned package so drift fails loudly. Preserves the prior custom-`repr`,
  structural-default, sentinel, and mutable-binding regressions. (Those three
  `pandas.core.apply` binding checks were only equal-to-themselves comparisons,
  not independent authorities; the fourth follow-up below replaces them.)

### Fourth adversarial post-review follow-up

- Replace the self-comparison in `frame_apply`'s `pandas.core.apply` global
  checks with genuinely independent frozen authorities. The prior check obtained
  each of `FrameColumnApply`, `FrameRowApply`, and `reconstruct_func` from the
  validated function's `__globals__` (which *is* `pandas.core.apply.__dict__`)
  and compared it for identity to the same attribute re-read from a fresh import
  of `pandas.core.apply` — the same object on both sides, so any same-module
  replacement passed. Because the then-prototyped `DataFrame.apply(axis=1)` route
  validates but never executes `frame_apply`, such a replacement changed ordinary
  pandas/fallback behavior while native lowering kept the compiled semantics.
  Now `reconstruct_func` is validated by a full frozen code-digest function
  authority (exact function type, module, qualname, empty closure/zero freevars,
  `None` kw/positional defaults, and the frozen type-tagged code digest), and
  `FrameColumnApply`/`FrameRowApply` by a frozen type-tagged *structural class
  digest* rebuilt from the live class — its identity, full MRO chain, the `axis`
  selector, and the code digests of the read-only property getters
  (`result_columns`, `result_index`, `series_generator`) and the plain method
  (`wrap_results_for_axis`) that drive the axis dispatch — so a same-module
  replacement that changes the class name/hierarchy, the axis constant, or any
  covered method body is rejected. A bare `__module__`/`__qualname__` check is
  not relied upon. The shared canonical code encoder now folds nested code
  objects (comprehensions/nested defs) into the digest recursively, and none of
  the expected digests is derived from the live mutable binding at lowering or
  runtime. Adds real-Cargo regressions replacing each of the three members in a
  fresh native process (each asserting the stable DataFrame `TypeError`), unit
  characterizations that a same-module class/function replacement changes the
  frozen digest, and drift tests that re-derive the class/function authority
  digests from the pinned package so a version bump fails loudly. Preserves the
  frozen `frame_apply` code-digest validation and all prior custom-`repr`,
  structural-default, sentinel, globals, import-target, artifact, and scheduling
  hardening.
