# Changelog

## Unreleased

- Restore compatibility with Core 0.1.5 / plugin API 1.4 while continuing to
  declare provider API 1.3 and require `rextio>=0.1.3,<0.2`.
- Defer provider compatibility enforcement to the Core loader. The benchmark
  and clean-environment proof now independently fail closed unless Core has
  the same API major and a minor of at least 3.
- Reject standalone/non-PyO3 lowering explicitly. This provider does not
  declare an artifact capability.
- Keep the NO-GO `DataFrame.apply` warning characterization bounded to the two
  exact NumPy 2.3.5 platform observations: scalar remainder is silent on
  macOS arm64 and emits four `RuntimeWarning`s on Linux x86_64.

## 0.1.0 (2026-07-17)

First **public alpha** release of `rextio-pandas` (PyPI package `rextio-pandas`,
repository [rextio/rextio-pandas](https://github.com/rextio/rextio-pandas)).

### Public packaging

- Bump the single version authority to `0.1.0` and publish-ready metadata:
  Development Status Alpha classifiers, public project URLs, and no
  `Private :: Do Not Upload` marker.
- Declare truthful runtime support only: `requires-python = ">=3.11,<3.12"` and
  the Python 3.11 classifier (no 3.12/3.13 classifiers). The generated native
  Series.map runtime is CPython 3.11-only and rejects other minors.
- Depend on public `rextio>=0.1.3,<0.2` (plugin API 1.3) instead of a private
  exact-commit VCS pin on `rextio-core-next`.
- Keep development pins `pandas==2.3.3` and `numpy==2.3.5`.
- Benchmark harness and clean-env proof use the **installed** `rextio` package
  by default; optional `REXTIO_CORE_ROOT` / `--find-links` overrides are
  documented and never hard-code a machine-local absolute path.
- Product boundary for this release: audited numeric `Series.map` is the sole
  GO route; `DataFrame.apply(axis=1)` remains NO-GO / ordinary fallback.
- Authoritative sustained measured break-even remains **10,000 rows** with the
  recorded ratios and provenance from the 2026-07-16 schema-4 evidence (not
  rewritten for this packaging release).

### Series.map authority closure (2026-07-16)

- Close the reachable `Series.map` authority chain over inherited
  `IndexOpsMixin._map_values`, `pandas.core.algorithms.map_array`, Cython
  `pandas._libs.lib.map_infer`, `Series._constructor`, inherited
  `NDFrame.__finalize__`, and `Series.to_numpy`. Plain Python authorities now
  require PyO3's exact, non-mutable CPython `PyFunction` type instead of the
  user-mutable `types.FunctionType` module attribute, and each function's cached
  `__builtins__` must be the exact canonical `builtins` module dictionary. A
  real-Cargo regression reproduces and rejects both the coordinated
  forged-callable attack and a canonical-code/globals/defaults replacement whose
  privately cached builtins changed fallback output while native execution
  remained unchanged.
- Independently validate every Python builtin name consumed by that frozen
  authority graph. Canonical `builtins.__dict__` container identity remains
  necessary but not sufficient: each resolved binding is checked by a
  structural/C-level authority (exact `builtin_function_or_method` plus fixed
  module/name/qualname and canonical builtins-module `__self__` for C-function
  names; PyO3 static type anchors for loaded types and exceptions) rather than
  by comparing two live lookups from the mutable `builtins` mapping. Real-Cargo
  regressions reject pure-Python `builtins.len` mutation both before and after
  native-module import with the stable Series-method `TypeError`. Malicious
  native extensions capable of fabricating CPython builtin objects remain out
  of scope.
- Bind the Cython `map_infer` authority to its exact callable type plus a frozen
  type/metatype structure, including immutable layout/MRO anchors, so replacing
  the callable and its live type name together does not pass validation.
- Carry the source Series through native execution and materialize through the
  exact fallback envelope:
  `_constructor(values, index=source.index, copy=False).__finalize__(source,
  method="map")`. The full authority graph is validated once per public call at
  extraction and is not repeated on the normal source-carrying materialization
  path. CPython optimize level 1 uses a separately frozen `__finalize__` digest;
  `-OO` remains unsupported and fails closed.

### Current authoritative Series.map benchmark evidence (2026-07-16)

- Retain the schema-4 full-run JSON plus hash-bound `check.json`/`build.json`
  evidence for plugin commit `35d651b1684c6a48a6222e19635df853840aed8e`,
  core `2bd1d1da0cf59e97d1659606bcb1ec12491e032c`, and API 1.3. The run used nine
  paired repetitions, a 10 ms minimum calibration target, and seed `20260715`;
  all six cells are headline-eligible and correctness-matched, with one
  accepted native route and zero rejections.
- Record native/fallback medians in microseconds and paired ratios (95% CI):
  1 element `93.938/9.610`, `9.802362` (`9.589628–10.017574`); 10 elements
  `96.817/10.773`, `9.183213` (`7.893036–9.481407`); 100 elements
  `92.650/15.694`, `5.914319` (`5.702861–6.057157`); 1,000 elements
  `100.707/74.341`, `1.354658` (`1.295011–1.412292`); 10,000 elements
  `109.263/666.875`, `0.162690` (`0.157637–0.166345`); and 100,000 elements
  `254.839/6,924.520`, `0.037236` (`0.036711–0.037654`).
- Set the sustained measured break-even to **10,000 elements**. Complete
  per-public-call validation supersedes the historical 1,000-element threshold:
  native is about 1.35× slower at 1,000, 6.15× faster at 10,000, and 26.86×
  faster at 100,000. Vectorized NumPy/pandas and Numba remain context-only lanes,
  never Rextio target claims.

### Prior authoritative Series.map benchmark evidence (2026-07-16; superseded by 35d651b)

- Retain the schema-4 full-run numbers previously bound to plugin commit
  `c1ae2e734c48f795d4c4ca418ba4cf20f53b4b93`, core
  `2bd1d1da0cf59e97d1659606bcb1ec12491e032c`, and API 1.3 (nine paired
  repetitions, 10 ms calibration target, seed `20260715`; all six cells
  headline-eligible). That run predated the builtin-authority fix at
  `35d651b1684c6a48a6222e19635df853840aed8e`.
- Recorded native/fallback medians and paired ratios (95% CI) were: 1 element
  `84.939/9.324`, `9.074455` (`8.828482–9.236807`); 10 elements
  `85.634/10.019`, `8.503623` (`8.345566–8.847237`); 100 elements
  `86.281/15.655`, `5.520391` (`5.311342–5.544713`); 1,000 elements
  `86.051/73.879`, `1.165308` (`1.154215–1.202724`); 10,000 elements
  `101.898/654.865`, `0.155077` (`0.154832–0.158761`); and 100,000 elements
  `244.310/6,729.854`, `0.035900` (`0.035272–0.037772`). Sustained measured
  break-even was **10,000 elements** (about 1.17× slower at 1,000, 6.45× faster
  at 10,000, and 27.86× faster at 100,000).

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

### Historical Series.map benchmark evidence (2026-07-15; superseded)

- Retain the schema-4 full-run JSON plus hash-bound `check.json`/`build.json`
  evidence for plugin commit `152ff9a0457ae4b4d83bfa2b21429cfee784a931`, core
  `2bd1d1da0cf59e97d1659606bcb1ec12491e032c`, and API 1.3. The sole checked
  product route is `native-plugin:rextio-pandas`; all six cells are
  headline-eligible with matching native/fallback correctness digests.
- Record the small-input losses honestly: paired native/fallback ratios are
  6.798× at 1 element, 6.330× at 10, and 4.127× at 100. The native route becomes
  favourable at the first larger measured point (0.890× at 1,000), then reaches
  0.126× at 10,000 and 0.032× at 100,000.
- Set the sustained measured break-even to **1,000 elements**, with no
  interpolation between 100 and 1,000 and no extrapolation beyond 100,000.
  Vectorized NumPy/pandas and warm Numba remain faster at every measured size
  and are context-only; DataFrame.apply has no benchmark cell or performance
  claim.

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
