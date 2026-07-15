# Product-route benchmark

Run the actual generated Series.map route against the exact original pandas call
through the same generated wrapper:

```bash
python -m benchmarks.bench_product_routes
```

For a fast correctness/provenance gate:

```bash
python -m benchmarks.bench_product_routes --smoke
```

A **fail-closed** preflight (no Python `assert`, so it survives `python -O`)
runs before any build or timing and *rejects* — not merely records — an invalid
state. It requires exact clean core and plugin Git worktrees; the core HEAD at
`2bd1d1da0cf59e97d1659606bcb1ec12491e032c`; `PLUGIN_API_VERSION == "1.3"`;
pandas 2.3.3 / NumPy 2.3.5; that the imported `rextio`/`rextio_pandas` resolve
under the expected editable checkout (or, for a VCS/wheel install, that the
parsed `direct_url.json` proves the exact credential-free URL and commit or the
built wheel and hash); and that the selected `rextio.plugins` entry point loads
this exact plugin object. A dirty or provenance-invalid run stops before timing
and cannot produce an eligible result. After the build it verifies the
`check.json` routes and the `build.json` native-build report agree (status
`built`, accepted native count, zero rejections), hashes both reports, and
confirms the generated Python module and native extension artifact imported for
timing resolve under the freshly built project (not a cache or global install).
The normalized provenance plus both report digests/paths, the native artifact
path/digest, and the harness SHA-256 are recorded; raw `check.json`/`build.json`
bodies are copied into `benchmarks/results/evidence/` only on a full run.

The timed native wrapper validates the complete reachable Series authority once
per public call: `Series.map`, inherited `_map_values`, `algorithms.map_array`,
Cython `lib.map_infer`, `_constructor`, inherited `__finalize__`, and
`to_numpy`. Plain Python members use PyO3's non-mutable exact `PyFunction`
authority; the Cython callable uses its exact callable type plus frozen
type/metatype structure. Native result construction then shares the fallback
`_constructor(..., index=source.index, copy=False).__finalize__(source,
method="map")` envelope without repeating validation on the normal
source-carrying path. Optimize level 1 is separately frozen; `-OO` is
unsupported and fails closed.

For every nine-repeat cell the native-first/fallback-first order is genuinely
counterbalanced — the counts differ by at most one (5:4 or 4:5) and, for odd
counts, the *extra* first position is assigned deterministically from the
per-cell seed (not always native), then the balanced orders are seeded-shuffled.
The raw schedule is recorded and eligibility recomputes the balance from it
rather than trusting any precomputed summary.

`headline_eligible` is fail-closed: a cell qualifies only when it is the exact
Series route, its route/build/provenance is verified, native and fallback
correctness digests match, every required sample is present, positive, and
finite, the raw schedule is counterbalanced, both medians clear the null-call
floor by a safety multiple, and the paired-bootstrap interval is finite,
correctly ordered, and narrow. Any failure records the blocking reasons in
`headline_ineligible_reasons` and keeps the cell out of headlines and
break-even.

The reported **sustained measured break-even** for the Series route is the
smallest measured size whose paired-bootstrap 95% CI is wholly below 1.0 and
remains so at every larger measured size, with every required larger cell
eligible; it is never interpolated, and is `none` otherwise.

## Authoritative full result (2026-07-16)

The tracked schema-4 non-smoke run used plugin commit
`c1ae2e734c48f795d4c4ca418ba4cf20f53b4b93`, core commit
`2bd1d1da0cf59e97d1659606bcb1ec12491e032c`, API 1.3, nine paired repetitions,
the 10 ms minimum calibration target, and seed `20260715`. The route report is exactly
`native-plugin:rextio-pandas`; all six cells are headline-eligible, their
native/fallback correctness digests match, and every recorded schedule is a
valid 5:4 or 4:5 counterbalance. The build accepted one native route and
rejected none.

| Size | Native median (µs/call) | pandas fallback median (µs/call) | Paired native/fallback ratio | Paired 95% CI | Interpretation |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 84.939 | 9.324 | 9.074455 | 8.828482–9.236807 | native 9.07× slower |
| 10 | 85.634 | 10.019 | 8.503623 | 8.345566–8.847237 | native 8.50× slower |
| 100 | 86.281 | 15.655 | 5.520391 | 5.311342–5.544713 | native 5.52× slower |
| 1,000 | 86.051 | 73.879 | 1.165308 | 1.154215–1.202724 | native 1.17× slower |
| 10,000 | 101.898 | 654.865 | 0.155077 | 0.154832–0.158761 | native 6.45× faster |
| 100,000 | 244.310 | 6,729.854 | 0.035900 | 0.035272–0.037772 | native 27.86× faster |

The ratio is the median of the nine within-pair native/fallback ratios, not the
quotient of the two separately rounded lane medians.

The result includes the **small-input losses** at 1, 10, 100, and 1,000
elements. Its **sustained measured break-even is 10,000 elements** because that
cell and the larger measured cell have an eligible CI wholly below 1.0. No
crossing is interpolated between 1,000 and 10,000, and nothing is extrapolated
past 100,000. Complete per-public-call authority validation supersedes the
historical 1,000-element threshold: native is about 1.17× slower at 1,000,
6.45× faster at 10,000, and 27.86× faster at 100,000.
Vectorized NumPy/pandas and warm Numba were faster at every measured size; they
remain context-only lanes, not Rextio performance targets.

The authoritative files are [latest.json](results/latest.json),
[`evidence/check.json`](results/evidence/check.json), and
[`evidence/build.json`](results/evidence/build.json). The report hashes in
`latest.json` match the retained files. Absolute temporary build paths and the
native-artifact digest are intentionally capture-time provenance; the temporary
build tree and binary are not retained after the run, so the repository does
not claim that those paths remain live or that the binary can be rehashed later.

Vectorized NumPy/pandas and cold/warm Numba are context-only lanes — never
Rextio target claims; the installed Numba version is recorded and Numba
honestly reports itself unavailable when it cannot be imported.

Only a successful **full** run atomically replaces the tracked
`benchmarks/results/latest.json` and its `evidence/` directory (and nothing
else). A `--smoke` run writes only to an ignored temporary location and can
never delete or overwrite the authoritative result. The tracked JSON contains
correctness digests, compile time, the full provenance block above, the
null-call floor, small-input losses, and the sustained break-even.

DataFrame.apply is a prototype/NO-GO because the complete executable pandas
authority cannot be bounded by the prototype's partial digest. It is excluded
entirely: no product cell, context lane, break-even entry, or speedup claim is
written by this harness.
