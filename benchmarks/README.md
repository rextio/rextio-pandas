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
