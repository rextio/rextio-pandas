"""Benchmark actual generated native routes against exact wrapper fallback.

This harness builds one real Rextio fixture, imports its generated wrapper, and
times the same public function under ``REXTIO_NATIVE_MODE=native`` and
``fallback``. Each sample is calibrated to at least 10 ms, native/fallback
order is counterbalanced, and paired bootstrap intervals are retained with raw
samples and provenance. Context lanes are never reported as Rextio claims.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata as importlib_metadata
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from rextio.plugins.testing import build_certification_project

from benchmarks.cases import KERNEL_SOURCE, BenchmarkCase, make_case

ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = Path("/Volumes/Data/workspace/rextio/rextio-core-next")
CORE_SHA = "2bd1d1da0cf59e97d1659606bcb1ec12491e032c"
CORE_VCS_URL = "https://github.com/rextio/rextio-core-next.git"

# Authoritative tracked output for a full run; smoke never touches these.
RESULTS_DIR = ROOT / "benchmarks" / "results"
FULL_RESULT = RESULTS_DIR / "latest.json"
EVIDENCE_DIR = RESULTS_DIR / "evidence"
# Only these result files are ever removed/replaced by a full run.
_HARNESS_OWNED_RESULTS = (FULL_RESULT, EVIDENCE_DIR)


# Eligibility is fail-closed. A headline cell must clear every one of these
# gates; anything unproven leaves the cell out of the headline set.
_NEAR_FLOOR_MULTIPLE = 5.0
_MAX_CI_WIDTH_FRACTION = 0.25


def _require(condition: object, message: str) -> None:
    """Fail-closed runtime gate that survives ``python -O`` (no ``assert``)."""
    if not condition:
        raise RuntimeError(f"benchmark preflight failed: {message}")


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _normalize_dist(name: str) -> str:
    return name.lower().replace("_", "-")


def _distribution_direct_url(distribution: str) -> dict | None:
    target = _normalize_dist(distribution)
    for dist in importlib_metadata.distributions():
        name = dist.metadata["Name"]
        if name is None or _normalize_dist(name) != target:
            continue
        raw = dist.read_text("direct_url.json")
        if raw:
            return json.loads(raw)
    return None


def _file_url_to_path(url: str) -> Path:
    parts = urlsplit(url)
    _require(parts.scheme == "file", f"expected a file:// URL, got {url!r}")
    return Path(unquote(parts.path)).resolve()


def _git_sha(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _git_dirty(path: Path) -> bool:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=path,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return bool(status.strip())


def _command_text(command: Sequence[str]) -> str:
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    return completed.stdout.strip() or completed.stderr.strip()


def _sha256_files(paths: Sequence[Path]) -> str:
    # A multi-file *manifest* digest (name + NUL framing). NOT a raw file hash.
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    """Standard SHA-256 of a single file's raw bytes (no name/framing)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_bound_native_artifact(installed_path: str, imported_file: str) -> Path:
    """Bind the timed native module to this build's exact artifact by bytes.

    Core stages byte-identical copies of the compiled extension at both the
    reported ``installed_path`` (``.rextio/generated/python``) and the import
    tree (``.rextio/build/python``), so their paths are not equal. Requiring the
    timed module to be byte-identical to the reported build artifact is stronger
    than a path prefix: any different native file (stale, cached, or global) has
    different bytes and is rejected.
    """
    installed = Path(installed_path).resolve()
    imported = Path(imported_file).resolve()
    _require(installed.exists(), f"reported native artifact missing: {installed}")
    _require(imported.exists(), f"imported native module missing: {imported}")
    _require(
        _sha256_file(imported) == _sha256_file(installed),
        f"timed native module {imported} is not byte-identical to the build artifact {installed}",
    )
    return imported


def _counterbalanced_schedule(repetitions: int, rng: random.Random) -> list[list[str]]:
    """Return a genuinely counterbalanced native/fallback order schedule.

    The native-first and fallback-first counts are forced to differ by at most
    one (e.g. 5:4 or 4:5 for nine repeats) instead of being drawn independently
    per repeat. For an odd repeat count the *extra* first position is assigned
    deterministically from the per-cell seed (not always to native), so no lane
    is systematically favoured across cells. The balanced orders are then
    seeded-shuffled so ordering cannot correlate with lane.
    """
    base = repetitions // 2
    native_first = base
    fallback_first = base
    if repetitions % 2 == 1:
        if rng.random() < 0.5:
            native_first += 1
        else:
            fallback_first += 1
    orders = [["native", "fallback"] for _ in range(native_first)]
    orders += [["fallback", "native"] for _ in range(fallback_first)]
    rng.shuffle(orders)
    return orders


_VALID_SCHEDULE_PAIRS = (("native", "fallback"), ("fallback", "native"))


def _schedule_balance(schedule: Sequence[Sequence[str]]) -> dict[str, Any]:
    # Every raw pair must be exactly one of the two valid orderings; a malformed
    # pair (duplicate/unknown/missing/extra lane) cannot look "balanced" via a
    # first-position count.
    all_pairs_valid = all(
        isinstance(order, (list, tuple)) and tuple(order) in _VALID_SCHEDULE_PAIRS
        for order in schedule
    )
    native_first = sum(1 for order in schedule if len(order) >= 1 and order[0] == "native")
    fallback_first = sum(1 for order in schedule if len(order) >= 1 and order[0] == "fallback")
    return {
        "native_first": native_first,
        "fallback_first": fallback_first,
        "all_pairs_valid": all_pairs_valid,
        "is_counterbalanced": (
            all_pairs_valid
            and len(schedule) > 0
            and native_first + fallback_first == len(schedule)
            and abs(native_first - fallback_first) <= 1
        ),
    }


def _headline_eligibility(
    cell: dict[str, Any],
    *,
    repetitions: int,
    null_floor_median_ns: float,
) -> tuple[bool, list[str]]:
    """Fail-closed headline gate; returns ``(eligible, blocking_reasons)``."""
    reasons: list[str] = []
    product = cell["product"]

    if cell["route"] != "series.map":
        # The authoritative harness has one product surface. Unknown/stale
        # route cells remain fail-closed if an older artifact is inspected.
        reasons.append("route-not-headline-surface")
    if not cell.get("route_verified", False):
        reasons.append("route-or-provenance-unverified")

    digest = product.get("correctness_digest", {})
    if digest.get("native") != digest.get("fallback") or not digest.get("native"):
        reasons.append("correctness-mismatch")

    samples = product.get("samples_ns_per_call", {})
    native = samples.get("native", [])
    fallback = samples.get("fallback", [])
    if len(native) != repetitions or len(fallback) != repetitions:
        reasons.append("missing-required-sample")
    if not all(_finite(value) and value > 0.0 for value in [*native, *fallback]):
        reasons.append("nonpositive-or-nonfinite-sample")

    # Recompute counterbalance from the raw schedule rather than trusting a
    # precomputed summary, and require it to match the number of paired repeats.
    schedule = product.get("schedule", [])
    balance = _schedule_balance(schedule)
    if len(schedule) != repetitions or not balance["is_counterbalanced"]:
        reasons.append("schedule-not-counterbalanced")

    medians = product.get("median_ns_per_call", {})
    floor = null_floor_median_ns * _NEAR_FLOOR_MULTIPLE
    if not all(_finite(medians.get(lane)) for lane in ("native", "fallback")):
        reasons.append("nonfinite-median")
    elif any(medians[lane] < floor for lane in ("native", "fallback")):
        reasons.append("near-null-floor")

    boot = product.get("paired_bootstrap", {})
    center = boot.get("median_native_over_fallback")
    low = boot.get("ci95_low")
    high = boot.get("ci95_high")
    if not (_finite(center) and _finite(low) and _finite(high)) or center <= 0.0:
        reasons.append("missing-or-nonfinite-bootstrap")
    elif low > high:
        reasons.append("reversed-ci-bounds")
    elif (high - low) > _MAX_CI_WIDTH_FRACTION * center:
        reasons.append("unstable-wide-ci")

    return (not reasons, reasons)


def _sustained_break_even(cells: list[dict[str, Any]]) -> int | None:
    """Smallest Series size that is a *sustained* measured break-even.

    A size qualifies only when that Series cell and every larger measured Series
    cell are headline-eligible with a paired-bootstrap 95% CI wholly below 1.0
    (native faster). This never interpolates; if any required larger cell is
    ineligible or not favourable, there is no sustained break-even.
    """
    series_cells = sorted(
        (cell for cell in cells if cell["route"] == "series.map"),
        key=lambda cell: cell["size"],
    )

    def favourable(cell: dict[str, Any]) -> bool:
        boot = cell["product"].get("paired_bootstrap", {})
        high = boot.get("ci95_high")
        return bool(cell["headline_eligible"]) and _finite(high) and high < 1.0

    for index, cell in enumerate(series_cells):
        if all(favourable(later) for later in series_cells[index:]):
            return int(cell["size"])
    return None


def _digest(result: pd.Series) -> str:
    payload = hashlib.sha256()
    payload.update(type(result).__qualname__.encode())
    payload.update(str(result.dtype).encode())
    payload.update(result.to_numpy(copy=False).tobytes(order="C"))
    payload.update(repr((result.index.start, result.index.stop, result.index.step)).encode())
    payload.update(repr(result.index.name).encode())
    payload.update(repr(result.name).encode())
    return payload.hexdigest()


def _assert_exact(left: pd.Series, right: pd.Series) -> None:
    assert_series_equal(left, right, check_exact=True)
    if left.dtype == np.dtype("float64"):
        if not np.array_equal(
            left.to_numpy(copy=False).view(np.uint64),
            right.to_numpy(copy=False).view(np.uint64),
        ):
            raise AssertionError("float bit patterns differ")


def _timed_batch(function: Callable[[], object], loops: int) -> int:
    gc.collect()
    enabled = gc.isenabled()
    gc.disable()
    try:
        start = time.perf_counter_ns()
        result: object | None = None
        for _ in range(loops):
            result = function()
        del result
        elapsed = time.perf_counter_ns() - start
    finally:
        if enabled:
            gc.enable()
    return elapsed


def _calibrate(function: Callable[[], object], target_ns: int) -> int:
    loops = 1
    while _timed_batch(function, loops) < target_ns:
        loops *= 2
        if loops > 1_048_576:
            break
    return loops


def _sample(function: Callable[[], object], loops: int) -> float:
    return _timed_batch(function, loops) / loops


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _paired_bootstrap(
    native: list[float],
    fallback: list[float],
    *,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    ratios = [left / right for left, right in zip(native, fallback, strict=True)]
    rng = random.Random(seed)
    boot: list[float] = []
    for _ in range(iterations):
        draw = [ratios[rng.randrange(len(ratios))] for _ in ratios]
        boot.append(statistics.median(draw))
    return {
        "median_native_over_fallback": statistics.median(ratios),
        "ci95_low": _percentile(boot, 0.025),
        "ci95_high": _percentile(boot, 0.975),
    }


def _set_mode(mode: str) -> None:
    os.environ["REXTIO_NATIVE_MODE"] = mode
    os.environ["REXTIO_DISABLE_BOUNDARY_FALLBACK"] = "1"


def _product_samples(
    function: Callable[[object], pd.Series],
    argument: object,
    *,
    repetitions: int,
    target_ns: int,
    seed: int,
) -> dict[str, Any]:
    def native_call() -> pd.Series:
        return function(argument)

    def fallback_call() -> pd.Series:
        return function(argument)

    _set_mode("native")
    native_result = native_call()
    _set_mode("fallback")
    fallback_result = fallback_call()
    _assert_exact(native_result, fallback_result)

    _set_mode("native")
    native_loops = _calibrate(native_call, target_ns)
    _set_mode("fallback")
    fallback_loops = _calibrate(fallback_call, target_ns)

    rng = random.Random(seed)
    schedule = _counterbalanced_schedule(repetitions, rng)
    samples: dict[str, list[float]] = {"native": [], "fallback": []}
    for order in schedule:
        for lane in order:
            _set_mode(lane)
            if lane == "native":
                samples[lane].append(_sample(native_call, native_loops))
            else:
                samples[lane].append(_sample(fallback_call, fallback_loops))
    return {
        "loops": {"native": native_loops, "fallback": fallback_loops},
        "schedule": schedule,
        "schedule_balance": _schedule_balance(schedule),
        "samples_ns_per_call": samples,
        "median_ns_per_call": {lane: statistics.median(values) for lane, values in samples.items()},
        "paired_bootstrap": _paired_bootstrap(
            samples["native"],
            samples["fallback"],
            iterations=1000 if repetitions >= 5 else 200,
            seed=seed + 1000,
        ),
        "correctness_digest": {
            "native": _digest(native_result),
            "fallback": _digest(fallback_result),
        },
    }


def _context_samples(
    contexts: tuple[tuple[str, Callable[[], pd.Series]], ...],
    *,
    repetitions: int,
    target_ns: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, function in contexts:
        loops = _calibrate(function, target_ns)
        samples = [_sample(function, loops) for _ in range(repetitions)]
        result = function()
        output[name] = {
            "classification": "context-only; not a Rextio target claim",
            "loops": loops,
            "samples_ns_per_call": samples,
            "median_ns_per_call": statistics.median(samples),
            "correctness_digest": _digest(result),
        }
    return output


def _numba_context(case: BenchmarkCase, target_ns: int, repetitions: int) -> dict[str, Any]:
    try:
        import numba
    except ImportError:
        return {
            "available": False,
            "version": None,
            "reason": "numba is not installed",
            "classification": "context-only; not a Rextio target claim",
        }

    values = case.argument.to_numpy(copy=False)

    @numba.njit
    def kernel(data: np.ndarray) -> np.ndarray:
        output = np.empty(data.shape[0], dtype=np.float64)
        for index in range(data.shape[0]):
            value = data[index]
            output[index] = value * 2.0 if value > 0.0 else -value
        return output

    def call() -> pd.Series:
        return pd.Series(kernel(values), index=case.argument.index, name=case.argument.name)

    cold_start = time.perf_counter_ns()
    cold_result = call()
    cold_ns = time.perf_counter_ns() - cold_start
    loops = _calibrate(call, target_ns)
    samples = [_sample(call, loops) for _ in range(repetitions)]
    return {
        "available": True,
        "version": numba.__version__,
        "classification": "context-only; JIT compile is not a Rextio target claim",
        "cold_compile_and_call_ns": cold_ns,
        "warm_loops": loops,
        "warm_samples_ns_per_call": samples,
        "warm_median_ns_per_call": statistics.median(samples),
        "correctness_digest": _digest(cold_result),
    }


def _write_project(root: Path) -> None:
    (root / "rextio.toml").write_text(
        '[rust]\nbuild_tool = "cargo"\n\n[plugins]\nenabled = ["rextio-pandas"]\n',
        encoding="utf-8",
    )
    package = root / "src" / "bench_app"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "kernels.py").write_text(KERNEL_SOURCE, encoding="utf-8")


def _preflight() -> dict[str, Any]:
    """Reject any invalid state before building or timing (survives ``python -O``).

    Dirty trees, a wrong installed core/plugin, a mismatched direct-URL, a
    foreign entry point, or off-pin pandas/NumPy all stop the run here so a stale
    or different package can never be measured and mis-attributed to the pinned
    checkout SHAs.
    """
    import rextio
    import rextio_pandas
    from rextio.plugins.api import PLUGIN_API_VERSION
    from rextio_pandas.plugin import plugin as this_plugin

    _require(
        PLUGIN_API_VERSION == "1.3",
        f"core advertises plugin API {PLUGIN_API_VERSION!r}, need '1.3'",
    )
    _require(pd.__version__ == "2.3.3", f"pandas is {pd.__version__!r}, need '2.3.3'")
    _require(np.__version__ == "2.3.5", f"numpy is {np.__version__!r}, need '2.3.5'")

    # Exact clean core and plugin worktrees, pinned once as the measured code.
    _require(not _git_dirty(CORE_ROOT), "core-next worktree is dirty")
    _require(not _git_dirty(ROOT), "plugin worktree is dirty")
    core_sha = _git_sha(CORE_ROOT)
    plugin_sha = _git_sha(ROOT)
    _require(core_sha == CORE_SHA, f"core-next HEAD {core_sha} != required {CORE_SHA}")

    core_file = Path(rextio.__file__).resolve()
    plugin_file = Path(rextio_pandas.__file__).resolve()

    core_direct_url = _distribution_direct_url("rextio")
    _require(core_direct_url is not None, "core has no direct_url.json provenance")
    core_mode = _validate_core_direct_url(core_direct_url, core_file)

    plugin_direct_url = _distribution_direct_url("rextio-pandas")
    _require(plugin_direct_url is not None, "plugin has no direct_url.json provenance")
    plugin_mode = _validate_plugin_direct_url(plugin_direct_url, plugin_file)

    entry_points = [
        {"name": ep.name, "value": ep.value, "dist": ep.dist.name if ep.dist else None}
        for ep in importlib_metadata.entry_points(group="rextio.plugins")
        if ep.name == "rextio-pandas"
    ]
    _require(
        entry_points
        and all(
            ep["value"] == "rextio_pandas.plugin:plugin" and ep["dist"] == "rextio-pandas"
            for ep in entry_points
        ),
        f"rextio-pandas entry point is not provided by this checkout: {entry_points}",
    )
    loaded = [
        ep
        for ep in importlib_metadata.entry_points(group="rextio.plugins")
        if ep.name == "rextio-pandas"
    ]
    _require(
        all(ep.load() is this_plugin for ep in loaded),
        "rextio-pandas entry point does not load this exact plugin object",
    )

    return {
        "core_sha": core_sha,
        "core_dirty": False,
        "plugin_sha": plugin_sha,
        "plugin_dirty": False,
        "core_install_mode": core_mode,
        "plugin_install_mode": plugin_mode,
        "core_import_file": str(core_file),
        "plugin_import_file": str(plugin_file),
        "core_direct_url": core_direct_url,
        "plugin_direct_url": plugin_direct_url,
        "plugin_api_version": PLUGIN_API_VERSION,
        "selected_entry_points": entry_points,
        "harness_manifest_sha256": _sha256_files(
            [Path(__file__).resolve(), ROOT / "benchmarks" / "cases.py"]
        ),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "python": sys.version,
        "rustc": _command_text(["rustc", "--version"]),
        "cargo": _command_text(["cargo", "--version"]),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def _validate_core_direct_url(direct_url: dict, core_file: Path) -> str:
    """Fail-closed check of the core direct URL for the active install mode."""
    if direct_url.get("dir_info", {}).get("editable"):
        checkout = _file_url_to_path(direct_url["url"])
        _require(
            core_file.is_relative_to(checkout),
            f"imported rextio {core_file} is not under editable checkout {checkout}",
        )
        _require(
            checkout.resolve() == CORE_ROOT.resolve(),
            f"editable core checkout {checkout} != expected {CORE_ROOT}",
        )
        return "editable"
    if "vcs_info" in direct_url:
        _require(direct_url["url"] == CORE_VCS_URL, f"core VCS URL {direct_url['url']!r}")
        _require("@" not in urlsplit(direct_url["url"]).netloc, "core URL leaks a credential")
        _require(direct_url["vcs_info"].get("vcs") == "git", "core direct URL is not git")
        _require(
            direct_url["vcs_info"].get("commit_id") == CORE_SHA,
            f"core VCS commit {direct_url['vcs_info'].get('commit_id')!r} != {CORE_SHA}",
        )
        return "vcs"
    raise RuntimeError(f"benchmark preflight failed: unrecognized core provenance: {direct_url}")


def _validate_plugin_direct_url(direct_url: dict, plugin_file: Path) -> str:
    """Fail-closed check of the plugin direct URL for the active install mode."""
    if direct_url.get("dir_info", {}).get("editable"):
        checkout = _file_url_to_path(direct_url["url"])
        _require(
            checkout.resolve() == ROOT.resolve(),
            f"editable plugin checkout {checkout} != this WP-5 checkout {ROOT}",
        )
        _require(
            plugin_file.is_relative_to(checkout / "src"),
            f"imported rextio_pandas {plugin_file} is not under {checkout / 'src'}",
        )
        return "editable"
    if "archive_info" in direct_url:
        wheel = direct_url["url"].rsplit("/", 1)[-1]
        _require(wheel.endswith(".whl"), f"plugin archive is not a wheel: {wheel}")
        _require("rextio_pandas-" in wheel, f"plugin wheel is not rextio_pandas: {wheel}")
        _require(bool(direct_url["archive_info"].get("hashes")), "plugin wheel has no hash")
        return "wheel"
    raise RuntimeError(f"benchmark preflight failed: unrecognized plugin provenance: {direct_url}")


def _bench(args: argparse.Namespace) -> dict[str, Any]:
    provenance = _preflight()
    sizes = [10, 1000] if args.smoke else [1, 10, 100, 1000, 10_000, 100_000]
    repetitions = 3 if args.smoke else args.repetitions
    target_ns = int(args.target_ms * 1_000_000)
    started = time.perf_counter_ns()
    with tempfile.TemporaryDirectory(prefix="rextio-pandas-bench-") as directory:
        project_root = Path(directory).resolve()
        _write_project(project_root)
        compile_started = time.perf_counter_ns()
        project = build_certification_project(project_root)
        compile_ns = time.perf_counter_ns() - compile_started
        reports_dir = project_root / ".rextio" / "reports"
        check_path = reports_dir / "check.json"
        build_path = reports_dir / "build.json"
        check_bytes = check_path.read_bytes()
        report = json.loads(check_bytes.decode("utf-8"))
        routes = {
            function["qualname"]: function["route"]
            for module in report["modules"]
            for function in module["functions"]
        }
        expected_routes = {
            "bench_app.kernels.series_map": "native-plugin:rextio-pandas",
        }
        for qualname, expected_route in expected_routes.items():
            _require(
                routes.get(qualname) == expected_route,
                f"{qualname} routed {routes.get(qualname)!r}, expected {expected_route!r}",
            )

        # Preserve and verify the native-build report; it must agree with check.
        _require(build_path.exists(), "build.json is missing")
        build_bytes = build_path.read_bytes()
        build_report = json.loads(build_bytes.decode("utf-8"))
        native_build = build_report.get("native_build", {})
        _require(build_report.get("status") == "built", "build status is not 'built'")
        _require(native_build.get("status") == "built", "native build did not complete")
        _require(
            build_report.get("accepted_native_count", 0) >= len(expected_routes),
            "build report accepted fewer native functions than the checked routes",
        )
        _require(
            build_report.get("rejected_native_count", 0) == 0,
            "build report rejected a native function",
        )
        native_artifact = Path(native_build["installed_path"]).resolve()
        _require(native_artifact.exists(), f"native artifact missing: {native_artifact}")
        _require(
            native_artifact.is_relative_to(project_root),
            "native artifact resolves outside the freshly built project",
        )
        generated_python_dir = (project_root / ".rextio" / "generated" / "python").resolve()

        provenance["check_report"] = {
            "path": str(check_path),
            "sha256": hashlib.sha256(check_bytes).hexdigest(),
            "routes": {qualname: routes[qualname] for qualname in expected_routes},
        }
        provenance["build_report"] = {
            "path": str(build_path),
            "sha256": hashlib.sha256(build_bytes).hexdigest(),
            "status": build_report.get("status"),
            "native_build_status": native_build.get("status"),
            "accepted_native_count": build_report.get("accepted_native_count"),
            "rejected_native_count": build_report.get("rejected_native_count"),
        }
        provenance["native_artifact"] = {
            "path": str(native_artifact),
            # Standard raw-bytes SHA-256 of the artifact file (not a manifest).
            "sha256": _sha256_file(native_artifact),
        }
        provenance["generated_python_dir"] = str(generated_python_dir)
        # Back-compat aliases retained for readers of the previous schema.
        provenance["check_report_sha256"] = provenance["check_report"]["sha256"]
        provenance["claimed_native_routes"] = provenance["check_report"]["routes"]
        provenance["compile_ns"] = compile_ns
        provenance["report_evidence_files"] = {
            "check.json": check_bytes.decode("utf-8"),
            "build.json": build_bytes.decode("utf-8"),
        }

        sys.path.insert(0, str(project.build_python_dir))
        try:
            module = importlib.import_module("bench_app.kernels")
            # The timed Python wrapper must be the exact generated file, and the
            # timed native extension must be the exact built artifact -- not
            # merely something under the same project tree, and never a cache or
            # global install.
            build_python_dir = Path(project.build_python_dir).resolve()
            expected_kernels = (build_python_dir / "bench_app" / "kernels.py").resolve()
            _require(
                Path(module.__file__).resolve() == expected_kernels,
                f"imported kernels {module.__file__} is not the generated {expected_kernels}",
            )
            native_module = sys.modules.get("_rextio_native")
            _require(native_module is not None, "native module was not imported")
            imported_native = Path(native_module.__file__).resolve()
            # Imported from the exact fresh build tree (not a cache/global), and
            # byte-identical to the reported build artifact.
            _require(
                imported_native.parent == build_python_dir,
                f"native module {imported_native} is not imported from {build_python_dir}",
            )
            _require_bound_native_artifact(native_build["installed_path"], native_module.__file__)
            provenance["timing_imports"] = {
                "kernels": str(Path(module.__file__).resolve()),
                "native_module": str(imported_native),
                "native_module_matches_build_artifact": True,
            }
            cells: list[dict[str, Any]] = []
            route = "series.map"
            for size_index, size in enumerate(sizes):
                case = make_case(route, size)
                function = getattr(module, case.function_name)
                product = _product_samples(
                    function,
                    case.argument,
                    repetitions=repetitions,
                    target_ns=target_ns,
                    seed=args.seed + size_index,
                )
                fallback_digest = product["correctness_digest"]["fallback"]
                contexts = _context_samples(
                    case.contexts,
                    repetitions=repetitions,
                    target_ns=target_ns,
                )
                for context in contexts.values():
                    context["matches_fallback_digest"] = (
                        context["correctness_digest"] == fallback_digest
                    )
                numba = _numba_context(case, target_ns, repetitions)
                if numba.get("available"):
                    numba["matches_fallback_digest"] = (
                        numba["correctness_digest"] == fallback_digest
                    )
                cells.append(
                    {
                        "route": route,
                        "size": size,
                        "product": product,
                        "contexts": contexts,
                        "numba": numba,
                        "route_verified": routes.get(f"bench_app.kernels.{case.function_name}")
                        == "native-plugin:rextio-pandas",
                        # Filled in fail-closed after the null-call floor exists.
                        "headline_eligible": False,
                        "headline_ineligible_reasons": ["not-yet-evaluated"],
                        "headline_note": "verified exact Series route",
                    }
                )
        finally:
            sys.path.remove(str(project.build_python_dir))
            for name in list(sys.modules):
                if name == "_rextio_native" or name == "bench_app" or name.startswith("bench_app."):
                    sys.modules.pop(name, None)

    def null() -> None:
        return None

    null_loops = _calibrate(null, target_ns)
    null_samples = [_sample(null, null_loops) for _ in range(repetitions)]
    null_floor_median_ns = statistics.median(null_samples)

    for cell in cells:
        eligible, reasons = _headline_eligibility(
            cell,
            repetitions=repetitions,
            null_floor_median_ns=null_floor_median_ns,
        )
        cell["headline_eligible"] = eligible
        cell["headline_ineligible_reasons"] = reasons

    # Series requires favourability at the size and every larger measured size
    # (no interpolation). No prototype route enters the result schema.
    sustained_break_even: dict[str, int | None] = {
        "series.map": _sustained_break_even(cells),
    }

    return {
        "schema": 4,
        "smoke": args.smoke,
        "generated_at_unix_ns": time.time_ns(),
        "elapsed_ns": time.perf_counter_ns() - started,
        "configuration": {
            "sizes": sizes,
            "repetitions": repetitions,
            "minimum_sample_target_ms": args.target_ms,
            "seed": args.seed,
            "gc_disabled_during_samples": True,
            "counterbalanced_pairs": True,
            "near_floor_multiple": _NEAR_FLOOR_MULTIPLE,
            "max_ci_width_fraction": _MAX_CI_WIDTH_FRACTION,
        },
        "provenance": provenance,
        "null_call_floor": {
            "loops": null_loops,
            "samples_ns_per_call": null_samples,
            "median_ns_per_call": null_floor_median_ns,
        },
        "cells": cells,
        "sustained_break_even": sustained_break_even,
        "sustained_break_even_definition": (
            "smallest measured Series size whose paired-bootstrap 95% CI is wholly "
            "below 1.0 and remains so at every larger measured Series size, with "
            "every required larger cell headline-eligible; not interpolated"
        ),
    }


def _pop_report_evidence(result: dict[str, Any]) -> dict[str, str]:
    """Detach the raw report bodies so they land in evidence files, not JSON."""
    provenance = result.get("provenance", {})
    return provenance.pop("report_evidence_files", {})


def _write_smoke_output(result: dict[str, Any], output: Path | None) -> Path:
    """Write a smoke result only to an ignored/temp location.

    A smoke run must never delete or overwrite the tracked authoritative
    ``latest.json`` (or its evidence directory).
    """
    if output is None:
        output = Path(tempfile.mkdtemp(prefix="rextio-pandas-smoke-")) / "smoke.json"
    output = output.resolve()
    for owned in _HARNESS_OWNED_RESULTS:
        _require(
            output != owned.resolve() and not output.is_relative_to(EVIDENCE_DIR.resolve()),
            f"smoke output {output} must not touch the authoritative result {owned}",
        )
    _pop_report_evidence(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _write_full_output(result: dict[str, Any]) -> Path:
    """Atomically replace only the harness-owned authoritative result + evidence.

    The tree was already validated clean by the preflight, so a previous tracked
    result does not block a rerun: this replaces just ``latest.json`` and the
    ``evidence/`` directory (never an arbitrary glob of the output folder).
    """
    evidence = _pop_report_evidence(result)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Refresh only the harness-owned evidence directory.
    if EVIDENCE_DIR.exists():
        shutil.rmtree(EVIDENCE_DIR)
    EVIDENCE_DIR.mkdir(parents=True)
    for name, body in evidence.items():
        (EVIDENCE_DIR / name).write_text(body, encoding="utf-8")

    # Atomic replace of the tracked result file.
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    tmp = FULL_RESULT.with_suffix(".json.tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, FULL_RESULT)
    return FULL_RESULT


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--repetitions", type=int, default=9)
    parser.add_argument("--target-ms", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Smoke-only explicit output path; ignored for full runs.",
    )
    args = parser.parse_args(argv)
    result = _bench(args)

    if args.smoke:
        output = _write_smoke_output(result, args.output)
    else:
        _require(
            args.output is None,
            "full runs always write the tracked authoritative result; --output is smoke-only",
        )
        output = _write_full_output(result)

    print(output)
    for route, size in result["sustained_break_even"].items():
        print(f"{route}: sustained measured break-even size={size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
