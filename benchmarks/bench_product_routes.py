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
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from rextio.plugins.testing import build_certification_project

from benchmarks.cases import KERNEL_SOURCE, BenchmarkCase, make_case

ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = Path("/Volumes/Data/workspace/rextio/rextio-core-next")
CORE_SHA = "ac2b79d304f13abaaecaf7714f897574c3b6256f"


def _git_sha(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _command_text(command: Sequence[str]) -> str:
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    return completed.stdout.strip() or completed.stderr.strip()


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
    schedule: list[list[str]] = []
    samples = {"native": [], "fallback": []}
    for _ in range(repetitions):
        order = ["native", "fallback"]
        if rng.random() < 0.5:
            order.reverse()
        schedule.append(order.copy())
        for lane in order:
            _set_mode(lane)
            if lane == "native":
                samples[lane].append(_sample(native_call, native_loops))
            else:
                samples[lane].append(_sample(fallback_call, fallback_loops))
    return {
        "loops": {"native": native_loops, "fallback": fallback_loops},
        "schedule": schedule,
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
        return {"available": False, "reason": "numba is not installed"}

    if case.route == "series.map":
        values = case.argument.to_numpy(copy=False)  # type: ignore[union-attr]

        @numba.njit
        def kernel(data: np.ndarray) -> np.ndarray:
            output = np.empty(data.shape[0], dtype=np.float64)
            for index in range(data.shape[0]):
                value = data[index]
                output[index] = value * 2.0 if value > 0.0 else -value
            return output

        def call() -> pd.Series:
            return pd.Series(kernel(values), index=case.argument.index, name=case.argument.name)  # type: ignore[union-attr]

    else:
        frame = case.argument
        values = frame.to_numpy(copy=False)  # type: ignore[union-attr]

        @numba.njit
        def kernel(data: np.ndarray) -> np.ndarray:
            output = np.empty(data.shape[0], dtype=np.float64)
            for index in range(data.shape[0]):
                left = data[index, 0]
                output[index] = left if left >= 0.0 else -data[index, 1]
            return output

        def call() -> pd.Series:
            return pd.Series(kernel(values), index=frame.index, name=None)  # type: ignore[union-attr]

    cold_start = time.perf_counter_ns()
    cold_result = call()
    cold_ns = time.perf_counter_ns() - cold_start
    loops = _calibrate(call, target_ns)
    samples = [_sample(call, loops) for _ in range(repetitions)]
    return {
        "available": True,
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


def _bench(args: argparse.Namespace) -> dict[str, Any]:
    if _git_sha(CORE_ROOT) != CORE_SHA:
        raise RuntimeError("core-next is not at the required integrated commit")
    sizes = [10, 1000] if args.smoke else [1, 10, 100, 1000, 10_000, 100_000]
    repetitions = 3 if args.smoke else args.repetitions
    target_ns = int(args.target_ms * 1_000_000)
    started = time.perf_counter_ns()
    with tempfile.TemporaryDirectory(prefix="rextio-pandas-bench-") as directory:
        project_root = Path(directory)
        _write_project(project_root)
        compile_started = time.perf_counter_ns()
        project = build_certification_project(project_root)
        compile_ns = time.perf_counter_ns() - compile_started
        report = json.loads(
            (project_root / ".rextio" / "reports" / "check.json").read_text(encoding="utf-8")
        )
        routes = {
            function["qualname"]: function["route"]
            for module in report["modules"]
            for function in module["functions"]
        }
        assert routes["bench_app.kernels.series_map"] == "native-plugin:rextio-pandas"
        assert routes["bench_app.kernels.dataframe_apply"] == "native-plugin:rextio-pandas"

        sys.path.insert(0, str(project.build_python_dir))
        try:
            module = importlib.import_module("bench_app.kernels")
            cells: list[dict[str, Any]] = []
            for route_index, route in enumerate(("series.map", "dataframe.apply")):
                for size_index, size in enumerate(sizes):
                    case = make_case(route, size)
                    function = getattr(module, case.function_name)
                    product = _product_samples(
                        function,
                        case.argument,
                        repetitions=repetitions,
                        target_ns=target_ns,
                        seed=args.seed + route_index * 100 + size_index,
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
                            "headline_eligible": route == "series.map",
                            "headline_note": (
                                "verified exact Series route"
                                if route == "series.map"
                                else "NO headline speedup claim: mixed-row coercion and "
                                "NumPy-scalar semantics remain out of scope"
                            ),
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
    break_even: dict[str, int | None] = {}
    for route in ("series.map", "dataframe.apply"):
        eligible = [
            cell
            for cell in cells
            if cell["route"] == route and cell["product"]["paired_bootstrap"]["ci95_high"] < 1.0
        ]
        break_even[route] = min((cell["size"] for cell in eligible), default=None)

    return {
        "schema": 1,
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
        },
        "provenance": {
            "core_sha": _git_sha(CORE_ROOT),
            "plugin_sha": _git_sha(ROOT),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "python": sys.version,
            "rustc": _command_text(["rustc", "--version"]),
            "cargo": _command_text(["cargo", "--version"]),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "compile_ns": compile_ns,
        },
        "null_call_floor": {
            "loops": null_loops,
            "samples_ns_per_call": null_samples,
            "median_ns_per_call": statistics.median(null_samples),
        },
        "cells": cells,
        "break_even_first_ci95_below_one": break_even,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--repetitions", type=int, default=9)
    parser.add_argument("--target-ms", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks" / "results" / "latest.json",
    )
    args = parser.parse_args(argv)
    result = _bench(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for old in args.output.parent.glob("*.json"):
        old.unlink()
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    for route, size in result["break_even_first_ci95_below_one"].items():
        print(f"{route}: first measured CI95-native-win size={size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
