"""Deterministic unit tests for the benchmark schedule, eligibility, provenance,
break-even, and smoke-isolation gates."""

from __future__ import annotations

import json
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from benchmarks import bench_product_routes as bench
from benchmarks.bench_product_routes import (
    ROOT,
    _counterbalanced_schedule,
    _headline_eligibility,
    _paired_bootstrap,
    _percentile,
    _preflight,
    _require,
    _schedule_balance,
    _sustained_break_even,
)


def _tree_clean() -> bool:
    for path in (ROOT, bench.CORE_ROOT):
        status = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        if status.strip():
            return False
    return True


def _valid_series_cell(repetitions: int = 9, *, size: int = 1000) -> dict[str, Any]:
    schedule = _counterbalanced_schedule(repetitions, random.Random(0))
    samples = [12_000_000.0 + index for index in range(repetitions)]
    return {
        "route": "series.map",
        "size": size,
        "route_verified": True,
        "headline_eligible": False,
        "product": {
            "schedule": schedule,
            "samples_ns_per_call": {"native": list(samples), "fallback": list(samples)},
            "median_ns_per_call": {"native": 12_000_000.0, "fallback": 12_500_000.0},
            "correctness_digest": {"native": "abc", "fallback": "abc"},
            "paired_bootstrap": {
                "median_native_over_fallback": 0.90,
                "ci95_low": 0.86,
                "ci95_high": 0.94,
            },
        },
    }


# --- E: counterbalance without odd-repeat lane bias -------------------------


def test_counterbalanced_schedule_is_balanced_for_nine() -> None:
    schedule = _counterbalanced_schedule(9, random.Random(20260715))
    assert len(schedule) == 9
    balance = _schedule_balance(schedule)
    assert balance["is_counterbalanced"] is True
    assert {balance["native_first"], balance["fallback_first"]} == {4, 5}
    assert all(sorted(order) == ["fallback", "native"] for order in schedule)


def test_counterbalanced_extra_lane_is_not_always_native() -> None:
    # Across seeds the extra first position must land on both lanes, not
    # systematically on native.
    outcomes = set()
    for seed in range(64):
        balance = _schedule_balance(_counterbalanced_schedule(9, random.Random(seed)))
        outcomes.add((balance["native_first"], balance["fallback_first"]))
    assert (5, 4) in outcomes and (4, 5) in outcomes


def test_counterbalanced_schedule_is_seed_deterministic() -> None:
    assert _counterbalanced_schedule(9, random.Random(7)) == _counterbalanced_schedule(
        9, random.Random(7)
    )


def test_even_repeats_are_exactly_balanced() -> None:
    balance = _schedule_balance(_counterbalanced_schedule(8, random.Random(3)))
    assert (balance["native_first"], balance["fallback_first"]) == (4, 4)


def test_schedule_balance_flags_independent_draw_imbalance() -> None:
    balance = _schedule_balance([["native", "fallback"]] * 9)
    assert (balance["native_first"], balance["fallback_first"]) == (9, 0)
    assert balance["is_counterbalanced"] is False


# --- G: fail-closed eligibility recomputed from the raw schedule ------------


def test_valid_series_cell_is_headline_eligible() -> None:
    eligible, reasons = _headline_eligibility(
        _valid_series_cell(), repetitions=9, null_floor_median_ns=1_000.0
    )
    assert eligible is True
    assert reasons == []


def test_dataframe_route_is_never_headline_eligible() -> None:
    cell = _valid_series_cell()
    cell["route"] = "dataframe.apply"
    eligible, reasons = _headline_eligibility(cell, repetitions=9, null_floor_median_ns=1_000.0)
    assert eligible is False
    assert "route-not-headline-surface" in reasons


@pytest.mark.parametrize(
    ("mutate", "expected_reason"),
    [
        (lambda c: c.update(route_verified=False), "route-or-provenance-unverified"),
        (
            lambda c: c["product"]["correctness_digest"].update(fallback="different"),
            "correctness-mismatch",
        ),
        (
            lambda c: c["product"]["samples_ns_per_call"]["native"].pop(),
            "missing-required-sample",
        ),
        (
            lambda c: c["product"]["samples_ns_per_call"]["native"].__setitem__(0, -1.0),
            "nonpositive-or-nonfinite-sample",
        ),
        (
            lambda c: c["product"]["samples_ns_per_call"]["native"].__setitem__(0, float("nan")),
            "nonpositive-or-nonfinite-sample",
        ),
        # Recomputed from the raw schedule: an imbalanced raw schedule is caught
        # even if a (stale) precomputed summary claimed balance.
        (
            lambda c: c["product"].update(schedule=[["native", "fallback"]] * 9),
            "schedule-not-counterbalanced",
        ),
        (
            lambda c: c["product"]["median_ns_per_call"].update(native=2_000.0),
            "near-null-floor",
        ),
        (
            lambda c: c["product"]["median_ns_per_call"].update(native=float("inf")),
            "nonfinite-median",
        ),
        (
            lambda c: c["product"]["paired_bootstrap"].update(ci95_low=0.10, ci95_high=1.80),
            "unstable-wide-ci",
        ),
        (
            lambda c: c["product"]["paired_bootstrap"].update(ci95_low=0.95, ci95_high=0.90),
            "reversed-ci-bounds",
        ),
        (
            lambda c: c["product"]["paired_bootstrap"].update(
                median_native_over_fallback=float("nan")
            ),
            "missing-or-nonfinite-bootstrap",
        ),
    ],
)
def test_headline_eligibility_fails_closed(mutate: Any, expected_reason: str) -> None:
    cell = _valid_series_cell()
    mutate(cell)
    eligible, reasons = _headline_eligibility(cell, repetitions=9, null_floor_median_ns=1_000.0)
    assert eligible is False
    assert expected_reason in reasons


def test_stale_summary_cannot_launder_an_imbalanced_schedule() -> None:
    cell = _valid_series_cell()
    cell["product"]["schedule"] = [["native", "fallback"]] * 9  # imbalanced raw
    cell["product"]["schedule_balance"] = {"is_counterbalanced": True}  # lying summary
    eligible, reasons = _headline_eligibility(cell, repetitions=9, null_floor_median_ns=1_000.0)
    assert eligible is False
    assert "schedule-not-counterbalanced" in reasons


# --- G: sustained break-even ------------------------------------------------


def _series_cell(size: int, *, eligible: bool, ci95_high: float) -> dict[str, Any]:
    return {
        "route": "series.map",
        "size": size,
        "headline_eligible": eligible,
        "product": {"paired_bootstrap": {"ci95_high": ci95_high}},
    }


def test_sustained_break_even_requires_all_larger_sizes_favorable() -> None:
    cells = [
        _series_cell(10, eligible=True, ci95_high=1.20),  # slower
        _series_cell(100, eligible=True, ci95_high=0.80),  # faster
        _series_cell(1000, eligible=True, ci95_high=0.70),  # faster
    ]
    assert _sustained_break_even(cells) == 100


def test_sustained_break_even_is_none_when_the_largest_cell_regresses() -> None:
    cells = [
        _series_cell(10, eligible=True, ci95_high=0.80),
        _series_cell(100, eligible=True, ci95_high=0.70),
        _series_cell(1000, eligible=True, ci95_high=1.10),  # largest regresses
    ]
    assert _sustained_break_even(cells) is None


def test_sustained_break_even_is_none_when_a_required_cell_is_ineligible() -> None:
    cells = [
        _series_cell(10, eligible=True, ci95_high=0.80),
        _series_cell(100, eligible=False, ci95_high=0.70),  # not eligible
    ]
    assert _sustained_break_even(cells) is None


# --- G: bootstrap / percentile determinism ----------------------------------


def test_percentile_linear_interpolation_is_exact() -> None:
    values = [10.0, 20.0, 30.0, 40.0]
    assert _percentile(values, 0.0) == 10.0
    assert _percentile(values, 1.0) == 40.0
    assert _percentile(values, 0.5) == 25.0


def test_paired_bootstrap_is_seed_deterministic() -> None:
    native = [9.0, 8.0, 10.0, 7.0, 9.5]
    fallback = [10.0, 10.0, 10.0, 10.0, 10.0]
    first = _paired_bootstrap(native, fallback, iterations=500, seed=42)
    second = _paired_bootstrap(native, fallback, iterations=500, seed=42)
    assert first == second
    assert first["ci95_low"] <= first["median_native_over_fallback"] <= first["ci95_high"]


# --- F: smoke isolation and scoped cleanup ----------------------------------


def test_require_raises_without_using_assert() -> None:
    _require(True, "ok")
    with pytest.raises(RuntimeError, match="benchmark preflight failed: boom"):
        _require(False, "boom")


def test_smoke_output_refuses_to_touch_authoritative_result() -> None:
    result = {"provenance": {"report_evidence_files": {"check.json": "{}"}}}
    with pytest.raises(RuntimeError):
        bench._write_smoke_output(result, bench.FULL_RESULT)


def test_smoke_writes_to_temp_and_leaves_sentinel_untouched(tmp_path: Path) -> None:
    sentinel = "SENTINEL-FULL-RESULT"
    existed = bench.FULL_RESULT.exists()
    original = bench.FULL_RESULT.read_text() if existed else None
    bench.FULL_RESULT.parent.mkdir(parents=True, exist_ok=True)
    bench.FULL_RESULT.write_text(sentinel, encoding="utf-8")
    try:
        result = {"provenance": {"report_evidence_files": {"check.json": "{}"}}, "ok": True}
        out = bench._write_smoke_output(result, tmp_path / "smoke.json")
        assert out == (tmp_path / "smoke.json").resolve()
        assert json.loads(out.read_text())["ok"] is True
        # The authoritative result and its raw evidence are untouched by smoke.
        assert bench.FULL_RESULT.read_text() == sentinel
        assert "report_evidence_files" not in result["provenance"]
    finally:
        if original is not None:
            bench.FULL_RESULT.write_text(original, encoding="utf-8")
        else:
            bench.FULL_RESULT.unlink()


def test_smoke_default_output_is_a_temp_location() -> None:
    result = {"provenance": {}, "ok": True}
    out = bench._write_smoke_output(result, None)
    try:
        assert out.exists()
        assert not out.is_relative_to(bench.RESULTS_DIR.resolve())
    finally:
        shutil.rmtree(out.parent, ignore_errors=True)


# --- C: fail-closed provenance validators -----------------------------------


def test_core_direct_url_validator_rejects_wrong_commit() -> None:
    with pytest.raises(RuntimeError):
        bench._validate_core_direct_url(
            {"url": bench.CORE_VCS_URL, "vcs_info": {"vcs": "git", "commit_id": "deadbeef"}},
            Path("/tmp/rextio/__init__.py"),
        )


def test_core_direct_url_validator_rejects_credentialed_url() -> None:
    with pytest.raises(RuntimeError):
        bench._validate_core_direct_url(
            {
                "url": "https://ghp_secret@github.com/rextio/rextio-core-next.git",
                "vcs_info": {"vcs": "git", "commit_id": bench.CORE_SHA},
            },
            Path("/tmp/rextio/__init__.py"),
        )


def test_plugin_direct_url_validator_accepts_wheel_mode() -> None:
    mode = bench._validate_plugin_direct_url(
        {
            "url": "https://example/rextio_pandas-0.0.1-py3-none-any.whl",
            "archive_info": {"hashes": {"sha256": "abc"}},
        },
        Path("/site-packages/rextio_pandas/__init__.py"),
    )
    assert mode == "wheel"


@pytest.mark.skipif(
    shutil.which("cargo") is None or shutil.which("rustc") is None,
    reason="preflight records the Rust toolchain versions",
)
def test_preflight_gathers_provenance_on_a_clean_tree() -> None:
    if not _tree_clean():
        pytest.skip("preflight is fail-closed on a dirty worktree; run on a clean tree")
    provenance = _preflight()
    for key in (
        "core_sha",
        "plugin_sha",
        "core_install_mode",
        "plugin_install_mode",
        "core_direct_url",
        "plugin_direct_url",
        "plugin_api_version",
        "selected_entry_points",
        "harness_digest",
        "pandas",
        "numpy",
        "rustc",
        "cargo",
    ):
        assert key in provenance
    assert provenance["plugin_api_version"] == "1.3"
    assert provenance["core_dirty"] is False and provenance["plugin_dirty"] is False
    assert provenance["pandas"] == "2.3.3"
    assert provenance["numpy"] == "2.3.5"
