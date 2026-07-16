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
from benchmarks.cases import KERNEL_SOURCE, make_case


def _tree_clean() -> bool:
    """Plugin checkout must be clean; optional core root only when overridden."""
    paths = [ROOT]
    core_root = bench._optional_core_root()
    if core_root is not None:
        paths.append(core_root)
    for path in paths:
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


def test_authoritative_fixture_is_series_only() -> None:
    assert "SeriesF64" in KERNEL_SOURCE
    assert "DataFrame" not in KERNEL_SOURCE
    assert ".apply(" not in KERNEL_SOURCE
    assert make_case("series.map", 10).route == "series.map"
    with pytest.raises(ValueError, match="unknown benchmark route"):
        make_case("dataframe.apply", 10)


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


@pytest.mark.parametrize(
    "schedule",
    [
        # duplicate-lane pairs that happen to look 5:4 by first position
        [["native", "native"]] * 5 + [["fallback", "fallback"]] * 4,
        # unknown lane
        [["native", "sideways"]] * 5 + [["fallback", "native"]] * 4,
        # missing lane (too short)
        [["native"]] * 5 + [["fallback", "native"]] * 4,
        # extra lane (too long)
        [["native", "fallback", "native"]] * 5 + [["fallback", "native"]] * 4,
        # duplicate-lane single pair among otherwise valid ones
        [["native", "native"]] + [["fallback", "native"]] * 4 + [["native", "fallback"]] * 4,
    ],
)
def test_schedule_balance_rejects_malformed_pairs(schedule: list) -> None:
    balance = _schedule_balance(schedule)
    assert balance["all_pairs_valid"] is False
    assert balance["is_counterbalanced"] is False


def test_schedule_balance_accepts_only_valid_pairs() -> None:
    schedule = [["native", "fallback"]] * 5 + [["fallback", "native"]] * 4
    balance = _schedule_balance(schedule)
    assert balance["all_pairs_valid"] is True
    assert balance["is_counterbalanced"] is True


# --- item 3: native artifact raw SHA + exact-path binding -------------------


def test_sha256_file_is_the_raw_bytes_digest(tmp_path: Path) -> None:
    import hashlib

    path = tmp_path / "artifact.bin"
    path.write_bytes(b"\x00\x01native-artifact\xff")
    assert bench._sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()
    # The manifest digest folds in the filename/framing, so the two differ.
    assert bench._sha256_file(path) != bench._sha256_files([path])


def test_sha256_file_changes_only_with_bytes(tmp_path: Path) -> None:
    a = tmp_path / "a.so"
    a.write_bytes(b"same-bytes")
    first = bench._sha256_file(a)
    # A different file (different name) with identical bytes hashes the same.
    b = tmp_path / "b.so"
    b.write_bytes(b"same-bytes")
    assert bench._sha256_file(b) == first
    # Changing the bytes changes the digest.
    b.write_bytes(b"other-bytes")
    assert bench._sha256_file(b) != first


def test_native_artifact_binding_rejects_a_different_file(tmp_path: Path) -> None:
    installed = tmp_path / "generated" / "_rextio_native.so"
    installed.parent.mkdir(parents=True)
    installed.write_bytes(b"the-real-artifact-bytes")
    # The byte-identical copy staged in the build tree binds successfully.
    imported = tmp_path / "build" / "_rextio_native.so"
    imported.parent.mkdir(parents=True)
    imported.write_bytes(b"the-real-artifact-bytes")
    assert bench._require_bound_native_artifact(str(installed), str(imported)) == imported.resolve()
    # A different native file beneath the same project root (different bytes) is
    # rejected even though it sits under the tree.
    stale = tmp_path / "build" / "_rextio_native_stale.so"
    stale.write_bytes(b"a-different-stale-artifact")
    with pytest.raises(RuntimeError):
        bench._require_bound_native_artifact(str(installed), str(stale))


# --- G: fail-closed eligibility recomputed from the raw schedule ------------


def test_valid_series_cell_is_headline_eligible() -> None:
    eligible, reasons = _headline_eligibility(
        _valid_series_cell(), repetitions=9, null_floor_median_ns=1_000.0
    )
    assert eligible is True
    assert reasons == []


def test_unknown_route_is_never_headline_eligible() -> None:
    cell = _valid_series_cell()
    cell["route"] = "prototype.route"
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


def test_core_provenance_accepts_index_install_without_direct_url() -> None:
    assert bench._validate_core_provenance(None, Path("/tmp/rextio/__init__.py"), None) == "index"


def test_core_provenance_accepts_vcs_with_commit() -> None:
    mode = bench._validate_core_provenance(
        {
            "url": "https://github.com/rextio/rextio.git",
            "vcs_info": {"vcs": "git", "commit_id": "abc123"},
        },
        Path("/tmp/rextio/__init__.py"),
        None,
    )
    assert mode == "vcs"


def test_core_provenance_rejects_credentialed_url() -> None:
    with pytest.raises(RuntimeError):
        bench._validate_core_provenance(
            {
                "url": "https://ghp_secret@github.com/rextio/rextio.git",
                "vcs_info": {"vcs": "git", "commit_id": "abc123"},
            },
            Path("/tmp/rextio/__init__.py"),
            None,
        )


def test_core_version_range_gate() -> None:
    assert bench._rextio_version_supported("0.1.3")
    assert bench._rextio_version_supported("0.1.9")
    assert not bench._rextio_version_supported("0.1.2")
    assert not bench._rextio_version_supported("0.2.0")


def test_plugin_direct_url_validator_accepts_wheel_mode() -> None:
    mode = bench._validate_plugin_direct_url(
        {
            "url": "https://example/rextio_pandas-0.1.0-py3-none-any.whl",
            "archive_info": {"hashes": {"sha256": "abc"}},
        },
        Path("/site-packages/rextio_pandas/__init__.py"),
    )
    assert mode == "wheel"


@pytest.mark.skipif(
    shutil.which("cargo") is None or shutil.which("rustc") is None,
    reason="preflight records the Rust toolchain versions",
)
@pytest.mark.needs_cargo
def test_preflight_gathers_provenance_on_a_clean_tree() -> None:
    if not _tree_clean():
        pytest.skip("preflight is fail-closed on a dirty worktree; run on a clean tree")
    provenance = _preflight()
    for key in (
        "core_version",
        "plugin_sha",
        "core_install_mode",
        "plugin_install_mode",
        "plugin_direct_url",
        "plugin_api_version",
        "required_rextio_spec",
        "selected_entry_points",
        "harness_manifest_sha256",
        "pandas",
        "numpy",
        "rustc",
        "cargo",
    ):
        assert key in provenance
    assert provenance["plugin_api_version"] == "1.3"
    assert provenance["required_rextio_spec"] == ">=0.1.3,<0.2"
    assert bench._rextio_version_supported(provenance["core_version"])
    assert provenance["core_dirty"] is False and provenance["plugin_dirty"] is False
    assert provenance["pandas"] == "2.3.3"
    assert provenance["numpy"] == "2.3.5"
    assert "/Volumes/Data/workspace" not in Path(bench.__file__).read_text(encoding="utf-8")
