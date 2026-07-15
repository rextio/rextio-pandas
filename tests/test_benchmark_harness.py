"""Deterministic unit tests for the benchmark schedule and eligibility gates."""

from __future__ import annotations

import random
import shutil
from typing import Any

import pytest

from benchmarks.bench_product_routes import (
    _counterbalanced_schedule,
    _headline_eligibility,
    _preflight,
    _require,
    _schedule_balance,
)


def _valid_series_cell(repetitions: int = 9) -> dict[str, Any]:
    schedule = _counterbalanced_schedule(repetitions, random.Random(0))
    samples = [12_000_000.0 + index for index in range(repetitions)]
    return {
        "route": "series.map",
        "size": 1000,
        "route_verified": True,
        "product": {
            "samples_ns_per_call": {"native": list(samples), "fallback": list(samples)},
            "median_ns_per_call": {"native": 12_000_000.0, "fallback": 12_500_000.0},
            "schedule_balance": _schedule_balance(schedule),
            "correctness_digest": {"native": "abc", "fallback": "abc"},
            "paired_bootstrap": {
                "median_native_over_fallback": 0.90,
                "ci95_low": 0.86,
                "ci95_high": 0.94,
            },
        },
    }


def test_counterbalanced_schedule_forces_five_four_split_for_nine() -> None:
    schedule = _counterbalanced_schedule(9, random.Random(20260715))
    assert len(schedule) == 9
    balance = _schedule_balance(schedule)
    assert (balance["native_first"], balance["fallback_first"]) == (5, 4)
    assert balance["is_counterbalanced"] is True
    assert all(sorted(order) == ["fallback", "native"] for order in schedule)


def test_counterbalanced_schedule_is_seed_deterministic() -> None:
    first = _counterbalanced_schedule(9, random.Random(7))
    second = _counterbalanced_schedule(9, random.Random(7))
    third = _counterbalanced_schedule(9, random.Random(8))
    assert first == second
    # A different seed shuffles to a different order (balance still holds).
    assert first != third
    assert _schedule_balance(third)["is_counterbalanced"] is True


def test_schedule_balance_flags_independent_draw_imbalance() -> None:
    imbalanced = [["native", "fallback"]] * 9
    balance = _schedule_balance(imbalanced)
    assert (balance["native_first"], balance["fallback_first"]) == (9, 0)
    assert balance["is_counterbalanced"] is False


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
            "nonpositive-sample",
        ),
        (
            lambda c: c["product"].__setitem__("schedule_balance", {"is_counterbalanced": False}),
            "schedule-not-counterbalanced",
        ),
        (
            lambda c: c["product"]["median_ns_per_call"].update(native=2_000.0),
            "near-null-floor",
        ),
        (
            lambda c: c["product"]["paired_bootstrap"].update(ci95_low=0.10, ci95_high=1.80),
            "unstable-wide-ci",
        ),
    ],
)
def test_headline_eligibility_fails_closed(mutate: Any, expected_reason: str) -> None:
    cell = _valid_series_cell()
    mutate(cell)
    eligible, reasons = _headline_eligibility(cell, repetitions=9, null_floor_median_ns=1_000.0)
    assert eligible is False
    assert expected_reason in reasons


def test_require_raises_without_using_assert() -> None:
    _require(True, "ok")  # does not raise
    with pytest.raises(RuntimeError, match="benchmark preflight failed: boom"):
        _require(False, "boom")


@pytest.mark.skipif(
    shutil.which("cargo") is None or shutil.which("rustc") is None,
    reason="preflight records the Rust toolchain versions",
)
def test_preflight_records_full_fail_closed_provenance() -> None:
    provenance = _preflight()
    required = {
        "core_sha",
        "core_dirty",
        "plugin_sha",
        "plugin_dirty",
        "core_import_file",
        "plugin_import_file",
        "core_direct_url",
        "plugin_direct_url",
        "plugin_api_version",
        "selected_entry_points",
        "harness_digest",
        "pandas",
        "numpy",
        "python",
        "rustc",
        "cargo",
        "platform",
        "machine",
        "processor",
    }
    assert required <= set(provenance)
    assert provenance["plugin_api_version"] == "1.3"
    assert "rextio-core-next" in provenance["core_import_file"]
    assert provenance["pandas"] == "2.3.3"
    assert provenance["numpy"] == "2.3.5"
    assert any(
        entry["value"] == "rextio_pandas.plugin:plugin"
        for entry in provenance["selected_entry_points"]
    )
