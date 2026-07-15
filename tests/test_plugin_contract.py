from __future__ import annotations

import importlib.metadata as importlib_metadata
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlsplit

import rextio

from rextio.plugins.api import PLUGIN_API_VERSION, BoundaryConversion

import rextio_pandas
from rextio_pandas.plugin import CORE_COMMIT, RextioPandasPlugin
from rextio_pandas.plugin_types import PLUGIN_TYPES
from rextio_pandas.rust_snippets.map_apply import boundary_helpers

from conftest import pandas_registry

ROOT = Path(__file__).resolve().parents[1]
CORE_VCS_URL = "https://github.com/rextio/rextio-core-next.git"


def _normalize(name: str) -> str:
    return name.lower().replace("_", "-")


def _direct_url(distribution: str) -> dict | None:
    # The test path can expose several metadata sources for one project (an
    # installed editable/wheel distribution plus a shadow ``src`` egg-info). Only
    # a real install records direct_url.json, so scan every candidate and return
    # the first that carries provenance rather than trusting distribution() to
    # pick the installed one.
    target = _normalize(distribution)
    for dist in importlib_metadata.distributions():
        name = dist.metadata["Name"]
        if name is None or _normalize(name) != target:
            continue
        raw = dist.read_text("direct_url.json")
        if raw:
            return json.loads(raw)
    return None


def _file_url_to_path(url: str) -> Path:
    parts = urlsplit(url)
    assert parts.scheme == "file", url
    return Path(unquote(parts.path)).resolve()


def _assert_credential_free(url: str) -> None:
    parts = urlsplit(url)
    assert "@" not in (parts.netloc or ""), url
    assert "ghp_" not in url and "x-access-token" not in url, url


def test_exact_incubator_dependencies_and_api() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = metadata["project"]["dependencies"]

    assert PLUGIN_API_VERSION == "1.3"
    assert RextioPandasPlugin.api_version == "1.3"
    assert f"git+https://github.com/rextio/rextio-core-next.git@{CORE_COMMIT}" in dependencies[0]
    assert "rextio.git@" not in dependencies[0]
    assert "@ghp_" not in dependencies[0]
    assert "x-access-token" not in dependencies[0]
    assert "@" not in dependencies[0].split("://", 1)[1].split("/", 1)[0]
    assert dependencies[1:] == ["pandas==2.3.3", "numpy==2.3.5"]


def test_installed_core_provenance_is_api_13_not_released_range() -> None:
    # The resolved core must be the integrated API 1.3 commit, never the
    # released 0.1.2 wheel that shares the same version string. Both editable
    # and non-editable VCS-wheel modes are proven without relying on a directory
    # name substring.
    assert PLUGIN_API_VERSION == "1.3"
    core_file = Path(rextio.__file__).resolve()
    direct_url = _direct_url("rextio")
    assert direct_url is not None, "core has no direct_url.json provenance"

    if direct_url.get("dir_info", {}).get("editable"):
        # Editable: prove the import resolves under the exact resolved checkout
        # and that checkout's Git HEAD is the pinned integrated commit.
        checkout = _file_url_to_path(direct_url["url"])
        assert core_file.is_relative_to(checkout), (checkout, core_file)
        head = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert head == CORE_COMMIT, head
    elif "vcs_info" in direct_url:
        # Non-editable VCS wheel: prove the credential-free URL and exact commit.
        assert direct_url["url"] == CORE_VCS_URL, direct_url["url"]
        _assert_credential_free(direct_url["url"])
        assert direct_url["vcs_info"]["vcs"] == "git"
        assert direct_url["vcs_info"]["commit_id"] == CORE_COMMIT
    else:
        raise AssertionError(f"unrecognized core install provenance: {direct_url}")


def test_installed_plugin_provenance_matches_this_checkout_or_its_wheel() -> None:
    plugin_file = Path(rextio_pandas.__file__).resolve()
    direct_url = _direct_url("rextio-pandas")
    assert direct_url is not None, "plugin has no direct_url.json provenance"

    if direct_url.get("dir_info", {}).get("editable"):
        checkout = _file_url_to_path(direct_url["url"])
        assert checkout == ROOT, (checkout, ROOT)
        # Resolved path-boundary containment, not a string prefix (a sibling
        # ``src-evil`` directory must not satisfy an ``src`` prefix).
        assert plugin_file.is_relative_to(checkout / "src"), plugin_file
    elif "archive_info" in direct_url:
        # Wheel install: the import lives in site-packages and the recorded
        # archive is a rextio_pandas wheel with a pinned hash.
        wheel_url = direct_url["url"]
        assert wheel_url.endswith(".whl"), wheel_url
        assert "rextio_pandas-" in wheel_url.rsplit("/", 1)[-1], wheel_url
        assert direct_url["archive_info"]["hashes"], direct_url
    else:
        raise AssertionError(f"unrecognized plugin install provenance: {direct_url}")


def test_editable_src_boundary_rejects_sibling_prefix() -> None:
    # A string ``startswith`` check would wrongly accept a sibling ``src-evil``
    # directory; resolved ``is_relative_to`` must not.
    checkout = ROOT
    inside = (checkout / "src" / "rextio_pandas" / "__init__.py").resolve()
    sibling = (checkout / "src-evil" / "rextio_pandas" / "__init__.py").resolve()
    assert inside.is_relative_to(checkout / "src")
    assert not sibling.is_relative_to(checkout / "src")
    assert str(sibling).startswith(str(checkout / "src"))  # the trap the old check fell into


def test_selected_entry_point_loads_this_exact_plugin_object() -> None:
    from rextio_pandas.plugin import plugin as this_plugin

    selected = [
        ep
        for ep in importlib_metadata.entry_points(group="rextio.plugins")
        if ep.name == "rextio-pandas"
    ]
    assert selected, "rextio-pandas entry point is not installed"
    assert all(ep.value == "rextio_pandas.plugin:plugin" for ep in selected)
    assert all(ep.dist is not None and ep.dist.name == "rextio-pandas" for ep in selected)
    # The entry point loads to the exact plugin factory in this checkout.
    assert all(ep.load() is this_plugin for ep in selected)


def test_clean_env_proof_script_is_credential_free_and_no_deps_free() -> None:
    text = (ROOT / "scripts" / "clean_env_proof.py").read_text(encoding="utf-8")
    assert CORE_COMMIT in text
    assert "rextio-core-next" in text
    # The proof resolves dependencies; it must not bypass them.
    assert '"--no-deps"' not in text
    assert "'--no-deps'" not in text


def test_loader_registers_only_supported_series_types_and_exact_crate() -> None:
    registry = pandas_registry()

    assert registry.active[0].api_version == "1.3"
    assert registry.active[0].lowering_provided is True
    assert registry.active[0].packages == ("pandas",)
    assert tuple(binding.plugin_type for binding in registry.types) == PLUGIN_TYPES
    assert len(PLUGIN_TYPES) == 2
    assert [plugin_type.key for plugin_type in PLUGIN_TYPES] == [
        "rextio-pandas/series-f64",
        "rextio-pandas/series-i64",
    ]
    assert all(
        isinstance(plugin_type.conversion, BoundaryConversion) for plugin_type in PLUGIN_TYPES
    )
    assert all(plugin_type.is_resident is False for plugin_type in PLUGIN_TYPES)
    assert all(plugin_type.uses == () for plugin_type in PLUGIN_TYPES)
    assert all(plugin_type.helpers == (boundary_helpers(),) for plugin_type in PLUGIN_TYPES)
    assert [
        (entry.dependency.name, entry.dependency.version) for entry in registry.crate_dependencies
    ] == [("numpy", "=0.29.0")]


def test_public_authority_exposes_series_map_and_apply_no_go_only() -> None:
    provider = RextioPandasPlugin()
    assert provider.covers().symbols == ("pandas.Series.map",)

    records = provider.describe(object())  # type: ignore[arg-type]
    native = [record for record in records if record.outcome == "native"]
    assert [record.id for record in native] == ["rextio-pandas/series-map"]
    [apply_no_go] = [record for record in records if "dataframe-apply" in record.id]
    assert apply_no_go.id == "rextio-pandas/dataframe-apply-prototype-no-go"
    assert apply_no_go.outcome == "fallback"
    assert apply_no_go.verified is False
    assert "not registered, claimed, lowered, built, benchmarked" in apply_no_go.constraint


def test_annotation_vocabulary_imports_without_pandas_or_core() -> None:
    script = """
import builtins
import sys

real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "pandas" or name == "rextio" or name.startswith("pandas.") or name.startswith("rextio."):
        raise AssertionError(f"unexpected import: {name}")
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded

from rextio_pandas.types import DataFrameF64, SeriesF64, SeriesI64

class Row:
    x: float

assert DataFrameF64[Row] is DataFrameF64
assert SeriesF64.__module__ == "rextio_pandas.types"
assert SeriesI64.__module__ == "rextio_pandas.types"
assert not any(name == "pandas" or name.startswith("pandas.") for name in sys.modules)
print("ok")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"


def test_future_and_eager_annotation_spellings() -> None:
    eager: dict[str, object] = {}
    eager_code = compile(
        "from rextio_pandas.types import SeriesF64\n"
        "def f(value: SeriesF64) -> SeriesF64:\n"
        "    return value\n",
        "<eager-annotations>",
        "exec",
        dont_inherit=True,
    )
    exec(eager_code, eager)
    assert eager["f"].__annotations__["value"].__name__ == "SeriesF64"  # type: ignore[index,union-attr]

    future: dict[str, object] = {}
    exec(
        "from __future__ import annotations\n"
        "from rextio_pandas.types import SeriesI64\n"
        "def f(value: SeriesI64) -> SeriesI64:\n"
        "    return value\n",
        future,
    )
    assert future["f"].__annotations__ == {"value": "SeriesI64", "return": "SeriesI64"}  # type: ignore[union-attr]
