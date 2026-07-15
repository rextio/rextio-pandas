from __future__ import annotations

import importlib.metadata as importlib_metadata
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import rextio

from rextio.plugins.api import PLUGIN_API_VERSION, BoundaryConversion

from rextio_pandas.plugin import CORE_COMMIT, RextioPandasPlugin
from rextio_pandas.plugin_types import PLUGIN_TYPES

from conftest import pandas_registry

ROOT = Path(__file__).resolve().parents[1]


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
    # released 0.1.2 wheel that shares the same version string.
    assert PLUGIN_API_VERSION == "1.3"
    core_file = Path(rextio.__file__).resolve()
    assert "rextio-core-next" in core_file.parts, core_file

    selected = [
        ep
        for ep in importlib_metadata.entry_points(group="rextio.plugins")
        if ep.name == "rextio-pandas"
    ]
    assert selected, "rextio-pandas entry point is not installed"
    assert all(ep.value == "rextio_pandas.plugin:plugin" for ep in selected)
    assert all(ep.dist is not None and ep.dist.name == "rextio-pandas" for ep in selected)


def test_clean_env_proof_script_is_credential_free_and_no_deps_free() -> None:
    text = (ROOT / "scripts" / "clean_env_proof.py").read_text(encoding="utf-8")
    assert CORE_COMMIT in text
    assert "rextio-core-next" in text
    # The proof resolves dependencies; it must not bypass them.
    assert '"--no-deps"' not in text
    assert "'--no-deps'" not in text


def test_loader_registers_materialized_series_types_and_exact_crate() -> None:
    registry = pandas_registry()

    assert registry.active[0].api_version == "1.3"
    assert registry.active[0].lowering_provided is True
    assert registry.active[0].packages == ("pandas",)
    assert tuple(binding.plugin_type for binding in registry.types) == PLUGIN_TYPES
    assert len(PLUGIN_TYPES) == 3
    assert all(
        isinstance(plugin_type.conversion, BoundaryConversion) for plugin_type in PLUGIN_TYPES
    )
    assert all(plugin_type.is_resident is False for plugin_type in PLUGIN_TYPES)
    assert [
        (entry.dependency.name, entry.dependency.version) for entry in registry.crate_dependencies
    ] == [("numpy", "=0.29.0")]


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
