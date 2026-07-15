from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from rextio.plugins.testing import CertifiedProject, build_certification_project

from rextio_pandas.diagnostics import RUNTIME_ERRORS

pytestmark = pytest.mark.skipif(
    shutil.which("cargo") is None,
    reason="real-Cargo proof requires cargo",
)

KERNELS = """
from rextio_pandas.types import SeriesF64, SeriesI64


def branch_f64(value: float) -> float:
    return value * 2.0 if value > 0.0 else -value


def identity_f64(value: float) -> float:
    return value


def identity_i64(value: int) -> int:
    return value


def classify_i64(value: int) -> float:
    return 1.5 if value >= 0 else 2.5


def map_f64(series: SeriesF64) -> SeriesF64:
    return series.map(branch_f64)


def map_f64_identity(series: SeriesF64) -> SeriesF64:
    return series.map(identity_f64)


def map_i64(series: SeriesI64) -> SeriesI64:
    return series.map(identity_i64)


def map_i64_to_f64(series: SeriesI64) -> SeriesF64:
    return series.map(classify_i64)
"""


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> CertifiedProject:
    root = tmp_path_factory.mktemp("pandas_series_map")
    (root / "rextio.toml").write_text(
        '[rust]\nbuild_tool = "cargo"\n\n[plugins]\nenabled = ["rextio-pandas"]\n',
        encoding="utf-8",
    )
    package = root / "src" / "pandas_app"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "kernels.py").write_text(KERNELS, encoding="utf-8")
    return build_certification_project(root)


def _series_equal(left: object, right: object) -> bool:
    if type(left) is not pd.Series or type(right) is not pd.Series:
        return False
    try:
        assert_series_equal(left, right, check_exact=True)
    except AssertionError:
        return False
    if left.dtype == np.dtype("float64"):
        return bool(
            np.array_equal(
                left.to_numpy().view(np.uint64),
                right.to_numpy().view(np.uint64),
            )
        )
    return True


def test_report_and_generated_hot_loops_are_real_native_route(project: CertifiedProject) -> None:
    check = json.loads(
        (project.project_root / ".rextio" / "reports" / "check.json").read_text(encoding="utf-8")
    )
    functions = {
        function["qualname"]: function
        for module in check["modules"]
        for function in module["functions"]
    }
    for name in ("map_f64", "map_f64_identity", "map_i64", "map_i64_to_f64"):
        record = functions[f"pandas_app.kernels.{name}"]
        assert record["native_status"] == "accepted"
        assert record["route"] == "native-plugin:rextio-pandas"

    rust = (project.project_root / ".rextio" / "generated" / "rust" / "src" / "lib.rs").read_text(
        encoding="utf-8"
    )
    assert rust.count("fn __rxtpd_map_values_") == 4
    assert rust.count("py.detach(|| __rxtpd_map_values_") == 4
    for body in rust.split("fn __rxtpd_map_values_")[1:]:
        hot = body.split("fn __rxtpd_map_series_", 1)[0]
        assert "for &value in input.iter()" in hot
        assert "PyObject" not in hot
        assert "Python::attach" not in hot
        assert "Python::with_gil" not in hot
        assert ".call" not in hot


@pytest.mark.parametrize(
    ("name", "source"),
    [
        (
            "map_f64",
            pd.Series(
                [-0.0, 0.0, math.nan, math.inf, -math.inf, -2.5, 3.0],
                dtype="float64",
                name="values",
            ),
        ),
        ("map_f64_identity", pd.Series([1.0], dtype="float64", name=None)),
        (
            "map_i64",
            pd.Series(
                [np.iinfo(np.int64).min, -1, 0, 1, np.iinfo(np.int64).max],
                dtype="int64",
                name="ints",
            ),
        ),
        ("map_i64_to_f64", pd.Series([-2, 0, 3], dtype="int64", name="classes")),
    ],
)
def test_native_equals_exact_original_pandas_call(
    project: CertifiedProject,
    name: str,
    source: pd.Series,
) -> None:
    checker = project.equivalence_checker(
        f"pandas_app.kernels.{name}",
        equals=_series_equal,
    )
    result = checker(source)
    assert type(result) is pd.Series
    assert result.name == source.name
    assert type(result.index) is pd.RangeIndex
    assert result.index.equals(source.index)
    assert result.index.name is None
    assert result.attrs == {}
    assert result.flags.allows_duplicate_labels is True


def test_strided_series_is_copied_by_logical_indexing(project: CertifiedProject) -> None:
    backing = np.arange(20, dtype=np.float64)
    source = pd.Series(backing[::2], name="strided")
    assert source.to_numpy(copy=False).flags.c_contiguous is False
    checker = project.equivalence_checker(
        "pandas_app.kernels.map_f64_identity",
        equals=_series_equal,
        copy_args=lambda args: (pd.Series(args[0].to_numpy(copy=False), name=args[0].name),),
    )
    result = checker(source)
    assert_series_equal(result, source, check_exact=True)


def test_large_series_runs_the_same_product_route(project: CertifiedProject) -> None:
    source = pd.Series(np.linspace(-1000.0, 1000.0, 20_001), name="large")
    checker = project.equivalence_checker(
        "pandas_app.kernels.map_f64",
        equals=_series_equal,
    )
    result = checker(source)
    assert type(result) is pd.Series
    assert len(result) == 20_001
    assert result.name == "large"


def _run_fresh(project: CertifiedProject, mode: str, body: str) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "PYTHONPATH": str(project.build_python_dir),
        "REXTIO_NATIVE_MODE": mode,
    }
    return subprocess.run(
        [sys.executable, "-c", body],
        cwd=project.project_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_native_and_fallback_run_in_fresh_processes(project: CertifiedProject) -> None:
    body = """
import json
import numpy as np
import pandas as pd
from pandas_app.kernels import map_f64
s = pd.Series([-0.0, 0.0, -2.5, 3.0], dtype="float64", name="values")
out = map_f64(s)
print(json.dumps({
    "class": type(out).__name__,
    "dtype": str(out.dtype),
    "bits": out.to_numpy().view(np.uint64).tolist(),
    "index": [out.index.start, out.index.stop, out.index.step, out.index.name],
    "name": out.name,
    "attrs": out.attrs,
    "duplicates": out.flags.allows_duplicate_labels,
}, sort_keys=True))
"""
    native = _run_fresh(project, "native", body)
    fallback = _run_fresh(project, "fallback", body)
    assert native.returncode == 0, native.stderr
    assert fallback.returncode == 0, fallback.stderr
    assert json.loads(native.stdout) == json.loads(fallback.stdout)


@pytest.mark.parametrize(
    ("setup", "expected"),
    [
        ("s = pd.Series([], dtype='float64')", RUNTIME_ERRORS["series_empty"]),
        ("s = pd.Series([1.0], dtype='float32')", RUNTIME_ERRORS["series_f64"]),
        (
            "s = pd.Series([1.0], index=pd.Index([4]), dtype='float64')",
            RUNTIME_ERRORS["series_index"],
        ),
        (
            "s = pd.Series([1.0], dtype='float64'); s.attrs['x'] = 1",
            RUNTIME_ERRORS["series_attrs"],
        ),
        (
            "s = pd.Series([1.0], dtype='float64').set_flags(allows_duplicate_labels=False)",
            RUNTIME_ERRORS["series_flags"],
        ),
        (
            "s = pd.Series([1.0], dtype='float64', name=('not', 'str'))",
            RUNTIME_ERRORS["series_name"],
        ),
        (
            "s = pd.Series([1.0], dtype='float64'); s.__dict__['map'] = lambda f: s",
            RUNTIME_ERRORS["series_method"],
        ),
        (
            "class Child(pd.Series): pass\ns = Child([1.0], dtype='float64')",
            RUNTIME_ERRORS["series_class"],
        ),
        (
            "s = pd.Series([1.0], dtype='float64')\npd.Series.map = lambda self, f: self",
            RUNTIME_ERRORS["series_method"],
        ),
    ],
)
def test_runtime_contract_misses_raise_exact_type_error(
    project: CertifiedProject,
    setup: str,
    expected: str,
) -> None:
    body = f"""
import pandas as pd
from pandas_app.kernels import map_f64
{setup}
try:
    map_f64(s)
except Exception as exc:
    print(type(exc).__name__)
    print(str(exc))
else:
    raise SystemExit("expected contract error")
"""
    completed = _run_fresh(project, "native", body)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["TypeError", expected]
