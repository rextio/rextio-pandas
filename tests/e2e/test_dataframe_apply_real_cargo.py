from __future__ import annotations

import importlib.util
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
from rextio_pandas.types import DataFrameF64, SeriesF64


class One:
    value: float


class Two:
    left: float
    right: float


class Three:
    first: float
    second: float
    third: float


def absolute_row(row) -> float:
    return row["value"] if row["value"] >= 0.0 else -row["value"]


def choose_row(row) -> float:
    return row["left"] if row["left"] >= 0.0 else -row["right"]


def classify_row(row) -> float:
    return row["first"] if row["first"] > 0.0 and row["second"] <= 5.0 else -1.5


class Unicode:
    값: float


def unicode_row(row) -> float:
    return row["값"] if row["값"] >= 0.0 else -row["값"]


def apply_one(frame: DataFrameF64[One]) -> SeriesF64:
    return frame.apply(absolute_row, axis=1)


def apply_two(frame: DataFrameF64[Two]) -> SeriesF64:
    return frame.apply(choose_row, axis=1)


def apply_three(frame: DataFrameF64[Three]) -> SeriesF64:
    return frame.apply(classify_row, axis=1)


def apply_unicode(frame: DataFrameF64[Unicode]) -> SeriesF64:
    return frame.apply(unicode_row, axis=1)
"""


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> CertifiedProject:
    root = tmp_path_factory.mktemp("pandas_dataframe_apply")
    (root / "rextio.toml").write_text(
        '[rust]\nbuild_tool = "cargo"\n\n[plugins]\nenabled = ["rextio-pandas"]\n',
        encoding="utf-8",
    )
    package = root / "src" / "pandas_app"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "frames.py").write_text(KERNELS, encoding="utf-8")
    return build_certification_project(root)


def _series_equal(left: object, right: object) -> bool:
    if type(left) is not pd.Series or type(right) is not pd.Series:
        return False
    try:
        assert_series_equal(left, right, check_exact=True)
    except AssertionError:
        return False
    return bool(
        np.array_equal(
            left.to_numpy().view(np.uint64),
            right.to_numpy().view(np.uint64),
        )
    )


def test_report_and_generated_apply_loops_are_real_native_route(
    project: CertifiedProject,
) -> None:
    report = json.loads(
        (project.project_root / ".rextio" / "reports" / "check.json").read_text(encoding="utf-8")
    )
    functions = {
        function["qualname"]: function
        for module in report["modules"]
        for function in module["functions"]
    }
    for name in ("apply_one", "apply_two", "apply_three", "apply_unicode"):
        record = functions[f"pandas_app.frames.{name}"]
        assert record["native_status"] == "accepted"
        assert record["route"] == "native-plugin:rextio-pandas"

    rust = (project.project_root / ".rextio" / "generated" / "rust" / "src" / "lib.rs").read_text(
        encoding="utf-8"
    )
    assert rust.count("fn __rxtpd_apply_values_") == 4
    assert rust.count("py.detach(|| __rxtpd_apply_values_") == 4
    assert 'EXPECTED_COLUMNS: &[&str] = &["value"]' in rust
    assert 'EXPECTED_COLUMNS: &[&str] = &["left", "right"]' in rust
    assert 'EXPECTED_COLUMNS: &[&str] = &["first", "second", "third"]' in rust
    # Unicode schema fields must be emitted as valid Rust ``\u{...}`` escapes.
    assert r'EXPECTED_COLUMNS: &[&str] = &["\u{ac12}"]' in rust
    import re as _re

    assert _re.search(r"\\u(?!\{)", rust) is None
    for body in rust.split("fn __rxtpd_apply_values_")[1:]:
        hot = body.split("fn __rxtpd_apply_frame_", 1)[0]
        assert "for row in input.outer_iter()" in hot
        assert "PyObject" not in hot
        assert "Python::attach" not in hot
        assert "Python::with_gil" not in hot
        assert ".call" not in hot
        assert "+" not in hot
        assert "/" not in hot


@pytest.mark.parametrize(
    ("name", "frame"),
    [
        (
            "apply_one",
            pd.DataFrame(
                {"value": [-0.0, 0.0, math.nan, math.inf, -math.inf, -2.5, 3.0]},
                dtype="float64",
            ),
        ),
        (
            "apply_two",
            pd.DataFrame(
                {
                    "left": [-0.0, 0.0, math.nan, math.inf, -math.inf, -2.5, 3.0],
                    "right": [1.0, -0.0, 2.0, 3.0, 4.0, math.inf, -math.inf],
                },
                dtype="float64",
            ),
        ),
        (
            "apply_three",
            pd.DataFrame(
                {
                    "first": [1.0, -1.0, math.nan],
                    "second": [5.0, 4.0, 3.0],
                    "third": [100.0, 200.0, 300.0],
                },
                dtype="float64",
            ),
        ),
    ],
)
def test_native_equals_exact_original_dataframe_apply(
    project: CertifiedProject,
    name: str,
    frame: pd.DataFrame,
) -> None:
    checker = project.equivalence_checker(
        f"pandas_app.frames.{name}",
        equals=_series_equal,
    )
    result = checker(frame)
    assert type(result) is pd.Series
    assert result.dtype == np.dtype("float64")
    assert result.name is None
    assert result.index is not frame.index
    assert result.index.equals(frame.index)
    assert type(result.index) is pd.RangeIndex
    assert result.index.name is None
    assert result.attrs == {}
    assert result.flags.allows_duplicate_labels is True


def test_fortran_order_frame_is_copied_in_logical_order(project: CertifiedProject) -> None:
    values = np.asfortranarray(
        np.array(
            [
                [1.0, 2.0, 3.0],
                [-1.0, 4.0, 5.0],
                [math.nan, 6.0, 7.0],
            ]
        )
    )
    frame = pd.DataFrame(values, columns=["first", "second", "third"])
    assert frame.to_numpy(copy=False).flags.f_contiguous is True
    checker = project.equivalence_checker(
        "pandas_app.frames.apply_three",
        equals=_series_equal,
        copy_args=lambda args: (
            pd.DataFrame(
                np.asfortranarray(args[0].to_numpy(copy=False)),
                columns=args[0].columns,
            ),
        ),
    )
    result = checker(frame)
    expected = frame.apply(
        lambda row: row["first"] if row["first"] > 0.0 and row["second"] <= 5.0 else -1.5,
        axis=1,
    )
    assert_series_equal(result, expected, check_exact=True)


def test_large_frame_runs_the_same_product_route(project: CertifiedProject) -> None:
    frame = pd.DataFrame(
        {
            "left": np.linspace(-1000.0, 1000.0, 20_001),
            "right": np.linspace(1000.0, -1000.0, 20_001),
        }
    )
    checker = project.equivalence_checker(
        "pandas_app.frames.apply_two",
        equals=_series_equal,
    )
    result = checker(frame)
    assert len(result) == 20_001


def _run_fresh(project: CertifiedProject, mode: str, body: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", body],
        cwd=project.project_root,
        env={
            **os.environ,
            "PYTHONPATH": str(project.build_python_dir),
            "REXTIO_NATIVE_MODE": mode,
        },
        capture_output=True,
        text=True,
        check=False,
    )


def test_native_and_fallback_apply_run_in_fresh_processes(project: CertifiedProject) -> None:
    body = """
import json
import numpy as np
import pandas as pd
from pandas_app.frames import apply_two
frame = pd.DataFrame({
    "left": [-0.0, 0.0, -2.5, 3.0],
    "right": [1.0, -0.0, 4.0, 5.0],
}, dtype="float64")
out = apply_two(frame)
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
        (
            "frame = pd.DataFrame({'left': [], 'right': []}, dtype='float64')",
            RUNTIME_ERRORS["frame_empty"],
        ),
        ("frame = pd.DataFrame()", RUNTIME_ERRORS["frame_empty"]),
        (
            "frame = pd.DataFrame(index=pd.RangeIndex(2))",
            RUNTIME_ERRORS["frame_zero_columns"],
        ),
        (
            "frame = pd.DataFrame({'left': [1.0], 'right': [2]}, dtype=None)",
            RUNTIME_ERRORS["frame_f64"],
        ),
        (
            "frame = pd.DataFrame({'left': pd.Series([1.0], dtype='Float64'), "
            "'right': pd.Series([2.0], dtype='Float64')})",
            RUNTIME_ERRORS["frame_f64"],
        ),
        (
            "frame = pd.DataFrame({'left': pd.Series([1.0, 2.0], dtype='category'), "
            "'right': pd.Series([3.0, 4.0], dtype='category')})",
            RUNTIME_ERRORS["frame_f64"],
        ),
        (
            "frame = pd.DataFrame({'left': pd.Series([1.0], dtype='Sparse[float64]'), "
            "'right': pd.Series([2.0], dtype='Sparse[float64]')})",
            RUNTIME_ERRORS["frame_f64"],
        ),
        (
            "frame = pd.DataFrame({'left': [1.0], 'right': [2.0]}, "
            "index=pd.Index([4]), dtype='float64')",
            RUNTIME_ERRORS["frame_index"],
        ),
        (
            "frame = pd.DataFrame({'left': [1.0], 'right': [2.0]}, dtype='float64'); "
            "frame.attrs['x'] = 1",
            RUNTIME_ERRORS["frame_attrs"],
        ),
        (
            "frame = pd.DataFrame({'left': [1.0], 'right': [2.0]}, dtype='float64')"
            ".set_flags(allows_duplicate_labels=False)",
            RUNTIME_ERRORS["frame_flags"],
        ),
        (
            "frame = pd.DataFrame([[1.0, 2.0]], columns=['right', 'left'])",
            RUNTIME_ERRORS["frame_schema"],
        ),
        (
            "frame = pd.DataFrame([[1.0]], columns=['left'])",
            RUNTIME_ERRORS["frame_schema"],
        ),
        (
            "frame = pd.DataFrame([[1.0, 2.0, 3.0]], columns=['left', 'right', 'extra'])",
            RUNTIME_ERRORS["frame_schema"],
        ),
        (
            "frame = pd.DataFrame([[1.0, 2.0]], columns=['left', 1])",
            RUNTIME_ERRORS["frame_schema"],
        ),
        (
            "frame = pd.DataFrame([[1.0, 2.0]], columns=['left', 'left'])",
            RUNTIME_ERRORS["frame_schema"],
        ),
        (
            "frame = pd.DataFrame([[1.0, 2.0]], columns=['left', 'right']); "
            "frame.columns.name = 'named'",
            RUNTIME_ERRORS["frame_schema"],
        ),
        (
            "frame = pd.DataFrame([[1.0, 2.0]], columns=['left', 'right']); "
            "frame.__dict__['apply'] = lambda *args: None",
            RUNTIME_ERRORS["frame_method"],
        ),
        (
            "class Child(pd.DataFrame): pass\n"
            "frame = Child([[1.0, 2.0]], columns=['left', 'right'])",
            RUNTIME_ERRORS["frame_class"],
        ),
        (
            "frame = pd.DataFrame([[1.0, 2.0]], columns=['left', 'right'])\n"
            "pd.DataFrame.apply = lambda *args, **kwargs: None",
            RUNTIME_ERRORS["frame_method"],
        ),
        (
            "frame = pd.DataFrame([[1.0, 2.0]], columns=['left', 'right'])\n"
            "pd.__version__ = '0.0.0'",
            RUNTIME_ERRORS["version"],
        ),
    ],
)
def test_runtime_dataframe_contract_misses_raise_exact_type_error(
    project: CertifiedProject,
    setup: str,
    expected: str,
) -> None:
    body = f"""
import pandas as pd
from pandas_app.frames import apply_two
{setup}
try:
    apply_two(frame)
except Exception as exc:
    print(type(exc).__name__)
    print(str(exc))
else:
    raise SystemExit("expected contract error")
"""
    completed = _run_fresh(project, "native", body)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["TypeError", expected]


def test_functools_wraps_forgery_of_frame_apply_is_rejected(project: CertifiedProject) -> None:
    body = """
import functools
import pandas as pd
from pandas_app.frames import apply_two

_original = pd.DataFrame.__dict__["apply"]

@functools.wraps(_original)
def _forged(self, *args, **kwargs):
    return pd.Series([777.0] * len(self), index=self.index)

pd.DataFrame.apply = _forged
assert _forged.__module__ == "pandas.core.frame"
assert _forged.__qualname__ == "DataFrame.apply"

frame = pd.DataFrame({"left": [1.0, 2.0], "right": [3.0, 4.0]}, dtype="float64")
try:
    apply_two(frame)
except Exception as exc:
    print(type(exc).__name__)
    print(str(exc))
else:
    raise SystemExit("forged DataFrame.apply was accepted")
"""
    completed = _run_fresh(project, "native", body)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["TypeError", RUNTIME_ERRORS["frame_method"]]


@pytest.mark.parametrize(
    "setup",
    [
        # Same apply code object, but a changed default (raw=True).
        (
            "import types\n"
            "_orig = pd.DataFrame.__dict__['apply']\n"
            "_defaults = tuple(True if d is False else d for d in _orig.__defaults__)\n"
            "_patched = types.FunctionType(_orig.__code__, _orig.__globals__, "
            "_orig.__name__, _defaults, _orig.__closure__)\n"
            "_patched.__qualname__ = _orig.__qualname__\n"
            "_patched.__module__ = _orig.__module__\n"
            "pd.DataFrame.apply = _patched"
        ),
        # apply with the original code object but foreign globals dict.
        (
            "import types\n"
            "_orig = pd.DataFrame.__dict__['apply']\n"
            "_patched = types.FunctionType(_orig.__code__, dict(_orig.__globals__), "
            "_orig.__name__, _orig.__defaults__, _orig.__closure__)\n"
            "_patched.__qualname__ = _orig.__qualname__\n"
            "_patched.__module__ = _orig.__module__\n"
            "pd.DataFrame.apply = _patched"
        ),
        # DataFrame.to_numpy ordinary replacement.
        "pd.DataFrame.to_numpy = lambda self, *a, **k: self.values",
        # Instance to_numpy shadowing.
        "frame.__dict__['to_numpy'] = lambda *a, **k: None",
    ],
)
def test_frame_semantic_identity_tampering_is_rejected(
    project: CertifiedProject, setup: str
) -> None:
    body = f"""
import pandas as pd
from pandas_app.frames import apply_two
frame = pd.DataFrame({{"left": [1.0, 2.0], "right": [3.0, 4.0]}}, dtype="float64")
{setup}
try:
    apply_two(frame)
except Exception as exc:
    print(type(exc).__name__)
    print(str(exc))
else:
    raise SystemExit("semantic tampering was accepted")
"""
    completed = _run_fresh(project, "native", body)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["TypeError", RUNTIME_ERRORS["frame_method"]]


@pytest.mark.parametrize(
    "body_lines",
    [
        # DataFrame.to_numpy uses the pinned pandas.core.frame.np binding.
        (
            "import pandas.core.frame as frame_mod\n"
            "_saved = frame_mod.np\n"
            "frame_mod.np = 'not numpy'\n"
            "try:\n"
            "    _run()\n"
            "finally:\n"
            "    frame_mod.np = _saved"
        ),
        # DataFrame.apply imports pandas.core.apply.frame_apply at call time;
        # tampering that import target must be rejected.
        (
            "import pandas.core.apply as apply_mod\n"
            "_saved = apply_mod.frame_apply\n"
            "apply_mod.frame_apply = lambda *a, **k: None\n"
            "try:\n"
            "    _run()\n"
            "finally:\n"
            "    apply_mod.frame_apply = _saved"
        ),
    ],
)
def test_frame_mutated_globals_and_imports_are_rejected(
    project: CertifiedProject, body_lines: str
) -> None:
    body = f"""
import pandas as pd
from pandas_app.frames import apply_two
frame = pd.DataFrame({{"left": [1.0, 2.0], "right": [3.0, 4.0]}}, dtype="float64")
def _run():
    try:
        apply_two(frame)
    except Exception as exc:
        print(type(exc).__name__)
        print(str(exc))
    else:
        raise SystemExit("mutated global/import was accepted")
{body_lines}
"""
    completed = _run_fresh(project, "native", body)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["TypeError", RUNTIME_ERRORS["frame_method"]]


def test_arrow_backed_frame_rejected_when_pyarrow_available(project: CertifiedProject) -> None:
    if importlib.util.find_spec("pyarrow") is None:
        pytest.skip("pyarrow is not installed; Arrow-backed storage cannot be constructed")
    body = """
import pandas as pd
from pandas_app.frames import apply_two
frame = pd.DataFrame({
    "left": pd.Series([1.0, 2.0], dtype="float64[pyarrow]"),
    "right": pd.Series([3.0, 4.0], dtype="float64[pyarrow]"),
})
try:
    apply_two(frame)
except Exception as exc:
    print(type(exc).__name__)
    print(str(exc))
else:
    raise SystemExit("expected contract error")
"""
    completed = _run_fresh(project, "native", body)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["TypeError", RUNTIME_ERRORS["frame_f64"]]


def test_unicode_schema_field_native_equals_fallback(project: CertifiedProject) -> None:
    frame = pd.DataFrame(
        {"값": [-0.0, 0.0, math.nan, math.inf, -math.inf, -2.5, 3.0]},
        dtype="float64",
    )
    checker = project.equivalence_checker(
        "pandas_app.frames.apply_unicode",
        equals=_series_equal,
    )
    result = checker(frame)
    expected = frame.apply(
        lambda row: row["값"] if row["값"] >= 0.0 else -row["값"],
        axis=1,
    )
    assert type(result) is pd.Series
    assert_series_equal(result, expected, check_exact=True)
    assert np.array_equal(
        result.to_numpy().view(np.uint64),
        expected.to_numpy().view(np.uint64),
    )
