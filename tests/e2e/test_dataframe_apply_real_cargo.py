"""Real-Cargo regression proving DataFrame.apply remains an ordinary fallback.

The project also contains one supported Series route so the certification build
produces and imports a real native extension. DataFrame.apply itself must never
appear as a native claim, generated hot loop, or build-report product route.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest

from rextio.plugins.testing import CertifiedProject, build_certification_project

pytestmark = [
    pytest.mark.needs_cargo,
    pytest.mark.skipif(
        shutil.which("cargo") is None,
        reason="real-Cargo proof requires cargo",
    ),
]

KERNELS = """
from __future__ import annotations

from rextio_pandas.types import DataFrameF64, SeriesF64


class Pair:
    left: float
    right: float


def identity(value: float) -> float:
    return value


def row_sum(row) -> float:
    return row["left"] + row["right"]


def series_identity(series: SeriesF64) -> SeriesF64:
    return series.map(identity)


def dataframe_apply(frame: DataFrameF64[Pair]) -> SeriesF64:
    return frame.apply(row_sum, axis=1)
"""


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> CertifiedProject:
    root = tmp_path_factory.mktemp("pandas_apply_no_go")
    (root / "rextio.toml").write_text(
        '[rust]\nbuild_tool = "cargo"\n\n[plugins]\nenabled = ["rextio-pandas"]\n',
        encoding="utf-8",
    )
    package = root / "src" / "pandas_app"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "kernels.py").write_text(KERNELS, encoding="utf-8")
    return build_certification_project(root)


def _run_fresh(project: CertifiedProject, body: str) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "PYTHONPATH": str(project.build_python_dir),
        "REXTIO_NATIVE_MODE": "auto",
    }
    return subprocess.run(
        [sys.executable, "-c", body],
        cwd=project.project_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_reports_and_generated_rust_expose_no_native_apply_route(
    project: CertifiedProject,
) -> None:
    reports = project.project_root / ".rextio" / "reports"
    check = json.loads((reports / "check.json").read_text(encoding="utf-8"))
    functions = {
        function["qualname"]: function
        for module in check["modules"]
        for function in module["functions"]
    }
    series = functions["pandas_app.kernels.series_identity"]
    apply = functions["pandas_app.kernels.dataframe_apply"]
    assert series["route"] == "native-plugin:rextio-pandas"
    assert apply["route"] == "fallback-python"
    assert apply["plugin_claims"] == []

    build = json.loads((reports / "build.json").read_text(encoding="utf-8"))
    assert build["status"] == "built"
    assert build["accepted_native_count"] >= 1

    rust = (project.project_root / ".rextio" / "generated" / "rust" / "src" / "lib.rs").read_text(
        encoding="utf-8"
    )
    assert "fn __rxtpd_map_values_" in rust
    assert "fn __rxtpd_apply_values_" not in rust
    assert "fn __rxtpd_apply_frame_" not in rust


def test_auto_mode_delegates_apply_to_exact_pandas_fallback(project: CertifiedProject) -> None:
    body = """
import json
import pandas as pd
import sys
from pandas_app.kernels import dataframe_apply
frame = pd.DataFrame({"left": [1.0, 2.0], "right": [10.0, 20.0]})
out = dataframe_apply(frame)
print(json.dumps({
    "values": out.tolist(),
    "dtype": str(out.dtype),
    "name": out.name,
    "native_loaded": "_rextio_native" in sys.modules,
}))
"""
    completed = _run_fresh(project, body)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "values": [11.0, 22.0],
        "dtype": "float64",
        "name": None,
        "native_loaded": True,
    }


def test_auto_mode_observes_live_dataframe_apply_behavior(project: CertifiedProject) -> None:
    body = """
import pandas as pd
from pandas_app.kernels import dataframe_apply
frame = pd.DataFrame({"left": [1.0, 2.0], "right": [10.0, 20.0]})
original = pd.DataFrame.apply
pd.DataFrame.apply = lambda self, *args, **kwargs: pd.Series([-999.0] * len(self))
try:
    print(dataframe_apply(frame).tolist())
finally:
    pd.DataFrame.apply = original
"""
    completed = _run_fresh(project, body)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "[-999.0, -999.0]"
