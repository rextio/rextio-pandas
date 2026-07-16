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

pytestmark = [
    pytest.mark.needs_cargo,
    pytest.mark.skipif(
        shutil.which("cargo") is None,
        reason="real-Cargo proof requires cargo",
    ),
]

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


def parameter_signature_probe(series: SeriesF64) -> float:
    return 1.0


def identity_map_roundtrip(series: SeriesF64) -> SeriesF64:
    return series.map(identity_f64)
"""

CLAIMLESS_KERNELS = """
from rextio_pandas.types import SeriesF64


def inspect_series(series: SeriesF64) -> float:
    return 1.0
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


@pytest.fixture(scope="module")
def claimless_project(tmp_path_factory: pytest.TempPathFactory) -> CertifiedProject:
    """Build a project whose only plugin involvement is a parameter type."""
    root = tmp_path_factory.mktemp("pandas_series_claimless_signature")
    (root / "rextio.toml").write_text(
        '[rust]\nbuild_tool = "cargo"\n\n[plugins]\nenabled = ["rextio-pandas"]\n',
        encoding="utf-8",
    )
    package = root / "src" / "pandas_claimless"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "kernels.py").write_text(CLAIMLESS_KERNELS, encoding="utf-8")
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
    for name in (
        "map_f64",
        "map_f64_identity",
        "map_i64",
        "map_i64_to_f64",
        "parameter_signature_probe",
        "identity_map_roundtrip",
    ):
        record = functions[f"pandas_app.kernels.{name}"]
        assert record["native_status"] == "accepted"
        assert record["route"] == "native-plugin:rextio-pandas"
    assert functions["pandas_app.kernels.parameter_signature_probe"]["plugin_claims"] == []
    assert functions["pandas_app.kernels.identity_map_roundtrip"]["plugin_claims"]

    rust = (project.project_root / ".rextio" / "generated" / "rust" / "src" / "lib.rs").read_text(
        encoding="utf-8"
    )
    assert rust.count("fn __rxtpd_map_values_") == 4
    assert rust.count("py.detach(|| __rxtpd_map_values_") == 4
    assert rust.count("struct RxtPandasSeriesF64") == 1
    for body in rust.split("fn __rxtpd_map_values_")[1:]:
        hot = body.split("fn __rxtpd_map_series_", 1)[0]
        assert "for &value in input.iter()" in hot
        assert "PyObject" not in hot
        assert "Python::attach" not in hot
        assert "Python::with_gil" not in hot
        assert ".call" not in hot


def test_parameter_probe_and_identity_map_product_use_real_boundary_support(
    project: CertifiedProject,
) -> None:
    source = pd.Series([-0.0, 1.5], dtype="float64", name="signature")
    probe = project.equivalence_checker("pandas_app.kernels.parameter_signature_probe")
    assert probe(source) == 1.0

    roundtrip = project.equivalence_checker(
        "pandas_app.kernels.identity_map_roundtrip",
        equals=_series_equal,
    )
    result = roundtrip(source)
    assert_series_equal(result, source, check_exact=True)
    assert np.array_equal(result.to_numpy().view(np.uint64), source.to_numpy().view(np.uint64))


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


def test_claimless_only_parameter_signature_builds_and_executes_boundary_support(
    claimless_project: CertifiedProject,
) -> None:
    reports = claimless_project.project_root / ".rextio" / "reports"
    check = json.loads((reports / "check.json").read_text(encoding="utf-8"))
    functions = {
        function["qualname"]: function
        for module in check["modules"]
        for function in module["functions"]
    }
    assert set(functions) == {"pandas_claimless.kernels.inspect_series"}
    record = functions["pandas_claimless.kernels.inspect_series"]
    assert record["native_status"] == "accepted"
    assert record["route"] == "native-plugin:rextio-pandas"
    assert record["plugin_claims"] == []

    build = json.loads((reports / "build.json").read_text(encoding="utf-8"))
    assert build["status"] == "built"
    assert build["accepted_native_count"] == 1

    rust = (
        claimless_project.project_root / ".rextio" / "generated" / "rust" / "src" / "lib.rs"
    ).read_text(encoding="utf-8")
    assert rust.count("struct RxtPandasSeriesF64") == 1
    assert rust.count("fn __rxtpd_extract_series_f64") == 1
    assert "let series = __rxtpd_extract_series_f64(py, &series)?;" in rust
    assert "fn __rxtpd_map_values_" not in rust

    valid = _run_fresh(
        claimless_project,
        "native",
        """
import pandas as pd
from pandas_claimless.kernels import inspect_series
print(inspect_series(pd.Series([1.0], dtype="float64")))
""",
    )
    assert valid.returncode == 0, valid.stderr
    assert valid.stdout.strip() == "1.0"

    rejected = _run_fresh(
        claimless_project,
        "native",
        """
import pandas as pd
from pandas_claimless.kernels import inspect_series
try:
    inspect_series(pd.Series([], dtype="float64"))
except Exception as exc:
    print(type(exc).__name__)
    print(str(exc))
else:
    raise SystemExit("expected contract error")
""",
    )
    assert rejected.returncode == 0, rejected.stderr
    assert rejected.stdout.splitlines() == ["TypeError", RUNTIME_ERRORS["series_empty"]]


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
        # Extension dtypes whose ``to_numpy()`` yields a numeric ndarray must be
        # rejected before conversion (float64 route).
        (
            "s = pd.Series([1.0, 2.0], dtype='Float64')",
            RUNTIME_ERRORS["series_f64"],
        ),
        (
            "s = pd.Series([1.0, 2.0], dtype='category')",
            RUNTIME_ERRORS["series_f64"],
        ),
        (
            "s = pd.Series([1.0, 2.0], dtype='Sparse[float64]')",
            RUNTIME_ERRORS["series_f64"],
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


@pytest.mark.parametrize(
    "setup",
    [
        "s = pd.Series([1, 2], dtype='Int64')",
        "s = pd.Series([1, 2], dtype='category')",
        "s = pd.Series([1, 2], dtype='Sparse[int64]')",
    ],
)
def test_i64_route_rejects_extension_storage_before_conversion(
    project: CertifiedProject,
    setup: str,
) -> None:
    body = f"""
import pandas as pd
from pandas_app.kernels import map_i64
{setup}
try:
    map_i64(s)
except Exception as exc:
    print(type(exc).__name__)
    print(str(exc))
else:
    raise SystemExit("expected contract error")
"""
    completed = _run_fresh(project, "native", body)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["TypeError", RUNTIME_ERRORS["series_i64"]]


def test_arrow_backed_series_rejected_when_pyarrow_available(project: CertifiedProject) -> None:
    if importlib.util.find_spec("pyarrow") is None:
        pytest.skip("pyarrow is not installed; Arrow-backed storage cannot be constructed")
    body = """
import pandas as pd
from pandas_app.kernels import map_f64
s = pd.Series([1.0, 2.0], dtype="float64[pyarrow]")
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
    assert completed.stdout.splitlines() == ["TypeError", RUNTIME_ERRORS["series_f64"]]


def test_functools_wraps_forgery_of_series_map_is_rejected(project: CertifiedProject) -> None:
    # A ``functools.wraps`` replacement copies ``__module__``/``__qualname__``
    # (which the old guard trusted) but never ``__code__``. It must now fail the
    # bytecode-fingerprint identity check and refuse to route natively.
    body = """
import functools
import pandas as pd
from pandas_app.kernels import map_f64

_original = pd.Series.__dict__["map"]

@functools.wraps(_original)
def _forged(self, *args, **kwargs):
    return pd.Series([777] * len(self), index=self.index, name=self.name)

pd.Series.map = _forged
assert _forged.__module__ == "pandas.core.series"
assert _forged.__qualname__ == "Series.map"

s = pd.Series([1.0, 2.0], dtype="float64", name="values")
try:
    map_f64(s)
except Exception as exc:
    print(type(exc).__name__)
    print(str(exc))
else:
    raise SystemExit("forged Series.map was accepted")
"""
    completed = _run_fresh(project, "native", body)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["TypeError", RUNTIME_ERRORS["series_method"]]


def test_inherited_map_values_authority_tampering_is_rejected(
    project: CertifiedProject,
) -> None:
    body = """
import numpy as np
import pandas as pd
import pandas.core.base as base
from pandas_app.kernels import map_f64

_map_descriptor = pd.Series.__dict__["map"]
_map_values_owner = next(cls for cls in pd.Series.__mro__ if "_map_values" in cls.__dict__)
assert _map_values_owner is base.IndexOpsMixin

def _forged_map_values(self, mapper, na_action=None):
    return np.full(len(self), -999.0, dtype=np.float64)

_map_values_owner._map_values = _forged_map_values
assert pd.Series.__dict__["map"] is _map_descriptor
assert pd.Series.map is _map_descriptor

s = pd.Series([1.0, 2.0], dtype="float64", name="values")
fallback = _map_descriptor(s, lambda value: value * 2.0)
assert fallback.tolist() == [-999.0, -999.0]

try:
    native = map_f64(s)
except Exception as exc:
    print(type(exc).__name__)
    print(str(exc))
else:
    raise SystemExit(
        f"mutated Series._map_values was accepted: "
        f"fallback={fallback.tolist()} native={native.tolist()}"
    )
"""
    completed = _run_fresh(project, "native", body)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["TypeError", RUNTIME_ERRORS["series_method"]]


def test_instance_map_values_shadow_is_rejected(project: CertifiedProject) -> None:
    body = """
import numpy as np
import pandas as pd
from pandas_app.kernels import map_f64

_map_descriptor = pd.Series.__dict__["map"]
s = pd.Series([1.0, 2.0], dtype="float64", name="values")
s.__dict__["_map_values"] = lambda mapper, na_action=None: np.full(
    len(s), -998.0, dtype=np.float64
)
fallback = _map_descriptor(s, lambda value: value * 2.0)
assert fallback.tolist() == [-998.0, -998.0]

try:
    native = map_f64(s)
except Exception as exc:
    print(type(exc).__name__)
    print(str(exc))
else:
    raise SystemExit(
        f"instance _map_values shadow was accepted: "
        f"fallback={fallback.tolist()} native={native.tolist()}"
    )
"""
    completed = _run_fresh(project, "native", body)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["TypeError", RUNTIME_ERRORS["series_method"]]


def test_direct_series_map_values_override_is_rejected(project: CertifiedProject) -> None:
    body = """
import numpy as np
import pandas as pd
import pandas.core.base as base
from pandas_app.kernels import map_f64

_map_descriptor = pd.Series.__dict__["map"]
_base_descriptor = base.IndexOpsMixin.__dict__["_map_values"]

def _forged_map_values(self, mapper, na_action=None):
    return np.full(len(self), -997.0, dtype=np.float64)

pd.Series._map_values = _forged_map_values
assert base.IndexOpsMixin.__dict__["_map_values"] is _base_descriptor
assert pd.Series.__dict__["map"] is _map_descriptor

s = pd.Series([1.0, 2.0], dtype="float64", name="values")
fallback = _map_descriptor(s, lambda value: value * 2.0)
assert fallback.tolist() == [-997.0, -997.0]

try:
    native = map_f64(s)
except Exception as exc:
    print(type(exc).__name__)
    print(str(exc))
else:
    raise SystemExit(
        f"direct Series._map_values override was accepted: "
        f"fallback={fallback.tolist()} native={native.tolist()}"
    )
"""
    completed = _run_fresh(project, "native", body)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["TypeError", RUNTIME_ERRORS["series_method"]]


def test_algorithms_map_array_replacement_is_rejected(project: CertifiedProject) -> None:
    body = """
import numpy as np
import pandas as pd
import pandas.core.algorithms as algorithms
from pandas_app.kernels import map_f64

_map_descriptor = pd.Series.__dict__["map"]
_map_values_descriptor = pd.Series._map_values

def _forged_map_array(arr, mapper, na_action=None, convert=True):
    return np.full(len(arr), -996.0, dtype=np.float64)

algorithms.map_array = _forged_map_array
assert pd.Series.__dict__["map"] is _map_descriptor
assert pd.Series._map_values is _map_values_descriptor

s = pd.Series([1.0, 2.0], dtype="float64", name="values")
fallback = _map_descriptor(s, lambda value: value * 2.0)
assert fallback.tolist() == [-996.0, -996.0]

try:
    native = map_f64(s)
except Exception as exc:
    print(type(exc).__name__)
    print(str(exc))
else:
    raise SystemExit(
        f"algorithms.map_array replacement was accepted: "
        f"fallback={fallback.tolist()} native={native.tolist()}"
    )
"""
    completed = _run_fresh(project, "native", body)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["TypeError", RUNTIME_ERRORS["series_method"]]


def test_algorithms_map_array_cached_builtins_replacement_is_rejected(
    project: CertifiedProject,
) -> None:
    body = """
import builtins
import types
import pandas as pd
import pandas.core.algorithms as algorithms
from pandas_app.kernels import map_f64

_map_descriptor = pd.Series.__dict__["map"]
_original_map_array = algorithms.map_array
_module_globals = _original_map_array.__globals__
_canonical_builtins = _module_globals["__builtins__"]
assert _canonical_builtins is builtins.__dict__

_forged_builtins = dict(builtins.__dict__)
_forged_builtins["len"] = lambda value: 0
_module_globals["__builtins__"] = _forged_builtins
try:
    _forged_map_array = types.FunctionType(
        _original_map_array.__code__,
        _module_globals,
        _original_map_array.__name__,
        _original_map_array.__defaults__,
        _original_map_array.__closure__,
    )
finally:
    _module_globals["__builtins__"] = _canonical_builtins

_forged_map_array.__module__ = _original_map_array.__module__
_forged_map_array.__qualname__ = _original_map_array.__qualname__
_forged_map_array.__kwdefaults__ = _original_map_array.__kwdefaults__
algorithms.map_array = _forged_map_array

assert _forged_map_array.__globals__ is _module_globals
assert _forged_map_array.__builtins__ is _forged_builtins
assert _forged_map_array.__builtins__ is not builtins.__dict__
assert _module_globals["__builtins__"] is builtins.__dict__

s = pd.Series([1.0, 2.0], dtype="float64", name="values")
fallback = _map_descriptor(s, lambda value: value * 2.0)
assert fallback.tolist() == [1.0, 2.0]

try:
    native = map_f64(s)
except Exception as exc:
    print(type(exc).__name__)
    print(str(exc))
else:
    raise SystemExit(
        f"cached function builtins replacement was accepted: "
        f"fallback={fallback.tolist()} native={native.tolist()}"
    )
"""
    completed = _run_fresh(project, "native", body)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["TypeError", RUNTIME_ERRORS["series_method"]]


def test_canonical_builtins_len_mutation_after_import_is_rejected(
    project: CertifiedProject,
) -> None:
    """Mutating builtins.len after native import must fail closed.

    Container identity of ``__builtins__ is builtins.__dict__`` still holds, but
    fallback can diverge while native previously still accepted. Independent
    structural authority on ``len`` rejects the pure-Python swap.
    """
    body = """
import builtins
import pandas as pd
from pandas_app.kernels import map_f64

series = pd.Series([1.0, 2.0], dtype="float64", name="values")
descriptor = pd.Series.__dict__["map"]
target_array = series._values
original_len = builtins.len

def selective_len(value):
    if value is target_array:
        return 0
    return original_len(value)

builtins.len = selective_len
try:
    fallback = descriptor(series, lambda value: value * 2.0).tolist()
    try:
        native = map_f64(series)
    except Exception as exc:
        print(type(exc).__name__)
        print(str(exc))
        print(repr(fallback))
    else:
        raise SystemExit(
            f"canonical builtins.len mutation after import was accepted: "
            f"fallback={fallback!r} native={native.tolist()!r}"
        )
finally:
    builtins.len = original_len
"""
    completed = _run_fresh(project, "native", body)
    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.splitlines()
    assert lines[0] == "TypeError"
    assert lines[1] == RUNTIME_ERRORS["series_method"]
    # Fallback under the mutated builtin can differ from honest map semantics.
    assert lines[2] == "[1.0, 2.0]"


def test_canonical_builtins_len_mutation_before_import_is_rejected(
    project: CertifiedProject,
) -> None:
    """Mutating builtins.len before native import must also fail closed.

    Prevents any import-time snapshot of ``len`` from bypassing call-time
    structural authority validation.
    """
    body = """
import builtins
import pandas as pd

series = pd.Series([1.0, 2.0], dtype="float64", name="values")
descriptor = pd.Series.__dict__["map"]
target_array = series._values
original_len = builtins.len

def selective_len(value):
    if value is target_array:
        return 0
    return original_len(value)

builtins.len = selective_len
try:
    fallback = descriptor(series, lambda value: value * 2.0).tolist()
    from pandas_app.kernels import map_f64

    try:
        native = map_f64(series)
    except Exception as exc:
        print(type(exc).__name__)
        print(str(exc))
        print(repr(fallback))
    else:
        raise SystemExit(
            f"canonical builtins.len mutation before import was accepted: "
            f"fallback={fallback!r} native={native.tolist()!r}"
        )
finally:
    builtins.len = original_len
"""
    completed = _run_fresh(project, "native", body)
    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.splitlines()
    assert lines[0] == "TypeError"
    assert lines[1] == RUNTIME_ERRORS["series_method"]
    assert lines[2] == "[1.0, 2.0]"


def test_mutable_function_type_anchor_cannot_accept_forged_callable(
    project: CertifiedProject,
) -> None:
    body = """
import types
import numpy as np
import pandas as pd
import pandas.core.algorithms as algorithms
import pandas.core.base as base
import pandas.core.generic as generic
from pandas_app.kernels import map_f64

_map_descriptor = pd.Series.__dict__["map"]
_map_values_descriptor = base.IndexOpsMixin.__dict__["_map_values"]
_map_array_descriptor = algorithms.map_array
_constructor_property = pd.Series.__dict__["_constructor"]
_finalize_descriptor = generic.NDFrame.__dict__["__finalize__"]
_to_numpy_descriptor = base.IndexOpsMixin.__dict__["to_numpy"]

def _forged_map_array(arr, mapper, na_action=None, convert=True):
    return np.full(len(arr), -981.0, dtype=np.float64)

class _ForgedFunction:
    def __init__(self, original, replacement=None):
        self.__module__ = original.__module__
        self.__qualname__ = original.__qualname__
        self.__globals__ = original.__globals__
        self.__closure__ = original.__closure__
        self.__code__ = original.__code__
        self.__kwdefaults__ = original.__kwdefaults__
        self.__defaults__ = original.__defaults__
        self._original = original
        self._replacement = replacement

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return lambda *args, **kwargs: self(instance, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        target = self._replacement or self._original
        return target(*args, **kwargs)

pd.Series.map = _ForgedFunction(_map_descriptor)
base.IndexOpsMixin._map_values = _ForgedFunction(_map_values_descriptor)
algorithms.map_array = _ForgedFunction(_map_array_descriptor, _forged_map_array)
pd.Series._constructor = property(_ForgedFunction(_constructor_property.fget))
generic.NDFrame.__finalize__ = _ForgedFunction(_finalize_descriptor)
base.IndexOpsMixin.to_numpy = _ForgedFunction(_to_numpy_descriptor)
types.FunctionType = _ForgedFunction

s = pd.Series([1.0, 2.0], dtype="float64", name="values")
fallback = _map_descriptor(s, lambda value: value * 2.0)
assert fallback.tolist() == [-981.0, -981.0]

try:
    native = map_f64(s)
except Exception as exc:
    print(type(exc).__name__)
    print(str(exc))
else:
    raise SystemExit(
        f"mutable FunctionType anchor accepted forged callables: "
        f"fallback={fallback.tolist()} native={native.tolist()}"
    )
"""
    completed = _run_fresh(project, "native", body)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["TypeError", RUNTIME_ERRORS["series_method"]]


def test_map_values_algorithms_global_replacement_is_rejected(
    project: CertifiedProject,
) -> None:
    body = """
import types
import numpy as np
import pandas as pd
import pandas.core.algorithms as algorithms
import pandas.core.base as base
from pandas_app.kernels import map_f64

_map_descriptor = pd.Series.__dict__["map"]
_map_values_descriptor = base.IndexOpsMixin.__dict__["_map_values"]
_canonical_map_array = algorithms.map_array

def _forged_map_array(arr, mapper, na_action=None, convert=True):
    return np.full(len(arr), -995.0, dtype=np.float64)

base.algorithms = types.SimpleNamespace(map_array=_forged_map_array)
assert base.IndexOpsMixin.__dict__["_map_values"] is _map_values_descriptor
assert algorithms.map_array is _canonical_map_array

s = pd.Series([1.0, 2.0], dtype="float64", name="values")
fallback = _map_descriptor(s, lambda value: value * 2.0)
assert fallback.tolist() == [-995.0, -995.0]

try:
    native = map_f64(s)
except Exception as exc:
    print(type(exc).__name__)
    print(str(exc))
else:
    raise SystemExit(
        f"_map_values algorithms global replacement was accepted: "
        f"fallback={fallback.tolist()} native={native.tolist()}"
    )
"""
    completed = _run_fresh(project, "native", body)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["TypeError", RUNTIME_ERRORS["series_method"]]


def test_lib_map_infer_replacement_is_rejected(project: CertifiedProject) -> None:
    body = """
import numpy as np
import pandas as pd
import pandas._libs.lib as lib
from pandas_app.kernels import map_f64

_map_descriptor = pd.Series.__dict__["map"]
_map_array_descriptor = pd.Series._map_values.__globals__["algorithms"].map_array

def _forged_map_infer(arr, mapper, convert=True, ignore_na=False):
    return np.full(len(arr), -994.0, dtype=np.float64)

lib.map_infer = _forged_map_infer
assert pd.Series.__dict__["map"] is _map_descriptor
assert pd.Series._map_values.__globals__["algorithms"].map_array is _map_array_descriptor

s = pd.Series([1.0, 2.0], dtype="float64", name="values")
fallback = _map_descriptor(s, lambda value: value * 2.0)
assert fallback.tolist() == [-994.0, -994.0]

try:
    native = map_f64(s)
except Exception as exc:
    print(type(exc).__name__)
    print(str(exc))
else:
    raise SystemExit(
        f"lib.map_infer replacement was accepted: "
        f"fallback={fallback.tolist()} native={native.tolist()}"
    )
"""
    completed = _run_fresh(project, "native", body)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["TypeError", RUNTIME_ERRORS["series_method"]]


def test_lib_map_infer_callable_object_metadata_mimic_is_rejected(
    project: CertifiedProject,
) -> None:
    body = """
import numpy as np
import pandas as pd
import pandas._libs.lib as lib
from pandas_app.kernels import map_f64

_map_descriptor = pd.Series.__dict__["map"]
_original_map_infer = lib.map_infer

class CallableMapInfer:
    def __init__(self, original):
        self.__module__ = original.__module__
        self.__qualname__ = original.__qualname__
        self.__globals__ = original.__globals__
        self.__closure__ = original.__closure__
        self.__code__ = original.__code__
        self.__kwdefaults__ = original.__kwdefaults__
        self.__defaults__ = original.__defaults__

    def __call__(self, arr, mapper, convert=True, ignore_na=False):
        return np.full(len(arr), -987.0, dtype=np.float64)

CallableMapInfer.__module__ = type(_original_map_infer).__module__
CallableMapInfer.__qualname__ = type(_original_map_infer).__qualname__
lib.map_infer = CallableMapInfer(_original_map_infer)

s = pd.Series([1.0, 2.0], dtype="float64", name="values")
fallback = _map_descriptor(s, lambda value: value * 2.0)
assert fallback.tolist() == [-987.0, -987.0]

try:
    native = map_f64(s)
except Exception as exc:
    print(type(exc).__name__)
    print(str(exc))
else:
    raise SystemExit(
        f"lib.map_infer callable-object metadata mimic was accepted: "
        f"fallback={fallback.tolist()} native={native.tolist()}"
    )
"""
    completed = _run_fresh(project, "native", body)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["TypeError", RUNTIME_ERRORS["series_method"]]


def test_lib_map_infer_runtime_type_anchor_replacement_is_rejected(
    project: CertifiedProject,
) -> None:
    body = """
import numpy as np
import pandas as pd
import pandas._libs.lib as lib
import _cython_3_1_4 as cython_runtime
from pandas_app.kernels import map_f64

_map_descriptor = pd.Series.__dict__["map"]
_original_map_infer = lib.map_infer
_original_function_type = cython_runtime.cython_function_or_method

class CallableMapInfer:
    def __init__(self, original):
        self.__module__ = original.__module__
        self.__qualname__ = original.__qualname__
        self.__globals__ = original.__globals__
        self.__closure__ = original.__closure__
        self.__code__ = original.__code__
        self.__kwdefaults__ = original.__kwdefaults__
        self.__defaults__ = original.__defaults__

    def __call__(self, arr, mapper, convert=True, ignore_na=False):
        return np.full(len(arr), -986.0, dtype=np.float64)

CallableMapInfer.__module__ = _original_function_type.__module__
CallableMapInfer.__qualname__ = _original_function_type.__qualname__
cython_runtime.cython_function_or_method = CallableMapInfer
lib.map_infer = CallableMapInfer(_original_map_infer)

s = pd.Series([1.0, 2.0], dtype="float64", name="values")
fallback = _map_descriptor(s, lambda value: value * 2.0)
assert fallback.tolist() == [-986.0, -986.0]

try:
    native = map_f64(s)
except Exception as exc:
    print(type(exc).__name__)
    print(str(exc))
else:
    raise SystemExit(
        f"lib.map_infer runtime type-anchor replacement was accepted: "
        f"fallback={fallback.tolist()} native={native.tolist()}"
    )
"""
    completed = _run_fresh(project, "native", body)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["TypeError", RUNTIME_ERRORS["series_method"]]


def test_algorithms_lib_global_replacement_is_rejected(project: CertifiedProject) -> None:
    body = """
import types
import numpy as np
import pandas as pd
import pandas._libs.lib as lib
import pandas.core.algorithms as algorithms
from pandas_app.kernels import map_f64

_map_descriptor = pd.Series.__dict__["map"]
_map_array_descriptor = algorithms.map_array
_canonical_map_infer = lib.map_infer

def _forged_map_infer(arr, mapper, convert=True, ignore_na=False):
    return np.full(len(arr), -993.0, dtype=np.float64)

algorithms.lib = types.SimpleNamespace(
    map_infer=_forged_map_infer,
    map_infer_mask=lib.map_infer_mask,
)
assert algorithms.map_array is _map_array_descriptor
assert lib.map_infer is _canonical_map_infer

s = pd.Series([1.0, 2.0], dtype="float64", name="values")
fallback = _map_descriptor(s, lambda value: value * 2.0)
assert fallback.tolist() == [-993.0, -993.0]

try:
    native = map_f64(s)
except Exception as exc:
    print(type(exc).__name__)
    print(str(exc))
else:
    raise SystemExit(
        f"algorithms.lib global replacement was accepted: "
        f"fallback={fallback.tolist()} native={native.tolist()}"
    )
"""
    completed = _run_fresh(project, "native", body)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["TypeError", RUNTIME_ERRORS["series_method"]]


def test_series_constructor_property_replacement_is_rejected(
    project: CertifiedProject,
) -> None:
    body = """
import numpy as np
import pandas as pd
from pandas_app.kernels import map_f64

_map_descriptor = pd.Series.__dict__["map"]

def _forged_constructor(values, index=None, copy=False):
    return pd.Series(np.full(len(values), -992.0), index=index, dtype="float64")

pd.Series._constructor = property(lambda self: _forged_constructor)
assert pd.Series.__dict__["map"] is _map_descriptor

s = pd.Series([1.0, 2.0], dtype="float64", name="values")
fallback = _map_descriptor(s, lambda value: value * 2.0)
assert fallback.tolist() == [-992.0, -992.0]

try:
    native = map_f64(s)
except Exception as exc:
    print(type(exc).__name__)
    print(str(exc))
else:
    raise SystemExit(
        f"Series._constructor replacement was accepted: "
        f"fallback={fallback.tolist()} native={native.tolist()}"
    )
"""
    completed = _run_fresh(project, "native", body)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["TypeError", RUNTIME_ERRORS["series_method"]]


def test_constructor_series_global_replacement_is_rejected(
    project: CertifiedProject,
) -> None:
    body = """
import numpy as np
import pandas as pd
import pandas.core.series as series_mod
from pandas_app.kernels import map_f64

_public_series = pd.Series
_map_descriptor = _public_series.__dict__["map"]
_constructor_property = _public_series.__dict__["_constructor"]
s = _public_series([1.0, 2.0], dtype="float64", name="values")
_forged_result = _public_series([-991.0, -991.0], dtype="float64")

def _forged_series(values, index=None, copy=False, **kwargs):
    return _forged_result

series_mod.Series = _forged_series
assert pd.Series is _public_series
assert _public_series.__dict__["_constructor"] is _constructor_property

fallback = _map_descriptor(s, lambda value: value * 2.0)
assert fallback.tolist() == [-991.0, -991.0]

try:
    native = map_f64(s)
except Exception as exc:
    print(type(exc).__name__)
    print(str(exc))
else:
    raise SystemExit(
        f"_constructor Series global replacement was accepted: "
        f"fallback={fallback.tolist()} native={native.tolist()}"
    )
"""
    completed = _run_fresh(project, "native", body)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["TypeError", RUNTIME_ERRORS["series_method"]]


def test_series_init_transitive_behavior_is_shared_by_native_materialization(
    project: CertifiedProject,
) -> None:
    body = """
import numpy as np
import pandas as pd
from pandas_app.kernels import map_f64

_map_descriptor = pd.Series.__dict__["map"]
_original_init = pd.Series.__init__
s = pd.Series([1.0, 2.0], dtype="float64", name="values")

def _conditional_init(self, data=None, *args, **kwargs):
    if "index" in kwargs and kwargs.get("copy") is False:
        data = np.full(len(data), -985.0, dtype=np.float64)
    _original_init(self, data, *args, **kwargs)

pd.Series.__init__ = _conditional_init
fallback = _map_descriptor(s, lambda value: value * 2.0)
assert fallback.tolist() == [-985.0, -985.0]

native = map_f64(s)
assert native.tolist() == fallback.tolist(), (
    f"constructor envelope diverged: fallback={fallback.tolist()} native={native.tolist()}"
)
assert native.index is s.index
assert native.name == s.name
print("shared-constructor-envelope")
"""
    completed = _run_fresh(project, "native", body)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["shared-constructor-envelope"]


def test_series_constructor_helper_behavior_is_shared_by_native_materialization(
    project: CertifiedProject,
) -> None:
    body = """
import numpy as np
import pandas as pd
import pandas.core.series as series_mod
from pandas_app.kernels import map_f64

_map_descriptor = pd.Series.__dict__["map"]
_original_sanitize_array = series_mod.sanitize_array
s = pd.Series([1.0, 2.0], dtype="float64", name="values")

def _conditional_sanitize_array(data, index, dtype=None, copy=False, *args, **kwargs):
    if index is s.index and copy is False:
        return np.full(len(data), -984.0, dtype=np.float64)
    return _original_sanitize_array(data, index, dtype, copy, *args, **kwargs)

series_mod.sanitize_array = _conditional_sanitize_array
fallback = _map_descriptor(s, lambda value: value * 2.0)
assert fallback.tolist() == [-984.0, -984.0]

native = map_f64(s)
assert native.tolist() == fallback.tolist()
assert native.index is s.index
assert native.name == s.name
print("shared-constructor-helper")
"""
    completed = _run_fresh(project, "native", body)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["shared-constructor-helper"]


def test_inherited_ndframe_finalize_replacement_is_rejected(
    project: CertifiedProject,
) -> None:
    body = """
import pandas as pd
import pandas.core.generic as generic
from pandas_app.kernels import map_f64

_map_descriptor = pd.Series.__dict__["map"]

def _forged_finalize(self, other, method=None, **kwargs):
    self.iloc[:] = -990.0
    return self

generic.NDFrame.__finalize__ = _forged_finalize
assert pd.Series.__dict__["map"] is _map_descriptor
assert pd.Series.__finalize__ is _forged_finalize

s = pd.Series([1.0, 2.0], dtype="float64", name="values")
fallback = _map_descriptor(s, lambda value: value * 2.0)
assert fallback.tolist() == [-990.0, -990.0]

try:
    native = map_f64(s)
except Exception as exc:
    print(type(exc).__name__)
    print(str(exc))
else:
    raise SystemExit(
        f"NDFrame.__finalize__ replacement was accepted: "
        f"fallback={fallback.tolist()} native={native.tolist()}"
    )
"""
    completed = _run_fresh(project, "native", body)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["TypeError", RUNTIME_ERRORS["series_method"]]


def test_direct_series_finalize_shadow_is_rejected(project: CertifiedProject) -> None:
    body = """
import pandas as pd
import pandas.core.generic as generic
from pandas_app.kernels import map_f64

_map_descriptor = pd.Series.__dict__["map"]
_base_finalize = generic.NDFrame.__dict__["__finalize__"]

def _forged_finalize(self, other, method=None, **kwargs):
    self.iloc[:] = -989.0
    return self

pd.Series.__finalize__ = _forged_finalize
assert generic.NDFrame.__dict__["__finalize__"] is _base_finalize
assert pd.Series.__dict__["map"] is _map_descriptor

s = pd.Series([1.0, 2.0], dtype="float64", name="values")
fallback = _map_descriptor(s, lambda value: value * 2.0)
assert fallback.tolist() == [-989.0, -989.0]

try:
    native = map_f64(s)
except Exception as exc:
    print(type(exc).__name__)
    print(str(exc))
else:
    raise SystemExit(
        f"direct Series.__finalize__ shadow was accepted: "
        f"fallback={fallback.tolist()} native={native.tolist()}"
    )
"""
    completed = _run_fresh(project, "native", body)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["TypeError", RUNTIME_ERRORS["series_method"]]


def test_authority_is_revalidated_after_tamper_and_restore_in_one_process(
    project: CertifiedProject,
) -> None:
    body = f"""
import numpy as np
import pandas as pd
import pandas.core.algorithms as algorithms
from pandas_app.kernels import map_f64

_map_descriptor = pd.Series.__dict__["map"]
_original_map_array = algorithms.map_array
_expected_error = {RUNTIME_ERRORS["series_method"]!r}
s = pd.Series([1.0, 2.0], dtype="float64", name="values")

first = map_f64(s)
assert first.tolist() == [2.0, 4.0]

def _forged_map_array(arr, mapper, na_action=None, convert=True):
    return np.full(len(arr), -988.0, dtype=np.float64)

algorithms.map_array = _forged_map_array
fallback = _map_descriptor(s, lambda value: value * 2.0)
assert fallback.tolist() == [-988.0, -988.0]
try:
    map_f64(s)
except TypeError as exc:
    assert str(exc) == _expected_error
else:
    raise SystemExit("tampered authority was accepted after an earlier valid call")

algorithms.map_array = _original_map_array
restored_fallback = _map_descriptor(s, lambda value: value * 2.0)
assert restored_fallback.tolist() == [2.0, 4.0]
second = map_f64(s)
assert second.tolist() == [2.0, 4.0]
print("valid-rejected-restored")
"""
    completed = _run_fresh(project, "native", body)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["valid-rejected-restored"]


@pytest.mark.parametrize(
    "setup",
    [
        # Same original code object, but a changed default (na_action="ignore").
        (
            "import types\n"
            "_orig = pd.Series.__dict__['map']\n"
            "_patched = types.FunctionType(_orig.__code__, _orig.__globals__, "
            "_orig.__name__, ('ignore',), _orig.__closure__)\n"
            "_patched.__qualname__ = _orig.__qualname__\n"
            "_patched.__module__ = _orig.__module__\n"
            "pd.Series.map = _patched"
        ),
        # Same bytecode, changed constants (forged code object).
        (
            "import types\n"
            "_orig = pd.Series.__dict__['map']\n"
            "_code = _orig.__code__.replace(co_consts=_orig.__code__.co_consts + ('rxt-tamper',))\n"
            "_patched = types.FunctionType(_code, _orig.__globals__, _orig.__name__, "
            "_orig.__defaults__, _orig.__closure__)\n"
            "_patched.__qualname__ = _orig.__qualname__\n"
            "_patched.__module__ = _orig.__module__\n"
            "pd.Series.map = _patched"
        ),
        # map replaced by a non-function.
        "pd.Series.map = 5",
        # Series.to_numpy with the original code object but foreign globals dict.
        (
            "import types\n"
            "_orig = pd.Series.to_numpy\n"
            "_patched = types.FunctionType(_orig.__code__, dict(_orig.__globals__), "
            "_orig.__name__, _orig.__defaults__, _orig.__closure__)\n"
            "_patched.__qualname__ = _orig.__qualname__\n"
            "_patched.__module__ = _orig.__module__\n"
            "pd.Series.to_numpy = _patched"
        ),
        # Series.to_numpy ordinary replacement.
        "pd.Series.to_numpy = lambda self, *a, **k: self.values",
        # Instance to_numpy shadowing.
        "s.__dict__['to_numpy'] = lambda *a, **k: None",
    ],
)
def test_series_semantic_identity_tampering_is_rejected(
    project: CertifiedProject, setup: str
) -> None:
    body = f"""
import pandas as pd
from pandas_app.kernels import map_f64
s = pd.Series([1.0, 2.0], dtype="float64", name="values")
{setup}
try:
    map_f64(s)
except Exception as exc:
    print(type(exc).__name__)
    print(str(exc))
else:
    raise SystemExit("semantic tampering was accepted")
"""
    completed = _run_fresh(project, "native", body)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["TypeError", RUNTIME_ERRORS["series_method"]]


@pytest.mark.parametrize(
    "tamper",
    [
        # The reproduced custom-__repr__ collision: a non-None default whose
        # repr is "None". Structural (not repr-based) default checking rejects it.
        (
            "class FakeNone:\n"
            "    def __repr__(self):\n"
            "        return 'None'\n"
            "_orig = pd.Series.__dict__['map']\n"
            "_patched = types.FunctionType(_orig.__code__, _orig.__globals__, "
            "_orig.__name__, (FakeNone(),), _orig.__closure__)\n"
            "_patched.__qualname__ = _orig.__qualname__\n"
            "_patched.__module__ = _orig.__module__\n"
            "pd.Series.map = _patched"
        ),
        # bool/int interchange: to_numpy default False replaced by int 0.
        (
            "_orig = pd.Series.to_numpy\n"
            "_patched = types.FunctionType(_orig.__code__, _orig.__globals__, "
            "_orig.__name__, (None, 0, _orig.__defaults__[2]), _orig.__closure__)\n"
            "_patched.__qualname__ = _orig.__qualname__\n"
            "_patched.__module__ = _orig.__module__\n"
            "pd.Series.to_numpy = _patched"
        ),
        # forged pandas sentinel: to_numpy default no_default replaced by a fake.
        (
            "class FakeSentinel:\n"
            "    def __repr__(self):\n"
            "        return '<no_default>'\n"
            "_orig = pd.Series.to_numpy\n"
            "_patched = types.FunctionType(_orig.__code__, _orig.__globals__, "
            "_orig.__name__, (None, False, FakeSentinel()), _orig.__closure__)\n"
            "_patched.__qualname__ = _orig.__qualname__\n"
            "_patched.__module__ = _orig.__module__\n"
            "pd.Series.to_numpy = _patched"
        ),
    ],
)
def test_default_object_forgeries_are_rejected(project: CertifiedProject, tamper: str) -> None:
    body = f"""
import types
import pandas as pd
from pandas_app.kernels import map_f64
s = pd.Series([1.0, 2.0], dtype="float64", name="values")
{tamper}
try:
    map_f64(s)
except Exception as exc:
    print(type(exc).__name__)
    print(str(exc))
else:
    raise SystemExit("default forgery was accepted")
"""
    completed = _run_fresh(project, "native", body)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["TypeError", RUNTIME_ERRORS["series_method"]]


@pytest.mark.parametrize(
    "binding",
    [
        # In-place mutation of the numpy module binding used by to_numpy.
        ("np", "'not the numpy module'"),
        # Mutation of a referenced pandas callable binding.
        ("isna", "lambda *a, **k: None"),
        # Mutation of a referenced pandas class binding.
        ("ExtensionDtype", "object"),
    ],
)
def test_mutated_trusted_globals_are_rejected(
    project: CertifiedProject, binding: tuple[str, str]
) -> None:
    name, replacement = binding
    body = f"""
import pandas as pd
import pandas.core.base as base
from pandas_app.kernels import map_f64
s = pd.Series([1.0, 2.0], dtype="float64", name="values")
_saved = base.{name}
base.{name} = {replacement}
try:
    map_f64(s)
except Exception as exc:
    print(type(exc).__name__)
    print(str(exc))
else:
    raise SystemExit("mutated trusted global was accepted")
finally:
    base.{name} = _saved
"""
    completed = _run_fresh(project, "native", body)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["TypeError", RUNTIME_ERRORS["series_method"]]


def test_runtime_guards_stay_active_under_python_dash_o(project: CertifiedProject) -> None:
    # The guards are generated Rust, so ``python -O`` (which strips Python
    # ``assert`` statements) cannot disable them.
    body = """
import pandas as pd
from pandas_app.kernels import map_f64
s = pd.Series([1.0, 2.0], dtype="Float64")
try:
    map_f64(s)
except Exception as exc:
    print(type(exc).__name__)
    print(str(exc))
else:
    raise SystemExit("guard was disabled under -O")
"""
    env = {
        **os.environ,
        "PYTHONPATH": str(project.build_python_dir),
        "REXTIO_NATIVE_MODE": "native",
    }
    completed = subprocess.run(
        [sys.executable, "-O", "-c", body],
        cwd=project.project_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["TypeError", RUNTIME_ERRORS["series_f64"]]
