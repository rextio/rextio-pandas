#!/usr/bin/env python3
"""Prove that ``rextio-pandas`` resolves public ``rextio>=0.1.3,<0.2`` in a clean env.

This builds the current checkout into a wheel, installs that wheel into a
throwaway virtual environment *with* full dependency resolution (it never skips
dependency resolution), and then asserts the installed provenance:

* the host interpreter is CPython 3.11 (``requires-python >=3.11,<3.12``),
* the installed ``rextio`` version satisfies ``>=0.1.3,<0.2``,
* ``PLUGIN_API_VERSION == "1.3"``,
* the imported ``rextio`` and ``rextio_pandas`` module paths live inside the
  fresh environment,
* the selected ``rextio.plugins`` entry point is provided by the installed
  ``rextio-pandas`` wheel built from this checkout.

Unsupported interpreters fail immediately with a clear message before any
build or install work.

By default pip resolves ``rextio`` from the public index. For offline or
pre-release local proofs, pass ``--find-links DIR`` (or set
``REXTIO_FIND_LINKS``) so pip can see a pre-built ``rextio`` wheel/sdist.
No machine-local absolute path is hard-coded.

Exit code is non-zero on any failed gate so it survives ``python -O``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import venv
from pathlib import Path
from typing import NoReturn

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_REXTIO_SPEC = ">=0.1.3,<0.2"
REQUIRED_PLUGIN_API = "1.3"
REQUIRED_PYTHON = (3, 11)
REQUIRED_IMPLEMENTATION = "cpython"


def _fail(message: str) -> NoReturn:
    print(f"CLEAN-ENV PROOF FAILED: {message}", file=sys.stderr)
    raise SystemExit(1)


def _interpreter_supported(
    version_info: tuple[int, ...] | None = None,
    implementation: str | None = None,
) -> bool:
    """Return True when the interpreter is the supported CPython 3.11 minor."""
    info = sys.version_info if version_info is None else version_info
    impl = sys.implementation.name if implementation is None else implementation
    return impl == REQUIRED_IMPLEMENTATION and info[:2] == REQUIRED_PYTHON


def _require_supported_interpreter() -> None:
    """Fail closed before any build/install work on unsupported interpreters."""
    if _interpreter_supported():
        return
    _fail(
        "rextio-pandas 0.1.0 requires CPython 3.11 "
        f"(requires-python >=3.11,<3.12); got {sys.implementation.name} "
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )


def _run(command: list[str], *, env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=env,
    )
    if completed.returncode != 0:
        _fail(
            f"command {command!r} exited {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed.stdout


def _build_wheel(dist_dir: Path) -> Path:
    _run([sys.executable, "-m", "build", "--wheel", "--outdir", str(dist_dir), str(ROOT)])
    wheels = sorted(dist_dir.glob("rextio_pandas-*.whl"))
    if not wheels:
        _fail("wheel build produced no rextio_pandas wheel")
    return wheels[-1]


def _parse_version(value: str) -> tuple[int, int, int]:
    core = value.split("+", 1)[0].split("-", 1)[0]
    parts = core.split(".")
    major = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 0
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    patch = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
    return major, minor, patch


def _rextio_version_supported(value: str) -> bool:
    version = _parse_version(value)
    return (0, 1, 3) <= version < (0, 2, 0)


def main(argv: list[str] | None = None) -> int:
    """Build, install into a clean env, and assert dependency provenance."""
    _require_supported_interpreter()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--find-links",
        type=Path,
        default=None,
        help=(
            "Optional local directory of wheels/sdists for offline resolution "
            "(also accepted via REXTIO_FIND_LINKS). No default machine path."
        ),
    )
    args = parser.parse_args(argv)

    find_links = args.find_links
    if find_links is None:
        env_links = os.environ.get("REXTIO_FIND_LINKS", "").strip()
        if env_links:
            find_links = Path(env_links).expanduser()

    with tempfile.TemporaryDirectory(prefix="rextio-pandas-cleanenv-") as tmp:
        tmp_path = Path(tmp)
        dist_dir = tmp_path / "dist"
        wheel = _build_wheel(dist_dir)

        env_dir = tmp_path / "venv"
        venv.EnvBuilder(with_pip=True).create(env_dir)
        py = env_dir / "bin" / "python"
        if not py.exists():  # pragma: no cover - non-posix layout
            py = env_dir / "Scripts" / "python.exe"

        install_cmd = [
            str(py),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
        ]
        if find_links is not None:
            if not find_links.is_dir():
                _fail(f"--find-links {find_links} is not a directory")
            install_cmd.extend(["--find-links", str(find_links.resolve())])
        install_cmd.append(str(wheel))
        _run(install_cmd)

        probe = r"""
import importlib.metadata as md
import json, sys

report = {}
import rextio, rextio_pandas
from rextio.plugins.api import PLUGIN_API_VERSION
report["api"] = PLUGIN_API_VERSION
report["rextio_file"] = rextio.__file__
report["rextio_pandas_file"] = rextio_pandas.__file__
report["rextio_pandas_version"] = rextio_pandas.__version__

core = md.distribution("rextio")
report["core_version"] = core.version
report["core_direct_url"] = core.read_text("direct_url.json")

pandas_dist = md.distribution("rextio-pandas")
report["plugin_version"] = pandas_dist.version
report["plugin_direct_url"] = pandas_dist.read_text("direct_url.json")

eps = [
    {"name": ep.name, "value": ep.value, "dist": ep.dist.name if ep.dist else None}
    for ep in md.entry_points(group="rextio.plugins")
    if ep.name == "rextio-pandas"
]
report["entry_points"] = eps
print(json.dumps(report))
"""
        raw = _run([str(py), "-c", probe])
        report = json.loads(raw.splitlines()[-1])

    env_root = str(env_dir.resolve())

    if report["api"] != REQUIRED_PLUGIN_API:
        _fail(f"PLUGIN_API_VERSION is {report['api']!r}, expected {REQUIRED_PLUGIN_API!r}")
    if not _rextio_version_supported(report["core_version"]):
        _fail(
            f"rextio {report['core_version']!r} is outside the supported range "
            f"{REQUIRED_REXTIO_SPEC}"
        )
    if report.get("plugin_version") != "0.1.0":
        _fail(
            f"installed rextio-pandas metadata version is {report.get('plugin_version')!r}, "
            "expected 0.1.0"
        )
    if report.get("rextio_pandas_version") != "0.1.0":
        _fail(
            f"imported rextio_pandas.__version__ is {report.get('rextio_pandas_version')!r}, "
            "expected 0.1.0"
        )
    if not report["rextio_file"].startswith(env_root):
        _fail(f"imported rextio is outside the fresh env: {report['rextio_file']}")
    if not report["rextio_pandas_file"].startswith(env_root):
        _fail(f"imported rextio_pandas is outside the fresh env: {report['rextio_pandas_file']}")

    core_du_raw = report.get("core_direct_url")
    if core_du_raw:
        core_du = json.loads(core_du_raw)
        url = core_du.get("url", "")
        if (
            "ghp_" in url
            or "x-access-token" in url
            or "@" in url.split("://", 1)[-1].split("/", 1)[0]
        ):
            _fail("core direct_url URL leaks a credential")

    eps = report["entry_points"]
    if not any(
        ep["value"] == "rextio_pandas.plugin:plugin" and ep["dist"] == "rextio-pandas" for ep in eps
    ):
        _fail(f"rextio-pandas entry point not provided by the installed wheel: {eps}")

    print(json.dumps(report, indent=2, sort_keys=True))
    print("CLEAN-ENV PROOF OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
