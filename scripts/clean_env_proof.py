#!/usr/bin/env python3
"""Prove that ``rextio-pandas`` resolves its exact VCS core in a clean env.

This builds the current checkout into a wheel, installs that wheel into a
throwaway virtual environment *with* full dependency resolution (it never skips
dependency resolution), and then asserts the installed provenance:

* the core's ``direct_url.json`` records a git VCS install pinned to the exact
  integrated commit and the ``rextio-core-next`` project (not released 0.1.2),
* ``PLUGIN_API_VERSION == "1.3"``,
* the imported ``rextio`` and ``rextio_pandas`` module paths live inside the
  fresh environment,
* the selected ``rextio.plugins`` entry point is provided by the installed
  ``rextio-pandas`` wheel built from this checkout.

The credential-free ``git+https://github.com/rextio/rextio-core-next.git`` URL
is resolved without exposing any token: a private ``rextio/rextio-core-next``
repository is normally cloned with the caller's own git credentials, but for an
offline/hermetic proof this script injects a transient ``insteadOf`` redirect
to a local mirror via ``GIT_CONFIG_*`` environment variables (never the global
git config, and never a URL containing a secret). The recorded commit id is the
real proof; the transport is interchangeable.

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
CORE_COMMIT = "ac2b79d304f13abaaecaf7714f897574c3b6256f"
CORE_PROJECT = "rextio-core-next"
DEFAULT_MIRROR = Path("/Volumes/Data/workspace/rextio/rextio-core-next")


def _fail(message: str) -> NoReturn:
    print(f"CLEAN-ENV PROOF FAILED: {message}", file=sys.stderr)
    raise SystemExit(1)


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


def _redirect_env(mirror: Path) -> dict[str, str]:
    env = dict(os.environ)
    if mirror.exists():
        # Credential-free, offline redirect of the public URL to a local mirror.
        env["GIT_CONFIG_COUNT"] = "1"
        env["GIT_CONFIG_KEY_0"] = f"url.file://{mirror}.insteadOf"
        env["GIT_CONFIG_VALUE_0"] = f"https://github.com/rextio/{CORE_PROJECT}.git"
    return env


def main(argv: list[str] | None = None) -> int:
    """Build, install into a clean env, and assert core-resolution provenance."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mirror",
        type=Path,
        default=DEFAULT_MIRROR,
        help="Local git mirror used for a credential-free offline redirect.",
    )
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="rextio-pandas-cleanenv-") as tmp:
        tmp_path = Path(tmp)
        dist_dir = tmp_path / "dist"
        wheel = _build_wheel(dist_dir)

        env_dir = tmp_path / "venv"
        venv.EnvBuilder(with_pip=True).create(env_dir)
        py = env_dir / "bin" / "python"
        if not py.exists():  # pragma: no cover - non-posix layout
            py = env_dir / "Scripts" / "python.exe"

        install_env = _redirect_env(args.mirror)
        _run(
            [str(py), "-m", "pip", "install", "--disable-pip-version-check", str(wheel)],
            env=install_env,
        )

        probe = r"""
import importlib.metadata as md
import json, sys

report = {}
import rextio, rextio_pandas
from rextio.plugins.api import PLUGIN_API_VERSION
report["api"] = PLUGIN_API_VERSION
report["rextio_file"] = rextio.__file__
report["rextio_pandas_file"] = rextio_pandas.__file__

core = md.distribution("rextio")
report["core_version"] = core.version
report["core_direct_url"] = core.read_text("direct_url.json")

pandas_dist = md.distribution("rextio-pandas")
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

    if report["api"] != "1.3":
        _fail(f"PLUGIN_API_VERSION is {report['api']!r}, expected '1.3'")
    if not report["rextio_file"].startswith(env_root):
        _fail(f"imported rextio is outside the fresh env: {report['rextio_file']}")
    if not report["rextio_pandas_file"].startswith(env_root):
        _fail(f"imported rextio_pandas is outside the fresh env: {report['rextio_pandas_file']}")

    core_du = json.loads(report["core_direct_url"] or "{}")
    vcs = core_du.get("vcs_info") or {}
    if vcs.get("vcs") != "git":
        _fail(f"core direct_url is not a git VCS install: {core_du}")
    if vcs.get("commit_id") != CORE_COMMIT:
        _fail(f"core commit_id is {vcs.get('commit_id')!r}, expected {CORE_COMMIT}")
    requested = vcs.get("requested_revision")
    if requested not in (None, CORE_COMMIT):
        _fail(f"core requested_revision is {requested!r}, expected the pinned commit")
    if CORE_PROJECT not in core_du.get("url", ""):
        _fail(f"core direct_url URL does not name {CORE_PROJECT}: {core_du.get('url')!r}")
    if "ghp_" in core_du.get("url", "") or "x-access-token" in core_du.get("url", ""):
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
