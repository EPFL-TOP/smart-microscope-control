"""Install Micro-Manager's demo/test device adapters for CI, authenticated.

``mmcore install --test-adapters`` lists releases through the *unauthenticated*
GitHub API, which the shared CI runners exhaust (``HTTP 403: rate limit
exceeded`` — seen on macOS in PR #40). This performs the same three steps
with ``gh`` and the job's ``GH_TOKEN``: pick the newest release of
micro-manager/mm-test-adapters whose tag starts with this pymmcore's device
interface version, download the platform zip, unpack it where pymmcore-plus
looks (``find_micromanager``).

    GH_TOKEN=... python scripts/ci/install_mm_test_adapters.py [--dest DIR]
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = "micro-manager/mm-test-adapters"


def device_interface_version() -> str:
    try:
        from pymmcore_plus.install import PYMMCORE_DIV

        return str(PYMMCORE_DIV)
    except ImportError:  # older/newer layout: parse "12.5.0.75.0"
        import pymmcore

        return pymmcore.__version__.split(".")[3]


def platform_string() -> str:
    from pymmcore_plus.install import _get_platform_arch_string

    return str(_get_platform_arch_string())


def gh(*args: str) -> str:
    return subprocess.check_output(["gh", *args], text=True, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dest", default=None, help="install directory (default: pymmcore-plus's)"
    )
    ns = ap.parse_args()

    from pymmcore_plus.install import USER_DATA_MM_PATH

    dest_root = Path(ns.dest) if ns.dest else Path(USER_DATA_MM_PATH)
    div = device_interface_version()
    releases = json.loads(gh("api", f"repos/{REPO}/releases?per_page=100"))
    tags = sorted(
        (
            str(r["tag_name"])
            for r in releases
            if str(r["tag_name"]).startswith(f"{div}.")
        ),
        reverse=True,
    )
    if not tags:
        print(f"no {REPO} release for device interface {div}", file=sys.stderr)
        return 1
    tag = tags[0]
    dest = dest_root / f"Micro-Manager-{tag}"
    if dest.exists() and any(dest.iterdir()):
        print(f"already installed: {dest}")
        return 0

    filename = f"mm-test-adapters-{platform_string()}.zip"
    with tempfile.TemporaryDirectory() as tmp:
        gh(
            "release",
            "download",
            tag,
            "--repo",
            REPO,
            "--pattern",
            filename,
            "--dir",
            tmp,
        )
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(Path(tmp) / filename) as zf:
            zf.extractall(dest)
    if platform.system() == "Darwin":
        subprocess.run(
            ["xattr", "-r", "-d", "com.apple.quarantine", str(dest)], check=False
        )
    print(f"installed {tag} -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
