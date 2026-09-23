"""Child process that loads device adapters for ``smc discover``.

``python -m smc.discovery._mm_child --result FILE (--adapters A,B | --all) [--probe NAME]``

It exists so a vendor DLL that hangs or crashes on load takes down this
process and not the survey (see :mod:`smc.discovery.mm_inventory` and
``docs/design/failure-modes.md`` FM-01 to FM-04):

* the result goes to ``FILE``, never stdout, which a DLL may print into;
* it is written atomically after the listing and again after the probe, so
  a crash during the probe keeps the listing;
* ``listing <adapter>`` / ``probing <adapter>/<device>`` go to stderr before
  each step, so the parent can say what was running when the child died;
* the process ends with ``os._exit(0)`` after the last write: unloading a
  vendor DLL at interpreter teardown can crash or hang, and there is nothing
  left to lose by skipping it.

The JSON is ASCII (``ensure_ascii=True``): device descriptions can hold
non-ASCII characters, and the file must read the same on every code page.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from smc.discovery.mm_inventory import ChildResult, list_adapters, probe_adapter

# Windows SetErrorMode flags: no "missing DLL" dialog, no crash-reporter dialog.
_SEM_FAILCRITICALERRORS = 0x0001
_SEM_NOGPFAULTERRORBOX = 0x0002


def _no_error_dialogs() -> None:
    """Keep Windows from opening a modal dialog that would look like a hang (FM-03)."""
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.kernel32.SetErrorMode(
            _SEM_FAILCRITICALERRORS | _SEM_NOGPFAULTERRORBOX
        )


def _progress(line: str) -> None:
    sys.stderr.write(line + "\n")
    sys.stderr.flush()


def _write(result: ChildResult, path: Path) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=True),
        encoding="utf-8",
    )
    os.replace(partial, path)


def main(argv: list[str] | None = None) -> int:
    """List adapter devices (and probe one adapter), writing the result file."""
    _no_error_dialogs()
    parser = argparse.ArgumentParser(prog="smc.discovery._mm_child")
    parser.add_argument("--result", required=True, type=Path, help="result file")
    parser.add_argument("--adapters", default="", help="comma-separated names")
    parser.add_argument("--all", action="store_true", help="every installed adapter")
    parser.add_argument("--probe", default=None, help="one adapter to initialise")
    args = parser.parse_args(argv)

    from pymmcore_plus import CMMCorePlus

    core = CMMCorePlus()
    names = [n for n in args.adapters.split(",") if n]
    if args.all:
        names = sorted(set(names) | {str(n) for n in core.getDeviceAdapterNames()})
    result = ChildResult()
    for name in names:
        _progress(f"listing {name}")
        result.listed += list_adapters(core, [name])
    _write(result, args.result)
    if args.probe:
        result.probe = probe_adapter(args.probe, progress=_progress)
        _write(result, args.result)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
