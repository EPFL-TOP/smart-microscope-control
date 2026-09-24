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
    """Keep the OS from opening a crash dialog for this child (FM-03).

    Windows: a missing DLL or a crash opens a modal dialog that looks like
    a hang. macOS: a crashing ``Python.app`` (Homebrew) makes ReportCrash
    write a report and show "Python quit unexpectedly" — once per crashing
    adapter, and once per ``pytest`` run on the NotificationTester probe.

    On macOS libc's own ``_exit`` becomes the raw C handler of the crash
    signals, so the process ends with ``_exit(signum)`` before the kernel
    treats it as a crash: exit code 6 for ``abort()`` instead of -6, and no
    report. A Python ``signal.signal`` handler would never run for a native
    ``abort()``, and ``faulthandler`` re-raises into the default action.
    Must run before ``pymmcore`` is imported or any adapter is loaded; the
    parent process never installs it. Linux shows no dialog and is left as
    it is.
    """
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.kernel32.SetErrorMode(
            _SEM_FAILCRITICALERRORS | _SEM_NOGPFAULTERRORBOX
        )
    elif sys.platform == "darwin":
        import ctypes
        import signal

        libc = ctypes.CDLL(None)
        libc.signal.argtypes = [ctypes.c_int, ctypes.c_void_p]
        libc.signal.restype = ctypes.c_void_p
        exit_address = ctypes.cast(libc._exit, ctypes.c_void_p).value
        for sig in (
            signal.SIGABRT,
            signal.SIGSEGV,
            signal.SIGBUS,
            signal.SIGILL,
            signal.SIGFPE,
        ):
            libc.signal(int(sig), exit_address)


def _progress(line: str) -> None:
    sys.stderr.write(line + "\n")
    sys.stderr.flush()


def _clean_string(value: str) -> str:
    """Recover text a vendor adapter returned as raw, non-UTF-8 bytes (FM-05).

    pymmcore decodes adapter strings with ``surrogateescape``: valid UTF-8
    bytes decode correctly, and anything else becomes a lone surrogate that
    ``json.dumps`` cannot serialise. Re-encoding with ``surrogateescape``
    recovers the original bytes, which are then read as UTF-8, or as cp1252
    (what a Windows adapter actually emits) when they are not valid UTF-8. A
    surrogate that ``surrogateescape`` itself cannot re-encode — not one it
    would have produced, but nothing here trusts that — is written out as
    literal escape text instead of raising.
    """
    try:
        raw = value.encode("utf-8", "surrogateescape")
    except UnicodeEncodeError:
        return value.encode("utf-8", "backslashreplace").decode("utf-8")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252", "replace")


def _clean_tree(obj: object) -> object:
    """Apply :func:`_clean_string` through the dicts, lists and strings of ``obj``."""
    if isinstance(obj, str):
        return _clean_string(obj)
    if isinstance(obj, dict):
        return {
            (_clean_string(k) if isinstance(k, str) else k): _clean_tree(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_clean_tree(v) for v in obj]
    return obj


def _write(result: ChildResult, path: Path) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(_clean_tree(result.model_dump(mode="json")), ensure_ascii=True),
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
