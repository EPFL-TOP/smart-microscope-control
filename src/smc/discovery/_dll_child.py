"""Child process that asks Windows whether one DLL loads (``smc.discovery.dll_probe``).

``python -m smc.discovery._dll_child PATH``

It exists so that the answer costs a process of its own. Windows resolves a
module that the process already holds by its base name, so a second probe in
one process can pass while the adapter's own dependency is missing; a vendor
``DllMain`` can crash or hang, and the parent must never hold a vendor DLL
(FM-02, design m2 §7.3). One child per DLL, then, and it:

* guards itself first with ``SetErrorMode`` (FM-03): a missing dependency would
  otherwise open a modal dialog that looks like a hang;
* loads the DLL with ``LOAD_WITH_ALTERED_SEARCH_PATH`` (``winmode=0x8``): its own
  folder first, then the standard order, ``PATH`` included, which is where each
  Nikon driver is found. That MMCore searches the same way is assumed;
* writes one line of ASCII JSON to stdout (FM-23), on a line of its own so that
  a DLL that printed without a newline does not hide it (FM-01);
* never frees the DLL and ends with ``os._exit(0)`` after the last write: the
  process is about to end and unloading a vendor DLL can hang.

It imports nothing from ``pymmcore_plus``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Windows SetErrorMode flags: no "missing DLL" dialog, no crash-reporter dialog.
_SEM_FAILCRITICALERRORS = 0x0001
_SEM_NOGPFAULTERRORBOX = 0x0002

#: ``LOAD_WITH_ALTERED_SEARCH_PATH`` for ``ctypes.WinDLL(winmode=...)``.
_LOAD_WITH_ALTERED_SEARCH_PATH = 0x8


def _no_error_dialogs() -> None:
    """Keep Windows from opening a dialog for this child (FM-03, FM-26)."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        set_error_mode = kernel32.SetErrorMode
        set_error_mode.argtypes = [wintypes.UINT]
        set_error_mode.restype = wintypes.UINT
        set_error_mode(_SEM_FAILCRITICALERRORS | _SEM_NOGPFAULTERRORBOX)


def _load(path: Path) -> OSError | None:
    """Load ``path`` the way MMCore would; the ``OSError`` Windows raised, or ``None``."""
    if sys.platform == "win32":
        import ctypes

        try:
            ctypes.WinDLL(str(path), winmode=_LOAD_WITH_ALTERED_SEARCH_PATH)
        except OSError as exc:
            return exc
        return None
    raise NotImplementedError("loading a .dll needs Windows")


def verdict(exc: OSError | None) -> dict[str, object]:
    """The protocol line's content for the outcome of :func:`_load`.

    ``winerror`` is the Windows error code (``OSError.winerror``, set only on
    Windows); ``strerror`` is Windows' own text, in the language of the PC.
    """
    if exc is None:
        return {"loads": True}
    winerror = getattr(exc, "winerror", None)
    return {
        "loads": False,
        "winerror": winerror if isinstance(winerror, int) else None,
        "strerror": exc.strerror or str(exc),
    }


def encode(answer: dict[str, object]) -> str:
    """One line of ASCII JSON: a localised Windows message survives any code page."""
    return json.dumps(answer, ensure_ascii=True)


def main(argv: list[str] | None = None) -> int:
    """Probe the one DLL named on the command line and print the verdict."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        sys.stderr.write("usage: python -m smc.discovery._dll_child PATH\n")
        return 2
    _no_error_dialogs()
    answer = verdict(_load(Path(args[0])))
    # A leading newline: a DLL that printed without one must not share the line.
    sys.stdout.write("\n" + encode(answer) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
