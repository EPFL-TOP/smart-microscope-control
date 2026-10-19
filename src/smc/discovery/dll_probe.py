"""Ask Windows why a DLL will not load (FM-09).

MMCore reports ``Failed to load device adapter "<name>"`` for a missing
dependency, a wrong SDK, the wrong bitness and a device-interface mismatch
alike. Windows' own error code tells them apart, and :func:`probe_dll` reads it
by loading the DLL in a child process of its own (design m2 §7.3):

* **One child per DLL.** Windows resolves a module the process already holds by
  its base name, so a second probe in one process can pass while the adapter's
  dependency is missing (FM-95). A vendor ``DllMain`` that crashes or hangs
  costs that one probe, and the parent never holds a vendor DLL (FM-02).
* The child is run as :func:`smc.discovery.mm_inventory.mm_section`'s is, and
  with the same helpers: files, never pipes (FM-04), a neutral working
  directory (FM-33), no console window (FM-22), waited in slices (FM-83) and
  killed with its tree at the deadline (FM-35).

Generic: any adapter, not only Nikon's. Nothing here imports ``smc.hardware``
or ``pymmcore_plus``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from smc.discovery import mm_inventory

__all__ = [
    "CHILD_COMMAND",
    "DLL_PROBE_TIMEOUT_S",
    "WINERROR_HINTS",
    "DllProbe",
    "describe_winerror",
    "probe_dll",
]

logger = logging.getLogger("smc.discovery.dll_probe")

#: Design m1 §10: every OS call has a 20 s timeout.
DLL_PROBE_TIMEOUT_S = 20.0

#: How the parent starts the child; a module attribute so tests can replace it
#: with a canned child, since no test may load a real DLL (FM-40).
CHILD_COMMAND: list[str] = [sys.executable, "-m", "smc.discovery._dll_child"]

#: A ``.dll`` can only be loaded on Windows. Module level so that a test can
#: run the Windows branch on every OS, and the other branch on Windows.
_WINDOWS = sys.platform == "win32"

#: What the four loader errors that this project has met mean for an adapter
#: (``nikon-control``'s table, the first one made generic).
WINERROR_HINTS: Mapping[int, str] = {
    126: (
        "a DLL it depends on could not be found; for a vendor adapter, usually "
        "the vendor's driver or runtime DLL, not beside the adapter or not "
        "installed"
    ),
    127: (
        "an entry point is missing; the vendor DLL is not the version this "
        "adapter was built against"
    ),
    193: (
        "wrong architecture; a 32-bit DLL under a 64-bit Micro-Manager, or the reverse"
    ),
    1114: (
        "the DLL loaded but its initialisation routine failed; usually a vendor "
        "SDK mismatch, or hardware that is off"
    ),
}


@dataclass(frozen=True, slots=True)
class DllProbe:
    """What Windows said when asked to load one DLL.

    ``loads`` is ``None`` when there is no answer: nothing was tried (not
    Windows, no such file) or the child gave none (it hung or crashed). That is
    not a failure of the DLL, and callers must not report it as one.
    """

    path: Path
    loads: bool | None
    winerror: int | None
    #: One line, ready to print.
    detail: str


def describe_winerror(code: int | None, strerror: str) -> str:
    """Windows' text for a load failure, with what the code means when it is known.

    ``WinError 126: <text> — <hint>``; an unknown code has no hint, and no code
    gives the text alone. The text is the PC's own language, so it is joined on
    one line: a vendor message with a line break must not split a report.
    """
    text = " ".join(strerror.split())
    if code is None:
        return text
    described = f"WinError {code}: {text}"
    hint = WINERROR_HINTS.get(code)
    return f"{described} — {hint}" if hint else described


def _entries(extra_path: Sequence[Path]) -> list[str]:
    """The folders for the child's ``PATH``, as ``open_core`` writes its own (FM-81)."""
    entries: list[str] = []
    for folder in extra_path:
        text = str(folder)
        if os.pathsep in text:
            logger.warning(
                "adapter folder %s contains %r, so it cannot be an entry of PATH; "
                "the probe leaves it out",
                text,
                os.pathsep,
            )
        else:
            entries.append(text)
    return entries


def _read_verdict(stdout: Path) -> tuple[bool, int | None, str] | None:
    """The child's verdict from its stdout file, whatever its exit code (FM-02).

    The last line that is the protocol wins: a DLL may print before it (FM-01).
    ``None`` when no line is: output that is not the protocol is no verdict.
    """
    try:
        text = stdout.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in reversed(text.splitlines()):
        try:
            answer = json.loads(line)
        except ValueError:
            continue
        if not isinstance(answer, dict) or not isinstance(answer.get("loads"), bool):
            continue
        code = answer.get("winerror")
        winerror = (
            code if isinstance(code, int) and not isinstance(code, bool) else None
        )
        strerror = answer.get("strerror")
        return (
            answer["loads"],
            winerror,
            strerror if isinstance(strerror, str) else "",
        )
    return None


def probe_dll(
    path: Path,
    *,
    extra_path: Sequence[Path] = (),
    timeout_s: float = DLL_PROBE_TIMEOUT_S,
) -> DllProbe:
    """Ask Windows to load ``path`` in a child process of its own.

    Args:
        path: The DLL: the adapter (``mmgr_dal_<name>.dll``) or its vendor
            driver. Loaded as MMCore loads it, with its own folder first.
        extra_path: Folders for the front of the child's ``PATH``, in order, as
            ``open_core`` puts a profile's adapter folders there. An entry that
            contains ``os.pathsep`` is left out, with a WARNING.
        timeout_s: How long the child may take. A vendor ``DllMain`` that hangs
            costs this long, then the child and its tree are killed.

    Returns:
        The verdict, or a probe with ``loads=None`` that says why there is none.
        It does not raise for a DLL that fails to load: that is the finding.

    Raises:
        ValueError: ``timeout_s`` is not a finite positive number.
        KeyboardInterrupt: Ctrl-C; the child's tree is killed first.
    """
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError(
            f"timeout_s must be a finite number of seconds, got {timeout_s}"
        )
    if not _WINDOWS:
        return DllProbe(path, None, None, "not Windows: a .dll cannot be loaded here")
    try:
        present = path.exists()
    except OSError as exc:
        return DllProbe(path, None, None, f"{path} cannot be looked at: {exc}")
    if not present:
        return DllProbe(path, None, None, f"{path} does not exist")

    command = [*CHILD_COMMAND, str(path)]
    environment = dict(os.environ)
    environment["PATH"] = os.pathsep.join(
        [
            *_entries(extra_path),
            *([environment["PATH"]] if environment.get("PATH") else []),
        ]
    )
    logger.info("dll probe: %s", path)

    with tempfile.TemporaryDirectory(
        prefix="smc-dllprobe-", ignore_cleanup_errors=True
    ) as tmp:
        folder = Path(tmp)
        stdout_path = folder / "stdout.txt"
        with stdout_path.open("wb") as out, (folder / "stderr.txt").open("wb") as err:
            try:
                child = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=err,
                    cwd=folder,
                    env=environment,
                    # So the whole tree can be killed together (FM-35);
                    # ignored on Windows, which uses creationflags instead.
                    start_new_session=True,
                    creationflags=mm_inventory._NO_WINDOW,
                )
            except OSError as exc:
                return DllProbe(
                    path, None, None, f"the probe child could not start: {exc}"
                )
            unstoppable = False
            kill_detail: str | None = None
            try:
                try:
                    returncode: int | None = mm_inventory._wait_in_slices(
                        child, timeout_s
                    )
                except subprocess.TimeoutExpired:
                    kill_detail = mm_inventory._kill_child_tree(child)
                    returncode = None
                    try:
                        # Bounded: a process stuck in a driver call may not die.
                        mm_inventory._wait_in_slices(child, mm_inventory._KILL_WAIT_S)
                    except subprocess.TimeoutExpired:
                        unstoppable = True
            except BaseException:
                # Ctrl-C while waiting must not leave a process running (FM-35).
                mm_inventory._kill_child_tree(child)
                raise

        answer = _read_verdict(stdout_path)

    if answer is not None:
        loads, winerror, strerror = answer
        if loads:
            return DllProbe(path, True, None, "loads")
        return DllProbe(path, False, winerror, describe_winerror(winerror, strerror))
    if returncode is None:
        detail = f"did not return within {timeout_s:g} s: its initialisation hung"
        if unstoppable or kill_detail:
            detail += (
                "; the probe child could not be stopped (it may still hold the "
                "hardware)"
            )
            if kill_detail:
                # taskkill's text can span two lines; the detail is one (FM-37).
                detail += f"; {' '.join(kill_detail.split())}"
        return DllProbe(path, None, None, detail)
    return DllProbe(
        path,
        None,
        None,
        "the probe child ended with exit code "
        f"{mm_inventory.exit_detail(returncode, sys.platform)} and no verdict",
    )
