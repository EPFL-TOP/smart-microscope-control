"""Child process that loads device adapters for ``smc discover``.

``python -m smc.discovery._mm_child --result FILE (--adapters A,B | --all) [--probe NAME [--port=PORT]]``

``python -m smc.discovery._mm_child --result FILE --build=ADAPTER [--camera=ADAPTER] [--skip=NAME]... [--search-path=DIR]...``

It exists so a vendor DLL that hangs or crashes on load takes down this
process and not the survey or the build (see :mod:`smc.discovery.mm_inventory`
and ``docs/design/failure-modes.md`` FM-01 to FM-04):

* the result goes to ``FILE``, never stdout, which a DLL may print into;
* it is written atomically after the listing, after every probed device and
  once more at the end, so a crash during the probe keeps the listing and
  the devices already probed, the probe marked ``complete=False``;
* ``listing <adapter>`` / ``probing <adapter>/<device>`` /
  ``reading <adapter>/<device>`` go to stderr before each step, so the parent
  can say what was running when the child died;
* the process ends with ``os._exit(0)`` after the last write: unloading a
  vendor DLL at interpreter teardown can crash or hang, and there is nothing
  left to lose by skipping it.

The JSON is ASCII (``ensure_ascii=True``): device descriptions can hold
non-ASCII characters, and the file must read the same on every code page.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from smc.discovery.mm_inventory import ChildResult, list_adapters, walk_adapter
from smc.discovery.models import BuildScan, BuiltDevice, ProbeResult
from smc.hardware.core import CoreError, close_core, new_core

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus

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


def _use_utf8_stderr() -> None:
    r"""Write the progress lines in UTF-8, which is how the parent reads them.

    Redirected to a file, a Windows stream is in the ANSI code page (cp1252) and
    writes what it cannot encode as ``\uXXXX`` text: the device named in the
    parent's ``re-run with --skip <device>`` hint would not be the device that
    crashed, and pasting it would load that device again. Only what cannot be
    encoded at all (a lone surrogate) is escaped. A stream that cannot change
    keeps the encoding it has.
    """
    reconfigure = getattr(sys.stderr, "reconfigure", None)
    if reconfigure is not None:
        with contextlib.suppress(OSError, ValueError):
            reconfigure(encoding="utf-8", errors="backslashreplace")


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


#: Windows refuses to replace a file that antivirus or a search indexer holds
#: open for a moment (``PermissionError``), and the file was written a moment
#: ago, often one device ago (FM-24): a few short waits, not a lost result.
_REPLACE_TRIES = 5
_REPLACE_WAIT_S = 0.05


def _write(result: ChildResult, path: Path) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(_clean_tree(result.model_dump(mode="json")), ensure_ascii=True),
        encoding="utf-8",
    )
    for attempt in range(1, _REPLACE_TRIES + 1):
        try:
            os.replace(partial, path)
        except PermissionError:
            if attempt == _REPLACE_TRIES:
                raise
            time.sleep(_REPLACE_WAIT_S)
        else:
            return


def _write_checkpoint(result: ChildResult, path: Path) -> None:
    """Write the file between two devices; a failure is a line, not the end.

    A failed write between two devices is a line on stderr and not the end of
    the probe: Windows antivirus can hold a file that was just written (FM-24),
    and the probe is minutes of hardware work. Only ``OSError`` is swallowed;
    the final write, after the probe, is not guarded. The line does not start
    with ``probing ``, which the parent reads as the device that was running.
    """
    try:
        _write(result, path)
    except OSError as exc:
        _progress(f"checkpoint skipped: the result file could not be written: {exc}")


def _checkpoint_writer(
    result: ChildResult,
    path: Path,
    store: Callable[[ProbeResult], None] | None = None,
) -> Callable[[ProbeResult], None]:
    """A walk's checkpoint: put the snapshot in the result and write the file.

    ``store`` says where the snapshot goes: ``result.probe`` by default, and
    ``result.build`` for the walks of a build (see :func:`_build`).
    """

    def checkpoint(snapshot: ProbeResult) -> None:
        if store is None:
            result.probe = snapshot
        else:
            store(snapshot)
        _write_checkpoint(result, path)

    return checkpoint


def _probe(
    core: CMMCorePlus, result: ChildResult, path: Path, adapter: str, port: str | None
) -> None:
    """Walk one adapter and write the whole result *before* the core is released.

    The walk's last checkpoint holds the rows, not the ``error`` and the notes
    that come after the last one, and not ``complete``. Unloading a vendor's
    devices can hang or crash (FM-02): written after it, the whole result would
    be lost to a death in the unload, and the report would say INCOMPLETE for a
    probe that had finished. So this is :func:`probe_adapter` with the final
    write between the walk and the release; the core is released in a
    ``finally`` all the same.
    """
    try:
        result.probe = walk_adapter(
            core,
            adapter,
            port=port,
            progress=_progress,
            checkpoint=_checkpoint_writer(result, path),
        )
        _write(result, path)
    finally:
        close_core(core)


def _read_devices(
    core: CMMCorePlus, walks: Sequence[ProbeResult]
) -> tuple[list[BuiltDevice], list[str]]:
    """What the configuration needs of each device that initialised (§10.3, point 4).

    A device's parent and a State device's labels exist only on the loaded
    device, so they are read here, before the core is released. A read that
    raises leaves its field empty and adds a note: the build goes on, and the
    file says what it could not know. ``reading <adapter>/<device>`` goes to
    stderr first, so a crash in a driver's getter names the device to leave out.
    """
    devices: list[BuiltDevice] = []
    notes: list[str] = []
    for walk in walks:
        for row in walk.devices:
            if row.status != "ok":
                continue
            _progress(f"reading {walk.adapter}/{row.name}")
            parent: str | None = None
            try:
                parent = str(core.getParentLabel(row.name)) or None
            except Exception as exc:
                notes.append(f"{row.name}: its parent could not be read: {exc}")
            labels: list[str] = []
            if row.type == "State":
                try:
                    labels = [str(label) for label in core.getStateLabels(row.name)]
                except Exception as exc:
                    notes.append(
                        f"{row.name}: its state labels could not be read: {exc}"
                    )
            devices.append(
                BuiltDevice(
                    label=row.name,
                    library=walk.adapter,
                    name=row.name,
                    type=row.type,
                    parent=parent,
                    state_labels=labels,
                )
            )
    return devices, notes


def _build(
    core: CMMCorePlus,
    result: ChildResult,
    path: Path,
    stand: str,
    camera: str | None,
    skip: Sequence[str],
) -> None:
    """Walk the stand (and a camera's adapter), read the devices, write the scan.

    The stand and the camera are walked on the same core, both always: a stand
    that failed still has its camera diagnosed in the same visit. Every
    checkpoint writes the scan so far as ``complete=False``; the whole scan is
    written **before** the core is released, in a ``finally`` as :func:`_probe`
    does (FM-02), so a death while the devices are unloaded costs a note and
    the file is still written.
    """
    try:

        def put_stand(snapshot: ProbeResult) -> None:
            result.build = BuildScan(stand=snapshot, complete=False)

        _progress(f"listing {stand}")
        stand_walk = walk_adapter(
            core,
            stand,
            skip=skip,
            progress=_progress,
            checkpoint=_checkpoint_writer(result, path, put_stand),
        )
        camera_walk: ProbeResult | None = None
        if camera:

            def put_camera(snapshot: ProbeResult) -> None:
                result.build = BuildScan(
                    stand=stand_walk, camera=snapshot, complete=False
                )

            _progress(f"listing {camera}")
            camera_walk = walk_adapter(
                core,
                camera,
                skip=skip,
                progress=_progress,
                checkpoint=_checkpoint_writer(result, path, put_camera),
            )
        walks = [stand_walk] if camera_walk is None else [stand_walk, camera_walk]
        # The walks are whole, the devices are not read yet: a crash in a
        # getter leaves both walks and says the scan is not complete.
        result.build = BuildScan(stand=stand_walk, camera=camera_walk, complete=False)
        _write_checkpoint(result, path)
        devices, notes = _read_devices(core, walks)
        result.build = BuildScan(
            stand=stand_walk, camera=camera_walk, devices=devices, notes=notes
        )
        _write(result, path)
    finally:
        close_core(core)


def _build_main(args: argparse.Namespace) -> int:
    """Open a core as the profile's stand will be opened, and build on it."""
    result = ChildResult()
    try:
        core = new_core(args.search_path)
    except CoreError as exc:
        # An adapter folder that is not a folder: the build's finding, written
        # as a scan with an error rather than a child that ends in a traceback.
        result.build = BuildScan(stand=ProbeResult(adapter=args.build, error=str(exc)))
        _write(result, args.result)
        return 0
    _build(core, result, args.result, args.build, args.camera, args.skip)
    return 0


def main(argv: list[str] | None = None) -> int:
    """List adapter devices, probe one adapter, or build from one adapter."""
    _no_error_dialogs()
    _use_utf8_stderr()
    parser = argparse.ArgumentParser(prog="smc.discovery._mm_child")
    parser.add_argument("--result", required=True, type=Path, help="result file")
    parser.add_argument("--adapters", default="", help="comma-separated names")
    parser.add_argument("--all", action="store_true", help="every installed adapter")
    parser.add_argument("--probe", default=None, help="one adapter to initialise")
    # The parent sends it as `--port=<port>`: a port starting with `-` is a
    # value there, where `--port -x` would be read as an option.
    parser.add_argument("--port", default=None, help="serial port for --probe")
    # The build options travel the same way, `--opt=value`, each one argument.
    parser.add_argument(
        "--build", default=None, help="the stand's adapter to build from"
    )
    parser.add_argument(
        "--camera", default=None, help="a camera's adapter, with --build"
    )
    parser.add_argument(
        "--skip", action="append", default=[], help="a device to leave out (repeat)"
    )
    parser.add_argument(
        "--search-path", action="append", default=[], help="an adapter folder (repeat)"
    )
    args = parser.parse_args(argv)
    if args.build is not None:
        return _build_main(args)

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
        _probe(CMMCorePlus(), result, args.result, args.probe, args.port)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
