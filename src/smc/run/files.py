"""The files a stage writes next to the run document (m3 §3.6).

The run document names what a stage produced (a cells file, a TIFF, a report)
by a path relative to the run folder, and a run folder is copied between PCs
and attached to issues. So this module holds the three things every stage
needs and none should write twice:

* **a write that is never half-done** (``write_atomically``, FM-38): the old
  file or the whole new one, never a prefix of it, even after a Ctrl-C or a
  full disk;
* **a name that is new for every run** (``new_token``), so a file that a
  ``done`` result names is never rewritten by the next run;
* **a path that cannot leave the folder** (``relative_to_run``,
  ``resolve_in_run``): a document written on Windows must be refused on macOS
  and the reverse, and a document copied from elsewhere must not make a stage
  read or write outside its folder (FM-24).

Nothing here imports ``smc.plugins`` or ``smc.hardware``: the run document and
the plugins depend on this module, not the reverse.
"""

from __future__ import annotations

import os
import secrets
import time
import uuid
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Any, TypeVar

__all__ = [
    "AtomicWriteError",
    "new_token",
    "relative_to_run",
    "resolve_in_run",
    "retry_on_permission_error",
    "write_atomically",
]

T = TypeVar("T")

#: Windows refuses to read or replace a file that antivirus or a search indexer
#: holds open for a moment (``PermissionError``), and the file was written a
#: moment ago (FM-24): a few short waits, not a lost result. As
#: ``_mm_child._write``.
_LOCK_TRIES = 5
_LOCK_WAIT_S = 0.05


class AtomicWriteError(OSError):
    """``write_atomically`` could not write the file; it is as it was.

    ``kept`` is the temporary file that holds the whole new content when only
    the final rename failed (so the caller can tell the operator where it is),
    ``None`` when nothing was kept. The cause is the ``OSError`` this was
    raised from; ``errno``, ``strerror`` and ``filename`` are the cause's.
    """

    def __init__(self, cause: OSError, kept: Path | None = None) -> None:
        super().__init__(cause.errno, cause.strerror, cause.filename)
        self.cause = cause
        self.kept = kept

    def __reduce__(self) -> tuple[Any, ...]:
        # Rebuilt from the arguments, not from OSError's own ``args``, so that
        # ``kept`` survives a worker process (pickle) as the errors of
        # ``smc.hardware`` do.
        return (type(self), (self.cause, self.kept))


def write_atomically(path: Path, data: bytes, *, make_parents: bool = True) -> None:
    """Replace ``path`` by ``data`` so that it is either the old file or the new one (FM-38).

    A temporary file in the same folder, created with ``O_EXCL`` and the mode
    a plain write gives (``mkstemp``'s 0600 would survive the rename), then
    ``os.replace``, all in one ``try`` so that a Ctrl-C between two blocks does
    not leave a file nothing names. On a failure the temporary file is
    removed, except when only the rename failed: it then holds the whole new
    content, so it is kept and the error names it. The rename is retried on
    ``PermissionError`` (FM-24).

    Args:
        path: The file to write.
        data: Its whole new content.
        make_parents: Create the missing folders above ``path`` first. The run
            document passes ``False``: a mistyped ``--run-doc`` folder is
            refused, not created. The folders are made inside the same
            ``try``, so a parent that is a file is an ``AtomicWriteError``
            with no temporary file, like any other failure.

    Raises:
        AtomicWriteError: The file could not be written; it is as it was.
            ``kept`` is set when only the rename failed.
    """
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}.smc-tmp")
    renaming = False
    try:
        if make_parents:
            path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o666,
        )
        try:
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view) :]
            os.fsync(fd)
        finally:
            os.close(fd)
        renaming = True
        retry_on_permission_error(lambda: os.replace(temporary, path))
    except BaseException as exc:
        if renaming and isinstance(exc, OSError):
            raise AtomicWriteError(exc, kept=temporary) from exc
        with suppress(OSError):
            temporary.unlink()
        if isinstance(exc, OSError):
            raise AtomicWriteError(exc) from exc
        raise


def retry_on_permission_error(action: Callable[[], T]) -> T:
    """``action()``, tried five times while it raises ``PermissionError``.

    A file that antivirus or an indexer holds open for a moment is refused, on
    Windows, to a read as well as to a rename (FM-24). The waits are 0.05 s;
    the last refusal is raised, and any other error at once.
    """
    attempt = 1
    while True:
        try:
            return action()
        except PermissionError:
            if attempt == _LOCK_TRIES:
                raise
            attempt += 1
            time.sleep(_LOCK_WAIT_S)


def new_token() -> str:
    """Twelve hex digits, new for every run of a stage.

    A file that a ``done`` result names is never rewritten (§2.3): the next run
    writes under another token, so a reader holding the old result still finds
    the old file.
    """
    return uuid.uuid4().hex[:12]


def relative_to_run(run_dir: Path, path: Path) -> str:
    """``path`` relative to ``run_dir``, as POSIX text on every OS (FM-24).

    Results name their files by this text, and the document is read on
    another OS than it was written on, so it never holds a backslash that
    came from a ``Path``.

    Raises:
        ValueError: ``path`` is not inside ``run_dir``, or is ``run_dir``
            itself.
    """
    relative = _below(path, run_dir)
    if relative is None:
        # macOS: a tmp folder is /var/... and resolves to /private/var/...; a
        # symlinked run folder is the same thing. Compare what both point at.
        relative = _below(path.resolve(), run_dir.resolve())
    if relative is None:
        raise ValueError(f"{path} is not inside the run folder {run_dir}")
    if not relative.parts:
        raise ValueError(f"{path} is the run folder itself, not a file in it")
    return relative.as_posix()


def _below(path: Path, folder: Path) -> PurePath | None:
    """``path`` relative to ``folder`` when it is lexically below it, else ``None``.

    A ``..`` in the result is not below anything: ``folder/a/../../x`` starts
    with ``folder`` and is outside it, so it is left to the resolved check.
    """
    try:
        relative = path.relative_to(folder)
    except ValueError:
        return None
    return None if ".." in relative.parts else relative


def resolve_in_run(run_dir: Path, relative: str) -> Path:
    """The absolute path of ``relative`` inside ``run_dir``; never one outside it.

    A run document may have been copied from elsewhere or written on another
    OS, and its paths are text: an absolute one, a drive, a ``..`` or a
    symlink must not make a stage read or write outside its folder. Both path
    flavours are consulted whatever the OS, so a Windows drive is refused on
    macOS too, and a ``..`` written with backslashes on Linux.

    Raises:
        ValueError: The text is empty, blank or holds a NUL; is absolute or
            has a drive in either flavour; has a ``..`` part; is
            ``run_dir`` itself once resolved; or leads outside it through a
            symlink. Every message starts with the text, quoted.
    """
    if not relative.strip() or "\x00" in relative:
        raise ValueError(
            f"{relative!r} is not a path in the run folder: it is empty, blank "
            f"or holds a NUL character"
        )
    posix = PurePosixPath(relative)
    windows = PureWindowsPath(relative)
    if posix.anchor or windows.anchor or windows.drive:
        raise ValueError(
            f"{relative!r} is absolute or names a drive; run-folder paths are "
            f"relative to the run folder"
        )
    if ".." in posix.parts or ".." in windows.parts:
        raise ValueError(
            f"{relative!r} has a '..' part; run-folder paths stay inside the run folder"
        )
    root = run_dir.resolve()
    resolved = (root / relative).resolve()
    if resolved == root:
        raise ValueError(f"{relative!r} is the run folder itself, not a file in it")
    if not resolved.is_relative_to(root):
        raise ValueError(
            f"{relative!r} leads outside the run folder {run_dir} (through a symlink)"
        )
    return resolved
