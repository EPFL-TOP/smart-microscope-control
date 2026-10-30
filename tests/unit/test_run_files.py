"""The run folder's files (m3 §3.6): atomic writes, tokens, paths that stay inside."""

from __future__ import annotations

import errno
import os
import pickle
import re
from pathlib import Path, PureWindowsPath
from typing import Any

import pytest

from smc.run import files as files_mod
from smc.run.files import (
    AtomicWriteError,
    new_token,
    relative_to_run,
    resolve_in_run,
    retry_on_permission_error,
    temporary_path,
    write_atomically,
)


def _names(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir())


def _exactly(text: str) -> str:
    """A ``match`` that holds only for the whole of ``text`` (FM-45: exact phrases)."""
    return rf"\A{re.escape(text)}\Z"


def _refuse_replace(src: Any, dst: Any) -> None:
    raise PermissionError(errno.EACCES, "Permission denied")


# --- write_atomically --------------------------------------------------------


def test_write_atomically_replaces_the_file_and_leaves_no_temporary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"old")
    write_atomically(path, b"new content")
    assert path.read_bytes() == b"new content"
    assert _names(tmp_path) == ["f.bin"]


def test_write_atomically_creates_missing_parent_folders_by_default(
    tmp_path: Path,
) -> None:
    path = tmp_path / "scan" / "tok" / "f.bin"
    write_atomically(path, b"x")
    assert path.read_bytes() == b"x"
    assert _names(path.parent) == ["f.bin"]


def test_write_atomically_without_make_parents_refuses_a_missing_folder_and_creates_nothing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing" / "f.bin"
    with pytest.raises(AtomicWriteError) as raised:
        write_atomically(path, b"x", make_parents=False)
    assert raised.value.errno == errno.ENOENT
    assert raised.value.kept is None
    assert isinstance(raised.value.__cause__, FileNotFoundError)
    assert _names(tmp_path) == []


@pytest.mark.parametrize("below", [(), ("sub",)], ids=["parent", "grandparent"])
def test_a_parent_that_is_a_file_is_an_atomic_write_error_and_leaves_no_temporary_file(
    tmp_path: Path, below: tuple[str, ...]
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_bytes(b"i am a file")
    with pytest.raises(AtomicWriteError) as raised:
        write_atomically(blocker.joinpath(*below, "f.bin"), b"x")
    assert isinstance(raised.value, OSError)
    assert raised.value.kept is None
    assert blocker.read_bytes() == b"i am a file"
    assert _names(tmp_path) == ["blocker"]


def test_a_failed_rename_keeps_the_new_file_names_it_and_leaves_the_old_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # FM-38: the old file stays whole; the new content is on disk and named.
    path = tmp_path / "run.json"
    path.write_bytes(b"old")
    waits: list[float] = []
    monkeypatch.setattr(files_mod.os, "replace", _refuse_replace)
    monkeypatch.setattr(files_mod.time, "sleep", waits.append)
    with pytest.raises(AtomicWriteError) as raised:
        write_atomically(path, b"new")
    error = raised.value
    assert path.read_bytes() == b"old"
    assert error.kept is not None
    assert error.kept.parent == tmp_path
    assert re.fullmatch(r"\.run\.json\.[0-9a-f]{8}\.smc-tmp", error.kept.name)
    assert error.kept.read_bytes() == b"new"
    assert _names(tmp_path) == sorted(["run.json", error.kept.name])
    assert (error.errno, error.strerror) == (errno.EACCES, "Permission denied")
    assert isinstance(error.__cause__, PermissionError)
    assert waits == [0.05] * 4  # five tries, then the refusal is the error


def test_a_ctrl_c_while_writing_leaves_the_old_file_and_no_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The operator's stop passes through, and nothing is left behind.
    path = tmp_path / "f.bin"
    path.write_bytes(b"old")

    def interrupt(fd: int) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(files_mod.os, "fsync", interrupt)
    with pytest.raises(KeyboardInterrupt):
        write_atomically(path, b"new")
    assert path.read_bytes() == b"old"
    assert _names(tmp_path) == ["f.bin"]


def test_a_full_disk_is_an_atomic_write_error_with_nothing_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"old")

    def full(fd: int) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(files_mod.os, "fsync", full)
    with pytest.raises(AtomicWriteError) as raised:
        write_atomically(path, b"new")
    assert (raised.value.errno, raised.value.strerror) == (
        errno.ENOSPC,
        "No space left on device",
    )
    assert raised.value.kept is None
    assert path.read_bytes() == b"old"
    assert _names(tmp_path) == ["f.bin"]


def test_an_atomic_write_error_survives_pickle() -> None:
    # Like the errors of smc.hardware: one may cross a worker process.
    kept = Path("run") / ".run.json.0a1b2c3d.smc-tmp"
    error = AtomicWriteError(
        PermissionError(errno.EACCES, "Permission denied", "run.json"), kept
    )
    copy = pickle.loads(pickle.dumps(error))
    assert type(copy) is AtomicWriteError
    assert copy.kept == kept
    assert (copy.errno, copy.strerror, copy.filename) == (
        errno.EACCES,
        "Permission denied",
        "run.json",
    )
    assert isinstance(copy.cause, PermissionError)
    assert str(copy) == str(error)


def test_a_ctrl_c_during_the_rename_passes_through_and_leaves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The operator's stop at the last step is still a stop: it is not an OSError,
    # so it is no AtomicWriteError and no temporary file is kept (the new content
    # was not "refused", it was abandoned).
    path = tmp_path / "f.bin"
    path.write_bytes(b"old")

    def interrupt(src: Any, dst: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(files_mod.os, "replace", interrupt)
    with pytest.raises(KeyboardInterrupt):
        write_atomically(path, b"new")
    assert path.read_bytes() == b"old"
    assert _names(tmp_path) == ["f.bin"]


def test_the_temporary_file_is_opened_binary_and_exclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Windows would turn every \n of a TIFF into \r\n without O_BINARY (FM-24).
    flags: list[int] = []
    real_open = os.open

    def spy(path: Any, flag: int, mode: int = 0o777, **kwargs: Any) -> int:
        flags.append(flag)
        return real_open(path, flag, mode, **kwargs)

    # POSIX has no O_BINARY: give the module one to look for, as Windows does.
    monkeypatch.setattr(files_mod.os, "O_BINARY", 0x8000, raising=False)
    monkeypatch.setattr(files_mod.os, "open", spy)
    payload = b"a\nb\r\n\x1a"
    write_atomically(tmp_path / "f.bin", payload)
    (flag,) = flags
    assert flag & 0x8000
    assert flag & os.O_EXCL
    assert (tmp_path / "f.bin").read_bytes() == payload


def test_a_temporary_name_that_is_taken_is_refused_and_not_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # O_EXCL refuses a name that exists; it is someone else's file, not ours to
    # overwrite or to clean up.
    path = tmp_path / "f.bin"
    path.write_bytes(b"old")
    squatter = tmp_path / ".f.bin.deadbeef.smc-tmp"
    squatter.write_bytes(b"not ours")
    monkeypatch.setattr(files_mod.secrets, "token_hex", lambda nbytes: "deadbeef")
    with pytest.raises(AtomicWriteError) as raised:
        write_atomically(path, b"new")
    assert isinstance(raised.value.__cause__, FileExistsError)
    assert raised.value.kept is None
    assert squatter.read_bytes() == b"not ours"
    assert path.read_bytes() == b"old"


def test_temporary_path_is_the_sibling_write_atomically_writes_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A check of a path's length must be of the longest path a write makes (#228)."""
    monkeypatch.setattr(files_mod.secrets, "token_hex", lambda nbytes: "deadbeef")
    path = Path("run") / "A1_0001.tif"

    temporary = temporary_path(path)

    assert temporary == Path("run") / ".A1_0001.tif.deadbeef.smc-tmp"
    assert len(temporary.name) == len(path.name) + 18

    target = tmp_path / "A1_0001.tif"
    target.write_bytes(b"old")
    monkeypatch.setattr(files_mod.os, "replace", _refuse_replace)
    monkeypatch.setattr(files_mod.time, "sleep", lambda _s: None)
    with pytest.raises(AtomicWriteError) as raised:
        write_atomically(target, b"new")
    # The file the write kept is the one temporary_path names: the two cannot differ.
    assert raised.value.kept == temporary_path(target)


def test_a_short_write_is_continued_until_every_byte_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # os.write may take fewer bytes than offered (a full pipe, a network share):
    # a single call would silently truncate a TIFF.
    real_write = os.write

    def three_at_a_time(fd: int, data: Any) -> int:
        return real_write(fd, bytes(data[:3]))

    monkeypatch.setattr(files_mod.os, "write", three_at_a_time)
    write_atomically(tmp_path / "f.bin", b"0123456789")
    assert (tmp_path / "f.bin").read_bytes() == b"0123456789"


# --- retry_on_permission_error -----------------------------------------------


def test_retry_on_permission_error_tries_five_times_and_waits_four(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    waits: list[float] = []

    def refuse() -> None:
        calls.append(1)
        raise PermissionError(errno.EACCES, "Permission denied")

    monkeypatch.setattr(files_mod.time, "sleep", waits.append)
    with pytest.raises(PermissionError):
        retry_on_permission_error(refuse)
    assert (len(calls), waits) == (5, [0.05] * 4)


def test_retry_on_permission_error_stops_at_the_first_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    waits: list[float] = []

    def flaky() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError(errno.EACCES, "Permission denied")
        return "read"

    monkeypatch.setattr(files_mod.time, "sleep", waits.append)
    assert retry_on_permission_error(flaky) == "read"
    assert (len(calls), waits) == (3, [0.05] * 2)


def test_retry_on_permission_error_does_not_retry_another_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    waits: list[float] = []

    def missing() -> None:
        calls.append(1)
        raise FileNotFoundError(errno.ENOENT, "No such file")

    monkeypatch.setattr(files_mod.time, "sleep", waits.append)
    with pytest.raises(FileNotFoundError):
        retry_on_permission_error(missing)
    assert (len(calls), waits) == (1, [])


# --- new_token ---------------------------------------------------------------


def test_new_token_is_twelve_hex_digits_and_new_each_time() -> None:
    tokens = [new_token() for _ in range(50)]
    assert all(re.fullmatch(r"[0-9a-f]{12}", token) for token in tokens)
    assert len(set(tokens)) == 50


# --- relative_to_run ---------------------------------------------------------


def test_relative_to_run_is_posix_text_on_every_os(tmp_path: Path) -> None:
    # The document is read on another OS than it was written on: compare the
    # text, never a Path (FM-24).
    path = tmp_path / "scan" / "tok" / "A1_0000.tif"
    assert relative_to_run(tmp_path, path) == "scan/tok/A1_0000.tif"
    assert relative_to_run(tmp_path, tmp_path / "run.json") == "run.json"


def test_relative_to_run_refuses_a_path_outside_and_the_folder_itself(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    outside = tmp_path / "elsewhere" / "f.bin"
    with pytest.raises(
        ValueError, match=_exactly(f"{outside} is not inside the run folder {run}")
    ):
        relative_to_run(run, outside)
    with pytest.raises(
        ValueError,
        match=_exactly(f"{run} is the run folder itself, not a file in it"),
    ):
        relative_to_run(run, run)
    # A `..` that comes back out is not below the folder, though it starts there.
    sneaky = run / "a" / ".." / ".." / "elsewhere" / "f.bin"
    with pytest.raises(
        ValueError, match=_exactly(f"{sneaky} is not inside the run folder {run}")
    ):
        relative_to_run(run, sneaky)
    # One that stays inside is named by where it ends up.
    assert relative_to_run(run, run / "a" / ".." / "b.bin") == "b.bin"


def test_relative_to_run_is_posix_text_for_windows_paths() -> None:
    # A native Path on Linux or macOS never holds a backslash separator, so only a
    # Windows-flavoured pair shows that the text is built with as_posix (FM-24).
    run = PureWindowsPath("C:/runs/r1")
    path = PureWindowsPath("C:/runs/r1/scan/tok/A1_0000.tif")
    assert relative_to_run(run, path) == "scan/tok/A1_0000.tif"  # type: ignore[arg-type]


def test_relative_to_run_follows_a_symlinked_run_folder(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"this account may not create symlinks ({exc})")
    # Lexically neither is below the other; both resolve to the same folder.
    assert relative_to_run(link, real / "sub" / "f.bin") == "sub/f.bin"
    assert relative_to_run(real, link / "sub" / "f.bin") == "sub/f.bin"


# --- resolve_in_run ----------------------------------------------------------

_EMPTY = "is not a path in the run folder: it is empty, blank or holds a NUL character"
_ABSOLUTE = (
    "is absolute or names a drive; run-folder paths are relative to the run folder"
)
_DOTDOT = "has a '..' part; run-folder paths stay inside the run folder"
_ITSELF = "is the run folder itself, not a file in it"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", f"'' {_EMPTY}"),
        (" ", f"' ' {_EMPTY}"),
        ("a\x00b", f"'a\\x00b' {_EMPTY}"),
        (".", f"'.' {_ITSELF}"),
        ("a/..", f"'a/..' {_DOTDOT}"),
        ("../x", f"'../x' {_DOTDOT}"),
        ("a/../x", f"'a/../x' {_DOTDOT}"),
        ("a\\..\\x", f"'a\\\\..\\\\x' {_DOTDOT}"),
        ("/abs", f"'/abs' {_ABSOLUTE}"),
        ("C:\\x", f"'C:\\\\x' {_ABSOLUTE}"),
        ("C:/x", f"'C:/x' {_ABSOLUTE}"),
        ("C:x", f"'C:x' {_ABSOLUTE}"),
        ("\\\\server\\share\\x", f"'\\\\\\\\server\\\\share\\\\x' {_ABSOLUTE}"),
        ("\\x", f"'\\\\x' {_ABSOLUTE}"),
    ],
)
def test_resolve_in_run_refuses(tmp_path: Path, text: str, message: str) -> None:
    # Both path flavours are consulted, so a document written on Windows is
    # refused on macOS and the reverse: these run on every OS.
    with pytest.raises(ValueError, match=_exactly(message)):
        resolve_in_run(tmp_path, text)


def test_resolve_in_run_accepts_a_run_folder_that_is_itself_a_symlink(
    tmp_path: Path,
) -> None:
    # `~/runs` linked to another drive, or /var against /private/var on macOS:
    # the folder is resolved too, or every file below it would look "outside".
    real = tmp_path / "real"
    real.mkdir()
    (real / "f.tif").write_bytes(b"x")
    link = tmp_path / "link"
    try:
        link.symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"this account may not create symlinks ({exc})")
    assert resolve_in_run(link, "f.tif") == real.resolve() / "f.tif"


def test_resolve_in_run_turns_an_os_error_of_the_os_into_its_value_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Windows re-raises a few network errors from Path.resolve (a share that
    # drops mid-run); the caller is promised a ValueError that names the text.
    def drop(self: Path, strict: bool = False) -> Path:
        raise OSError(
            errno.ENETDOWN, "The specified network name is no longer available"
        )

    monkeypatch.setattr(Path, "resolve", drop)
    message = (
        f"'a/b.json' cannot be resolved in the run folder {tmp_path} "
        f"(The specified network name is no longer available)"
    )
    with pytest.raises(ValueError, match=_exactly(message)) as raised:
        resolve_in_run(tmp_path, "a/b.json")
    assert isinstance(raised.value.__cause__, OSError)


def test_resolve_in_run_refuses_a_symlink_that_leaves_the_folder(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_bytes(b"not for the run")
    try:
        (run / "link").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"this account may not create symlinks ({exc})")
    # The file is there: reading it would have succeeded.
    assert (run / "link" / "secret.txt").read_bytes() == b"not for the run"
    message = (
        f"'link/secret.txt' leads outside the run folder {run} (through a symlink)"
    )
    with pytest.raises(ValueError, match=_exactly(message)):
        resolve_in_run(run, "link/secret.txt")


@pytest.mark.parametrize(
    "text",
    ["scan/tok/A1_0000.tif", "a//b", "a/./b", "cells/0123456789ab.json", "run.json"],
)
def test_resolve_in_run_accepts_a_file_below_the_folder_and_round_trips_with_relative_to_run(
    tmp_path: Path, text: str
) -> None:
    resolved = resolve_in_run(tmp_path, text)
    assert resolved.is_absolute()
    assert resolved.is_relative_to(tmp_path.resolve())
    # The text that names the file is the one that was asked for, normalised.
    normalised = "/".join(part for part in text.split("/") if part not in ("", "."))
    assert relative_to_run(tmp_path, resolved) == normalised
    assert os.path.normcase(resolved) == os.path.normcase(
        (tmp_path / normalised).resolve()
    )
