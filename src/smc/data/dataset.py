"""A scan's run folder as a self-contained dataset for the lab's share (algorithms design §2).

A collaborator develops a selection algorithm on a laptop with no stand, and the
scan they work on travels as a folder: the run document, the frames and the
cells file at the same relative paths, and a manifest that says what each file
is, so that a copy that lost a byte on the way is found by ``check`` and not by
a wrong result a week later. A dataset *is* a run folder: ``load_scan`` reads it
as it reads the source.

Nothing here imports the hardware layer; ``tests/test_kit_imports.py`` keeps it so.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from pydantic import AwareDatetime, BaseModel, ConfigDict

from smc.run.files import retry_on_permission_error

__all__ = ["DatasetFile", "DatasetManifest", "DatasetWriteError"]

#: Read and written a megabyte at a time: a frame is 4 MB or so, and a scan is
#: hundreds of them. A module attribute read at the call, so that a test can
#: make it seven bytes and cut every file into many chunks.
_CHUNK: int = 1 << 20

#: Windows refuses a path of 260 characters or more unless long paths are
#: enabled, and enabling them is the lab PC's choice, not an assumption of ours.
#: Module attributes read at the call, so a test can say it is on Windows with a
#: shorter limit.
MAX_PATH: int = 260
ON_WINDOWS: bool = os.name == "nt"

#: The schema of ``manifest.json`` this smc writes and reads.
MANIFEST_SCHEMA = 1


class DatasetFile(BaseModel):
    """One file of a dataset: its size and the SHA-256 of its bytes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bytes: int
    sha256: str


class DatasetManifest(BaseModel):
    """What a dataset holds and where it came from; ``manifest.json`` is its dump.

    ``files`` holds every file of the folder but ``manifest.json``, by its
    POSIX path relative to the folder, in the order they were written (the
    frames in the scan's order, the cells file, ``run.json``, ``README.md``).
    ``source`` is the source run document's file name, never its path (FM-50),
    and ``scrubbed`` the JSON pointers of the values of ``run.json`` that the
    export rewrote (``smc.data.scrub``). ``cells_absent`` says why there are no
    cells when there are none. The camera (``pixel_size_um``,
    ``image_shape_px``) is the scan's, not the plan's (FM-124).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = MANIFEST_SCHEMA
    smc_version: str
    exported_utc: AwareDatetime
    source: str
    profile: str
    objective: str | None
    pixel_size_um: float
    image_shape_px: tuple[int, int]
    wells: tuple[str, ...]
    frames: int
    cells: bool
    cells_absent: str | None
    files: dict[str, DatasetFile]
    scrubbed: tuple[str, ...]


class DatasetWriteError(Exception):
    """An export stopped part-way on an I/O error; the message says how far and what to do.

    Not an ``OSError`` and not a ``DatasetError``: the command maps a refusal
    (nothing was written) to exit 2 and this (the folder holds part of the
    dataset) to exit 1, and the cause is chained.
    """


def _copy_and_hash(source: Path, destination: Path) -> tuple[int, str]:
    """Copy ``source`` to a new file ``destination``; return the bytes written and the SHA-256 of the **source's** bytes.

    The hash is taken of what is read, not of what is written back: a bad write
    (a full disk that lied, a flaky share) is then found by ``check``, where
    hashing the copy afterwards would record the damage as the truth (and read
    the file once more). ``destination`` is opened ``"xb"``: a file that exists
    is never opened for writing. The source is opened first, so that a source
    that cannot be read leaves no empty destination behind.
    """
    digest = hashlib.sha256()
    size = 0
    with retry_on_permission_error(lambda: source.open("rb")) as reader:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as writer:
            while chunk := reader.read(_CHUNK):
                digest.update(chunk)
                writer.write(chunk)
                size += len(chunk)
    return size, digest.hexdigest()


def _sha256_of(path: Path) -> str:
    """The SHA-256 of a file, read once, a chunk at a time.

    ``hashlib.file_digest`` is 3.11 and later, and the floor is 3.10.
    """
    digest = hashlib.sha256()
    with retry_on_permission_error(lambda: path.open("rb")) as reader:
        while chunk := reader.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()
