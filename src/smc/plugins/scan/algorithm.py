"""The scan's pure part: the fields left, the frames' paths, the disk, the focus state (m3 §8).

Nothing here reads a stand. What it decides is what a long, unattended run
must not get wrong before it starts:

* **A frame's path cannot leave its folder.** The well and the field's name
  come from a run document that may have been copied or edited, and each
  becomes a folder or a file name: only letters, digits, ``_`` and ``-`` are
  accepted, so no part is ``..``, a drive, a separator of either OS or a name
  that Windows trims (a trailing space or dot, FM-24) or keeps for a device
  (``NUL``, ``COM1``). Two fields whose names differ only in case would write
  one file on NTFS and APFS, so ``collision_key`` compares without case.
* **A move that arrived is a move whose readback is near its target.** The
  distance is measured on both axes against a fraction of the smaller side of
  the field of view, and a readback that is not a number is off target
  (FM-104, FM-73).
* **The disk is checked for the whole scan.** A full disk at well 60 of 96
  loses the night; a tenth more than the frames covers the TIFF headers and
  what else writes to the disk meanwhile.
* **A pixel size is compared relative to the one it should be.** The Ti2-E's
  magnifier lever changes it by half (FM-103); a calibration measures it
  within a few per cent of the profile's.
"""

from __future__ import annotations

import io
import math
import re
from collections.abc import Collection, Sequence
from typing import Any

import numpy.typing as npt
import tifffile

from smc.plugins.slots import FocusState, PlannedField, ScanFrame

__all__ = [
    "CALIBRATION_TOLERANCE",
    "FRAME_PART",
    "PROFILE_TOLERANCE",
    "RESERVED_NAMES",
    "bytes_needed",
    "collision_key",
    "encode_frame",
    "focus_state",
    "frame_path",
    "itemsize_of",
    "landed_off_target",
    "landing_error_um",
    "landing_tolerance_um",
    "size_differs",
    "wells_to_scan",
]

#: What a well, a field's name or a token may be to become part of a path.
FRAME_PART = re.compile(r"[A-Za-z0-9_-]+")
#: The names Windows keeps for devices, in upper case: a file named ``NUL.tif``
#: is the null device there, so no part of a frame's path may be one (FM-24).
#: The list is its own: ``PurePath.is_reserved`` is deprecated in Python 3.13.
RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{n}" for n in range(10)}
    | {f"LPT{n}" for n in range(10)}
)
#: How far the camera's pixel size may be from a plan made from the profile,
#: and from the size read at the start of the scan: the lever is 50 %, a
#: rounding in a profile far below 1 %.
PROFILE_TOLERANCE = 0.01
#: How far it may be from a plan made from a calibration: a measured size is
#: a few per cent from the profile's, which is the camera's.
CALIBRATION_TOLERANCE = 0.10
#: The disk needed is this much more than the frames' pixels.
_DISK_MARGIN = 1.1


def wells_to_scan(
    wells: Sequence[str],
    fields: Sequence[PlannedField],
    done: Collection[str],
) -> list[tuple[str, tuple[PlannedField, ...]]]:
    """Each well still to scan with its fields, in the order ``wells`` gives.

    ``wells`` is the plan's visiting order and ``fields`` the plan's fields;
    each well keeps its fields in plan order. A well in ``done`` is skipped. A
    well with no field is kept with an empty group: it is marked done like any
    other, so the checkpoint says it was considered.
    """
    by_well: dict[str, list[PlannedField]] = {well: [] for well in wells}
    for field in fields:
        if field.well in by_well:
            by_well[field.well].append(field)
    return [
        (well, tuple(by_well[well]))
        for well in dict.fromkeys(wells)
        if well not in done
    ]


def frame_path(token: str, well: str, name: str) -> str:
    """``scan/<token>/<well>/<name>.tif``: relative to the run folder, POSIX (FM-24).

    Raises:
        ValueError: A part is not letters, digits, ``_`` and ``-`` only; the
            message names it.
    """
    for what, part in (("token", token), ("well", well), ("field name", name)):
        if FRAME_PART.fullmatch(part) is None:
            raise ValueError(
                f"{part!r} cannot be a frame's {what} in a file path: only "
                f"letters, digits, '_' and '-' are"
            )
        if part.upper() in RESERVED_NAMES:
            raise ValueError(
                f"{part!r} is a Windows device name and cannot be a frame's "
                f"{what} in a file path"
            )
    return f"scan/{token}/{well}/{name}.tif"


def collision_key(well: str, name: str) -> tuple[str, str]:
    """What two fields share when they would write one file on a volume that folds case.

    NTFS and the default APFS do not tell ``A1_0001`` from ``a1_0001``: two
    fields whose keys are equal write one file, and the result would name two
    frames for one TIFF on disk (FM-24).
    """
    return (well.casefold(), name.casefold())


def landing_error_um(
    target: tuple[float, float], readback: tuple[float, float]
) -> float:
    """The distance of both axes between where a move was sent and where the stage says it is.

    One axis is not enough: a stage short by 3.5 µm on x and 3.5 µm on y is
    4.95 µm off, past a 4.8 µm tolerance that neither axis reaches.
    """
    return math.hypot(readback[0] - target[0], readback[1] - target[1])


def landing_tolerance_um(fov_um: tuple[float, float], fraction: float) -> float:
    """``fraction`` of the smaller side of the field of view, in µm.

    The smaller side, because the frame is not square (FM-73): a short landing
    along it is the one that leaves a gap in the coverage first.
    """
    return fraction * min(fov_um)


def landed_off_target(error_um: float, tolerance_um: float) -> bool:
    """Whether a landing is farther than the tolerance; a landing that is not a number is.

    Written as ``not error <= tolerance`` and not as ``error > tolerance``: a
    readback that is ``nan`` compares False both ways, and only this form
    fails it (FM-104).
    """
    return not error_um <= tolerance_um


def itemsize_of(bit_depth: int) -> int:
    """Bytes per pixel of a frame of ``bit_depth`` bits: 1, 2 or 4."""
    if bit_depth <= 8:
        return 1
    if bit_depth <= 16:
        return 2
    return 4


def bytes_needed(n_fields: int, shape_px: tuple[int, int], itemsize: int) -> int:
    """The disk the frames of ``n_fields`` fields need, with a tenth to spare."""
    height, width = shape_px
    return math.ceil(_DISK_MARGIN * n_fields * height * width * itemsize)


def size_differs(now_um: float, reference_um: float, tolerance: float) -> bool:
    """Whether ``now_um`` is more than ``tolerance`` (relative) from ``reference_um``."""
    return abs(now_um / reference_um - 1.0) > tolerance


def focus_state(has_autofocus: bool, used: bool, locked: bool) -> FocusState:
    """What the autofocus was for a frame: ``none``, ``off``, ``locked`` or ``unlocked``.

    ``off`` is an autofocus the scan did not use, whatever it was doing: a
    lock the scan did not ask for is not one it can vouch for.
    """
    if not has_autofocus:
        return "none"
    if not used:
        return "off"
    return "locked" if locked else "unlocked"


def encode_frame(frame: npt.NDArray[Any], meta: ScanFrame) -> bytes:
    """The frame as a TIFF in its own dtype, with ``meta`` as its JSON metadata, as ``smc snap`` writes it.

    Encoded in memory so that the file is written whole by
    ``write_atomically`` (FM-38): ``tifffile`` opens its target with ``"wb"``.
    """
    buffer = io.BytesIO()
    tifffile.imwrite(buffer, frame, metadata=meta.model_dump(mode="json"))
    return buffer.getvalue()
