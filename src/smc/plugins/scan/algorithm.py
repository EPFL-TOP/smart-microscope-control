"""The scan's pure part: the fields left, the frames' paths, the disk, the focus state (m3 §8).

Nothing here reads a stand. What it decides is what a long, unattended run
must not get wrong before it starts:

* **A frame's path cannot leave its folder.** The well and the field's name
  come from a run document that may have been copied or edited, and each
  becomes a folder or a file name: only letters, digits, ``_`` and ``-`` are
  accepted, so no part is ``..``, a drive, a separator of either OS or a name
  that Windows trims (a trailing space or dot, FM-24).
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
    "bytes_needed",
    "encode_frame",
    "focus_state",
    "frame_path",
    "itemsize_of",
    "size_differs",
    "wells_to_scan",
]

#: What a well, a field's name or a token may be to become part of a path.
FRAME_PART = re.compile(r"[A-Za-z0-9_-]+")
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
    return f"scan/{token}/{well}/{name}.tif"


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
