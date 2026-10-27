"""The scan slot: what was imaged, where, and what could not be (m3 §3.4).

The frames live in files under the run folder and the result names them; the
cells stage reads them back with ``load_frame``, which will not follow a path
out of the folder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy.typing as npt
import tifffile
from pydantic import Field

from smc.plugins.api import PluginResult
from smc.plugins.geometry import Matrix2
from smc.plugins.slots.base import SlotValue
from smc.run.files import resolve_in_run, retry_on_permission_error

__all__ = ["FailedField", "FocusState", "ScanFrame", "ScanResult", "load_frame"]

#: ``locked``: the autofocus was used and locked when the frame was taken;
#: ``unlocked``: used, not locked; ``off``: the stand has one and the scan did
#: not use it; ``none``: the stand has none.
FocusState = Literal["locked", "unlocked", "off", "none"]


class ScanFrame(SlotValue):
    """One frame taken: the field it belongs to, its file and where the stage was.

    ``x_um``, ``y_um`` are the stage's **readback** after the move, not the
    target (the demo rounds a command, M1 §8): the point imaged at the frame
    centre. ``z_um`` is the focus stage's reading, ``None`` without one;
    ``pfs_offset`` the autofocus offset in its native unit, ``None`` when the
    autofocus was not used. ``t_s`` counts from the start of the attempt that
    took the frame, and ``file`` is relative to the run folder, POSIX.
    """

    well: str
    index: int = Field(ge=0)
    name: str
    file: str
    x_um: float
    y_um: float
    z_um: float | None
    focus: FocusState
    pfs_offset: float | None
    t_s: float = Field(ge=0.0)


class FailedField(SlotValue):
    """A field the scan could not image; ``x_um``, ``y_um`` are the target."""

    well: str
    index: int = Field(ge=0)
    name: str
    x_um: float
    y_um: float
    error: str


class ScanResult(PluginResult):
    """A finished scan: its frames, the fields that failed, and the camera it used.

    ``matrix_um_per_px`` and ``pixel_size_um`` are the scan plan's, copied so
    that the cells stage needs only ``scan``. ``folder`` is the frames' folder,
    relative to the run folder.
    """

    folder: str
    objective: str | None
    pixel_size_um: float = Field(gt=0.0)
    matrix_um_per_px: Matrix2
    image_shape_px: tuple[int, int]
    exposure_ms: float = Field(ge=0.0)
    frames: tuple[ScanFrame, ...]
    failed: tuple[FailedField, ...]

    def frame_index(self) -> dict[str, list[int]]:
        """Each well mapped to the positions of its frames in ``frames``.

        Wells come in the order of their first frame and the positions ascend;
        a well that has no frame (every field failed) is absent, not empty.
        """
        index: dict[str, list[int]] = {}
        for position, frame in enumerate(self.frames):
            index.setdefault(frame.well, []).append(position)
        return index


def load_frame(run_dir: Path, frame: ScanFrame) -> npt.NDArray[Any]:
    """Read a frame back from the run folder, with the dtype and shape it was written with.

    The read is retried on ``PermissionError``: the scan wrote the file a
    moment ago, and antivirus may hold it (FM-24).

    Raises:
        ValueError: ``frame.file`` leads outside the run folder (a run
            document copied from elsewhere must not make a stage read any
            file on the PC).
        FileNotFoundError: The file is not in the run folder; the message
            names it. A corrupt file raises ``tifffile``'s own error.
    """
    path = resolve_in_run(run_dir, frame.file)
    if not path.is_file():
        raise FileNotFoundError(
            f"frame file {frame.file} is missing from the run folder {run_dir}"
        )
    return retry_on_permission_error(lambda: tifffile.imread(path))
