"""The camera-calibration slot: how the camera's pixels map to the stage (m3 §3.3).

Camera calibration measures the matrix of ``smc.plugins.geometry`` once per
objective and camera setting; the scan plan and the cells stage read it when it
exists and fall back to an assumed upright camera when it does not.
"""

from __future__ import annotations

from pydantic import Field

from smc.plugins.api import PluginResult
from smc.plugins.geometry import Matrix2

__all__ = ["CameraCalibration"]


class CameraCalibration(PluginResult):
    """The measured camera matrix, with how well it was measured.

    ``image_shape_px`` is ``(height, width)`` at the time of the measurement: a
    binning change makes the calibration inapplicable (§7). ``pixel_size_um``
    is ``sqrt(|det M|)``; ``rotation_deg`` and ``mirrored`` follow the
    convention of ``smc.plugins.geometry``. ``residual_um`` and ``response``
    are the correlation's (§4); ``return_error_um`` is the distance, in µm,
    between the start and the content after the return, and ``None`` when
    the measurement was reused (``reused_from`` names the run it came from)
    or when the frame after the return did not correlate: unknown, not zero.
    """

    objective: str | None
    image_shape_px: tuple[int, int]
    matrix_um_per_px: Matrix2
    pixel_size_um: float = Field(gt=0.0)
    rotation_deg: float = Field(ge=-180.0, le=180.0)
    mirrored: bool
    residual_um: float = Field(ge=0.0)
    response: float = Field(ge=0.0, le=1.0)
    return_error_um: float | None = Field(ge=0.0)
    step_um: float = Field(gt=0.0)
    reused_from: str | None = None
