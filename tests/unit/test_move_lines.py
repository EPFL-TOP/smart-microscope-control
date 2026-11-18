"""A move's INFO line writes its whole target the way its segments are written (#545, F2).

Ti2-E session 5 read ``move_by (100.0, 0.0) µm to (200.60000000000002,
0.6000000000000001) µm`` at the console and on the page: the stage's readback
plus a jog is a float with its rounding noise, and the line printed all of it
beside a segment rounded to hundredths. Only the words are rounded: the stage is
sent the full value (``test_segmented_move.py`` pins that for the segments).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import cast

import pytest
from pymmcore_plus import CMMCorePlus

from smc.hardware import Microscope
from smc.hardware.capabilities import XYStage, ZStage
from smc.hardware.profile import Profile, SafetySection
from smc.hardware.roles import Role
from smc.testing import FakeCore

LOGGER = "smc.hardware.microscope"
#: A stage readback with the noise of a sum, as the page's drive made it.
NOISY = 0.2000000000007276


@contextmanager
def _stand(*, cut_um: float | None = None) -> Iterator[tuple[FakeCore, Microscope]]:
    """A fake demo stand, its XY and Z cut into segments of ``cut_um`` when it is set."""
    core = FakeCore.demo_like()
    profile = Profile.demo()
    if cut_um is not None:
        core.stop_refused = {"XY", "Z"}
        safety = SafetySection.model_validate(
            {
                **profile.safety.model_dump(),
                "stop_refused_by": [Role.xy_stage.value, Role.focus.value],
                "max_segment_um": cut_um,
                "max_z_segment_um": cut_um,
            }
        )
        profile = profile.model_copy(update={"safety": safety})
    microscope = Microscope.from_core(cast("CMMCorePlus", core), profile)
    core.log.clear()
    try:
        yield core, microscope
    finally:
        microscope.close()


def _lines(caplog: pytest.LogCaptureFixture, prefix: str) -> list[str]:
    return [m for m in caplog.messages if m.startswith(prefix)]


def test_a_move_to_logs_its_target_rounded_to_two_decimals(
    caplog: pytest.LogCaptureFixture,
) -> None:
    assert NOISY != 0.2  # the raw float is not the rounded one: the test can fail
    with _stand() as (core, microscope):
        with caplog.at_level(logging.INFO, logger=LOGGER):
            microscope.require(XYStage).move_to_um(NOISY, -0.1)
        # The words are rounded, the command is not.
        assert core.log == [f"setXYPosition('XY', {NOISY!r}, -0.1)"]
    assert _lines(caplog, "xy_stage: move_to") == ["xy_stage: move_to (0.2, -0.1) µm"]


def test_a_move_by_logs_its_offset_and_the_whole_target_rounded(
    caplog: pytest.LogCaptureFixture,
) -> None:
    offset = 0.1 + 0.2
    assert offset != 0.3
    assert NOISY + offset != 0.5
    with _stand() as (core, microscope):
        stage = microscope.require(XYStage)
        stage.move_to_um(NOISY, -0.1)
        core.log.clear()
        with caplog.at_level(logging.INFO, logger=LOGGER):
            stage.move_by_um(offset, 0.0, force=True)
        sent = core.log[0]
        assert sent.startswith(f"setXYPosition('XY', {NOISY + offset!r}")
    assert _lines(caplog, "xy_stage: move_by") == [
        "xy_stage: move_by (0.3, 0.0) µm to (0.5, -0.1) µm"
    ]


def test_a_cut_move_by_logs_a_rounded_whole_target_beside_its_segments(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The jog of session 5: -12000 µm from X 12000.2, cut at 4000 µm into three segments."""
    start_x = 12000.0 + NOISY
    with _stand(cut_um=4000.0) as (core, microscope):
        stage = microscope.require(XYStage)
        core.setXYPosition("XY", start_x, -0.1)
        with caplog.at_level(logging.INFO, logger=LOGGER):
            stage.move_by_um(-12000.0, 0.0, force=True)
    whole = "xy_stage: move_by (-12000.0, 0.0) µm to (0.2, -0.1) µm"
    assert _lines(caplog, "xy_stage: move_by") == [
        f"{whole}, segment 1 of 3 to (8000.2, -0.1) µm",
        f"{whole}, segment 2 of 3 to (4000.2, -0.1) µm",
        f"{whole}, segment 3 of 3 to (0.2, -0.1) µm",
    ]


def test_a_move_to_that_is_cut_logs_a_rounded_whole_target_beside_its_segments(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The page's drive of session 5: ``move_to (29575.000000000004, -10049.999999999996)``."""
    x_um, y_um = 29575.000000000004 / 2, -10049.999999999996 / 2
    assert (x_um, y_um) != (14787.5, -5025.0)
    with (
        _stand(cut_um=5000.0) as (_, microscope),
        caplog.at_level(logging.INFO, logger=LOGGER),
    ):
        microscope.require(XYStage).move_to_um(x_um, y_um)
    whole = "xy_stage: move_to (14787.5, -5025.0) µm"
    lines = _lines(caplog, "xy_stage: move_to")
    assert len(lines) == 4
    assert all(line.startswith(f"{whole}, segment ") for line in lines)
    assert lines[-1].endswith("segment 4 of 4 to (14787.5, -5025.0) µm")


def test_a_z_move_logs_its_target_rounded_to_two_decimals(
    caplog: pytest.LogCaptureFixture,
) -> None:
    noisy = 485.0 + 1e-11
    assert noisy != 485.0
    with _stand() as (core, microscope):
        stage = microscope.require(ZStage)
        with caplog.at_level(logging.INFO, logger=LOGGER):
            stage.move_to_um(noisy)
        assert core.log[0] == f"setPosition('Z', {noisy!r})"
        stage.move_by_um(0.1 + 0.2, force=True)
    assert _lines(caplog, "z: move_") == [
        "z: move_to 485.0 µm",
        "z: move_by 0.3 µm to 485.3 µm",
    ]
