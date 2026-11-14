"""A long move cut into segments on the demo stand (m2 §11.12, #221).

The demo is the other kind of stand from the Ti2-E: its ``DXYStage`` call
returns at once and reads busy until it arrives, and its ``DStage`` (Z) lands
at once (measured 2026-11-14, m2 §11.12.1). Each segment must still be waited
for and land within ``SEGMENT_END_TOLERANCE_UM`` before the next is sent.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from smc.hardware import Microscope
from smc.hardware.capabilities import XYStage, ZStage
from smc.hardware.profile import Profile, SafetySection
from smc.hardware.roles import Role

pytestmark = pytest.mark.demo

LOGGER = "smc.hardware.microscope"


def _segmented_demo() -> Profile:
    """The demo profile, declaring both stages refused and cutting at 1000 / 100 µm."""
    demo = Profile.demo()
    safety = SafetySection.model_validate(
        {
            **demo.safety.model_dump(),
            "stop_refused_by": [Role.xy_stage.value, Role.focus.value],
            "max_segment_um": 1000.0,
            "max_z_segment_um": 100.0,
        }
    )
    return demo.model_copy(update={"safety": safety})


def test_a_segmented_move_on_the_demo_stage_lands_at_its_target(
    demo_core: Any, caplog: pytest.LogCaptureFixture
) -> None:
    microscope = Microscope.from_core(demo_core, _segmented_demo())
    try:
        xy = microscope.require(XYStage)
        z = microscope.require(ZStage)
        start = xy.move_to_um(0.0, 0.0)
        z.move_to_um(0.0)
        with caplog.at_level(logging.INFO, logger=LOGGER):
            landed = xy.move_to_um(start.x_um + 3000.0, start.y_um)
            z_landed = z.move_to_um(250.0)
    finally:
        microscope.close()
    assert landed.x_um == pytest.approx(3000.0, abs=0.05)
    assert landed.y_um == pytest.approx(0.0, abs=0.05)
    assert z_landed == pytest.approx(250.0, abs=0.05)
    segments = [m for m in caplog.messages if ", segment " in m]
    assert [m.split(", segment ")[1].split(" to ")[0] for m in segments] == [
        "1 of 3",
        "2 of 3",
        "3 of 3",
        "1 of 3",
        "2 of 3",
        "3 of 3",
    ]
