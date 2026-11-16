"""The Plate screen's rows on a stand whose soft limits cut the plate's corners, over the real ``goto-well`` (m5 §19, #512).

The Ti2-E's Greiner 24-well in its holder has A6 25 µm and D1 500 µm outside the stage's
XY soft limits: the rows used to offer them, and *Go to* was refused for both (found in the
hardware-session prep of Ti2-E session 5). The profile is the demo's with the Ti2-E's
``[safety] xy_soft_limits_um`` and ``[holder] plate_centre_xy_um``, so the layer's own guard
is the one that refuses. Every position is typed out from ``holder_estimate("greiner-24-well",
(400, -325), "not-turned")`` as measured on 2026-11-16 (Python 3.13.0, useq-schema 0.9.2),
never read back from the code under test (FM-43).
"""

from __future__ import annotations

from collections.abc import Iterator
from importlib.metadata import EntryPoint
from pathlib import Path

import pytest

from smc.hardware import Microscope
from smc.hardware.capabilities import XYStage
from smc.hardware.profile import HolderSection, Profile
from smc.plugins.registry import GROUP
from smc.run import RunDocument, StageStatus
from smc.ui.controller import PlatePage

pytestmark = [
    pytest.mark.demo,
    # A failure a handler swallowed fails the test that did not expect it (#277).
    pytest.mark.usefixtures("handler_failures"),
]

LIMITS = ((-49000.0, 49000.0), (-29000.0, 29000.0))
HOLDER_CENTRE = (400.0, -325.0)
#: The wells' centres as the holder places the plate, not turned (µm).
CENTRES = {
    "B2": (-28775.0, 9400.0),
    "B5": (29575.0, 9400.0),
    "C2": (-28775.0, -10050.0),
    "A6": (49025.0, 28850.0),  # 25 µm outside the +X limit
    "D1": (-48225.0, -29500.0),  # 500 µm outside the -Y limit
}
GOTO = EntryPoint(
    name="goto-well", value="smc.plugins.goto_well:GotoWellPlugin", group=GROUP
)


@pytest.fixture
def ti2e_stand(
    mm_available: bool, tmp_path: Path
) -> Iterator[tuple[PlatePage, Microscope]]:
    """The demo devices on the Ti2-E's holder and soft limits, and a page over them with the real ``goto-well``."""
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    demo = Profile.demo()
    profile = demo.model_copy(
        update={
            "holder": HolderSection(
                plate_centre_xy_um=HOLDER_CENTRE,
                measured="a test's copy of the Ti2-E's holder",
            ),
            "safety": demo.safety.model_copy(update={"xy_soft_limits_um": LIMITS}),
        }
    )
    microscope = Microscope.open(profile, dry_run=False)
    page = PlatePage(microscope, tmp_path / "run.json", entry_points=[GOTO])
    try:
        # The demo stage's default speed would take minutes over 60 mm.
        microscope.core.setProperty("XY", "Velocity", "100000")
        yield page, microscope
    finally:
        page.close()
        microscope.close()


def _where(microscope: Microscope) -> tuple[float, float]:
    xy = microscope.require(XYStage).position_um()
    return (xy.x_um, xy.y_um)


def test_the_suggested_wells_are_driven_to_and_the_corners_are_refused(
    ti2e_stand: tuple[PlatePage, Microscope],
) -> None:
    page, microscope = ti2e_stand
    page.choose_plate("greiner-24-well")
    suggested = page.suggested_centres()
    assert suggested == ("B2", "B5", "C2")
    for well in suggested:
        page.goto(well)
        assert page.join(60.0)
        record = RunDocument.load(page.run_doc).stages["goto-well"]
        assert record.status is StageStatus.done, page.status().goto
        assert _where(microscope) == pytest.approx(CENTRES[well], abs=1.0)

    # The corners the rows used to offer: the layer refuses them, with its own words, and
    # the stage stays where the last drive left it (C2).
    for corner in ("A6", "D1"):
        page.goto(corner)
        assert page.join(60.0)
        record = RunDocument.load(page.run_doc).stages["goto-well"]
        assert record.status is StageStatus.failed
        assert "outside the soft limits" in page.status().goto
        assert _where(microscope) == pytest.approx(CENTRES["C2"], abs=1.0)
