"""The owner's path on the plate page: two wells centred by hand, one click, a plate with a residual (m5 §12, #346).

« Positionner à la main avec le joystick sur 2 ou 3 wells, puis la calibration se fait
toute seule. » This runs the page's controller the way the page's buttons do, on the
``demo-sample`` stand (a camera that images a synthetic 96-well plate where the stage
is), from Connect to the registered plate: *camera-calibration* by the next step's own
pairs, two wells reached by the Drive pad's relative moves and captured, then the step
the page offers, which is ``plate-walls``.

The operator's two centres are a few hundred micrometres off the wells' true centres, as
a hand on a joystick would leave them. The expected values are the sample's own geometry,
written out here (FM-124): the profile puts A1 at (-49 500, 31 500) µm, the pitch is 9 mm.

The run took 16.5 s on the demo stage for two wells (measured 2026-11-09, Python 3.13.0,
Panel 1.10.0); no assertion is made on a duration.
"""

from __future__ import annotations

from importlib.metadata import EntryPoint, entry_points
from pathlib import Path

import pytest

from smc.hardware.profile import Profile
from smc.plugins.registry import GROUP
from smc.plugins.slots import PlateResult
from smc.run import RunDocument
from smc.ui.controller import PlatePage

pytestmark = [
    pytest.mark.demo,
    # A failure a handler swallowed fails the test that did not expect it (#277).
    pytest.mark.usefixtures("handler_failures"),
]

DEMO_SAMPLE = Path(__file__).resolve().parents[1] / "profiles" / "demo-sample.toml"
#: The sample's A1 and the plate's pitch, from the profile (a 96-well plate centred on the
#: stage origin, not turned): D6 and D7 are three rows down and five and six columns right.
A1_UM = (-49_500.0, 31_500.0)
PITCH_UM = 9000.0
#: Where the wells are, written out: column and row indices of D6 and D7.
CENTRES_UM = {
    "D6": (A1_UM[0] + 5 * PITCH_UM, A1_UM[1] - 3 * PITCH_UM),
    "D7": (A1_UM[0] + 6 * PITCH_UM, A1_UM[1] - 3 * PITCH_UM),
}


def _entry_points() -> list[EntryPoint]:
    """The installed plugins, and ``plate-walls`` once: a venv installed before #351 has no entry point for it (FM-112)."""
    installed = [ep for ep in entry_points(group=GROUP) if ep.name != "plate-walls"]
    walls = EntryPoint(
        name="plate-walls",
        value="smc.plugins.plate_walls.plugin:PlateWallsPlugin",
        group=GROUP,
    )
    return [*installed, walls]


def _jog(page: PlatePage, dx_um: float, dy_um: float) -> None:
    """One relative move of the Drive pad, ended: the pad's limit is 5000 µm a move."""
    page.jog_xy(dx_um, dy_um)
    assert page.join(30.0), "the move did not end"
    outcome = page.status().outcome
    assert outcome is not None
    assert outcome.startswith("stage jog"), outcome


def test_two_captured_wells_register_the_plate_in_one_click_on_demo_sample(
    mm_available: bool, tmp_path: Path
) -> None:
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    run_doc = tmp_path / "run.json"
    page = PlatePage.closed(
        Profile.load(DEMO_SAMPLE), run_doc, entry_points=_entry_points()
    )
    try:
        page.connect()
        assert page.join(60.0), "connect did not end"
        assert page.connected, page.status().outcome
        # The sample's own plate is the page's: nothing to choose (m5 §11.3).
        assert (page.plate, page.plate_chosen) == ("96-well", True)

        # The calibration the tool reads: the page does not run it for the operator.
        page.run("camera-calibration", [])
        assert page.join(60.0), "camera-calibration did not end"
        outcome = page.status().outcome
        assert outcome is not None
        assert outcome.startswith("Camera calibration done in "), outcome

        # D6, a few hundred µm off its centre as a hand leaves it (300 and -200 µm), by
        # relative moves of at most 4500 µm on one axis, as the Drive pad makes them.
        _jog(page, -4200.0, 0.0)
        _jog(page, 0.0, 4300.0)
        ref = page.capture_centre("D6")
        assert (ref.x_um, ref.y_um) == pytest.approx((-4200.0, 4300.0), abs=1.0)
        # D7, 200 µm and 100 µm off.
        _jog(page, 4500.0, 0.0)
        _jog(page, 4400.0, 0.0)
        _jog(page, 0.0, 100.0)
        page.capture_centre("D7")
        assert [r.well for r in page.refs] == ["D6", "D7"]

        # Two centres, no wall touched, the calibration done: the page offers the tool.
        step = page.view().next_step
        assert step is not None
        assert step.plugin == "plate-walls", step.why
        assert step.blocked is None
        assert list(step.params) == page.page_params("plate-walls")
        page.run(step.plugin, list(step.params))
        assert page.join(120.0), "plate-walls did not end"
        outcome = page.status().outcome
        assert outcome is not None
        assert outcome.startswith("Registration from the walls done in "), outcome

        # What it left: the plate, registered from the walls, with a residual that was
        # measured and A1 where the sample puts it.
        provider = RunDocument.load(run_doc).provider("plate")
        assert provider is not None
        assert provider.plugin == "plate-walls"
        result = PlateResult.model_validate(provider.result)
        assert result.method == "wall-images"
        assert result.residual_um is not None
        assert result.a1_center_xy_um == pytest.approx(A1_UM, abs=2.0)
        assert result.walls is not None
        measured = {fit.well: fit.center_xy_um for fit in result.walls.wells}
        assert set(measured) == {"D6", "D7"}
        for well, centre in CENTRES_UM.items():
            assert measured[well] == pytest.approx(centre, abs=2.0)

        # The card says the wells and the plate, and where the frames are.
        card = page.view().cards["plate-walls"]
        assert [line.partition(":")[0] for line in card.lines] == [
            "D6",
            "D7",
            "plate 96-well",
        ]
        assert card.lines[-1].startswith("plate 96-well: residual ")
        assert len(card.files) == 1
        assert card.files[0].startswith("folder: ")
    finally:
        page.close()
        page.release()
