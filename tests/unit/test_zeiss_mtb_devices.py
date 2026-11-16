"""The MTB unicore devices on a ``UniMMCore`` over ``FakeMTB`` (m4 §7, §8).

Every test drives the devices through the core, as the layer does, from the
profile's own ``.cfg`` without its camera (``fake_stand_config``). A move runs
on the device's worker; a test that moves slowly waits for it
(``waitForDevice``) before it ends, so no worker outlives it (FM-44).
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from pymmcore_plus import CMMCorePlus

from smc.hardware import Microscope
from smc.hardware.backends.zeiss_mtb import session
from smc.hardware.backends.zeiss_mtb.api import AxisInfo, ChangerElement, MTBError
from smc.hardware.backends.zeiss_mtb.devices import arrived, changer_labels
from smc.hardware.capabilities import ZStage
from smc.hardware.core import CoreError, close_core, open_core
from smc.hardware.errors import HardwareError
from smc.testing.fake_mtb import FakeMTB, fake_stand_config

CFG = Path(__file__).resolve().parents[2] / "profiles" / "zeiss-observer7.cfg"
FOCUS = AxisInfo(-14_000.0, 14_000.0, 0.01, 0.02, 0.1)


@pytest.fixture
def fake() -> FakeMTB:
    return FakeMTB()


@pytest.fixture
def stand_cfg(tmp_path: Path) -> Path:
    return fake_stand_config(CFG, tmp_path)


@pytest.fixture
def core(fake: FakeMTB, stand_cfg: Path) -> Iterator[CMMCorePlus]:
    """The fake stand's core, inside the test's own MTB session."""
    with session.use_api_factory(lambda: fake):
        opened = open_core(stand_cfg)
        try:
            yield opened
        finally:
            close_core(opened)


def _set_elements(fake: FakeMTB, mtb_id: str, *names: str) -> None:
    """Give a fake changer other element names, before the stand opens."""
    changer = fake.changers[mtb_id]
    changer._elements = tuple(
        ChangerElement(position=i, name=n, element_type="Element")
        for i, n in enumerate(names, start=1)
    )
    changer.set_position(1)


def _moves(fake: FakeMTB, mtb_id: str) -> list[tuple[object, ...]]:
    return [c for c in fake.calls if c[0] == mtb_id and c[1] in ("move_um", "move")]


# -- arrival, the pure rule ---------------------------------------------------


def test_arrival_takes_the_larger_of_the_step_and_the_typical_deviation() -> None:
    assert arrived(100.0, 100.019, FOCUS)
    assert not arrived(100.0, 100.021, FOCUS)
    coarse = AxisInfo(-1.0, 1.0, 0.25, 0.0, 0.0)
    assert arrived(0.0, 0.25, coarse)
    assert not arrived(0.0, 0.26, coarse)


def test_arrival_never_demands_equality() -> None:
    exact = AxisInfo(-1.0, 1.0, 0.0, 0.0, 0.0)
    assert arrived(0.5, 0.5 + 5e-7, exact)
    assert not arrived(0.5, 0.5 + 2e-6, exact)


# -- axes ----------------------------------------------------------------------


@pytest.mark.wallclock
def test_a_read_does_not_wait_behind_a_move(core: CMMCorePlus, fake: FakeMTB) -> None:
    fake.axes["MTBFocus"].move_s = 0.5
    core.setPosition("Focus", 1100.0)
    time.sleep(0.1)
    started = time.perf_counter()
    read = core.getPosition("Focus")
    elapsed = time.perf_counter() - started
    busy = core.deviceBusy("Focus")
    core.waitForDevice("Focus")
    assert elapsed < 0.25
    assert 1000.0 < read < 1100.0
    assert busy is True
    assert core.getPosition("Focus") == pytest.approx(1100.0)


def test_busy_is_true_while_the_mtb_move_runs_and_false_after(
    core: CMMCorePlus, fake: FakeMTB
) -> None:
    fake.axes["MTBFocus"].move_s = 0.3
    core.setPosition("Focus", 1050.0)
    assert core.deviceBusy("Focus") is True
    core.waitForDevice("Focus")
    assert core.deviceBusy("Focus") is False
    assert core.getPosition("Focus") == pytest.approx(1050.0)


def test_a_refused_move_raises_at_the_next_busy_once(
    core: CMMCorePlus, fake: FakeMTB, caplog: pytest.LogCaptureFixture
) -> None:
    fake.axes["MTBFocus"].refuse_next = True
    with caplog.at_level(logging.ERROR, logger="smc.hardware.zeiss_mtb"):
        core.setPosition("Focus", 1100.0)
        with pytest.raises(MTBError, match=r"Focus: MTB refused the move to 1100.0 µm"):
            core.waitForDevice("Focus")
    assert core.deviceBusy("Focus") is False
    assert any("the move failed" in r.getMessage() for r in caplog.records)


def test_a_false_return_within_the_typical_deviation_is_an_arrival(
    core: CMMCorePlus, fake: FakeMTB
) -> None:
    focus = fake.axes["MTBFocus"]
    focus.false_next = True
    focus.deviation_um = 0.015
    core.setPosition("Focus", 1100.0)
    core.waitForDevice("Focus")
    assert core.getPosition("Focus") == pytest.approx(1100.015)


def test_a_false_return_beyond_the_typical_deviation_is_a_failure(
    core: CMMCorePlus, fake: FakeMTB
) -> None:
    focus = fake.axes["MTBFocus"]
    focus.false_next = True
    focus.deviation_um = 0.05
    core.setPosition("Focus", 1100.0)
    with pytest.raises(MTBError, match="refused the move"):
        core.waitForDevice("Focus")


def test_a_true_return_within_the_max_deviation_is_an_arrival(
    core: CMMCorePlus, fake: FakeMTB
) -> None:
    fake.axes["MTBFocus"].deviation_um = 0.08
    core.setPosition("Focus", 1100.0)
    core.waitForDevice("Focus")


def test_a_true_return_beyond_the_max_deviation_is_a_failure(
    core: CMMCorePlus, fake: FakeMTB
) -> None:
    fake.axes["MTBFocus"].deviation_um = 0.5
    core.setPosition("Focus", 1100.0)
    with pytest.raises(
        MTBError, match=r"beyond its declared maximum deviation of 0\.1"
    ):
        core.waitForDevice("Focus")


def test_a_target_already_arrived_at_is_not_sent(
    core: CMMCorePlus, fake: FakeMTB
) -> None:
    core.setPosition("Focus", 1000.015)
    assert core.deviceBusy("Focus") is False
    assert _moves(fake, "MTBFocus") == []


def test_a_target_outside_the_axis_range_is_refused_not_clamped(
    core: CMMCorePlus, fake: FakeMTB
) -> None:
    with pytest.raises(
        MTBError, match=r"outside the axis's range \[-14000.0, 14000.0\]"
    ):
        core.setPosition("Focus", 14_000.5)
    with pytest.raises(MTBError, match="outside the axis's range"):
        core.setXYPosition("XYStage", 0.0, -350_001.0)
    assert _moves(fake, "MTBFocus") == []
    assert _moves(fake, "MTBStageAxisX") == []
    assert _moves(fake, "MTBStageAxisY") == []


def test_a_move_while_moving_is_refused(core: CMMCorePlus, fake: FakeMTB) -> None:
    fake.axes["MTBPiezoFocusCan"].move_s = 0.5
    core.setPosition("Piezo", 300.0)
    try:
        with pytest.raises(MTBError, match="is moving"):
            core.setPosition("Piezo", 200.0)
    finally:
        core.waitForDevice("Piezo")
    assert [c[2] for c in _moves(fake, "MTBPiezoFocusCan")] == [300.0]


def test_xy_moves_x_then_y_and_skips_an_axis_already_there(
    core: CMMCorePlus, fake: FakeMTB
) -> None:
    core.setXYPosition("XYStage", 120.0, -35.0)
    core.waitForDevice("XYStage")
    legs = [(c[0], c[2]) for c in fake.calls if c[1] == "move_um"]
    assert legs == [("MTBStageAxisX", 120.0), ("MTBStageAxisY", -35.0)]
    core.setXYPosition("XYStage", 120.0, 400.0)
    core.waitForDevice("XYStage")
    legs = [(c[0], c[2]) for c in fake.calls if c[1] == "move_um"]
    assert legs[2:] == [("MTBStageAxisY", 400.0)]
    assert core.getXYPosition("XYStage") == pytest.approx((120.0, 400.0))


def test_stop_during_the_x_leg_cancels_the_y_leg(
    core: CMMCorePlus, fake: FakeMTB
) -> None:
    fake.axes["MTBStageAxisX"].move_s = 0.5
    core.setXYPosition("XYStage", 100.0, 200.0)
    time.sleep(0.1)
    try:
        with pytest.raises(MTBError, match="no axis stop"):
            core.stop("XYStage")
    finally:
        core.waitForDevice("XYStage")
    assert _moves(fake, "MTBStageAxisY") == []
    assert core.getXYPosition("XYStage") == pytest.approx((100.0, 0.0))


def test_a_stop_is_refused_on_an_idle_axis_too(core: CMMCorePlus) -> None:
    # The layer reads the refusal as the one the profile declares (FM-117).
    with pytest.raises(MTBError, match="no axis stop"):
        core.stop("Focus")


def test_home_and_set_origin_are_refused(core: CMMCorePlus, fake: FakeMTB) -> None:
    for call in (
        lambda: core.home("XYStage"),
        lambda: core.home("Focus"),
        lambda: core.setOrigin("Focus"),
        lambda: core.setOriginX("XYStage"),
        lambda: core.setOriginY("XYStage"),
        lambda: core.setOriginXY("XYStage"),
    ):
        with pytest.raises(MTBError, match="never homes"):
            call()
    assert _moves(fake, "MTBFocus") == []


def test_unload_waits_for_a_move_still_running(
    core: CMMCorePlus, fake: FakeMTB
) -> None:
    fake.axes["MTBFocus"].move_s = 0.3
    core.setPosition("Focus", 1200.0)
    core.unloadAllDevices()
    assert fake.axes["MTBFocus"].position_um() == pytest.approx(1200.0)


# -- changers ------------------------------------------------------------------


def test_changer_states_are_mtb_positions_minus_min_position(
    core: CMMCorePlus, fake: FakeMTB
) -> None:
    assert core.getState("Objective") == 1  # MTB position 2
    assert core.getNumberOfStates("Objective") == 6
    core.setState("Objective", 3)
    core.waitForDevice("Objective")
    assert fake.changers["MTBObjectiveChanger"].position() == 4
    assert core.getState("Objective") == 3
    assert core.getStateLabel("Objective") == "Plan-Apochromat 40x/0.95"


def test_changer_labels_are_element_names_made_unique(
    fake: FakeMTB, stand_cfg: Path
) -> None:
    _set_elements(fake, "MTBReflectorChanger", "none", " GFP ", "none", "DAPI", "")
    with session.use_api_factory(lambda: fake):
        core = open_core(stand_cfg)
        try:
            labels = core.getStateLabels("Reflector")
            core.setStateLabel("Reflector", "none (3)")
            core.waitForDevice("Reflector")
            position = fake.changers["MTBReflectorChanger"].position()
        finally:
            close_core(core)
    assert labels == ("none (1)", "GFP", "none (3)", "DAPI", "Position-5")
    assert position == 3


def test_changer_labels_the_pure_rule() -> None:
    elements = tuple(
        ChangerElement(position=p, name=n, element_type="Element")
        for p, n in ((1, "PH1"), (2, "  "), (3, "PH1"), (4, "BF"))
    )
    assert changer_labels(elements) == ["PH1 (1)", "Position-2", "PH1 (3)", "BF"]


@pytest.mark.wallclock
def test_set_state_returns_at_once_while_the_changer_moves(
    core: CMMCorePlus, fake: FakeMTB
) -> None:
    fake.changers["MTBSideportChanger"].move_s = 0.5
    started = time.perf_counter()
    core.setStateLabel("Sideport", "VIS 100%")
    elapsed = time.perf_counter() - started
    busy = core.deviceBusy("Sideport")
    during = core.getState("Sideport")
    core.waitForDevice("Sideport")
    assert elapsed < 0.25
    assert busy is True
    assert during == 1  # still on the left sideport, live
    assert core.getState("Sideport") == 0
    assert len(_moves(fake, "MTBSideportChanger")) == 1


def test_set_state_sends_one_move_even_when_it_ends_at_once(
    core: CMMCorePlus, fake: FakeMTB
) -> None:
    # unicore sets the State, then the Label, which sets the State again.
    core.setStateLabel("Objective", "Plan-Apochromat 40x/0.95")
    core.waitForDevice("Objective")
    core.setState("Objective", 1)
    core.waitForDevice("Objective")
    assert [c[2] for c in _moves(fake, "MTBObjectiveChanger")] == [4, 2]


def test_a_changer_that_lands_elsewhere_is_a_failure(
    core: CMMCorePlus, fake: FakeMTB
) -> None:
    fake.changers["MTBCondenserContrastChanger"].lands_at = 2
    core.setState("Condenser", 4)
    with pytest.raises(MTBError, match="reads position 2"):
        core.waitForDevice("Condenser")


def test_a_refused_changer_move_is_a_failure(core: CMMCorePlus, fake: FakeMTB) -> None:
    fake.changers["MTBReflectorChanger"].refuse_next = True
    core.setState("Reflector", 2)
    with pytest.raises(MTBError, match="refused the move to position 3"):
        core.waitForDevice("Reflector")
    assert core.getState("Reflector") == 0


# -- TL shutter ----------------------------------------------------------------


@pytest.mark.parametrize(
    "names",
    [("Closed", "Shut"), ("Open", " open "), ("Open", "Closed", "Half")],
    ids=["no-open", "two-open", "three-positions"],
)
def test_tl_shutter_without_exactly_one_open_element_fails_initialize(
    fake: FakeMTB,
    stand_cfg: Path,
    names: tuple[str, ...],
    caplog: pytest.LogCaptureFixture,
) -> None:
    _set_elements(fake, "MTBTLShutter", *names)
    with (
        session.use_api_factory(lambda: fake),
        caplog.at_level(logging.ERROR),
        pytest.raises(CoreError, match="TLShutter"),
    ):
        open_core(stand_cfg)
    errors = " ".join(
        r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR
    )
    assert "TLShutter did not initialise" in errors
    assert repr(list(names)) in errors


def test_tl_shutter_opens_and_closes_by_its_named_positions(
    core: CMMCorePlus, fake: FakeMTB
) -> None:
    tl = fake.changers["MTBTLShutter"]
    assert core.getShutterOpen("TLShutter") is False
    core.setShutterOpen("TLShutter", True)
    assert tl.position() == 1  # "Open"
    assert core.getShutterOpen("TLShutter") is True
    core.setShutterOpen("TLShutter", False)
    assert tl.position() == 2


def test_tl_shutter_that_does_not_move_is_a_failure(
    core: CMMCorePlus, fake: FakeMTB
) -> None:
    fake.changers["MTBTLShutter"].refuse_next = True
    with pytest.raises(MTBError, match="did not move the TL shutter"):
        core.setShutterOpen("TLShutter", True)


# -- Colibri -------------------------------------------------------------------


def test_colibri_open_lights_the_selected_led_and_switches_every_other_off(
    core: CMMCorePlus, fake: FakeMTB
) -> None:
    leds = fake.colibri.led_list
    leds[0].on = leds[5].on = True  # lit by hand after the connect
    core.setProperty("Colibri", "LED", "475nm")
    core.setProperty("Colibri", "Intensity", 30.0)
    core.setShutterOpen("Colibri", True)
    assert fake.colibri.lit() == [475]
    assert leds[2].intensity == pytest.approx(30.0)
    assert fake.colibri.manual is True
    assert core.getShutterOpen("Colibri") is True


def test_colibri_led_values_are_none_and_one_per_led(core: CMMCorePlus) -> None:
    assert set(core.getAllowedPropertyValues("Colibri", "LED")) == {
        "None",
        "385nm",
        "430nm",
        "475nm",
        "511nm",
        "567nm",
        "630nm",
    }
    assert core.getProperty("Colibri", "LED") == "None"
    assert float(core.getProperty("Colibri", "Intensity")) == 0.0


def test_colibri_open_without_a_led_is_refused(
    core: CMMCorePlus, fake: FakeMTB
) -> None:
    with pytest.raises(MTBError, match="no LED is selected"):
        core.setShutterOpen("Colibri", True)
    assert fake.colibri.lit() == []


def test_colibri_close_tries_every_led_and_reports_each_failure(
    core: CMMCorePlus, fake: FakeMTB
) -> None:
    core.setProperty("Colibri", "LED", "475nm")
    core.setShutterOpen("Colibri", True)
    leds = fake.colibri.led_list
    leds[0].on = leds[5].on = True
    leds[0].fail_off = leds[5].fail_off = True
    before = len(fake.calls)
    with pytest.raises(MTBError, match="2 LED") as caught:
        core.setShutterOpen("Colibri", False)
    assert "385nm" in str(caught.value)
    assert "630nm" in str(caught.value)
    offs = [c[0] for c in fake.calls[before:] if c[1] == "set_on" and c[2] is False]
    assert offs == [f"LED{nm}" for nm in (385, 430, 475, 511, 567, 630)]
    assert fake.colibri.lit() == [385, 630]


def test_colibri_a_failed_switch_on_darkens_every_led(
    core: CMMCorePlus, fake: FakeMTB
) -> None:
    fake.colibri.led_list[2].fail_on = True
    core.setProperty("Colibri", "LED", "475nm")
    with pytest.raises(MTBError, match="SetOnOff"):
        core.setShutterOpen("Colibri", True)
    assert fake.colibri.lit() == []


def test_colibri_reads_open_when_any_led_is_on(
    core: CMMCorePlus, fake: FakeMTB
) -> None:
    assert core.getShutterOpen("Colibri") is False
    fake.colibri.led_list[5].on = True
    assert core.getShutterOpen("Colibri") is True


def test_intensity_set_while_open_applies_to_the_lit_led(
    core: CMMCorePlus, fake: FakeMTB
) -> None:
    led = fake.colibri.led_list[2]
    core.setProperty("Colibri", "Intensity", 55.0)
    assert led.intensity == 0.0  # closed: kept for the next open
    core.setProperty("Colibri", "LED", "475nm")
    core.setShutterOpen("Colibri", True)
    assert led.intensity == pytest.approx(55.0)
    core.setProperty("Colibri", "Intensity", 12.5)
    assert led.intensity == pytest.approx(12.5)
    with pytest.raises(ValueError, match="allowed range"):
        core.setProperty("Colibri", "Intensity", 101.0)


def test_led_set_while_open_lights_the_new_led_and_none_closes(
    core: CMMCorePlus, fake: FakeMTB
) -> None:
    core.setProperty("Colibri", "LED", "475nm")
    core.setShutterOpen("Colibri", True)
    core.setProperty("Colibri", "LED", "567nm")
    assert fake.colibri.lit() == [567]
    core.setProperty("Colibri", "LED", "None")
    assert fake.colibri.lit() == []
    assert core.getShutterOpen("Colibri") is False


def test_initialize_darkens_a_stand_left_lit(fake: FakeMTB, stand_cfg: Path) -> None:
    for led in fake.colibri.led_list:
        led.on = True
    fake.changers["MTBTLShutter"].set_position(1)  # "Open", left by ZEN
    with session.use_api_factory(lambda: fake):
        core = open_core(stand_cfg)
        try:
            assert fake.colibri.lit() == []
            assert fake.changers["MTBTLShutter"].position() == 2
        finally:
            close_core(core)


def test_a_colibri_that_cannot_darken_at_initialize_fails_the_open(
    fake: FakeMTB, stand_cfg: Path
) -> None:
    fake.colibri.led_list[1].on = True
    fake.colibri.led_list[1].fail_off = True
    with (
        session.use_api_factory(lambda: fake),
        pytest.raises(CoreError, match="Colibri"),
    ):
        open_core(stand_cfg)


def test_unload_darkens_the_light_and_never_raises(
    core: CMMCorePlus, fake: FakeMTB, caplog: pytest.LogCaptureFixture
) -> None:
    core.setProperty("Colibri", "LED", "475nm")
    core.setShutterOpen("Colibri", True)
    core.setShutterOpen("TLShutter", True)
    fake.colibri.led_list[0].on = True
    fake.colibri.led_list[0].fail_off = True
    with caplog.at_level(logging.ERROR, logger="smc.hardware.zeiss_mtb"):
        core.unloadAllDevices()
    assert fake.colibri.lit() == [385]
    assert fake.changers["MTBTLShutter"].position() == 2
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert any(e.startswith("LIGHT MAY STILL BE ON: Colibri:") for e in errors)
    assert core.getLoadedDevices() == ("Core",)


def test_an_unloaded_light_leaves_no_darkener_for_the_exit(
    core: CMMCorePlus, fake: FakeMTB
) -> None:
    core.unloadAllDevices()
    assert session.shared()._darkeners == {}


# -- errors and imports --------------------------------------------------------


def test_a_seam_error_leaves_the_device_as_an_mtb_error_and_the_layer_as_a_hardware_error(
    zeiss_fake_microscope: Microscope,
    fake_mtb: FakeMTB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken() -> float:
        raise OSError("the CAN bus did not answer")

    monkeypatch.setattr(fake_mtb.axes["MTBFocus"], "position_um", broken)
    with pytest.raises(MTBError, match="the CAN bus did not answer") as at_device:
        zeiss_fake_microscope.core.getPosition("Focus")
    assert isinstance(at_device.value.__cause__, OSError)
    with pytest.raises(HardwareError, match="focus") as at_layer:
        zeiss_fake_microscope.require(ZStage).position_um()
    assert "the CAN bus did not answer" in str(at_layer.value)


def test_importing_the_devices_does_not_import_pythonnet(tmp_path: Path) -> None:
    script = tmp_path / "probe.py"
    script.write_text(
        "import sys\n"
        "import smc.hardware.backends.zeiss_mtb.devices\n"
        "import smc.hardware.backends.zeiss_mtb.session\n"
        "import smc.testing.fake_mtb\n"
        "import smc.testing.fixtures\n"
        "leaked = sorted(m for m in ('clr', 'System', 'ZEISS', 'pythonnet') if m in sys.modules)\n"
        "assert not leaked, leaked\n",
        encoding="utf-8",
    )
    out, err = tmp_path / "out.txt", tmp_path / "err.txt"
    with out.open("wb") as stdout, err.open("wb") as stderr:
        child = subprocess.Popen(
            [sys.executable, str(script)], stdout=stdout, stderr=stderr
        )
        try:
            code = child.wait(timeout=60)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=10)
            raise
    assert code == 0, err.read_text(encoding="utf-8", errors="replace")
