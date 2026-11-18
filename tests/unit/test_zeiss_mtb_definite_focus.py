"""``MTBDefiniteFocus`` on a ``UniMMCore`` over ``FakeMTB`` (m4 §11.4).

The device is driven through the core, as ``MMFocusStabilizer`` drives it, from
a ``.cfg`` that loads DF2 alone, so these tests do not depend on the rest of
the stand. A test whose action runs slowly waits for it (``waitForDevice``)
before it ends, so no worker outlives it (FM-44).
"""

from __future__ import annotations

import base64
import logging
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from pymmcore_plus import CMMCorePlus

from smc.hardware.backends.zeiss_mtb import devices, session
from smc.hardware.backends.zeiss_mtb.api import STABILIZER_TIMEOUT_MS, MTBError
from smc.hardware.core import CoreError, close_core, open_core
from smc.testing.fake_mtb import FakeMTB

DF2 = "DefiniteFocus"
MTB_ID = "MTBFocusStabilizer2"
CFG = (
    "Property,Core,Initialize,0\n"
    f"#py pyDevice,{DF2},smc.hardware.backends.zeiss_mtb.devices,MTBDefiniteFocus\n"
    "Property,Core,Initialize,1\n"
)
#: How long a slow fake action runs: long enough to read during it.
SLOW_S = 0.5
#: A read during an action answers well within this (m4 §11.4 rule 2).
QUICK_S = 0.1


@pytest.fixture
def cfg(tmp_path: Path) -> Path:
    path = tmp_path / "df2.cfg"
    path.write_text(CFG, encoding="utf-8")
    return path


@pytest.fixture
def fake() -> FakeMTB:
    return FakeMTB()


@pytest.fixture
def core(fake: FakeMTB, cfg: Path) -> Iterator[CMMCorePlus]:
    """DF2 alone on a core, inside the test's own MTB session."""
    with session.use_api_factory(lambda: fake):
        opened = open_core(cfg)
        try:
            yield opened
        finally:
            close_core(opened)


def _calls(fake: FakeMTB, *names: str) -> list[tuple[object, ...]]:
    return [c for c in fake.calls if c[0] == MTB_ID and c[1] in names]


def _learn(core: CMMCorePlus) -> str:
    core.setProperty(DF2, "Action", "Learn")
    core.waitForDevice(DF2)
    return str(core.getProperty(DF2, "Reference"))


def _timed(call: Callable[[], object]) -> tuple[object, float]:
    started = time.perf_counter()
    value = call()
    return value, time.perf_counter() - started


def _wait_for(condition: Callable[[], bool], what: str) -> None:
    deadline = time.monotonic() + 5.0
    while not condition():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.01)


# -- initialize ----------------------------------------------------------------


def test_initialize_switches_periodic_off_and_holds_no_reference(
    fake: FakeMTB, cfg: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Another MTB client (ZEN) left periodic stabilisation on and a reference
    # learned: neither is this process's to trust (FM-118).
    fake.stabilizer.periodic_on = True
    fake.stabilizer.learn(STABILIZER_TIMEOUT_MS)
    with (
        session.use_api_factory(lambda: fake),
        caplog.at_level(logging.WARNING, logger="smc.hardware.zeiss_mtb"),
    ):
        core = open_core(cfg)
        try:
            assert _calls(fake, "stop_periodic") == [
                (MTB_ID, "stop_periodic", STABILIZER_TIMEOUT_MS)
            ]
            assert core.getProperty(DF2, "Periodic") == "Off"
            assert core.getProperty(DF2, "Reference") == ""
            assert core.getProperty(DF2, "Action") == "Idle"
            assert core.getProperty(DF2, "OperationMode") == "Balanced"
        finally:
            close_core(core)
    assert any("periodic stabilisation was on" in m for m in caplog.messages)


def test_initialize_without_the_component_names_the_device(
    fake: FakeMTB, cfg: Path, caplog: pytest.LogCaptureFixture
) -> None:
    fake.absent.add(MTB_ID)
    with (
        session.use_api_factory(lambda: fake),
        caplog.at_level(logging.ERROR),
        pytest.raises(CoreError, match=DF2),
    ):
        open_core(cfg)
    assert (
        f"{DF2} did not initialise: MTB component '{MTB_ID}' is absent"
        in caplog.messages
    )


# -- the actions on the worker -------------------------------------------------


@pytest.mark.wallclock
def test_learn_returns_at_once_and_reference_reads_base64_after_the_wait(
    cfg: Path,
) -> None:
    fake = FakeMTB(move_s=SLOW_S)
    with session.use_api_factory(lambda: fake):
        core = open_core(cfg)
        try:
            _, took = _timed(lambda: core.setProperty(DF2, "Action", "Learn"))
            assert took < QUICK_S
            assert core.getProperty(DF2, "Action") == "Learn"
            assert core.getProperty(DF2, "Reference") == ""
            assert core.deviceBusy(DF2) is True
            core.waitForDevice(DF2)
            text = str(core.getProperty(DF2, "Reference"))
        finally:
            close_core(core)
    assert fake.stabilizer.held is not None
    assert base64.b64decode(text, validate=True) == fake.stabilizer.held
    assert text.isascii()
    assert any(b > 0x7F for b in fake.stabilizer.held)


@pytest.mark.wallclock
def test_reads_answer_while_a_stabilize_runs(cfg: Path) -> None:
    fake = FakeMTB()
    with session.use_api_factory(lambda: fake):
        core = open_core(cfg)
        try:
            _learn(core)
            focus = fake.axes["MTBFocus"]
            focus.set_position_um(995.0)  # a drift, as a hand on the knob
            fake.stabilizer.move_s = SLOW_S
            core.setProperty(DF2, "Action", "StabilizeNow")
            _wait_for(lambda: focus.position_um() > 995.0, "the stabilize to start")
            action, took_action = _timed(lambda: core.getProperty(DF2, "Action"))
            periodic, took_periodic = _timed(lambda: core.getProperty(DF2, "Periodic"))
            busy, took_busy = _timed(lambda: core.deviceBusy(DF2))
            between = focus.position_um()
            core.waitForDevice(DF2)
        finally:
            close_core(core)
    assert (action, periodic, busy) == ("StabilizeNow", "Off", True)
    assert max(took_action, took_periodic, took_busy) < QUICK_S
    assert 995.0 < between < 1000.0
    assert focus.position_um() == pytest.approx(1000.0)


def test_a_second_action_while_one_runs_is_refused(
    core: CMMCorePlus, fake: FakeMTB
) -> None:
    text = _learn(core)
    fake.stabilizer.move_s = SLOW_S
    core.setProperty(DF2, "Action", "StabilizeNow")
    try:
        for name, value in (
            ("Action", "Learn"),
            ("Action", "StabilizeNow"),
            ("Reference", text),
            ("OperationMode", "Exact"),
        ):
            with pytest.raises(
                MTBError, match=f"{DF2}: StabilizeNow is running; .* is refused"
            ):
                core.setProperty(DF2, name, value)
    finally:
        core.waitForDevice(DF2)
    assert len(_calls(fake, "learn")) == 1
    assert len(_calls(fake, "stabilize_now")) == 1
    assert _calls(fake, "restore", "set_operation_mode") == []
    assert core.getProperty(DF2, "Reference") == text


def test_stabilize_answered_false_is_raised_once_by_busy(
    core: CMMCorePlus, fake: FakeMTB, caplog: pytest.LogCaptureFixture
) -> None:
    _learn(core)
    fake.stabilizer.refuse_next = True
    with caplog.at_level(logging.ERROR, logger="smc.hardware.zeiss_mtb"):
        core.setProperty(DF2, "Action", "StabilizeNow")
        with pytest.raises(MTBError) as failed:
            core.waitForDevice(DF2)
    assert str(failed.value) == (
        f"{DF2}: DF2 answered False to StabilizeNow; the focus drive is where DF2 "
        "left it"
    )
    assert core.deviceBusy(DF2) is False
    assert any("the action failed" in m for m in caplog.messages)


def test_a_restore_of_text_that_is_not_base64_sends_nothing(
    core: CMMCorePlus, fake: FakeMTB
) -> None:
    text = _learn(core)
    for bad, why in (
        ("not base64!", "is not base64 text"),
        ("AAEC AwQF", "is not base64 text"),
        ("", "an empty Reference is refused"),
    ):
        with pytest.raises(MTBError, match=why):
            core.setProperty(DF2, "Reference", bad)
    assert _calls(fake, "restore") == []
    assert core.getProperty(DF2, "Reference") == text
    assert core.getProperty(DF2, "Action") == "Idle"


def test_a_failed_restore_leaves_no_reference(core: CMMCorePlus, fake: FakeMTB) -> None:
    text = _learn(core)
    unknown = base64.b64encode(b"\x00" * 9).decode("ascii")
    core.setProperty(DF2, "Reference", unknown)
    with pytest.raises(MTBError, match="DF2 answered False to InitWithStabilizingData"):
        core.waitForDevice(DF2)
    assert core.getProperty(DF2, "Reference") == ""
    # A blob DF2 knows comes back; one it refuses leaves none, whatever was held.
    core.setProperty(DF2, "Reference", text)
    core.waitForDevice(DF2)
    assert core.getProperty(DF2, "Reference") == text
    fake.stabilizer.refuse_next = True
    core.setProperty(DF2, "Reference", text)
    with pytest.raises(MTBError, match="was not restored"):
        core.waitForDevice(DF2)
    assert core.getProperty(DF2, "Reference") == ""


@pytest.mark.wallclock
def test_stop_is_sent_while_the_worker_runs(
    core: CMMCorePlus, fake: FakeMTB, caplog: pytest.LogCaptureFixture
) -> None:
    _learn(core)
    focus = fake.axes["MTBFocus"]
    focus.set_position_um(990.0)
    fake.stabilizer.move_s = 5.0
    core.setProperty(DF2, "Action", "StabilizeNow")
    _wait_for(lambda: focus.position_um() > 990.0, "the stabilize to start")
    with caplog.at_level(logging.WARNING, logger="smc.hardware.zeiss_mtb"):
        _, took = _timed(lambda: core.setProperty(DF2, "Action", "Stop"))
        with pytest.raises(MTBError, match="after a Stop was sent during it"):
            core.waitForDevice(DF2)
    assert took < QUICK_S
    assert _calls(fake, "stop") == [(MTB_ID, "stop", STABILIZER_TIMEOUT_MS)]
    # Stopped where the line had got to, far from the plane at 1000.
    assert 990.0 < focus.position_um() < 995.0
    assert any(
        "whether DF2 ends a synchronous call from another thread is not known" in m
        for m in caplog.messages
    )


def test_a_stop_before_the_worker_sends_its_call_cancels_it(
    core: CMMCorePlus, fake: FakeMTB, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The set returns before the worker reaches MTB: a Stop in between finds
    # nothing to stop in MTB, and the StabilizeNow sent after it would run.
    _learn(core)
    focus = fake.axes["MTBFocus"]
    focus.set_position_um(995.0)
    gate = threading.Event()
    real_start = devices._Mover.start

    def held_start(
        mover: devices._Mover,
        target: object,
        work: Callable[[threading.Event], None],
    ) -> None:
        def held(cancel: threading.Event) -> None:
            gate.wait(5.0)
            work(cancel)

        real_start(mover, target, held)

    monkeypatch.setattr(devices._Mover, "start", held_start)
    core.setProperty(DF2, "Action", "StabilizeNow")
    core.setProperty(DF2, "Action", "Stop")
    gate.set()
    with pytest.raises(MTBError, match="a Stop came before StabilizeNow was sent"):
        core.waitForDevice(DF2)
    assert _calls(fake, "stabilize_now") == []
    assert focus.position_um() == pytest.approx(995.0)


def test_stabilize_now_with_no_reference_is_refused_and_sends_nothing(
    core: CMMCorePlus, fake: FakeMTB
) -> None:
    # Another client left DF2 holding a plane: not this process's to trust
    # (FM-118), and a generic property set must not drive Z to it.
    fake.stabilizer.learn(STABILIZER_TIMEOUT_MS)
    focus = fake.axes["MTBFocus"]
    focus.set_position_um(990.0)
    before = len(_calls(fake, "learn"))
    with pytest.raises(MTBError, match="no reference was learned or restored"):
        core.setProperty(DF2, "Action", "StabilizeNow")
    assert _calls(fake, "stabilize_now") == []
    assert core.getProperty(DF2, "Action") == "Idle"
    assert focus.position_um() == 990.0
    # A failed re-learn leaves none either: the old blob is no longer held.
    _learn(core)
    fake.stabilizer.refuse_next = True
    core.setProperty(DF2, "Action", "Learn")
    with pytest.raises(MTBError, match="answered False"):
        core.waitForDevice(DF2)
    assert len(_calls(fake, "learn")) == before + 2
    assert core.getProperty(DF2, "Reference") == ""
    with pytest.raises(MTBError, match="no reference was learned or restored"):
        core.setProperty(DF2, "Action", "StabilizeNow")


def test_periodic_on_is_refused(core: CMMCorePlus, fake: FakeMTB) -> None:
    with pytest.raises(MTBError, match="never switches DF2's periodic"):
        core.setProperty(DF2, "Periodic", "On")
    assert _calls(fake, "stop_periodic") == []
    fake.stabilizer.periodic_on = True
    assert core.getProperty(DF2, "Periodic") == "On"
    core.setProperty(DF2, "Periodic", "Off")
    assert _calls(fake, "stop_periodic") == [
        (MTB_ID, "stop_periodic", STABILIZER_TIMEOUT_MS)
    ]
    assert core.getProperty(DF2, "Periodic") == "Off"


def test_piezo_focus_usage_is_read_only(core: CMMCorePlus, fake: FakeMTB) -> None:
    assert core.getProperty(DF2, "PiezoFocusUsage") == "Off"
    assert core.isPropertyReadOnly(DF2, "PiezoFocusUsage") is True
    with pytest.raises(ValueError, match="read-only"):
        core.setProperty(DF2, "PiezoFocusUsage", "On")
    fake.stabilizer.piezo_focus_usage_on = True
    assert core.getProperty(DF2, "PiezoFocusUsage") == "On"


@pytest.mark.wallclock
def test_shutdown_stops_a_running_stabilize_and_never_raises(
    cfg: Path, caplog: pytest.LogCaptureFixture
) -> None:
    fake = FakeMTB()
    with session.use_api_factory(lambda: fake):
        core = open_core(cfg)
        _learn(core)
        focus = fake.axes["MTBFocus"]
        focus.set_position_um(990.0)
        fake.stabilizer.move_s = 5.0
        fake.stabilizer.periodic_on = True  # left on by another client meanwhile
        core.setProperty(DF2, "Action", "StabilizeNow")
        _wait_for(lambda: focus.position_um() > 990.0, "the stabilize to start")
        with caplog.at_level(logging.WARNING, logger="smc.hardware.zeiss_mtb"):
            _, took = _timed(lambda: close_core(core))
    assert took < 2.0
    assert [c[1] for c in _calls(fake, "stop", "stop_periodic")] == [
        "stop",
        "stop_periodic",
    ]
    assert focus.position_um() < 995.0
    assert any(
        "periodic stabilisation switched off at unload" in m for m in caplog.messages
    )


def test_a_stop_that_fails_at_unload_is_logged_and_the_unload_goes_on(
    cfg: Path, caplog: pytest.LogCaptureFixture
) -> None:
    fake = FakeMTB()
    with session.use_api_factory(lambda: fake):
        core = open_core(cfg)
        _learn(core)
        fake.stabilizer.move_s = 0.3
        focus = fake.axes["MTBFocus"]
        focus.set_position_um(990.0)
        core.setProperty(DF2, "Action", "StabilizeNow")
        # Only once the stabilize moves the axis: set before, the knob could
        # fail the stabilize instead of the stop.
        _wait_for(lambda: focus.position_um() > 990.0, "the stabilize to start")
        fake.stabilizer.fail_next = True
        with caplog.at_level(logging.ERROR, logger="smc.hardware.zeiss_mtb"):
            close_core(core)
    assert any(f"{DF2}: Stop at unload failed" in m for m in caplog.messages)
    # Not stopped, so it ran to the plane; the unload waited for it.
    assert fake.axes["MTBFocus"].position_um() == pytest.approx(1000.0)
