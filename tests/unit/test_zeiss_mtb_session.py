"""The process's one MTB session: one login, a logout only at exit, after the light (m4 §6, §8)."""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from smc.hardware import Microscope
from smc.hardware.backends.zeiss_mtb import session
from smc.hardware.backends.zeiss_mtb.api import MTBError, MTBSessionError
from smc.hardware.core import close_core, open_core
from smc.testing.fake_mtb import FakeMTB, fake_stand_config

CFG = Path(__file__).resolve().parents[2] / "profiles" / "zeiss-observer7.cfg"


def test_shared_logs_in_once_per_process() -> None:
    fake = FakeMTB()
    with session.use_api_factory(lambda: fake):
        first = session.shared()
        assert session.shared() is first
        first.axis("MTBFocus")
        session.shared().changer("MTBObjectiveChanger")
        assert fake.logins == 1
        assert fake.logouts == 0
    assert fake.logouts == 1


def test_components_are_looked_up_once_and_cached() -> None:
    fake = FakeMTB()
    with session.use_api_factory(lambda: fake):
        mtb = session.shared()
        assert mtb.axis("MTBFocus") is mtb.axis("MTBFocus")
        assert mtb.changer("MTBTLShutter") is mtb.changer("MTBTLShutter")
        assert mtb.led_controller("MTBFLLEDController") is mtb.led_controller(
            "MTBFLLEDController"
        )


def test_concurrent_first_calls_log_in_once() -> None:
    fake = FakeMTB()
    barrier = threading.Barrier(8)
    found: list[session.MTBSession] = []
    errors: list[BaseException] = []

    def first_call() -> None:
        try:
            barrier.wait(timeout=10)
            found.append(session.shared())
        except BaseException as exc:  # reported below, never lost in the thread
            errors.append(exc)

    with session.use_api_factory(lambda: fake):
        threads = [threading.Thread(target=first_call) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
            assert not thread.is_alive()
        assert errors == []
        assert len(found) == 8
        assert all(s is found[0] for s in found)
        assert fake.logins == 1


def test_a_failed_login_leaves_no_session_and_the_next_call_tries_again() -> None:
    fake = FakeMTB()
    fake.fail_next_login = True
    with session.use_api_factory(lambda: fake):
        with pytest.raises(MTBError, match="the stand is off"):
            session.shared()
        assert fake.logins == 0
        assert session.shared() is not None
        assert fake.logins == 1


def test_a_login_failure_that_is_no_mtb_error_is_a_session_error() -> None:
    def broken() -> FakeMTB:
        raise OSError("the DLL would not load")

    with (
        session.use_api_factory(broken),
        pytest.raises(MTBSessionError, match="the DLL would not load"),
    ):
        session.shared()


def test_closing_the_microscope_does_not_log_out(
    zeiss_fake_microscope: Microscope, fake_mtb: FakeMTB
) -> None:
    zeiss_fake_microscope.close()
    assert fake_mtb.logins == 1
    assert fake_mtb.logouts == 0
    # Connect again in the same process: the session still serves.
    session.shared().axis("MTBFocus").position_um()
    assert fake_mtb.logins == 1


def test_after_logout_every_call_is_refused_with_a_session_error() -> None:
    fake = FakeMTB()
    with session.use_api_factory(lambda: fake):
        mtb = session.shared()
        axis = mtb.axis("MTBFocus")
        changer = mtb.changer("MTBTLShutter")
        led = mtb.led_controller("MTBFLLEDController").leds()[0]
        mtb._close_at_exit()
        assert fake.logouts == 1
        for call in (
            session.shared,
            lambda: mtb.axis("MTBFocus"),
            axis.position_um,
            lambda: axis.move_um(1.0, 1000),
            changer.position,
            led.is_on,
            lambda: led.set_on(True, 1000),
        ):
            with pytest.raises(MTBSessionError, match="cannot log in twice"):
                call()
        assert fake.logins == 1


def test_use_api_factory_refuses_while_another_session_exists() -> None:
    first, second = FakeMTB(), FakeMTB()
    with session.use_api_factory(lambda: first):
        session.shared()
        with (
            pytest.raises(MTBSessionError, match="another factory"),
            session.use_api_factory(lambda: second),
        ):
            pass  # pragma: no cover - the block must not be entered
        assert second.logins == 0
    # The first block's exit forgot its session: the next one starts clean.
    with session.use_api_factory(lambda: second):
        session.shared()
        assert second.logins == 1


def test_use_api_factory_leaves_no_exit_handler_behind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered: list[object] = []
    monkeypatch.setattr(session.atexit, "register", registered.append)
    monkeypatch.setattr(session.atexit, "unregister", registered.remove)
    with session.use_api_factory(FakeMTB):
        session.shared()
        assert len(registered) == 1
    assert registered == []


def test_the_exit_handler_runs_every_darkener_even_when_one_fails_then_logs_out(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake = FakeMTB()
    ran: list[str] = []

    def fails() -> None:
        ran.append("fails")
        raise MTBError("the shutter did not move")

    with session.use_api_factory(lambda: fake):
        mtb = session.shared()
        mtb.register_darkener("A", fails)
        mtb.register_darkener("B", lambda: ran.append("B"))
        controller = mtb.led_controller("MTBFLLEDController")
        fake.colibri.led_list[2].on = True
        fake.colibri.led_list[0].fail_off = True
        with caplog.at_level(logging.ERROR, logger="smc.hardware.zeiss_mtb"):
            mtb._close_at_exit()
    assert ran == ["fails", "B"]
    assert fake.colibri.lit() == []
    assert controller is not None
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert any(
        "LIGHT MAY STILL BE ON: A: the shutter did not move" in e for e in errors
    )
    assert any("LIGHT MAY STILL BE ON" in e and "385" in e for e in errors)
    calls = fake.calls
    logout = calls.index(("FakeMTB", "logout"))
    switched_off = [
        i for i, c in enumerate(calls) if c[1] == "set_on" and c[2] is False
    ]
    assert len(switched_off) == 6
    assert max(switched_off) < logout


def test_exit_closes_a_tl_shutter_left_open_through_its_own_darkener(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    fake = FakeMTB()
    with session.use_api_factory(lambda: fake):
        core = open_core(fake_stand_config(CFG, tmp_path))
        core.setShutterOpen("TLShutter", True)
        core.setProperty("Colibri", "LED", "475nm")
        core.setShutterOpen("Colibri", True)
        with caplog.at_level(logging.ERROR, logger="smc.hardware.zeiss_mtb"):
            session.shared()._close_at_exit()
        assert fake.changers["MTBTLShutter"].position() == 2
        assert fake.colibri.lit() == []
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    close_core(core)


def test_a_ctrl_c_in_a_darkener_still_cuts_the_light_and_logs_out_then_goes_on() -> (
    None
):
    fake = FakeMTB()
    ran: list[str] = []

    def interrupted() -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):  # noqa: PT012, SIM117 - the block's exit raises it
        with session.use_api_factory(lambda: fake):
            mtb = session.shared()
            mtb.led_controller("MTBFLLEDController")
            fake.colibri.led_list[3].on = True
            mtb.register_darkener("A", interrupted)
            mtb.register_darkener("B", lambda: ran.append("B"))
    assert ran == ["B"]
    assert fake.colibri.lit() == []
    assert fake.logouts == 1
    # The block's exit forgot the session even though it raised.
    with session.use_api_factory(FakeMTB):
        session.shared()


def test_an_unregistered_darkener_does_not_run_at_exit() -> None:
    ran: list[str] = []
    with session.use_api_factory(FakeMTB):
        mtb = session.shared()
        mtb.register_darkener("A", lambda: ran.append("A"))
        mtb.unregister_darkener("A")
        mtb.unregister_darkener("never registered")
    assert ran == []


#: Seconds after which the child dumps every thread's stack and exits (#538):
#: below the parent's own bound, so a hang names its stuck thread in the
#: failure instead of a bare ``TimeoutExpired``.
_CHILD_DUMP_AFTER_S = 45

_CHILD = (
    f"""
import faulthandler

faulthandler.dump_traceback_later({_CHILD_DUMP_AFTER_S}, exit=True)
"""
    + """
import atexit
import json
import sys
from pathlib import Path

from smc.hardware import Microscope
from smc.hardware.backends.zeiss_mtb import session
from smc.hardware.profile import Profile
from smc.testing.fake_mtb import FakeMTB, fake_stand_config

out = Path(sys.argv[1])
fake = FakeMTB()


def dump():
    out.write_text(json.dumps([list(map(str, c)) for c in fake.calls]), encoding="utf-8")


# Registered before the first login, so atexit's LIFO order runs it last.
atexit.register(dump)
swap = session.use_api_factory(lambda: fake)
swap.__enter__()  # never left: the process exits with the session open
profile = Profile.load(Path(sys.argv[2]))
cfg = fake_stand_config(profile.config_path(), out.parent)
mm = profile.micromanager.model_copy(update={"config": str(cfg)})
stand = Microscope.open(profile.model_copy(update={"micromanager": mm}))
stand.core.setProperty("Colibri", "LED", "475nm")
stand.core.setShutterOpen("Colibri", True)
stand.core.setShutterOpen("TLShutter", True)
assert fake.colibri.lit() == [475], fake.colibri.lit()
assert fake.changers["MTBTLShutter"].position() == 1  # "Open"
"""
)


def test_process_exit_darkens_the_light_then_logs_out(tmp_path: Path) -> None:
    script = tmp_path / "child.py"
    script.write_text(_CHILD, encoding="utf-8")
    calls_file = tmp_path / "calls.json"
    profile = Path(__file__).resolve().parents[2] / "profiles" / "zeiss-observer7.toml"
    stdout, stderr = tmp_path / "stdout.txt", tmp_path / "stderr.txt"
    with stdout.open("wb") as out, stderr.open("wb") as err:
        child = subprocess.Popen(
            [sys.executable, str(script), str(calls_file), str(profile)],
            stdout=out,
            stderr=err,
        )
        try:
            code = child.wait(timeout=60)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=10)
            raise
    assert code == 0, stderr.read_text(encoding="utf-8", errors="replace")
    import json

    calls = [tuple(c) for c in json.loads(calls_file.read_text(encoding="utf-8"))]
    logout = calls.index(("FakeMTB", "logout"))
    lit = max(i for i, c in enumerate(calls) if c == ("LED475", "set_on", "True"))
    for nm in (385, 430, 475, 511, 567, 630):
        offs = [i for i, c in enumerate(calls) if c == (f"LED{nm}", "set_on", "False")]
        assert any(lit < i < logout for i in offs), (
            f"LED {nm} not switched off before logout"
        )
    opened = calls.index(("MTBTLShutter", "move", "1", "30000"))
    closed = [
        i for i, c in enumerate(calls) if c == ("MTBTLShutter", "move", "2", "30000")
    ]
    assert any(opened < i < logout for i in closed), (
        "TL shutter not closed before logout"
    )
    # The darkeners worked: nothing says the light may still be on.
    assert "LIGHT MAY STILL BE ON" not in stderr.read_text(
        encoding="utf-8", errors="replace"
    )
