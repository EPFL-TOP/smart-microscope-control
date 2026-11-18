"""``MMFocusStabilizer`` on the Zeiss fake stand: DF2 through the ``Autofocus`` capability (m4 §11.6).

Every call goes through the facade, as a tool's does, on the profile's own
``.cfg`` over ``FakeMTB``. DF2 corrects through the focus drive (m4 §11.1), so
the fake's stabilize moves ``Focus``; the stand opens with it at 1000 µm.
"Nothing sent" is read on the fake's record of DF2's calls. A test whose
stabilize runs slowly joins it before it ends (FM-44).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

import pytest

from smc.hardware import Microscope
from smc.hardware.backends import mm_stabilizer
from smc.hardware.backends.mm_stabilizer import MMFocusStabilizer
from smc.hardware.backends.zeiss_mtb.api import MTBError
from smc.hardware.capabilities import (
    Autofocus,
    FocusReference,
    FocusStabilized,
    ObjectiveTurret,
    ZStage,
)
from smc.hardware.errors import (
    AutofocusExcursionError,
    AutofocusUnsupportedError,
    HardwareError,
    SafetyRefusedError,
    StopRefusedError,
)
from smc.hardware.profile import SafetySection
from smc.testing.fake_mtb import FakeMTB

DF2 = "DefiniteFocus"
MTB_ID = "MTBFocusStabilizer2"
Z0 = 1000.0
TEN_X = "Fluar 10x/0.50"
FORTY_X = "Plan-Apochromat 40x/0.95"
#: The calls that reach DF2 itself, not its reads.
SENT = ("learn", "restore", "stabilize_now", "stop", "stop_periodic")
JOIN_S = 5.0


def _sent(fake: FakeMTB) -> list[tuple[object, ...]]:
    return [c for c in fake.calls if c[0] == MTB_ID and c[1] in SENT]


def _wait_for(condition: Callable[[], bool], what: str) -> None:
    deadline = time.monotonic() + JOIN_S
    while not condition():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.01)


def _with_safety(stand: Microscope, **safety: object) -> Microscope:
    """Another facade over the stand's core, with these ``[safety]`` keys; never closed."""
    merged = SafetySection.model_validate(
        {**stand.profile.safety.model_dump(), **safety}
    )
    return Microscope.from_core(
        stand.core, stand.profile.model_copy(update={"safety": merged})
    )


class _Thread:
    """Run ``target`` in a thread and keep what it returned or raised."""

    def __init__(self, target: Callable[[], object]) -> None:
        self.result: object = None
        self.error: BaseException | None = None
        self._target = target
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            self.result = self._target()
        except BaseException as exc:
            self.error = exc

    def join(self) -> None:
        self._thread.join(timeout=JOIN_S)
        assert not self._thread.is_alive(), "thread did not finish: deadlock?"


@pytest.fixture
def stand(zeiss_fake_microscope: Microscope) -> Microscope:
    return zeiss_fake_microscope


def test_stabilize_now_returns_to_the_learned_plane(
    stand: Microscope, fake_mtb: FakeMTB, caplog: pytest.LogCaptureFixture
) -> None:
    af, z = stand.require(Autofocus), stand.require(ZStage)
    reference = af.learn_reference()
    assert reference.device == DF2
    assert reference.objective == TEN_X
    assert reference.data != ""
    z.move_by_um(-5.0)
    with caplog.at_level(logging.INFO, logger="smc.hardware.backends.mm_stabilizer"):
        done = af.stabilize_now()
    assert abs(z.position_um() - Z0) <= 0.1
    assert done == FocusStabilized(
        reference, pytest.approx(Z0 - 5.0), pytest.approx(Z0)
    )
    # DF2 held it already: learned here, so no restore was sent.
    assert [c[1] for c in _sent(fake_mtb)] == ["learn", "stabilize_now"]
    assert (
        f"autofocus {DF2}: stabilize_now moved Z from 995.000 to 1000.000 µm "
        "(+5.000 µm)"
    ) in caplog.messages


def test_stabilize_now_without_a_reference_is_refused_and_sends_nothing(
    stand: Microscope, fake_mtb: FakeMTB
) -> None:
    with pytest.raises(SafetyRefusedError) as refused:
        stand.require(Autofocus).stabilize_now()
    assert refused.value.how_to_force == ""
    assert "no reference was given and none is held" in refused.value.reason
    assert _sent(fake_mtb) == []


def test_a_reference_from_another_objective_is_refused(
    stand: Microscope, fake_mtb: FakeMTB
) -> None:
    # FM-223: DF2 learned the plane through another objective.
    af = stand.require(Autofocus)
    reference = af.learn_reference()
    assert stand.require(ObjectiveTurret).select(FORTY_X, confirm=True) == FORTY_X
    before = _sent(fake_mtb)
    with pytest.raises(SafetyRefusedError) as refused:
        af.stabilize_now(reference)
    assert refused.value.how_to_force == ""
    assert (
        f"learned through {TEN_X!r}, and {FORTY_X!r} is in place"
        in refused.value.reason
    )
    assert _sent(fake_mtb) == before
    # Unknown on either side is no mismatch: a reference with no objective goes.
    unknown = FocusReference(DF2, reference.data, None)
    assert af.stabilize_now(unknown).reference == unknown


def test_a_reference_of_another_device_is_refused(
    stand: Microscope, fake_mtb: FakeMTB
) -> None:
    af = stand.require(Autofocus)
    reference = af.learn_reference()
    before = _sent(fake_mtb)
    with pytest.raises(SafetyRefusedError, match="learned by 'PFS', not by this"):
        af.stabilize_now(FocusReference("PFS", reference.data, reference.objective))
    with pytest.raises(SafetyRefusedError, match="the reference is empty"):
        af.stabilize_now(FocusReference(DF2, "", reference.objective))
    assert _sent(fake_mtb) == before


def test_piezo_focus_usage_on_refuses_a_stabilize(
    stand: Microscope, fake_mtb: FakeMTB
) -> None:
    # DF2 would correct through the piezo, which no role reads (m4 §11.1).
    af = stand.require(Autofocus)
    af.learn_reference()
    fake_mtb.stabilizer.piezo_focus_usage_on = True
    before = _sent(fake_mtb)
    with pytest.raises(SafetyRefusedError, match="PiezoFocusUsage reads On") as refused:
        af.stabilize_now()
    assert refused.value.how_to_force == ""
    assert _sent(fake_mtb) == before


def test_a_stabilize_beyond_the_excursion_bound_raises_and_leaves_z(
    stand: Microscope,
) -> None:
    facade = _with_safety(stand, max_engage_excursion_um=1.0)
    af, z = facade.require(Autofocus), facade.require(ZStage)
    reference = af.learn_reference()
    z.move_by_um(-5.0)
    with pytest.raises(AutofocusExcursionError) as tripped:
        af.stabilize_now()
    assert (tripped.value.z_start_um, tripped.value.z_um) == (
        pytest.approx(Z0 - 5.0),
        pytest.approx(Z0),
    )
    assert "beyond [safety] max_engage_excursion_um (1 µm)" in str(tripped.value)
    assert "the layer left it there" in str(tripped.value)
    assert not isinstance(tripped.value, SafetyRefusedError)  # FM-64: it moved
    assert z.position_um() == pytest.approx(Z0)
    # The Z soft limits bound it too.
    limited = _with_safety(stand, z_soft_limits_um=(Z0 - 10.0, Z0 - 1.0))
    z.move_by_um(-5.0)
    with pytest.raises(AutofocusExcursionError, match="outside the Z soft limits"):
        limited.require(Autofocus).stabilize_now(reference)
    assert z.position_um() == pytest.approx(Z0)


def test_the_z_guard_bounds_a_move_either_way_and_a_z_it_cannot_read(
    stand: Microscope, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Which way DF2 moves Z is not measured (m4 §12): the guard is two-sided.
    af, z = stand.require(Autofocus), stand.require(ZStage)
    reference = af.learn_reference()
    bounded = _with_safety(stand, max_engage_excursion_um=1.0)
    z.move_by_um(5.0)
    with pytest.raises(AutofocusExcursionError, match=r"by 5\.000 µm, beyond"):
        bounded.require(Autofocus).stabilize_now(reference)
    assert z.position_um() == pytest.approx(Z0)
    # A landing below the lower soft limit.
    above = _with_safety(stand, z_soft_limits_um=(Z0 + 1.0, Z0 + 10.0))
    z.move_by_um(5.0)
    with pytest.raises(AutofocusExcursionError, match="outside the Z soft limits"):
        above.require(Autofocus).stabilize_now(reference)
    # FM-167: a Z that reads NaN after the stabilize is beyond any bound.
    bounded_z = bounded.require(ZStage)
    real = bounded_z.position_um
    reads: list[float] = []

    def nan_after_the_first() -> float:
        reads.append(real())
        return reads[0] if len(reads) == 1 else float("nan")

    monkeypatch.setattr(bounded_z, "position_um", nan_after_the_first)
    with pytest.raises(AutofocusExcursionError, match="to nan µm"):
        bounded.require(Autofocus).stabilize_now(reference)


def test_every_refusal_comes_before_the_restore(
    stand: Microscope, fake_mtb: FakeMTB
) -> None:
    # DF2 holds another plane, so each stabilize below would need a restore:
    # a refusal after it would already have replaced what DF2 holds.
    af = stand.require(Autofocus)
    first = af.learn_reference()
    af.learn_reference()
    before = _sent(fake_mtb)
    with pytest.raises(SafetyRefusedError, match="learned by 'PFS'"):
        af.stabilize_now(FocusReference("PFS", first.data, first.objective))
    fake_mtb.stabilizer.piezo_focus_usage_on = True
    with pytest.raises(SafetyRefusedError, match="PiezoFocusUsage reads On"):
        af.stabilize_now(first)
    fake_mtb.stabilizer.piezo_focus_usage_on = False
    assert stand.require(ObjectiveTurret).select(FORTY_X, confirm=True) == FORTY_X
    with pytest.raises(SafetyRefusedError, match=f"learned through {TEN_X!r}"):
        af.stabilize_now(first)
    assert _sent(fake_mtb) == before
    # An objective that cannot be read now is no mismatch (m4 §11.6).
    blind = MMFocusStabilizer(
        stand.core,
        DF2,
        stand._executor,
        stand._safety,
        objective_label=lambda: None,
        focus=lambda: stand.get(ZStage),
    )
    assert blind.stabilize_now(first).reference == first
    assert [c[1] for c in _sent(fake_mtb)][len(before) :] == [
        "restore",
        "stabilize_now",
    ]


def test_a_failed_learn_leaves_no_reference_held(
    stand: Microscope, fake_mtb: FakeMTB
) -> None:
    af = stand.require(Autofocus)
    first = af.learn_reference()
    fake_mtb.stabilizer.refuse_next = True
    with pytest.raises(HardwareError, match="answered False"):
        af.learn_reference()
    with pytest.raises(SafetyRefusedError, match="none is held"):
        af.stabilize_now()
    # DF2's own record went with it: the first reference is restored first.
    af.stabilize_now(first)
    assert [c[1] for c in _sent(fake_mtb)][-2:] == ["restore", "stabilize_now"]


def test_disengage_switches_periodic_off_even_when_its_read_fails(
    stand: Microscope, fake_mtb: FakeMTB, monkeypatch: pytest.MonkeyPatch
) -> None:
    af = stand.require(Autofocus)
    fake_mtb.stabilizer.periodic_on = True
    af.disengage()
    assert [c[1] for c in _sent(fake_mtb)] == ["stop_periodic"]
    assert af.is_engaged() is False

    def unreadable() -> bool:
        raise MTBError("GetStabilizePeriodicallyOnOff timed out (fake)")

    # A read that fails counts as on (FM-19): the Off is sent anyway.
    monkeypatch.setattr(fake_mtb.stabilizer, "periodic", unreadable)
    af.disengage()
    assert [c[1] for c in _sent(fake_mtb)] == ["stop_periodic", "stop_periodic"]


def test_a_stop_that_fails_is_raised_after_periodic_is_switched_off(
    stand: Microscope, fake_mtb: FakeMTB
) -> None:
    af, z = stand.require(Autofocus), stand.require(ZStage)
    af.learn_reference()
    z.move_by_um(-5.0)
    fake_mtb.stabilizer.move_s = 0.3
    focus = fake_mtb.axes["MTBFocus"]
    worker = _Thread(af.stabilize_now)
    try:
        _wait_for(lambda: focus.position_um() > Z0 - 5.0, "the stabilize to start")
        fake_mtb.stabilizer.periodic_on = True
        fake_mtb.stabilizer.fail_next = True  # the Stop raises
        with pytest.raises(HardwareError) as failed:
            af.disengage()
    finally:
        worker.join()
        fake_mtb.stabilizer.move_s = 0.0
    assert "the switch-off failed (Action Stop:" in str(failed.value)
    assert "Stop raised (fake)" in str(failed.value)
    assert "stop_periodic" in [c[1] for c in _sent(fake_mtb)]
    assert fake_mtb.stabilizer.periodic_on is False


def test_an_emergency_stop_that_df2_does_not_honour_is_a_failure(
    stand: Microscope, fake_mtb: FakeMTB, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Whether DF2 ends a synchronous call from another thread is not known
    # (m4 §12): the stop says it switched DF2 off only when Action reads Idle.
    monkeypatch.setattr(mm_stabilizer, "STOP_CONFIRM_S", 0.2)
    af, z = stand.require(Autofocus), stand.require(ZStage)
    af.learn_reference()
    z.move_by_um(-5.0)
    fake_mtb.stabilizer.move_s = 1.0
    fake_mtb.stabilizer.stop_ignored = True
    focus = fake_mtb.axes["MTBFocus"]
    worker = _Thread(af.stabilize_now)
    try:
        _wait_for(lambda: focus.position_um() > Z0 - 5.0, "the stabilize to start")
        with pytest.raises(HardwareError) as failed:
            stand.stop()
    finally:
        worker.join()
        fake_mtb.stabilizer.move_s = 0.0
        fake_mtb.stabilizer.stop_ignored = False
    assert not isinstance(failed.value, StopRefusedError)
    assert f"the emergency stop failed for Autofocus {DF2!r}" in str(failed.value)
    assert "Action still reads StabilizeNow 0.2 s after the Stop" in str(failed.value)
    stand.resume()


def test_a_failed_restore_leaves_no_reference_held(
    stand: Microscope, fake_mtb: FakeMTB
) -> None:
    af, z = stand.require(Autofocus), stand.require(ZStage)
    first = af.learn_reference()
    z.move_by_um(-5.0)
    af.learn_reference()  # DF2 now holds another plane
    fake_mtb.stabilizer.refuse_next = True
    with pytest.raises(HardwareError, match="was not restored"):
        af.stabilize_now(first)
    # No stabilize after the failed restore; the layer's give-up sent a Stop.
    assert [c[1] for c in _sent(fake_mtb)] == ["learn", "learn", "restore", "stop"]
    with pytest.raises(SafetyRefusedError, match="none is held"):
        af.stabilize_now()
    # Restored, it is held again, and DF2 goes to its plane.
    assert af.stabilize_now(first).z_um == pytest.approx(Z0)
    assert af.stabilize_now().reference == first


def test_dry_run_learns_and_stabilizes_nothing(
    stand: Microscope, fake_mtb: FakeMTB
) -> None:
    dry = Microscope.from_core(stand.core, stand.profile, dry_run=True)
    af = dry.require(Autofocus)
    reference = af.learn_reference()
    assert reference == FocusReference(DF2, "", TEN_X)
    done = af.stabilize_now()
    assert done == FocusStabilized(reference, pytest.approx(Z0), pytest.approx(Z0))
    assert _sent(fake_mtb) == []
    # A reference learned in a dry run is refused outside one.
    with pytest.raises(SafetyRefusedError, match="the reference is empty"):
        stand.require(Autofocus).stabilize_now(reference)


def test_engage_is_unsupported_and_sends_nothing(
    stand: Microscope, fake_mtb: FakeMTB
) -> None:
    dry = Microscope.from_core(stand.core, stand.profile, dry_run=True)
    for af in (stand.require(Autofocus), dry.require(Autofocus)):
        with pytest.raises(AutofocusUnsupportedError) as refused:
            af.engage()
        assert str(refused.value) == (
            f"autofocus {DF2}: engage() is not supported: it holds focus one shot "
            f"at a time: learn_reference(), stabilize_now(); nothing was sent"
        )
        assert not isinstance(refused.value, SafetyRefusedError)
        assert af.is_engaged() is False
        assert af.is_locked() is False
        assert af.is_in_range() is None
    assert _sent(fake_mtb) == []


def test_a_non_zero_offset_is_unsupported_and_zero_sends_nothing(
    stand: Microscope, fake_mtb: FakeMTB
) -> None:
    af = stand.require(Autofocus)
    assert (af.offset_native(), af.offset_unit()) == (0.0, "µm")
    assert af.offset_limits_native() == (0.0, 0.0)
    assert af.set_offset_native(0.0) == 0.0
    assert af.offset_by_native(0.0) == 0.0
    for call in (
        lambda: af.set_offset_native(1.0),
        lambda: af.offset_by_native(-0.5, force=True),
        lambda: af.set_offset_native(float("nan")),
    ):
        with pytest.raises(AutofocusUnsupportedError, match="nothing was sent"):
            call()
    assert _sent(fake_mtb) == []
    # focus_by_um is a Z move, its jog guard included.
    step = af.focus_by_um(-1.0)
    assert (step.via, step.offset_native) == ("z", None)
    assert step.z_um == pytest.approx(Z0 - 1.0)
    with pytest.raises(SafetyRefusedError, match="exceeds the Z jog limit"):
        af.focus_by_um(-150.0)


@pytest.mark.wallclock
def test_disengage_stops_a_running_stabilize_while_halted(
    stand: Microscope, fake_mtb: FakeMTB
) -> None:
    af, z = stand.require(Autofocus), stand.require(ZStage)
    af.learn_reference()
    z.move_by_um(-5.0)
    fake_mtb.stabilizer.move_s = 5.0
    focus = fake_mtb.axes["MTBFocus"]
    worker = _Thread(af.stabilize_now)
    try:
        _wait_for(lambda: focus.position_um() > Z0 - 5.0, "the stabilize to start")
        assert stand.state().moving == (f"autofocus {DF2}",)
        started = time.perf_counter()
        # The emergency stop: halt, the autofocus switched off first, then the
        # stages, whose stops MTB refuses, as the profile declares.
        with pytest.raises(StopRefusedError):
            stand.stop()
        took = time.perf_counter() - started
        worker.join()
    finally:
        fake_mtb.stabilizer.move_s = 0.0
    assert took < 1.0
    assert isinstance(worker.error, HardwareError)
    assert "after a Stop was sent during it" in str(worker.error)
    assert Z0 - 5.0 < focus.position_um() < Z0 - 2.0
    assert ("stop" in [c[1] for c in _sent(fake_mtb)]) is True
    assert stand.require(Autofocus).is_engaged() is False
    stand.resume()


@pytest.mark.wallclock
def test_reads_do_not_wait_behind_a_stabilize(
    stand: Microscope, fake_mtb: FakeMTB
) -> None:
    af, z = stand.require(Autofocus), stand.require(ZStage)
    af.learn_reference()
    z.move_by_um(-5.0)
    fake_mtb.stabilizer.move_s = 0.5
    focus = fake_mtb.axes["MTBFocus"]
    worker = _Thread(af.stabilize_now)
    try:
        _wait_for(lambda: focus.position_um() > Z0 - 5.0, "the stabilize to start")
        started = time.perf_counter()
        engaged = af.is_engaged()
        here = z.position_um()
        state = stand.state()
        took = time.perf_counter() - started
    finally:
        worker.join()
    assert took < 0.1
    assert engaged is False
    assert Z0 - 5.0 < here < Z0
    assert state.moving == (f"autofocus {DF2}",)
    assert state.errors == []
    assert worker.error is None


def test_a_z_move_switches_periodic_off_and_does_not_switch_it_on(
    stand: Microscope, fake_mtb: FakeMTB, caplog: pytest.LogCaptureFixture
) -> None:
    # Another MTB client (ZEN) switched periodic stabilisation on: it would fight
    # the move (m4 §11.6), and smc did not switch it on, so it stays off.
    fake_mtb.stabilizer.periodic_on = True
    af, z = stand.require(Autofocus), stand.require(ZStage)
    assert af.is_engaged() is True
    with caplog.at_level(logging.WARNING, logger="smc.hardware.backends.mm_stabilizer"):
        landed = z.move_by_um(-1.0)
    assert landed == pytest.approx(Z0 - 1.0)
    assert [c[1] for c in _sent(fake_mtb)] == ["stop_periodic"]
    assert af.is_engaged() is False
    assert any(
        "periodic stabilisation was on (another client left it on) and was switched "
        "off for z: move_by -1.0 µm" in m
        for m in caplog.messages
    )
