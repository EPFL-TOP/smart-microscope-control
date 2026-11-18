"""The ``Autofocus`` contract, on every backend (m2 §3.2, §3.13, m4 §11.7).

The offset moves at most 1.0 from where it started (``envelope.offset_native``),
in its own units and in whole units (the Ti2-E's ``PFSOffset`` holds nothing
finer, FM-119) and with the autofocus off, except in the locked
``focus_by_um`` case, which moves it by 1.0. Z moves only below its start, by
at most 2 µm, as in the ``ZStage`` contract. A case that needs a lock skips on
``hardware`` when none comes ("no lock on this sample"), or when the engage
check refuses it (m2 §12.4: an offset the operator has not locked at the
stand); on ``fake`` and ``demo`` the lock comes at once, so no lock fails it.

An autofocus holds focus continuously (PFS: ``engage``, an offset) or one shot
at a time (DF2: ``learn_reference``, ``stabilize_now``), and refuses the other
way with ``AutofocusUnsupportedError``. A case of the continuous way skips on
it where the stand may lack that way (``zeiss-fake``, ``hardware``) and fails
on ``fake`` and ``demo``; a case of the one-shot way skips on it everywhere,
since PFS has none. One contract keeps a stand from skipping both.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from smc.hardware import Microscope
from smc.hardware.capabilities import Autofocus, Properties, ZStage
from smc.hardware.errors import AutofocusUnsupportedError, SafetyRefusedError
from smc.hardware.roles import Role

if TYPE_CHECKING:
    # For the annotations only; the values come from the fixture.
    from conftest import Envelope, Unsupported


def _offset_start(envelope: Envelope) -> float:
    assert envelope.offset_native is not None
    return envelope.offset_native


def _z_start(envelope: Envelope) -> float:
    assert envelope.z_um is not None
    return envelope.z_um


def _engage(autofocus: Autofocus, backend: str) -> bool:
    """``engage()``; on ``hardware``, a refusal of the engage check skips the case.

    On a stand that declares ``engage_drives_z_to_offset``, an offset the
    operator has not locked at the stand is refused, and so is Z outside the
    contracts' window (m2 §12.4); the skip names the reason. On ``fake`` and
    ``demo`` a refusal fails the case.

    No contract runs a first lock from below (m2 §12.10.8): it moves Z by what
    the stand decides, and the visit runs it by hand. On ``hardware`` an offset
    at its floor skips before anything is sent; an autofocus whose offset holds
    nothing but zero (DF2, limits ``(0.0, 0.0)``) has no floor to be at, and its
    ``engage()`` refuses on its own.

    ``AutofocusUnsupportedError`` is the caller's: inside
    ``unsupported.skips()``, or caught where the other way is tried.
    """
    if backend == "hardware":
        limits = autofocus.offset_limits_native()
        if (
            limits is not None
            and limits[0] < limits[1]
            and autofocus.offset_native() <= limits[0]
        ):
            pytest.skip(
                "the offset is at its floor: engaging there may be a first lock (m2 "
                "§12.10), which no contract runs"
            )
    try:
        return autofocus.engage()
    except SafetyRefusedError as refused:
        if backend != "hardware":
            raise
        pytest.skip(f"engage() refused: {refused.reason}")


def _lock(autofocus: Autofocus, backend: str) -> None:
    """Engage it; on ``hardware`` a sample with no reference skips the case."""
    if _engage(autofocus, backend):
        return
    if backend == "hardware":
        pytest.skip("no lock on this sample: focus on it by eye, then run again")
    pytest.fail("no lock, on a backend that locks at once")


def _objective(stand: Microscope) -> str | None:
    """The current objective's label, as ``offset_per_um`` is keyed; ``None`` without a turret."""
    turret = stand.roles.get(Role.objective_turret)
    if turret is None:
        return None
    return stand.require(Properties).get(turret, "Label")


def _with_factors(
    stand: Microscope, envelope: Envelope, factors: dict[str, float]
) -> Microscope:
    """Another facade over ``stand``'s core, under the envelope, with these factors.

    Never closed, as the ``microscope`` fixture's: ``stand`` owns the core, and
    the ``microscope`` fixture the test also requests puts it back. The caller
    keeps the facade in a variable for as long as it uses a capability of it: a
    capability looks its siblings up through its facade, which it holds only
    weakly, and refuses once the facade is gone (#142).
    """
    profile = envelope.profile(stand.profile)
    autofocus = profile.autofocus.model_copy(update={"offset_per_um": factors})
    facade = Microscope.from_core(
        stand.core, profile.model_copy(update={"autofocus": autofocus})
    )
    # Its own autofocus has seen nothing yet: as the fixture's, read the lock
    # once, so that the operator's is seen (m2 §12.2 D1).
    facade.require(Autofocus).is_locked()
    return facade


def test_an_autofocus_holds_focus_one_way_or_the_other(
    autofocus: Autofocus, backend: str
) -> None:
    # m4 §11.7: each way may be unsupported, never both. Without this case a
    # stand that skipped every case of both would pass with an autofocus that
    # holds focus no way at all.
    try:
        locked = _engage(autofocus, backend)
    except AutofocusUnsupportedError:
        pass
    else:
        if not locked and backend == "hardware":
            pytest.skip("no lock on this sample: focus on it by eye, then run again")
        assert locked, "no lock, on a backend that locks at once"
        return
    try:
        reference = autofocus.learn_reference()
        done = autofocus.stabilize_now(reference)
    except AutofocusUnsupportedError as exc:
        pytest.fail(
            f"this autofocus holds focus neither way: engage() and the one-shot "
            f"mode are both unsupported ({exc})"
        )
    assert done.reference == reference


def test_stabilize_now_returns_to_the_learned_plane_within_1_um(
    autofocus: Autofocus,
    z: ZStage,
    envelope: Envelope,
    unsupported: Unsupported,
) -> None:
    # The stand check that replaces the predecessor's df2_test.py (m4 §12): learn,
    # a 2 µm move away from the sample, a stabilize. The disturbance (2.0)
    # differs from the tolerance (1.0), so a stabilize that does nothing fails.
    z0, tol = _z_start(envelope), envelope.tolerance_um
    with unsupported.skips(one_shot=True):
        reference = autofocus.learn_reference()
    # Absolute: the contracts' Z jog limit is 1 µm.
    disturbed = z.move_to_um(z0 - 2.0)
    assert abs(disturbed - (z0 - 2.0)) <= tol
    done = autofocus.stabilize_now(reference)
    here = z.position_um()
    assert abs(here - z0) <= 1.0
    assert done.reference == reference
    assert done.z_before_um is not None
    assert abs(done.z_before_um - disturbed) <= tol
    assert done.z_um is not None
    assert abs(done.z_um - here) <= tol


def test_engage_returns_a_bool_and_says_the_truth(
    autofocus: Autofocus, backend: str, unsupported: Unsupported
) -> None:
    with unsupported.skips():
        locked = _engage(autofocus, backend)
    assert isinstance(locked, bool)
    if locked:
        assert autofocus.is_engaged() is True
        assert autofocus.is_locked() is True
    else:
        # False always means "no lock, and it is off" (m2 §3.2).
        assert autofocus.is_engaged() is False


def test_disengage_switches_it_off_and_twice_is_harmless(
    autofocus: Autofocus, backend: str, unsupported: Unsupported
) -> None:
    with unsupported.skips():
        _lock(autofocus, backend)
    autofocus.disengage()
    assert autofocus.is_engaged() is False
    assert autofocus.is_locked() is False
    autofocus.disengage()
    assert autofocus.is_engaged() is False


def test_is_in_range_is_a_bool_or_none(autofocus: Autofocus) -> None:
    assert autofocus.is_in_range() in (True, False, None)


def test_offset_unit_is_a_string(autofocus: Autofocus) -> None:
    assert isinstance(autofocus.offset_unit(), str)


def test_offset_round_trips_within_tolerance(
    autofocus: Autofocus, envelope: Envelope, unsupported: Unsupported
) -> None:
    o0, tol = _offset_start(envelope), envelope.offset_tolerance
    autofocus.disengage()
    with unsupported.skips():
        landed = autofocus.set_offset_native(o0 - 1.0)
    assert abs(landed - (o0 - 1.0)) <= tol
    assert abs(autofocus.offset_native() - (o0 - 1.0)) <= tol


def test_offset_jog_above_the_limit_is_refused_and_sends_nothing(
    autofocus: Autofocus, envelope: Envelope, unsupported: Unsupported
) -> None:
    # Whole units only: the Ti2-E's PFSOffset holds whole units (FM-119). From
    # start + 1, a jog of -2 is above the 1.0 limit and lands at start - 1: the
    # offset never leaves 1.0 of where it started.
    o0, tol = _offset_start(envelope), envelope.offset_tolerance
    assert envelope.max_offset_jog < 2.0
    autofocus.disengage()
    with unsupported.skips():
        autofocus.set_offset_native(o0 + 1.0)
    with pytest.raises(SafetyRefusedError) as refused:
        autofocus.offset_by_native(-2.0)
    assert refused.value.how_to_force == "pass force=True"
    # A jog just above the limit is refused too, so a guard with slack fails.
    # Refused, it sends nothing: no fraction reaches a device that holds whole
    # units.
    with pytest.raises(SafetyRefusedError):
        autofocus.offset_by_native(-1.25)
    assert abs(autofocus.offset_native() - (o0 + 1.0)) <= tol
    landed = autofocus.offset_by_native(-2.0, force=True)
    assert abs(landed - (o0 - 1.0)) <= tol


def test_the_fake_stand_holds_whole_units(
    backend: str, request: pytest.FixtureRequest
) -> None:
    # The contracts above pass on a device that holds floats too; this is what
    # keeps the fake stand the one that shows a fraction (FM-119, #207).
    if backend != "fake":
        pytest.skip("only the fake stand is configured to hold whole units")
    # After the skip: a hardware run never opens the stand for this case.
    stand: Microscope = request.getfixturevalue("stand")
    assert stand.require(Autofocus).set_offset_native(7927.5) == 7927.0


def test_offset_outside_the_soft_limits_is_refused_even_forced(
    autofocus: Autofocus, envelope: Envelope, unsupported: Unsupported
) -> None:
    o0, tol = _offset_start(envelope), envelope.offset_tolerance
    assert envelope.offset_span < 3.0
    autofocus.disengage()
    # A refusal that is no SafetyRefusedError leaves pytest.raises unmatched.
    with unsupported.skips(), pytest.raises(SafetyRefusedError) as absolute:
        autofocus.set_offset_native(o0 - 3.0)
    assert absolute.value.how_to_force == ""
    with pytest.raises(SafetyRefusedError) as relative:
        autofocus.offset_by_native(-3.0, force=True)
    assert relative.value.how_to_force == ""
    assert abs(autofocus.offset_native() - o0) <= tol


def test_offset_limits_report_the_envelope(
    autofocus: Autofocus, envelope: Envelope, unsupported: Unsupported
) -> None:
    o0, span = _offset_start(envelope), envelope.offset_span
    limits = autofocus.offset_limits_native()
    if limits == (0.0, 0.0):
        # An autofocus whose offset is zero by construction (DF2, m4 §11.6) has
        # no use for the envelope's limits; it says so by refusing any other
        # value. On a stand that holds an offset, these limits would refuse
        # the same value as a SafetyRefusedError, so nothing is ever sent.
        with unsupported.skips():
            autofocus.set_offset_native(o0 + 1.0)
        pytest.fail(f"the offset's limits read {limits}, not the envelope's")
    assert limits == (o0 - span, o0 + span)


def test_z_move_keeps_an_engaged_autofocus_engaged(
    autofocus: Autofocus,
    z: ZStage,
    envelope: Envelope,
    backend: str,
    unsupported: Unsupported,
) -> None:
    # On the fake, a Z move switches it off (the Nikon quirk, FM-12): this is
    # the interlock engaging it again.
    with unsupported.skips():
        _lock(autofocus, backend)
    z.move_to_um(_z_start(envelope) - 2)
    assert autofocus.is_engaged() is True
    assert autofocus.is_locked() is True


def test_z_move_with_keep_autofocus_false_leaves_it_off(
    autofocus: Autofocus,
    z: ZStage,
    envelope: Envelope,
    backend: str,
    unsupported: Unsupported,
) -> None:
    z0, tol = _z_start(envelope), envelope.tolerance_um
    with unsupported.skips():
        _lock(autofocus, backend)
    assert abs(z.move_to_um(z0 - 2, keep_autofocus=False) - (z0 - 2)) <= tol
    assert autofocus.is_engaged() is False


def test_focus_by_with_the_autofocus_off_moves_z(
    autofocus: Autofocus, envelope: Envelope
) -> None:
    z0, tol = _z_start(envelope), envelope.tolerance_um
    autofocus.disengage()
    step = autofocus.focus_by_um(-1.0)
    assert (step.via, step.offset_native) == ("z", None)
    assert step.z_um is not None
    assert abs(step.z_um - (z0 - 1.0)) <= tol
    assert autofocus.is_engaged() is False


def test_focus_by_while_locked_without_a_factor_is_refused(
    stand: Microscope,
    autofocus: Autofocus,
    envelope: Envelope,
    backend: str,
    unsupported: Unsupported,
) -> None:
    facade = _with_factors(stand, envelope, {})
    af = facade.require(Autofocus)
    with unsupported.skips():
        _lock(af, backend)
    o0, tol = af.offset_native(), envelope.offset_tolerance
    with pytest.raises(SafetyRefusedError) as refused:
        af.focus_by_um(-1.0)
    assert refused.value.how_to_force == ""
    assert "offset_per_um" in refused.value.reason
    assert abs(af.offset_native() - o0) <= tol


def test_focus_by_while_locked_moves_the_offset(
    stand: Microscope,
    autofocus: Autofocus,
    envelope: Envelope,
    backend: str,
    unsupported: Unsupported,
) -> None:
    objective = _objective(stand)
    if backend == "hardware":
        factor = stand.profile.autofocus.offset_per_um.get(objective or "")
        if factor is None:
            pytest.skip(
                f"the profile has no [autofocus] offset_per_um for objective "
                f"{objective!r}"
            )
        # An offset change of 1.0 in its own units, whatever the factor: the
        # contracts' offset jog limit (inclusive), and twice the tolerance, so
        # that an offset that did not move fails the check below.
        af, dz_um = autofocus, -1.0 / abs(factor)
        if abs(dz_um) > envelope.max_z_jog_um:
            pytest.skip(f"{dz_um:g} µm is above the contracts' Z jog limit")
    else:
        assert objective is not None, "fake and demo have an objective turret"
        factor = 1.0
        facade = _with_factors(stand, envelope, {objective: factor})
        af = facade.require(Autofocus)
        dz_um = -1.0
    with unsupported.skips():
        _lock(af, backend)
    o0, tol = af.offset_native(), envelope.offset_tolerance
    assert abs(dz_um * factor) > tol
    step = af.focus_by_um(dz_um)
    assert step.via == "offset"
    assert step.offset_native is not None
    assert abs(step.offset_native - (o0 + dz_um * factor)) <= tol
    assert abs(step.offset_native - o0) > tol  # it moved
    assert af.is_locked() is True


def test_dry_run_engage_sends_nothing(
    dry_autofocus: Autofocus,
    envelope: Envelope,
    backend: str,
    unsupported: Unsupported,
) -> None:
    # The engage check runs in dry-run too (m2 §12.4): read the lock first.
    dry_autofocus.is_locked()
    with unsupported.skips():
        locked = _engage(dry_autofocus, backend)
    assert locked is True  # the commanded outcome
    assert dry_autofocus.is_engaged() is envelope.autofocus_engaged


def test_engage_at_an_unseen_offset_is_refused(
    stand: Microscope, autofocus: Autofocus, envelope: Envelope
) -> None:
    # m2 §12.2 D1: on such a stand, engaging drives Z to the plane the offset
    # names, so an offset never read locked is refused before anything is sent.
    if not stand.profile.autofocus.engage_drives_z_to_offset:
        pytest.skip(
            "the profile does not declare that engaging drives Z to the offset's "
            "plane ([autofocus] engage_drives_z_to_offset)"
        )
    o0, tol = _offset_start(envelope), envelope.offset_tolerance
    # Above the floor, so never a first lock from below (m2 §12.10 D4).
    limits = autofocus.offset_limits_native()
    assert limits is None or o0 + 1.0 > limits[0]
    autofocus.disengage()
    # With PFS off, an offset change moves no Z (m2 §4.5); the microscope
    # fixture puts the offset back.
    assert abs(autofocus.set_offset_native(o0 + 1.0) - (o0 + 1.0)) <= tol
    with pytest.raises(SafetyRefusedError) as refused:
        autofocus.engage()
    assert refused.value.how_to_force == ""
    assert "engage_drives_z_to_offset" in refused.value.reason
    assert autofocus.is_engaged() is False


def test_dry_run_offset_change_returns_the_command(
    dry_autofocus: Autofocus, envelope: Envelope, unsupported: Unsupported
) -> None:
    o0, tol = _offset_start(envelope), envelope.offset_tolerance
    with unsupported.skips():
        commanded = dry_autofocus.set_offset_native(o0 - 1.0)
    assert commanded == o0 - 1.0
    assert abs(dry_autofocus.offset_native() - o0) <= tol
