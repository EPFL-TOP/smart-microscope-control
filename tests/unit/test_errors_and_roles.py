import dataclasses
import pickle

import pytest

from smc.hardware.core import CoreError
from smc.hardware.errors import (
    AutofocusExcursionError,
    AutofocusLostError,
    CapabilityMissingError,
    HardwareError,
    SafetyRefusedError,
    StopNote,
    StopOutcome,
    StopRefusedError,
    note_stop,
    stop_notes,
)
from smc.hardware.roles import DeviceInfo, Role


def test_role_values_are_the_profile_keys() -> None:
    assert Role("xy_stage") is Role.xy_stage
    assert {r.value for r in Role} == {r.name for r in Role}


def test_device_info_is_immutable() -> None:
    dev = DeviceInfo(label="XY", type="XYStage")
    with pytest.raises(AttributeError):
        dev.label = "other"  # type: ignore[misc]


def test_core_error_is_a_hardware_error() -> None:
    assert issubclass(CoreError, HardwareError)


def test_capability_missing_names_the_role_and_what_is_available() -> None:
    err = CapabilityMissingError("XYStage", Role.xy_stage, ("camera", "focus"))
    assert "'xy_stage'" in str(err)
    assert "camera, focus" in str(err)
    assert "[roles.assign]" in str(err)


def test_a_hint_replaces_the_roles_advice() -> None:
    # m2 §8.6: Channels needs a config group, not a role.
    err = CapabilityMissingError(
        "Channels",
        None,
        ("Camera", "System"),
        hint="no config group is named for channels; name one in the profile's "
        "[channels] group",
    )
    assert str(err) == (
        "Channels is not available: no config group is named for channels; name "
        "one in the profile's [channels] group (available: Camera, System)."
    )
    assert "[roles.assign]" not in str(err)


def test_capability_missing_survives_a_worker_process() -> None:
    for error in (
        CapabilityMissingError("XYStage", Role.xy_stage, ("camera",)),
        CapabilityMissingError("Channels", None, ("Camera",), hint="name a group"),
    ):
        copy = pickle.loads(pickle.dumps(error))
        assert (str(copy), copy.role, copy.available, copy.hint) == (
            str(error),
            error.role,
            error.available,
            error.hint,
        )


def test_safety_refused_carries_how_to_force() -> None:
    err = SafetyRefusedError(
        "jog of 9000 µm exceeds the 5000 µm guard", "pass force=True"
    )
    assert err.how_to_force == "pass force=True"
    assert "pass force=True" in str(err)


def test_autofocus_lost_is_a_hardware_error_and_not_a_refusal() -> None:
    # FM-64: the move was made, so a tool that skips a target on a refusal
    # must not catch this one and carry on.
    assert issubclass(AutofocusLostError, HardwareError)
    assert not issubclass(AutofocusLostError, SafetyRefusedError)


def test_autofocus_lost_survives_a_worker_process() -> None:
    error = AutofocusLostError("z: moved, no lock", 12.0, False)
    copy = pickle.loads(pickle.dumps(error))
    assert (str(copy), copy.z_um, copy.in_range) == ("z: moved, no lock", 12.0, False)


def test_autofocus_excursion_error_pickles_with_its_positions() -> None:
    error = AutofocusExcursionError("autofocus PFS: Z went", 4269.44, 4504.76)
    copy = pickle.loads(pickle.dumps(error))
    assert type(copy) is AutofocusExcursionError
    assert (str(copy), copy.z_start_um, copy.z_um) == (
        "autofocus PFS: Z went",
        4269.44,
        4504.76,
    )


def test_autofocus_excursion_error_is_not_a_safety_refusal() -> None:
    # FM-64: Z moved, so a tool that skips a target on a refusal must not
    # catch this one and carry on (m2 §12.2 D2).
    assert issubclass(AutofocusExcursionError, HardwareError)
    assert not issubclass(AutofocusExcursionError, SafetyRefusedError)


# --- a stop the device refuses (m2 §11.4) -------------------------------------


def test_stop_refused_error_says_the_move_runs_to_its_target() -> None:
    error = StopRefusedError(("xy_stage XYStage", "z ZDrive", "autofocus_offset PFS"))
    assert str(error) == (
        "`xy_stage XYStage`, `z ZDrive` and `autofocus_offset PFS` refused the stop, "
        "as the profile declares ([safety] stop_refused_by): a move under way runs "
        "to its target; stop it with the stand's own controls if it must not finish"
    )
    assert str(StopRefusedError(("xy_stage XY",), "the facade's own text")) == (
        "the facade's own text"
    )


def test_stop_refused_error_pickles_with_its_names() -> None:
    for error in (
        StopRefusedError(("xy_stage XY", "z Z")),
        StopRefusedError(("xy_stage XY",), "the facade's own text"),
    ):
        copy = pickle.loads(pickle.dumps(error))
        assert (type(copy), str(copy), copy.refused) == (
            StopRefusedError,
            str(error),
            error.refused,
        )


def _raised_with_context(first: BaseException, second: BaseException) -> BaseException:
    """``second`` raised while ``first`` is handled, as a second Ctrl-C in a give-up."""
    try:
        try:
            raise first
        except BaseException:
            raise second from None
    except BaseException as caught:
        return caught


def test_stop_notes_reads_the_context_chain_oldest_first() -> None:
    accepted = StopNote("xy_stage XY", StopOutcome.accepted)
    refused = StopNote("z Z", StopOutcome.refused)
    cut = StopNote("xy_stage XY", StopOutcome.interrupted)
    first, second = KeyboardInterrupt(), KeyboardInterrupt()
    note_stop(first, accepted)
    note_stop(first, refused)
    note_stop(second, cut)
    # `from None` sets the cause aside, not the context: the walk reads it.
    raised = _raised_with_context(first, second)
    assert raised is second
    assert raised.__context__ is first
    assert stop_notes(raised) == (accepted, refused, cut)


def test_stop_notes_ends_on_a_context_chain_that_loops() -> None:
    first, second = KeyboardInterrupt(), KeyboardInterrupt()
    note = StopNote("xy_stage XY", StopOutcome.refused)
    note_stop(first, note)
    first.__context__, second.__context__ = second, first
    assert stop_notes(first) == (note,)


class _FrozenError(Exception):
    """An exception that takes no attribute: a plugin's or a library's own."""

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(name)


def test_a_note_on_an_exception_that_takes_none_is_dropped_not_raised() -> None:
    # `_move` notes every exception a move raises: a note that raised would
    # replace that exception with its own AttributeError.
    frozen = _FrozenError("the plugin's own error")
    note_stop(frozen, StopNote("xy_stage XY", StopOutcome.refused))
    assert stop_notes(frozen) == ()


def test_stop_notes_of_an_exception_without_notes_is_empty() -> None:
    assert stop_notes(KeyboardInterrupt()) == ()
    assert stop_notes(_raised_with_context(ValueError(), KeyboardInterrupt())) == ()


def test_a_stop_note_holds_no_exception() -> None:
    # FM-98: a note that held the error would keep its traceback, and the core.
    note = StopNote("xy_stage XY", StopOutcome.refused)
    # Two fields, a string and an enum: whether the role is declared is the
    # outcome (`refused` or `failed`), not a bool beside it (m2 §11.11 E4).
    assert [field.name for field in dataclasses.fields(StopNote)] == [
        "motion",
        "outcome",
    ]
    assert type(note.motion) is str
    assert isinstance(note.outcome, StopOutcome)
