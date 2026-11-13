"""An autofocus that records every call and fails where a test says (#454).

Not a test module: imported as ``from recording_autofocus import
RecordingAutofocus`` by the tests of ``goto-well`` and of the plate page's jog,
which put it on a stand with ``Microscope.override(Autofocus, ...)``. That checks
``isinstance`` against the ``Autofocus`` Protocol, so every method of it is here,
as in ``test_scan_demo``'s ``_Autofocus``, whose copy it is.

A failure is a **fresh** ``CoreError`` at every call: an error kept and raised
again keeps the traceback of every frame it went through, and with it the
microscope those frames held (FM-98's test hazard).
"""

from __future__ import annotations

from typing import Literal

from smc.hardware.capabilities import FocusStep
from smc.hardware.errors import CoreError

__all__ = ["RecordingAutofocus"]

Failing = Literal["is_engaged", "disengage"]


class RecordingAutofocus:
    """An autofocus that records every call in ``calls``; on or off as a test sets it.

    Args:
        engaged: Whether it is switched on.
        locked: Whether it reads locked while on; ``engaged=True, locked=False`` is
            PFS on and searching.
        fails: ``"is_engaged"`` or ``"disengage"``: every call of that method
            raises a fresh ``CoreError(f"{name} failed on the fake")``.
        read_fails_after_off: ``is_engaged()`` raises a fresh ``CoreError`` once
            ``disengage()`` was called: a read back that fails.
        stays_on: ``disengage()`` records the call and leaves it as it was: a
            switch-off that does not hold.
    """

    def __init__(
        self,
        *,
        engaged: bool,
        locked: bool = True,
        fails: Failing | None = None,
        read_fails_after_off: bool = False,
        stays_on: bool = False,
    ) -> None:
        self.engaged = engaged
        self.locked = locked
        self.fails = fails
        self.read_fails_after_off = read_fails_after_off
        self.stays_on = stays_on
        self.switched_off = False
        self.offset = 4242.0
        self.calls: list[str] = []

    def _fail_if_asked(self, name: str) -> None:
        if self.fails == name:
            raise CoreError(f"{name} failed on the fake")

    def is_engaged(self) -> bool:
        self.calls.append("is_engaged")
        self._fail_if_asked("is_engaged")
        if self.read_fails_after_off and self.switched_off:
            raise CoreError("is_engaged failed on the fake after the switch-off")
        return self.engaged

    def is_locked(self) -> bool:
        self.calls.append("is_locked")
        return self.engaged and self.locked

    def is_in_range(self) -> bool | None:
        self.calls.append("is_in_range")
        return None

    def engage(self, timeout_s: float | None = None) -> bool:
        self.calls.append("engage")
        self.engaged = True
        return self.locked

    def disengage(self) -> None:
        self.calls.append("disengage")
        self._fail_if_asked("disengage")
        self.switched_off = True
        if not self.stays_on:
            self.engaged = False

    def offset_native(self) -> float:
        self.calls.append("offset_native")
        return self.offset

    def set_offset_native(self, value: float) -> float:
        self.calls.append("set_offset_native")
        return value

    def offset_by_native(self, delta: float, *, force: bool = False) -> float:
        self.calls.append("offset_by_native")
        return self.offset + delta

    def offset_unit(self) -> str:
        self.calls.append("offset_unit")
        return ""

    def offset_limits_native(self) -> tuple[float, float] | None:
        self.calls.append("offset_limits_native")
        return None

    def focus_by_um(self, dz_um: float, *, force: bool = False) -> FocusStep:
        self.calls.append("focus_by_um")
        raise AssertionError("neither goto-well nor the page's jog focuses")
