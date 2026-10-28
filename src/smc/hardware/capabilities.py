"""Capabilities: the typed interfaces every tool programs against (ADR-0003).

A tool never asks for "the Ti2 stage" or "the Hamamatsu"; it asks for an
``XYStage`` or a ``Camera`` and gets whatever device fills that role on the
stand it runs on. These Protocols are that contract, and the contract tests
hold every backend to the semantics written here (design §3).

Two conventions are encoded in the signatures, because they prevent real
mistakes at a microscope:

* **Units in names** (``position_um``, ``exposure_ms``): Micro-Manager speaks
  µm and ms, vendor SDKs often do not, and a silent unit mix-up drives an
  objective into a plate.
* **Methods, not properties**, for anything that reaches hardware: a read
  can be slow or fail, and a property hides that from the caller.

Mutating calls return the *readback* — the value the device reports after
the change — so a caller sees where the stage actually landed, not where it
was asked to go. In dry-run they return the commanded value.

This module is pure: it imports no Micro-Manager code, so tools and tests
can type against it without the device adapters installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

__all__ = [
    "XY",
    "Autofocus",
    "Camera",
    "ChannelPreset",
    "Channels",
    "FocusStep",
    "LightSource",
    "Limits",
    "ObjectiveTurret",
    "Properties",
    "PropertyInfo",
    "Shutter",
    "XYStage",
    "ZStage",
]


@dataclass(frozen=True, slots=True)
class XY:
    """A stage position in Micro-Manager coordinates: µm, X right, Y up."""

    x_um: float
    y_um: float


@dataclass(frozen=True, slots=True)
class Limits:
    """A closed travel range on one axis, in µm."""

    low_um: float
    high_um: float


@dataclass(frozen=True, slots=True)
class FocusStep:
    """What ``Autofocus.focus_by_um`` moved, and where focus ended (m2 §3.5).

    ``via`` says which device took the change: the focus drive (``"z"``)
    with the autofocus off, or the autofocus offset (``"offset"``) while it
    holds focus, since a Z move would be undone by the relock.
    """

    via: Literal["z", "offset"]
    #: The focus drive's position afterwards; ``None``: the stand has no focus drive.
    z_um: float | None
    #: The offset afterwards, in its own units, when ``via == "offset"``; else ``None``.
    offset_native: float | None


@dataclass(frozen=True, slots=True)
class PropertyInfo:
    """One device property as Micro-Manager describes it.

    Values stay strings, as MMCore stores them: converting them here would
    guess a type the adapter never promised. ``number`` is offered for the
    common numeric case and says ``None`` rather than guessing.
    """

    device: str
    name: str
    value: str
    read_only: bool = False
    allowed: tuple[str, ...] = ()
    lower: float | None = None
    upper: float | None = None

    @property
    def numeric(self) -> bool:
        """Whether the adapter declares a numeric range (both limits set)."""
        return self.lower is not None and self.upper is not None

    @property
    def number(self) -> float | None:
        """The value as a float, or ``None`` when it is not a number."""
        try:
            return float(self.value)
        except ValueError:
            return None

    def describe(self) -> str:
        """One human-readable line: ``Device.Name = value`` plus its constraints."""
        text = f"{self.device}.{self.name} = {self.value}"
        if self.numeric:
            text += f" [{self.lower:g} .. {self.upper:g}]"
        if self.allowed:
            text += " {" + " | ".join(self.allowed) + "}"
        if self.read_only:
            text += " (read-only)"
        return text


@dataclass(frozen=True, slots=True)
class ChannelPreset:
    """One channel: a preset of the stand's channel config group (m2 §8.2)."""

    group: str
    name: str
    #: ``(device, property, value)``, in the order MMCore applies them.
    settings: tuple[tuple[str, str, str], ...]
    #: The configuration file ``capture()`` wrote; ``None``: the running core only.
    saved_to: Path | None = None


@runtime_checkable
class XYStage(Protocol):
    """A motorised XY stage.

    ``move_by_um`` is jog-guarded: a relative move larger than the configured
    limit is refused unless ``force=True``, because a typo in a relative move
    (``1000`` for ``100``) is the classic way to leave the well. Absolute
    moves are not jog-guarded — crossing a plate is legitimate travel — but
    both kinds refuse a target outside the soft limits, which cannot be
    forced.
    """

    def position_um(self) -> XY:
        """Where the stage reports it is."""
        ...

    def move_to_um(self, x_um: float, y_um: float) -> XY:
        """Move to an absolute position; return the readback once it has arrived."""
        ...

    def move_by_um(self, dx_um: float, dy_um: float, *, force: bool = False) -> XY:
        """Move relative to the current position; return the readback once it has arrived."""
        ...

    def wait(self, timeout_s: float | None = None) -> None:
        """Block until the stage is idle; ``DeviceTimeoutError`` past the deadline.

        A read: it never waits for the microscope lock and never stops the
        stage (design §13).
        """
        ...

    def is_busy(self) -> bool:
        """Whether the stage reports it is still moving."""
        ...

    def stop(self) -> None:
        """Stop the stage now; never refused, never waits for the microscope lock.

        The move it interrupts raises ``MotionStoppedError``. It runs in
        dry-run too: a stop cannot create motion, and a dry-run session may be
        watching a stage moved by hand (design §13).
        """
        ...

    def limits_um(self) -> tuple[Limits, Limits] | None:
        """The soft limits as ``(x, y)``; ``None`` means unknown, not unlimited."""
        ...


@runtime_checkable
class ZStage(Protocol):
    """A focus drive. Same contract as ``XYStage``, jog guard included.

    The jog limit is the profile's ``max_z_jog_um``, separate from the XY
    ``max_jog_um``: a mistyped focus jog drives the objective into the sample.

    On a stand with an autofocus, both moves suspend it and, with
    ``keep_autofocus`` (the default), engage it again once the move has
    completed: moving Z switches PFS off on the Nikon stands (FM-12, m2
    §3.4). A move that completes but finds no lock again raises
    ``AutofocusLostError``; a move that fails, is stopped or is interrupted
    leaves the autofocus off, because engaging it would move Z after the
    operator's stop (FM-88). With ``keep_autofocus=False`` it stays off.
    """

    def position_um(self) -> float:
        """Where the drive reports it is."""
        ...

    def move_to_um(self, z_um: float, *, keep_autofocus: bool = True) -> float:
        """Move to an absolute position; return the readback once it has arrived."""
        ...

    def move_by_um(
        self, dz_um: float, *, force: bool = False, keep_autofocus: bool = True
    ) -> float:
        """Move relative to the current position; return the readback once it has arrived.

        Refuses ``|dz_um| > max_z_jog_um`` with ``SafetyRefusedError`` unless
        ``force``. ``force`` passes the jog guard only: the soft limits still
        apply to the target, and a non-finite distance is never forceable.
        """
        ...

    def wait(self, timeout_s: float | None = None) -> None:
        """Block until the drive is idle; a read, as ``XYStage.wait``."""
        ...

    def is_busy(self) -> bool:
        """Whether the drive reports it is still moving."""
        ...

    def stop(self) -> None:
        """Stop the drive now; same contract as ``XYStage.stop``."""
        ...

    def limits_um(self) -> Limits | None:
        """The soft limits; ``None`` means unknown, not unlimited."""
        ...


@runtime_checkable
class Camera(Protocol):
    """A camera that snaps single frames.

    ``pixel_size_um`` returns ``0.0`` when nothing measured it: a guessed
    pixel size silently corrupts every distance computed from an image.
    """

    def snap(self) -> np.ndarray:
        """One 2-D frame in the camera's native dtype and orientation."""
        ...

    def exposure_ms(self) -> float:
        """The current exposure time."""
        ...

    def set_exposure_ms(self, value_ms: float) -> float:
        """Set the exposure time; return the readback."""
        ...

    def image_shape(self) -> tuple[int, int]:
        """The frame shape as ``(height, width)``."""
        ...

    def bit_depth(self) -> int:
        """Significant bits per pixel."""
        ...

    def pixel_size_um(self) -> float:
        """Object-space pixel size; ``0.0`` means unknown, never guessed."""
        ...


@runtime_checkable
class Shutter(Protocol):
    """A shutter, and the core's auto-shutter switch that drives it during snaps."""

    def is_open(self) -> bool:
        """Whether the shutter reports open."""
        ...

    def set_open(self, open_: bool) -> bool:
        """Open or close; return the readback."""
        ...

    def auto_shutter(self) -> bool:
        """Whether the core opens the shutter around each exposure."""
        ...

    def set_auto_shutter(self, on: bool) -> bool:
        """Switch auto-shutter; return the readback."""
        ...


@runtime_checkable
class Autofocus(Protocol):
    """A continuous (hardware) autofocus: Nikon PFS, Zeiss Definite Focus (m2 §3).

    Engaged, it holds the focal plane by driving the focus drive itself, at
    an *offset* in the device's own units, which no profile can convert to
    µm without a measurement. So every value in those units ends in
    ``_native``, and ``focus_by_um`` takes µm and converts only through a
    factor the profile states as measured (``[autofocus] offset_per_um``).

    ``engage`` returns ``False`` when no lock came, and the autofocus is then
    off: one left searching can move Z with no action holding the
    microscope, which the layer forbids (design §13). ``disengage`` is a
    safe call, like closing a shutter: it is never refused.
    """

    def is_engaged(self) -> bool:
        """Whether continuous focus is switched on."""
        ...

    def is_locked(self) -> bool:
        """The device's own lock flag; not "settled" (m2 §12).

        On the Ti2-E it read ``True`` 0.5 s before PFS drove Z +235 µm (#205):
        ``engage()`` returning ``True`` is the call that waits for Z to settle.
        """
        ...

    def is_in_range(self) -> bool | None:
        """Whether its reference is in range; ``None``: the stand does not say.

        Read from the property the profile names (``[autofocus]
        in_range_property``); the layer never guesses which one it is.
        """
        ...

    def engage(self, timeout_s: float | None = None) -> bool:
        """Switch it on and wait for the lock; ``False``: no lock, and it is off.

        ``timeout_s`` defaults to the profile's ``lock_timeout_s``, else 5 s
        (assumed); it must be finite and in (0, 600] s. Its search drives Z,
        so a ``ZStage.stop()`` during the search ends it with the autofocus
        off, as a raise (``MotionStoppedError``), never a ``False``.

        ``True``: locked and settled (``[autofocus] settle_s``). On a stand
        that declares ``engage_drives_z_to_offset``, it engages only at an
        offset seen locked with this objective since the stand was opened,
        and refuses otherwise (``SafetyRefusedError``, not forceable). Z is
        guarded while it engages: out of its bound, it is switched off and
        ``AutofocusExcursionError`` is raised (m2 §12).
        """
        ...

    def disengage(self) -> None:
        """Switch it off now; never refused, runs while halted and in dry-run."""
        ...

    def offset_native(self) -> float:
        """The offset, in the device's own units."""
        ...

    def set_offset_native(self, value: float) -> float:
        """Move the offset to ``value``; return the readback (soft limits apply).

        With the autofocus on, it follows the offset and drives Z, so a
        ``ZStage.stop()`` during the move ends it with the autofocus off
        (``MotionStoppedError``).
        """
        ...

    def offset_by_native(self, delta: float, *, force: bool = False) -> float:
        """Move the offset by ``delta``; jog-guarded unless ``force``.

        With no ``[safety] max_autofocus_offset_jog`` in the profile, every
        non-zero change needs ``force``: the unit is the stand's own, so no
        default can be assumed. A ``ZStage.stop()`` ends it as it ends
        ``set_offset_native``.
        """
        ...

    def offset_unit(self) -> str:
        """What the offset's numbers are, from the profile; ``""``: unknown."""
        ...

    def offset_limits_native(self) -> tuple[float, float] | None:
        """The offset's soft limits; ``None`` means unknown, not unlimited."""
        ...

    def focus_by_um(self, dz_um: float, *, force: bool = False) -> FocusStep:
        """Change focus by ``dz_um`` µm through whatever holds focus now.

        Off: a Z move. Locked: an offset change of ``dz_um`` times the
        profile's measured factor for the current objective, then a relock.
        On but not locked, or locked with no factor: refused, not forceable.
        """
        ...


@runtime_checkable
class ObjectiveTurret(Protocol):
    """The nosepiece: which objective is in the light path (m2 §8.3).

    Turning it can drive an objective into a loaded plate (FM-14), so
    ``select`` needs ``confirm=True`` unless the profile's ``[safety]
    turret_requires_confirm`` is false. A turn switches the autofocus off
    first and never engages it again: the new objective's offset is not
    known. A turret cannot be stopped from smc.
    """

    def labels(self) -> list[str]:
        """Every position's label, in position order, empty positions included."""
        ...

    def current(self) -> str:
        """The label of the objective in the light path."""
        ...

    def select(self, label: str, *, confirm: bool = False) -> str:
        """Turn to ``label``; return the readback. Refused without ``confirm``."""
        ...

    def pixel_size_um(self, label: str | None = None) -> float:
        """The profile's pixel size for ``label`` (default: current), at binning 1; ``0.0``: unknown.

        The profile's size is for a magnification factor of 1.0; the result is
        divided by the core's factor (m2 §8.10), so it is for the stand as it is.
        """
        ...


@runtime_checkable
class LightSource(Protocol):
    """A lamp or LED, and the knob that sets how bright it is (m2 §8.4).

    The intensity is in the device's own units (``_native``): no profile says
    what the ``DiaLamp``'s ``Intensity`` numbers are. A device with no such
    knob reports ``None``, never a fake one.
    """

    def device(self) -> str:
        """The light source's device label."""
        ...

    def intensity_knob(self) -> PropertyInfo | None:
        """The property that sets how bright, read now; ``None``: none that smc knows."""
        ...

    def intensity_native(self) -> float | None:
        """The knob's value, in its own units; ``None``: no knob."""
        ...

    def set_intensity_native(self, value: float) -> float:
        """Set the knob; return the readback. Never clamped: outside its limits, refused."""
        ...

    def is_on(self) -> bool | None:
        """Whether it is on; ``None``: no switch that smc knows."""
        ...

    def set_on(self, on: bool) -> bool:
        """Switch it on or off; return the readback. Off is never refused."""
        ...


@runtime_checkable
class Channels(Protocol):
    """The stand's channels: the presets of the config group named for them (m2 §8.5)."""

    def group(self) -> str:
        """The config group whose presets are the channels."""
        ...

    def names(self) -> list[str]:
        """The channels, as the group lists them; ``[]`` while the group does not exist."""
        ...

    def current(self) -> str | None:
        """The first channel whose every setting matches the stand; ``None``: none does."""
        ...

    def preset(self, name: str) -> ChannelPreset:
        """One channel's settings, as the running core holds them."""
        ...

    def select(self, name: str) -> str | None:
        """Apply a channel; return the one the stand then matches (``None``: none).

        Refused, and not forceable, for a preset that would move a stage, an
        autofocus or the objective turret, or change a core slot.
        """
        ...

    def capture(self, name: str) -> ChannelPreset:
        """Record the stand's current illumination as channel ``name``, replacing one of that name.

        Written to the profile's configuration file first, then to the
        running core.
        """
        ...


@runtime_checkable
class Properties(Protocol):
    """Raw device properties: the escape hatch for what no capability covers yet."""

    def devices(self) -> list[str]:
        """Loaded device labels, without the ``Core`` pseudo-device."""
        ...

    def describe(self, device: str) -> list[PropertyInfo]:
        """Every property of one device."""
        ...

    def get(self, device: str, name: str) -> str:
        """One property value, as the adapter reports it."""
        ...

    def set(self, device: str, name: str, value: str | float | int) -> str:
        """Set one property; return the readback. Read-only properties refuse."""
        ...
