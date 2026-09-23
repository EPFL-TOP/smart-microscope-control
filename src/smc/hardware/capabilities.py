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
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "XY",
    "Camera",
    "Limits",
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

    def move_to_um(self, x_um: float, y_um: float, *, wait: bool = True) -> XY:
        """Move to an absolute position; return the readback."""
        ...

    def move_by_um(
        self, dx_um: float, dy_um: float, *, wait: bool = True, force: bool = False
    ) -> XY:
        """Move relative to the current position; return the readback."""
        ...

    def wait(self, timeout_s: float | None = None) -> None:
        """Block until the stage is idle; ``DeviceTimeoutError`` past the deadline."""
        ...

    def is_busy(self) -> bool:
        """Whether the stage reports it is still moving."""
        ...

    def stop(self) -> None:
        """Stop the stage now; never refused, never waits for the microscope lock.

        A waiter on the stopped move raises ``MotionStoppedError``. It runs in
        dry-run too: a stop cannot create motion, and a dry-run session may be
        watching a stage moved by hand (design §13).
        """
        ...

    def limits_um(self) -> tuple[Limits, Limits] | None:
        """The soft limits as ``(x, y)``; ``None`` means unknown, not unlimited."""
        ...


@runtime_checkable
class ZStage(Protocol):
    """A focus drive. Same contract as ``XYStage`` without the jog guard."""

    def position_um(self) -> float:
        """Where the drive reports it is."""
        ...

    def move_to_um(self, z_um: float, *, wait: bool = True) -> float:
        """Move to an absolute position; return the readback."""
        ...

    def move_by_um(self, dz_um: float, *, wait: bool = True) -> float:
        """Move relative to the current position; return the readback."""
        ...

    def wait(self, timeout_s: float | None = None) -> None:
        """Block until the drive is idle; ``DeviceTimeoutError`` past the deadline."""
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
