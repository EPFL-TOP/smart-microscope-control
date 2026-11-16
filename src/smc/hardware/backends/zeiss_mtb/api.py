"""The seam the MTB devices speak to: Protocols, value types and ``MTBError`` (m4 §4).

``dotnet`` implements these Protocols over ``MTBApi.dll``; ``smc.testing.fake_mtb``
implements them in Python, so every device is tested on any OS without .NET.
Every call is synchronous and may block; every failure is an ``MTBError``.

``MTBError`` is a ``RuntimeError`` so that the layer's ``as_hardware_errors``
turns it into a ``HardwareError`` naming the role and the call (FM-79); a
device never raises anything else out of a core call, except the
``ValueError`` unicore raises itself for a value outside a property's range.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

__all__ = [
    "AXIS_TIMEOUT_MS",
    "CHANGER_TIMEOUT_MS",
    "LED_TIMEOUT_MS",
    "MTBLED",
    "AxisInfo",
    "ChangerElement",
    "MTBApi",
    "MTBAxis",
    "MTBChanger",
    "MTBError",
    "MTBLEDController",
    "MTBSessionError",
]

#: A traverse of the plate; the predecessor's 10 s was for short moves (FM-10).
AXIS_TIMEOUT_MS = 60_000
#: The predecessor's, used daily on this stand for the turrets and the objective.
CHANGER_TIMEOUT_MS = 30_000
#: The predecessor's, for switching an LED and setting its intensity.
LED_TIMEOUT_MS = 10_000


class MTBError(RuntimeError):
    """MTB refused a call or failed in it; the text names the component and the call."""


class MTBSessionError(MTBError):
    """The process's session cannot serve: logged out, or a second session refused."""


@dataclass(frozen=True)
class AxisInfo:
    """What an axis declares of itself, in µm."""

    min_um: float
    max_um: float
    step_um: float
    typical_deviation_um: float
    max_deviation_um: float


@dataclass(frozen=True)
class ChangerElement:
    """One position of a changer, as MTB names it."""

    position: int  # MTB's, 1-based
    name: str
    element_type: str  # "None" for an empty objective slot


class MTBAxis(Protocol):
    """One continuously positionable MTB axis, in µm."""

    mtb_id: str

    def info(self) -> AxisInfo:
        """Range, step and deviations, as MTB declares them."""
        ...

    def position_um(self) -> float:
        """Where MTB reads the axis now."""
        ...

    def move_um(self, target_um: float, timeout_ms: int) -> bool:
        """A synchronous absolute move; returns MTB's own return value."""
        ...


class MTBChanger(Protocol):
    """A discrete-position MTB changer (turret, sideport, shutter)."""

    mtb_id: str

    def elements(self) -> tuple[ChangerElement, ...]:
        """``MinPosition`` to ``MaxPosition``, in order."""
        ...

    def position(self) -> int:
        """MTB's current position, 1-based."""
        ...

    def move(self, position: int, timeout_ms: int) -> bool:
        """A synchronous move to a 1-based position; returns MTB's own return value."""
        ...


class MTBLED(Protocol):
    """One LED of a Colibri: on/off and an intensity in %."""

    wavelength_nm: int

    def is_on(self) -> bool:
        """Whether MTB reads the LED on."""
        ...

    def set_on(self, on: bool, timeout_ms: int) -> None:
        """Switch the LED on or off."""
        ...

    def intensity_pct(self) -> float:
        """The intensity it is lit at, 0 to 100."""
        ...

    def set_intensity_pct(self, pct: float, timeout_ms: int) -> None:
        """Set the intensity, 0 to 100."""
        ...


class MTBLEDController(Protocol):
    """The Colibri: its LEDs and its operation mode."""

    mtb_id: str

    def leds(self) -> tuple[MTBLED, ...]:
        """The LEDs, in MTB's order."""
        ...

    def set_manual_mode(self, timeout_ms: int) -> None:
        """Continuous manual LED control (``LEDLight``, ``Continuous``)."""
        ...


class MTBApi(Protocol):
    """One MTB connection: the login and the components."""

    def login(self) -> None:
        """Log in and initialise the hardware (``GetRoot``)."""
        ...

    def logout(self) -> None:
        """Log out; MTB cannot log in again in the same process."""
        ...

    def axis(self, mtb_id: str) -> MTBAxis:
        """The axis ``mtb_id``; an ``MTBError`` naming the id when absent."""
        ...

    def changer(self, mtb_id: str) -> MTBChanger:
        """The changer ``mtb_id``; an ``MTBError`` naming the id when absent."""
        ...

    def led_controller(self, mtb_id: str) -> MTBLEDController:
        """The LED controller ``mtb_id``; an ``MTBError`` naming the id when absent."""
        ...
