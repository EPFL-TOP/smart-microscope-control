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
    "STABILIZER_OPERATION_MODES",
    "STABILIZER_TIMEOUT_MS",
    "AxisInfo",
    "ChangerElement",
    "MTBApi",
    "MTBAxis",
    "MTBChanger",
    "MTBError",
    "MTBFocusStabilizer",
    "MTBLEDController",
    "MTBSessionError",
]

#: A traverse of the plate; the predecessor's 10 s was for short moves (FM-10).
AXIS_TIMEOUT_MS = 60_000
#: The predecessor's, used daily on this stand for the turrets and the objective.
CHANGER_TIMEOUT_MS = 30_000
#: The predecessor's, for switching an LED and setting its intensity.
LED_TIMEOUT_MS = 10_000
#: Definite Focus 2's calls: the predecessor's, and the Micro-Manager wiki
#: recommends 25 s for DF2. How long a ``StabilizeNow`` takes is not measured
#: (m4 §12).
STABILIZER_TIMEOUT_MS = 30_000
#: DF2's precision settings, as ``MTBFocusStabilizer2OperationMode`` names them:
#: slowest and most precise first. Not one shot versus continuous (m4 §11.1).
STABILIZER_OPERATION_MODES = ("Exact", "Balanced", "Fast")


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


class MTBFocusStabilizer(Protocol):
    """Definite Focus 2 (``MTBFocusStabilizer2``): learn a plane, restore it, stabilize once.

    A reference is DF2's own blob, opaque to smc. ``restore`` and
    ``stabilize_now`` return MTB's answer and the device judges it (m4 §11.4);
    ``learn`` with no blob, or an empty one, is an ``MTBError``.
    """

    mtb_id: str

    def learn(self, timeout_ms: int) -> bytes:
        """``InitOnCurrentFocusPosition``: the reference blob of the plane in focus now."""
        ...

    def restore(self, reference: bytes, timeout_ms: int) -> bool:
        """``InitWithStabilizingData``: make ``reference`` DF2's; MTB's own answer."""
        ...

    def stabilize_now(self, timeout_ms: int) -> bool:
        """``StabilizeNow``: one correction to the plane held; MTB's own answer."""
        ...

    def stop(self, timeout_ms: int) -> None:
        """``Stop``: end what DF2 is doing."""
        ...

    def operation_mode(self) -> str:
        """``"Exact"``, ``"Balanced"`` or ``"Fast"``, read live."""
        ...

    def set_operation_mode(self, mode: str) -> None:
        """Set one of ``STABILIZER_OPERATION_MODES``; any other name is an ``MTBError``."""
        ...

    def piezo_focus_usage(self) -> bool:
        """Whether DF2 corrects through the piezo (``True``) or the focus drive."""
        ...

    def periodic(self) -> bool:
        """Whether periodic (continuous) stabilisation is on: ``GetStabilizePeriodicallyOnOff``."""
        ...

    def stop_periodic(self, timeout_ms: int) -> None:
        """``StabilizePeriodically(Off, 0)``: periodic stabilisation off."""
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

    def focus_stabilizer(self, mtb_id: str) -> MTBFocusStabilizer:
        """The focus stabilizer ``mtb_id``; an ``MTBError`` naming the id when absent."""
        ...
