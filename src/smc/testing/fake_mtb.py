"""``FakeMTB``: the Axio Observer 7's MTB components in Python (m4 §9.1).

It implements ``smc.hardware.backends.zeiss_mtb.api.MTBApi``, so the unicore
devices and the session run against it on every OS, without pythonnet or
Zeiss software (FM-40). ``fake_stand_config`` writes the profile's own
``.cfg`` without the PVCAM camera, so a stand of these devices needs no
Micro-Manager adapter either.

The stand, as m4 §1.1 describes it: the four axes with their ranges, the five
changers with the stand's element names where they are known, six LEDs, and
Definite Focus 2 acting on the focus drive (``FakeFocusStabilizer``). The
steps and deviations of the axes are **chosen for the fake**, from the
predecessor's measurements where it had one (X/Y step 0.25 µm; a piezo
error of 0.082 µm); they are not MTB's declared values:

========  ======  =========  =========
axis      step    typical    max dev.
========  ======  =========  =========
X, Y      0.25    1.0        2.0
focus     0.01    0.02       0.1
piezo     0.01    0.055      0.2
========  ======  =========  =========

A synchronous move sleeps ``move_s`` and moves in a straight line, so a read
from another thread during it sees a position between the start and the
target. Knobs on each component make it fail as a test needs:
``refuse_next`` (returns ``False``, does not move), ``false_next`` (moves,
then returns ``False``, as MTB does for a zero-distance move or a dithering
piezo), ``fail_next`` (raises ``MTBError``), ``deviation_um`` (the offset of
where a move lands), ``absent`` on ``FakeMTB`` (ids it refuses), and on an LED
``fail_off`` and ``fail_on``. Every call is recorded in order in ``calls``.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from smc.hardware.backends.zeiss_mtb.api import (
    STABILIZER_OPERATION_MODES,
    AxisInfo,
    ChangerElement,
    MTBError,
    MTBSessionError,
)

__all__ = [
    "FakeAxis",
    "FakeChanger",
    "FakeFocusStabilizer",
    "FakeLED",
    "FakeLEDController",
    "FakeMTB",
    "fake_stand_config",
]

#: The Colibri's six LEDs, in MTB's order (m4 §1.1).
COLIBRI_NM = (385, 430, 475, 511, 567, 630)


@dataclass
class _Calls:
    """The calls of one fake stand, in order, shared by its components."""

    items: list[tuple[object, ...]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, *call: object) -> None:
        with self.lock:
            self.items.append(call)


class FakeAxis:
    """One MTB axis: a straight-line synchronous move, and the failure knobs."""

    def __init__(
        self,
        mtb_id: str,
        info: AxisInfo,
        position_um: float,
        calls: _Calls,
        *,
        move_s: float = 0.0,
    ) -> None:
        self.mtb_id = mtb_id
        self._info = info
        self._calls = calls
        self._lock = threading.Lock()
        self._position_um = position_um
        self._start_um = position_um
        self._end_um = position_um
        self._started_at: float | None = None
        self._duration_s = 0.0
        self.move_s = move_s
        self.refuse_next = False
        self.false_next = False
        self.fail_next = False
        self.deviation_um = 0.0

    @property
    def moves(self) -> list[float]:
        """The targets sent, in order."""
        return [
            float(c[2])  # type: ignore[arg-type]
            for c in self._calls.items
            if c[0] == self.mtb_id and c[1] == "move_um"
        ]

    def info(self) -> AxisInfo:
        self._calls.add(self.mtb_id, "info")
        return self._info

    def _here(self) -> float:
        """Where the axis is now, on the straight line of a move under way; under the lock."""
        if self._started_at is None or self._duration_s <= 0:
            return self._position_um
        done = min(1.0, (time.monotonic() - self._started_at) / self._duration_s)
        return self._start_um + (self._end_um - self._start_um) * done

    def position_um(self) -> float:
        with self._lock:
            return self._here()

    def set_position_um(self, position_um: float) -> None:
        """Put the axis somewhere, as a hand on the stand's knob would."""
        with self._lock:
            self._position_um = position_um

    def glide(
        self, target_um: float, move_s: float, stop: threading.Event | None = None
    ) -> bool:
        """A straight line to ``target_um`` over ``move_s``; ``False`` when ``stop`` ended it.

        A stopped glide ends where the line had got to. The position is kept
        under the axis's own lock, so a read from another thread (the device
        of this axis) sees a point of the line while another component (the
        fake DF2) drives it.
        """
        with self._lock:
            self._start_um = self._here()
            self._end_um = target_um
            self._duration_s = move_s
            self._started_at = time.monotonic()
        stopped = False
        if move_s > 0:
            if stop is None:
                time.sleep(move_s)
            else:
                stopped = stop.wait(move_s)
        with self._lock:
            self._position_um = self._here() if stopped else self._end_um
            self._started_at = None
        return not stopped

    def move_um(self, target_um: float, timeout_ms: int) -> bool:
        self._calls.add(self.mtb_id, "move_um", target_um, timeout_ms)
        if self.fail_next:
            self.fail_next = False
            raise MTBError(f"{self.mtb_id}: SetPosition({target_um}) raised (fake)")
        if self.refuse_next:
            self.refuse_next = False
            return False
        if not self._info.min_um <= target_um <= self._info.max_um:
            raise MTBError(f"{self.mtb_id}: {target_um} is outside the range (fake)")
        answer = not self.false_next
        self.false_next = False
        self.glide(target_um + self.deviation_um, self.move_s)
        return answer


class FakeChanger:
    """One MTB changer: 1-based positions, a synchronous move, the failure knobs."""

    def __init__(
        self,
        mtb_id: str,
        elements: tuple[ChangerElement, ...],
        position: int,
        calls: _Calls,
        *,
        move_s: float = 0.0,
    ) -> None:
        self.mtb_id = mtb_id
        self._elements = elements
        self._calls = calls
        self._position = position
        self.move_s = move_s
        self.refuse_next = False
        self.fail_next = False
        #: Where a move lands instead of its target, for the readback check.
        self.lands_at: int | None = None

    def elements(self) -> tuple[ChangerElement, ...]:
        self._calls.add(self.mtb_id, "elements")
        return self._elements

    def position(self) -> int:
        return self._position

    def set_position(self, position: int) -> None:
        """Put the changer somewhere, as a hand on the stand would."""
        self._position = position

    def move(self, position: int, timeout_ms: int) -> bool:
        self._calls.add(self.mtb_id, "move", position, timeout_ms)
        if self.fail_next:
            self.fail_next = False
            raise MTBError(f"{self.mtb_id}: SetPosition({position}) raised (fake)")
        if self.refuse_next:
            self.refuse_next = False
            return False
        positions = [e.position for e in self._elements]
        if position not in positions:
            raise MTBError(f"{self.mtb_id}: no position {position} (fake)")
        if self.move_s > 0:
            time.sleep(self.move_s)
        self._position = self.lands_at if self.lands_at is not None else position
        return True


class FakeLED:
    """One Colibri LED: on/off and an intensity in %."""

    def __init__(self, wavelength_nm: int, calls: _Calls) -> None:
        self.wavelength_nm = wavelength_nm
        self._calls = calls
        self.on = False
        self.intensity = 0.0
        self.fail_off = False
        self.fail_on = False

    @property
    def name(self) -> str:
        return f"LED{self.wavelength_nm}"

    def is_on(self) -> bool:
        return self.on

    def set_on(self, on: bool, timeout_ms: int) -> None:
        self._calls.add(self.name, "set_on", on)
        if on and self.fail_on:
            raise MTBError(f"LED {self.wavelength_nm}nm: SetOnOff(On) failed (fake)")
        if not on and self.fail_off:
            raise MTBError(f"LED {self.wavelength_nm}nm: SetOnOff(Off) failed (fake)")
        self.on = bool(on)

    def intensity_pct(self) -> float:
        return self.intensity

    def set_intensity_pct(self, pct: float, timeout_ms: int) -> None:
        self._calls.add(self.name, "set_intensity_pct", pct)
        self.intensity = float(pct)


class FakeLEDController:
    """The Colibri: six LEDs and the manual mode."""

    def __init__(
        self, calls: _Calls, wavelengths: tuple[int, ...] = COLIBRI_NM
    ) -> None:
        self.mtb_id = "MTBFLLEDController"
        self._calls = calls
        self.led_list = tuple(FakeLED(nm, calls) for nm in wavelengths)
        self.manual = False

    def leds(self) -> tuple[FakeLED, ...]:
        return self.led_list

    def set_manual_mode(self, timeout_ms: int) -> None:
        self._calls.add(self.mtb_id, "set_manual_mode")
        self.manual = True

    def lit(self) -> list[int]:
        """The wavelengths of the LEDs that are on."""
        return [led.wavelength_nm for led in self.led_list if led.on]


class FakeFocusStabilizer:
    """Definite Focus 2 (m4 §11.3): learns the focus drive's plane, restores it, stabilizes once.

    It acts on the focus axis it is given, as DF2 corrects through the focus
    drive on the Observer 7 (m4 §11.1). ``learn`` takes ``move_s``, as DF2
    measures for a moment, returns a new 9-byte blob at each call (bytes
    above 0x7F included) and records the focus position as its plane, which
    it then holds; ``restore`` makes a blob it learned the held one, and answers
    ``False`` to any other; ``stabilize_now`` moves the focus axis to the held
    plane as ``FakeAxis`` moves (a straight line over ``move_s``) and answers
    ``False`` with none held, or when ``stop`` ended it where the axis was.

    Knobs: ``refuse_next`` (answers ``False``, moves nothing; a refused learn
    gives no blob, an ``MTBError``), ``fail_next`` (raises ``MTBError``),
    ``piezo_focus_usage_on``, ``periodic_on``, ``stop_ignored`` (a ``stop``
    that ends nothing). Every call is in the stand's ``calls``.
    """

    def __init__(self, focus: FakeAxis, calls: _Calls, *, move_s: float = 0.0) -> None:
        self.mtb_id = "MTBFocusStabilizer2"
        self._focus = focus
        self._calls = calls
        self._lock = threading.Lock()
        #: Every blob learned, and the focus position it was learned at.
        self.planes: dict[bytes, float] = {}
        self.held: bytes | None = None
        self._stop: threading.Event | None = None
        self.move_s = move_s
        self.mode = "Balanced"
        self.refuse_next = False
        self.fail_next = False
        #: Named apart from ``piezo_focus_usage()``, the seam's read of it.
        self.piezo_focus_usage_on = False
        self.periodic_on = False
        self.stop_ignored = False

    def _fail_if_asked(self, call: str) -> None:
        if self.fail_next:
            self.fail_next = False
            raise MTBError(f"{self.mtb_id}: {call} raised (fake)")

    def _refused(self) -> bool:
        refused, self.refuse_next = self.refuse_next, False
        return refused

    def learn(self, timeout_ms: int) -> bytes:
        self._calls.add(self.mtb_id, "learn", timeout_ms)
        self._fail_if_asked("InitOnCurrentFocusPosition")
        if self._refused():
            raise MTBError(
                f"{self.mtb_id}: InitOnCurrentFocusPosition answered False and gave "
                "no reference (fake)"
            )
        if self.move_s > 0:
            # DF2 measures for a moment: 0.4 s at the stand (m4 §11.1).
            time.sleep(self.move_s)
        with self._lock:
            blob = bytes((0xDF, 0x02, 0xF0)) + len(self.planes).to_bytes(6, "big")
            self.planes[blob] = self._focus.position_um()
            self.held = blob
        return blob

    def restore(self, reference: bytes, timeout_ms: int) -> bool:
        self._calls.add(self.mtb_id, "restore", bytes(reference), timeout_ms)
        self._fail_if_asked("InitWithStabilizingData")
        with self._lock:
            if self._refused() or bytes(reference) not in self.planes:
                # What DF2 holds after a refusal is unknown: here, nothing.
                self.held = None
                return False
            self.held = bytes(reference)
        return True

    def stabilize_now(self, timeout_ms: int) -> bool:
        self._calls.add(self.mtb_id, "stabilize_now", timeout_ms)
        self._fail_if_asked("StabilizeNow")
        if self._refused():
            return False
        with self._lock:
            if self.held is None:
                return False
            plane = self.planes[self.held]
            stop = self._stop = threading.Event()
        return self._focus.glide(plane, self.move_s, stop)

    def stop(self, timeout_ms: int) -> None:
        self._calls.add(self.mtb_id, "stop", timeout_ms)
        self._fail_if_asked("Stop")
        if self.stop_ignored:
            return
        with self._lock:
            stop = self._stop
        if stop is not None:
            stop.set()

    def operation_mode(self) -> str:
        return self.mode

    def set_operation_mode(self, mode: str) -> None:
        self._calls.add(self.mtb_id, "set_operation_mode", mode)
        if mode not in STABILIZER_OPERATION_MODES:
            raise MTBError(
                f"{self.mtb_id}: no operation mode {mode!r}; one of "
                f"{', '.join(STABILIZER_OPERATION_MODES)} (fake)"
            )
        self.mode = mode

    def piezo_focus_usage(self) -> bool:
        return self.piezo_focus_usage_on

    def periodic(self) -> bool:
        return self.periodic_on

    def stop_periodic(self, timeout_ms: int) -> None:
        self._calls.add(self.mtb_id, "stop_periodic", timeout_ms)
        self._fail_if_asked("StabilizePeriodically(Off)")
        self.periodic_on = False


def _elements(*names: str, empty: tuple[int, ...] = ()) -> tuple[ChangerElement, ...]:
    return tuple(
        ChangerElement(
            position=i, name=name, element_type="None" if i in empty else "Element"
        )
        for i, name in enumerate(names, start=1)
    )


class FakeMTB:
    """The stand's MTB, in Python; implements ``api.MTBApi``."""

    def __init__(self, *, move_s: float = 0.0) -> None:
        self._calls = _Calls()
        self.logins = 0
        self.logouts = 0
        self.absent: set[str] = set()
        self.fail_next_login = False
        self._logged_out = False
        xy = AxisInfo(-350_000.0, 350_000.0, 0.25, 1.0, 2.0)
        self.axes: dict[str, FakeAxis] = {
            "MTBStageAxisX": FakeAxis(
                "MTBStageAxisX", xy, 0.0, self._calls, move_s=move_s
            ),
            "MTBStageAxisY": FakeAxis(
                "MTBStageAxisY", xy, 0.0, self._calls, move_s=move_s
            ),
            "MTBFocus": FakeAxis(
                "MTBFocus",
                AxisInfo(-14_000.0, 14_000.0, 0.01, 0.02, 0.1),
                1000.0,
                self._calls,
                move_s=move_s,
            ),
            "MTBPiezoFocusCan": FakeAxis(
                "MTBPiezoFocusCan",
                AxisInfo(0.0, 500.0, 0.01, 0.055, 0.2),
                250.0,
                self._calls,
                move_s=move_s,
            ),
        }
        # Objectives and sideport as the stand reports them (m4 §1.1); the
        # reflector's and the condenser's names are chosen for the fake, with
        # the "none" the stand shows on more than one position of each.
        changers = {
            "MTBObjectiveChanger": (
                _elements(
                    "",
                    "Fluar 10x/0.50",
                    "",
                    "Plan-Apochromat 40x/0.95",
                    "",
                    "",
                    empty=(1, 3, 5, 6),
                ),
                2,
            ),
            "MTBReflectorChanger": (
                _elements(
                    "none", "38 HE eGFP", "43 HE DsRed", "none", "49 DAPI", "none"
                ),
                1,
            ),
            "MTBSideportChanger": (
                _elements("VIS 100%", "Sideport 100% Left", "Sideport 100% Right"),
                2,
            ),
            "MTBCondenserContrastChanger": (
                _elements("none", "PH1", "PH2", "PH3", "DIC", "none"),
                1,
            ),
            "MTBTLShutter": (_elements("Open", "Closed"), 2),
        }
        self.changers: dict[str, FakeChanger] = {
            mtb_id: FakeChanger(mtb_id, elements, position, self._calls, move_s=move_s)
            for mtb_id, (elements, position) in changers.items()
        }
        self.colibri = FakeLEDController(self._calls)
        self.stabilizer = FakeFocusStabilizer(
            self.axes["MTBFocus"], self._calls, move_s=move_s
        )

    @property
    def calls(self) -> list[tuple[object, ...]]:
        """Every call made, in order: ``(component, method, *arguments)``."""
        with self._calls.lock:
            return list(self._calls.items)

    def login(self) -> None:
        self._calls.add("FakeMTB", "login")
        if self._logged_out:
            raise MTBSessionError(
                "MTB could not be initialized (fake: logged out before)"
            )
        if self.fail_next_login:
            self.fail_next_login = False
            raise MTBError(
                "Error while finishing initialization (fake: the stand is off)"
            )
        self.logins += 1

    def logout(self) -> None:
        self._calls.add("FakeMTB", "logout")
        self.logouts += 1
        self._logged_out = True

    def _present(self, mtb_id: str) -> None:
        if mtb_id in self.absent:
            raise MTBError(f"MTB component {mtb_id!r} is absent")

    def axis(self, mtb_id: str) -> FakeAxis:
        self._present(mtb_id)
        if mtb_id not in self.axes:
            raise MTBError(f"MTB component {mtb_id!r} is absent")
        return self.axes[mtb_id]

    def changer(self, mtb_id: str) -> FakeChanger:
        self._present(mtb_id)
        if mtb_id not in self.changers:
            raise MTBError(f"MTB component {mtb_id!r} is absent")
        return self.changers[mtb_id]

    def led_controller(self, mtb_id: str) -> FakeLEDController:
        self._present(mtb_id)
        if mtb_id != self.colibri.mtb_id:
            raise MTBError(f"MTB component {mtb_id!r} is absent")
        return self.colibri

    def focus_stabilizer(self, mtb_id: str) -> FakeFocusStabilizer:
        self._present(mtb_id)
        if mtb_id != self.stabilizer.mtb_id:
            raise MTBError(f"MTB component {mtb_id!r} is absent")
        return self.stabilizer


def fake_stand_config(profile_cfg: Path, out_dir: Path) -> Path:
    """The profile's ``.cfg`` with only its ``#py`` lines and ``Property,Core,Initialize``.

    The contract stand runs **the profile's own device lines**, without the
    PVCAM camera, so it needs no Micro-Manager adapter and runs on every CI
    job (m4 §9.1). The profile's ``.cfg`` is ASCII (m4 §10); the copy is
    written as UTF-8.
    """
    kept = [
        line
        for line in profile_cfg.read_text(encoding="utf-8").splitlines()
        if line.lstrip(" \t").startswith("#py ")
        or line.strip().startswith("Property,Core,Initialize,")
    ]
    out = out_dir / f"{profile_cfg.stem}-fake.cfg"
    out.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return out
