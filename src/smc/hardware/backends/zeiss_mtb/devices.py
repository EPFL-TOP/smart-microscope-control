"""The Observer 7's MTB components as pymmcore-plus unicore devices (m4 §7, §8).

A ``.cfg`` names each class on a ``#py pyDevice`` line, and unicore builds it
with no arguments, so the MTB id is a class attribute and nothing reaches MTB
before ``initialize()``, where the process's session is found
(``session.shared()``).

Hardware behaviour encoded here (m4 §1):

* **Every MTB move is synchronous**, and unicore holds a device's own lock for
  the whole of a core call on it. A move made in the call would block every
  read and every ``deviceBusy`` of that device until it ended, so a move
  starts a worker thread (``_Mover``) and returns; ``busy()`` is ``True``
  while the worker runs, and a failed move is raised once, by the next
  ``busy()``, which ``waitForDevice`` then raises (m4 §7.2 rules 3, 4).
* **A device never calls the core**: unicore's device lock is not reentrant,
  so a device method that called the core on its own label would deadlock
  (and hang the interpreter's exit).
* **MTB has no axis stop.** ``stop()`` cancels only what the device has not
  sent (the Y leg of an XY move), then raises an ``MTBError`` saying the move
  sent runs to its end; the profile declares ``stop_refused_by``, so the layer
  reads it as a declared refusal and cuts long moves (FM-117).
* **A zero-distance ``SetPosition`` returns ``False``** ("nothing to do") and
  the piezo dithers above its step, so a move is judged by its readback
  (``arrived``), never by MTB's answer alone (FM-167).
* **Light kills embryos.** The Colibri and the TL shutter are darkened at
  ``initialize()``, at every unload and at process exit (through the
  session), and a switch-off that fails is an ERROR saying ``LIGHT MAY STILL
  BE ON``.
* **Definite Focus 2 is a ``GenericDevice``** of five properties (unicore has
  no autofocus type, m4 §11.4): its learn, restore and stabilize run on the
  same kind of worker as a move, and a ``False`` from MTB is a failure.

Only ``MTBError`` (or unicore's own ``ValueError`` for a property value out of
range) leaves a device; every other exception from the seam is wrapped,
chained (FM-79, FM-113).
"""

from __future__ import annotations

import base64
import binascii
import logging
import threading
from collections.abc import Callable
from typing import ClassVar, TypeVar

from pymmcore_plus.experimental.unicore import (
    GenericDevice,
    ShutterDevice,
    StageDevice,
    StateDevice,
    XYStageDevice,
)

from smc.hardware.backends.zeiss_mtb import session as mtb_session
from smc.hardware.backends.zeiss_mtb.api import (
    AXIS_TIMEOUT_MS,
    CHANGER_TIMEOUT_MS,
    LED_TIMEOUT_MS,
    MTBLED,
    STABILIZER_OPERATION_MODES,
    STABILIZER_TIMEOUT_MS,
    AxisInfo,
    ChangerElement,
    MTBAxis,
    MTBChanger,
    MTBError,
    MTBFocusStabilizer,
    MTBLEDController,
)
from smc.hardware.backends.zeiss_mtb.session import LIGHT_MAY_STILL_BE_ON

__all__ = [
    "IDLE",
    "LEARN",
    "OFF",
    "ON",
    "RESTORE",
    "STABILIZE_NOW",
    "STOP",
    "MTBColibri",
    "MTBCondenserChanger",
    "MTBDefiniteFocus",
    "MTBFocus",
    "MTBObjectiveChanger",
    "MTBPiezo",
    "MTBReflectorChanger",
    "MTBSideportChanger",
    "MTBTLShutter",
    "MTBXYStage",
    "arrived",
    "changer_labels",
]

log = logging.getLogger("smc.hardware.zeiss_mtb")

T = TypeVar("T")

#: The Colibri's ``LED`` value that lights nothing.
NO_LED = "None"

#: DF2's ``Action`` values (m4 §11.4): it reads the action whose worker runs.
IDLE, LEARN, STABILIZE_NOW, STOP = "Idle", "Learn", "StabilizeNow", "Stop"
#: What ``Action`` reads while a restore (a set of ``Reference``) runs.
RESTORE = "Restore"
#: ``PiezoFocusUsage`` and ``Periodic``.
ON, OFF = "On", "Off"


def arrived(target_um: float, read_um: float, info: AxisInfo) -> bool:
    """Whether ``read_um`` is at ``target_um``, within what the axis declares it holds.

    The larger of the step and the typical deviation: a closed-loop piezo
    resolves 0.01 µm but dithers an order of magnitude more, and the step
    alone made its settles read as refusals (the predecessor's lesson). Never
    below 1e-6 µm, so that an axis that declares zero does not demand equality.
    """
    tolerance = max(info.step_um, info.typical_deviation_um, 1e-6)
    return abs(read_um - target_um) <= tolerance


def changer_labels(elements: tuple[ChangerElement, ...]) -> list[str]:
    """One unique label per element: its name stripped, ``Position-<n>`` when empty.

    A name two positions share (the condenser's and the reflector's "none")
    becomes ``"<name> (<n>)"`` for each, ``<n>`` MTB's 1-based position, so a
    preset and a log line name exactly one position (m4 §7.4).
    """
    names = [e.name.strip() or f"Position-{e.position}" for e in elements]
    counts: dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    return [
        f"{name} ({element.position})" if counts[name] > 1 else name
        for name, element in zip(names, elements, strict=True)
    ]


def _seam(what: str, call: Callable[[], T]) -> T:
    """``call()``, with any failure that is no ``MTBError`` wrapped in one naming ``what``."""
    try:
        return call()
    except MTBError:
        raise
    except Exception as exc:
        raise MTBError(f"{what}: {type(exc).__name__}: {exc}") from exc


def _initialized(label: str, setup: Callable[[], None]) -> None:
    """Run ``setup``; a failure is logged at ERROR, then raised as an ``MTBError``.

    unicore only logs a failing ``initialize()``; the opener then names the
    device and this line carries the cause (m4 §7.2 rule 1).
    """
    try:
        setup()
    except Exception as exc:
        error = (
            exc
            if isinstance(exc, MTBError)
            else MTBError(f"{label}: {type(exc).__name__}: {exc}")
        )
        log.error("%s did not initialise: %s", label, error)
        if error is exc:
            raise
        raise error from exc


class _Mover:
    """The worker thread of one device's moves, its own lock, and the outcome.

    One move at a time: a second one while the first runs is refused. A move
    that failed is kept until the next ``busy()`` after the worker has ended,
    which raises it once. ``what`` names a job in the log: DF2's are actions
    (a learn, a stabilize), not moves.
    """

    def __init__(self, name: str, what: str = "move") -> None:
        self._name = name
        self._what = what
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._error: MTBError | None = None
        self._cancel = threading.Event()
        self._waited = True
        self.target: object = None

    def running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def unwaited(self, target: object) -> bool:
        """Whether the last move went to ``target`` and no ``busy()`` has seen it end.

        Running, or ended with an outcome nobody has read yet: either way a
        repeat of that move is the same move, never a second one.
        """
        with self._lock:
            return not self._waited and self.target == target

    def start(self, target: object, work: Callable[[threading.Event], None]) -> None:
        """Run ``work`` on a new worker; ``MTBError`` if one already runs."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise MTBError(
                    f"{self._name} is moving to {self.target!r}; a move to "
                    f"{target!r} is refused until it ends"
                )
            self._error = None
            self._cancel = threading.Event()
            self._waited = False
            self.target = target
            cancel = self._cancel
            thread = threading.Thread(
                target=self._run,
                args=(work, cancel),
                name=f"smc-mtb-{self._name}",
                daemon=True,
            )
            self._thread = thread
        thread.start()

    def _run(
        self, work: Callable[[threading.Event], None], cancel: threading.Event
    ) -> None:
        try:
            work(cancel)
        except Exception as exc:
            error = (
                exc
                if isinstance(exc, MTBError)
                else MTBError(f"{self._name}: {type(exc).__name__}: {exc}")
            )
            if error is not exc:
                error.__cause__ = exc
            log.error("%s: the %s failed: %s", self._name, self._what, error)
            with self._lock:
                self._error = error

    def busy(self) -> bool:
        """``True`` while the worker runs; then the failure once, if the move failed."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return True
            self._waited = True
            error, self._error = self._error, None
        if error is not None:
            raise error
        return False

    def cancel(self) -> None:
        """Ask the worker not to send what it has not sent yet."""
        self._cancel.set()

    def join(self, timeout_s: float) -> bool:
        """Wait for the worker; ``True`` when none runs any more."""
        with self._lock:
            thread = self._thread
        if thread is None:
            return True
        thread.join(timeout_s)
        return not thread.is_alive()


def _move_axis(axis: MTBAxis, info: AxisInfo, name: str, target_um: float) -> None:
    """One synchronous MTB move, judged by its readback (m4 §7.3, FM-167)."""
    answer = _seam(
        f"{name}: SetPosition({target_um} µm)",
        lambda: axis.move_um(target_um, AXIS_TIMEOUT_MS),
    )
    read_um = _seam(f"{name}: GetPosition", axis.position_um)
    if answer is False:
        if arrived(target_um, read_um, info):
            return
        raise MTBError(
            f"{name}: MTB refused the move to {target_um} µm and the axis reads "
            f"{read_um} µm. Possible causes: an axis lock, a latched emergency "
            "stop, an active joystick, or for the focus, Definite Focus 2 "
            "holding the axis"
        )
    bound = max(info.max_deviation_um, info.step_um, info.typical_deviation_um, 1e-6)
    # `not <=`: a readback of NaN is no arrival (FM-167).
    if not abs(read_um - target_um) <= bound:
        raise MTBError(
            f"{name}: MTB moved to {target_um} µm but the axis reads {read_um} µm, "
            f"beyond its declared maximum deviation of {bound} µm"
        )


def _check_range(name: str, target_um: float, info: AxisInfo) -> None:
    """Refuse a target outside the axis's range; never clamp (a clamp reads as success)."""
    if not info.min_um <= target_um <= info.max_um:
        raise MTBError(
            f"{name}: {target_um} µm is outside the axis's range "
            f"[{info.min_um}, {info.max_um}] µm; nothing was sent"
        )


def _never_home(name: str) -> MTBError:
    return MTBError(f"{name}: smc never homes the stand nor redefines its origin")


def _refuse_stop(name: str) -> MTBError:
    """The refusal of a stop: MTB cannot stop a move it runs (m4 §7.2 rule 6).

    Raised, not only logged, so that the layer reads it as the refusal the
    profile declares (``stop_refused_by``) and never tells the operator that
    the stage stopped while it goes on to its target (FM-117).
    """
    message = f"{name}: MTB has no axis stop; the move already sent runs to its end"
    log.warning("%s", message)
    return MTBError(message)


def _shutdown_mover(name: str, mover: _Mover, timeout_ms: int) -> None:
    """Wait for a move still running, up to its MTB timeout; log, never raise.

    The light is darkened first when a move still runs: unicore unloads in
    load order and the lights come last in the ``.cfg``, so they would stay
    lit for the whole move MTB cannot stop (m4 §8).
    """
    if mover.running():
        try:
            mtb_session._darken_if_logged_in()
        except Exception as exc:  # never raised out of an unload
            log.error("%s: %s: %s", LIGHT_MAY_STILL_BE_ON, name, exc)
    if not mover.join(timeout_ms / 1000.0):
        log.error(
            "%s: a move still runs after %s ms; unloading without waiting for it",
            name,
            timeout_ms,
        )


class _MTBZAxis(StageDevice):
    """One MTB axis as a Micro-Manager stage; ``MTBFocus`` and ``MTBPiezo``."""

    MTB_ID: ClassVar[str]

    def __init__(self) -> None:
        super().__init__()
        self._axis: MTBAxis | None = None
        self._info: AxisInfo | None = None
        self._mover = _Mover(type(self).__name__)

    @property
    def _name(self) -> str:
        return self.get_label() or type(self).__name__

    def initialize(self) -> None:
        def setup() -> None:
            axis = mtb_session.shared().axis(self.MTB_ID)
            self._info = _seam(f"{self._name}: {self.MTB_ID} info", axis.info)
            self._axis = axis
            self._mover = _Mover(self._name)

        _initialized(self._name, setup)

    def _ready(self) -> tuple[MTBAxis, AxisInfo]:
        if self._axis is None or self._info is None:
            raise MTBError(f"{self._name} is not initialised")
        return self._axis, self._info

    def set_position_um(self, val: float) -> None:
        axis, info = self._ready()
        target_um = float(val)
        if self._mover.running():
            raise MTBError(
                f"{self._name} is moving; a move to {target_um} µm is refused"
            )
        _check_range(self._name, target_um, info)
        here = _seam(f"{self._name}: GetPosition", axis.position_um)
        if arrived(target_um, here, info):
            # MTB answers False for a zero-distance move: nothing to send.
            log.debug("%s: already at %s µm (reads %s)", self._name, target_um, here)
            return
        name = self._name
        self._mover.start(
            target_um, lambda _cancel: _move_axis(axis, info, name, target_um)
        )

    def get_position_um(self) -> float:
        axis, _ = self._ready()
        return float(_seam(f"{self._name}: GetPosition", axis.position_um))

    def busy(self) -> bool:
        return self._mover.busy()

    def home(self) -> None:
        raise _never_home(self._name)

    def set_origin(self) -> None:
        raise _never_home(self._name)

    def stop(self) -> None:
        self._mover.cancel()
        raise _refuse_stop(self._name)

    def shutdown(self) -> None:
        _shutdown_mover(self._name, self._mover, AXIS_TIMEOUT_MS)


class MTBFocus(_MTBZAxis):
    """The motorised focus drive (``MTBFocus``), ±14 000 µm."""

    MTB_ID: ClassVar[str] = "MTBFocus"


class MTBPiezo(_MTBZAxis):
    """The 500 µm piezo focus insert (``MTBPiezoFocusCan``)."""

    MTB_ID: ClassVar[str] = "MTBPiezoFocusCan"


class MTBXYStage(XYStageDevice):
    """The scanning stage: two MTB axes, moved X then Y in one worker."""

    X_ID: ClassVar[str] = "MTBStageAxisX"
    Y_ID: ClassVar[str] = "MTBStageAxisY"

    def __init__(self) -> None:
        super().__init__()
        self._axes: tuple[tuple[MTBAxis, AxisInfo], tuple[MTBAxis, AxisInfo]] | None = (
            None
        )
        self._mover = _Mover(type(self).__name__)

    @property
    def _name(self) -> str:
        return self.get_label() or type(self).__name__

    def initialize(self) -> None:
        """Find the session and both axes; nothing reaches MTB before this."""

        def setup() -> None:
            session = mtb_session.shared()
            x = session.axis(self.X_ID)
            y = session.axis(self.Y_ID)
            x_info = _seam(f"{self._name}: {self.X_ID} info", x.info)
            y_info = _seam(f"{self._name}: {self.Y_ID} info", y.info)
            self._axes = ((x, x_info), (y, y_info))
            self._mover = _Mover(self._name)

        _initialized(self._name, setup)

    def _ready(self) -> tuple[tuple[MTBAxis, AxisInfo], tuple[MTBAxis, AxisInfo]]:
        if self._axes is None:
            raise MTBError(f"{self._name} is not initialised")
        return self._axes

    def set_position_um(self, x: float, y: float) -> None:
        """Start the move on the worker and return; ``busy()`` is ``True`` until it ends."""
        (x_axis, x_info), (y_axis, y_info) = self._ready()
        x_um, y_um = float(x), float(y)
        if self._mover.running():
            raise MTBError(
                f"{self._name} is moving; a move to ({x_um}, {y_um}) µm is refused"
            )
        _check_range(f"{self._name} X", x_um, x_info)
        _check_range(f"{self._name} Y", y_um, y_info)
        legs: list[tuple[str, MTBAxis, AxisInfo, float]] = []
        for leg, axis, info, target in (
            ("X", x_axis, x_info, x_um),
            ("Y", y_axis, y_info, y_um),
        ):
            here = _seam(f"{self._name} {leg}: GetPosition", axis.position_um)
            if not arrived(target, here, info):
                legs.append((leg, axis, info, target))
        if not legs:
            log.debug("%s: already at (%s, %s) µm", self._name, x_um, y_um)
            return
        name = self._name

        def work(cancel: threading.Event) -> None:
            # X, then Y: the soft limits are a rectangle, so both legs of the
            # L stay inside it.
            for leg, axis, info, target in legs:
                if cancel.is_set():
                    log.warning("%s: stopped; the %s leg was not sent", name, leg)
                    return
                _move_axis(axis, info, f"{name} {leg}", target)

        self._mover.start((x_um, y_um), work)

    def get_position_um(self) -> tuple[float, float]:
        """Read both axes live, also during a move (FM-195)."""
        (x_axis, _), (y_axis, _) = self._ready()
        return (
            float(_seam(f"{self._name} X: GetPosition", x_axis.position_um)),
            float(_seam(f"{self._name} Y: GetPosition", y_axis.position_um)),
        )

    def busy(self) -> bool:
        """Whether the worker runs; then a failed move's ``MTBError``, once."""
        return self._mover.busy()

    def home(self) -> None:
        """Refused: smc never homes the stand."""
        raise _never_home(self._name)

    def set_origin(self) -> None:
        """Refused: smc never redefines the stand's origin."""
        raise _never_home(self._name)

    def set_origin_x(self) -> None:
        """Refused: smc never redefines the stand's origin."""
        raise _never_home(self._name)

    def set_origin_y(self) -> None:
        """Refused: smc never redefines the stand's origin."""
        raise _never_home(self._name)

    def stop(self) -> None:
        """Cancel the Y leg if it is not sent yet, then refuse: MTB cannot stop a move it runs."""
        self._mover.cancel()
        raise _refuse_stop(self._name)

    def shutdown(self) -> None:
        """Wait for a move still running, up to its MTB timeout; never raises."""
        _shutdown_mover(self._name, self._mover, AXIS_TIMEOUT_MS)


class _MTBChanger(StateDevice):
    """An MTB changer as a Micro-Manager state device (m4 §7.4).

    States are MTB positions minus ``MinPosition``. The device is built with
    one placeholder position, since unicore builds it with no arguments and
    the positions are MTB's; ``initialize()`` assigns every position and its
    label (measured on 0.18.1, m4 §1.2).
    """

    MTB_ID: ClassVar[str]

    def __init__(self) -> None:
        super().__init__({0: "Position-1"})
        self._changer: MTBChanger | None = None
        self._min_position = 1
        self._count = 1
        self._mover = _Mover(type(self).__name__)

    @property
    def _name(self) -> str:
        return self.get_label() or type(self).__name__

    def initialize(self) -> None:
        def setup() -> None:
            changer = mtb_session.shared().changer(self.MTB_ID)
            elements = _seam(f"{self._name}: {self.MTB_ID} elements", changer.elements)
            if not elements:
                raise MTBError(f"{self._name}: {self.MTB_ID} reports no position")
            self._min_position = elements[0].position
            for state, label in enumerate(changer_labels(elements)):
                self.assign_label_to_position(state, label)
            self._count = len(elements)
            self.set_property_allowed_values("State", list(range(self._count)))
            self._changer = changer
            self._mover = _Mover(self._name)

        _initialized(self._name, setup)

    def _ready(self) -> MTBChanger:
        if self._changer is None:
            raise MTBError(f"{self._name} is not initialised")
        return self._changer

    def get_state(self) -> int:
        changer = self._ready()
        position = _seam(f"{self._name}: Position", changer.position)
        return int(position) - self._min_position

    def set_state(self, position: int) -> None:
        changer = self._ready()
        state = int(position)
        if not 0 <= state < self._count:
            raise MTBError(
                f"{self._name}: state {state} is not one of its {self._count} positions"
            )
        target = state + self._min_position
        if self._mover.unwaited(target):
            # unicore's StateDevice sets State, then Label, and the Label
            # setter sets the State again within the same core call: the same
            # move, running or ended (a fast one, a refusal) with its outcome
            # still to be read by busy(). Sent again, it would hide that outcome.
            return
        if self._mover.running():
            raise MTBError(
                f"{self._name} is moving; a move to state {state} is refused"
            )
        if _seam(f"{self._name}: Position", changer.position) == target:
            return
        name = self._name

        def work(_cancel: threading.Event) -> None:
            answer = _seam(
                f"{name}: SetPosition({target})",
                lambda: changer.move(target, CHANGER_TIMEOUT_MS),
            )
            landed = _seam(f"{name}: Position", changer.position)
            if landed != target:
                said = "refused the move" if answer is False else "answered the move"
                raise MTBError(
                    f"{name}: MTB {said} to position {target} and the changer "
                    f"reads position {landed}. It may be locked, or the stand "
                    "may be blocking the change"
                )

        self._mover.start(target, work)

    def _set_current_label(self, label: str) -> None:
        """Set the state of ``label``; unicore's base compares with a live read.

        The base class sets the State again whenever the live state differs
        from the label's, which it does for the whole of a move made on a
        worker: it would recurse until the move ended. ``set_state`` takes a
        repeat of the move under way as done.
        """
        state = self._label_to_state.get(label)
        if state is not None:
            self.set_state(state)

    def busy(self) -> bool:
        return self._mover.busy()

    def shutdown(self) -> None:
        _shutdown_mover(self._name, self._mover, CHANGER_TIMEOUT_MS)


class MTBObjectiveChanger(_MTBChanger):
    """The objective turret (``MTBObjectiveChanger``)."""

    MTB_ID: ClassVar[str] = "MTBObjectiveChanger"


class MTBReflectorChanger(_MTBChanger):
    """The reflector turret, the filter cubes (``MTBReflectorChanger``)."""

    MTB_ID: ClassVar[str] = "MTBReflectorChanger"


class MTBSideportChanger(_MTBChanger):
    """The sideport turret; the camera is on the left sideport (``MTBSideportChanger``)."""

    MTB_ID: ClassVar[str] = "MTBSideportChanger"


class MTBCondenserChanger(_MTBChanger):
    """The condenser's contrast turret (``MTBCondenserContrastChanger``)."""

    MTB_ID: ClassVar[str] = "MTBCondenserContrastChanger"


class MTBTLShutter(ShutterDevice):
    """The transmitted-light shutter, a two-position changer whose *open* is found by name.

    The predecessor's ``set_open`` raised at every exposure when the names did
    not match, the caller swallowed it, and the TL light would never have been
    gated for a whole overnight run. Here a shutter that cannot be gated fails
    its initialise (m4 §7.5).
    """

    MTB_ID: ClassVar[str] = "MTBTLShutter"

    def __init__(self) -> None:
        super().__init__()
        self._changer: MTBChanger | None = None
        self._session: mtb_session.MTBSession | None = None
        self._open_position = 0
        self._closed_position = 0

    @property
    def _name(self) -> str:
        return self.get_label() or type(self).__name__

    def initialize(self) -> None:
        """Find *open* by name, then close the shutter before anything else."""

        def setup() -> None:
            session = mtb_session.shared()
            changer = session.changer(self.MTB_ID)
            elements = _seam(f"{self._name}: {self.MTB_ID} elements", changer.elements)
            names = [e.name for e in elements]
            opens = [e for e in elements if e.name.strip().casefold() == "open"]
            if len(elements) != 2 or len(opens) != 1:
                raise MTBError(
                    f"{self._name}: a TL shutter needs exactly two positions, one "
                    f"named 'open'; {self.MTB_ID} has {names!r}, so the transmitted "
                    "light could not be gated"
                )
            self._open_position = opens[0].position
            self._closed_position = next(
                e.position for e in elements if e.position != self._open_position
            )
            self._changer = changer
            self._darken()
            session.register_darkener(self._name, self._darken)
            self._session = session

        _initialized(self._name, setup)

    def _ready(self) -> MTBChanger:
        if self._changer is None:
            raise MTBError(f"{self._name} is not initialised")
        return self._changer

    def _move(self, position: int) -> None:
        changer = self._ready()
        if _seam(f"{self._name}: Position", changer.position) == position:
            return
        _seam(
            f"{self._name}: SetPosition({position})",
            lambda: changer.move(position, CHANGER_TIMEOUT_MS),
        )
        landed = _seam(f"{self._name}: Position", changer.position)
        if landed != position:
            raise MTBError(
                f"{self._name}: MTB did not move the TL shutter to position "
                f"{position}; it reads position {landed}"
            )

    def _darken(self) -> None:
        self._move(self._closed_position)

    def get_open(self) -> bool:
        """Whether MTB reads the shutter at its *open* position, live."""
        changer = self._ready()
        return bool(
            _seam(f"{self._name}: Position", changer.position) == self._open_position
        )

    def set_open(self, open: bool) -> None:
        """Move to the open or the closed position synchronously and check the readback."""
        self._move(self._open_position if open else self._closed_position)

    def shutdown(self) -> None:
        """Close the shutter, then forget its darkener; never raises."""
        _darken_at_unload(
            self._name,
            self._darken if self._changer is not None else None,
            self._session,
        )


def _darken_at_unload(
    label: str,
    darken: Callable[[], None] | None,
    session: mtb_session.MTBSession | None,
) -> None:
    """Darken, then forget the darkener; a failure is an ERROR, never raised (m4 §8.2).

    The session is the one ``initialize()`` found, never ``shared()`` again: an
    unload must not log in, and a device that never initialised has nothing
    to darken.
    """
    if session is not None and session._closed:
        # The exit handler darkened and logged out; MTB can take no call now.
        log.debug(
            "%s: unloaded after the MTB logout; the exit handler darkened it", label
        )
        return
    if darken is not None:
        try:
            darken()
        except Exception as exc:
            log.error("%s: %s: %s", LIGHT_MAY_STILL_BE_ON, label, exc)
    if session is not None:
        session.unregister_darkener(label)


def _switch_off(name: str, key: str, led: MTBLED) -> None:
    _seam(f"{name}: {key} SetOnOff(Off)", lambda: led.set_on(False, LED_TIMEOUT_MS))


class MTBColibri(ShutterDevice):
    """The Colibri, one LED at a time: ``LED`` picks it, ``Intensity`` sets its % (m4 §7.6).

    A ``ShutterDevice``, so that Micro-Manager's auto-shutter, which limits the
    light to the exposure, opens it, and the layer's light source switches it.
    """

    MTB_ID: ClassVar[str] = "MTBFLLEDController"

    def __init__(self) -> None:
        super().__init__()
        self._controller: MTBLEDController | None = None
        self._session: mtb_session.MTBSession | None = None
        self._leds: dict[str, MTBLED] = {}
        self._led = NO_LED
        self._intensity = 0.0
        self._lit: MTBLED | None = None
        cls = type(self)
        self.register_property(
            "LED",
            default_value=NO_LED,
            allowed_values=[NO_LED],
            getter=cls._get_led,
            setter=cls._set_led,
        )
        self.register_property(
            "Intensity",
            default_value=0.0,
            limits=(0.0, 100.0),
            getter=cls._get_intensity,
            setter=cls._set_intensity,
        )

    @property
    def _name(self) -> str:
        return self.get_label() or type(self).__name__

    def initialize(self) -> None:
        """Find the LEDs and switch every one off before anything else."""

        def setup() -> None:
            session = mtb_session.shared()
            controller = session.led_controller(self.MTB_ID)
            leds = _seam(f"{self._name}: {self.MTB_ID} LEDs", controller.leds)
            self._leds = {f"{led.wavelength_nm}nm": led for led in leds}
            self.set_property_allowed_values("LED", [NO_LED, *self._leds])
            self._controller = controller
            self._darken()
            session.register_darkener(self._name, self._darken)
            self._session = session

        _initialized(self._name, setup)

    def _ready(self) -> MTBLEDController:
        if self._controller is None:
            raise MTBError(f"{self._name} is not initialised")
        return self._controller

    def _darken(self) -> None:
        """Every LED off, not only the selected one, each tried (FM-49); one error for all."""
        self._ready()
        self._lit = None
        failures: list[str] = []
        for key, led in self._leds.items():
            try:
                _switch_off(self._name, key, led)
            except MTBError as exc:
                failures.append(str(exc))
        if failures:
            raise MTBError(
                f"{self._name}: {len(failures)} LED(s) did not switch off: "
                + "; ".join(failures)
            )

    def _light(self, key: str) -> None:
        """Light ``key`` alone at the intensity; any failure darkens every LED and raises."""
        led = self._leds[key]
        try:
            controller = self._ready()
            _seam(
                f"{self._name}: manual mode",
                lambda: controller.set_manual_mode(LED_TIMEOUT_MS),
            )
            for other_key, other in self._leds.items():
                if other is not led:
                    _switch_off(self._name, other_key, other)
            intensity = self._intensity
            _seam(
                f"{self._name}: {key} intensity {intensity}%",
                lambda: led.set_intensity_pct(intensity, LED_TIMEOUT_MS),
            )
            _seam(
                f"{self._name}: {key} SetOnOff(On)",
                lambda: led.set_on(True, LED_TIMEOUT_MS),
            )
            if not _seam(f"{self._name}: {key} GetOnOff", led.is_on):
                raise MTBError(
                    f"{self._name}: {key} reads off after it was switched on"
                )
        except MTBError as exc:
            try:
                self._darken()
            except MTBError as dark:
                log.error("%s: %s: %s", LIGHT_MAY_STILL_BE_ON, self._name, dark)
                raise MTBError(f"{exc}; then {dark}") from exc
            raise
        self._lit = led

    def get_open(self) -> bool:
        """``True`` when any LED reads on, so light from any LED reads as open."""
        self._ready()
        return any(
            _seam(f"{self._name}: {key} GetOnOff", led.is_on)
            for key, led in self._leds.items()
        )

    def set_open(self, open: bool) -> None:
        """Light the selected LED alone, or switch every LED off."""
        self._ready()
        if not open:
            self._darken()
            return
        if self._led == NO_LED:
            raise MTBError(
                f"{self._name}: no LED is selected (property LED is {NO_LED!r}); "
                "choose one before opening"
            )
        self._light(self._led)

    def _get_led(self) -> str:
        return self._led

    def _set_led(self, value: str) -> None:
        previous, self._led = self._led, str(value)
        if self._lit is None:
            return
        if self._led == NO_LED:
            self._darken()
        elif self._led != previous:
            self._light(self._led)

    def _get_intensity(self) -> float:
        return self._intensity

    def _set_intensity(self, value: float) -> None:
        self._intensity = float(value)
        lit = self._lit
        if lit is None:
            return
        pct = self._intensity
        _seam(
            f"{self._name}: {self._led} intensity {pct}%",
            lambda: lit.set_intensity_pct(pct, LED_TIMEOUT_MS),
        )

    def shutdown(self) -> None:
        """Switch every LED off, then forget the darkener; never raises."""
        _darken_at_unload(
            self._name,
            self._darken if self._controller is not None else None,
            self._session,
        )


class MTBDefiniteFocus(GenericDevice):
    """Definite Focus 2 as five properties: learn, restore, stabilize once (m4 §11.4).

    unicore has no autofocus device type, so DF2 is a ``GenericDevice``, and
    the layer's ``MMFocusStabilizer`` drives it through ``Action``,
    ``Reference``, ``OperationMode``, ``PiezoFocusUsage`` and ``Periodic``.

    Hardware behaviour encoded here:

    * **Every MTB call of DF2 is synchronous** and may take seconds, so a
      learn, a restore and a stabilize run on the device's worker (``_Mover``)
      and the set returns at once; ``busy()`` is ``True`` while one runs and
      raises its failure once after. Reads never wait behind it.
    * **A ``False`` from MTB is a failure**, never a success: the predecessor
      ignored the answers and imaged at the commanded Z. Whether DF2 answers
      ``False`` on a success is not measured (m4 §12).
    * **What DF2 held before this process is not this process's to trust**
      (FM-118): ``Reference`` reads ``""`` after ``initialize()``, from the
      start of every learn or restore, and after one that failed.
    * **Periodic stabilisation is never switched on**: it would hold one
      plane while the stage visits others. ``initialize()`` and ``shutdown()``
      switch it off when another MTB client left it on.
    * **``SignalQuality`` is not read**: it reads 0 while DF2 works (m4 §11.1).
    """

    MTB_ID: ClassVar[str] = "MTBFocusStabilizer2"

    def __init__(self) -> None:
        super().__init__()
        self._stabilizer: MTBFocusStabilizer | None = None
        self._session: mtb_session.MTBSession | None = None
        self._mover = _Mover(type(self).__name__, "action")
        #: The reference learned or restored since ``initialize()``, as base64.
        self._reference = ""
        cls = type(self)
        self.register_property(
            "Action",
            default_value=IDLE,
            allowed_values=[IDLE, LEARN, STABILIZE_NOW, STOP],
            getter=cls._get_action,
            setter=cls._set_action,
        )
        self.register_property(
            "Reference",
            default_value="",
            getter=cls._get_reference,
            setter=cls._set_reference,
        )
        self.register_property(
            "OperationMode",
            default_value="Balanced",
            allowed_values=list(STABILIZER_OPERATION_MODES),
            getter=cls._get_operation_mode,
            setter=cls._set_operation_mode,
        )
        self.register_property(
            "PiezoFocusUsage",
            default_value=OFF,
            allowed_values=[ON, OFF],
            getter=cls._get_piezo_focus_usage,
            is_read_only=True,
        )
        self.register_property(
            "Periodic",
            default_value=OFF,
            allowed_values=[ON, OFF],
            getter=cls._get_periodic,
            setter=cls._set_periodic,
        )

    @property
    def _name(self) -> str:
        return self.get_label() or type(self).__name__

    def initialize(self) -> None:
        """Find DF2, read its operation mode, and switch its periodic mode off if on."""

        def setup() -> None:
            session = mtb_session.shared()
            stabilizer = session.focus_stabilizer(self.MTB_ID)
            name = self._name
            mode = _seam(
                f"{name}: {self.MTB_ID} operation mode", stabilizer.operation_mode
            )
            if _seam(f"{name}: {self.MTB_ID} periodic", stabilizer.periodic):
                _seam(
                    f"{name}: StabilizePeriodically(Off)",
                    lambda: stabilizer.stop_periodic(STABILIZER_TIMEOUT_MS),
                )
                log.warning(
                    "%s: periodic stabilisation was on (another MTB client left it "
                    "on), and was switched off: smc stabilizes one shot at a time "
                    "(m4 §11.4)",
                    name,
                )
            self._stabilizer = stabilizer
            self._session = session
            self._reference = ""
            self._mover = _Mover(name, "action")
            log.info("%s: Definite Focus 2, operation mode %s", name, mode)

        _initialized(self._name, setup)

    def _ready(self) -> MTBFocusStabilizer:
        if self._stabilizer is None:
            raise MTBError(f"{self._name} is not initialised")
        return self._stabilizer

    def _refuse_while_running(self, action: str) -> None:
        if self._mover.running():
            raise MTBError(
                f"{self._name}: {self._mover.target} is running; {action} is "
                "refused until it ends; nothing was sent"
            )

    def busy(self) -> bool:
        """``True`` while an action runs on the worker; then its failure, once."""
        return self._mover.busy()

    # --- Action ---------------------------------------------------------------

    def _get_action(self) -> str:
        if self._mover.running():
            return str(self._mover.target)
        return IDLE

    def _set_action(self, value: str) -> None:
        action = str(value)
        if action == IDLE:
            return
        stabilizer = self._ready()
        if action == STOP:
            self._send_stop(stabilizer)
            return
        self._refuse_while_running(action)
        name = self._name
        if action == LEARN:
            self._reference = ""

            def learn(_cancel: threading.Event) -> None:
                blob = _seam(
                    f"{name}: InitOnCurrentFocusPosition",
                    lambda: stabilizer.learn(STABILIZER_TIMEOUT_MS),
                )
                if not blob:
                    raise MTBError(f"{name}: DF2 learned an empty reference")
                self._reference = base64.b64encode(bytes(blob)).decode("ascii")

            self._mover.start(LEARN, learn)
            return

        def stabilize(cancel: threading.Event) -> None:
            answer = _seam(
                f"{name}: StabilizeNow",
                lambda: stabilizer.stabilize_now(STABILIZER_TIMEOUT_MS),
            )
            if not answer:
                after = ", after a Stop was sent during it" if cancel.is_set() else ""
                raise MTBError(
                    f"{name}: DF2 answered False to StabilizeNow{after}; the focus "
                    "drive is where DF2 left it"
                )

        self._mover.start(STABILIZE_NOW, stabilize)

    def _send_stop(self, stabilizer: MTBFocusStabilizer) -> None:
        """``Stop``, in the set and not on the worker, even while the worker runs (rule 4)."""
        running = self._mover.running()
        if running:
            self._mover.cancel()
        _seam(f"{self._name}: Stop", lambda: stabilizer.stop(STABILIZER_TIMEOUT_MS))
        if running:
            log.warning(
                "%s: Stop sent while %s runs; whether DF2 ends a synchronous call "
                "from another thread is not known (m4 §12)",
                self._name,
                self._mover.target,
            )

    # --- Reference ------------------------------------------------------------

    def _get_reference(self) -> str:
        return self._reference

    def _set_reference(self, value: str) -> None:
        """A restore: the text is decoded here, before anything is sent, then sent on the worker."""
        stabilizer = self._ready()
        name, text = self._name, str(value)
        if not text:
            raise MTBError(
                f"{name}: an empty Reference is refused: learn one (Action Learn) "
                "or set one learned before; nothing was sent"
            )
        try:
            blob = base64.b64decode(text, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise MTBError(
                f"{name}: the Reference is not base64 text ({exc}); nothing was sent"
            ) from exc
        if not blob:
            raise MTBError(
                f"{name}: the Reference decodes to nothing; nothing was sent"
            )
        self._refuse_while_running(RESTORE)
        self._reference = ""

        def restore(_cancel: threading.Event) -> None:
            answer = _seam(
                f"{name}: InitWithStabilizingData",
                lambda: stabilizer.restore(blob, STABILIZER_TIMEOUT_MS),
            )
            if not answer:
                raise MTBError(
                    f"{name}: DF2 answered False to InitWithStabilizingData: the "
                    "reference was not restored, and DF2 holds none that smc knows"
                )
            self._reference = text

        self._mover.start(RESTORE, restore)

    # --- OperationMode, PiezoFocusUsage, Periodic: MTB's, live ----------------

    def _get_operation_mode(self) -> str:
        stabilizer = self._ready()
        return _seam(f"{self._name}: GetOperationMode", stabilizer.operation_mode)

    def _set_operation_mode(self, value: str) -> None:
        stabilizer = self._ready()
        mode = str(value)
        self._refuse_while_running(f"OperationMode {mode}")
        _seam(
            f"{self._name}: SetOperationMode({mode})",
            lambda: stabilizer.set_operation_mode(mode),
        )

    def _get_piezo_focus_usage(self) -> str:
        stabilizer = self._ready()
        uses = _seam(f"{self._name}: GetPiezoFocusUsage", stabilizer.piezo_focus_usage)
        return ON if uses else OFF

    def _get_periodic(self) -> str:
        stabilizer = self._ready()
        on = _seam(f"{self._name}: GetStabilizePeriodicallyOnOff", stabilizer.periodic)
        return ON if on else OFF

    def _set_periodic(self, value: str) -> None:
        stabilizer = self._ready()
        if str(value) == ON:
            raise MTBError(
                f"{self._name}: smc never switches DF2's periodic stabilisation on: "
                "it would hold one plane while the stage visits others (m4 §11.2); "
                "nothing was sent"
            )
        _seam(
            f"{self._name}: StabilizePeriodically(Off)",
            lambda: stabilizer.stop_periodic(STABILIZER_TIMEOUT_MS),
        )

    def shutdown(self) -> None:
        """Stop a running action and wait for it, then periodic off if on; never raises."""
        name, stabilizer, session = self._name, self._stabilizer, self._session
        if stabilizer is None:
            return
        if session is not None and session._closed:
            log.debug("%s: unloaded after the MTB logout; nothing to stop", name)
            return
        if self._mover.running():
            try:
                self._send_stop(stabilizer)
            except Exception as exc:
                log.error("%s: Stop at unload failed: %s", name, exc)
        _shutdown_mover(name, self._mover, STABILIZER_TIMEOUT_MS)
        try:
            if stabilizer.periodic():
                stabilizer.stop_periodic(STABILIZER_TIMEOUT_MS)
                log.warning("%s: periodic stabilisation switched off at unload", name)
        except Exception as exc:
            log.error(
                "%s: periodic stabilisation could not be read or switched off at "
                "unload: %s",
                name,
                exc,
            )
