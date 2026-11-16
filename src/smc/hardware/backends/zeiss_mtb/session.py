"""The process's one MTB session: logged in once, logged out only at exit (m4 §6).

MTB tears its realtime subsystem down on ``Logout`` and cannot bring it back in
the same process: a second ``Login`` fails with ``ConnectToRtSystem() …
E_NOINTERFACE`` (observed 2026-09-04, ``lightsheet-live-tracking-tool``). So a
process has exactly one session (ADR-0007 point 3), shared by every device,
and nothing logs out but the process's exit: not ``Microscope.close()``, not a
device's ``shutdown()``, not *Disconnect*, since any of those would leave the
next *Connect* of the same ``smc ui`` unable to reach the stand.

The exit handler is also where the light is switched off for the last time
(m4 §8): every darkener a device registered, then every LED of every
controller the session handed out, then the logout. Each step is tried
whatever the others do, and a failure is an ERROR saying ``LIGHT MAY STILL BE
ON``: a real overnight run once killed its embryos with light left on
outside the exposure.
"""

from __future__ import annotations

import atexit
import logging
import threading
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import TypeVar

from smc.hardware.backends.zeiss_mtb.api import (
    LED_TIMEOUT_MS,
    MTBLED,
    AxisInfo,
    ChangerElement,
    MTBApi,
    MTBAxis,
    MTBChanger,
    MTBError,
    MTBLEDController,
    MTBSessionError,
)

__all__ = ["LIGHT_MAY_STILL_BE_ON", "MTBSession", "shared", "use_api_factory"]

log = logging.getLogger("smc.hardware.zeiss_mtb")

#: The words an operator looks for in the log when a switch-off failed.
LIGHT_MAY_STILL_BE_ON = "LIGHT MAY STILL BE ON"

T = TypeVar("T")

_LOGGED_OUT = "MTB was logged out; it cannot log in twice in one process"

#: Guards the session's creation, so that devices initialised in parallel
#: (unicore initialises on a thread pool) log in once.
_lock = threading.Lock()
_session: MTBSession | None = None
_logged_out = False
#: ``use_api_factory``'s factory while its block runs; ``None`` means .NET.
_factory: Callable[[], MTBApi] | None = None


def _default_factory() -> MTBApi:
    """``dotnet.connect()``, imported here so that nothing else imports pythonnet."""
    from smc.hardware.backends.zeiss_mtb import dotnet

    return dotnet.connect()


class MTBSession:
    """The logged-in MTB connection, its components cached by id, and the darkeners.

    Built only by :func:`shared`. The components it hands out refuse every
    call with an ``MTBSessionError`` once the session has logged out.
    """

    def __init__(self, api: MTBApi, factory: Callable[[], MTBApi]) -> None:
        self._api = api
        self._factory = factory
        self._lock = threading.Lock()
        self._axes: dict[str, MTBAxis] = {}
        self._changers: dict[str, MTBChanger] = {}
        self._controllers: dict[str, MTBLEDController] = {}
        #: The raw controllers, which the exit handler darkens after the logout flag.
        self._raw_controllers: dict[str, MTBLEDController] = {}
        self._darkeners: dict[str, Callable[[], None]] = {}
        self._closed = False
        self._closing = False

    def _check(self) -> None:
        if self._closed:
            raise MTBSessionError(_LOGGED_OUT)

    def axis(self, mtb_id: str) -> MTBAxis:
        """The axis ``mtb_id``, looked up once; an ``MTBError`` naming it when absent."""
        with self._lock:
            self._check()
            if mtb_id not in self._axes:
                self._axes[mtb_id] = _GuardedAxis(
                    self, _wrapped(self._api.axis, mtb_id)
                )
            return self._axes[mtb_id]

    def changer(self, mtb_id: str) -> MTBChanger:
        """The changer ``mtb_id``, looked up once; an ``MTBError`` naming it when absent."""
        with self._lock:
            self._check()
            if mtb_id not in self._changers:
                raw = _wrapped(self._api.changer, mtb_id)
                self._changers[mtb_id] = _GuardedChanger(self, raw)
            return self._changers[mtb_id]

    def led_controller(self, mtb_id: str) -> MTBLEDController:
        """The LED controller ``mtb_id``, looked up once; darkened at exit."""
        with self._lock:
            self._check()
            if mtb_id not in self._controllers:
                raw = _wrapped(self._api.led_controller, mtb_id)
                self._raw_controllers[mtb_id] = raw
                self._controllers[mtb_id] = _GuardedController(self, raw)
            return self._controllers[mtb_id]

    def register_darkener(self, label: str, darken: Callable[[], None]) -> None:
        """Run ``darken`` at process exit, before the logout; one per label."""
        with self._lock:
            self._darkeners[label] = darken

    def unregister_darkener(self, label: str) -> None:
        """Forget ``label``'s darkener; a label without one is ignored."""
        with self._lock:
            self._darkeners.pop(label, None)

    def _run_darkeners(self) -> BaseException | None:
        """Run every registered darkener, each tried; a failure is an ERROR.

        Returns a ``KeyboardInterrupt`` (or other ``BaseException``) met on the
        way, after every darkener has been tried, so that the caller still cuts
        the light and then lets the operator's stop through (FM-49, FM-71).
        """
        with self._lock:
            darkeners = list(self._darkeners.items())
        interrupted: BaseException | None = None
        for label, darken in darkeners:
            try:
                darken()
            except Exception as exc:
                log.error("%s: %s: %s", LIGHT_MAY_STILL_BE_ON, label, exc)
            except BaseException as exc:
                log.error("%s: %s: interrupted (%r)", LIGHT_MAY_STILL_BE_ON, label, exc)
                interrupted = interrupted or exc
        return interrupted

    def _darken_now(self) -> None:
        """Darken every lit device now, before an unload waits for a move (m4 §8).

        An unload goes in load order, and an axis or a changer waits there for
        a move MTB cannot stop, up to its timeout: the light must not wait with
        it. Never raises a failure; an interrupt goes on once all were tried.
        """
        interrupted = self._run_darkeners()
        if interrupted is not None:
            raise interrupted

    def _close_at_exit(self) -> None:
        """Darken, switch every LED off, then log out; each step tried, no failure raised.

        The darkeners run while the session still serves: they go through the
        components it handed out, which refuse every call once it is closed.
        Only then is it closed and logged out. A ``KeyboardInterrupt`` met on
        the way is raised after the logout, so the light is still cut and the
        operator's stop is not lost (FM-49, FM-71).

        Idempotent: ``use_api_factory`` runs it and unregisters it from
        ``atexit``, and a second run finds it already run.
        """
        global _logged_out
        with self._lock:
            if self._closing:
                return
            self._closing = True
            controllers = list(self._raw_controllers.items())
        interrupted = self._run_darkeners()
        for mtb_id, controller in controllers:
            try:
                _all_off(controller, mtb_id)
            except BaseException as exc:
                interrupted = interrupted or exc
        with self._lock:
            self._closed = True
            self._darkeners.clear()
        _logged_out = True
        try:
            self._api.logout()
        except Exception as exc:
            log.error("MTB logout failed at exit: %s", exc)
        else:
            log.info("MTB logged out")
        if interrupted is not None:
            raise interrupted


def _all_off(controller: MTBLEDController, mtb_id: str) -> None:
    """Every LED of ``controller`` off, each tried; a failure is an ERROR (FM-49)."""
    try:
        leds = controller.leds()
    except Exception as exc:
        log.error(
            "%s: %s: could not list the LEDs: %s", LIGHT_MAY_STILL_BE_ON, mtb_id, exc
        )
        return
    for led in leds:
        try:
            led.set_on(False, LED_TIMEOUT_MS)
        except Exception as exc:
            log.error(
                "%s: %s LED %s nm: %s",
                LIGHT_MAY_STILL_BE_ON,
                mtb_id,
                getattr(led, "wavelength_nm", "?"),
                exc,
            )


def _wrapped(lookup: Callable[[str], T], mtb_id: str) -> T:
    """``lookup(mtb_id)``, with any failure that is no ``MTBError`` wrapped in one (FM-113)."""
    try:
        return lookup(mtb_id)
    except MTBError:
        raise
    except Exception as exc:
        raise MTBError(f"MTB component {mtb_id!r}: lookup failed: {exc}") from exc


class _GuardedAxis:
    """An axis that refuses every call once the session has logged out."""

    def __init__(self, session: MTBSession, raw: MTBAxis) -> None:
        self._session = session
        self._raw = raw
        self.mtb_id = raw.mtb_id

    def info(self) -> AxisInfo:
        self._session._check()
        return self._raw.info()

    def position_um(self) -> float:
        self._session._check()
        return self._raw.position_um()

    def move_um(self, target_um: float, timeout_ms: int) -> bool:
        self._session._check()
        return self._raw.move_um(target_um, timeout_ms)


class _GuardedChanger:
    """A changer that refuses every call once the session has logged out."""

    def __init__(self, session: MTBSession, raw: MTBChanger) -> None:
        self._session = session
        self._raw = raw
        self.mtb_id = raw.mtb_id

    def elements(self) -> tuple[ChangerElement, ...]:
        self._session._check()
        return self._raw.elements()

    def position(self) -> int:
        self._session._check()
        return self._raw.position()

    def move(self, position: int, timeout_ms: int) -> bool:
        self._session._check()
        return self._raw.move(position, timeout_ms)


class _GuardedLED:
    """An LED that refuses every call once the session has logged out."""

    def __init__(self, session: MTBSession, raw: MTBLED) -> None:
        self._session = session
        self._raw = raw
        self.wavelength_nm = raw.wavelength_nm

    def is_on(self) -> bool:
        self._session._check()
        return self._raw.is_on()

    def set_on(self, on: bool, timeout_ms: int) -> None:
        self._session._check()
        self._raw.set_on(on, timeout_ms)

    def intensity_pct(self) -> float:
        self._session._check()
        return self._raw.intensity_pct()

    def set_intensity_pct(self, pct: float, timeout_ms: int) -> None:
        self._session._check()
        self._raw.set_intensity_pct(pct, timeout_ms)


class _GuardedController:
    """An LED controller that refuses every call once the session has logged out."""

    def __init__(self, session: MTBSession, raw: MTBLEDController) -> None:
        self._session = session
        self._raw = raw
        self.mtb_id = raw.mtb_id
        self._leds: tuple[MTBLED, ...] | None = None

    def leds(self) -> tuple[MTBLED, ...]:
        self._session._check()
        if self._leds is None:
            self._leds = tuple(
                _GuardedLED(self._session, led) for led in self._raw.leds()
            )
        return self._leds

    def set_manual_mode(self, timeout_ms: int) -> None:
        self._session._check()
        self._raw.set_manual_mode(timeout_ms)


def _darken_if_logged_in() -> None:
    """Run the darkeners of the process's session, if it has one; never logs in.

    For a device's unload, which must neither log in nor wait for a move
    with the light on (m4 §8).
    """
    with _lock:
        session = _session
    if session is not None and not session._closing:
        session._darken_now()


def shared() -> MTBSession:
    """The process's session, logging in on the first call.

    A login that fails leaves no session, so the next call tries again (the
    stand was off; it is now on). Only a logout is final.

    Raises:
        MTBSessionError: MTB was logged out in this process, or the
            connection could not be made (pythonnet or the DLL missing).
        MTBError: MTB refused the login.
    """
    global _session
    with _lock:
        if _logged_out:
            raise MTBSessionError(_LOGGED_OUT)
        if _session is not None:
            return _session
        factory = _factory if _factory is not None else _default_factory
        try:
            api = factory()
            api.login()
        except MTBError:
            raise
        except Exception as exc:
            raise MTBSessionError(f"MTB login failed: {exc}") from exc
        session = MTBSession(api, factory)
        atexit.register(session._close_at_exit)
        _session = session
        log.info("MTB logged in")
        return session


def use_api_factory(factory: Callable[[], MTBApi]) -> AbstractContextManager[None]:
    """For tests only: :func:`shared` builds its session from ``factory`` inside the block.

    On exit it runs the session's exit handler (darken, logout), unregisters it
    from ``atexit`` and forgets the session, so the next block starts clean
    (FM-46). Production code never calls it.

    Raises:
        MTBSessionError: On entry, while a session made by another factory
            exists, or another block is open.
    """
    return _using(factory)


@contextmanager
def _using(factory: Callable[[], MTBApi]) -> Iterator[None]:
    global _factory, _session, _logged_out
    with _lock:
        if _factory is not None or (
            _session is not None and _session._factory is not factory
        ):
            raise MTBSessionError(
                "an MTB session made by another factory exists; a test can only "
                "swap the API before the process has logged in"
            )
        _factory = factory
    try:
        yield
    finally:
        with _lock:
            session = _session
        try:
            if session is not None:
                atexit.unregister(session._close_at_exit)
                session._close_at_exit()
        finally:
            with _lock:
                _session = None
                _logged_out = False
                _factory = None
