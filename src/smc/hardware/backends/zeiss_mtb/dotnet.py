"""The MTB seam over ``MTBApi.dll``, through pythonnet (m4 §5).

Ported from ``lightsheet-live-tracking-tool``'s
``tracking_tools/microscope_interface/mtb.py`` (HEAD f476600), which drove
this stand daily. What it learned at the stand, kept here:

* ``MTBConnection().Login("en", "")`` hands the client id back as the return
  value or inside a tuple, by pythonnet version; ``GetRoot(client_id)``
  initialises **all** the hardware and fails when the stand is off.
* ``root.GetComponent(id)`` returns concrete ``MTBCtrl*`` objects whose
  members appear only after an interface cast (``IMTBContinual``,
  ``IMTBChanger``, ``IMTBFluorescenceLEDControl`` …); ``None`` is an absent
  component.
* **pythonnet ≥ 3 refuses int → Enum**: ``MTBCmdSetModes.Synchronous`` is the
  enum member, never ``1`` (observed 2026-09-04).
* The µm unit is found among ``GetPositionUnit(i)``; the focus also offers nm.
* A changer's ``Position`` is 1-based and ``GetElement`` is **0-based**: the
  name of position ``p`` is ``GetElement(p - 1)``; reading ``GetElement(p)``
  mislabelled every objective (2026-09-08).
* ``SetOnOff`` on an LED failed with three guessed argument shapes; its
  arguments are built from the overload's reflected signature (``_args_for``).
* Definite Focus 2's members appear after the cast ``IMTBFocusStabilizer2``,
  and its learn hands the reference back through a ``Byte[]&``
  out-parameter, which pythonnet appends to the returned tuple (2026-09-11).

The adapters take the .NET objects, the ``ZEISS.MTB.Api`` namespace and
``System.Enum`` as arguments, so the tests drive them with duck-typed fakes
on any OS; only :func:`connect` needs Windows. No clamping and no tolerance
logic here: the device judges MTB's answer (m4 §7.3). Every .NET call is
wrapped, so any exception becomes an ``MTBError`` naming the component and
the call (pythonnet raises .NET exceptions that are no ``RuntimeError``,
FM-113).
"""

from __future__ import annotations

import functools
import importlib
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypeVar

from smc.hardware.backends.zeiss_mtb.api import (
    STABILIZER_OPERATION_MODES,
    AxisInfo,
    ChangerElement,
    MTBApi,
    MTBError,
    MTBSessionError,
)

__all__ = [
    "DEFAULT_MTB_API",
    "DotNetAxis",
    "DotNetChanger",
    "DotNetFocusStabilizer",
    "DotNetLED",
    "DotNetLEDController",
    "DotNetMTB",
    "connect",
    "resolve_mode",
]

log = logging.getLogger("smc.hardware.zeiss_mtb")

T = TypeVar("T")

DEFAULT_MTB_API = Path(
    r"C:\Program Files\Carl Zeiss\MTB 2011 - 2.12.0.7\MTBApi\MTBApi.dll"
)

#: How MTB may spell a micrometre (the stand says ``µm``).
_UM_ALIASES = (
    "um",
    "\N{MICRO SIGN}m",
    "\N{GREEK SMALL LETTER MU}m",
    "micron",
    "micrometer",
    "micrometre",
)


def _net(where: str, call: str, action: Callable[[], T]) -> T:
    """``action()``, with any exception raised as an ``MTBError`` naming ``where`` and ``call``."""
    try:
        return action()
    except MTBError:
        raise
    except Exception as exc:
        raise MTBError(f"{where}: {call} failed: {type(exc).__name__}: {exc}") from exc


def resolve_mode(namespace: Any, name: str = "Synchronous") -> Any:
    """The ``MTBCmdSetModes`` member ``name``, never its int: pythonnet ≥ 3 refuses one."""
    try:
        return getattr(namespace.MTBCmdSetModes, name)
    except AttributeError as exc:
        raise MTBError(f"MTBCmdSetModes has no member {name!r}") from exc


def _cast(namespace: Any, obj: Any, names: Sequence[str]) -> Any | None:
    """``obj`` cast to the first interface of ``names`` that takes it, else ``None``."""
    for name in names:
        try:
            cast = getattr(namespace, name)(obj)
        except Exception:
            continue
        if cast is not None:
            return cast
    return None


def _units(where: str, continual: Any) -> list[str]:
    count = int(_net(where, "GetPositionUnitCount", continual.GetPositionUnitCount))
    return [
        str(
            _net(
                where,
                f"GetPositionUnit({i})",
                functools.partial(continual.GetPositionUnit, i),
            )
        )
        for i in range(count)
    ]


class DotNetAxis:
    """One MTB axis, cast to ``IMTBContinual``, in its µm unit."""

    _IFACES = ("IMTBContinual", "IMTBStageAxis", "IMTBAxis", "IMTBFocus")

    def __init__(self, mtb_id: str, component: Any, namespace: Any) -> None:
        self.mtb_id = mtb_id
        self._ns = namespace
        continual = _cast(namespace, component, self._IFACES)
        if continual is None:
            raise MTBError(
                f"{mtb_id}: the component exposes no position interface (tried {', '.join(self._IFACES)})"
            )
        self._c = continual
        units = _units(mtb_id, continual)
        unit = next((u for u in units if u.strip().lower() in _UM_ALIASES), None)
        if unit is None:
            raise MTBError(
                f"{mtb_id}: no micrometre unit among the units it offers {units!r}"
            )
        self.unit = unit

    def _deviation(self, call: str) -> float:
        """A declared deviation; 0.0 when the axis does not declare it (the predecessor's rule)."""
        try:
            return abs(float(getattr(self._c, call)(self.unit)))
        except Exception as exc:
            log.debug("%s: %s is not declared (%s); taken as 0", self.mtb_id, call, exc)
            return 0.0

    def info(self) -> AxisInfo:
        """Range, step and deviations, read in the axis's µm unit."""
        unit, c = self.unit, self._c
        return AxisInfo(
            min_um=float(
                _net(self.mtb_id, "GetMinPosition", lambda: c.GetMinPosition(unit))
            ),
            max_um=float(
                _net(self.mtb_id, "GetMaxPosition", lambda: c.GetMaxPosition(unit))
            ),
            step_um=abs(
                float(_net(self.mtb_id, "StepWidth", lambda: c.StepWidth(unit)))
            ),
            typical_deviation_um=self._deviation("TypicalDeviation"),
            max_deviation_um=self._deviation("MaxDeviation"),
        )

    def position_um(self) -> float:
        """``GetPosition`` in µm."""
        return float(
            _net(self.mtb_id, "GetPosition", lambda: self._c.GetPosition(self.unit))
        )

    def move_um(self, target_um: float, timeout_ms: int) -> bool:
        """A synchronous ``SetPosition`` with the enum member; MTB's answer."""
        mode = resolve_mode(self._ns)
        answer = _net(
            self.mtb_id,
            f"SetPosition({target_um} {self.unit})",
            lambda: self._c.SetPosition(
                float(target_um), self.unit, mode, int(timeout_ms)
            ),
        )
        return answer is not False


class DotNetChanger:
    """One MTB changer, cast to ``IMTBChanger``; positions 1-based, elements 0-based."""

    _IFACES = ("IMTBObjectiveChanger", "IMTBChanger")
    #: An objective's name needs this cast; an empty slot does not take it.
    _ELEMENT_IFACES = ("IMTBObjective",)

    def __init__(self, mtb_id: str, component: Any, namespace: Any) -> None:
        self.mtb_id = mtb_id
        self._ns = namespace
        changer = _cast(namespace, component, self._IFACES)
        if changer is None:
            raise MTBError(f"{mtb_id}: the component exposes no changer interface")
        self._c = changer

    def _element(self, position: int) -> ChangerElement:
        element = _net(
            self.mtb_id,
            f"GetElement({position - 1})",
            lambda: self._c.GetElement(position - 1),
        )
        try:
            element_type = str(element.ElementType)
        except Exception:
            element_type = ""
        name: Any = None
        for target in (element, _cast(self._ns, element, self._ELEMENT_IFACES)):
            if target is None:
                continue
            try:
                name = target.Name
            except Exception:
                continue
            if name:
                break
        return ChangerElement(
            position=position, name=str(name or ""), element_type=element_type
        )

    def elements(self) -> tuple[ChangerElement, ...]:
        """Every position from ``MinPosition`` to ``MaxPosition``, named by ``GetElement(p - 1)``."""
        low = int(_net(self.mtb_id, "MinPosition", lambda: self._c.MinPosition))
        high = int(_net(self.mtb_id, "MaxPosition", lambda: self._c.MaxPosition))
        return tuple(self._element(p) for p in range(low, high + 1))

    def position(self) -> int:
        """``Position``, 1-based."""
        return int(_net(self.mtb_id, "Position", lambda: self._c.Position))

    def move(self, position: int, timeout_ms: int) -> bool:
        """A synchronous ``SetPosition`` with the enum member; MTB's answer."""
        mode = resolve_mode(self._ns)
        answer = _net(
            self.mtb_id,
            f"SetPosition({position})",
            lambda: self._c.SetPosition(int(position), mode, int(timeout_ms)),
        )
        return answer is not False


def _args_for(
    led: Any, method: str, on: bool, timeout_ms: int, enum: Any, namespace: Any
) -> list[tuple[Any, ...]]:
    """Argument tuples for ``method``, built from its reflected overloads, richest first.

    Three guessed shapes (an ``MTBOnOff``, a bool, an int) all failed on the
    stand; every other MTB setter takes ``(value, mode, timeout)``. So each
    parameter's type is read: a two-state enum gets On or Off, an
    ``MTBCmdSetModes`` gets ``Synchronous``, a trailing integer the timeout. An
    overload that takes a client id (a ``String``) or a type not known here is
    skipped (the predecessor's ``_args_for``).
    """
    candidates: list[tuple[int, tuple[Any, ...]]] = []
    try:
        methods = [m for m in led.GetType().GetMethods() if m.Name == method]
    except Exception:
        methods = []
    for overload in methods:
        params = list(overload.GetParameters())
        args: list[Any] = []
        usable = True
        for index, param in enumerate(params):
            kind = param.ParameterType
            type_name = str(kind.Name or "")
            try:
                is_enum = bool(kind.IsEnum)
            except Exception:
                is_enum = False
            if is_enum:
                try:
                    names = [str(n) for n in enum.GetNames(kind)]
                except Exception:
                    usable = False
                    break
                lowered = [n.lower() for n in names]
                if "on" in lowered and "off" in lowered:
                    args.append(
                        enum.Parse(kind, names[lowered.index("on" if on else "off")])
                    )
                elif "cmdsetmodes" in type_name.lower():
                    args.append(resolve_mode(namespace))
                else:
                    usable = False
                    break
            elif type_name == "Boolean":
                args.append(bool(on))
            elif (
                type_name in ("Int32", "Int16", "Byte", "UInt32")
                and index == len(params) - 1
            ):
                args.append(int(timeout_ms))
            else:
                usable = False
                break
        if usable:
            candidates.append((len(args), tuple(args)))
    candidates.sort(key=lambda c: c[0], reverse=True)
    return [args for _, args in candidates]


def _signatures(led: Any, method: str) -> list[str]:
    """The reflected overloads of ``method``, for an error message that informs the next fix."""
    out: list[str] = []
    try:
        for m in led.GetType().GetMethods():
            if m.Name == method:
                params = ", ".join(
                    f"{p.ParameterType.Name} {p.Name}" for p in m.GetParameters()
                )
                out.append(f"{m.ReturnType.Name} {m.Name}({params})")
    except Exception:
        pass
    return out


class DotNetLED:
    """One Colibri LED: ``IMTBFluorescenceLED`` for on/off, ``IMTBContinual`` for the %."""

    def __init__(self, controller_id: str, led: Any, namespace: Any, enum: Any) -> None:
        self._ns = namespace
        self._enum = enum
        self._led = _cast(namespace, led, ("IMTBFluorescenceLED",)) or led
        self._cont = _cast(namespace, led, ("IMTBContinual",))
        self.wavelength_nm = int(
            _net(controller_id, "LED Wavelength", lambda: self._led.Wavelength)
        )
        self._where = f"{controller_id} LED {self.wavelength_nm} nm"
        self._unit = "%"
        if self._cont is not None:
            units = [u for u in _units(self._where, self._cont) if u]
            if units:
                self._unit = units[0]

    def is_on(self) -> bool:
        """``GetOnOff``, as a bool or an ``MTBOnOff`` member."""
        value = _net(self._where, "GetOnOff", lambda: self._led.GetOnOff())
        if isinstance(value, tuple):  # an out-parameter appended by pythonnet
            value = value[0]
        if isinstance(value, bool):
            return value
        text = str(value).lower()
        if text.endswith("off") or text == "0":
            return False
        if text.endswith("on") or text == "1":
            return True
        raise MTBError(
            f"{self._where}: GetOnOff answered {value!r}, neither on nor off"
        )

    def set_on(self, on: bool, timeout_ms: int) -> None:
        """``SetOnOff`` with arguments built from its reflected signature."""
        attempts = _args_for(
            self._led, "SetOnOff", on, timeout_ms, self._enum, self._ns
        )
        last: Exception | None = None
        for args in attempts:
            try:
                self._led.SetOnOff(*args)
                return
            except Exception as exc:
                last = exc
        signatures = _signatures(self._led, "SetOnOff") or ["<unavailable>"]
        raise MTBError(
            f"{self._where}: SetOnOff({'On' if on else 'Off'}) failed with every argument "
            f"form built from its reflected signature(s) {signatures}"
        ) from last

    def _continual(self) -> Any:
        if self._cont is None:
            raise MTBError(
                f"{self._where}: the LED has no intensity axis (IMTBContinual)"
            )
        return self._cont

    def intensity_pct(self) -> float:
        """The intensity axis's position, in %."""
        cont = self._continual()
        return float(
            _net(self._where, "GetPosition", lambda: cont.GetPosition(self._unit))
        )

    def set_intensity_pct(self, pct: float, timeout_ms: int) -> None:
        """A synchronous ``SetPosition`` on the intensity axis, in %."""
        cont = self._continual()
        mode = resolve_mode(self._ns)
        _net(
            self._where,
            f"SetPosition({pct} {self._unit})",
            lambda: cont.SetPosition(float(pct), self._unit, mode, int(timeout_ms)),
        )


class DotNetLEDController:
    """The Colibri, cast to ``IMTBFluorescenceLEDControl``."""

    def __init__(self, mtb_id: str, component: Any, namespace: Any, enum: Any) -> None:
        self.mtb_id = mtb_id
        self._ns = namespace
        self._enum = enum
        control = _cast(namespace, component, ("IMTBFluorescenceLEDControl",))
        if control is None:
            raise MTBError(
                f"{mtb_id}: the component exposes no IMTBFluorescenceLEDControl"
            )
        self._c = control
        self._leds: tuple[DotNetLED, ...] | None = None

    def leds(self) -> tuple[DotNetLED, ...]:
        """``LED[0]`` to ``LED[NumberOfLEDs - 1]``, built once."""
        if self._leds is None:
            count = int(_net(self.mtb_id, "NumberOfLEDs", lambda: self._c.NumberOfLEDs))
            self._leds = tuple(
                DotNetLED(self.mtb_id, self._led(i), self._ns, self._enum)
                for i in range(count)
            )
        return self._leds

    def _led(self, index: int) -> Any:
        return _net(self.mtb_id, f"LED[{index}]", lambda: self._c.LED[index])

    def set_manual_mode(self, timeout_ms: int) -> None:
        """``SetOperationMode(LEDLight, Continuous, …)``: the LEDs are driven directly."""
        ns = self._ns
        mode = resolve_mode(ns)
        try:
            light = ns.MTBFluorescenceLEDOperationMode.LEDLight
            continuous = ns.MTBFluorescenceLEDTriggerMode.Continuous
        except AttributeError as exc:
            raise MTBError(
                f"{self.mtb_id}: the LED operation modes are missing: {exc}"
            ) from exc
        last: Exception | None = None
        for args in (
            (light, continuous, mode, int(timeout_ms)),
            (light, continuous, mode),
        ):
            try:
                self._c.SetOperationMode(*args)
                return
            except Exception as exc:
                last = exc
        raise MTBError(
            f"{self.mtb_id}: SetOperationMode(LEDLight, Continuous) failed: {last}"
        ) from last


def _extract_bytes(result: Any) -> bytes | None:
    """The ``Byte[]`` of a .NET answer, or ``None``: pythonnet appends out-parameters to a tuple.

    The data overload of ``InitOnCurrentFocusPosition`` comes back as
    ``(True, Byte[])``; a bare ``bool`` is an answer, never a blob. A .NET
    ``Byte[]`` is a sequence of ints, so anything with a length that is not
    text is taken as its bytes (the predecessor's rule).
    """
    if result is None or isinstance(result, bool):
        return None
    candidates = result if isinstance(result, tuple) else (result,)
    for item in candidates:
        if isinstance(item, (bytes, bytearray)):
            return bytes(item)
        try:
            if len(item) and not isinstance(item, str):
                return bytes(bytearray(item))
        except Exception:
            continue
    return None


def _on_off(where: str, call: str, value: Any) -> bool:
    """An ``MTBOnOff`` answer as a bool; the first item of a tuple (an out-parameter after it).

    Read by the end of its text, never by ``"on" in``: ``"MTBOnOff.Off"``
    holds ``On`` in its type's name.
    """
    if isinstance(value, tuple):
        value = value[0] if value else None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text.endswith("off") or text == "0":
        return False
    if text.endswith("on") or text == "1":
        return True
    raise MTBError(f"{where}: {call} answered {value!r}, neither on nor off")


def _shape(args: tuple[Any, ...]) -> str:
    """An argument shape for a message: ``(None, 30000)``, the enum members by type."""
    shown = [
        repr(a) if a is None or isinstance(a, (int, float, str)) else type(a).__name__
        for a in args
    ]
    return f"({', '.join(shown)})"


class DotNetFocusStabilizer:
    """Definite Focus 2, cast to ``IMTBFocusStabilizer2`` only (m4 §11.3).

    Ported from the predecessor's ``MTBDefiniteFocus``, whose learn ran at the
    stand (2026-09-11). Its argument shapes are kept, in its order, because
    the overloads MTB 2.12 offers are known from reflection only; each one
    that raises is kept, and when none answers, the ``MTBError`` names the call
    and every shape tried, chained from the last failure (FM-113). Unlike the
    predecessor, nothing is swallowed: a learn with no blob, a stop or a mode
    switch that fails is an ``MTBError``. Its fallback to ``IMTBAutoFocus`` is
    not ported: a stand without DF2 fails its ``initialize()`` naming the cast.
    """

    _IFACE = "IMTBFocusStabilizer2"

    def __init__(self, mtb_id: str, component: Any, namespace: Any) -> None:
        self.mtb_id = mtb_id
        self._ns = namespace
        stabilizer = _cast(namespace, component, (self._IFACE,))
        if stabilizer is None:
            raise MTBError(
                f"{mtb_id}: the component does not cast to {self._IFACE}, so it is "
                "no Definite Focus 2"
            )
        self._c = stabilizer

    def _first_answer(
        self, call: str, method: Callable[..., Any], shapes: Sequence[tuple[Any, ...]]
    ) -> Any:
        """``method`` with each shape in turn; the answer of the first that does not raise."""
        tried: list[str] = []
        last: Exception | None = None
        for args in shapes:
            try:
                return method(*args)
            except Exception as exc:
                tried.append(f"{_shape(args)}: {type(exc).__name__}: {exc}")
                last = exc
        raise MTBError(
            f"{self.mtb_id}: {call} failed with every argument shape tried: "
            + "; ".join(tried)
        ) from last

    def _method(self, name: str) -> Callable[..., Any]:
        method: Callable[..., Any] = _net(
            self.mtb_id, name, lambda: getattr(self._c, name)
        )
        return method

    def learn(self, timeout_ms: int) -> bytes:
        """``InitOnCurrentFocusPosition``: the blob from its ``Byte[]&`` out-parameter.

        The data overload first, as the predecessor did, since only it hands
        the reference back; then the mode overload and ``GetStabilizerData``.
        """
        mode = resolve_mode(self._ns)
        init = self._method("InitOnCurrentFocusPosition")
        tried: list[str] = []
        last: Exception | None = None
        for args in ((None, int(timeout_ms)), (None,)):
            try:
                result = init(*args)
            except Exception as exc:
                tried.append(f"{_shape(args)}: {type(exc).__name__}: {exc}")
                last = exc
                continue
            data = _extract_bytes(result)
            if data:
                return data
            tried.append(f"{_shape(args)}: answered {result!r} with no reference")
        for args in ((mode, int(timeout_ms)), (mode,)):
            try:
                answer = init(*args)
            except Exception as exc:
                tried.append(f"{_shape(args)}: {type(exc).__name__}: {exc}")
                last = exc
                continue
            if answer is False:
                tried.append(f"{_shape(args)}: answered False")
                continue
            try:
                data = _extract_bytes(self._c.GetStabilizerData())
            except Exception as exc:
                tried.append(f"GetStabilizerData(): {type(exc).__name__}: {exc}")
                last = exc
                continue
            if data:
                return data
            tried.append(f"{_shape(args)}, then GetStabilizerData(): no reference")
        raise MTBError(
            f"{self.mtb_id}: InitOnCurrentFocusPosition (learn) gave no reference "
            f"with any argument shape tried: {'; '.join(tried)}"
        ) from last

    def restore(self, reference: bytes, timeout_ms: int) -> bool:
        """``InitWithStabilizingData(mode, data, timeout)``; MTB's answer."""
        if not reference:
            raise MTBError(f"{self.mtb_id}: no stabilizing data to restore")
        mode = resolve_mode(self._ns)
        payload = bytearray(reference)
        answer = self._first_answer(
            "InitWithStabilizingData",
            self._method("InitWithStabilizingData"),
            ((mode, payload, int(timeout_ms)), (mode, payload)),
        )
        return answer is not False

    def stabilize_now(self, timeout_ms: int) -> bool:
        """``StabilizeNow(mode, timeout)``; MTB's answer."""
        mode = resolve_mode(self._ns)
        answer = self._first_answer(
            "StabilizeNow",
            self._method("StabilizeNow"),
            ((mode, int(timeout_ms)), (mode,)),
        )
        return answer is not False

    def stop(self, timeout_ms: int) -> None:
        """``Stop(mode, timeout)``; a ``False`` is logged, since whether DF2 stopped is unknown."""
        mode = resolve_mode(self._ns)
        answer = self._first_answer(
            "Stop", self._method("Stop"), ((mode, int(timeout_ms)), (mode,))
        )
        if answer is False:
            log.warning(
                "%s: Stop answered False; whether DF2 stopped is not known", self.mtb_id
            )

    def operation_mode(self) -> str:
        """``GetOperationMode()``: the member's name, one of the three."""
        value = _net(self.mtb_id, "GetOperationMode", self._c.GetOperationMode)
        if isinstance(value, tuple):
            value = value[0] if value else None
        name = str(value).rsplit(".", 1)[-1].strip()
        if name not in STABILIZER_OPERATION_MODES:
            raise MTBError(
                f"{self.mtb_id}: GetOperationMode answered {value!r}, not one of "
                f"{', '.join(STABILIZER_OPERATION_MODES)}"
            )
        return name

    def set_operation_mode(self, mode: str) -> None:
        """``SetOperationMode`` with the member taken by name (``getattr``), never an int."""
        if mode not in STABILIZER_OPERATION_MODES:
            raise MTBError(
                f"{self.mtb_id}: no operation mode {mode!r}; one of "
                f"{', '.join(STABILIZER_OPERATION_MODES)}"
            )
        try:
            member = getattr(self._ns.MTBFocusStabilizer2OperationMode, mode)
        except AttributeError as exc:
            raise MTBError(
                f"{self.mtb_id}: MTBFocusStabilizer2OperationMode has no member {mode!r}"
            ) from exc
        answer = self._first_answer(
            f"SetOperationMode({mode})",
            self._method("SetOperationMode"),
            ((member,), (None, member)),
        )
        if answer is False:
            raise MTBError(f"{self.mtb_id}: SetOperationMode({mode}) answered False")

    def piezo_focus_usage(self) -> bool:
        """``GetPiezoFocusUsage()``: ``True`` when DF2 corrects through the piezo."""
        value = _net(self.mtb_id, "GetPiezoFocusUsage", self._c.GetPiezoFocusUsage)
        return _on_off(self.mtb_id, "GetPiezoFocusUsage", value)

    def periodic(self) -> bool:
        """``GetStabilizePeriodicallyOnOff()``: its first item; the reason it stopped follows."""
        value = _net(
            self.mtb_id,
            "GetStabilizePeriodicallyOnOff",
            self._c.GetStabilizePeriodicallyOnOff,
        )
        return _on_off(self.mtb_id, "GetStabilizePeriodicallyOnOff", value)

    def stop_periodic(self, timeout_ms: int) -> None:
        """``StabilizePeriodically(MTBOnOff.Off)``, the member, then with a period of 0."""
        try:
            off = self._ns.MTBOnOff.Off
        except AttributeError as exc:
            raise MTBError(f"{self.mtb_id}: MTBOnOff has no member 'Off'") from exc
        answer = self._first_answer(
            "StabilizePeriodically(Off)",
            self._method("StabilizePeriodically"),
            ((off,), (off, 0)),
        )
        if answer is False:
            raise MTBError(
                f"{self.mtb_id}: StabilizePeriodically(Off) answered False; periodic "
                "stabilisation may still be on"
            )


class DotNetMTB:
    """``api.MTBApi`` over ``ZEISS.MTB.Api``; built by :func:`connect`, not logged in."""

    def __init__(self, namespace: Any, enum: Any, *, locale: str = "en") -> None:
        self._ns = namespace
        self._enum = enum
        self._locale = locale
        self._connection: Any = None
        self._client_id: str | None = None
        self._root: Any = None

    def login(self) -> None:
        """``Login``, then ``GetRoot``, which initialises all the hardware."""
        connection = _net("MTB", "MTBConnection()", self._ns.MTBConnection)
        answer = _net("MTB", "Login", lambda: connection.Login(self._locale, ""))
        if isinstance(answer, tuple):
            client_id = next(
                (v for v in reversed(answer) if isinstance(v, str) and v), None
            )
        elif isinstance(answer, str):
            client_id = answer
        else:
            client_id = None
        if not client_id:
            raise MTBError(f"MTB: Login returned no client id ({answer!r})")
        root = _net("MTB", "GetRoot", lambda: connection.GetRoot(client_id))
        if root is None:
            raise MTBError("MTB: GetRoot returned nothing; is the stand switched on?")
        self._connection, self._client_id, self._root = connection, client_id, root
        log.info("MTB connected, client id %s", client_id)

    def logout(self) -> None:
        """``Logout``; MTB cannot log in again in this process."""
        connection, client_id = self._connection, self._client_id
        self._connection = self._client_id = self._root = None
        if connection is not None and client_id:
            _net("MTB", "Logout", lambda: connection.Logout(client_id))

    def _component(self, mtb_id: str) -> Any:
        root = self._root
        if root is None:
            raise MTBError(f"MTB component {mtb_id!r}: not logged in")
        component = _net(mtb_id, "GetComponent", lambda: root.GetComponent(mtb_id))
        if component is None:
            raise MTBError(f"MTB component {mtb_id!r} is absent from this stand")
        return component

    def axis(self, mtb_id: str) -> DotNetAxis:
        """The component ``mtb_id`` as an axis."""
        return DotNetAxis(mtb_id, self._component(mtb_id), self._ns)

    def changer(self, mtb_id: str) -> DotNetChanger:
        """The component ``mtb_id`` as a changer."""
        return DotNetChanger(mtb_id, self._component(mtb_id), self._ns)

    def led_controller(self, mtb_id: str) -> DotNetLEDController:
        """The component ``mtb_id`` as an LED controller."""
        return DotNetLEDController(
            mtb_id, self._component(mtb_id), self._ns, self._enum
        )

    def focus_stabilizer(self, mtb_id: str) -> DotNetFocusStabilizer:
        """The component ``mtb_id`` as Definite Focus 2."""
        return DotNetFocusStabilizer(mtb_id, self._component(mtb_id), self._ns)


def connect(dll: Path = DEFAULT_MTB_API) -> MTBApi:
    """Load ``MTBApi.dll`` through pythonnet and return the API, not yet logged in.

    Raises:
        MTBSessionError: pythonnet is missing (``pip install smc[zeiss]``), the
            DLL is not at ``dll``, or it does not load; the message says which.
    """
    try:
        import clr
    except ImportError as exc:
        raise MTBSessionError(
            "pythonnet is not installed, so MTB cannot be reached: run `pip install smc[zeiss]` on the stand's PC"
        ) from exc
    if not dll.is_file():
        raise MTBSessionError(
            f"MTBApi.dll was not found at {dll}; is MTB 2011 installed?"
        )
    try:
        clr.AddReference(str(dll.with_suffix("")))
    except Exception as first:
        try:
            from System.Reflection import Assembly

            Assembly.LoadFrom(str(dll))
        except Exception as exc:
            raise MTBSessionError(
                f"MTBApi.dll at {dll} did not load: {first}; then {exc}"
            ) from exc
    try:
        namespace = importlib.import_module("ZEISS.MTB.Api")
        from System import Enum
    except Exception as exc:
        raise MTBSessionError(
            f"MTBApi.dll at {dll} loaded, but ZEISS.MTB.Api is not importable: {exc}"
        ) from exc
    return DotNetMTB(namespace, Enum)
