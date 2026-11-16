"""The .NET seam on duck-typed fakes, on any OS (m4 §5).

Ported from ``lightsheet-live-tracking-tool``'s ``tests/test_mtb.py``: the
fakes stand in for the .NET objects, the ``ZEISS.MTB.Api`` namespace and
``System.Enum``, and the interface casts return the object they are given.
The behaviours pinned here are the ones the predecessor verified at the stand.
"""

# ruff: noqa: N802 - the fakes mirror the names of the .NET members

from __future__ import annotations

import builtins
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from smc.hardware.backends.zeiss_mtb import dotnet
from smc.hardware.backends.zeiss_mtb.api import MTBError, MTBSessionError


class _Modes:
    """``MTBCmdSetModes``: members are objects, never ints (pythonnet ≥ 3)."""

    Default = object()
    Synchronous = object()


def _cast(obj: Any) -> Any:
    return obj


def _namespace(**extra: Any) -> SimpleNamespace:
    """``ZEISS.MTB.Api`` with every cast the seam tries returning its argument."""
    ns = SimpleNamespace(
        MTBCmdSetModes=_Modes,
        IMTBContinual=_cast,
        IMTBStageAxis=_cast,
        IMTBAxis=_cast,
        IMTBFocus=_cast,
        IMTBChanger=_cast,
        IMTBObjectiveChanger=_cast,
        IMTBFluorescenceLED=_cast,
        IMTBFluorescenceLEDControl=_cast,
        MTBFluorescenceLEDOperationMode=SimpleNamespace(LEDLight="LEDLight"),
        MTBFluorescenceLEDTriggerMode=SimpleNamespace(Continuous="Continuous"),
    )
    for key, value in extra.items():
        setattr(ns, key, value)
    return ns


class FakeContinual:
    """A component cast to ``IMTBContinual``."""

    def __init__(self, position: float = 0.0, units: tuple[str, ...] = ("µm",)) -> None:
        self._pos = position
        self._units = list(units)
        self.set_calls: list[tuple[Any, ...]] = []

    def GetPositionUnitCount(self) -> int:
        return len(self._units)

    def GetPositionUnit(self, i: int) -> str:
        return self._units[i]

    def GetPosition(self, unit: str) -> float:
        return self._pos

    def GetMinPosition(self, unit: str) -> float:
        return -14000.0

    def GetMaxPosition(self, unit: str) -> float:
        return 14000.0

    def StepWidth(self, unit: str) -> float:
        return 0.01

    def TypicalDeviation(self, unit: str) -> float:
        return -0.02  # MTB's sign is not trusted

    def MaxDeviation(self, unit: str) -> float:
        raise RuntimeError("not declared on this axis")

    def SetPosition(self, pos: float, unit: str, mode: Any, timeout: int) -> bool:
        self.set_calls.append((pos, unit, mode, timeout))
        self._pos = pos
        return True


def test_resolve_mode_takes_the_enum_member_not_an_int() -> None:
    ns = _namespace()
    assert dotnet.resolve_mode(ns) is _Modes.Synchronous
    axis = dotnet.DotNetAxis("MTBFocus", FakeContinual(), ns)
    assert axis.move_um(12.5, 60_000) is True
    pos, unit, mode, timeout = axis._c.set_calls[-1]
    assert mode is _Modes.Synchronous
    assert not isinstance(mode, int)
    assert (pos, unit, timeout) == (12.5, "µm", 60_000)
    with pytest.raises(MTBError, match="no member 'Teleport'"):
        dotnet.resolve_mode(ns, "Teleport")


def test_the_axis_unit_is_found_among_the_units_offered() -> None:
    axis = dotnet.DotNetAxis(
        "MTBFocus", FakeContinual(units=("nm", "µm")), _namespace()
    )
    assert axis.unit == "µm"
    info = axis.info()
    assert (info.min_um, info.max_um, info.step_um) == (-14000.0, 14000.0, 0.01)
    assert info.typical_deviation_um == 0.02
    assert (
        info.max_deviation_um == 0.0
    )  # not declared: taken as 0, as the predecessor did


def test_a_component_without_a_um_unit_is_an_mtb_error() -> None:
    with pytest.raises(
        MTBError, match=r"MTBFocus: no micrometre unit .*\['nm', 'mm'\]"
    ):
        dotnet.DotNetAxis("MTBFocus", FakeContinual(units=("nm", "mm")), _namespace())


class FakeLightChanger:
    """A light-path changer: ``Position`` 1-based, ``GetElement`` 0-based."""

    def __init__(self, position: int, names: list[str]) -> None:
        self.Position = position
        self.MinPosition = 1
        self.MaxPosition = len(names)
        self._names = names
        self.set_calls: list[tuple[Any, ...]] = []

    def GetElement(self, n: int) -> Any:
        if n < 0 or n >= len(self._names):
            raise IndexError("Index was outside the bounds of the array")
        return SimpleNamespace(Name=self._names[n], ElementType="Element")

    def SetPosition(self, position: int, mode: Any, timeout: int) -> bool:
        self.set_calls.append((position, mode, timeout))
        self.Position = int(position)
        return True


def test_changer_names_come_from_get_element_position_minus_one() -> None:
    # The sideport as the stand's display shows it (2026-09-08).
    raw = FakeLightChanger(2, ["VIS 100%", "Sideport 100% Left", "Sideport 100% Right"])
    changer = dotnet.DotNetChanger("MTBSideportChanger", raw, _namespace())
    elements = changer.elements()
    assert [(e.position, e.name) for e in elements] == [
        (1, "VIS 100%"),
        (2, "Sideport 100% Left"),
        (3, "Sideport 100% Right"),
    ]
    assert changer.position() == 2
    assert changer.move(3, 30_000) is True
    assert raw.set_calls[-1] == (3, _Modes.Synchronous, 30_000)


def test_an_empty_objective_slot_has_element_type_none() -> None:
    class Nosepiece(FakeLightChanger):
        def GetElement(self, n: int) -> Any:
            if n == 1:
                return SimpleNamespace(ElementType="None")  # no Name, not castable
            return SimpleNamespace(ElementType="Objective")

    class Objective:
        def __init__(self, element: Any) -> None:
            if element.ElementType == "None":
                raise TypeError("cannot cast an empty slot")
            self.Name = "Fluar 10x/0.50"

    raw = Nosepiece(1, ["", ""])
    changer = dotnet.DotNetChanger(
        "MTBObjectiveChanger", raw, _namespace(IMTBObjective=Objective)
    )
    first, second = changer.elements()
    assert (first.name, first.element_type) == ("Fluar 10x/0.50", "Objective")
    assert (second.name, second.element_type) == ("", "None")


class FakeParam:
    def __init__(
        self, type_name: str, name: str, enum_names: tuple[str, ...] = ()
    ) -> None:
        self.Name = name
        self.ParameterType = SimpleNamespace(
            Name=type_name, IsEnum=bool(enum_names), names=list(enum_names)
        )


class FakeMethod:
    def __init__(self, name: str, params: list[FakeParam]) -> None:
        self.Name = name
        self._params = params
        self.ReturnType = SimpleNamespace(Name="Void")

    def GetParameters(self) -> list[FakeParam]:
        return self._params


class FakeEnum:
    """``System.Enum``: names of an enum type, and a member parsed from a name."""

    @staticmethod
    def GetNames(kind: Any) -> list[str]:
        return list(kind.names)

    @staticmethod
    def Parse(kind: Any, name: str) -> str:
        return f"<{kind.Name}.{name}>"


class ReflectiveLED:
    """An LED whose ``SetOnOff`` accepts only ``(MTBOnOff, MTBCmdSetModes, Int32)``."""

    def __init__(self, wavelength: int = 475) -> None:
        self.Wavelength = wavelength
        self.calls: list[tuple[Any, ...]] = []
        self.rejected: list[tuple[Any, ...]] = []

    def GetType(self) -> Any:
        methods = [
            FakeMethod("SetOnOff", [FakeParam("MTBOnOff", "onOff", ("Off", "On"))]),
            FakeMethod(
                "SetOnOff",
                [
                    FakeParam("MTBOnOff", "onOff", ("Off", "On")),
                    FakeParam("MTBCmdSetModes", "mode", ("Default", "Synchronous")),
                    FakeParam("Int32", "timeout"),
                ],
            ),
            FakeMethod(
                "SetOnOff",
                [FakeParam("String", "clientID"), FakeParam("Boolean", "on")],
            ),
            FakeMethod("GetOnOff", []),
        ]
        return SimpleNamespace(GetMethods=lambda: methods)

    def SetOnOff(self, *args: Any) -> None:
        if len(args) != 3:
            self.rejected.append(args)
            raise RuntimeError("SetOnOff failed")
        self.calls.append(args)

    def GetOnOff(self) -> str:
        return "MTBOnOff.On"

    def GetPositionUnitCount(self) -> int:
        return 1

    def GetPositionUnit(self, i: int) -> str:
        return "%"


def test_set_on_off_arguments_follow_the_reflected_signature() -> None:
    raw = ReflectiveLED()
    led = dotnet.DotNetLED("MTBFLLEDController", raw, _namespace(), FakeEnum)
    led.set_on(True, 4321)
    assert raw.calls == [("<MTBOnOff.On>", _Modes.Synchronous, 4321)]
    assert raw.rejected == []  # the richest overload is tried first
    led.set_on(False, 1000)
    assert raw.calls[-1][0] == "<MTBOnOff.Off>"
    assert led.is_on() is True
    assert led.wavelength_nm == 475


def test_a_set_on_off_that_every_form_refuses_names_the_signatures() -> None:
    class Refuses(ReflectiveLED):
        def SetOnOff(self, *args: Any) -> None:
            raise RuntimeError("nope")

    led = dotnet.DotNetLED("MTBFLLEDController", Refuses(), _namespace(), FakeEnum)
    with pytest.raises(MTBError, match=r"reflected signature.*Int32 timeout") as caught:
        led.set_on(False, 1000)
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_a_missing_component_is_an_mtb_error_naming_it() -> None:
    root = SimpleNamespace(GetComponent=lambda mtb_id: None)
    mtb = dotnet.DotNetMTB(_namespace(), FakeEnum)
    mtb._root = root
    with pytest.raises(MTBError, match="'MTBFocusStabilizer2' is absent"):
        mtb.axis("MTBFocusStabilizer2")


def test_login_takes_the_client_id_from_a_tuple_and_logs_out_with_it() -> None:
    logouts: list[str] = []

    class Connection:
        def Login(self, locale: str, client: str) -> tuple[None, str]:
            return (None, "client-7")

        def GetRoot(self, client_id: str) -> object:
            assert client_id == "client-7"
            return SimpleNamespace(GetComponent=lambda mtb_id: FakeContinual())

        def Logout(self, client_id: str) -> None:
            logouts.append(client_id)

    mtb = dotnet.DotNetMTB(_namespace(MTBConnection=Connection), FakeEnum)
    mtb.login()
    assert mtb.axis("MTBFocus").position_um() == 0.0
    mtb.logout()
    assert logouts == ["client-7"]


def test_a_dotnet_exception_becomes_an_mtb_error_naming_the_call() -> None:
    class FakeDotNetError(Exception):
        """pythonnet's .NET exceptions are no RuntimeError (FM-113)."""

    class Jammed(FakeContinual):
        def SetPosition(self, pos: float, unit: str, mode: Any, timeout: int) -> bool:
            raise FakeDotNetError("MTBException: axis locked")

    axis = dotnet.DotNetAxis("MTBStageAxisX", Jammed(), _namespace())
    with pytest.raises(
        MTBError, match=r"MTBStageAxisX: SetPosition\(5.0 µm\) failed"
    ) as caught:
        axis.move_um(5.0, 1000)
    assert "axis locked" in str(caught.value)
    assert isinstance(caught.value.__cause__, FakeDotNetError)


def test_manual_mode_falls_back_to_the_overload_without_a_timeout() -> None:
    calls: list[tuple[Any, ...]] = []

    class Control:
        NumberOfLEDs = 1
        LED: ClassVar[list[ReflectiveLED]] = [ReflectiveLED(385)]

        def SetOperationMode(self, *args: Any) -> None:
            calls.append(args)
            if len(args) == 4:
                raise RuntimeError("no such overload")

    controller = dotnet.DotNetLEDController(
        "MTBFLLEDController", Control(), _namespace(), FakeEnum
    )
    controller.set_manual_mode(10_000)
    assert calls[-1] == ("LEDLight", "Continuous", _Modes.Synchronous)
    assert [led.wavelength_nm for led in controller.leds()] == [385]


def test_connect_without_pythonnet_is_a_session_error_naming_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def no_clr(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "clr":
            raise ImportError("No module named 'clr'")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "clr", raising=False)
    monkeypatch.setattr(builtins, "__import__", no_clr)
    with pytest.raises(MTBSessionError, match=r"pip install smc\[zeiss\]"):
        dotnet.connect(Path("C:/nowhere/MTBApi.dll"))


def test_connect_with_a_missing_dll_names_the_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "clr", SimpleNamespace())
    missing = tmp_path / "MTBApi.dll"
    with pytest.raises(MTBSessionError, match="was not found at") as caught:
        dotnet.connect(missing)
    assert str(missing) in str(caught.value)
