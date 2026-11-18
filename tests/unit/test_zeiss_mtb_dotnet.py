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
from smc.hardware.backends.zeiss_mtb.api import (
    STABILIZER_TIMEOUT_MS,
    MTBError,
    MTBSessionError,
)


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


def test_mtb_false_and_off_cross_the_seam_unchanged() -> None:
    class Refuses(FakeContinual):
        def SetPosition(self, pos: float, unit: str, mode: Any, timeout: int) -> bool:
            return False  # a zero-distance move, or a refusal: the device judges it

    class Stuck(FakeLightChanger):
        def SetPosition(self, position: int, mode: Any, timeout: int) -> bool:
            return False

    class Dark(ReflectiveLED):
        def GetOnOff(self) -> str:
            return "MTBOnOff.Off"

    ns = _namespace()
    assert dotnet.DotNetAxis("MTBFocus", Refuses(), ns).move_um(1.0, 1000) is False
    assert (
        dotnet.DotNetChanger("MTBTLShutter", Stuck(1, ["Open", "Closed"]), ns).move(
            2, 1000
        )
        is False
    )
    assert dotnet.DotNetLED("MTBFLLEDController", Dark(), ns, FakeEnum).is_on() is False


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


# -- Definite Focus 2 (m4 §11.3), from the predecessor's FakeStabilizer --------


class _OperationModes:
    """``MTBFocusStabilizer2OperationMode``: members are objects, never ints."""

    Exact = object()
    Balanced = object()
    Fast = object()


class _OnOff:
    """``MTBOnOff``: a member's text is the type's name and the member's."""

    On = "MTBOnOff.On"
    Off = "MTBOnOff.Off"


def _df2_namespace(**extra: Any) -> SimpleNamespace:
    return _namespace(
        **{
            "IMTBFocusStabilizer2": _cast,
            "MTBFocusStabilizer2OperationMode": _OperationModes,
            "MTBOnOff": _OnOff,
            **extra,
        }
    )


class FakeStabilizer:
    """``IMTBFocusStabilizer2`` with the data overload of the learn.

    pythonnet surfaces a .NET ``out`` parameter by appending it to the return
    value, so the data overload comes back as a tuple: modelling that shape is
    the point of this double (the predecessor's ``FakeStabilizer``).
    """

    def __init__(
        self, data: bytes = b"\x01\x02\x03\xf4", *, with_data_overload: bool = True
    ) -> None:
        self._data = data
        self._with_data = with_data_overload
        self.init_calls: list[tuple[Any, ...]] = []
        self.restore_calls: list[tuple[Any, ...]] = []
        self.stabilize_calls: list[tuple[Any, ...]] = []
        self.mode_calls: list[tuple[Any, ...]] = []
        self.periodic_calls: list[tuple[Any, ...]] = []

    def InitOnCurrentFocusPosition(self, *args: Any) -> Any:
        self.init_calls.append(args)
        if self._with_data and args and args[0] is None:
            return (True, bytearray(self._data))
        if args and args[0] is None:
            raise TypeError("no such overload")
        return True

    def InitWithStabilizingData(self, *args: Any) -> bool:
        self.restore_calls.append(args)
        return True

    def GetStabilizerData(self) -> bytearray:
        return bytearray(self._data)

    def StabilizeNow(self, *args: Any) -> bool:
        self.stabilize_calls.append(args)
        return True

    def Stop(self, *args: Any) -> bool:
        return True

    def GetOperationMode(self) -> str:
        return "Balanced"

    def SetOperationMode(self, *args: Any) -> bool:
        self.mode_calls.append(args)
        return True

    def GetPiezoFocusUsage(self) -> str:
        return "MTBOnOff.Off"

    def GetStabilizePeriodicallyOnOff(self) -> tuple[str, str]:
        return ("MTBOnOff.On", "stopped by: nothing")

    def StabilizePeriodically(self, *args: Any) -> bool:
        self.periodic_calls.append(args)
        return True


def _df2(raw: FakeStabilizer | None = None) -> dotnet.DotNetFocusStabilizer:
    return dotnet.DotNetFocusStabilizer(
        "MTBFocusStabilizer2", raw or FakeStabilizer(), _df2_namespace()
    )


def test_focus_stabilizer_without_the_cast_is_an_mtb_error() -> None:
    # No fallback to IMTBAutoFocus (m4 §11.3): a stand without DF2 fails its
    # initialize() naming the cast.
    def refuses(obj: Any) -> Any:
        raise TypeError("cannot cast")

    ns = _df2_namespace(IMTBFocusStabilizer2=refuses, IMTBAutoFocus=_cast)
    with pytest.raises(MTBError, match="does not cast to IMTBFocusStabilizer2"):
        dotnet.DotNetFocusStabilizer("MTBFocusStabilizer2", FakeStabilizer(), ns)


def test_focus_stabilizer_learn_takes_the_blob_from_the_out_parameter() -> None:
    # The data overload first: the mode overload leaves the reference inside
    # DF2, where only one position can use it.
    raw = FakeStabilizer()
    blob = _df2(raw).learn(30_000)
    assert blob == b"\x01\x02\x03\xf4"
    assert isinstance(blob, bytes)
    assert raw.init_calls == [(None, 30_000)]


def test_focus_stabilizer_learn_falls_back_to_the_mode_overload_and_its_data() -> None:
    raw = FakeStabilizer(with_data_overload=False)
    assert _df2(raw).learn(30_000) == b"\x01\x02\x03\xf4"
    assert raw.init_calls == [(None, 30_000), (None,), (_Modes.Synchronous, 30_000)]


def test_focus_stabilizer_every_shape_failing_names_each_and_chains_the_last() -> None:
    class Dead(FakeStabilizer):
        def InitOnCurrentFocusPosition(self, *args: Any) -> Any:
            self.init_calls.append(args)
            raise RuntimeError(f"no focus signal {len(self.init_calls)}")

    with pytest.raises(MTBError) as caught:
        _df2(Dead()).learn(30_000)
    text = str(caught.value)
    assert text.startswith(
        "MTBFocusStabilizer2: InitOnCurrentFocusPosition (learn) gave no reference"
    )
    # A mode member shows as its type: `object` here, MTBCmdSetModes on .NET.
    for shape, n in (
        ("(None, 30000)", 1),
        ("(None)", 2),
        ("(object, 30000)", 3),
        ("(object)", 4),
    ):
        assert f"{shape}: RuntimeError: no focus signal {n}" in text
    assert str(caught.value.__cause__) == "no focus signal 4"


def test_focus_stabilizer_a_learn_with_an_empty_blob_is_an_mtb_error() -> None:
    raw = FakeStabilizer(b"")
    with pytest.raises(MTBError, match="gave no reference"):
        _df2(raw).learn(30_000)


def test_extract_bytes_handles_the_shapes_pythonnet_produces() -> None:
    ex = dotnet._extract_bytes
    assert ex(None) is None
    assert ex(True) is None, "a bare bool is an answer, not a blob"
    assert ex(b"ab") == b"ab"
    assert ex(bytearray(b"ab")) == b"ab"
    assert ex((True, bytearray(b"ab"))) == b"ab"
    assert ex([1, 2, 255]) == b"\x01\x02\xff"


def test_focus_stabilizer_restore_sends_the_blob_and_the_mode_member() -> None:
    raw = FakeStabilizer()
    assert _df2(raw).restore(b"\x09\x08", 30_000) is True
    mode, data, timeout = raw.restore_calls[-1]
    assert mode is _Modes.Synchronous, (
        "a raw int reached .NET: pythonnet ≥ 3 refuses it"
    )
    assert (bytes(data), timeout) == (b"\x09\x08", 30_000)


def test_focus_stabilizer_restore_refuses_an_empty_reference() -> None:
    raw = FakeStabilizer()
    with pytest.raises(MTBError, match="no stabilizing data"):
        _df2(raw).restore(b"", 30_000)
    assert raw.restore_calls == []


def test_the_stabilizer_timeout_is_long_enough_for_df2() -> None:
    # The Micro-Manager wiki recommends 25 s for DF2: a 10 s motion timeout
    # would abort a good measurement.
    assert STABILIZER_TIMEOUT_MS >= 25_000
    raw = FakeStabilizer()
    _df2(raw).stabilize_now(STABILIZER_TIMEOUT_MS)
    assert raw.stabilize_calls == [(_Modes.Synchronous, STABILIZER_TIMEOUT_MS)]


def test_focus_stabilizer_answers_cross_the_seam_and_a_raise_names_stabilize_now() -> (
    None
):
    class Refuses(FakeStabilizer):
        def StabilizeNow(self, *args: Any) -> bool:
            return False

        def InitWithStabilizingData(self, *args: Any) -> bool:
            return False

    class Dead(FakeStabilizer):
        def StabilizeNow(self, *args: Any) -> bool:
            raise RuntimeError("lost the surface")

    assert _df2(Refuses()).stabilize_now(30_000) is False
    assert _df2(Refuses()).restore(b"\x01", 30_000) is False
    with pytest.raises(MTBError, match=r"StabilizeNow failed .*lost the surface"):
        _df2(Dead()).stabilize_now(30_000)


def test_operation_mode_is_set_by_member_name_never_an_int() -> None:
    raw = FakeStabilizer()
    df2 = _df2(raw)
    df2.set_operation_mode("Exact")
    assert raw.mode_calls == [(_OperationModes.Exact,)]
    assert not isinstance(raw.mode_calls[0][0], int)
    assert df2.operation_mode() == "Balanced"
    with pytest.raises(MTBError, match="'Sharpish'; one of Exact, Balanced, Fast"):
        df2.set_operation_mode("Sharpish")
    assert len(raw.mode_calls) == 1


def test_on_off_reads_take_the_end_of_the_text_and_the_first_item() -> None:
    # "MTBOnOff.Off" holds "On" in its type's name: the predecessor's `"on" in`
    # read it as on.
    raw = FakeStabilizer()
    df2 = _df2(raw)
    assert df2.piezo_focus_usage() is False
    assert df2.periodic() is True
    df2.stop_periodic(30_000)
    assert raw.periodic_calls == [(_OnOff.Off,)]


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
