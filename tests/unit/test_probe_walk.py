"""The probe's walk over a stub core (design m2 §9.3).

``_ProbeCore`` answers the calls the walk makes and records every one in
``calls``, so each rule is shown by what was and was not sent to the core
(FM-94: the assertions read ``calls`` and the rows after the walk, never from
inside a callback). No test here lists a vendor adapter (FM-40).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest
from pymmcore_plus import DeviceType

from smc.discovery import diagnosis
from smc.discovery.mm_inventory import list_adapters, probe_adapter, walk_adapter
from smc.discovery.models import ProbedDevice, ProbeResult
from smc.hardware.backends import nikon

_UNSET = ("", "undefined")


def _type_value(type_name: str) -> int:
    # A type is the name without "Device": "Hub" is DeviceType.HubDevice.
    return int(DeviceType[f"{type_name}Device"])


class _ProbeCore:
    """One adapter, answering what ``walk_adapter`` and ``list_adapters`` ask.

    ``listed`` is the adapter's own ``(name, type)`` list. ``reported`` maps a
    hub to the names it reports once initialised, ``ports`` a label to its
    pre-init ``Port`` (a label absent from it has no such property), and
    ``types`` a label to the type of a device the adapter does not list.
    ``init_errors``, ``load_errors`` and ``set_errors`` map a label to the text
    of the ``RuntimeError`` its call raises; ``list_errors`` does the same for a
    hub's ``getInstalledDevices``; ``unreadable`` names the labels whose
    ``Port`` cannot be read, and ``runtime_ports`` the labels whose ``Port`` is
    not a pre-init property. A device with a pre-init ``Port`` that is unset
    raises on ``initializeDevice``, as a real serial adapter does, so code that
    initialises it anyway shows ``failed``, never ``ok``. A peripheral of a hub
    (one a hub reports, or any listed device of an adapter that lists a hub)
    raises on ``initializeDevice`` until ``setParentLabel`` named its hub, as the
    Ti2's peripherals do, so a walk that sets the parent late shows ``failed``.
    """

    def __init__(
        self,
        adapter: str,
        listed: Sequence[tuple[str, str]],
        *,
        reported: Mapping[str, Sequence[str]] | None = None,
        ports: Mapping[str, str] | None = None,
        init_errors: Mapping[str, str] | None = None,
        load_errors: Mapping[str, str] | None = None,
        set_errors: Mapping[str, str] | None = None,
        list_errors: Mapping[str, str] | None = None,
        types: Mapping[str, str] | None = None,
        unreadable: Sequence[str] = (),
        runtime_ports: Sequence[str] = (),
        interrupt: Sequence[str] = (),
        installed: bool = True,
        listing_error: str | None = None,
    ) -> None:
        self.adapter = adapter
        self.listed = list(listed)
        self.reported = {k: list(v) for k, v in (reported or {}).items()}
        self.ports = dict(ports or {})
        self.init_errors = dict(init_errors or {})
        self.load_errors = dict(load_errors or {})
        self.set_errors = dict(set_errors or {})
        self.list_errors = dict(list_errors or {})
        self.types = dict(types or {})
        self.unreadable = set(unreadable)
        self.runtime_ports = set(runtime_ports)
        self.interrupt = set(interrupt)
        self.installed = installed
        self.listing_error = listing_error
        self.calls: list[tuple[str, ...]] = []
        self.initialised: set[str] = set()
        self.loaded: dict[str, str] = {}
        self.parents: dict[str, str] = {}

    # --- the adapter's listing -------------------------------------------------
    def getDeviceAdapterNames(self) -> tuple[str, ...]:  # noqa: N802
        return (self.adapter,) if self.installed else ()

    def getAvailableDevices(self, name: str) -> tuple[str, ...]:  # noqa: N802
        if self.listing_error is not None:
            raise RuntimeError(self.listing_error)
        return tuple(n for n, _ in self.listed)

    def getAvailableDeviceTypes(self, name: str) -> tuple[int, ...]:  # noqa: N802
        return tuple(_type_value(t) for _, t in self.listed)

    def getAvailableDeviceDescriptions(self, name: str) -> tuple[str, ...]:  # noqa: N802
        return tuple("N/A" for _ in self.listed)

    # --- devices ---------------------------------------------------------------
    def loadDevice(self, label: str, adapter: str, device: str) -> None:  # noqa: N802
        self.calls.append(("loadDevice", label, adapter, device))
        if not label or label == "Core" or "," in label:
            raise RuntimeError(f'Device label "{label}" contains reserved characters')
        if label in self.load_errors:
            raise RuntimeError(self.load_errors[label])
        self.loaded[label] = device

    def setParentLabel(self, label: str, parent: str) -> None:  # noqa: N802
        self.calls.append(("setParentLabel", label, parent))
        self.parents[label] = parent

    def _needs_parent(self, label: str) -> bool:
        if dict(self.listed).get(label) == "Hub":
            return False  # a hub has no parent, even one that reports itself
        if any(label in names for names in self.reported.values()):
            return True
        hub_listed = any(kind == "Hub" for _, kind in self.listed)
        return hub_listed and dict(self.listed).get(label, "Hub") != "Hub"

    def hasProperty(self, label: str, prop: str) -> bool:  # noqa: N802
        return prop == "Port" and label in self.ports

    def isPropertyPreInit(self, label: str, prop: str) -> bool:  # noqa: N802
        if not self.hasProperty(label, prop):
            raise RuntimeError(f"Property {prop} not found on {label}")
        return label not in self.runtime_ports

    def getProperty(self, label: str, prop: str) -> str:  # noqa: N802
        if label in self.unreadable:
            raise RuntimeError("cannot read the port")
        return self.ports[label]

    def setProperty(self, label: str, prop: str, value: str) -> None:  # noqa: N802
        self.calls.append(("setProperty", label, prop, value))
        if label in self.set_errors:
            raise RuntimeError(self.set_errors[label])
        self.ports[label] = value

    def initializeDevice(self, label: str) -> None:  # noqa: N802
        self.calls.append(("initializeDevice", label))
        if label in self.interrupt:
            raise KeyboardInterrupt
        if label in self.init_errors:
            raise RuntimeError(self.init_errors[label])
        if self._needs_parent(label) and label not in self.parents:
            raise RuntimeError(f"the hub of {label} is not set")
        pre_init_port = label in self.ports and label not in self.runtime_ports
        if pre_init_port and self.ports[label].strip().casefold() in _UNSET:
            raise RuntimeError("the serial port is not set")
        self.initialised.add(label)

    def getInstalledDevices(self, hub: str) -> tuple[str, ...]:  # noqa: N802
        self.calls.append(("getInstalledDevices", hub))
        if hub not in self.initialised:
            raise RuntimeError(
                "Operation (GetInstalledPeripheralNames) not permitted on "
                "uninitialized device"
            )
        if hub in self.list_errors:
            raise RuntimeError(self.list_errors[hub])
        return tuple(self.reported.get(hub, ()))

    def getDeviceType(self, label: str) -> int:  # noqa: N802
        listed = dict(self.listed)
        kind = self.types.get(label) or listed.get(label)
        return _type_value(kind) if kind else int(DeviceType.UnknownType)

    def unloadAllDevices(self) -> None:  # noqa: N802
        self.calls.append(("unloadAllDevices",))

    # --- what a test reads -----------------------------------------------------
    def calls_of(self, kind: str, label: str) -> list[tuple[str, ...]]:
        return [c for c in self.calls if c[0] == kind and c[1] == label]


def _walk(core: _ProbeCore, name: str = "Fake", **kwargs: object) -> ProbeResult:
    return walk_adapter(core, name, **kwargs)  # type: ignore[arg-type]


def _rows(result: ProbeResult) -> dict[str, ProbedDevice]:
    return {d.name: d for d in result.devices}


# --- the hub's peripherals (rule 4) ----------------------------------------------


def test_a_hub_reports_peripherals_the_adapter_does_not_list() -> None:
    core = _ProbeCore(
        "Fake",
        [("H", "Hub")],
        reported={"H": ["A", "B"]},
        types={"A": "Camera", "B": "Shutter"},
    )

    result = _walk(core)

    assert [d.name for d in result.devices] == ["H", "A", "B"]
    rows = _rows(result)
    assert (rows["H"].status, rows["H"].source, rows["H"].type) == (
        "ok",
        "adapter",
        "Hub",
    )
    assert (rows["A"].status, rows["A"].source, rows["A"].type) == (
        "ok",
        "hub",
        "Camera",
    )
    assert (rows["B"].status, rows["B"].source, rows["B"].type) == (
        "ok",
        "hub",
        "Shutter",
    )
    # The parent is named before the peripheral initialises (a Ti2 peripheral
    # looks its hub up in `Initialize`), and the hub itself has none.
    for peripheral in ("A", "B"):
        assert core.parents[peripheral] == "H"
        assert core.calls.index(("setParentLabel", peripheral, "H")) < core.calls.index(
            ("initializeDevice", peripheral)
        )
    assert "H" not in core.parents
    assert result.error is None


def test_a_peripheral_the_adapter_lists_and_the_hub_does_not_report_is_not_initialised() -> (
    None
):
    core = _ProbeCore(
        "Fake",
        [("H", "Hub"), ("X", "Camera"), ("Y", "Camera")],
        reported={"H": ["X"]},
    )

    result = _walk(core)

    rows = _rows(result)
    assert rows["X"].status == "ok"
    assert rows["X"].source == "both"
    assert rows["Y"].status == "not-probed"
    assert rows["Y"].reason is not None
    assert "not reported by the hub" in rows["Y"].reason
    # The row is the candidates first, then what the hub does not report.
    assert [d.name for d in result.devices] == ["H", "X", "Y"]
    # Never loaded, never initialised: a row that says ok for it would be a lie.
    assert core.calls_of("loadDevice", "Y") == []
    assert core.calls_of("initializeDevice", "Y") == []


def test_a_hub_that_reports_nothing_leaves_the_adapters_own_list_to_be_probed() -> None:
    core = _ProbeCore("Fake", [("H", "Hub"), ("X", "Camera"), ("Z", "Shutter")])

    result = _walk(core)

    rows = _rows(result)
    assert [rows[n].status for n in ("H", "X", "Z")] == ["ok", "ok", "ok"]
    assert (rows["X"].source, rows["Z"].source) == ("adapter", "adapter")
    assert ("setParentLabel", "X", "H") in core.calls
    assert result.error is None
    assert len(result.notes) == 1
    assert "reports no peripherals" in result.notes[0]


def test_a_hub_that_reports_nothing_with_nothing_else_listed_is_an_error_with_the_stands_meaning() -> (
    None
):
    core = _ProbeCore("NikonTi2", [("Ti2-E__0", "Hub")])

    result = _walk(core, "NikonTi2")

    assert result.error is not None
    assert nikon.TI2.empty_list_meaning in result.error
    assert "nothing below it was probed" in result.error
    # The hub itself initialised, and its row says so.
    assert [(d.name, d.status) for d in result.devices] == [("Ti2-E__0", "ok")]


@pytest.mark.parametrize(
    ("adapter", "meaning"),
    [
        ("NikonTi2", nikon.TI2.empty_list_meaning),
        ("NikonTI", nikon.TIE.empty_list_meaning),
        ("NoStandKnowsThis", diagnosis.GENERIC_EMPTY_LIST),
    ],
)
def test_an_adapter_that_lists_no_devices_is_an_error_with_the_stands_meaning(
    adapter: str, meaning: str
) -> None:
    core = _ProbeCore(adapter, [])

    result = _walk(core, adapter)

    assert result.error is not None
    assert f"{adapter} lists no devices" in result.error
    assert meaning in result.error
    assert result.devices == []
    assert core.calls == []


def test_an_adapter_that_is_not_installed_is_an_error_and_loads_nothing() -> None:
    core = _ProbeCore("Fake", [("A", "Camera")], installed=False)

    result = _walk(core)

    assert result.error == "adapter not installed"
    assert result.devices == []
    assert core.calls == []


def test_an_adapter_whose_listing_fails_is_an_error_with_mmcores_text() -> None:
    core = _ProbeCore("Fake", [], listing_error='Failed to load device adapter "Fake"')

    result = _walk(core)

    assert result.error == 'Failed to load device adapter "Fake"'
    assert result.devices == []


# --- hubs (rules 2, 6, 7) --------------------------------------------------------


def test_a_hub_that_will_not_initialise_carries_the_ordered_causes() -> None:
    raw = "(Error message unavailable) (-10)"
    core = _ProbeCore(
        "NikonTi2",
        [("Ti2-E__0", "Hub"), ("Nosepiece", "State"), ("Lamp", "Shutter")],
        init_errors={"Ti2-E__0": raw},
    )

    result = _walk(core, "NikonTi2")

    rows = _rows(result)
    assert rows["Ti2-E__0"].status == "failed"
    assert rows["Ti2-E__0"].error == raw  # MMCore's text, raw
    for name in ("Nosepiece", "Lamp"):
        assert rows[name].status == "not-probed"
        reason = rows[name].reason
        assert reason is not None
        assert "its hub" in reason
        assert "Ti2-E__0" in reason
        assert core.calls_of("loadDevice", name) == []
    assert result.causes == [nikon.ONE_CONNECTION, *nikon.TI2.causes]
    assert result.causes[0] == nikon.ONE_CONNECTION


def test_another_adapters_hub_failure_carries_the_three_generic_causes_in_order() -> (
    None
):
    core = _ProbeCore(
        "Fake", [("H", "Hub"), ("A", "Camera")], init_errors={"H": "no answer"}
    )

    result = _walk(core)

    assert result.causes == list(diagnosis.GENERIC_HUB_CAUSES)
    assert len(result.causes) == 3


def test_the_second_hub_is_not_tried_once_the_first_initialised() -> None:
    core = _ProbeCore(
        "Fake",
        [("H1", "Hub"), ("H2", "Hub")],
        reported={"H1": ["A"]},
        types={"A": "Camera"},
    )

    result = _walk(core)

    rows = _rows(result)
    assert rows["H1"].status == "ok"
    assert rows["H2"].status == "not-probed"
    reason = rows["H2"].reason
    assert reason is not None
    assert "already initialised" in reason
    assert "H1" in reason
    assert core.calls_of("loadDevice", "H2") == []
    assert core.calls_of("initializeDevice", "H2") == []
    assert not [d for d in result.devices if d.status == "failed"]
    assert result.causes == []
    # Hubs first, tried or not, then the hub's peripherals.
    assert [d.name for d in result.devices] == ["H1", "H2", "A"]


def test_a_failed_first_hub_and_a_working_second_one_give_a_note_and_the_second_is_used() -> (
    None
):
    core = _ProbeCore(
        "Fake",
        [("H1", "Hub"), ("H2", "Hub")],
        reported={"H2": ["A"]},
        types={"A": "Camera"},
        init_errors={"H1": "the stand is off"},
    )

    result = _walk(core)

    rows = _rows(result)
    assert (rows["H1"].status, rows["H2"].status, rows["A"].status) == (
        "failed",
        "ok",
        "ok",
    )
    assert ("setParentLabel", "A", "H2") in core.calls
    assert core.calls_of("getInstalledDevices", "H1") == []
    (note,) = result.notes
    assert "did not initialise but" in note
    assert "H1" in note
    assert "H2" in note
    # A hub row failed, so the causes are given: the stand may be off.
    assert result.causes == list(diagnosis.GENERIC_HUB_CAUSES)


def test_a_hub_that_cannot_list_its_peripherals_is_a_note_and_the_adapters_list_is_probed() -> (
    None
):
    core = _ProbeCore(
        "Fake",
        [("H", "Hub"), ("X", "Camera")],
        list_errors={"H": "listing exploded"},
    )

    result = _walk(core)

    rows = _rows(result)
    assert rows["H"].status == "ok"
    assert (rows["X"].status, rows["X"].source) == ("ok", "adapter")
    assert any(
        "did not list its peripherals" in n and "listing exploded" in n
        for n in result.notes
    )
    assert result.error is None


def test_a_hub_that_cannot_list_with_nothing_else_listed_does_not_claim_it_reported_none() -> (
    None
):
    core = _ProbeCore("Fake", [("H", "Hub")], list_errors={"H": "listing exploded"})

    result = _walk(core)

    assert result.error is not None
    assert "could not list its peripherals" in result.error
    assert "reports no peripherals" not in result.error


def test_with_no_hub_listed_every_device_is_probed_without_a_parent() -> None:
    core = _ProbeCore("Fake", [("A", "Camera"), ("B", "Shutter")])

    result = _walk(core)

    assert [(d.name, d.status, d.source) for d in result.devices] == [
        ("A", "ok", "adapter"),
        ("B", "ok", "adapter"),
    ]
    assert not [c for c in core.calls if c[0] == "setParentLabel"]
    assert not [c for c in core.calls if c[0] == "getInstalledDevices"]


def test_a_name_the_hub_reports_twice_or_that_is_already_loaded_is_probed_once() -> (
    None
):
    core = _ProbeCore(
        "Fake",
        [("H", "Hub")],
        reported={"H": ["A", "A", "H"]},
        types={"A": "Camera"},
    )

    result = _walk(core)

    assert [d.name for d in result.devices] == ["H", "A"]
    assert len(core.calls_of("loadDevice", "A")) == 1
    assert len(core.calls_of("loadDevice", "H")) == 1


# --- pre-init ports (rule 3) -----------------------------------------------------


@pytest.mark.parametrize("unset", ["Undefined", "UNDEFINED", " undefined "])
def test_a_device_with_port_undefined_is_reported_not_failed(unset: str) -> None:
    core = _ProbeCore("Fake", [("S", "Shutter")], ports={"S": unset})

    result = _walk(core)

    (row,) = result.devices
    assert row.status == "not-probed"
    assert row.reason is not None
    assert "needs a port" in row.reason
    assert "--port" in row.reason
    assert core.calls_of("initializeDevice", "S") == []
    assert row.error is None


def test_an_empty_port_counts_as_unset() -> None:
    core = _ProbeCore("Fake", [("S", "Shutter")], ports={"S": ""})

    result = _walk(core)

    (row,) = result.devices
    assert row.status == "not-probed"
    assert row.reason is not None
    assert "needs a port" in row.reason
    assert core.calls_of("initializeDevice", "S") == []


def test_a_port_that_is_already_set_is_left_alone() -> None:
    core = _ProbeCore("Fake", [("S", "Shutter")], ports={"S": "COM4"})

    result = _walk(core)

    (row,) = result.devices
    assert (row.status, row.port) == ("ok", None)
    assert not [c for c in core.calls if c[0] == "setProperty"]

    # --port does not overwrite a port the device already has.
    again = _ProbeCore("Fake", [("S", "Shutter")], ports={"S": "COM4"})
    _walk(again, port="COM9")
    assert not [c for c in again.calls if c[0] == "setProperty"]
    assert again.ports["S"] == "COM4"


def test_the_port_is_set_before_the_device_initialises() -> None:
    core = _ProbeCore("Fake", [("S", "Shutter")], ports={"S": ""})

    result = _walk(core, port="COM9")

    (row,) = result.devices
    assert (row.status, row.port) == ("ok", "COM9")
    set_at = core.calls.index(("setProperty", "S", "Port", "COM9"))
    init_at = core.calls.index(("initializeDevice", "S"))
    assert set_at < init_at
    assert result.notes == []


def test_a_port_nobody_needs_is_a_note() -> None:
    core = _ProbeCore("Fake", [("A", "Camera")])

    result = _walk(core, port="COM9")

    (note,) = result.notes
    assert "COM9" in note
    assert "was not used" in note
    assert not [c for c in core.calls if c[0] == "setProperty"]


def test_a_port_note_does_not_claim_devices_that_were_never_tried_were_checked() -> (
    None
):
    # The hub failed, so its peripherals were never looked at: "no device needs
    # a port" would be a guess about them.
    core = _ProbeCore(
        "Fake", [("H", "Hub"), ("A", "Camera")], init_errors={"H": "no answer"}
    )

    result = _walk(core, port="COM9")

    (note,) = result.notes
    assert "was not used" in note
    assert "no device of Fake needs a port" not in note


def test_a_port_that_cannot_be_set_is_that_devices_failure() -> None:
    core = _ProbeCore(
        "Fake",
        [("S", "Shutter"), ("A", "Camera")],
        ports={"S": ""},
        set_errors={"S": "port rejected"},
    )

    result = _walk(core, port="COM9")

    rows = _rows(result)
    assert rows["S"].status == "failed"
    assert rows["S"].error == "port rejected"
    assert core.calls_of("initializeDevice", "S") == []
    # The walk went on: one device's refusal is that device's finding.
    assert rows["A"].status == "ok"
    # The device did need the port, so it was not "unused".
    assert result.notes == []


def test_an_unreadable_port_property_does_not_stop_the_device_initialising() -> None:
    core = _ProbeCore("Fake", [("S", "Shutter")], ports={"S": "COM4"}, unreadable=["S"])

    result = _walk(core, port="COM9")

    (row,) = result.devices
    assert row.status == "ok"
    assert row.port is None
    assert core.calls_of("initializeDevice", "S") == [("initializeDevice", "S")]
    assert not [c for c in core.calls if c[0] == "setProperty"]


def test_a_hub_that_needs_a_port_is_not_a_failure_and_gives_no_causes() -> None:
    core = _ProbeCore("Fake", [("H", "Hub"), ("A", "Camera")], ports={"H": ""})

    result = _walk(core)

    rows = _rows(result)
    assert rows["H"].status == "not-probed"
    assert rows["A"].status == "not-probed"
    reason = rows["A"].reason
    assert reason is not None
    assert "its hub" in reason
    assert result.causes == []
    assert not [c for c in core.calls if c[0] == "initializeDevice"]


def test_a_port_that_is_not_a_pre_init_property_does_not_stop_the_device_initialising() -> (
    None
):
    # An empty runtime Port is no reason to skip the device: only a pre-init
    # Port has to be given before `initializeDevice`.
    core = _ProbeCore("Fake", [("S", "Shutter")], ports={"S": ""}, runtime_ports=["S"])

    result = _walk(core, port="COM9")

    (row,) = result.devices
    assert (row.status, row.port) == ("ok", None)
    assert core.calls_of("initializeDevice", "S") == [("initializeDevice", "S")]
    assert not [c for c in core.calls if c[0] == "setProperty"]


def test_a_device_that_fails_after_its_port_was_set_says_which_port() -> None:
    core = _ProbeCore(
        "Fake", [("S", "Shutter")], ports={"S": ""}, init_errors={"S": "no answer"}
    )

    result = _walk(core, port="COM9")

    (row,) = result.devices
    assert row.status == "failed"
    assert row.error == "no answer"
    # What was tried is on the row, though it did not work.
    assert row.port == "COM9"
    assert ("setProperty", "S", "Port", "COM9") in core.calls
    assert result.notes == []


def test_a_port_note_does_not_claim_a_device_that_failed_to_load_was_checked() -> None:
    # `S` could not even be loaded, so nobody asked it about its Port.
    core = _ProbeCore(
        "Fake",
        [("S", "Shutter"), ("A", "Camera")],
        ports={"S": ""},
        load_errors={"S": "vendor SDK missing"},
    )

    result = _walk(core, port="COM9")

    (note,) = result.notes
    assert "was not used" in note
    assert "no device of Fake needs a port" not in note
    assert "1 not tried" in note


@pytest.mark.parametrize("empty", ["", "  "])
def test_the_walk_refuses_a_port_check_port_refuses(empty: str) -> None:
    core = _ProbeCore("Fake", [("A", "Camera")])

    with pytest.raises(ValueError, match="empty"):
        _walk(core, port=empty)

    assert core.calls == []


# --- failures of one device (rule 9) ---------------------------------------------


def test_a_native_module_failure_names_the_vendor_runtime() -> None:
    raw = "Native module failed to load (6)"
    core = _ProbeCore(
        "HamamatsuHam",
        [("HamamatsuHam_DCAM", "Camera")],
        init_errors={"HamamatsuHam_DCAM": raw},
    )

    result = _walk(core, "HamamatsuHam")

    (row,) = result.devices
    assert row.status == "failed"
    assert row.error == raw
    assert row.hint is not None
    assert "DCAM-API" in row.hint
    # The causes are a hub's: a camera whose runtime is missing is not "the
    # stand is off" (the Hamamatsu is not fitted on the Ti2-E, and that is fine).
    assert result.causes == []

    other = _ProbeCore("Other", [("Cam", "Camera")], init_errors={"Cam": raw})
    (other_row,) = _walk(other, "Other").devices
    assert other_row.hint is not None
    assert "DCAM" not in other_row.hint


def test_a_label_mmcore_refuses_is_that_devices_failure() -> None:
    core = _ProbeCore("Fake", [("a,b", "Camera"), ("fine", "Camera")])

    result = _walk(core)

    rows = _rows(result)
    assert rows["a,b"].status == "failed"
    assert rows["a,b"].error is not None
    assert "reserved characters" in rows["a,b"].error
    assert rows["a,b"].type == "Camera"
    assert rows["fine"].status == "ok"
    assert result.causes == []


def test_a_peripheral_that_will_not_load_is_unknown_and_failed() -> None:
    core = _ProbeCore(
        "Fake",
        [("H", "Hub")],
        reported={"H": ["Ghost"]},
        load_errors={"Ghost": "no such device"},
    )

    result = _walk(core)

    ghost = _rows(result)["Ghost"]
    assert (ghost.status, ghost.type, ghost.source) == ("failed", "Unknown", "hub")
    assert ghost.error == "no such device"
    # The hub initialised: one peripheral that will not load is not a hub failure.
    assert result.causes == []


# --- checkpoints and ownership of the core (§9.5) --------------------------------


def test_checkpoints_follow_every_row_and_only_the_returned_result_is_complete() -> (
    None
):
    core = _ProbeCore(
        "Fake",
        [("H", "Hub"), ("X", "Camera"), ("Y", "Camera")],
        reported={"H": ["X"]},
    )
    copies: list[ProbeResult] = []

    result = _walk(core, checkpoint=copies.append)

    assert len(result.devices) == 3
    # Once before the first device, then once after every row.
    assert len(copies) == len(result.devices) + 1
    assert copies[0].devices == []
    assert [len(c.devices) for c in copies] == [0, 1, 2, 3]
    assert not any(c.complete for c in copies)
    assert result.complete
    # A copy, not the object the walk goes on filling.
    assert copies[0].devices == []
    assert all(c is not result for c in copies)


def test_the_checkpoint_of_a_failed_hub_already_carries_the_causes() -> None:
    # A crash right after the failed hub must leave the diagnosis with it.
    core = _ProbeCore(
        "Fake", [("H", "Hub"), ("A", "Camera")], init_errors={"H": "no answer"}
    )
    copies: list[ProbeResult] = []

    result = _walk(core, checkpoint=copies.append)

    after_hub = copies[1]
    assert [(d.name, d.status) for d in after_hub.devices] == [("H", "failed")]
    assert after_hub.causes == list(diagnosis.GENERIC_HUB_CAUSES)
    assert copies[0].causes == []
    assert result.causes == list(diagnosis.GENERIC_HUB_CAUSES)


def test_progress_names_each_device_and_the_hubs_listing_step() -> None:
    core = _ProbeCore("Fake", [("H", "Hub")], reported={"H": ["A"]})
    lines: list[str] = []

    _walk(core, progress=lines.append)

    # The parent reads the last line that starts with `probing ` to name the
    # device that was running when the child died (`_PROGRESS_PREFIXES`).
    assert lines == [
        "probing Fake/H",
        "probing Fake/H: listing its peripherals",
        "probing Fake/A",
    ]


def test_walk_adapter_leaves_the_devices_loaded_and_probe_adapter_releases_them() -> (
    None
):
    walked = _ProbeCore("Fake", [("A", "Camera")])
    _walk(walked)
    # #15 builds on what initialised: the walk never closes the core.
    assert ("unloadAllDevices",) not in walked.calls

    given = _ProbeCore("Fake", [("A", "Camera")])
    result = probe_adapter("Fake", core=given)  # type: ignore[arg-type]
    assert [d.status for d in result.devices] == ["ok"]
    assert given.calls[-1] == ("unloadAllDevices",)


def test_probe_adapter_releases_the_core_it_was_given_when_nothing_can_be_probed() -> (
    None
):
    core = _ProbeCore("Fake", [], installed=False)

    result = probe_adapter("Fake", core=core)  # type: ignore[arg-type]

    assert result.error == "adapter not installed"
    assert core.calls == [("unloadAllDevices",)]


def test_an_interrupt_during_a_probe_still_releases_the_core() -> None:
    core = _ProbeCore("Fake", [("A", "Camera")], interrupt=["A"])

    with pytest.raises(KeyboardInterrupt):
        probe_adapter("Fake", core=core)  # type: ignore[arg-type]

    assert core.calls[-1] == ("unloadAllDevices",)


# --- the listing's meaning -------------------------------------------------------


def test_list_adapters_sets_the_meaning_of_an_empty_or_fixed_list() -> None:
    (empty,) = list_adapters(_ProbeCore("NikonTi2", []), ["NikonTi2"])  # type: ignore[arg-type]
    (fixed,) = list_adapters(
        _ProbeCore("NikonTI", [("TIScope", "Hub"), ("TIZDrive", "Stage")]),  # type: ignore[arg-type]
        ["NikonTI"],
    )
    (plain,) = list_adapters(
        _ProbeCore("DemoCamera", [("DCam", "Camera")]), ["DemoCamera"]
    )  # type: ignore[arg-type]
    (broken,) = list_adapters(
        _ProbeCore("NikonTi2", [], listing_error="Failed to load"),  # type: ignore[arg-type]
        ["NikonTi2"],
    )

    assert empty.list_meaning == nikon.TI2.empty_list_meaning
    assert fixed.list_meaning == nikon.TIE.empty_list_meaning
    assert plain.list_meaning is None
    # A listing that raised has an error to show, and no meaning to add to it.
    assert broken.error == "Failed to load"
    assert broken.list_meaning is None


# --- skip (design m2 §10.2) -------------------------------------------------------


def test_a_skipped_device_is_never_loaded_and_says_why() -> None:
    core = _ProbeCore(
        "Fake",
        [("H", "Hub")],
        reported={"H": ["A", "Dia"]},
        types={"A": "Camera", "Dia": "Shutter"},
    )
    seen: list[str] = []

    result = _walk(core, skip=["Dia"], progress=seen.append)

    rows = _rows(result)
    assert (rows["A"].status, rows["Dia"].status) == ("ok", "not-probed")
    assert rows["Dia"].reason == "left out: --skip Dia"
    # Loading it is what crashes the Ti-E's TIDiaLamp: not one call may reach it.
    assert core.calls_of("loadDevice", "Dia") == []
    assert core.calls_of("setParentLabel", "Dia") == []
    assert core.calls_of("initializeDevice", "Dia") == []
    # The child's progress line names the device it is on when it dies: a device
    # nobody touched must not be that device.
    assert "probing Fake/Dia" not in seen
    assert "probing Fake/A" in seen


def test_a_skipped_hub_lets_the_next_listed_hub_connect_without_causes() -> None:
    core = _ProbeCore(
        "Fake",
        [("Stand", "Hub"), ("Simulator", "Hub")],
        reported={"Simulator": ["A"]},
        types={"A": "Camera"},
    )

    result = _walk(core, skip=["Stand"])

    rows = _rows(result)
    assert (rows["Stand"].status, rows["Simulator"].status, rows["A"].status) == (
        "not-probed",
        "ok",
        "ok",
    )
    assert core.calls_of("loadDevice", "Stand") == []
    # Not a failed hub: nothing says the stand is off, and no note says a hub
    # "did not initialise".
    assert result.causes == []
    assert result.notes == []
    assert ("setParentLabel", "A", "Simulator") in core.calls


def test_a_skipped_device_counts_among_those_the_port_note_did_not_look_at() -> None:
    core = _ProbeCore("Fake", [("H", "Hub"), ("S", "Shutter")])

    result = _walk(core, port="COMX", skip=["S"])

    (note,) = [n for n in result.notes if "--port" in n]
    assert "--port COMX was not used" in note
    # "no device of Fake needs a port" would claim a device nobody looked at.
    assert "no device that was initialised needs a port (1 not tried)" in note


def test_probe_adapter_passes_skip_to_the_walk() -> None:
    core = _ProbeCore("Fake", [("A", "Camera"), ("B", "Camera")])

    result = probe_adapter("Fake", skip=["B"], core=core)  # type: ignore[arg-type]

    assert _rows(result)["B"].status == "not-probed"
    assert core.calls_of("loadDevice", "B") == []
