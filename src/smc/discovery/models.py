"""What a survey of a microscope PC records.

The ``Inventory`` is written to ``inventory.json`` and read back by the design
session to draft a profile, so it is a pydantic model: the file validates
against the same class that produced it. Every section can be empty, and
``notes`` says why — a PC without ``lspci`` or with a crashing vendor DLL
still yields a survey (failures are findings, not exceptions).
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

__all__ = [
    "AdapterDevice",
    "AdapterInfo",
    "AdaptersSection",
    "DeviceEntry",
    "Hint",
    "Inventory",
    "ProbeResult",
    "ProbedDevice",
    "SerialPort",
    "SystemInfo",
]


class SystemInfo(BaseModel):
    """The machine and the software stack that produced the survey."""

    hostname: str
    #: UTC; serialised as ISO 8601.
    collected_at: datetime
    os: str
    python: str
    smc: str
    pymmcore_plus: str
    #: The Micro-Manager folder pymmcore-plus uses; ``None`` when none is found.
    mm_install: str | None = None
    #: e.g. ``"Device API version 75, Module API version 10"``. A ``.cfg`` saved
    #: by an MMStudio with a different device API fails to load here.
    device_api: str = ""
    #: Folder names of other Micro-Manager installs on disk (names only).
    other_mm_installs: list[str] = Field(default_factory=list)


class SerialPort(BaseModel):
    """One serial port, as the OS reports it (``pyserial``)."""

    device: str
    #: Hex, upper case, without ``0x``; ``None`` for a port that is not USB.
    vendor_id: str | None = None
    product_id: str | None = None
    manufacturer: str | None = None
    description: str | None = None
    serial_number: str | None = None
    #: Vendors the hints table matched on this port.
    hints: list[str] = Field(default_factory=list)


class DeviceEntry(BaseModel):
    """One USB, PnP or PCI device.

    The same shape for every OS so the hints and the report need not care
    where it came from; ``source`` says which tool reported it.
    """

    #: ``"pnp"`` (Windows), ``"system_profiler"`` (macOS), ``"lsusb"`` or
    #: ``"lspci"`` (Linux).
    source: str
    name: str
    #: Hex, upper case, without ``0x``. For PCI: the vendor and device IDs.
    vendor_id: str | None = None
    product_id: str | None = None
    manufacturer: str | None = None
    serial_number: str | None = None
    #: PnP class (``Ports``, ``Camera``, …) or PCI class.
    device_class: str | None = None
    #: PnP status (``OK``, ``Error``, ``Unknown``): a device with a missing
    #: driver shows up here, which is itself a finding.
    status: str | None = None
    #: Windows PnP instance ID (``USB\VID_0403&PID_6001\…``, ``PCI\VEN_…``).
    instance_id: str | None = None
    hints: list[str] = Field(default_factory=list)


class Hint(BaseModel):
    """A vendor recognised on this PC, and the adapters it suggests."""

    vendor: str
    #: Which entry matched, e.g. ``"serial COM3"`` or ``"pci Photometrics …"``.
    matched: str
    adapters: list[str] = Field(default_factory=list)
    note: str = ""


class AdapterDevice(BaseModel):
    """One device an adapter says it provides (listed, not loaded)."""

    name: str
    #: Micro-Manager device type name (``CameraDevice``, ``HubDevice``, …).
    type: str
    description: str = ""


class AdapterInfo(BaseModel):
    """The devices of one adapter, or why they could not be listed."""

    name: str
    installed: bool
    devices: list[AdapterDevice] = Field(default_factory=list)
    #: The enumeration error. On a Nikon PC without the vendor DLL next to the
    #: adapter, this text *is* the finding.
    error: str | None = None


class AdaptersSection(BaseModel):
    """Installed adapter names, and the devices of the ones worth listing."""

    #: Every installed adapter's name. Listing names loads no DLL.
    installed: list[str] = Field(default_factory=list)
    #: Adapters whose devices were enumerated (in the child process).
    listed: list[AdapterInfo] = Field(default_factory=list)


class ProbedDevice(BaseModel):
    """One device loaded and initialised by ``--probe-adapter``."""

    name: str
    type: str
    ok: bool
    error: str | None = None


class ProbeResult(BaseModel):
    """The outcome of initialising every device of one adapter."""

    adapter: str
    devices: list[ProbedDevice] = Field(default_factory=list)
    #: Set when the adapter could not be probed at all.
    error: str | None = None


class Inventory(BaseModel):
    """A read-only survey of one PC."""

    system: SystemInfo
    adapters: AdaptersSection = Field(default_factory=AdaptersSection)
    serial: list[SerialPort] = Field(default_factory=list)
    usb: list[DeviceEntry] = Field(default_factory=list)
    pci: list[DeviceEntry] = Field(default_factory=list)
    hints: list[Hint] = Field(default_factory=list)
    probe: ProbeResult | None = None
    #: One line per thing that could not be collected.
    notes: list[str] = Field(default_factory=list)
