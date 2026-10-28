"""``FakeCore``: an in-memory stand-in for ``CMMCorePlus`` (design §8).

The demo devices are the regression bed, but they never misbehave. This
fake covers what the simulator cannot show: a device that stays busy, a
device that fails, a vendor quirk. It implements the calls the layer makes
(design §6, ``roles._InventoryCore``, the facade's ``setTimeoutMs``,
``getStateLabel`` and ``unloadAllDevices``) with MMCore's names, argument
order and error messages, so a test written against it also reads like the
real thing. It does not subclass ``CMMCorePlus``: nothing here loads a
device adapter, so the default suite stays away from vendor DLLs (FM-40).

Behaviour copied from the demo, measured 2026-10-01 (pymmcore-plus 0.18.1,
pymmcore 12.5.0.75.0), unless labelled *assumed*:

* **A set on a read-only property is ignored**, without an error. Only the
  backend's own check refuses it, so a contract that expects the refusal
  fails if that check is removed (FM-43).
* **One store.** The camera's ``Exposure`` and a ``State`` device's
  ``State`` and ``Label`` are properties; the dedicated calls
  (``setExposure``, ``setState``, ``getStateLabel``) read and write those
  same properties, so ``Properties`` and the capabilities never disagree.
* **Motion is instant.** A device reads busy only while its label is in
  ``busy_devices``, and ``stop()`` does not take it out: a stuck device
  stays stuck (FM-17).
* **The Nikon quirk** (FM-12): with ``quirk_z_move_disables_autofocus``, a
  ``setPosition`` on the focus device switches continuous focus off, as
  moving the Z drive does with PFS on the Nikon stands. The demo keeps it on.
* **Continuous focus acts on the AutoFocus slot** (FM-87; measured on the
  demo 2026-10-16). ``enableContinuousFocus``, ``isContinuousFocus*`` and
  ``get``/``setAutoFocusOffset`` take no label: they act on the device the
  slot names, each such device keeping its own state and one offset
  (``0.0`` at construction; a set moves nothing). With the slot empty,
  ``enableContinuousFocus(True)`` raises ``Auto-focus focus device not
  defined.``, ``enableContinuousFocus(False)`` does nothing, both
  ``isContinuousFocus*`` read ``False``, and both offset calls raise
  ``Error (code 43)``. Enabled reads locked at once, as on the demo
  (assumed: a real lock takes time), unless ``autofocus_lockable`` is
  ``False``.
* **Config groups** (measured on the demo 2026-10-19, m2 §8.1). Groups and
  presets list sorted. ``defineConfig`` *merges*: it changes or adds one
  setting and keeps the others, creates a missing group, and accepts a
  device that is not loaded without a word. ``deleteConfig`` of a missing
  preset names the group, not the preset, and a group stays once its last
  preset is gone. ``setConfig`` applies each setting in order through the
  property store, a ``Core,Shutter`` setting through the shutter slot; a
  device that is not loaded raises there. ``getCurrentConfig`` is not
  implemented: it compares strings (``10`` against ``10.0000``), and the
  layer never calls it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np
from pymmcore_plus import DeviceType

__all__ = ["FakeCore", "FakeDevice", "FakeProperty"]

#: MMCore's own pseudo-device, listed last by ``getLoadedDevices``.
_CORE_LABEL = "Core"

#: The core slots: constructor keyword, MMCore's Core property, the device type.
_SLOTS: dict[str, tuple[str, DeviceType]] = {
    "camera": ("Camera", DeviceType.Camera),
    "xy_stage": ("XYStage", DeviceType.XYStage),
    "focus": ("Focus", DeviceType.Stage),
    "autofocus": ("AutoFocus", DeviceType.AutoFocus),
    "shutter": ("Shutter", DeviceType.Shutter),
}

#: What MMCore says when the current camera slot is empty (measured).
_NO_CAMERA = "Camera not loaded or initialized."
#: What MMCore says to ``enableContinuousFocus(True)`` with the AutoFocus slot
#: empty, and to either offset call (measured on the demo, 2026-10-16).
_NO_AUTOFOCUS = "Auto-focus focus device not defined."
_NO_AUTOFOCUS_OFFSET = "Error (code 43)"

#: One preset's settings: ``(device, property, value)``, in the order applied.
Settings = list[tuple[str, str, str]]


@dataclass(slots=True)
class FakeProperty:
    """One device property: its value as MMCore reports it, and its constraints.

    ``lower``/``upper`` are the property limits (both or neither), and
    ``allowed`` the values a set may take; empty means anything.
    """

    value: str
    read_only: bool = False
    allowed: tuple[str, ...] = ()
    lower: float | None = None
    upper: float | None = None


@dataclass(slots=True)
class FakeDevice:
    """One loaded device, as the inventory calls describe it.

    ``state_labels`` gives a ``State`` device one label per position; its
    ``State`` and ``Label`` properties are added when missing, as a camera's
    ``Exposure`` is, because MMCore's dedicated calls read them.
    """

    label: str
    type: DeviceType
    library: str = "FakeCore"
    name: str = ""
    description: str = ""
    properties: dict[str, FakeProperty] = field(default_factory=dict)
    state_labels: tuple[str, ...] = ()


class FakeCore:
    """The MMCore calls the layer makes, in memory, with a log of every mutation.

    Knobs a test sets directly:

    * ``log``: every mutating call as it is made, ``"setXYPosition('XY',
      20.0, -10.0)"``, including one that then raises. Reads are not logged.
      Building a facade logs ``setTimeoutMs(...)``, so tests clear the log
      after building it.
    * ``frame_source``: called as ``(x_um, y_um, z_um)`` at each snap, with
      the positions of the XY and focus slot devices (0.0 for an empty slot);
      ``None`` gives a frame of zeros of ``image_shape``.
    * ``busy_devices``: labels that read busy until removed.
    * ``stop_refused``: labels whose ``stop()`` is logged, then raises the
      Ti2-E's ``Error in device "<label>": Unsupported device command (11)``
      (m2 §11.8), a fresh error each time. ``busy_devices`` is left alone, so
      a busy device stays busy: the move goes on.
    * ``failing``: label -> exception raised by every call whose first
      argument is that label; for the camera slot's device, by the
      current-camera calls too, and for the AutoFocus slot's device, by the
      continuous-focus and autofocus-offset calls.
    * ``pixel_size_um``, ``image_shape`` (height, width), ``bit_depth``: what
      the camera calls report.
    * ``magnification``: label -> what that ``Magnifier`` device reads (a
      label with no entry reads 1.0); ``getMagnificationFactor()`` is the
      product over the loaded ``Magnifier`` devices (m2 §8.10). The fake does
      not divide ``pixel_size_um`` by it: a test sets the calibrated value it
      wants MMCore to report.
    * ``quirk_z_move_disables_autofocus``: the Nikon quirk (FM-12).
    * ``autofocus_lockable``: ``False`` and the autofocus never reads
      locked, even enabled, as with no reference in range.
    * ``config_groups``: group -> preset -> its settings, which the
      config-group calls read and write; a test may fill it directly.
    """

    def __init__(
        self,
        devices: Iterable[FakeDevice] = (),
        *,
        camera: str = "",
        xy_stage: str = "",
        focus: str = "",
        autofocus: str = "",
        shutter: str = "",
    ) -> None:
        """Load ``devices`` in order and fill the five core slots.

        Raises:
            ValueError: A label is used twice or is ``"Core"``, or a slot names
                no device of its type: a test that misspells a label would
                otherwise build a stand that silently lacks a role.
        """
        self.log: list[str] = []
        self.frame_source: Callable[[float, float, float], np.ndarray] | None = None
        self.busy_devices: set[str] = set()
        self.stop_refused: set[str] = set()
        self.failing: dict[str, BaseException] = {}
        self.pixel_size_um = 1.0
        self.magnification: dict[str, float] = {}
        self.image_shape: tuple[int, int] = (512, 512)
        self.bit_depth = 16
        self.quirk_z_move_disables_autofocus = True
        self.autofocus_lockable = True
        self.config_groups: dict[str, dict[str, Settings]] = {}
        self._channel_group = ""

        self._devices: dict[str, FakeDevice] = {}
        for device in devices:
            if device.label == _CORE_LABEL or device.label in self._devices:
                raise ValueError(
                    f"device label {device.label!r} is reserved or used twice"
                )
            self._devices[device.label] = device
            _complete(device)
        self._core_device = FakeDevice(_CORE_LABEL, DeviceType.Core, library="")
        self._xy: dict[str, tuple[float, float]] = {
            d.label: (0.0, 0.0)
            for d in self._devices.values()
            if d.type == DeviceType.XYStage
        }
        self._z: dict[str, float] = {
            d.label: 0.0 for d in self._devices.values() if d.type == DeviceType.Stage
        }
        self._open: dict[str, bool] = {
            d.label: False
            for d in self._devices.values()
            if d.type == DeviceType.Shutter
        }
        self._auto_shutter = True
        autofocus_devices = [
            d.label for d in self._devices.values() if d.type == DeviceType.AutoFocus
        ]
        #: Per AutoFocus device: whether continuous focus is on, and its offset.
        self._continuous: dict[str, bool] = dict.fromkeys(autofocus_devices, False)
        self._af_offset: dict[str, float] = dict.fromkeys(autofocus_devices, 0.0)
        self._timeout_ms = 5000
        self._image: np.ndarray | None = None
        self._slots: dict[str, str] = {}
        given = {
            "camera": camera,
            "xy_stage": xy_stage,
            "focus": focus,
            "autofocus": autofocus,
            "shutter": shutter,
        }
        for slot, label in given.items():
            if label and not self._fits(slot, label):
                raise ValueError(
                    f"{slot}={label!r} names no loaded device of type "
                    f"{_SLOTS[slot][1].name}"
                )
            self._slots[slot] = label

    @classmethod
    def demo_like(cls, extra_devices: Iterable[FakeDevice] = ()) -> FakeCore:
        """The demo configuration's devices, slots, State labels and camera.

        Copied from the demo as measured on 2026-10-01; a demo test keeps
        them equal (``tests/test_fake_core_demo.py``), and where the two
        disagree the fake is wrong. One difference is deliberate: the demo's
        White Light Shutter starts open on some opens (FM-47), and the fake's
        always starts closed.

        ``extra_devices`` are loaded after the demo's own, so a test can add
        what the demo lacks (a ``Magnifier``, m2 §8.10) without changing the
        copy; the inventory test keeps comparing the default.
        """
        devices = [
            FakeDevice("DHub", DeviceType.Hub, "DemoCamera", "DHub", "DHub"),
            _demo(
                "Camera",
                DeviceType.Camera,
                "DCam",
                "Demo camera",
                "Demo Camera Device Adapter",
                extra={
                    "Binning": FakeProperty("1", allowed=("1", "2", "4", "8")),
                    "Exposure": FakeProperty("10.0000", lower=0.0, upper=10000.0),
                    "PixelType": FakeProperty("16bit"),
                    "CameraName": FakeProperty("DemoCamera-MultiMode", read_only=True),
                    "CameraID": FakeProperty("V1.0", read_only=True),
                },
            ),
            _demo_wheel(
                "Dichroic",
                ("400DCLP", "Q505LP", "Q585LP", "89402bs", *_numbered(4, 10)),
            ),
            _demo_wheel(
                "Emission",
                (
                    "Chroma-HQ620",
                    "Chroma-D460",
                    "Chroma-HQ535",
                    "Chroma-HQ700",
                    "89402m",
                    *_numbered(5, 10),
                ),
            ),
            _demo_wheel(
                "Excitation",
                (
                    "Chroma-D360",
                    "Chroma-HQ480",
                    "Chroma-HQ570",
                    "Chroma-HQ620",
                    *_numbered(4, 9),
                    "Empty",
                ),
            ),
            _demo(
                "Objective",
                DeviceType.State,
                "DObjective",
                "Demo objective turret",
                "Demo objective turret driver",
                labels=(
                    "Nikon 40X Plan Fluor ELWD",
                    "Nikon 10X S Fluor",
                    "Objective-2",
                    "Nikon 20X Plan Fluor ELWD",
                    "Objective-4",
                    "Objective-5",
                ),
                position=1,
            ),
            _demo("Z", DeviceType.Stage, "DStage", "Demo stage", "Demo stage driver"),
            _demo(
                "Path",
                DeviceType.State,
                "DLightPath",
                "Demo light path",
                "Demo light-path driver",
                labels=_numbered(0, 3),
            ),
            _demo(
                "XY",
                DeviceType.XYStage,
                "DXYStage",
                "Demo XY stage",
                "Demo XY stage driver",
            ),
            _demo(
                "White Light Shutter",
                DeviceType.Shutter,
                "DShutter",
                "Demo shutter",
                "Demo shutter driver",
            ),
            _demo(
                "Autofocus",
                DeviceType.AutoFocus,
                "DAutoFocus",
                "Demo auto focus",
                "Demo auto-focus adapter",
            ),
            _demo(
                "LED",
                DeviceType.State,
                "DStateDevice",
                "Demo State Device",
                "Demo state device driver",
                labels=(
                    "Closed",
                    "385nm",
                    "470nm",
                    "550nm",
                    "635nm",
                    *_numbered(5, 10),
                ),
            ),
            _demo(
                "LED Shutter",
                DeviceType.Shutter,
                "State Device Shutter",
                "State device used as a shutter",
                "State device that is used as a shutter",
                library="Utilities",
            ),
        ]
        core = cls(
            [*devices, *extra_devices],
            camera="Camera",
            xy_stage="XY",
            focus="Z",
            autofocus="Autofocus",
            shutter="White Light Shutter",
        )
        core.config_groups = _demo_config_groups()
        core._channel_group = "Channel"
        return core

    # --- helpers -------------------------------------------------------------

    def _record(self, name: str, *args: object) -> None:
        self.log.append(f"{name}({', '.join(map(repr, args))})")

    def _device(self, label: str) -> FakeDevice:
        """The device ``label`` names, after raising its ``failing`` entry, if any."""
        if label in self.failing:
            raise self.failing[label]
        if label == _CORE_LABEL:
            return self._core_device
        try:
            return self._devices[label]
        except KeyError:
            raise RuntimeError(f'No device with label "{label}"') from None

    def _typed(self, label: str, kind: DeviceType) -> FakeDevice:
        device = self._device(label)
        if device.type != kind:
            raise RuntimeError(
                f'Device "{label}" is of the wrong type for the requested operation'
            )
        return device

    def _property(self, label: str, name: str) -> FakeProperty:
        """A property for the calls whose unknown-property error is the device's (measured)."""
        found = self._device(label).properties.get(name)
        if found is None:
            raise RuntimeError(
                f'Error in device "{label}": Invalid property name encountered: '
                f"{name} (2)"
            )
        return found

    def _camera(self) -> FakeDevice | None:
        """The current camera, after raising its ``failing`` entry; ``None`` if the slot is empty."""
        label = self._slots["camera"]
        return self._device(label) if label else None

    def _fits(self, slot: str, label: str) -> bool:
        device = self._devices.get(label)
        return device is not None and device.type == _SLOTS[slot][1]

    def _set_slot(self, slot: str, label: str) -> None:
        if label and not self._fits(slot, label):
            # pymmcore-plus's message for a label of the wrong type or none.
            raise ValueError(
                f'Cannot set Core property {_SLOTS[slot][0]} to invalid value "{label}"'
            )
        self._slots[slot] = label

    def _slot_label(self, slot: str, label: str | None) -> str:
        return self._slots[slot] if label is None else label

    # --- inventory -----------------------------------------------------------

    def getLoadedDevices(self) -> tuple[str, ...]:
        return (*self._devices, _CORE_LABEL)

    def getDeviceType(self, label: str) -> DeviceType:
        return self._device(label).type

    def getDeviceLibrary(self, label: str) -> str:
        return self._device(label).library

    def getDeviceName(self, label: str) -> str:
        return self._device(label).name

    def getDeviceDescription(self, label: str) -> str:
        return self._device(label).description

    # --- the core slots ------------------------------------------------------

    def getCameraDevice(self) -> str:
        return self._slots["camera"]

    def setCameraDevice(self, label: str) -> None:
        self._record("setCameraDevice", label)
        self._set_slot("camera", label)

    def getXYStageDevice(self) -> str:
        return self._slots["xy_stage"]

    def setXYStageDevice(self, label: str) -> None:
        self._record("setXYStageDevice", label)
        self._set_slot("xy_stage", label)

    def getFocusDevice(self) -> str:
        return self._slots["focus"]

    def setFocusDevice(self, label: str) -> None:
        self._record("setFocusDevice", label)
        self._set_slot("focus", label)

    def getAutoFocusDevice(self) -> str:
        return self._slots["autofocus"]

    def setAutoFocusDevice(self, label: str) -> None:
        self._record("setAutoFocusDevice", label)
        self._set_slot("autofocus", label)

    def getShutterDevice(self) -> str:
        return self._slots["shutter"]

    def setShutterDevice(self, label: str) -> None:
        self._record("setShutterDevice", label)
        self._set_slot("shutter", label)

    def getTimeoutMs(self) -> int:
        return self._timeout_ms

    def setTimeoutMs(self, timeout_ms: int) -> None:
        self._record("setTimeoutMs", timeout_ms)
        self._timeout_ms = int(timeout_ms)

    # --- stages --------------------------------------------------------------

    def getXPosition(self, label: str | None = None) -> float:
        return self.getXYPosition(label)[0]

    def getYPosition(self, label: str | None = None) -> float:
        return self.getXYPosition(label)[1]

    def getXYPosition(self, label: str | None = None) -> list[float]:
        """``[x, y]`` of ``label``, or of the XY slot's device when omitted, as MMCore."""
        label = self._slot_label("xy_stage", label)
        self._typed(label, DeviceType.XYStage)
        return list(self._xy[label])

    def setXYPosition(self, label: str, x_um: float, y_um: float) -> None:
        self._record("setXYPosition", label, x_um, y_um)
        self._typed(label, DeviceType.XYStage)
        self._xy[label] = (float(x_um), float(y_um))

    def getPosition(self, label: str | None = None) -> float:
        """The position of ``label``, or of the focus slot's device when omitted."""
        label = self._slot_label("focus", label)
        self._typed(label, DeviceType.Stage)
        return self._z[label]

    def setPosition(self, label: str, z_um: float) -> None:
        """Move a stage at once; on the focus device, the Nikon quirk may apply (FM-12)."""
        self._record("setPosition", label, z_um)
        self._typed(label, DeviceType.Stage)
        self._z[label] = float(z_um)
        autofocus = self._slots["autofocus"]
        if (
            self.quirk_z_move_disables_autofocus
            and label == self._slots["focus"]
            and autofocus
        ):
            self._continuous[autofocus] = False

    def deviceBusy(self, label: str) -> bool:
        """Busy only while the label is in ``busy_devices``: motion is instant."""
        self._device(label)
        return label in self.busy_devices

    def stop(self, label: str) -> None:
        """Accepted for a stage; ``busy_devices`` is left alone, so a stuck device stays stuck.

        A label in ``stop_refused`` raises the Ti2-E's refusal instead, measured
        in session 1 for its XY stage, Z drive and PFS offset (#208).
        """
        self._record("stop", label)
        device = self._device(label)
        if label in self.stop_refused:
            raise RuntimeError(
                f'Error in device "{label}": Unsupported device command (11)'
            )
        if device.type not in (DeviceType.Stage, DeviceType.XYStage):
            raise RuntimeError(f'Cannot stop "{label}": not a stage')

    # --- camera --------------------------------------------------------------

    def snapImage(self) -> None:
        """Take a frame from ``frame_source`` at the slot devices' positions, or zeros."""
        self._record("snapImage")
        if self._camera() is None:
            raise RuntimeError(_NO_CAMERA)
        if self.frame_source is None:
            self._image = np.zeros(self.image_shape, dtype=_dtype(self.bit_depth))
            return
        xy_label, z_label = self._slots["xy_stage"], self._slots["focus"]
        x_um, y_um = self._xy[xy_label] if xy_label else (0.0, 0.0)
        z_um = self._z[z_label] if z_label else 0.0
        self._image = self.frame_source(x_um, y_um, z_um)

    def getImage(self) -> np.ndarray:
        """The last snapped frame; before any snap, MMCore's error (measured)."""
        self._camera()
        if self._image is None:
            raise RuntimeError("Issue snapImage before getImage.")
        return self._image

    def getExposure(self) -> float:
        """The camera's ``Exposure`` property; 0.0 with no camera (measured)."""
        camera = self._camera()
        if camera is None:
            return 0.0
        return float(camera.properties["Exposure"].value)

    def setExposure(self, exposure_ms: float) -> None:
        """Write the camera's ``Exposure`` property as the demo formats it.

        With no camera it raises like ``snapImage`` (assumed).
        """
        self._record("setExposure", exposure_ms)
        camera = self._camera()
        if camera is None:
            raise RuntimeError(_NO_CAMERA)
        camera.properties["Exposure"].value = f"{float(exposure_ms):.4f}"

    def getImageHeight(self) -> int:
        return self.image_shape[0] if self._camera() is not None else 0

    def getImageWidth(self) -> int:
        return self.image_shape[1] if self._camera() is not None else 0

    def getImageBitDepth(self) -> int:
        return self.bit_depth if self._camera() is not None else 0

    def getPixelSizeUm(self) -> float:
        """``pixel_size_um`` as set; 0.0 is MMCore's "no calibration"."""
        self._camera()
        return self.pixel_size_um

    def getMagnificationFactor(self) -> float:
        """The product of ``magnification`` over the ``Magnifier`` devices; 1.0 with none.

        MMCore multiplies every ``Magnifier`` device (two optovars at 1.6x
        read 2.56, measured 2026-10-20). A device that fails raises its
        ``failing`` entry, as a real one would when asked.
        """
        factor = 1.0
        for device in self._devices.values():
            if device.type == DeviceType.Magnifier:
                self._device(device.label)
                factor *= self.magnification.get(device.label, 1.0)
        return factor

    # --- shutters ------------------------------------------------------------

    def getShutterOpen(self, label: str | None = None) -> bool:
        """Whether ``label``, or the shutter slot's device when omitted, is open."""
        label = self._slot_label("shutter", label)
        self._typed(label, DeviceType.Shutter)
        return self._open[label]

    def setShutterOpen(self, label: str, open_: bool) -> None:
        self._record("setShutterOpen", label, open_)
        self._typed(label, DeviceType.Shutter)
        self._open[label] = bool(open_)

    def getAutoShutter(self) -> bool:
        return self._auto_shutter

    def setAutoShutter(self, on: bool) -> None:
        self._record("setAutoShutter", on)
        self._auto_shutter = bool(on)

    # --- properties ----------------------------------------------------------

    def getDevicePropertyNames(self, label: str) -> tuple[str, ...]:
        """Sorted, as MMCore keeps them."""
        return tuple(sorted(self._device(label).properties))

    def hasProperty(self, label: str, name: str) -> bool:
        return name in self._device(label).properties

    def getProperty(self, label: str, name: str) -> str:
        found = self._device(label).properties.get(name)
        if found is None:
            raise RuntimeError(f'Cannot get value of property "{name}"')
        return found.value

    def setProperty(self, label: str, name: str, value: str | float | int) -> None:
        """Set a property as MMCore does: read-only is ignored, a bad value raises.

        A value outside ``allowed`` or outside the limits raises MMCore's
        message (measured). A set on a read-only property raises nothing and
        changes nothing (measured): only the backend refuses it.
        """
        self._record("setProperty", label, name, value)
        self._set_property(label, name, value)

    def _set_property(self, label: str, name: str, value: str | float | int) -> None:
        """``setProperty`` without the log line: ``setConfig`` applies its settings so."""
        device = self._device(label)
        prop = self._property(label, name)
        if prop.read_only:
            return
        text = str(value)
        # Raised where it is built, never held in a local: the exception's
        # traceback reaches this frame, a cycle through the core (#148, FM-92).
        if (prop.allowed and text not in prop.allowed) or _outside(prop, text):
            raise _refused(name, text)
        try:
            if device.type == DeviceType.State and name == "State":
                self._set_position(device, int(float(text)))
            elif device.type == DeviceType.State and name == "Label":
                self._set_position(device, device.state_labels.index(text))
            elif device.type == DeviceType.Camera and name == "Exposure":
                prop.value = f"{float(text):.4f}"
            else:
                prop.value = text
        except ValueError:
            # Not a number, or not one of the device's labels.
            raise _refused(name, text) from None

    def isPropertyReadOnly(self, label: str, name: str) -> bool:
        return self._property(label, name).read_only

    def hasPropertyLimits(self, label: str, name: str) -> bool:
        prop = self._property(label, name)
        return prop.lower is not None and prop.upper is not None

    def getPropertyLowerLimit(self, label: str, name: str) -> float:
        """The lower limit; 0.0 for a property without limits (measured)."""
        lower = self._property(label, name).lower
        return 0.0 if lower is None else lower

    def getPropertyUpperLimit(self, label: str, name: str) -> float:
        """The upper limit; 0.0 for a property without limits (assumed, as the lower)."""
        upper = self._property(label, name).upper
        return 0.0 if upper is None else upper

    def getAllowedPropertyValues(self, label: str, name: str) -> tuple[str, ...]:
        """The allowed values; ``()`` for an unknown property, which does not raise (measured)."""
        found = self._device(label).properties.get(name)
        return () if found is None else found.allowed

    # --- State devices -------------------------------------------------------

    def getState(self, label: str) -> int:
        device = self._typed(label, DeviceType.State)
        return int(device.properties["State"].value)

    def setState(self, label: str, position: int) -> None:
        self._record("setState", label, position)
        self._set_position(self._typed(label, DeviceType.State), int(position))

    def getStateLabel(self, label: str) -> str:
        return self._typed(label, DeviceType.State).properties["Label"].value

    def getStateLabels(self, label: str) -> tuple[str, ...]:
        return self._typed(label, DeviceType.State).state_labels

    def setStateLabel(self, label: str, text: str) -> None:
        """Move a State device to the position named ``text``; an unknown one raises (measured)."""
        self._record("setStateLabel", label, text)
        device = self._typed(label, DeviceType.State)
        if text not in device.state_labels:
            raise RuntimeError(
                f'Error in device "{label}": Label not defined (10).  Available '
                f"Labels: {device.state_labels!r}"
            )
        self._set_position(device, device.state_labels.index(text))

    def _set_position(self, device: FakeDevice, position: int) -> None:
        """Move a State device: ``State`` and ``Label`` change together (one store)."""
        if not 0 <= position < len(device.state_labels):
            # The demo's message for a position out of range (measured).
            raise RuntimeError(
                f'Error in device "{device.label}": (Error message unavailable) (103)'
            )
        device.properties["State"].value = str(position)
        device.properties["Label"].value = device.state_labels[position]

    # --- continuous focus ----------------------------------------------------

    def _autofocus(self) -> str:
        """The AutoFocus slot's device, after raising its ``failing`` entry; ``""`` if empty."""
        label = self._slots["autofocus"]
        if label:
            self._device(label)
        return label

    def enableContinuousFocus(self, enable: bool) -> None:
        """Switch the slot's autofocus; with the slot empty, only switching on raises (measured)."""
        self._record("enableContinuousFocus", enable)
        label = self._autofocus()
        if not label:
            if enable:
                raise RuntimeError(_NO_AUTOFOCUS)
            return
        self._continuous[label] = bool(enable)

    def isContinuousFocusEnabled(self) -> bool:
        """The slot's device; ``False`` with the slot empty (measured)."""
        label = self._autofocus()
        return bool(label) and self._continuous[label]

    def isContinuousFocusLocked(self) -> bool:
        """Locked whenever enabled, unless ``autofocus_lockable`` is ``False``.

        As on the demo, a lock comes at once; a real one takes time (assumed).
        """
        return self.isContinuousFocusEnabled() and self.autofocus_lockable

    def getAutoFocusOffset(self) -> float:
        """The slot's device's offset; with the slot empty, MMCore's error (measured)."""
        label = self._autofocus()
        if not label:
            raise RuntimeError(_NO_AUTOFOCUS_OFFSET)
        return self._af_offset[label]

    def setAutoFocusOffset(self, offset: float) -> None:
        """Store the slot's device's offset; it moves nothing, as on the demo (measured)."""
        self._record("setAutoFocusOffset", offset)
        label = self._autofocus()
        if not label:
            raise RuntimeError(_NO_AUTOFOCUS_OFFSET)
        self._af_offset[label] = float(offset)

    # --- config groups -------------------------------------------------------

    def getAvailableConfigGroups(self) -> tuple[str, ...]:
        return tuple(sorted(self.config_groups))

    def getAvailableConfigs(self, group: str) -> tuple[str, ...]:
        """The group's presets, sorted; ``()`` for a group that does not exist (measured)."""
        return tuple(sorted(self.config_groups.get(group, {})))

    def getConfigData(self, group: str, name: str) -> tuple[tuple[str, str, str], ...]:
        """The preset's ``(device, property, value)`` settings, in the order applied."""
        settings = self.config_groups.get(group, {}).get(name)
        if settings is None:
            raise ValueError(
                f'Configuration group "{group}" or its preset "{name}" does not exist'
            )
        return tuple(settings)

    def defineConfig(
        self,
        group: str,
        name: str | None = None,
        device: str | None = None,
        prop: str | None = None,
        value: str | float | None = None,
    ) -> None:
        """Merge one setting into a preset, creating the group and the preset (measured).

        A setting of the same device and property is changed where it is, any
        other is appended, and nothing checks that the device is loaded.
        """
        self._record("defineConfig", group, name, device, prop, value)
        presets = self.config_groups.setdefault(group, {})
        if name is None:
            return
        settings = presets.setdefault(name, [])
        if device is None or prop is None:
            return
        text = "" if value is None else str(value)
        for index, (dev, name_, _) in enumerate(settings):
            if (dev, name_) == (device, prop):
                settings[index] = (device, prop, text)
                return
        settings.append((device, prop, text))

    def deleteConfig(self, group: str, name: str) -> None:
        """Remove one preset; the group stays even once it is empty (measured).

        A missing preset raises the message of a missing group, which does
        not name the preset (measured).
        """
        self._record("deleteConfig", group, name)
        presets = self.config_groups.get(group)
        if presets is None or name not in presets:
            raise ValueError(f'Configuration group "{group}" does not exist')
        del presets[name]

    def setConfig(self, group: str, name: str) -> None:
        """Apply a preset's settings in order; a device that is not loaded raises there."""
        self._record("setConfig", group, name)
        settings = self.config_groups.get(group, {}).get(name)
        if settings is None:
            raise ValueError(
                f'Preset "{name}" of configuration group "{group}" does not exist'
            )
        for device, prop, value in list(settings):
            if device == _CORE_LABEL:
                self._set_core_property(prop, value)
            else:
                self._set_property(device, prop, value)

    def _set_core_property(self, prop: str, value: str) -> None:
        """A ``Core,<prop>`` setting: a slot, the auto-shutter or the channel group."""
        slots = {core_name: slot for slot, (core_name, _) in _SLOTS.items()}
        if prop in slots:
            self._set_slot(slots[prop], value)
        elif prop == "AutoShutter":
            self._auto_shutter = value not in ("0", "")
        elif prop == "ChannelGroup":
            self._set_channel_group(value)
        else:
            raise RuntimeError(
                f'Error in device "{_CORE_LABEL}": Invalid property name '
                f"encountered: {prop} (2)"
            )

    def getChannelGroup(self) -> str:
        return self._channel_group

    def setChannelGroup(self, group: str) -> None:
        self._record("setChannelGroup", group)
        self._set_channel_group(group)

    def _set_channel_group(self, group: str) -> None:
        if group and group not in self.config_groups:
            raise ValueError(
                f'Cannot set Core property ChannelGroup to invalid value "{group}"'
            )
        self._channel_group = group

    # --- unloading -----------------------------------------------------------

    def unloadDevice(self, label: str) -> None:
        """Remove one device and clear the slots that named it (measured on the demo)."""
        self._record("unloadDevice", label)
        self._device(label)
        self._forget(label)

    def unloadAllDevices(self) -> None:
        """Remove every device; calling it twice is harmless (measured)."""
        self._record("unloadAllDevices")
        for label in list(self._devices):
            self._forget(label)

    def _forget(self, label: str) -> None:
        self._devices.pop(label, None)
        self._xy.pop(label, None)
        self._z.pop(label, None)
        self._open.pop(label, None)
        self._continuous.pop(label, None)
        self._af_offset.pop(label, None)
        for slot, named in self._slots.items():
            if named == label:
                self._slots[slot] = ""


def _complete(device: FakeDevice) -> None:
    """Add the properties MMCore's dedicated calls read, when a device lacks them."""
    props = device.properties
    if device.type == DeviceType.Camera:
        props.setdefault("Exposure", FakeProperty("10.0000"))
    if device.type == DeviceType.State:
        labels = device.state_labels
        props.setdefault("State", FakeProperty("0"))
        props.setdefault(
            "Label", FakeProperty(labels[0] if labels else "", allowed=labels)
        )


def _dtype(bit_depth: int) -> type[np.unsignedinteger]:
    """The narrowest unsigned type that holds ``bit_depth`` bits, as MMCore returns."""
    if bit_depth <= 8:
        return np.uint8
    if bit_depth <= 16:
        return np.uint16
    return np.uint32


def _refused(name: str, text: str) -> RuntimeError:
    """MMCore's error for a value a property does not take (measured)."""
    return RuntimeError(f'Cannot set property "{name}" to "{text}"')


def _outside(prop: FakeProperty, text: str) -> bool:
    """Whether a numeric ``text`` lies outside the property's limits."""
    if prop.lower is None or prop.upper is None:
        return False
    try:
        number = float(text)
    except ValueError:
        return False
    return not prop.lower <= number <= prop.upper


def _numbered(start: int, stop: int) -> tuple[str, ...]:
    """The demo's placeholder labels ``State-<n>`` for positions without a name."""
    return tuple(f"State-{n}" for n in range(start, stop))


def _demo(
    label: str,
    kind: DeviceType,
    name: str,
    description: str,
    driver: str,
    *,
    library: str = "DemoCamera",
    extra: dict[str, FakeProperty] | None = None,
    labels: tuple[str, ...] = (),
    position: int = 0,
) -> FakeDevice:
    """A demo device: its read-only ``Name`` and ``Description``, then ``extra``.

    ``driver`` is the ``Description`` property, which the demo words
    differently from the device description.
    """
    properties = {
        "Name": FakeProperty(name, read_only=True),
        "Description": FakeProperty(driver, read_only=True),
        **(extra or {}),
    }
    if kind == DeviceType.State:
        properties["State"] = FakeProperty(str(position))
        properties["Label"] = FakeProperty(labels[position], allowed=labels)
    return FakeDevice(label, kind, library, name, description, properties, labels)


def _demo_config_groups() -> dict[str, dict[str, Settings]]:
    """The demo configuration's six groups, as ``getConfigData`` read them (2026-10-19)."""

    def channel(dichroic: str, emission: str, excitation: str) -> Settings:
        return [
            ("Dichroic", "Label", dichroic),
            ("Emission", "Label", emission),
            ("Excitation", "Label", excitation),
            (_CORE_LABEL, "Shutter", "White Light Shutter"),
        ]

    def multiband(led: str) -> Settings:
        return [
            ("Dichroic", "Label", "89402bs"),
            ("Emission", "Label", "89402m"),
            ("Excitation", "Label", "Empty"),
            ("LED", "Label", led),
            (_CORE_LABEL, "Shutter", "LED Shutter"),
        ]

    def camera(binning: str, bit_depth: str) -> Settings:
        return [("Camera", "Binning", binning), ("Camera", "BitDepth", bit_depth)]

    return {
        "Camera": {
            "HighRes": camera("1", "12"),
            "LowRes": camera("4", "8"),
            "MedRes": camera("2", "10"),
        },
        "Channel": {
            "Cy5": channel("400DCLP", "Chroma-HQ700", "Chroma-HQ570"),
            "DAPI": channel("400DCLP", "Chroma-HQ620", "Chroma-D360"),
            "FITC": channel("Q505LP", "Chroma-HQ535", "Chroma-HQ480"),
            "Rhodamine": channel("Q585LP", "Chroma-HQ700", "Chroma-HQ570"),
        },
        "Channel-Multiband": {
            "Cy5": multiband("635nm"),
            "DAPI": multiband("385nm"),
            "FITC": multiband("470nm"),
            "Rhodamine": multiband("550nm"),
        },
        "LightPath": {
            "Camera-left": [("Path", "State", "1")],
            "Camera-right": [("Path", "State", "2")],
            "Eyepiece": [("Path", "State", "0")],
        },
        "Objective": {
            "10X": [("Objective", "State", "1")],
            "20X": [("Objective", "State", "3")],
            "40X": [("Objective", "State", "0")],
        },
        "System": {
            "Startup": [
                ("Camera", "BitDepth", "16"),
                ("Camera", "ScanMode", "1"),
                ("Objective", "Label", "Nikon 10X S Fluor"),
                ("Camera", "Binning", "1"),
                ("LED Shutter", "State Device", "LED"),
                (_CORE_LABEL, "ChannelGroup", "Channel"),
            ]
        },
    }


def _demo_wheel(label: str, labels: tuple[str, ...]) -> FakeDevice:
    return _demo(
        label,
        DeviceType.State,
        "DWheel",
        "Demo filter wheel",
        "Demo filter wheel driver",
        labels=labels,
    )
