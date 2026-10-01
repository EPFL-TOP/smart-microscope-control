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
    * ``failing``: label -> exception raised by every call whose first
      argument is that label; for the camera slot's device, by the
      current-camera calls too.
    * ``pixel_size_um``, ``image_shape`` (height, width), ``bit_depth``: what
      the camera calls report.
    * ``quirk_z_move_disables_autofocus``: the Nikon quirk (FM-12).
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
        self.failing: dict[str, BaseException] = {}
        self.pixel_size_um = 1.0
        self.image_shape: tuple[int, int] = (512, 512)
        self.bit_depth = 16
        self.quirk_z_move_disables_autofocus = True

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
        self._continuous_focus = False
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
    def demo_like(cls) -> FakeCore:
        """The demo configuration's devices, slots, State labels and camera.

        Copied from the demo as measured on 2026-10-01; a demo test keeps
        them equal (``tests/test_fake_core_demo.py``), and where the two
        disagree the fake is wrong. One difference is deliberate: the demo's
        White Light Shutter starts open on some opens (FM-47), and the fake's
        always starts closed.
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
        return cls(
            devices,
            camera="Camera",
            xy_stage="XY",
            focus="Z",
            autofocus="Autofocus",
            shutter="White Light Shutter",
        )

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
        if self.quirk_z_move_disables_autofocus and label == self._slots["focus"]:
            self._continuous_focus = False

    def deviceBusy(self, label: str) -> bool:
        """Busy only while the label is in ``busy_devices``: motion is instant."""
        self._device(label)
        return label in self.busy_devices

    def stop(self, label: str) -> None:
        """Accepted for a stage; ``busy_devices`` is left alone, so a stuck device stays stuck."""
        self._record("stop", label)
        device = self._device(label)
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
        device = self._device(label)
        prop = self._property(label, name)
        if prop.read_only:
            return
        text = str(value)
        refused = RuntimeError(f'Cannot set property "{name}" to "{text}"')
        if (prop.allowed and text not in prop.allowed) or _outside(prop, text):
            raise refused
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
            raise refused from None

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

    def enableContinuousFocus(self, enable: bool) -> None:
        self._record("enableContinuousFocus", enable)
        self._continuous_focus = bool(enable)

    def isContinuousFocusEnabled(self) -> bool:
        return self._continuous_focus

    def isContinuousFocusLocked(self) -> bool:
        """Locked whenever enabled (as on the demo; a real lock takes time, assumed instant)."""
        return self._continuous_focus

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


def _demo_wheel(label: str, labels: tuple[str, ...]) -> FakeDevice:
    return _demo(
        label,
        DeviceType.State,
        "DWheel",
        "Demo filter wheel",
        "Demo filter wheel driver",
        labels=labels,
    )
