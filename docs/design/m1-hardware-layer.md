# M1 design — the hardware layer, proven on the simulator

- **Status**: design for issues #5, #6, #7, #8, #9, #10, #11, #30 and #54
  (§13, added 2026-09-23, revised 2026-09-24; §9 revised 2026-09-30;
  §8 revised 2026-10-01)
- **Owner**: the design session. **Executors**: `/develop` sessions, one per issue.
- **Rule**: this document is the contract between issues that are built in
  parallel. Names, module paths and signatures below are fixed; an executor
  who needs to change one stops and comments on its issue instead.

Read first: [ADR-0002](../adr/0002-pymmcore-plus-as-the-hardware-core.md),
[ADR-0003](../adr/0003-capabilities-roles-and-profiles.md),
[ADR-0005](../adr/0005-testing-strategy.md).

## 0. Scope and non-goals

M1 delivers, on Micro-Manager's demo devices: five capabilities
(`XYStage`, `ZStage`, `Camera`, `Shutter`, `Properties`) implemented over
`CMMCorePlus`; role resolution; TOML profiles; the `Microscope` facade with
safety guards and dry-run; a contract test suite with a `FakeCore`; a
stage-aware synthetic sample; CLI commands; and a read-only hardware
inventory command (`smc discover`) so the microscope PCs can be surveyed
now.

Not in M1: `Autofocus`, `ObjectiveTurret`, `LightSource`, `Channels`
(M2 — their semantics need Nikon/Zeiss quirks the demo cannot show), the
plugin API (M3), any UI beyond the CLI, unicore Python devices (M4).

## 1. Module map

```
src/smc/
  hardware/
    __init__.py
    core.py            (exists) open/close a core; MM install status
    errors.py          HardwareError hierarchy                        (seeded by the design PR)
    capabilities.py    value types + Protocols                          #5
    roles.py           Role, DeviceInfo (seeded) · RoleMap, resolve(), core helpers   #6
    profile.py         Profile (pydantic) + TOML loading                #7
    safety.py          Safety checks + Executor (lock, dry-run)         #5
    microscope.py      Microscope facade + capability registry          #8
    backends/
      __init__.py
      mm.py            MM* capability classes, loaded_devices, core_roles  #5
  discovery/
    __init__.py        inventory() — OS sections, hints, then the MM child  #30
    models.py          Inventory (pydantic)                              #30
    os_inventory.py    serial / USB-PnP / PCI, per OS                    #30
    mm_inventory.py    system section, adapter listing, optional probe   #30
    _mm_child.py       child process that loads adapters (crash-isolated) #30
    vendors.py         VID/PID and name → vendor → likely adapters       #30
    report.py          text rendering + UTF-8 files for --out            #30
  testing/
    __init__.py
    fakes.py           FakeCore                                          #9
    fixtures.py        pytest plugin: demo_core, demo_microscope, fake_*  #9
    synthetic.py       PlateSample renderer + SampleCamera               #10
  cli.py               (exists) + profiles/devices/stage/z/snap/discover  #11 #30
tests/
  conftest.py          loads smc.testing.fixtures (which adds --profile) and pytester
  contracts/           one file per capability, parametrised backends     #9
  unit/                pure logic (roles, profile, safety, synthetic)
  test_*.py            existing simulator/CLI tests
profiles/demo.toml     example profile the loader must accept             #7
```

## 2. Conventions

- **Units in names.** `_um`, `_ms`, `_s`, `_deg`, `_px`. Stage coordinates
  are Micro-Manager's: µm, X right, Y up, Z up.
- **Methods, not properties**, for anything that talks to hardware
  (`exposure_ms()` / `set_exposure_ms()`), because a hardware read can be
  slow or fail and a property hides that.
- **Mutating calls return the readback** (`move_to_um` returns the position
  read after the move). In dry-run they return the *commanded* value.
- **Errors** live in `smc.hardware.errors`:

  ```python
  class HardwareError(RuntimeError): ...


  class CoreError(HardwareError): ...  # re-exported from core.py


  class CapabilityMissingError(HardwareError):
      capability: str
      role: Role | None
      available: tuple[str, ...]


  class SafetyRefusedError(HardwareError):
      reason: str
      how_to_force: str  # "" when it cannot be forced


  class DeviceTimeoutError(HardwareError): ...


  # §13 (#54)
  class MotionInProgressError(SafetyRefusedError):
      moving: tuple[str, ...]  # "xy_stage XY"


  class MicroscopeHaltedError(HardwareError): ...  # not a SafetyRefusedError (§13)


  class MotionStoppedError(HardwareError):
      device: str


  class MicroscopeBusyError(HardwareError):
      holder: str  # the description of the call holding the lock
      held_s: float
  ```

  Messages say what to do next (`how_to_force`, the CLI command, the
  profile key), never just what went wrong.
- **Logging**: `logging.getLogger("smc.hardware.<module>")`. Every mutating
  call logs at INFO: `xy_stage: move_to (1234.0, -56.0) µm`; dry-run logs
  `[dry-run] xy_stage: move_to …`. No `print` outside `cli.py`.
- **Threading**: a `Microscope` owns one `threading.RLock`; every backend
  call goes through `Executor`. An *action* (a mutation, `snap`)
  holds the lock from its first check to its readback, **including the wait
  of a move**. Reads, `stop()` and closing a shutter never take it (§13).
  No action runs while a movement has not finished. Capabilities are
  therefore safe to call from a UI thread and a worker at once; actions are
  still *sequential*.
- **Timeouts**: `core.setTimeoutMs(profile.micromanager.device_timeout_ms)`
  (default 60 000 — a plate traverse exceeds MMCore's 5 s default).
  `waitForDevice` errors surface as `DeviceTimeoutError`.
- **Typing**: `mypy --strict`; Protocols are `@runtime_checkable`; value
  types are frozen dataclasses (`slots=True`); configuration is pydantic v2.
- Python 3.10 compatible: no `StrEnum` (use `class Role(str, Enum)`), no
  `match` needed, `tomllib` behind `sys.version_info` with `tomli` fallback.

## 3. Capabilities — `smc/hardware/capabilities.py` (#5)

```python
@dataclass(frozen=True, slots=True)
class XY:
    x_um: float
    y_um: float


@dataclass(frozen=True, slots=True)
class Limits:
    low_um: float
    high_um: float


@dataclass(frozen=True, slots=True)
class PropertyInfo:
    device: str
    name: str
    value: str
    read_only: bool = False
    allowed: tuple[str, ...] = ()
    lower: float | None = None
    upper: float | None = None
    # helpers: .numeric (both limits set), .number (float(value) or None), .describe()


@runtime_checkable
class XYStage(Protocol):
    def position_um(self) -> XY: ...
    def move_to_um(self, x_um: float, y_um: float) -> XY: ...
    def move_by_um(self, dx_um: float, dy_um: float, *, force: bool = False) -> XY: ...
    def wait(self, timeout_s: float | None = None) -> None: ...
    def is_busy(self) -> bool: ...
    def stop(self) -> None: ...  # §13 (#54)
    def limits_um(self) -> tuple[Limits, Limits] | None: ...  # (x, y); None = unknown


@runtime_checkable
class ZStage(Protocol):
    def position_um(self) -> float: ...
    def move_to_um(self, z_um: float) -> float: ...
    def move_by_um(self, dz_um: float) -> float: ...
    def wait(self, timeout_s: float | None = None) -> None: ...
    def is_busy(self) -> bool: ...
    def stop(self) -> None: ...  # §13 (#54)
    def limits_um(self) -> Limits | None: ...


@runtime_checkable
class Camera(Protocol):
    def snap(self) -> np.ndarray: ...  # 2-D, native dtype
    def exposure_ms(self) -> float: ...
    def set_exposure_ms(self, value_ms: float) -> float: ...
    def image_shape(self) -> tuple[int, int]: ...  # (height, width)
    def bit_depth(self) -> int: ...
    def pixel_size_um(self) -> float: ...  # 0.0 = unknown; never guessed


@runtime_checkable
class Shutter(Protocol):
    def is_open(self) -> bool: ...
    def set_open(self, open_: bool) -> bool: ...
    def auto_shutter(self) -> bool: ...
    def set_auto_shutter(self, on: bool) -> bool: ...


@runtime_checkable
class Properties(Protocol):
    def devices(self) -> list[str]: ...  # without "Core"
    def describe(self, device: str) -> list[PropertyInfo]: ...
    def get(self, device: str, name: str) -> str: ...
    def set(self, device: str, name: str, value: str | float | int) -> str: ...
```

Semantics every implementation must honour (these are the contract tests):

- `move_by_um` on `XYStage` refuses `max(|dx|, |dy|) > safety.max_jog_um`
  with `SafetyRefusedError` unless `force=True`. Absolute moves are never
  jog-guarded (crossing a plate is legitimate travel).
- Absolute and relative moves refuse a target outside the profile's soft
  limits (`SafetyRefusedError`, not forceable). No limits configured → no check.
- A move returns when the device has arrived (there is no `wait=False` in
  M1, §13). `wait(timeout_s)` is a lock-free poll: it returns when the
  device reports not busy, raises `DeviceTimeoutError` after `timeout_s`
  (default: the core timeout), and never stops anything. To ask "is it done
  yet", use `is_busy()`.
- While a device the layer moved is still moving, every mutation (except
  closing a shutter) and every `snap()` raises `MotionInProgressError`;
  reads and `stop()` are allowed (§13).
- `stop()` sends the device's stop at once, without waiting for the
  microscope lock, even in dry-run (§13).
- `Camera.pixel_size_um()` returns MMCore's value when > 0, else the
  profile's value for the current objective label, else `0.0`.
- Every reader tolerates a device that answers slowly but never swallows a
  failure silently: exceptions propagate; the *facade's* `state()` is the
  tolerant layer.

M2 will extend `ZStage.move_to_um` with `keep_autofocus: bool = True`
(a keyword with a default is a compatible extension).

## 4. Roles — `smc/hardware/roles.py` (#6)

```python
class Role(str, Enum):
    camera = "camera"
    xy_stage = "xy_stage"
    focus = "focus"
    autofocus = "autofocus"
    autofocus_offset = "autofocus_offset"
    objective_turret = "objective_turret"
    shutter = "shutter"
    light_source = "light_source"
    light_path = "light_path"
    filter_turret = "filter_turret"


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    label: str
    type: str  # DeviceType name without "Device": "XYStage", "Stage", "Camera",
    # "State", "Shutter", "AutoFocus", "Hub", "Generic", …
    library: str = ""
    name: str = ""
    description: str = ""


Source = Literal["profile", "core", "heuristic"]


@dataclass
class RoleMap:
    assigned: dict[Role, str]
    sources: dict[Role, Source]
    candidates: dict[Role, list[str]]  # best first; includes the assigned one
    warnings: list[str]

    def get(self, role: Role) -> str | None: ...
    def missing(self) -> list[Role]: ...
    def ambiguous(self) -> dict[Role, list[str]]: ...  # roles with > 1 candidate
    def describe(self) -> list[str]: ...  # one aligned line per role


def devices_from_core(core) -> list[DeviceInfo]     # skips "Core"; type = DeviceType(...).name without "Device"
def core_roles(core) -> dict[Role, str]              # camera, xy_stage, focus, autofocus, shutter — non-empty only

def resolve(
    devices: Iterable[DeviceInfo],
    *,
    core_roles: Mapping[Role, str] | None = None,  # from MMCore's own slots
    overrides: Mapping[Role, str] | None = None,  # profile [roles.assign]
    exclusions: Mapping[Role, Sequence[str]] | None = None,  # profile [roles.exclude]
) -> RoleMap: ...
```

Algorithm (pure; no core import):

1. **Candidates** per role from the type table, minus devices whose
   normalised name (`[a-z0-9]` only) contains a built-in or profile
   exclusion substring; ranked by the first matching name hint, then label.
2. **Assignment**, first source that yields a device:
   1. `overrides[role]` — if the label is not loaded, a warning and fall
      through.
   2. `core_roles[role]` — if excluded for that role (e.g. a `.cfg` that
      still names a TIRF positioner as the XY stage), a warning and fall
      through — this is the case that reads like dead hardware.
   3. Best heuristic candidate.
3. Record `sources`, `candidates`, `warnings`. A role with several
   candidates is not an error, but `describe()` shows the runner-ups.

Type table and built-in rules (generalised from `nikon-control/scope/stand.py`):

| Role | Type | Must contain | Exclude | Hints (rank order) |
|---|---|---|---|---|
| camera | Camera | | | |
| xy_stage | XYStage | | tirf | xystage, xydrive, stage |
| focus | Stage | | pfs, offset, tirf | zdrive, focus, z |
| autofocus | AutoFocus | | | |
| autofocus_offset | Stage | pfs \| offset | | pfsoffset, offset |
| objective_turret | State | nose \| objective \| turret | filter, condenser, reflector | nosepiece, objective, turret |
| shutter | Shutter | | | epishutter, epi, diashutter, dia, tl |
| light_source | Shutter \| State \| Generic | lamp \| led \| light \| laser \| colibri | shutter-only names | lamp, led, laser |
| light_path | State | lightpath \| sideport \| port \| eyepiece | | lightpath, sideport |
| filter_turret | State | filter \| reflector | objective, nose | filter, reflector |

Unit tests use synthetic `DeviceInfo` lists copied from
`docs/hardware/inventory.md`: the Ti2's four `XYStage`-typed devices, the
Ti-E's three `Stage`-typed devices, the demo configuration.

## 5. Profiles — `smc/hardware/profile.py` (#7)

TOML shape (this replaces the draft `profiles/demo.toml`):

```toml
[microscope]
name = "demo"
vendor = "Micro-Manager"
description = "DemoCamera adapter: simulated camera, XY, Z, turret, shutter, autofocus."

[micromanager]
config = ""                    # "" = demo configuration; relative paths resolve against this file
device_timeout_ms = 60000
adapter_search_paths = []      # extra Micro-Manager directories, e.g. a separate MMStudio install

[roles.assign]                 # overrides only; keys are Role values
# xy_stage = "XY"

[roles.exclude]                # extra name substrings per role
# focus = ["piezo"]

[safety]
max_jog_um = 5000.0
# z_soft_limits_um = [-1000.0, 1000.0]
# xy_soft_limits_um = [[-60000.0, 60000.0], [-40000.0, 40000.0]]
turret_requires_confirm = true

[camera]
pixel_size_um = { "Nikon 10X S Fluor" = 0.65, "Nikon 40X Plan Fluor ELWD" = 0.1625 }

[quirks]                       # free-form, consumed by backends; documented per key in M2
```

Model:

```python
class MicroscopeSection(BaseModel):
    name: str
    vendor: str = ""
    description: str = ""


class MicroManagerSection(BaseModel):
    config: str = ""
    device_timeout_ms: int = 60_000
    adapter_search_paths: list[str] = []


class RolesSection(BaseModel):
    assign: dict[Role, str] = {}
    exclude: dict[Role, list[str]] = {}


class SafetySection(BaseModel):
    max_jog_um: float = 5000.0
    z_soft_limits_um: tuple[float, float] | None = None
    xy_soft_limits_um: tuple[tuple[float, float], tuple[float, float]] | None = None
    turret_requires_confirm: bool = True


class CameraSection(BaseModel):
    pixel_size_um: dict[str, float] = {}


class Profile(BaseModel):
    microscope: MicroscopeSection
    micromanager: MicroManagerSection = MicroManagerSection()
    roles: RolesSection = RolesSection()
    safety: SafetySection = SafetySection()
    camera: CameraSection = CameraSection()
    quirks: dict[str, Any] = {}
    source: Path | None = Field(default=None, exclude=True)  # where it was loaded from

    @classmethod
    def demo(cls) -> Profile: ...  # built in code; no file needed
    @classmethod
    def load(cls, source: str | Path) -> Profile: ...
    def config_path(self) -> Path | None: ...  # None for the demo configuration


def search_paths() -> list[
    Path
]: ...  # $SMC_PROFILES (os.pathsep-separated) → ./profiles → .
def list_profiles() -> list[tuple[str, Path]]: ...
```

Rules: `load("demo")` returns `Profile.demo()` unless a `demo.toml` is
found first; a bare name resolves to `<dir>/<name>.toml` over the search
paths; a path is read as given. Validation errors are re-raised as
`ProfileError(HardwareError)` naming file, key and fix. Limits must be
ordered; unknown role keys list the allowed values; a non-empty `config`
must exist at load time.

## 6. Micro-Manager backend — `smc/hardware/backends/mm.py` (#5)

The core inventory helpers (`devices_from_core`, `core_roles`) live in
`roles.py` (#6); this module holds only the capability implementations.

```python
class MMXYStage:  # implements XYStage
    def __init__(self, core, label: str, executor: Executor, safety: Safety): ...


class MMZStage:  # implements ZStage
    def __init__(self, core, label: str, executor: Executor, safety: Safety): ...


class MMCamera:  # implements Camera
    def __init__(
        self,
        core,
        label: str,
        executor: Executor,
        pixel_sizes_um: Mapping[str, float],
        objective_label: Callable[[], str | None],
    ): ...


class MMShutter:  # implements Shutter
    def __init__(self, core, label: str, executor: Executor): ...


class MMProperties:  # implements Properties
    def __init__(self, core, executor: Executor): ...
```

Core calls per method:

| Method | MMCore |
|---|---|
| `XYStage.position_um` | `getXPosition(label)`, `getYPosition(label)` |
| `XYStage.move_to_um` | `setXYPosition(label, x, y)`, then the wait, then the readback: one action under the lock (§13) |
| `ZStage.position_um` / `move_to_um` | `getPosition(label)` / `setPosition(label, z)`, waited like XY |
| `wait` / `is_busy` | polls `deviceBusy(label)`, lock-free, never stops / `deviceBusy(label)`, lock-free |
| `XYStage.stop` / `ZStage.stop` | `stop(label)`, through `Executor.stop` (no lock, §13) |
| `Camera.snap` | `snapImage()` then `getImage()` (the core's current camera must be `label`); an action under the lock (§13) |
| `Camera.exposure_ms` / `set_exposure_ms` | `getExposure()` / `setExposure(ms)` |
| `Camera.image_shape` / `bit_depth` | `getImageHeight()`, `getImageWidth()` / `getImageBitDepth()` |
| `Camera.pixel_size_um` | `getPixelSizeUm()`, fallback profile map by `objective_label()` |
| `Shutter.*` | `getShutterOpen(label)`, `setShutterOpen(label, b)` (closing is a safe call, §13), `getAutoShutter()`, `setAutoShutter(b)` |
| `Properties.*` | `getLoadedDevices`, `getDevicePropertyNames`, `getProperty`, `setProperty`, `isPropertyReadOnly`, `hasPropertyLimits`, `getPropertyLowerLimit/UpperLimit`, `getAllowedPropertyValues` |

Mutations go through `executor.do(description, action, dry_result=…)`
(moves and moving `Properties.set` pass `motion=`, §13); reads through
`executor.read(action)`, which takes no lock. The internal `Executor`
methods may change shape in #59's fix round; §13 fixes their behaviour. `objective_label` is a callable
supplied by the facade (`getStateLabel` of the `objective_turret` role, or
`lambda: None`).

## 7. Safety and the facade — `safety.py` (#5), `microscope.py` (#8)

```python
class Safety:
    # plain arguments, so #5 does not depend on the profile model (#7); the
    # facade (#8) builds it from profile.safety
    def __init__(self, *, max_jog_um: float, z_soft_limits_um: tuple[float, float] | None = None,
                 xy_soft_limits_um: tuple[tuple[float, float], tuple[float, float]] | None = None): ...
    def check_jog_um(self, dx_um: float, dy_um: float, *, force: bool) -> None
    def check_xy_target_um(self, x_um: float, y_um: float) -> None
    def check_z_target_um(self, z_um: float) -> None

class Executor:
    def __init__(self, *, dry_run: bool, lock: threading.RLock, logger: logging.Logger,
                 lock_timeout_s: float = 60.0): ...
    def halt(self) -> None; def resume(self) -> None
    def hold_log(self, run: Callable[[], T]) -> T   # §13: several stops, then their lines
    halted: bool
    def moving(self) -> tuple[str, ...]
    dry_run: bool
    # internal (actions, reads, safe calls, stops): shape owned by #59, behaviour fixed by §13
```

`Executor.do` acquires the lock, logs `description` (prefixed `[dry-run]`
when dry), calls `action()` unless dry-run, and returns its result or
`dry_result`. Reads always reach the hardware, so a dry-run session shows
real positions and never moves. Motion, stop, halt and the lock timeout:
§13.

```python
CAPABILITY_ROLES: dict[type, tuple[Role, ...]] = {
    XYStage: (Role.xy_stage,), ZStage: (Role.focus,), Camera: (Role.camera,),
    Shutter: (Role.shutter,), Properties: (),
}

class Microscope:
    profile: Profile
    roles: RoleMap
    devices: list[DeviceInfo]
    dry_run: bool
    core: CMMCorePlus          # escape hatch; using it from a plugin is a review failure

    @classmethod
    def open(cls, profile: Profile | str | Path = "demo", *, dry_run: bool = False) -> Microscope
    @classmethod
    def from_core(cls, core, profile: Profile, *, dry_run: bool = False) -> Microscope   # tests, FakeCore
    def close(self) -> None
    def __enter__ / __exit__

    def has(self, capability: type[T]) -> bool
    def get(self, capability: type[T]) -> T | None
    def require(self, capability: type[T]) -> T          # CapabilityMissingError
    def available(self) -> list[type]
    def override(self, capability: type[T], implementation: T) -> None   # testing hook (synthetic camera)

    def state(self) -> MicroscopeState
    def describe(self) -> str

    def stop(self) -> None      # §13: stop every stage, then halt
    def resume(self) -> None    # §13: clear the halt

@dataclass
class MicroscopeState:
    xy: XY | None = None; z_um: float | None = None
    exposure_ms: float | None = None; image_shape: tuple[int, int] | None = None
    pixel_size_um: float | None = None; shutter_open: bool | None = None
    moving: tuple[str, ...] = ()   # Executor.moving(), §13
    halted: bool = False
    roles: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)   # "z: <exception>" per failed read
```

`open()` = `Profile.load` → `open_core(config)` → `setTimeoutMs` →
`loaded_devices` → `resolve(devices, core_roles=core_roles(core),
overrides=profile.roles.assign, exclusions=profile.roles.exclude)` → log
each warning. `require()` builds the capability once from
`CAPABILITY_ROLES` and caches it; a missing role raises
`CapabilityMissingError(capability, role, available=roles.assigned)` whose
message lists what *is* available and points at `[roles.assign]`.
`close()` stops the stages if anything is still moving (§13), then unloads
all devices (one connection per stand), and is idempotent. The `Executor` gets
`lock_timeout_s = profile.micromanager.device_timeout_ms / 1000`.

## 8. Testing — `smc/testing/` and `tests/` (#9, #10)

**Revised 2026-10-01** for the plan of #9. These facts were measured on
the demo that day (pymmcore-plus 0.18.1, pymmcore 12.5.0.75.0), and the
fake copies them:
- The demo's `DHub` has no properties, so the first version's case "at
  least one property per device" failed on the demo.
- MMCore ignores a set on a read-only property: it raises nothing and the
  value does not change. Only the backend's own check refuses it.
- An unknown label raises `RuntimeError('No device with label "X"')`. An
  unknown property raises `RuntimeError('Cannot get value of property "X"')`.
- `getDeviceType` returns a `DeviceType` member, and `str()` of it gives
  `"XYStage"`.
- `unloadDevice` clears the core slot that named the device, and calling
  `unloadAllDevices` twice is harmless.
- XY and Z start at 0. The XY stage reads busy right after a move, for
  about 1 s per 10 mm; Z reads idle at once. A fresh core's timeout is
  5000 ms.
- A Z move on the demo leaves continuous focus enabled. The Nikon quirk
  exists only in the fake.

### FakeCore (`smc/testing/fakes.py`)

`FakeCore` is an in-memory stand-in for `CMMCorePlus`. It covers what the
simulator cannot show: a stuck device, a device that fails, a vendor
quirk. It does not subclass `CMMCorePlus`. It implements the calls the
layer makes (§6, `roles._InventoryCore`, and the facade's `setTimeoutMs`,
`getStateLabel` and `unloadAllDevices`), with MMCore's names, argument
order and errors. It lives in the package, not under `tests/`, so that
plugins and M2's quirk tests can use it.

```python
@dataclass(slots=True)
class FakeProperty:
    value: str
    read_only: bool = False
    allowed: tuple[str, ...] = ()
    lower: float | None = None
    upper: float | None = None


@dataclass(slots=True)
class FakeDevice:
    label: str
    type: DeviceType  # pymmcore_plus.DeviceType
    library: str = "FakeCore"
    name: str = ""
    description: str = ""
    properties: dict[str, FakeProperty] = field(default_factory=dict)
    state_labels: tuple[str, ...] = ()  # State devices: one label per position


class FakeCore:
    def __init__(
        self,
        devices: Iterable[FakeDevice] = (),
        *,
        camera: str = "",
        xy_stage: str = "",
        focus: str = "",
        autofocus: str = "",
        shutter: str = "",
    ) -> None: ...  # a slot naming no device: ValueError
    @classmethod
    def demo_like(cls) -> FakeCore: ...

    log: list[str]  # every mutating call, e.g. "setXYPosition('XY', 20.0, -10.0)"
    # called with (x_um, y_um, z_um) at each snap; None gives a zero frame
    frame_source: Callable[[float, float, float], np.ndarray] | None
    busy_devices: set[str]  # labels that read busy until removed; stop() leaves them
    failing: dict[str, BaseException]  # label -> raised by every call on that device
    pixel_size_um: float  # what getPixelSizeUm() returns
    image_shape: tuple[int, int]  # (height, width)
    bit_depth: int
    quirk_z_move_disables_autofocus: bool = True  # FM-12, kept for M2
```

Behaviour. Each point is either the measured MMCore behaviour above or
labelled as assumed:
- **Log.** Every mutating call appends `name(args)`, with its arguments
  as `repr`. The mutating calls are: positions, `stop`, exposure,
  shutter, auto-shutter, properties, state, the slot setters,
  `setTimeoutMs`, continuous focus, `snapImage` and the unloads. Reads
  are not logged.
- **Calls.**
  - Inventory: `getLoadedDevices` (with `"Core"` last, as on the demo),
    `getDeviceType`, `getDeviceLibrary`, `getDeviceName`,
    `getDeviceDescription`.
  - The getter and setter of each of the five slots.
  - `setTimeoutMs` / `getTimeoutMs` (5000 at construction).
  - Stages: `getXPosition`, `getYPosition`, `getXYPosition`,
    `setXYPosition`, `getPosition`, `setPosition`, `deviceBusy`, `stop`.
  - Camera: `snapImage`, `getImage`, `getExposure`, `setExposure`,
    `getImageHeight`, `getImageWidth`, `getImageBitDepth`,
    `getPixelSizeUm`.
  - Shutter: `getShutterOpen`, `setShutterOpen`, `getAutoShutter`,
    `setAutoShutter`.
  - Properties: `getDevicePropertyNames`, `hasProperty`, `getProperty`,
    `setProperty`, `isPropertyReadOnly`, `hasPropertyLimits`,
    `getPropertyLowerLimit`, `getPropertyUpperLimit`,
    `getAllowedPropertyValues`.
  - State devices: `getState`, `setState`, `getStateLabel`,
    `getStateLabels`.
  - Continuous focus: `enableContinuousFocus`,
    `isContinuousFocusEnabled`, `isContinuousFocusLocked` (locked when
    enabled; assumed).
  - `unloadDevice`, `unloadAllDevices`.
- **One store.** The camera's `Exposure` (written `f"{ms:.4f}"`, as the
  demo does) and `Binning`, and a State device's `State` and `Label`, are
  properties. The dedicated calls read and write those same properties,
  so `Properties` and the capabilities agree.
- **Motion is instant.** A set changes the position at once. A device
  reads busy only while it is in `busy_devices`. The demo has real
  motion; the fake has stuck devices.
- **Frames.** `snapImage()` calls `frame_source(x, y, z)` with the
  positions of the devices in the XY and focus slots (0.0 for an empty
  slot) and keeps the frame for `getImage()`. With no source, the frame
  is zeros of `image_shape`: `uint8` up to 8 bits, else `uint16`.
  `getImage()` before any snap raises `RuntimeError`.
- **Errors.**
  - An unknown label or property raises, with the measured message.
  - A set on a read-only property is ignored.
  - A value outside `allowed` raises
    `RuntimeError('Cannot set property "Binning" to "3"')` (measured).
  - A label in `failing` makes every call that names it raise its
    exception. If it is the camera's label, the current-camera calls
    raise it too.
- **The quirk.** With `quirk_z_move_disables_autofocus`, a `setPosition`
  on the focus slot's device disables continuous focus, as moving the Z
  drive does with PFS on the Nikon stands.
- **Unloading** removes the devices and clears the slots that named them.

`FakeCore.demo_like()` mirrors the demo configuration:
- the demo's labels, types, libraries, names and descriptions;
- the five slots;
- the State devices' labels and positions;
- `pixel_size_um = 1.0`;
- a 512 × 512, 16-bit camera with an exposure of 10 ms and binning 1;
- every position at 0, the shutter closed and auto-shutter on. The demo's
  White Light Shutter starts open on some opens (FM-47); the fake always
  starts closed.

Its properties are a subset of the demo's:
- `Name` and `Description` (read-only) wherever the demo has them;
- the camera's `Binning` (`1 2 4 8`), `Exposure` (0–10 000), `PixelType`,
  and `CameraName` and `CameraID` (read-only);
- each State device's `State` and `Label`.

A demo test keeps the inventory and the slots identical. It also checks
that every property the fake declares exists on the demo, with the same
read-only flag. Where the fake and the demo disagree, the fake is wrong.

### Fixtures (`smc/testing/fixtures.py`, a pytest plugin)

`tests/conftest.py` loads the plugin with
`pytest_plugins = ["smc.testing.fixtures", "pytester"]`, and a plugin's
repository loads it the same way. The plugin imports `pytest`, so
`smc.testing/__init__.py` exports only the fake. `import smc.testing`
then works without the dev extra.

| Fixture | Scope | What it gives |
|---|---|---|
| `mm_available` | session | Whether the adapters are installed. Fails under `SMC_REQUIRE_MM=1` when they are not. |
| `demo_core` | function | `open_core(None)`, released afterwards. Skips without the adapters. |
| `demo_microscope` / `demo_microscope_dry` | function | `Microscope.open(Profile.demo())`, live or with `dry_run=True`, closed afterwards. Skips without the adapters. |
| `fake_core` | function | `FakeCore.demo_like()` |
| `fake_microscope` | function | `Microscope.from_core(fake_core, Profile.demo())`, closed afterwards |
| `hardware_microscope` | function | `Microscope.open(<--profile>)`, closed afterwards |

The plugin also adds the option `--profile <name or path>`.

`hardware_microscope` **fails** a test that is not marked
`@pytest.mark.hardware`, and it does so before it opens anything. A stand
is therefore never reached by a test that the default run selects. It
**skips** when `--profile` is not given, with the message "hardware tests
need --profile <name>; run them only at the microscope (ADR-0005)".
`--profile demo` rehearses a hardware run on the simulator.

The demo fixtures use `Profile.demo()` rather than `"demo"`, because
`Profile.load("demo")` picks up whatever `demo.toml` the working directory
has.

### Contract suite (`tests/contracts/`)

Fixtures in `conftest.py`. The contract tests never import a backend
class; they only go through `Microscope`.
- **`backend`** is parametrised over `fake`, `demo` (marked `demo`) and
  `hardware` (marked `hardware`). The default run therefore deselects
  `hardware`, and `pytest -m hardware --profile X tests/contracts`
  selects only it.
- **`stand`** is the backend's facade on its own profile:
  `fake_microscope`, `demo_microscope` or `hardware_microscope`, fetched
  with `request.getfixturevalue`. It owns the core and closes it.
- **`envelope`** records where the stand started: XY, Z, exposure,
  shutter and auto-shutter, each `None` when the stand lacks the
  capability, all read through `stand`. It also holds the contract's
  `[safety]`:
  - `max_jog_um = 50`;
  - XY soft limits at the start ± 100 µm on each axis;
  - Z soft limits at the start ± 3 µm.
- **`microscope`** and **`dry_microscope`** are
  `Microscope.from_core(stand.core, …)`, built with the stand's profile
  whose `[safety]` is replaced by the envelope's: one live, one with
  `dry_run=True`. Neither closes the core; `stand` does.
  - At teardown, `microscope` puts back what the contracts change: XY,
    Z, exposure, shutter and auto-shutter.
  - If the facade is halted, or something is still moving, it calls
    `stop()` instead.
- **`xy`, `z`, `camera`, `shutter`, `properties`** (and `dry_xy`,
  `dry_z`) are that capability of `microscope`.
  - On `hardware`, a stand without the capability skips the test.
  - On `fake` and `demo`, the capability is fetched with `require()`, so
    a broken role resolution fails the test instead of skipping it.

**What a contract may move.** Every target a contract sends lies within
the envelope:
- the XY stage moves at most 60 µm from the start;
- Z moves at most 2 µm, and only below the start, which is away from the
  sample on an inverted stand.

Every target outside the envelope is one the layer must refuse, so none
of them is ever sent. The stand ends where it started.

Required cases. Each case is one test, and every method of every
capability Protocol runs on every backend.
- **XY**:
  - an absolute move lands within 0.5 µm, and returns only once
    `is_busy()` is false;
  - a relative move adds to the position;
  - a jog above `max_jog_um` raises `SafetyRefusedError` with a
    `how_to_force` and sends nothing; with `force=True` it passes;
  - a target outside the soft limits raises with an empty
    `how_to_force`, for an absolute or a relative move, forced or not;
  - `wait()` returns on an idle stage;
  - `stop()` on an idle stage is harmless;
  - `limits_um()` reports the envelope;
  - in dry-run, a move returns the commanded value and the stage does not
    move.
- **Z**: the same cases without the jog guard, which #84 adds.
- **Camera**:
  - `snap()` is 2-D, matches `image_shape()`, and has a dtype wide
    enough for `bit_depth()`;
  - the exposure round-trips within 1 %;
  - `pixel_size_um()` is `0.0` or a positive finite number;
  - on the fake only, the pixel size falls back to the profile's value
    for the current objective, scaled by the binning.
- **Shutter**: open and close round-trip; auto-shutter round-trips.
- **Properties**:
  - `devices()` is not empty and excludes `Core`;
  - `describe()` answers for every device, each entry names its device,
    and at least one device has a property (the demo's `DHub` has none);
  - a read-only property refuses `set()` with `HardwareError` and keeps
    its value. The test is skipped on a stand with no read-only property.
- **Facade**:
  - `state()` of a healthy stand lists no errors;
  - on the fake only, `require()` for a missing role raises
    `CapabilityMissingError` naming the role;
  - on the fake only, `state()` fills what it can when one device fails
    (Z in `failing`).

### Synthetic sample (`smc/testing/synthetic.py`, #10)

```python
@dataclass(frozen=True)
class Blob: x_um: float; y_um: float; radius_um: float; intensity: float

class PlateSample:
    def __init__(self, plate: str = "96-well", a1_center_xy_um: tuple[float, float] = (0.0, 0.0),
                 rotation_deg: float = 0.0, *, well_level: float = 3000.0, plastic_level: float = 800.0,
                 blobs: Sequence[Blob] = (), texture_std: float = 40.0, noise_std: float = 20.0,
                 seed: int = 0): ...
    def render(self, x_um: float, y_um: float, z_um: float, *, shape: tuple[int, int],
               pixel_size_um: float, camera_rotation_deg: float = 0.0, mirrored: bool = False) -> np.ndarray  # uint16

class SampleCamera:   # implements Camera; snap() renders at the stage's current position
    def __init__(self, microscope: Microscope, sample: PlateSample, *, pixel_size_um: float, shape=(512, 512)): ...
```

Well geometry comes from `useq.WellPlatePlan` (never re-derived). Texture
is deterministic per stage position (hash of the world-space tile), so
phase correlation between two overlapping frames works. Fixture:
`demo_microscope_with_sample(sample=…)` calls `microscope.override(Camera,
SampleCamera(...))`. A test proves the frame mean changes when the stage
crosses a well wall.

## 9. CLI — `smc/cli.py` (#11)

*Revised 2026-09-30* (plan of #11), after measuring typer 0.27 and
rich 15: the first version made `--profile` a root option, which Click
accepts only before the command (`smc -p X devices`, not the
`smc devices -p X` used in #11 and #16), and it did not say that a
negative number (`smc stage jog -100 0`) parses as an unknown option.

**Options.**

- `--profile/-p NAME_OR_PATH` belongs to each command that opens a
  microscope (`doctor`, `devices`, `stage …`, `z …`, `snap`) and goes
  after it: `smc devices -p nikon-ti2`. Default: `$SMC_PROFILE`, else
  `demo`.
- `--dry-run` belongs to each command that moves something (`stage move`,
  `stage jog`, `z move`, `z jog`). The move is logged and not sent; the
  command prints the commanded target and the position read afterwards.
  It does not cover what opening the configuration applies (#78).
- Root options go before the command: `--verbose/-v` (INFO log on stderr,
  one line per command sent) and `--debug` (DEBUG log, and the traceback
  instead of the one-line error).
- **Negative numbers.** The move commands set Click's
  `ignore_unknown_options`, so `-100` is a number, not an option. A
  misspelt option (`--dryrun`) then fails as a bad number or an extra
  argument, which is a usage error: exit 2, nothing opened, nothing sent.

| Command | Does |
|---|---|
| `smc profiles` | one line per profile found on the search paths (name, path or `(built in)`, and the error of one that does not load), then the search paths |
| `smc doctor [-p]` | MM status, then open the profile and print `RoleMap.describe()` and the role warnings; `--config CFG` keeps the raw `.cfg` check |
| `smc devices [-p]` | one line per device (label, type, library/name, roles), then `RoleMap.describe()` and the warnings |
| `smc stage get` / `stage move X Y` / `stage jog DX DY [--force]` | `XYStage` |
| `smc z get` / `z move Z` / `z jog DZ` | `ZStage` (no jog guard in M1; the soft limits apply) |
| `smc snap [--out frame.tif] [--exposure-ms MS]` | `Camera`; one TIFF via `tifffile` (new dependency) |
| `smc discover …` | §10 |

**Output.** Plain aligned text, never parsed as rich markup (FM-36) and
never wrapped to the console width (FM-37). Positions have two decimals:
`XY (1234.50, -56.25) µm`, `Z 12.50 µm`. JSON output comes later.

**Exit codes.**

| Code | Meaning |
|---|---|
| 0 | done |
| 1 | a hardware, profile or file error, or an unexpected exception (one line naming its type; `--debug` for the traceback) |
| 2 | refused, nothing sent: a guard (`SafetyRefusedError`, `MotionInProgressError` included) or a bad argument (Click's usage errors exit 2 as well) |
| 130 | interrupted (Ctrl-C): the Executor stopped any move under way (FM-17), and the stand was released |

Errors print one line, `✗ reason — how to force / fix`; for a jog, the
fix is `--force`, not the API's `force=True`. Every command opens the
microscope in a `with` block, so the stand is released before the line is
printed. No signal handler: Ctrl-C is the `KeyboardInterrupt` the Executor
already turns into a stop (FM-70). No `smc stop`: a second process cannot
connect to a stand that the first one holds.

**`smc snap`.** `--out` must end in `.tif` or `.tiff`, and its folder must
exist; both are checked before the stand is opened (FM-30). `--exposure-ms`
must be finite and in (0, 60 000] ms, because a snap cannot be interrupted
(FM-32). A 2-D `uint8` or `uint16` frame is written in its own dtype and
never rescaled; the demo and the Hamamatsu give `uint16` (measured on the
demo, 2026-09-30). Any other frame is refused. The TIFF's description holds
tifffile's JSON metadata: profile, camera, `exposure_ms`, `pixel_size_um`
(`0.0` means unknown, as the camera reports it), `xy_um`, `z_um`,
`time_utc`, `smc_version`. A failed write still prints the frame summary
(FM-34).

**`smc profiles`** skips a TOML file without a `[microscope]` table without
a word, since `./pyproject.toml` is on the search path. It prints the search
paths, so a profile that is not found can be diagnosed.

Tests use `typer.testing.CliRunner` with `COLUMNS=200` on the demo profile;
the fake through `SMC_PROFILE` comes with #9.

## 10. Discover — `smc/discovery/` (#30, pulled into M1)

Read-only survey of a PC, run on each microscope computer following
`docs/hardware/microscope-pc-setup.md` — one folder per PC:

```
smc discover                                   # text report on screen
smc discover --out local\surveys\nikon-ti2     # also writes inventory.json + inventory.txt there (UTF-8)
smc discover --all-adapters                    # devices of every installed adapter (slow: loads every DLL)
smc discover --probe-adapter NikonTi2          # opt-in: initialise each device of ONE adapter in a throwaway core
smc discover --no-os                           # skip serial / USB / PnP / PCI
```

Sections: **system** (host name, UTC time, OS, Python, `smc`,
`pymmcore-plus`, MM install dir, device interface version; other
Micro-Manager installs found on disk, names only); **adapters** (every
installed adapter's *name* — listing names loads no DLL — and the devices
of the adapters in `vendors.LAB_ADAPTERS` plus those the hints suggest, or
of all of them with `--all-adapters`; an enumeration error is recorded as
the finding); **serial** (`pyserial`: device, VID:PID, manufacturer,
description, serial number); **usb / pnp** (Windows: `Get-PnpDevice
-PresentOnly` as JSON, with the console output encoding set to UTF-8;
macOS: `system_profiler SPUSBDataType -json`; Linux: `lsusb`); **pci**
(Windows: PnP entries whose `InstanceId` starts with `PCI\`; Linux:
`lspci -nn`); **hints** (vendor IDs and name fragments → vendor → adapters,
from `vendors.py`, e.g. FTDI → Märzhäuser/ASI/Sutter candidates, PCI `1B6B`
→ Photometrics → `PVCAM`, PCI `1093` → National Instruments → `NIDAQ`, PCI
`10EE` + `CZMI` → Zeiss realtime card); **notes** (one line per thing that
could not be collected).

**Isolation.** Everything that loads a device adapter — the device lists
and the probe — runs in a child process (`python -m
smc.discovery._mm_child`) that prints ASCII JSON, under a timeout (180 s,
`--timeout-s`). Loading a vendor DLL can hang or crash the process; that
must cost one section, not the survey — the lesson behind
`nikon-control`'s one-device-per-throwaway-core probe. A crash or a
timeout becomes a note; the OS sections are still reported.

**Output.** `--out DIR` creates DIR and (over)writes `inventory.json` (the
`Inventory` model) and `inventory.txt` (the text report), both UTF-8 and
written by the tool itself — never through shell redirection. They contain
serial numbers and the host name, so they stay in the git-ignored
`local/`; the design session publishes a redacted version.

**Redirected output.** When stdout or stderr is not UTF-8 (a redirected
stream on Windows is cp1252), the CLI reconfigures it with
`errors="replace"`, so redirected output degrades instead of crashing
(#42).

Every OS call has a 20 s timeout and degrades to a note; nothing is
initialised unless `--probe-adapter` is given; no network access. New
dependency: `pyserial`.

## 11. Sequencing

| Wave | Issues | Parallel? | Notes |
|---|---|---|---|
| 1 | #30 discover (+ #42) · #7 profiles · #6 roles · #5 capabilities + safety + MM backend | yes — separate `git worktree`s | `errors.py` and `Role`/`DeviceInfo` are seeded by the design PR; #5 takes labels from the core's own slots until #8 exists; only #30 touches `cli.py` in wave 1 |
| 2a | #54 motion guard, stop, lock timeout (§13) | after 5 | `errors.py`, `capabilities.py`, `safety.py`, `backends/mm.py` only |
| 2b | #8 facade | after 5, 6, 7, 54 | wires everything; adds `from_core`, `override`, `stop`/`resume` |
| 3 | #9 FakeCore + fixtures + contracts · #10 synthetic · #11 CLI | after 8, parallel | #11 also extends `doctor`; #10 needs `override()` from #8 |

Each wave-1 executor adds its own tests under `tests/unit/` or against
`demo_core` and does **not** touch the other wave-1 modules.

## 12. Decided defaults (so nobody re-decides them)

- Profiles are TOML, not YAML; the demo profile exists both as code
  (`Profile.demo()`) and as `profiles/demo.toml` (loader test).
- `Camera.snap()` returns the raw MMCore array (no copy, no flip); display
  orientation is the UI's job with the camera calibration (M3).
- Dry-run is a property of the `Microscope`, not of individual calls.
- `Microscope.core` stays public for the CLI and tests; a plugin importing
  `pymmcore_plus` or touching `.core` fails review.
- Role names are the ADR-0003 glossary names; `Role` values are also the
  TOML keys.
- No action while a movement has not finished (owner, 2026-09-23): the
  guard refuses, it does not queue. An action holds the lock through its
  wait; reads, `stop()` and closing a shutter never take it (§13, revised
  2026-09-24).

## 13. Motion, stop and the lock (#54)

**Rule (owner, 2026-09-23): no action on a microscope whose movement has
not finished.**

**Revised again 2026-09-28** after the second `/code-review` of #59 (15
more findings, 9 reproduced): almost all came from `wait=False` — a motion
that outlives its action needs an owner, a later `wait()` that must learn
how it ended, and side records keyed by device and thread, and each fix
round closed one path while the next review found another. **M1 has no
`wait=False`**: every move waits inside its action. A UI stays responsive
by moving from a worker thread; reads and stops never wait for the lock.

**Revised 2026-09-24**, after the design review of #59, where `/code-review`
reproduced 15 races. The first version of this section released the lock
during the wait that follows a move, to keep readers and stops responsive.
As a result, waiters, stops and other callers had to find "the" motion of a
device by its label while other threads acted in between. A stop was lost,
a stopped move reported an arrival, a stale waiter stopped someone else's
move, and a readback returned another move's position. This version gets
responsiveness the other way round. **An action holds the lock from its
first check to its readback, wait included. Reads and the safe calls
(stop, closing a shutter) never take the lock.** Only the thread inside an
action sends commands and waits, so these races cannot happen.

The earlier first design held the lock through the wait too, but it took
the lock for reads and stops as well. That gave the three failures #54 was
opened for:
- a plate traverse blocked every reader for up to 60 s, and no stop could
  get through;
- a move that timed out left the stage moving;
- a hung `snapImage` froze every capability (FM-15).

**Measured on the demo** (2026-09-23, pymmcore-plus 0.18.1, pymmcore
12.5.0.75.0):
- pymmcore releases the GIL during a device call: a Python thread ran freely
  through a 2 s `snapImage`.
- MMCore serialises calls per adapter module. During that snap,
  `getXPosition`, `deviceBusy` and `stop` on the demo XY stage (same
  `DemoCamera` module as the camera) waited 1.7 s for it, while `deviceBusy`
  on the `Utilities` LED shutter and `getProperty("Core", …)` answered at once.

So MMCore, not smc, is what keeps concurrent calls to one device safe, and
a read or a stop that skips smc's lock is safe. Such a stop reaches a stage
driven by another adapter than the camera even while a snap hangs, which is
the case on the Ti2 (a `NikonTi2` stage, a Hamamatsu camera). On a stand
where one adapter drives both, the stop waits for the camera call inside
MMCore; nothing in Python can change that.

### Kinds of call

| Kind | Calls | Microscope lock | While another thread's action holds the lock | While a motion outlived its action | While halted | Dry-run |
|---|---|---|---|---|---|---|
| read | `position_um`, `is_busy`, `wait`, `limits_um`, `exposure_ms`, `image_shape`, `bit_depth`, `pixel_size_um`, `is_open`, `auto_shutter`, `Properties.devices/describe/get`, `state()` | **never taken** | runs | runs | runs | runs |
| action | moves, `set_exposure_ms`, `set_open(True)`, `set_auto_shutter`, `Properties.set`, `snap` | held from the first check to the readback, including the wait of a move | if that action is waiting for a motion, **refused at once** (`MotionInProgressError`); otherwise waits for the lock up to `lock_timeout_s` | **refused** | **refused** | mutations logged and skipped; `snap` runs |
| safe | `XYStage.stop`, `ZStage.stop`, `Microscope.stop`, `set_open(False)` | **never taken** | runs | runs | runs | runs (sent to the device) |

Refusals raise at once; motion is never queued. A UI that wants to snap
after a move waits for the move, then snaps.

### The lock

- **One section per action.** An action runs, under the lock and in this
  order:
  1. the halt check;
  2. the guard (below);
  3. its timeout, resolved from the core;
  4. its stop-generation check (below);
  5. the command;
  6. for a move or a moving `Properties.set`, the wait;
  7. the readback.

  A relative move reads the position, checks the soft limits and moves
  inside that section, so its target and its readback are its own.
- **A caller that finds the lock held by an action waiting for a motion is
  refused at once**, with `MotionInProgressError` naming the motion. The
  `Executor` knows because the holder records "waiting for `<motion>`" under
  the registry lock. Any other contention, such as another thread's `snap`
  or a slow set, waits for the lock up to `lock_timeout_s`. Then it raises
  `MicroscopeBusyError`, naming the holder and how long it has held the
  lock.
- **Nothing between `acquire()` and `try`.** Acquire the lock in the frame
  whose `try`/`finally` releases it, with no generator-based context manager
  and no Python call in between. A `KeyboardInterrupt` in that window would
  leave the lock held for good; #59 reproduced this with
  `_thread.interrupt_main()`. The holder's description and start time are
  recorded inside the `try`, under the registry lock, so a caller that times
  out reads a consistent pair.
- `lock_timeout_s` must be finite, positive and at most
  `threading.TIMEOUT_MAX`.
- **Re-entry.** A thread that already holds the lock re-enters it: a
  callback that runs inside `snap` and moves the stage is part of the outer
  action and waits inside it.

### The guard

- **Motions.** An XY or Z move is a motion, and so is a `Properties.set` on a
  `State` device (a turret, a filter wheel, a light path). A property of a
  `Stage` or `XYStage` device set through `Properties` is **not** a motion:
  moving a stage through `Properties` bypasses every guard, and a plugin
  that does it fails review.
- A motion waits inside its action, so normally nothing is left when the
  action returns. **A motion outlives its action** only when the action
  gave up (deadline, interrupt, error) and the device still reads busy, or
  its busy state cannot be read. Such a motion is **registered**, and until
  the device reports idle every action is refused with
  `MotionInProgressError` ("stop it (`stop()`), or `resume()` after
  checking the stand").
- **The guard** runs at step 2 of every action: it asks each registered
  motion `is_busy()` and drops the idle ones.
- **Unreadable busy state** (`is_busy()` raises) counts as moving; the
  refusal names the error.
- **The way out.** `stop()` on a stage drops its registration once the stop
  is sent (a device still reading busy after that stays: it is moving).
  **`resume()` drops every registration**, with a WARNING naming each, so a
  `State` device that cannot be stopped, or a stage that keeps answering
  busy, never locks the stand for good: the operator checks the stand and
  resumes.

### Waiting and giving up

- The wait polls `is_busy()` every `POLL_INTERVAL_S`, holding the lock. Its
  deadline is resolved at step 3, before the command is sent.
- **One `try`** covers the command, the wait and the readback: nothing
  unprotected runs between the command returning and the wait starting, so
  a Ctrl-C landing there still sends the stop.
- **Idle**: if the device's stop generation moved since the action was
  called, raise `MotionStoppedError`; otherwise read back and return.
- **Giving up** (the deadline, `KeyboardInterrupt`, any other exception):
  send the device's stop **first**, register the motion if the device still
  reads busy (or cannot be read), then log, then re-raise.
  `DeviceTimeoutError` says whether the stop was sent, failed, or is
  impossible (a `State` device). A give-up does **not** advance the stop
  generation: it cancels nobody else's action.
- **`wait(timeout_s)`** is a read (above): it polls, takes no lock, and
  never stops anything.

### Stop and halt

- **Stop generation.** Every device has a counter, which only an explicit
  `stop()` increments, before it sends the stop. An action records its device's
  counter when it is called, before it waits for the lock.
  - If the counter has moved by step 4, the command is never sent, and the
    action raises `MotionStoppedError` ("stopped before it started"). A stop
    pressed while a move waits for the lock therefore cancels that move.
  - If the counter moves during the command or the wait, the stop is sent
    again after the command, which may have reached the device first. The
    action then raises `MotionStoppedError` once the device is idle. It does
    so even when the stop command itself failed: the caller asked for a
    stop, so the move did not end as planned.
- **`XYStage.stop()` / `ZStage.stop()`** increment the counter, send MMCore's
  `stop(label)`, and only then log at WARNING. They never take the
  microscope lock, are never refused, and run in dry-run too. The registry
  lock, which is never held across a device call, is the only lock they
  take.
- **The registry lock is an `RLock`** (FM-70, #76). Python runs a signal
  handler on the thread it interrupts, between two bytecodes, so a Ctrl-C
  handler that calls `Microscope.stop()` or `close()` can land while that
  thread is inside a registry section, and a plain `Lock` then blocks the
  stop for ever. Re-entry means a stop can run at any point of a registry
  section on the same thread, even inside one statement such as a
  comprehension, so every section stays correct when it does: it iterates
  a snapshot taken in one call (`list(registry.values())`), and a drop
  tolerates an entry that is already gone.
- **`set_open(False)`** is a safe call like stop. Closing a shutter never
  waits for the lock and is never refused, so the light can always be cut:
  a laser shutter during a traverse, or a cleanup in a `finally` block.
  Opening the shutter is an action; a close requested while an open is
  being sent is sent again after it, and the open raises
  (`MicroscopeHaltedError` when halted, else `HardwareError` "closed while
  opening"), so the shutter never ends open after a close that reported
  success.
- **Every stop path**, whether `stop()`, a give-up in a wait, or the stop
  re-sent after a command, sends the device's stop before it writes a log
  line. A blocked console (a QuickEdit selection on Windows) or a second
  Ctrl-C can then delay or cut the log line, not the stop. One helper sends
  every stop, so the order lives in one place.
- **Several stops in a row** (the emergency stop, and `close()` when
  something moves) send **every** stop before **any** of their log lines,
  the halt's included. Otherwise a blocked console holds the Z stop behind
  the line of the XY stop: on the demo (measured 2026-10-01) the order was
  `microscope: halted`, stop XY, its line, stop Z, its line. The facade
  runs the stops inside `Executor.hold_log(run)`, which holds every line
  the `Executor` writes on the calling thread while `run` runs and writes
  them, in order, once it has returned or raised. Lines that other threads
  write meanwhile are not held, and a `hold_log` nested in another on the
  same thread leaves the writing to the outer one (#76).
- **Logging an action**: its INFO line is written after the halt, guard and
  generation checks, just before the command, so the log never shows a move
  that was not sent.
- **Releasing the lock**: release first, then clear the holder record (only
  if it is still this thread's), so a second Ctrl-C in the `finally` cannot
  keep the lock.
- **`Microscope.stop()`** (#8) is the stand's emergency stop:
  1. it halts first;
  2. it calls `stop()` on every stage, continuing past failures;
  3. only then are the log lines written, `microscope: halted` first, then
     one per stop (both steps above run inside `hold_log`);
  4. it raises one `HardwareError` listing the failures, if any.

  While halted, every action raises `MicroscopeHaltedError`. The halt never
  waits for the microscope lock, so it can land at any moment. Actions check
  it at step 1 and again immediately before their command or acquisition. `resume()` clears
  the halt, drops every registered motion (above), and starts nothing. A
  capability's own `stop()` does not halt.
- **`MicroscopeHaltedError` derives from `HardwareError`, not
  `SafetyRefusedError`.** A plugin that skips a target on
  `except SafetyRefusedError` (the soft limits) must not swallow the
  emergency stop.
- **`Microscope.close()`**: when anything is registered as moving, it calls
  `stop()` on every stage before it unloads the devices.

### Lock timeout (FM-15)

`lock_timeout_s` bounds the **wait for the lock**, not the action that
holds it:
- a hung `snap` makes other actions fail after `lock_timeout_s` with
  `MicroscopeBusyError` ("camera: snap has held the microscope for 61.0 s;
  the camera driver may be hung");
- reads and safe calls are not affected;
- a move holds the lock for its whole traverse, and other actions are
  refused at once (`MotionInProgressError`) rather than timing out.

When no holder is recorded (the lock was taken outside the `Executor`), the message says so and gives the time as "at least" this caller's wait: an assumed figure is labelled. The facade passes `device_timeout_ms / 1000`; the default is 60 s. `snap`
itself stays unbounded, because a thread inside a driver call cannot be
cancelled.

### What #8 relies on

The facade uses only the capabilities' public methods and:
- `Executor(dry_run=..., lock=..., logger=..., lock_timeout_s=...)`;
- `halt()`, `resume()`, `halted`, `moving()` and `hold_log()`;
- the four errors in §2.

The internal methods (`do`, `read`, `wait`, `stop` and their helpers) may
change shape in #59's fix round, provided the behaviour above holds.

### Not verified (M2, #16)

- That each stand's stages report busy as soon as the command is sent. If
  one reports idle for a moment first, the wait returns early and the
  readback is wrong. The Ti2 session checks it: after a 10 mm move, the
  readback must equal the target.
- That `stop()` reaches the Ti2 stage while a Hamamatsu snap is running
  (different adapters, as measured above on the demo).
- That closing the shutter during a snap with auto-shutter leaves the
  camera and MMCore in a usable state (the frame is cut short).
