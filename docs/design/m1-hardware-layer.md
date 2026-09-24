# M1 design — the hardware layer, proven on the simulator

- **Status**: design for issues #5, #6, #7, #8, #9, #10, #11, #30 and #54
  (§13, added 2026-09-23, revised 2026-09-24)
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
  conftest.py          loads smc.testing.fixtures; --profile option
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
  call goes through `Executor`. An *action* (a mutation, `snap`, `wait()`)
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
    def move_to_um(self, x_um: float, y_um: float, *, wait: bool = True) -> XY: ...
    def move_by_um(
        self, dx_um: float, dy_um: float, *, wait: bool = True, force: bool = False
    ) -> XY: ...
    def wait(self, timeout_s: float | None = None) -> None: ...
    def is_busy(self) -> bool: ...
    def stop(self) -> None: ...  # §13 (#54)
    def limits_um(self) -> tuple[Limits, Limits] | None: ...  # (x, y); None = unknown


@runtime_checkable
class ZStage(Protocol):
    def position_um(self) -> float: ...
    def move_to_um(self, z_um: float, *, wait: bool = True) -> float: ...
    def move_by_um(self, dz_um: float, *, wait: bool = True) -> float: ...
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
- `wait()` returns when the device reports not busy. After `timeout_s`
  (default: the core timeout) it raises `DeviceTimeoutError`, having sent
  the device's stop only if this thread started the motion. If the motion
  was stopped meanwhile, it raises `MotionStoppedError` (§13). To ask "is it
  done yet", use `is_busy()`.
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
| `XYStage.move_to_um` | `setXYPosition(label, x, y)`, then the wait if `wait`, then the readback: one action under the lock (§13) |
| `ZStage.position_um` / `move_to_um` | `getPosition(label)` / `setPosition(label, z)`, waited like XY |
| `wait` / `is_busy` | polls `deviceBusy(label)` holding the lock / `deviceBusy(label)`, a lock-free read |
| `XYStage.stop` / `ZStage.stop` | `stop(label)`, through `Executor.stop` (no lock, §13) |
| `Camera.snap` | `snapImage()` then `getImage()` (the core's current camera must be `label`); `Executor.read(..., at_rest=True)` |
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
    def do(self, description: str, action: Callable[[], T], *, dry_result: T,
           motion: Motion | None = None) -> T
    def read(self, action: Callable[[], T], *, at_rest: bool = False,
             description: str = "") -> T
    def wait(self, motion: Motion, timeout_s: float) -> None      # §13
    def stop(self, motion: Motion) -> None                         # §13
    def halt(self) -> None; def resume(self) -> None               # §13
    def moving(self) -> tuple[str, ...]                            # §13
    dry_run: bool
    halted: bool
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

### FakeCore (`smc/testing/fakes.py`)

The subset of the MMCore API the backend uses (§6), with a call `log`,
per-device positions/state, a property store, `snapImage/getImage` served
by a `frame_source: Callable[[float, float, float], np.ndarray] | None`,
and quirk switches (`quirk_z_move_disables_autofocus: bool = True`, kept
for M2). Construction: `FakeCore.demo_like()` mirrors the demo
configuration's labels and types so the same tests run on both.

### Fixtures (`smc/testing/fixtures.py`, loaded by `tests/conftest.py`)

`mm_available`, `demo_core`, `demo_microscope` (+ `dry_run` variant),
`fake_core`, `fake_microscope`, `hardware_microscope` (skips unless
`--profile` was given; only a human at the stand passes it).

### Contract suite (`tests/contracts/`)

`conftest.py` defines `backend` parametrised over `fake`, `demo`,
`hardware` (the last skips without `--profile`); `microscope` fixture maps
it. One file per capability; each test asks `microscope.require(Cap)`.
Required cases:

- XY: absolute move lands within 0.5 µm; relative move adds; jog above
  `max_jog_um` raises `SafetyRefusedError` and `force=True` passes; target
  outside soft limits raises and is not forceable; `wait()` returns;
  dry-run returns the commanded value and the real position is unchanged.
- Z: same shape as XY without the jog guard.
- Camera: `snap()` is 2-D and matches `image_shape()`; exposure round-trips;
  `pixel_size_um()` is `0.0` or positive, never negative; profile fallback
  by objective label works on the fake.
- Shutter: open/close round-trip; auto-shutter round-trip.
- Properties: `devices()` excludes `Core`; `describe()` lists at least one
  property per device; read-only properties refuse `set()`.
- Facade: `require` on a missing role raises `CapabilityMissingError` naming
  the role; `state()` fills what it can when one device fails
  (fake raises on Z).

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

Global option `--profile/-p NAME_OR_PATH` (default `$SMC_PROFILE` or
`demo`); `--dry-run` where a command moves something.

| Command | Does |
|---|---|
| `smc profiles` | list profiles found on the search paths |
| `smc doctor [-p]` | (extend) MM status, then open the profile and print `RoleMap.describe()` with candidates and warnings |
| `smc devices [-p]` | table: label, type, library/name, role |
| `smc stage get` / `stage move X Y` / `stage jog DX DY [--force]` | `XYStage` |
| `smc z get` / `z move Z` / `z jog DZ` | `ZStage` |
| `smc snap [--out frame.tif] [--exposure-ms MS]` | `Camera`; writes 16-bit TIFF via `tifffile` (new dependency) |
| `smc discover …` | §10 |

Errors print one line (`✗ reason — how to force / fix`) and exit 1;
`SafetyRefusedError` exits 2. Tests use `typer.testing.CliRunner` on the demo
profile and on a fake through `SMC_PROFILE`.

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

| Kind | Calls | Microscope lock | While another thread's action holds the lock | While a `wait=False` motion is busy | While halted | Dry-run |
|---|---|---|---|---|---|---|
| read | `position_um`, `is_busy`, `limits_um`, `exposure_ms`, `image_shape`, `bit_depth`, `pixel_size_um`, `is_open`, `auto_shutter`, `Properties.devices/describe/get`, `state()` | **never taken** | runs | runs | runs | runs |
| action | moves, `set_exposure_ms`, `set_open(True)`, `set_auto_shutter`, `Properties.set`, `snap` | held from the first check to the readback, including the wait of a move | if that action is waiting for a motion, **refused at once** (`MotionInProgressError`); otherwise waits for the lock up to `lock_timeout_s` | **refused** | **refused** | mutations logged and skipped; `snap` runs |
| wait | `wait()` | held while it polls | waits for the lock, within its own `timeout_s` | polls it | a stopped motion raises `MotionStoppedError` | nothing moves, so it returns |
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
  6. for a `wait=True` move or a moving `Properties.set`, the wait;
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

### The guard and `wait=False`

- **Motions.** An XY or Z move is a motion, and so is a `Properties.set` on a
  `State` device (a turret, a filter wheel, a light path). A property of a
  `Stage` or `XYStage` device set through `Properties` is **not** a motion.
  Moving a stage through `Properties` bypasses every guard, and a plugin
  that does it fails review.
- A `wait=True` move (the default) waits inside its action, so nothing is
  left registered when it returns.
- A `wait=False` move sends the command, **registers** the motion together
  with the thread that started it, and ends its action. Until the device
  reports idle, every action from any thread is refused with
  `MotionInProgressError`: "wait for it (`wait()`) or stop it (`stop()`)".
  `wait()` and the safe calls are the exceptions.
- **The guard** runs at step 2 of every action. It asks each registered
  motion `is_busy()` and drops the idle ones. A dropped registration's stop
  generation is kept for its device, so a later `wait()` still learns that
  the motion was stopped.
- **Unreadable busy state** (`is_busy()` raises): the motion counts as
  moving, and the refusal names the error. `stop()` on it, or `resume()`,
  drops it with a warning, so a broken device, including a `State` device
  that cannot be stopped, never locks the stand for good. A device that
  still reads busy after a stop stays registered, because it is moving.

### Waiting and giving up

- The wait polls `is_busy()` every `POLL_INTERVAL_S`, holding the lock. Its
  deadline is `timeout_s`, which defaults to the core timeout resolved at
  step 3, before the command is sent.
- **Idle**: if the device's stop generation moved since the action was
  called, the wait raises `MotionStoppedError`; otherwise the action reads
  back and returns.
- **Giving up** (the deadline, `KeyboardInterrupt`, or any other exception):
  the thread that started the motion sends the stop **first**, then logs,
  then re-raises. `DeviceTimeoutError` says whether the stop was sent,
  failed, or is impossible (a `State` device).
- **Only the thread that started a motion stops it.** A `wait()` from any
  other thread never stops anything. At its deadline it raises
  `DeviceTimeoutError`, which says the motion belongs to another caller. It
  also waits for the lock within its own `timeout_s`, so another thread's
  short `wait(0.2)` cannot stop a traverse. This rule includes the thread
  that sent a `wait=False` move and calls `wait()` later: that thread owns
  the motion.

### Stop and halt

- **Stop generation.** Every device has a counter, which `Executor.stop()`
  increments before it sends the stop. An action records its device's
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
- **`set_open(False)`** is a safe call like stop. Closing a shutter never
  waits for the lock and is never refused, so the light can always be cut:
  a laser shutter during a traverse, or a cleanup in a `finally` block.
  Opening the shutter is an action.
- **Every stop path**, whether `stop()`, a give-up in a wait, or the stop
  re-sent after a command, sends the device's stop before it writes a log
  line. A blocked console (a QuickEdit selection on Windows) or a second
  Ctrl-C can then delay or cut the log line, not the stop.
- **`Microscope.stop()`** (#8) is the stand's emergency stop:
  1. it halts first;
  2. it calls `stop()` on every stage, continuing past failures;
  3. it raises one `HardwareError` listing the failures, if any.

  While halted, every action raises `MicroscopeHaltedError`. The halt takes
  no lock, so it can land at any moment. Actions check it at step 1 and
  again immediately before their command or acquisition. `resume()` clears
  the halt, drops unreadable motions (above), and starts nothing. A
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

The facade passes `device_timeout_ms / 1000`; the default is 60 s. `snap`
itself stays unbounded, because a thread inside a driver call cannot be
cancelled.

### What #8 relies on

The facade uses only the capabilities' public methods and:
- `Executor(dry_run=..., lock=..., logger=..., lock_timeout_s=...)`;
- `halt()`, `resume()`, `halted` and `moving()`;
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
