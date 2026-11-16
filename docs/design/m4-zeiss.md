# M4 design — the Zeiss Axio Observer 7

- **Status**: design for issue #25 (§0 to §10, 2026-11-16, with the owner);
  §11 is reserved for #26 (Definite Focus 2), fixed by `/plan 26`; §12 lists
  what the first session at the stand (#27) measures.
- **Owner**: the design session. **Executors**: `/develop` sessions, one per issue.
- **Rule**: as in M1 and M2, this document is the contract between issues
  that are built in parallel. Names, module paths and signatures below are
  fixed; an executor who needs to change one stops and comments on its issue
  instead. Where this document extends M1 or M2, it says which section, and
  their rules still hold everywhere else.

Read first: [ADR-0007](../adr/0007-non-micromanager-hardware.md) (accepted
2026-11-16), M1 §2, §6, §7, §8 and §13, M2 §11.12 (a stand that refuses
`stop()`), [`docs/hardware/inventory.md`](../hardware/inventory.md) (the
Zeiss rows) and [`zeiss-observer7.md`](../hardware/zeiss-observer7.md).

## 0. Scope

#25 brings the Observer 7's MTB components into the layer as pymmcore-plus
unicore Python devices, in the same core as the PVCAM camera, so that roles,
capabilities, guards, `useq` MDA and every plugin work unchanged (ADR-0007
point 1). It delivers: the core opener's choice of `UniMMCore` (§3); a
narrow seam over MTB (§4) with its .NET implementation (§5); the
process-wide session owner (§6); the devices (§7); the light on every exit
path (§8); the fake MTB and a contract stand on it (§9); the profile and the
`.cfg` (§10).

Not in #25: Definite Focus 2, the generalised Z interlock and
`stabilize_now` (§11, #26); the TL halogen lamp's intensity (never
controlled by the predecessor either; the TL shutter gates it); `smc doctor`
and `smc discover` checks for MTB; channel presets that switch the core's
shutter (pymmcore-plus#635, §1.3); darkening on `Microscope.stop()` (a
layer-wide question for every stand, not a Zeiss one); anything run at the
stand (#27).

## 1. What is known

### 1.1 The stand (inventory, 2026-09-24, and the predecessor)

From `docs/hardware/inventory.md` and `lightsheet-live-tracking-tool`'s
`tracking_tools/microscope_interface/mtb.py` (HEAD f476600, used daily on
this stand):

- One assembly, `MTBApi.dll` (`C:\Program Files\Carl Zeiss\MTB 2011 -
  2.12.0.7\MTBApi\`), loaded with `clr.AddReference`; everything else is in
  the `ZEISS.MTB.Api` namespace. `MTBConnection().Login("en", "")` returns
  a client id (as the return value or in a tuple, by pythonnet version);
  `GetRoot(client_id)` initialises **all** the hardware (it fails when the
  stand is off: "Error while finishing initialization of component Piezo
  Focus … hardware switched on"); `root.GetComponent(id)` returns `None`
  for an absent component.
- **One login per process.** `Logout` tears down the realtime subsystem and
  a second `Login` in the same process fails with `MTB could not be
  initialized ConnectToRtSystem() … (0x80000004 = E_NOINTERFACE)` (observed
  2026-09-04). ZEN's MTB client competes with ours: ZEN is closed.
- Components are concrete `ZEISS.MTB.MicControl.MTBCtrl*` objects whose
  members appear only after an interface cast, `getattr(ZEISS.MTB.Api,
  "IMTBxxx")(obj)`: `IMTBContinual` (axes, LED intensity), `IMTBChanger`,
  `IMTBObjectiveChanger`, `IMTBFluorescenceLEDControl`,
  `IMTBFluorescenceLED`.
- Component ids: `MTBStageAxisX`, `MTBStageAxisY`, `MTBFocus`,
  `MTBPiezoFocusCan`, `MTBObjectiveChanger`, `MTBReflectorChanger`,
  `MTBSideportChanger`, `MTBCondenserContrastChanger`, `MTBTLShutter`,
  `MTBFLLEDController` (Colibri), `MTBFocusStabilizer2` (DF2, #26),
  `MTBTLHalogenLamp` (not used).
- **Every move is synchronous**: `SetPosition(value, unit, mode, timeout_ms)`
  with `mode = MTBCmdSetModes.Synchronous` blocks until the move ends.
  There is **no axis stop and no axis busy** in the predecessor's use of
  MTB. A synchronous move and its readback cost about 0.05 s for a short
  move.
- **pythonnet ≥ 3 refuses int → Enum**: `MTBCmdSetModes.Synchronous` is
  taken as the member (`getattr(MTBCmdSetModes, "Synchronous")`), never as
  an int. `SetOnOff` on an LED failed on the stand with three guessed
  argument shapes; the predecessor builds its arguments from the overload's
  reflected signature (`_args_for`, mtb.py:538-608).
- Axes: the unit string is found among `GetPositionUnit(i)` (the focus
  offers µm and nm). Ranges: X/Y ±350 000 µm (step 0.25), focus ±14 000 µm
  (step 0.010), piezo 0 to 500 µm (step 0.010). A zero-distance
  `SetPosition` returns `False` ("nothing to do"); the piezo dithers above
  its step width, so `False` with a readback within `max(step,
  TypicalDeviation)` of the target is an arrival. The predecessor clamps
  targets to the range; this layer refuses them instead (§7.3).
- Changers: `Position`, `MinPosition`, `MaxPosition` are 1-based Int16;
  `GetElement(n)` is **0-based**, so a position's name is
  `GetElement(position - 1).Name` (reading `GetElement(position)`
  mislabelled every objective, with the wrong magnification and pixel size).
  An empty objective slot has `ElementType == "None"`. Sideport 1 "VIS
  100%", 2 "Sideport 100% Left" (the camera), 3 right; reflector and
  condenser 6 positions; objectives: 2 Fluar 10x/0.50, 4 Plan-Apochromat
  40x/0.95.
- The TL shutter is a two-position changer; the predecessor finds *open*
  by an element named "open" and takes the other position as closed. If the
  names do not match, its `set_open` raised every time and the caller
  swallowed it, so the TL light "would never be gated at all, for a whole
  overnight run" (`tools/light_check.py`).
- Colibri: six LEDs, 385, 430, 475, 511, 567, 630 nm (`NumberOfLEDs`,
  `LED[i]`). Each LED casts to `IMTBFluorescenceLED` (`Wavelength`,
  `GetOnOff`/`SetOnOff`) and to `IMTBContinual` (intensity, unit "%", 0 to
  100). Manual mode is `SetOperationMode(LEDLight, Continuous, mode,
  timeout)`. The predecessor lights one LED at a time and switches every
  other one off.
- **Phototoxicity.** "A real overnight experiment killed the embryos … The
  setup dose was the experiment." Light outside the exposure is ~300× the
  dose needed. The predecessor gates the LEDs to the exposure, switches the
  light off on every exit path (end of live, run start and stop, atexit,
  browser session end), and shows a failed switch-off as "LIGHT MAY STILL
  BE ON".
- Timeouts: axes and LEDs 10 000 ms, changers and objective 30 000 ms.

### 1.2 unicore in pymmcore-plus 0.18.1 (measured 2026-11-16)

Measured in a one-off script against `.venv`'s pymmcore-plus 0.18.1
(pymmcore 12.5.0.75.0), the latest release:

- `UniMMCore.loadSystemConfiguration` runs lines prefixed `#py ` as normal
  commands; `#py pyDevice,<label>,<module>,<Class>` imports the module and
  calls `Class()` **with no arguments**. Pre-init properties, `Property`
  lines, `Label` and `ConfigGroup` lines work with the prefix. A plain
  `CMMCorePlus` reads the same file **without a word** and loads nothing but
  `Core`. `#py Property,Core,Focus,Z` and `#py Property,Core,Shutter,Sh`
  fill the slots.
- The file is read with `path.read_text()`, without an encoding: cp1252 on
  Windows.
- Base classes and their abstract methods: `StageDevice`
  (`set_position_um`, `get_position_um`, `home`, `stop`, `set_origin`),
  `XYStageDevice` (`set_position_um(x, y)`, `get_position_um`, `home`,
  `stop`, `set_origin_x`, `set_origin_y`), `StateDevice` (`get_state`,
  `set_state`; positions and labels passed to `__init__`), `ShutterDevice`
  (`get_open`, `set_open`). `Device.busy()` defaults to `False`;
  `initialize()` and `shutdown()` default to no-ops.
- **Every core call on a Python device holds that device's own
  non-reentrant lock** (`with dev:`): set, get, `deviceBusy`, properties,
  every poll of `waitForDevice`, and `unload`. `stop` and `home` do not.
  So a synchronous `set_position_um` blocks `getPosition` and `deviceBusy`
  for the whole move, and a device method that calls the core on its own
  label deadlocks (and hangs the interpreter's exit).
- An exception raised by `set_position_um` reaches the caller of
  `setPosition` unchanged (a `RuntimeError` stays a `RuntimeError`); one
  raised by `busy()` ends `waitForDevice` with that exception (after
  0.208 s for a 0.2 s worker).
- A failing `initialize()` is **only logged**:
  `initializeAllDevices()` returns, and
  `getDeviceInitializationState(label)` reads `InitializationFailed`.
- A `StateDevice` built with one placeholder position can be given its
  positions in `initialize()`: `assign_label_to_position(i, label)` for
  each, then `get_property_info(Keyword.State).allowed_values =
  list(range(n))`; `getNumberOfStates`, `getStateLabels`, `setState`,
  `setStateLabel` then see exactly those, and the placeholder label is
  refused.
- `@pymm_property` shares one `PropertyInfo` across every instance of a
  class; `register_property` is per instance.
- `core.stop(label)` reaches the device's `stop()` while a blocking set
  runs.

### 1.3 What 0.18.1 lacks

- **pymmcore-plus#635**: `getProperty("Core", "Shutter")` (and `Focus`,
  `XYStage`) reads `""` for a Python device, and `setProperty("Core",
  "Shutter", <python device>)` raises `ValueError` — so a channel preset
  that sets `Core,Shutter` to the Colibri or the TL shutter fails on
  `setConfig`. The `.cfg`'s slot lines and `setShutterDevice` work.
- **pymmcore-plus#629**: after a Python `StateDevice` moves, the core's
  property *cache* keeps the old `State` and `Label`; the live
  `getState`/`getStateLabel` are right. MMCore's pixel-size match reads the
  cache, so the pixel size after an objective change is stale.

Both are merged upstream, unreleased. ADR-0007 point 5: #25 is built and
tested on 0.18.1 against the fake MTB; #27 needs the release.

## 2. Module map

```
src/smc/hardware/core.py                     (edit) §3: UniMMCore when the .cfg has #py lines
src/smc/hardware/backends/zeiss_mtb/
    __init__.py                              (new)  docstring; re-exports nothing that imports .NET
    api.py                                   (new)  §4: the seam — Protocols, value types, MTBError
    dotnet.py                                (new)  §5: the seam over MTBApi.dll (pythonnet); imported lazily
    session.py                               (new)  §6: the process-wide session owner
    devices.py                               (new)  §7: the unicore devices
src/smc/testing/fake_mtb.py                  (new)  §9: FakeMTB, the fake stand's .cfg
src/smc/testing/fixtures.py                  (edit) §9: zeiss_fake_microscope
profiles/zeiss-observer7.toml, .cfg          (new)  §10
pyproject.toml                               (edit) mypy override for clr, System, ZEISS
tests/contracts/conftest.py                  (edit) §9: the zeiss-fake backend
```

Nothing outside `smc.hardware.backends.zeiss_mtb` imports `clr`, `System`
or `ZEISS`, and only `dotnet.py` does, inside functions. `smc.testing`
imports `api` and `session`, never `dotnet`.

## 3. The core opener — `core.py` (extends M1 §6)

```python
def declares_python_devices(config: Path) -> bool:
    """Whether a line of ``config``, stripped of leading spaces, starts with ``#py ``."""


def _new_core(
    adapter_search_paths, *, python_devices: bool = False
) -> tuple[CMMCorePlus, str]: ...
```

- `declares_python_devices` reads the file's **bytes** (`b"#py "` at a line's
  start, after `lstrip(b" \t")`), never text: a `.cfg` saved in cp1252 has
  bytes that UTF-8 refuses (FM-05, FM-80).
- `open_core(config, …)` resolves `config` first, then builds the core
  with `python_devices=declares_python_devices(path)`. `UniMMCore` is
  imported from `pymmcore_plus.experimental.unicore` inside `_new_core`,
  only when asked: the Nikon stands and the demo never import unicore
  (ADR-0007 point 2). `new_core()` (public, `smc config build`) and
  `status()` stay plain.
- After a `UniMMCore` has loaded the file, the opener reads
  `getDeviceInitializationState(label)` for every loaded device; any that is
  not `InitializedSuccessfully` makes the open fail: the core is closed
  (`close_core`) and a `CoreError` names every such device, in load order,
  and says that the device's own ERROR line in the log carries the cause.
  unicore only logs a failing `initialize()` (§1.2), so without this the
  stand opens with a device that does nothing.

## 4. The seam over MTB — `zeiss_mtb/api.py`

The devices speak to MTB only through these Protocols. `dotnet.py`
implements them over `MTBApi.dll`; `smc.testing.fake_mtb` implements them
in Python. Every call is synchronous and may block; every failure is an
`MTBError`.

```python
class MTBError(RuntimeError):
    """MTB refused a call or failed in it; the text names the component and the call."""


class MTBSessionError(MTBError):
    """The process's session cannot serve: logged out, or a second session refused."""


@dataclass(frozen=True)
class AxisInfo:
    min_um: float
    max_um: float
    step_um: float
    typical_deviation_um: float
    max_deviation_um: float


@dataclass(frozen=True)
class ChangerElement:
    position: int  # MTB's, 1-based
    name: str
    element_type: str  # "None" for an empty objective slot


class MTBAxis(Protocol):
    mtb_id: str

    def info(self) -> AxisInfo: ...
    def position_um(self) -> float: ...
    # returns MTB's own return value
    def move_um(self, target_um: float, timeout_ms: int) -> bool: ...


class MTBChanger(Protocol):
    mtb_id: str

    # MinPosition..MaxPosition, in order
    def elements(self) -> tuple[ChangerElement, ...]: ...
    def position(self) -> int: ...  # 1-based
    def move(self, position: int, timeout_ms: int) -> bool: ...


class MTBLED(Protocol):
    wavelength_nm: int

    def is_on(self) -> bool: ...
    def set_on(self, on: bool, timeout_ms: int) -> None: ...
    def intensity_pct(self) -> float: ...
    def set_intensity_pct(self, pct: float, timeout_ms: int) -> None: ...


class MTBLEDController(Protocol):
    mtb_id: str

    def leds(self) -> tuple[MTBLED, ...]: ...  # in MTB's order
    def set_manual_mode(self, timeout_ms: int) -> None: ...


class MTBApi(Protocol):
    def login(self) -> None: ...
    def logout(self) -> None: ...
    def axis(self, mtb_id: str) -> MTBAxis: ...  # MTBError naming the id when absent
    def changer(self, mtb_id: str) -> MTBChanger: ...
    def led_controller(self, mtb_id: str) -> MTBLEDController: ...
```

`MTBError` is a `RuntimeError` so that the layer's `as_hardware_errors`
turns it into a `HardwareError` naming the role and the call (FM-79); a
device never raises anything else out of a core call except `ValueError`
for a value outside a property's range, which unicore raises itself.

Timeouts, decided (§1.1 and FM-10): `AXIS_TIMEOUT_MS = 60_000` (a traverse
of the plate; the predecessor's 10 s was for short moves),
`CHANGER_TIMEOUT_MS = 30_000`, `LED_TIMEOUT_MS = 10_000`, constants in
`api.py`.

## 5. The .NET seam — `zeiss_mtb/dotnet.py`

```python
DEFAULT_MTB_API = Path(
    r"C:\Program Files\Carl Zeiss\MTB 2011 - 2.12.0.7\MTBApi\MTBApi.dll"
)


def connect(dll: Path = DEFAULT_MTB_API) -> MTBApi: ...
```

- `connect` imports `clr` (pythonnet) and `ZEISS.MTB.Api` inside the
  function; a missing pythonnet, a missing DLL or a failing load is an
  `MTBSessionError` that says which (`pip install smc[zeiss]`, the path
  tried). It returns a `DotNetMTB`, not yet logged in.
- Ported from mtb.py **with its tests**: the login (client id from the
  return value or the tuple), `GetRoot`, `GetComponent` with `None` as
  absent, the interface casts, `resolve_mode` (the enum member, never an
  int), the µm unit found among the units offered (a component that offers
  none is an `MTBError`), `GetElement(position - 1)`, the LED arguments
  built from the reflected overload signature (`_args_for`), and the
  out-parameters taken from the returned tuple.
- Every .NET call is wrapped: any exception becomes an `MTBError` naming the
  component id and the call, chained `from exc` (pythonnet raises .NET
  exceptions that are no `RuntimeError`, FM-113).
- The adapters take the .NET objects and a namespace object for the enums
  as arguments, so the tests drive them with duck-typed fakes on any OS, as
  the predecessor's `tests/test_mtb.py` does; only `connect` needs Windows.
- No clamping and no tolerance logic here: `move_um` returns MTB's bool and
  the device judges it (§7.3).

## 6. The session owner — `zeiss_mtb/session.py`

```python
class MTBSession:
    def axis(self, mtb_id: str) -> MTBAxis: ...
    def changer(self, mtb_id: str) -> MTBChanger: ...
    def led_controller(self, mtb_id: str) -> MTBLEDController: ...
    def register_darkener(self, label: str, darken: Callable[[], None]) -> None: ...
    def unregister_darkener(self, label: str) -> None: ...


def shared() -> MTBSession: ...
def use_api_factory(factory: Callable[[], MTBApi]) -> AbstractContextManager[None]: ...
```

- **One session per process (ADR-0007 point 3).** `shared()` returns the
  process's session, creating it and logging in on the first call, under a
  module lock, so two devices initialised in parallel (unicore initialises
  on a thread pool, §1.2) log in once. The default factory is
  `dotnet.connect`. Components are looked up once and cached by id.
- **Logged out only at process exit.** The first login registers an
  `atexit` handler that runs every registered darkener (§8), switches off
  every LED of every LED controller the session handed out, then logs out.
  Nothing else logs out: not `Microscope.close()`, not a device's
  `shutdown()`, not *Disconnect*. After the logout, `shared()` and every
  component call raise `MTBSessionError("MTB was logged out; it cannot log
  in twice in one process")`.
- A login that fails leaves no session: the next `shared()` tries again (the
  stand was off; it is now on). Only a logout is final.
- **For tests only**: `use_api_factory(factory)` makes `shared()` build its
  session from `factory` inside the block. It refuses to enter
  (`MTBSessionError`) while a session made by another factory exists. On
  exit it runs that session's exit handler (darken, logout), unregisters it
  from `atexit`, and forgets the session, so the next block starts clean
  (FM-46). Production code never calls it.

## 7. The devices — `zeiss_mtb/devices.py`

### 7.1 Classes and labels

| Class | Base | MTB id(s) | Label in the `.cfg` | Role |
|---|---|---|---|---|
| `MTBXYStage` | `XYStageDevice` | `MTBStageAxisX`, `MTBStageAxisY` | `XYStage` | `xy_stage` |
| `MTBFocus` | `StageDevice` | `MTBFocus` | `Focus` | `focus` |
| `MTBPiezo` | `StageDevice` | `MTBPiezoFocusCan` | `Piezo` | none |
| `MTBObjectiveChanger` | `StateDevice` | `MTBObjectiveChanger` | `Objective` | `objective_turret` |
| `MTBReflectorChanger` | `StateDevice` | `MTBReflectorChanger` | `Reflector` | `filter_turret` |
| `MTBSideportChanger` | `StateDevice` | `MTBSideportChanger` | `Sideport` | `light_path` |
| `MTBCondenserChanger` | `StateDevice` | `MTBCondenserContrastChanger` | `Condenser` | none |
| `MTBTLShutter` | `ShutterDevice` | `MTBTLShutter` | `TLShutter` | none |
| `MTBColibri` | `ShutterDevice` | `MTBFLLEDController` | `Colibri` | `shutter`, `light_source` |

The MTB id is a class attribute (`MTB_ID: ClassVar[str]`), since unicore
builds a device with no arguments; one class per changer, sharing a private
base `_MTBChanger`, and `MTBFocus` and `MTBPiezo` sharing `_MTBZAxis`. The
Colibri is a `ShutterDevice`, not the `GenericDevice` the issue first named
(decided with the owner, 2026-11-16): the layer switches a light source
only when it is `Shutter`-typed (`MMLightSource.set_on`), and Micro-Manager's
auto-shutter, which limits the light to the exposure, opens the core's
shutter.

### 7.2 Rules every device follows (ADR-0007 point 6)

1. **The session is found in `initialize()`**, through `session.shared()`;
   nothing reaches MTB in `__init__`. `initialize()` raises an `MTBError`
   with a diagnosis (absent component, stand off, pythonnet missing) and
   logs it at ERROR first, since unicore only logs it and the opener (§3)
   names the device.
2. **A device never calls the core** (no `self.core.…`): unicore's device
   lock is not reentrant (§1.2).
3. **A move returns at once.** `set_position_um` and `set_state` validate
   the target, start the synchronous MTB call on a worker thread owned by
   the device, and return; `busy()` is `True` while that worker runs. This
   is what keeps reads from waiting behind a move: unicore holds the
   device's lock for the length of the set call. A shared private helper,
   `_Mover`, holds the worker, its own `threading.Lock`, and the outcome.
4. **A move that failed is raised once, by the next `busy()`** once its
   worker has ended (`MTBError` naming the device, the target and MTB's
   answer), and logged at ERROR when it happens; `waitForDevice` then
   raises it (§1.2). A new move while one runs is refused with an
   `MTBError` ("… is moving"); the layer never asks for one.
5. **Reads go to MTB live**, without the mover's lock:
   `get_position_um` and `get_state` read MTB's position while a move runs
   (a position between the start and the target is the truth then, FM-195).
   Whether MTB answers a read while a synchronous move of the same
   component runs is not known; #27 measures it (§12).
6. **`stop()` cannot stop what MTB is doing**: MTB has no axis stop (§1.1).
   It cancels what the device has not yet sent (the Y leg of an XY move,
   §7.3) and logs at WARNING that the move sent runs to its end. The
   profile declares `stop_refused_by = ["xy_stage", "focus"]`, so the layer
   cuts long moves into segments (M2 §11.12) and its stop contract holds.
7. **`home()`, `set_origin()`, `set_origin_x()`, `set_origin_y()` raise
   `MTBError`**: smc never homes or redefines the stand's origin.
8. **Errors**: only `MTBError` (or unicore's own `ValueError`) leaves a
   device; every other exception from the seam is wrapped, chained.
9. **Shutters are synchronous.** `set_open` of the TL shutter and of the
   Colibri runs in the call (a switch takes milliseconds to a fraction of a
   second); only the shutter's own reads wait behind it.

### 7.3 Axes

- **Units**: µm, as MTB reports them in its µm unit; no conversion.
- **Range**: a target outside `[info.min_um, info.max_um]` is refused with
  an `MTBError` naming the range; never clamped (a clamp reads as success).
- **Arrival**, a pure function in `devices.py`:
  `arrived(target_um, read_um, info) -> bool` is `abs(read_um - target_um)
  <= max(info.step_um, info.typical_deviation_um, 1e-6)`. A target already
  arrived at is not sent (MTB returns `False` for a zero-distance move). A
  `move_um` that returns `False` is an arrival if the readback has arrived,
  otherwise a failure; one that returns `True` is checked the same way,
  with `info.max_deviation_um` as the bound, and a readback beyond it is a
  failure (FM-167).
- **`MTBXYStage`** moves X, then Y, as two synchronous MTB moves in one
  worker; an axis already arrived is skipped. The soft limits are a
  rectangle, so both legs of the L stay inside it. `get_position_um()`
  returns `(x, y)` read live. `stop()` during the X leg cancels the Y leg.
- **`MTBFocus`, `MTBPiezo`**: one axis each. `is_continuous_focus_drive()`
  stays `False`.

### 7.4 Changers

- **States are MTB positions minus `MinPosition`** (0-based for
  Micro-Manager, 1-based for MTB).
- **Positions and labels are set in `initialize()`** (§1.2): the device is
  built with `{0: "Position-1"}`, then reads `elements()` and assigns every
  position. **A label is the element's name, stripped**; an empty name is
  `"Position-<n>"`; a name that two positions share (the condenser's and
  the reflector's "none") becomes `"<name> (<n>)"` for each of them, with
  `<n>` MTB's 1-based position.
- `set_state` is a move (§7.2 rule 3) with `CHANGER_TIMEOUT_MS`; a `False`
  from MTB with the position unchanged is a failure. The objective turret's
  confirmation is the layer's (`turret_requires_confirm`), unchanged.
- The magnification of an objective is not read in #25 (no pixel size is
  derived from it: §10).

### 7.5 TL shutter — `MTBTLShutter`

- **Open is found by name, or the device does not initialise.** The changer
  must have exactly two positions and exactly one element whose stripped,
  case-folded name is `"open"`; closed is the other position. Anything else
  is an `MTBError` in `initialize()` that lists the element names: a TL
  shutter that cannot be gated is a failed connect, never a swallowed error
  at every exposure (§1.1).
- `set_open(open)` moves to that position synchronously and checks the
  readback; `get_open()` reads MTB's position live.

### 7.6 Colibri — `MTBColibri`

One LED at a time (decided with the owner, 2026-11-16).

- Properties, registered per instance with `register_property` (§1.2):
  - `LED`: allowed values `"None"` and one `"<nm>nm"` per LED in MTB's
    order (`"385nm"`, …, `"630nm"`), set in `initialize()`; default
    `"None"`.
  - `Intensity`: float, limits 0 to 100 (%), default `0.0`; the intensity
    the selected LED is lit at. `[light_source] intensity_property =
    "Intensity"` points the layer's knob at it.
- `set_open(True)`: refused with an `MTBError` when `LED` is `"None"`;
  otherwise manual mode, every other LED off, the selected LED's intensity,
  the selected LED on, then a readback. A failure in any step switches
  every LED off (each tried) and raises.
- `set_open(False)`: **every** LED off, not only the selected one, each
  tried even when one fails (FM-49); the failures are raised together in
  one `MTBError` after the last attempt.
- `get_open()`: `True` when any LED reads on, so light from any LED reads
  as open.
- Setting `Intensity` while open applies it to the lit LED at once; setting
  `LED` while open lights the new LED and switches the old one off; setting
  `LED` to `"None"` while open closes.

## 8. The light on every exit path

The phototoxicity lesson (§1.1), as code. Each step is idempotent:
switching a light off can only ever be safe.

1. **`initialize()`** of `MTBColibri` and `MTBTLShutter` darkens first (every
   LED off; the TL shutter closed), so a stand left lit by ZEN or a crash
   starts dark. A failure there fails the initialise.
2. **`shutdown()`** (every unload: `Microscope.close()`, *Disconnect*,
   `close_core`, the end of `with Microscope.open(…)`) darkens, then
   unregisters the device's darkener. It never raises, since the unload of
   the other devices must go on; a failure is logged at ERROR as
   `LIGHT MAY STILL BE ON: <label>: <cause>`. If a move's worker still runs
   (an axis or a changer), `shutdown()` waits for it up to its MTB timeout,
   then logs and returns.
3. **Process exit**: each lit device registers a darkener with the session
   in `initialize()`; the session's `atexit` handler runs every darkener
   still registered, then switches off every LED of every controller it
   handed out, then logs out (§6). Each step is tried whatever the others
   do; each failure is logged at ERROR with the same `LIGHT MAY STILL BE
   ON` text.
4. **During imaging**, the Colibri is the core's shutter (`[roles.assign]
   shutter`, and the `.cfg`'s `Core,Shutter` line): Micro-Manager's
   auto-shutter lights it only for the exposure. Nothing new in the layer.

Not covered, and said so in the PR: a process killed without its `atexit`
(a hard crash, a kill from the task manager) leaves the light as it was;
`Microscope.stop()` does not darken (§0).

## 9. The fake MTB and the contract stand

### 9.1 `smc/testing/fake_mtb.py`

```python
class FakeMTB:  # implements api.MTBApi
    def __init__(self, *, move_s: float = 0.0) -> None: ...

    axes: dict[str, FakeAxis]
    changers: dict[str, FakeChanger]
    colibri: FakeLEDController
    logins: int
    logouts: int


def fake_stand_config(profile_cfg: Path, out_dir: Path) -> Path: ...
```

- The stand as §1.1 describes it: the four axes with their ranges, steps
  and deviations (X, Y at 0.0; focus at 1000.0 µm; piezo at 250.0 µm), the
  five changers with the stand's element names (TL shutter: `"Open"`,
  `"Closed"`), six LEDs.
- A synchronous move sleeps `move_s` and moves in a straight line, so a
  read from another thread during it sees a position between start and
  target. Knobs on each fake component, for the failure tests:
  `refuse_next` (returns `False`, does not move), `fail_next` (raises
  `MTBError`), `deviation_um` (the readback's offset), `absent` (on
  `FakeMTB`: ids that `axis`/`changer` refuse), and on an LED `fail_off`.
  Every call is recorded in order (`calls`).
- `fake_stand_config` writes the profile's `.cfg` with every line that is
  not a `#py` line or a `Property,Core,Initialize` line dropped: the
  contract stand runs **the profile's own device lines**, without the PVCAM
  camera, so it needs no Micro-Manager adapter and runs on every CI job.
  It writes UTF-8 (ASCII content, §10).

### 9.2 `zeiss_fake_microscope` (`smc.testing.fixtures`)

Inside `session.use_api_factory(lambda: fake)`, it opens a `Microscope` on
`Profile.load("zeiss-observer7")` with `[micromanager] config` replaced by
`fake_stand_config(...)` in `tmp_path`, yields it, and closes it. It is
added to `tests/contracts/conftest.py`'s `backend` params as `zeiss-fake`
and to `_STANDS`. The stand has no camera and no autofocus; the contracts
skip what a stand lacks, as they do already.

A contract that fails on `zeiss-fake` **only** because of pymmcore-plus#629
or #635 (§1.3) gets `pytest.mark.xfail(strict=True, reason=…)` under a
condition on `pymmcore_plus.__version__` being 0.18.1 or older, so that the
release turns it red until the mark goes; the Report lists each such mark.
Any other failure is a defect of #25.

## 10. Profile and configuration

`profiles/zeiss-observer7.cfg` — **ASCII only** (unicore reads it as cp1252
on Windows, §1.2):

```
# Zeiss Axio Observer 7: MTB 2011 components as pymmcore-plus unicore
# Python devices (ADR-0007, docs/design/m4-zeiss.md); Prime 95B on PVCAM.
# ASCII only: UniMMCore reads this file without an encoding.
Property,Core,Initialize,0
Device,Camera-1,PVCAM,Camera-1
#py pyDevice,XYStage,smc.hardware.backends.zeiss_mtb.devices,MTBXYStage
#py pyDevice,Focus,smc.hardware.backends.zeiss_mtb.devices,MTBFocus
#py pyDevice,Piezo,smc.hardware.backends.zeiss_mtb.devices,MTBPiezo
#py pyDevice,Objective,smc.hardware.backends.zeiss_mtb.devices,MTBObjectiveChanger
#py pyDevice,Reflector,smc.hardware.backends.zeiss_mtb.devices,MTBReflectorChanger
#py pyDevice,Sideport,smc.hardware.backends.zeiss_mtb.devices,MTBSideportChanger
#py pyDevice,Condenser,smc.hardware.backends.zeiss_mtb.devices,MTBCondenserChanger
#py pyDevice,TLShutter,smc.hardware.backends.zeiss_mtb.devices,MTBTLShutter
#py pyDevice,Colibri,smc.hardware.backends.zeiss_mtb.devices,MTBColibri
Property,Core,Initialize,1
Property,Core,Camera,Camera-1
#py Property,Core,XYStage,XYStage
#py Property,Core,Focus,Focus
#py Property,Core,Shutter,Colibri
```

No channel preset and no pixel-size preset: presets that switch the core's
shutter wait for #635's release, and the pixel sizes are measured at #27.

`profiles/zeiss-observer7.toml` — no new key (ADR-0007 point 1):

```toml
[microscope]
name = "zeiss-observer7"
vendor = "Zeiss"
description = "Axio Observer 7: MTB 2011 through pythonnet as unicore devices, Prime 95B on PVCAM (ADR-0007)"

[micromanager]
config = "zeiss-observer7.cfg"
device_timeout_ms = 60000
adapter_search_paths = []

[roles.assign]
xy_stage = "XYStage"
focus = "Focus"
objective_turret = "Objective"
filter_turret = "Reflector"
light_path = "Sideport"
shutter = "Colibri"
light_source = "Colibri"

[light_source]
intensity_property = "Intensity"

[safety]
max_jog_um = 5000.0      # the layer's default, not chosen for this stand
max_z_jog_um = 100.0     # the layer's default, not chosen for this stand
turret_requires_confirm = true
# MTB 2011 has no axis stop: a synchronous SetPosition runs to its end
# (lightsheet-live-tracking-tool, mtb.py). Not measured with smc (#27).
stop_refused_by = ["xy_stage", "focus"]
max_segment_um = 5000.0  # chosen as m2 §11.12 D1: equal to max_jog_um; not measured
max_z_segment_um = 100.0 # chosen as m2 §11.12 D1: equal to max_z_jog_um; not measured
# z_soft_limits_um, xy_soft_limits_um and z_away_from_sample: measured at #27,
# before any move beyond the jog limit.
```

## 11. Definite Focus 2 (#26)

Reserved; fixed by `/plan 26`: DF2 as a `GenericDevice` with its own
`Autofocus` implementation, the interlock generalised to any `Autofocus`,
`stabilize_now`, and the per-position reference blobs in the run document
(ADR-0007 point 4). §6's session and §4's seam are what it builds on: its
`MTBApi` method is added then.

## 12. Not verified (for #27)

- A pymmcore-plus release with #629 and #635; the `[zeiss]` extra and the
  first session wait for it (ADR-0007 point 5).
- pythonnet's version on the stand's Python 3.14, and that `connect` loads
  MTB 2.12.0.7 there.
- Whether MTB answers a read of a component while a synchronous move of the
  same component runs (§7.2 rule 5), and how long it takes.
- That MTB has no stop for a move in progress (the profile's
  `stop_refused_by`); whether an asynchronous `MTBCmdSetModes` member
  exists.
- Whether ZEN can take the stand while `smc ui` holds the login, and
  whether MTB opens under the NSSM service (session 0) (ADR-0007).
- Whether an objective change moves the focus by itself (FM-195), and the
  focus direction away from the sample.
- The soft limits, the pixel sizes per objective, the TL shutter's element
  names as MTB reports them.
