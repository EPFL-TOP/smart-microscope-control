# M4 design — the Zeiss Axio Observer 7

- **Status**: design for issue #25 (§0 to §10, 2026-11-16, with the owner);
  §11 is #26's (Definite Focus 2), fixed by `/plan 26` on 2026-11-18; §12
  lists what the first session at the stand (#27) measures.
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

Issue #26 adds (§11):

```
src/smc/hardware/capabilities.py             (edit) §11.5: FocusReference, FocusStabilized, two Autofocus methods
src/smc/hardware/errors.py                   (edit) §11.5: AutofocusUnsupportedError
src/smc/hardware/backends/mm.py              (edit) §11.5, §11.7: MMAutofocus refuses the one-shot mode; ZInterlock runtime-checkable
src/smc/hardware/backends/mm_stabilizer.py   (new)  §11.6: MMFocusStabilizer
src/smc/hardware/microscope.py               (edit) §11.7: the build, the slot, the interlock
src/smc/hardware/backends/zeiss_mtb/         (edit) §11.3, §11.4: the seam, the .NET adapter, MTBDefiniteFocus
src/smc/testing/fake_mtb.py                  (edit) §11.3: FakeFocusStabilizer
profiles/zeiss-observer7.toml, .cfg          (edit) §11.7
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

Fixed by `/plan 26` (2026-11-18) on ADR-0007 point 4 (option F1). It
extends M2 §3 (the `Autofocus` capability) and §4 to §9 of this document,
whose rules still hold.

### 11.1 What is known

From the predecessor (`lightsheet-live-tracking-tool`, branch `zeiss`:
`mtb.py`'s `MTBDefiniteFocus`, `mtb_backend.py`, `tools/df2_test.py`) and its
run at the stand on 2026-09-11:

- Component `MTBFocusStabilizer2`; its members appear after the cast
  `IMTBFocusStabilizer2`. Learn: `InitOnCurrentFocusPosition(None, timeout)`,
  the overload with a `Byte[]&` out-parameter, which pythonnet returns as
  `(bool, Byte[])`. Restore: `InitWithStabilizingData(mode, data, timeout)`.
  One shot: `StabilizeNow(mode, timeout)`. `Stop(mode, timeout)`. Continuous:
  `StabilizePeriodically(MTBOnOff, period)`, with a period whose unit MTB
  does not document; `GetStabilizePeriodicallyOnOff()` returns a tuple, the
  on/off first. `GetOperationMode`/`SetOperationMode` take
  `MTBFocusStabilizer2OperationMode.{Exact, Balanced, Fast}`: a precision
  setting, not one shot versus continuous. `GetPiezoFocusUsage()`. `mode` is
  `MTBCmdSetModes.Synchronous`. The predecessor's timeout was 30 000 ms (the
  Micro-Manager wiki recommends 25 s).
- **A reference is an opaque blob**: 9 bytes on the stand, learned in 0.4 s
  (Balanced, with the Fluar 10x/0.50, which `ObjectiveSuitability` called
  "Suitable"). The predecessor kept one per position, as base64, keyed by
  nothing else.
- **DF2 holds a distance from the coverslip, not from the sample**, and on
  this stand it **corrects through the focus drive** (`GetPiezoFocusUsage`
  is Off): disturbed by +5.009 µm on the piezo, `StabilizeNow` moved the
  focus drive by −5.398 µm and the piezo by +0.019 µm, a residual of
  −0.370 µm on the total. The first run looked failed because it watched
  the piezo alone.
- `SignalQuality` read 0 after a successful recovery: it is no health signal.
- **Not measured**: how long `StabilizeNow` takes; whether it answers `False`
  on a success; whether `Stop` ends a synchronous `StabilizeNow` from another
  thread; how far one `StabilizeNow` may move Z; whether a reference holds
  after an objective change; what happens with no coverslip;
  `StabilizePeriodically` at all.
- The predecessor ignored the answers and swallowed the failures (the visit
  then imaged at the commanded Z). This layer does neither.

### 11.2 Scope

DF2 is a unicore `GenericDevice` (§11.4), since unicore has no autofocus
type (§1.2), and the layer reaches it through a vendor-neutral `Autofocus`
implementation over its properties, `MMFocusStabilizer` (§11.6).
`Autofocus` gains `learn_reference` and `stabilize_now` (§11.5).

**One shot only.** In smc, DF2 never holds focus on its own: its periodic
mode is switched off, never on. Continuous stabilisation, as an
engage/lock, is a later issue that measures the period first.

Not in #26: `SetFocusPositionOffset` (an offset on a reference, not verified
at the stand), `FindFocus` (it finds the coverslip, not the sample),
`ObjectiveSuitability`, the tool that visits positions with references and
its "who owns Z" choice (§11.8), the web UI, and the `AutoFocusDevice`
proposal upstream (ADR-0007 point 4: in parallel, not a dependency).

### 11.3 The seam: `api.py`, `dotnet.py`, the fake

```python
STABILIZER_TIMEOUT_MS = 30_000


class MTBFocusStabilizer(Protocol):
    mtb_id: str

    # InitOnCurrentFocusPosition: the reference blob
    def learn(self, timeout_ms: int) -> bytes: ...
    # InitWithStabilizingData: MTB's own answer
    def restore(self, reference: bytes, timeout_ms: int) -> bool: ...
    # StabilizeNow: MTB's own answer
    def stabilize_now(self, timeout_ms: int) -> bool: ...
    def stop(self, timeout_ms: int) -> None: ...
    # "Exact", "Balanced" or "Fast"
    def operation_mode(self) -> str: ...
    def set_operation_mode(self, mode: str) -> None: ...
    def piezo_focus_usage(self) -> bool: ...
    # GetStabilizePeriodicallyOnOff, its first item
    def periodic(self) -> bool: ...
    # StabilizePeriodically(Off, 0)
    def stop_periodic(self, timeout_ms: int) -> None: ...
```

`MTBApi` gains `focus_stabilizer(mtb_id) -> MTBFocusStabilizer` and
`MTBSession` the same method, cached by id as the other components are.

- `learn` returns the blob from the out-parameter; no blob, or an empty
  one, is an `MTBError`. `restore` and `stabilize_now` return MTB's answer:
  the device judges it (§11.4). `operation_mode` returns the member's name;
  `set_operation_mode` takes the member by name (`getattr`, never an int,
  §1.1), and a name that is not one of the three is an `MTBError` listing
  them.
- `dotnet.py`: `DotNetFocusStabilizer`, ported from `MTBDefiniteFocus`
  **with its tests** (the predecessor's `FakeStabilizer`, `tests/test_mtb.py`).
  The cast is to `IMTBFocusStabilizer2` only: the predecessor's fallback to
  `IMTBAutoFocus` is not ported, and a stand without DF2 fails
  `initialize()` naming the cast. The argument shapes are the predecessor's,
  tried in its order; when every shape fails, the `MTBError` names the call
  and each shape tried, chained from the last failure (FM-113). `MTBOnOff`
  is taken as the member. Out-parameters come from the returned tuple.
- `smc.testing.fake_mtb`: `FakeFocusStabilizer`, at `FakeMTB.stabilizer`,
  id `MTBFocusStabilizer2`, refused like any id in `FakeMTB.absent`. It acts
  on `FakeMTB.axes["MTBFocus"]`: `learn` returns a new 9-byte blob at each
  call and records the focus position as that blob's plane; `restore` makes
  a blob it learned the held one (any other blob: `False`); `stabilize_now`
  moves the focus axis to the held blob's plane as `FakeAxis` moves (a
  straight line over `move_s`) and answers `False` with none held; `stop`
  ends a running `stabilize_now` where the axis is. Knobs, as §9.1's:
  `refuse_next` (answers `False`, moves nothing), `fail_next` (raises
  `MTBError`), `piezo_focus_usage = False`, `periodic_on = False`,
  `stop_ignored = False` (a `stop` that ends nothing); every call is in
  `calls`.

### 11.4 The device: `MTBDefiniteFocus`

| Class | Base | MTB id | Label in the `.cfg` | Role |
|---|---|---|---|---|
| `MTBDefiniteFocus` | `GenericDevice` | `MTBFocusStabilizer2` | `DefiniteFocus` | `autofocus` (§11.7) |

Properties, registered per instance (§1.2); unicore refuses a value outside
the allowed ones and a set of a read-only one with `ValueError` (measured,
0.18.1):

| Property | Values | Read | Set |
|---|---|---|---|
| `Action` | `Idle`, `Learn`, `StabilizeNow`, `Stop` | the action whose worker runs, else `Idle` | `Learn`, `StabilizeNow`: start it on the worker; `Stop`: rule 4; `Idle`: nothing |
| `Reference` | base64 text, or `""` | the reference this device learned or restored since `initialize()`; `""`: none | a restore, on the worker; `""` is refused |
| `OperationMode` | `Exact`, `Balanced`, `Fast` | MTB's, live | MTB's, in the call |
| `PiezoFocusUsage` | `On`, `Off`; read-only | MTB's, live | — |
| `Periodic` | `On`, `Off` | MTB's, live | `Off`: `stop_periodic`; `On`: refused (`MTBError`) |

Rules, beside §7.2's:

1. **`initialize()`** finds the component, reads the operation mode (a
   failed cast fails here), and switches `periodic` off when it reads on,
   with a WARNING (another MTB client left it on). `Reference` reads `""`:
   what DF2 held before this process is not this process's to trust (FM-118).
2. **One worker at a time** (`_Mover`, §7.2 rule 3): a learn, a stabilize and
   a restore each run on it, and one is refused while another runs
   (`MTBError` naming the action that runs); `busy()` is `True` while it
   runs and raises its failure once after (§7.2 rule 4). The set returns at
   once, and reads do not wait behind the worker (measured on a unicore
   `GenericDevice`, 0.18.1: the set returned in under 1 ms, `deviceBusy` and
   `getProperty` answered during a 0.5 s worker, `waitForDevice` returned
   after 0.51 s).
3. **What counts as done.** `Learn` calls `learn(STABILIZER_TIMEOUT_MS)`;
   `Reference` then reads the blob as base64 (ASCII). A restore decodes the
   text with `base64.b64decode(text, validate=True)` in the set, before
   anything is sent (a text that is not base64 is an `MTBError` naming the
   device), then calls `restore`: a `False` is a failure, and after any
   failed restore `Reference` reads `""`, since DF2's reference is then
   unknown. `StabilizeNow` calls `stabilize_now`: a `False` is a failure
   ("DF2 answered False to StabilizeNow"), never a success, although
   whether DF2 answers `False` on a success is not measured (§12).
4. **`Stop`** is sent in the set, not on the worker, even while the worker
   runs (unicore holds the device's lock for the set only). When a worker
   runs, it logs at WARNING that whether DF2 ends a synchronous call from
   another thread is not known (§12).
5. **`shutdown()`** sends `Stop` to a running worker and waits for it up to
   `STABILIZER_TIMEOUT_MS` (`_shutdown_mover`), then switches `periodic` off
   if it reads on. It never raises; a failure is logged at ERROR. DF2 needs
   no darkener (§8).
6. **No `SignalQuality`**: it reads 0 while DF2 works (§11.1).

### 11.5 The capability: `Autofocus` gains two methods

In `smc.hardware.capabilities`:

```python
@dataclass(frozen=True, slots=True)
class FocusReference:
    """A focal plane as the autofocus records it (m4 §11.5); opaque to the layer."""

    #: The autofocus device that learned it; another device refuses it.
    device: str
    #: The device's own bytes as base64 (ASCII): a run document holds it as text.
    data: str
    #: The objective in place when it was learned; ``None``: no turret.
    objective: str | None
```

```python
@dataclass(frozen=True, slots=True)
class FocusStabilized:
    """What one ``Autofocus.stabilize_now`` did (m4 §11.5)."""

    reference: FocusReference
    #: The focus drive before and after; ``None``: the stand has no focus drive.
    z_before_um: float | None
    z_um: float | None
```

```python
class Autofocus(Protocol):
    # ... the methods of M2 §3, unchanged ...

    def learn_reference(self) -> FocusReference:
        """Record the plane in focus now as the reference; nothing moves."""
        ...

    def stabilize_now(self, reference: FocusReference | None = None) -> FocusStabilized:
        """Bring focus to ``reference`` (else the one held) once, and leave it there."""
        ...
```

`smc.hardware.errors` gains `AutofocusUnsupportedError(HardwareError)`:
this autofocus does not hold focus that way, and nothing was sent. It is not
a `SafetyRefusedError`: no guard refused a move that the stand could make,
so a tool that skips a target on a refusal must not skip on it; a tool that
can work either way catches it and uses the other way.

- `MMAutofocus` (PFS): `learn_reference` and `stabilize_now` raise it ("PFS
  holds focus continuously: engage()"). Nothing else in it changes.
- `MMFocusStabilizer` (§11.6): `engage` raises it ("it holds focus one shot
  at a time: learn_reference(), stabilize_now()").
- A pydantic model with a `dict[str, FocusReference]` field round-trips
  through `model_dump(mode="json")` and `model_validate` unchanged (measured,
  pydantic 2.13.5), so a plugin's result holds references and the run
  document stores them (§11.8).

### 11.6 `MMFocusStabilizer` (`smc/hardware/backends/mm_stabilizer.py`)

Vendor-neutral: it knows §11.4's five properties and nothing of MTB, so any
Generic device that offers them is a focus stabilizer.

```python
class MMFocusStabilizer:  # implements Autofocus and ZInterlock
    def __init__(
        self,
        core: CMMCorePlus,
        label: str,
        executor: Executor,
        safety: Safety,
        *,
        objective_label: Callable[[], str | None],
        focus: Callable[[], ZStage | None],
    ) -> None: ...
```

`objective_label` and `focus` are the facade's, as `MMAutofocus` takes them:
the objective is read live (`getStateLabel`, never the cached `Label`,
§1.3), and the focus drive is looked up at the call.

- **Reads** go through `executor.read` and never wait behind a stabilize
  (§7.2 rule 5, FM-70): `is_engaged()` is `Periodic` reading `On`;
  `is_locked()` is `is_engaged()` (smc reads no lock flag of DF2);
  `is_in_range()` is `None`.
- **No offset.** smc sets no offset on a DF2 reference, so the offset is
  zero by construction, not by assumption: `offset_native()` is `0.0`,
  `offset_unit()` `"µm"`, `offset_limits_native()` `(0.0, 0.0)`;
  `set_offset_native(0.0)` and `offset_by_native(0.0)` return `0.0` and send
  nothing, and any other value raises `AutofocusUnsupportedError`.
  `focus_by_um(dz_um, force=…)` is a Z move, as on an autofocus that is off:
  `FocusStep("z", focus().move_by_um(dz_um, force=force), None)`; with no
  focus drive, `CapabilityMissingError` for `ZStage`.
- **`engage()`** raises `AutofocusUnsupportedError` before anything is sent.
- **`disengage()`** is a safe call (`executor.safe`): never refused, run
  while halted and in dry-run. It sets `Action` to `Stop` when it reads
  anything but `Idle`, then `Periodic` to `Off` when it reads `On`; both
  are tried, and their failures raised together after the second.
- **`learn_reference()`** is one action (`executor.do`) with the motion
  below: `Action` set to `Learn`, the wait, then `Reference` read back; a
  `""` after a learn that ended without an error is a `HardwareError`. The
  reference is held from then on. In dry-run nothing is sent and it returns
  `FocusReference(label, "", objective)`.
- **`stabilize_now(reference=None)`** is one compound
  (`executor.compound`). It is refused before anything is sent
  (`SafetyRefusedError`, not forceable) when: no `reference` is given and
  none is held; `reference.device` is not this label; `reference.data` is
  `""` outside dry-run; `reference.objective` and the objective now are both
  known and differ (FM-223: DF2 learned the plane through another
  objective); `PiezoFocusUsage` reads `On` (DF2 would correct through the
  piezo, which no role reads, so the Z readback would not see the
  correction, §11.1). Then: Z is read (`z_before_um`); when `reference`
  differs from the held one, a restore action (`Reference` set to its data,
  the wait); then the stabilize action (`Action` set to `StabilizeNow`, the
  wait, Z read back). A failed restore leaves no reference held. Every
  stabilize logs its Z change at INFO. In dry-run nothing is sent and it
  returns `FocusStabilized(reference, z, z)`.
- **The Z guard reads after, it cannot stop during.** A Z change beyond
  `[safety] max_engage_excursion_um`, or a Z outside `z_soft_limits_um`,
  raises `AutofocusExcursionError(message, z_start_um=z_before_um,
  z_um=z_um)`, whose message says that DF2 has already moved Z there and
  that the layer left it there. A stand whose profile sets neither key has
  no bound (the Zeiss profile, until #27 measures one, §12).
- **The motion** of its actions: `Motion(device=label, name=f"autofocus
  {label}", is_busy=deviceBusy(label), stop=<Action set to Stop>,
  stop_refused=False)`, with the core's timeout (`device_timeout_ms`, 60 s
  on this stand). `Microscope.stop()` reaches it through `disengage()`, as
  it reaches PFS.
- **The interlock** (`ZInterlock`): `suspend(reason)` reads `Periodic`; on
  `On` (another MTB client) it switches it off with a WARNING and returns
  `False`, since smc did not switch it on and does not switch it on again;
  on `Off` it returns `False`. `relock` is never reached after a `False`,
  and does nothing.

### 11.7 The facade, the profile and the contracts

- `ZInterlock` becomes `@runtime_checkable`. `Microscope._interlock()`
  returns the autofocus when it is a `ZInterlock`, whatever its class
  (ADR-0007 point 4); `override` logs its WARNING only for an
  implementation that is not one.
- `Microscope._build`: the `autofocus` role's device type decides. A
  `GenericDevice` gets `MMFocusStabilizer`, any other `MMAutofocus`,
  unchanged. No profile key decides it (ADR-0007, *What M1–M3 taught*,
  point 2: no backend selection).
- `_point_autofocus_slot` leaves the core's AutoFocus slot alone for a
  Generic device: `setAutoFocusDevice` refuses one with `ValueError`
  (measured, 0.18.1).
- `Role.autofocus`'s rule is unchanged: the heuristics never pick a Generic
  device as an autofocus; `[roles.assign] autofocus` names it, with the note
  the role map already writes for an assigned device of another type.
- `profiles/zeiss-observer7.cfg` gains, after the Colibri line,
  `#py pyDevice,DefiniteFocus,smc.hardware.backends.zeiss_mtb.devices,MTBDefiniteFocus`;
  `profiles/zeiss-observer7.toml` gains `autofocus = "DefiniteFocus"` under
  `[roles.assign]`, and no `[autofocus]` key. `max_engage_excursion_um`
  stays unset until #27 measures a bound (§12); its docstring in
  `profile.py` adds "or a one-shot stabilize moves it (m4 §11.6)".
- The contract stand `zeiss-fake` now has an autofocus. In
  `tests/contracts/`, a case that needs `engage()` or a non-zero offset skips
  on `AutofocusUnsupportedError` on `zeiss-fake` and `hardware` (the set
  `_SKIPS_WHAT_IT_LACKS`) and fails on it on `fake` and `demo`; the cases of
  the one-shot mode skip on it the same way. A contract holds on every
  stand: an autofocus holds focus one way or the other (`engage` or
  `stabilize_now`).

### 11.8 References in the run document, and who owns Z

- A tool that visits positions keeps one `FocusReference` per position in
  its result (`dict[str, FocusReference]`), so the run document holds them
  as ASCII text (§11.5). The layer keeps the held reference for the
  facade's life only.
- **Who owns Z** (the predecessor's `df2_owns_z`) is that tool's choice, a
  param, not a profile key: after `stabilize_now`, the tool either keeps the
  Z that DF2 left (DF2 owns Z) or adds its own correction with a Z move. The
  layer offers both calls and decides neither. The param's name and default
  come with the first tool that visits positions with references, a later
  issue; the predecessor defaulted to DF2 owning Z, and called the choice a
  scientific one, to settle by running both.

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
- Definite Focus 2 (§11.1): how long `StabilizeNow` takes, whether it
  answers `False` on a success, whether `Stop` ends it from another thread,
  how far one moves Z (then `max_engage_excursion_um`, before any
  stabilize runs unattended), whether a reference holds after an objective
  change, and the `piezo_focus_usage` reading. The stand check is the
  one-shot contracts on `hardware` (learn, a 2 µm move down, a stabilize).
