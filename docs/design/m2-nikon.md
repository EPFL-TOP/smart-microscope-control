# M2 design — Nikon on hardware

- **Status**: design for issue #13 (§1–§6, 2026-10-16); §3.3.1 and
  §3.11's stop rules decided in #144 (2026-10-18). #12, #14, #15 and
  #48 add their own sections with their plans.
- **Owner**: the design session. **Executors**: `/develop` sessions, one per issue.
- **Rule**: as in M1, this document is the contract between issues that are
  built in parallel. Names, module paths and signatures below are fixed; an
  executor who needs to change one stops and comments on its issue instead.
  Where this document extends [M1's](m1-hardware-layer.md), it says which
  section it extends, and M1's rules still hold everywhere else.

Read first: [ADR-0003](../adr/0003-capabilities-roles-and-profiles.md);
M1 §2 (conventions), §3, §6, §7, §8 and §13 (motion, stop and the lock);
[`docs/hardware/nikon-ti2.md`](../hardware/nikon-ti2.md), *Session 0*.

## 0. Scope

M2 brings the Nikon stands under the layer: the `Autofocus` capability and
the Z interlock (#13), `ObjectiveTurret`, `LightSource` and `Channels`
(#14), the stand knowledge from `nikon-control` (#12), the configuration
builder (#15) and a truthful probe (#48). This version fixes #13 only.

Not in #13: one-shot autofocus (`fullFocus`), a CLI command for the
autofocus, the piezo (§2), camera-clocked sequences, and the Zeiss
Definite Focus 2 (M4, ADR-0007). The Protocol below does not preclude
DF2: its continuous mode is an engage/lock, and its one-shot
`stabilize_now` can be added later as a method of its own.

## 1. What is known about focus on the Ti2-E

From session 0 (#126, `nikon-ti2.md`), unless another source is named.
Every line is measured unless it says otherwise.

- **Three things move focus.** `ZDrive` (`Stage`), `PFSOffset` (`Stage`,
  the PFS offset lens, read 7803.0 in device units) and a Mad City Labs
  piezo insert driven open loop by the NI card's `AO 0`. The piezo is in
  no configuration, and its position is never read back (its `sensor`
  output is not connected).
- **PFS** is an `AutoFocus` device labelled `PFS`, and the core's
  AutoFocus slot names it in the lab's configuration. Its properties, read
  with PFS off: `FocusMaintenance` `Off`/`On` (writable), `PFS in Range`
  `Out of Range` (read-only), `PFS Status` `0000000000000000` (read-only),
  `DichroicMirrorInserted` `Yes`, `LEDIntensity` 40. What they read with PFS
  locked is **unknown** (§4).
- Neither `ZDrive` nor `PFSOffset` reports a focus direction (0, unknown).
  MMCore offers `PFSOffset` as a `Core,Focus` device too; the `focus` role
  already excludes names containing `pfs` or `offset` (M1 §4).
- **Moving Z switches PFS off** on the Nikon stands (micro-manager #1815,
  FM-12). Known from `nikon-control` (`inventory.md`, *Both Nikon
  stands*); not re-measured on this Ti2-E.
- **pymmcore-plus redirects `setAutoFocusOffset` to the offset stage only
  for `NikonTI` and `NikonTE2000`**, not `NikonTi2` (read from
  `_OFFSET_DEVICES` in pymmcore-plus 0.18.1). The layer therefore never
  relies on `setAutoFocusOffset` when an offset device is resolved.
- **`nikon-control`** (read 2026-10-16): `engage_pfs` waits 5 s by default
  (a figure never measured), polls every 50 ms, returns `False` on no lock
  and **leaves PFS switched on**. `move_z` re-engages PFS after a Z move and
  ignores whether it locked. `focus_by(dz)` sends `dz` to the offset device
  when PFS is on and to Z otherwise, so one argument has two units. On the
  Ti-E, offset positions are 1/40 of the driver's raw steps, and the valid
  range depends on the objective.
- **The demo** (measured 2026-10-16, pymmcore-plus 0.18.1): `Autofocus`
  (`DAutoFocus`) fills the `autofocus` role from the core slot; no device
  fills `autofocus_offset`. Continuous focus starts off; enabling it reads
  enabled and locked at once. `setAutoFocusOffset(12.5)` reads back 12.5
  through `getAutoFocusOffset()`, enabled or not, and moves nothing. A Z
  move leaves it enabled. `deviceBusy("Autofocus")` reads `False`. With the
  AutoFocus slot empty, `isContinuousFocusEnabled()` and
  `isContinuousFocusLocked()` read `False`, `getAutoFocusOffset()`,
  `setAutoFocusOffset()` and `fullFocus()` raise `RuntimeError('Error (code
  43)')`, and `enableContinuousFocus(True)` raises `RuntimeError('Auto-focus
  focus device not defined.')`.

## 2. The focus axes in M2 (decided)

- **`focus` is the Z drive** (`ZDrive` on the Ti2-E, `TIZDrive` on the
  Ti-E). `ZStage` is that role, in µm. "Z" in a run document is the
  `focus` role's position.
- **PFS corrects through the Z drive.** That is Nikon's design; the PFS
  session confirms it (§4).
- **The PFS offset is `autofocus_offset`**, in its own units, never µm
  (§3).
- **The piezo has no role in M2.** No route to it has been tried (the
  `NIDAQ` analog output wrapped by `Utilities`' DA Z Stage is one), and a
  device that never reads back cannot honour "mutating calls return the
  readback" (M1 §2) as it stands. Whether it becomes a second focus role,
  and which device a fast Z-stack then moves, is decided in #135.
- A Z-stack in M2 moves the `focus` role with the autofocus suspended
  (`keep_autofocus`, §3.4); how acquisitions use it is M3's.

## 3. Autofocus (#13)

### 3.1 Protocol and value type — `capabilities.py`

```python
@dataclass(frozen=True, slots=True)
class FocusStep:
    via: Literal["z", "offset"]  # what moved
    z_um: float | None  # the focus drive's position afterwards; None: no focus drive
    offset_native: float | None  # the offset afterwards when via == "offset", else None


@runtime_checkable
class Autofocus(Protocol):
    def is_engaged(self) -> bool: ...  # continuous focus switched on
    def is_locked(self) -> bool: ...  # holding focus now
    def is_in_range(self) -> bool | None: ...  # None: the stand does not say (§3.9)
    def engage(self, timeout_s: float | None = None) -> bool: ...
    def disengage(self) -> None: ...
    def offset_native(self) -> float: ...
    def set_offset_native(self, value: float) -> float: ...
    def offset_by_native(self, delta: float, *, force: bool = False) -> float: ...
    def offset_unit(self) -> str: ...  # the profile's label; "" = unknown
    def offset_limits_native(self) -> tuple[float, float] | None: ...
    def focus_by_um(self, dz_um: float, *, force: bool = False) -> FocusStep: ...
```

`ZStage` gains one keyword on each move (M1 §3 announced it; a keyword
with a default is a compatible extension):

```python
def move_to_um(self, z_um: float, *, keep_autofocus: bool = True) -> float: ...
def move_by_um(
    self, dz_um: float, *, force: bool = False, keep_autofocus: bool = True
) -> float: ...
```

`Autofocus` and `FocusStep` join `__all__`. **Names.** ADR-0003's
`engaged`, `locked`, `in_range`, `offset` and `focus_by(delta)` become
methods (M1 §2: anything that talks to hardware is a method), `is_*` like
`is_busy()` and `is_open()`. A value in the offset device's own units ends
in `_native`, so that no caller can take it for µm (ADR-0003: named for
what it is, never converted implicitly). `focus_by_um` takes µm, never a
native delta (§3.5).

### 3.2 Semantics (the contract tests)

- `is_engaged()` is `isContinuousFocusEnabled()`; `is_locked()` is
  `isContinuousFocusLocked()`. Both read the core's AutoFocus slot (§3.6).
- `is_in_range()` reads the property the profile names (§3.9) and returns
  whether its value is one of the profile's in-range values. Without that
  profile entry it returns `None`: the layer never guesses which property
  or value means "in range".
- **`engage(timeout_s)`** switches continuous focus on and waits for the
  lock, up to `timeout_s` (default: the profile's `lock_timeout_s`, else
  `DEFAULT_LOCK_TIMEOUT_S = 5.0`, assumed and labelled as such). It returns
  `True` once locked. **It returns `False` when no lock came, and then the
  autofocus is off**: the wait gives up like a move, so it sends the
  autofocus's stop, which switches it off (FM-17). `False` therefore always
  means "no lock, and it is off". A WARNING names the timeout (and whether
  it was assumed) and `is_in_range()`. If the switch-off itself fails, or the
  autofocus still reads engaged afterwards, `engage` raises the
  `DeviceTimeoutError` instead, and the motion stays registered (M1 §13), so
  the stand refuses actions until `stop()` or `resume()`. `timeout_s` must be
  finite and in (0, `MAX_LOCK_TIMEOUT_S` = 600] s, else `ValueError` before
  anything is sent (FM-32): the wait holds the microscope lock.

  This differs from `nikon-control`, which leaves PFS on after a failed
  lock. A PFS left searching can move Z with no action holding the lock and
  no motion registered, which is what M1 §13 forbids, and an off state is
  the same on every stand and testable.
- **`disengage()`** switches continuous focus off. It is a *safe call*,
  like closing a shutter (M1 §13): it never takes the lock, is never
  refused, runs while halted and in dry-run (switching off cannot create
  motion), and an `engage` already waiting for its lock is cancelled with
  `MotionStoppedError`.
- **The offset.** `offset_native()` reads the `autofocus_offset` device's
  position when that role is filled, else the autofocus device's own offset
  (`getAutoFocusOffset()`, the demo's case). `set_offset_native(value)`
  moves it and waits like a stage move, and returns the readback.
  `offset_by_native(delta, *, force)` is the relative move. The guards
  mirror Z's (M1 §3, §7):
  - a relative change above `[safety] max_autofocus_offset_jog` is refused
    unless `force=True`. When the profile does not set that key, **every
    non-zero relative change needs `force=True`**: the unit is the stand's
    own, so no default can be assumed (unknown beats guessed). The refusal
    names the key;
  - a target outside `[safety] autofocus_offset_soft_limits` is refused and
    cannot be forced; absolute changes are not jog-guarded, as for Z;
  - a non-finite value or delta is refused and cannot be forced.

  With the autofocus locked, an offset change moves Z: the autofocus
  follows the new offset. With it off, it moves only the offset lens
  (assumed for the Ti2-E, §4).
- `offset_unit()` is the profile's `[autofocus] offset_unit`, `""` when
  unknown; messages then say "device units". `offset_limits_native()`
  is the profile's soft limits, or `None`.
- `focus_by_um`: §3.5. The Z interlock: §3.4.
- Every reader propagates a failure as a `HardwareError` naming the role
  and the call (M1 §2, FM-79); the facade's `state()` is the tolerant layer.

### 3.3 Kinds of call (extends M1 §13)

| Kind | Calls |
|---|---|
| read | `is_engaged`, `is_locked`, `is_in_range`, `offset_native`, `offset_unit`, `offset_limits_native` |
| action | `engage` (a motion: the wait for the lock), `set_offset_native` and `offset_by_native` (a motion: the offset device), each run as a compound of that one motion so that the Z stop reaches it (§3.3.1); `focus_by_um` and a Z move on a stand with an autofocus (a *compound*: several nested actions in one lock section, §3.7) |
| safe | `disengage` |

The lock wait is a **motion** in the sense of M1 §13: `Motion(device=<af
label>, name="autofocus <label>", is_busy=lambda: enabled and not locked,
stop=<switch it off>)`. So, without new machinery, another thread's action
during the wait is refused at once with `MotionInProgressError` naming
`autofocus <label>`; a Ctrl-C or a fault during the wait sends the stop;
an unreadable state counts as busy (FM-19); and `disengage()`, keyed by
the same label, cancels it (FM-18). The offset device's move is an ordinary
stage motion keyed by its own label. Once locked, the autofocus correcting
Z is **not** a motion: the layer would otherwise refuse every action while
PFS holds focus. Measured with a prototype on a `FakeCore` (2026-10-16):
the give-up sends `enableContinuousFocus(False)` and registers nothing, a
`SafeCall` on the AF label during the wait raises `MotionStoppedError`,
another thread's action is refused at once, and in dry-run nothing is sent.

#### 3.3.1 What each stop reaches (FM-91, #144)

Stops are keyed by device (M1 §13), and PFS drives Z under keys of its own
(`PFS`, `PFSOffset`). So every motion that can drive Z names the stops that
end it. **Rule: while PFS can drive Z because of an action, the operator's
`ZStage.stop()` switches it off, first.** Measured on the Ti2-E fake before
this decision (2026-10-18): a `ZStage.stop()` during a direct `engage()`
search was ignored, and PFS searched on until `lock_timeout_s` ran out. One
during a direct offset move with PFS locked was ignored too: the move
completed and PFS stayed on, following it.

| Motion (its key) | Its own stop | `ZStage.stop()` | `Microscope.stop()` | `close()`, something moving |
|---|---|---|---|---|
| Z move (`ZDrive`) | Z's | Z's | yes | yes |
| Relock after an interlocked Z move (`PFS`) | `disengage()` | PFS off, then Z's stop (#140) | yes | yes |
| Locked `focus_by_um`: the offset move and the relock | the offset device's; `disengage()` | PFS off, offset device stopped, then Z's stop (#140) | yes | yes |
| A direct `engage()`'s lock wait (`PFS`) | `disengage()` | **PFS off, then Z's stop** | yes | yes |
| A direct `set_offset_native` / `offset_by_native`, PFS on (`PFSOffset`, or `PFS` without an offset device) | the offset device's | **PFS off, offset device stopped, then Z's stop** | yes | yes |
| The same move, PFS off | the offset device's | no: it moves no Z (assumed, §4 item 5) | yes | yes |
| PFS holding focus, no action running | (not a motion) | no | PFS off | no: left as it is |

"yes" in the last two columns: the emergency stop's order (§3.11), PFS off
first, then the offset device, then the stages.

- **Mechanism.** `engage`, `set_offset_native` and `offset_by_native` each
  run as one compound (§3.7). Inside it, the call links the focus drive's
  stop (`Executor.linked_stop`) for as long as PFS can drive Z, exactly as
  the relock and a locked `focus_by_um` do (`MMAutofocus._ended_by_a_z_stop`).
  `engage` always links. An offset move links when `is_engaged()` reads
  `True`, *or cannot be read* (unknown counts as on, FM-19): with PFS on but
  not locked, an offset change can still steer its search. The read is made
  inside the compound, under the microscope lock, where no other action can
  switch PFS on. Nothing is linked when the focus drive is not this module's
  `MMZStage` (`Microscope.override`): there is no stop of it to link.
- **Outcome of a Z stop once the call has sent something.** PFS is switched
  off, and the offset device is stopped while its move is pending or under
  way, before Z's own stop and inside one `hold_log` (FM-62). The call raises
  `MotionStoppedError` naming Z, e.g. `` `z ZDrive`: it was stopped during
  autofocus: engage, so autofocus PFS was left off; read the position before
  moving on ``, and the WARNING `autofocus PFS: left off, because autofocus:
  engage did not complete (MotionStoppedError)` is logged. `engage` keeps its
  contract: `False` means "no lock, and it is off". A Z stop is a raise, never
  a `False`. When the Z stop's switch-off itself fails, no "left off" is
  claimed; the error says PFS may still move Z (as for the relock).
- **A Z stop that came while the call waited for the microscope lock**
  cancels it with nothing sent (`MotionStoppedError`, "it was stopped before
  it started"). PFS is then as it was, so no "left off" WARNING is logged.
  This is M1 §13's rule (a stop cancels the actions on its device called
  before it), applied to calls that drive Z through another device.
- **Not chosen.** *Refusing the stop*: a stop is never refused (M1 §13).
  *Cancelling the lock wait but leaving PFS on*: PFS left searching moves Z
  with no action holding the lock, which §3.2 forbids. *Linking the offset
  move with PFS off*: the Z stop would stop a lens move that moves no Z, and
  report PFS "left off" when it was never on. If #16 finds that an offset
  move with PFS off does move Z (§4 item 5), the condition becomes "always".
- **Halted, no link is called** (#140): the emergency stop sends every stop
  itself, in its own order. `close()` keeps the links (§3.11).

### 3.4 The Z interlock (FM-12)

On a stand whose `autofocus` role is filled, `ZStage.move_to_um` and
`move_by_um` run as one compound (§3.7), in this order:

1. The jog guard (relative) and, for an absolute move, the soft limits,
   before anything else, as today: **a refused move never touches the
   autofocus.** A relative move reads the position and checks its target
   inside the compound, before step 2.
2. `suspend`: if `is_engaged()`, switch it off as a nested action (logged
   `autofocus: suspend for <move>`; skipped in dry-run, unlike
   `disengage()`). Nothing is sent when it is off.
3. The Z move, as in M1 (a nested motion action).
4. `relock`, only if it was engaged, `keep_autofocus` is true **and the move
   returned normally**: `engage()` with the profile's lock timeout. No lock
   raises `AutofocusLostError` (§3.10), naming where Z landed, the timeout
   and `is_in_range()`; the autofocus is then off. A Z move under PFS that
   silently leaves PFS off is the "PFS off overnight" that ADR-0003 names.
5. The return value is the Z readback of step 3, before the relock: once
   the autofocus holds focus again, `position_um()` reads where it holds it.

**No relock after a failure** (FM-88). If step 3 raises (a stop,
`MotionStoppedError`, a halt, a timeout, a fault, a Ctrl-C), the autofocus
stays off, a WARNING says so (`autofocus <label>: left off, because <move>
did not complete`), and the exception propagates. Re-engaging would move Z
after the operator's stop. With `keep_autofocus=False`, steps 1–3 run, the
autofocus stays off, and an INFO line says so.

On a stand with no autofocus role, a Z move is exactly M1's: no compound,
no extra read. A stand whose autofocus is overridden (`Microscope.override`,
a testing hook) is not interlocked; `override` logs a WARNING saying so.

### 3.5 `focus_by_um` — one call, one unit

`focus_by_um(dz_um, *, force=False)` changes focus by `dz_um` µm, through
whatever holds focus now. It runs as one compound:

1. Before the compound: Z's jog guard on `dz_um` (`max_z_jog_um`, M1 §7),
   forceable as for Z.
2. Autofocus **off**: `ZStage.move_by_um(dz_um, force=force,
   keep_autofocus=False)`. Returns `FocusStep("z", <readback>, None)`.
   No `focus` role: `CapabilityMissingError` naming `focus`.
3. **On but not locked**: refused (`SafetyRefusedError`, not forceable):
   `autofocus <label> is on but not locked (in range: no); wait for the
   lock (engage()), or disengage() it to focus with Z`.
4. **Locked**: the change goes to the offset, because a Z move would be
   undone by the relock. `delta = dz_um × [autofocus] offset_per_um[<current
   objective label>]`, a factor **measured** per objective, sign included
   (§4). No factor for the current objective, or no objective turret
   resolved: refused, not forceable, naming the objective and the key, and
   pointing at `offset_by_native()`. Then `offset_by_native(delta,
   force=force)` (its own guards apply to `delta`), then a relock as in
   §3.4 step 4 (`AutofocusLostError` on no lock). Returns
   `FocusStep("offset", <Z read afterwards, or None>, <offset readback>)`.

`nikon-control`'s `focus_by` sent one number to two devices in two units.
Here the unit is always µm, and the only conversion is a factor the
profile states as measured. Where it is missing, the call refuses rather
than guesses, and `offset_by_native()` remains for a tool that works in
the device's units.

### 3.6 Backend — `backends/mm.py`

```python
# nikon-control's figure, never measured: labelled as assumed wherever it shows
DEFAULT_LOCK_TIMEOUT_S = 5.0
MAX_LOCK_TIMEOUT_S = 600.0


class ZInterlock(Protocol):  # internal: what MMZStage needs; MMAutofocus implements it
    # True: it was engaged and is now off
    def suspend(self, reason: str) -> bool: ...

    # engage with the profile's timeout; no lock: AutofocusLostError
    def relock(self, reason: str, *, z_um: float | None) -> None: ...


class MMAutofocus:  # implements Autofocus and ZInterlock
    def __init__(
        self,
        core: CMMCorePlus,
        label: str,
        executor: Executor,
        safety: Safety,
        *,
        offset_label: str | None,
        offset_unit: str = "",
        lock_timeout_s: float | None = None,  # None: DEFAULT_LOCK_TIMEOUT_S, assumed
        offset_per_um: Mapping[str, float] = {},
        in_range_property: str = "",
        in_range_values: Sequence[str] = (),
        objective_label: Callable[[], str | None] = lambda: None,
        focus: Callable[[], ZStage | None] = lambda: None,
    ) -> None: ...


class MMZStage:  # gains one keyword
    def __init__(
        self,
        core: CMMCorePlus,
        label: str,
        executor: Executor,
        safety: Safety,
        *,
        autofocus: Callable[[], ZInterlock | None] = lambda: None,
    ) -> None: ...
```

Core calls (extends M1 §6's table):

| Method | MMCore |
|---|---|
| `is_engaged` / `is_locked` | `isContinuousFocusEnabled()` / `isContinuousFocusLocked()` |
| `is_in_range` | `getProperty(label, in_range_property)`, compared with `in_range_values` |
| `engage` | `enableContinuousFocus(True)`, then the lock wait (§3.3), then `isContinuousFocusLocked()` |
| `disengage` | `enableContinuousFocus(False)` through `Executor.safe`, `SafeCall(device=label, …)` |
| `offset_native` | `getPosition(offset_label)`, else `getAutoFocusOffset()` |
| `set_offset_native` / `offset_by_native` | `setPosition(offset_label, v)`, waited as `_motion(…, offset_label, "autofocus_offset <offset_label>", stoppable=True)`; else `setAutoFocusOffset(v)`, waited as `Motion(device=label, name="autofocus <label> offset", is_busy=deviceBusy(label), stop=None)` |

- **The slot (FM-87).** The continuous-focus and autofocus-offset calls
  take no label: they act on the device the core's AutoFocus slot names.
  Every `MMAutofocus` call that does so, reads included, first checks that
  `getAutoFocusDevice()` is still `label` and raises `HardwareError`
  otherwise, as `MMCamera` does for the camera (M1 §6). The facade points
  the slot at the role once, when it is built (§3.11).
- **Faults (FM-79).** Calls on the autofocus device go through
  `as_hardware_errors(core, "autofocus")`, calls on the offset device
  through `as_hardware_errors(core, "autofocus_offset")`.
- The lock-wait timeout is passed to `Executor.do` as `timeout_s`; the
  offset move uses the core timeout, as a stage move does.
- `engage` catches the `DeviceTimeoutError` of its own lock wait only, then
  reads `is_engaged()`: off → WARNING and `False`; still on, or unreadable →
  re-raise. A readback of "not locked" after the wait ended idle (the
  autofocus switched itself off) is also `False`, with a WARNING saying it
  switched itself off. Dry-run: `engage` returns `True` (the commanded
  outcome, M1 §2), and nothing is sent.
- `suspend` is a nested `Executor.do` (no motion) whose step sends
  `enableContinuousFocus(False)`; `relock` runs `engage`'s action and turns
  a `False` into `AutofocusLostError`. Both run only inside a compound.

### 3.7 `Executor.compound` — `safety.py`

```python
def compound(self, description: str, run: Callable[[], T]) -> T: ...
```

One lock section around several nested actions: under the lock, the halt
check and the guard run once (as `acquire` does), then `run()`. Each nested
action runs its own halt check, guard, dry-run handling and motion wait
through M1 §13's re-entry, so `compound` itself sends nothing and runs in
dry-run too. A nested action inside a *motion's command* is still refused
(M1 §13); nothing in §3.4 or §3.5 nests there. `acquire` keeps its meaning
(a frame) and may share the implementation.

### 3.8 Safety — `safety.py`

```python
class Safety:
    def __init__(self, *, ..., max_autofocus_offset_jog: float | None = None,
                 autofocus_offset_soft_limits: tuple[float, float] | None = None,
                 offset_unit: str = ""): ...
    max_autofocus_offset_jog: float | None  # None: every non-zero offset jog needs force
    autofocus_offset_soft_limits: tuple[float, float] | None
    def check_offset_jog_native(self, delta: float, *, force: bool) -> None
    def check_offset_target_native(self, value: float) -> None
```

Refusals (`SafetyRefusedError`), with `<unit>` the profile's
`offset_unit` or `device units`:
- `autofocus offset jog 1.5 <unit> exceeds the offset jog limit of 1.0
  <unit>`, `how_to_force="pass force=True"`;
- with no limit set: `autofocus offset jog 1.5 <unit>: the profile sets no
  [safety] max_autofocus_offset_jog, so every relative offset change needs
  force=True`, `how_to_force="pass force=True, or set [safety]
  max_autofocus_offset_jog in the profile"`;
- non-finite, or a target outside the soft limits: an empty `how_to_force`.

### 3.9 Profile — `profile.py`

```toml
[safety]
# max_autofocus_offset_jog = 50.0                # in [autofocus] offset_unit; omitted: every relative offset change needs force=True
# autofocus_offset_soft_limits = [0.0, 15000.0]  # in [autofocus] offset_unit; not forceable

[autofocus]
offset_unit = ""          # what the offset's numbers are ("PFS offset steps"); "" = unknown, shown as "device units"
# lock_timeout_s = 5.0    # engage() and the Z interlock wait this long for a lock; omitted: 5 s, assumed
offset_per_um = {}        # objective label -> offset change per µm of focus, measured with the autofocus locked; sign included
in_range_property = ""    # a property of the autofocus device that says whether its reference is in range; "" = the stand does not say
in_range_values = []      # the values of that property that mean "in range"; any other value means out of range
```

```python
class AutofocusSection(_Section):  # extra="forbid", like the other sections
    offset_unit: str = ""
    lock_timeout_s: float | None = None
    offset_per_um: dict[str, float] = {}
    in_range_property: str = ""
    in_range_values: list[str] = []


class SafetySection(_Section):  # gains, after its M1 keys
    max_autofocus_offset_jog: float | None = None
    autofocus_offset_soft_limits: tuple[float, float] | None = None


class Profile(BaseModel):  # gains
    autofocus: AutofocusSection = AutofocusSection()
```

Validation, each a `ProfileError` naming the key (M1 §5):
- `lock_timeout_s`: finite, in (0, 600];
- `max_autofocus_offset_jog`: positive and finite;
- `autofocus_offset_soft_limits`: ordered and finite (`_check_ordered`);
- each `offset_per_um` value: finite and non-zero, the message naming the
  objective (`autofocus.offset_per_um."Nikon 40X …"`);
- `in_range_property` and `in_range_values`: both set or both empty, and no
  empty value;
- **a TOML boolean is refused** for every new numeric key, while an integer
  (`100`) is accepted: pydantic's lax mode reads `true` as `1.0` (#106, which
  fixes the existing keys; the new ones do not add to it).

`Profile.demo()` and `profiles/demo.toml` stay equal: the demo sets no
`[autofocus]` value and no offset key (the commented lines above go into
`demo.toml`). `profiles/README.md` documents every new key. The Ti2-E and
Ti-E values come from the session in §4, through #12's profiles.

### 3.10 Errors — `errors.py`

```python
class AutofocusLostError(HardwareError):  # not a SafetyRefusedError (FM-64)
    z_um: float | None  # where the focus drive was read after the move
    in_range: bool | None
```

Message: `z: moved to 12.00 µm, but autofocus PFS did not lock again within
5 s (assumed: the profile sets no [autofocus] lock_timeout_s) and is off
(in range: no); engage it (engage()) once the sample is in range, or move Z
with keep_autofocus=False`. From `focus_by_um`, it names the offset instead
of the Z move. The CLI prints it as any `HardwareError` (exit 1).

### 3.11 Facade — `microscope.py`

- `CAPABILITY_ROLES[Autofocus] = (Role.autofocus,)`, inserted before
  `Properties`; `available()` on the demo becomes `[XYStage, ZStage,
  Camera, Shutter, Autofocus, Properties]`.
- `_build(Autofocus)`: `MMAutofocus(core, label, executor, safety,
  offset_label=roles.get(Role.autofocus_offset), …profile.autofocus…,
  objective_label=self._objective_label(), focus=lambda: self.get(ZStage))`.
- `_build(ZStage)`: `MMZStage(…, autofocus=self._interlock)`, where
  `_interlock()` returns `self.get(Autofocus)` when it is an `MMAutofocus`,
  else `None`. Lazy on both sides, so neither build calls the other.
- `Safety` gets the two new `[safety]` keys and `offset_unit`.
- **The slot (FM-87).** When the `autofocus` role is filled and the core's
  AutoFocus slot names another device (or none), the facade sets the slot
  to the role's label at construction and logs it at INFO
  (`autofocus: the core's AutoFocus slot now names 'PFS' (was '')`). The
  role's order of authority (profile, then core slot, then heuristics,
  ADR-0003) decides which device that is.
- **The emergency stop** (`stop()`, and `close()` when something moves,
  through `_stop_stages`) switches the autofocus off **first**, then stops
  the stages, past any failure, every call before any log line (FM-62). A
  Z stop sent while PFS still drives Z can be overridden by its next
  correction. A failure is one more entry in the `HardwareError` that names
  each failure. `close()` with nothing moving leaves the autofocus as it is:
  PFS holding focus after smc exits is a legitimate state.
- **The offset device is stopped next** (#140): right after the switch-off
  and before the stages, by its `autofocus_offset` role, whether or not the
  autofocus was built or overridden. Its failure entry is
  `autofocus_offset 'PFSOffset' (<exc>)`.
- **The links (§3.3.1).** While halted, `Executor.stop` calls no link, so
  `stop()` sends each stop once, in the order above. **`close()` keeps the
  links** (decided in #144): it does not halt, so when a compound of the
  autofocus is running, `_stop_stages`'s Z stop calls that compound's link
  as well. Measured on the Ti2-E fake during a relock (2026-10-18): `close()`
  sent `enableContinuousFocus(False)`, `stop('PFSOffset')`, `stop('XYStage')`,
  `enableContinuousFocus(False)`, `stop('ZDrive')`. The second switch-off
  (and a second `stop('PFSOffset')` while an offset move is pending) repeats
  what was sent just before, in the same order, and is idempotent. In
  exchange, the compound knows a Z stop ended it: its caller gets
  `MotionStoppedError` naming Z, and the "left off" WARNING is logged. A
  failing link is then logged a second time, under the Z entry, with its
  own device in the message. Skipping the links would need a second
  mechanism next to the halt, for no change in what the stand does.
- **`state()`** gains `autofocus_engaged: bool | None`, `autofocus_locked:
  bool | None` and `autofocus_offset_native: float | None` (each `None`
  with an `errors` entry when its read fails, and `None` without an
  autofocus). `describe()` adds, after the shutter, `autofocus locked`,
  `autofocus on, not locked`, `autofocus off` or `autofocus ?`.
- `override(Autofocus, impl)` logs a WARNING that Z moves no longer
  suspend and re-engage it (§3.4).

### 3.12 FakeCore — `smc/testing/fakes.py` (extends M1 §8)

Measured on the demo (§1), unless labelled:
- `getAutoFocusOffset()` / `setAutoFocusOffset(offset)` act on the slot's
  device and store one float, `0.0` at construction; the set is logged.
  With the slot empty, both raise `RuntimeError('Error (code 43)')`.
- `enableContinuousFocus(True)` with the slot empty raises
  `RuntimeError('Auto-focus focus device not defined.')`;
  `isContinuousFocusEnabled()` and `isContinuousFocusLocked()` read
  `False`.
- `failing[<slot's label>]` makes the continuous-focus and offset calls
  raise too, as the current-camera calls do for the camera slot.
- `autofocus_lockable: bool = True`: when `False`, the autofocus never
  reads locked, even enabled (no reference in range). Locked whenever
  enabled otherwise, as today (assumed).
- `demo_like()` is unchanged: the demo has no offset device.

A test that needs the Ti2-E's layout builds its own `FakeCore` from the
session 0 labels: `ZDrive` (focus slot), `PFS` (`AutoFocus`, autofocus
slot, with a read-only `PFS in Range` property), `PFSOffset` (`Stage`),
`XYStage`, `Camera-1` and a `Nosepiece` `State` device with objective
labels. The prototype above built one with `FakeCore(devices, camera=…,
xy_stage=…, focus=…, autofocus=…)`, and it resolved `autofocus_offset` to
`PFSOffset`.

### 3.13 Tests

- **Contracts** (`tests/contracts/test_autofocus.py`, through `Microscope`
  only, on `fake`, `demo` and `hardware`). The envelope (M1 §8) also
  records whether the autofocus was engaged and its offset at the start,
  and its `[safety]` adds `max_autofocus_offset_jog = 1.0` and
  `autofocus_offset_soft_limits` at the start ± 2.0, which must lie inside
  the profile's own (FM-48). A contract moves the offset at most 1.0 from
  the start, with the autofocus off, except in the locked `focus_by_um`
  case. The teardown restores, in order: shutter closed, autofocus off
  (`disengage`), XY, Z (`keep_autofocus=False`), offset, exposure,
  auto-shutter, then the autofocus engaged if it was (no lock is a named
  teardown failure, FM-49), then the shutter. On `hardware`, a case that
  needs a lock skips when `engage()` returns `False` ("no lock on this
  sample"); on `fake` and `demo` it fails.
- **FakeCore** (`tests/unit/test_autofocus.py`, the Ti2-E layout above,
  through `Microscope.from_core`): the interlock with the quirk on
  (`test_moving_z_re_engages_pfs`), every refusal and failure path of §3.4
  and §3.5, the slot, the offset device, the emergency stop, threading.
- **Demo** (`tests/test_autofocus_demo.py`): what only the demo shows
  (the role from the core slot, no offset device, the offset through
  `DAutoFocus`, `state()`), beside the contracts' `demo` backend.

## 4. What the PFS session must read (#16)

Nothing in §3 assumes these; each has a key or a test waiting for it.
`/hardware-session` turns them into steps for the Ti2-E, then the Ti-E.
The sample is a dish or plate with cells, focused by eye first.

1. **Lock.** PFS off, focused by eye. `enableContinuousFocus(True)`, then
   every 100 ms for 5 s: `isContinuousFocusEnabled()`,
   `isContinuousFocusLocked()`, `FocusMaintenance`, `PFS in Range`,
   `PFS Status`, `getPosition("ZDrive")`, `deviceBusy("ZDrive")`,
   `deviceBusy("PFS")`. Gives the lock time (`lock_timeout_s`), the
   in-range value (`in_range_values`), and whether `isContinuousFocusLocked`
   tells the truth. Then, locked, `enableContinuousFocus(True)` once more:
   does it disturb the lock? (`engage` and the relock send it again.)
2. **No reference.** The same with the objective far from the sample, or
   without a sample: what reads "out of range", and **whether Z moves while
   PFS searches**.
3. **#1815 on the Ti2-E.** PFS locked, `setPosition("ZDrive", z + 5)`: is
   PFS switched off?
4. **Offset under lock.** PFS locked: `setPosition("PFSOffset", o + N)` for
   N = 10, then 50, and back; Z before and after each, the time to lock
   again, `deviceBusy("PFSOffset")` during the move. Per objective in use:
   the factor `offset_per_um = N / ΔZ`, its sign, and a safe
   `max_autofocus_offset_jog`.
5. **Offset with PFS off.** The same move: does Z move? (§3.2 assumes not,
   and §3.3.1 links no Z stop to it then.)
6. **Limits and stop.** `PFSOffset`'s property limits, and whether
   `stop("PFSOffset")` is accepted (`autofocus_offset_soft_limits`). The
   emergency stop and the Z stop's link send it (§3.3.1, §3.11).
7. **The slot API.** `getAutoFocusOffset()` against
   `getPosition("PFSOffset")`, and whether `setAutoFocusOffset(o)` moves
   `PFSOffset` on the Ti2-E.
8. **Switch-off.** PFS locked and following a slow Z disturbance:
   `enableContinuousFocus(False)` stops Z at once?
9. **Z busy under lock.** `deviceBusy("ZDrive")` while PFS holds focus with
   the stage still (M1 §13 "Not verified": a Z drive that reads busy for
   good would refuse every action).

## 5. Sequencing

| Issue | Touches | Runs |
|---|---|---|
| #13 | `capabilities.py`, `errors.py`, `safety.py`, `backends/mm.py`, `microscope.py`, `profile.py`, `testing/fakes.py`, `tests/contracts/conftest.py`, `profiles/demo.toml`, `profiles/README.md` | first |
| #14 | the same capability, backend, facade, fake and contract modules | after #13 merges |
| #106, #111, #112 | `profile.py` | one at a time, not beside #13 |
| #12 | `profiles/nikon-*.toml`, a Nikon backend module | its plan decides, and fixes the Ti2-E profile's form and where its configuration file lives (`nikon-ti2.md`, *Profile entries, measured*); its profiles use §3.9's keys once #13 is merged |

## 6. Not verified

- Everything in §4.
- That PFS corrects only through the Z drive on both Nikon stands (§2).
- That `nikon-control`'s 5 s lock wait is enough (§3.2 labels it assumed).
- The lock-wait motion on a real stand: that `isContinuousFocusLocked()`
  reads `False` while PFS searches and `True` once it holds (§3.3).
