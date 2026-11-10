# M2 design — Nikon on hardware

- **Status**: design for issue #13 (§1–§6, 2026-10-16); §3.3.1 and
  §3.11's stop rules decided in #144 (2026-10-18); §7 for #12
  (2026-10-19); §8 for #14 (2026-10-19); §8.10 for #167 (2026-10-20); §9
  for #48 (2026-10-20); §10 for #15 (2026-10-20); §11 for #204 and #206
  (2026-10-28), §11.11 for #224 (2026-10-30); §12 for #205 (2026-10-28);
  §13 for #340 (2026-11-05); §8.11 D8 and §12.10's amendments for #407
  (2026-11-10).
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
builder (#15) and a truthful probe (#48). §1–§6 fix #13; §7 fixes #12;
§8 fixes #14; §9 fixes #48; §10 is #15's; §11 is #204's, a stand that refuses `stop()`; §12 is #205's, engaging an
autofocus that drives Z to its offset's plane.

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
  **On the Ti2-E, engaging drives Z to the plane the stored offset names**:
  when the layer may engage, the Z guard around it, and when `True` means
  settled are §12 (#205).

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
| The same move, PFS off | the offset device's | no: it moves no Z (measured, §4.5, §11.9) | yes | yes |
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
  `MMZStage` (`Microscope.override`): there is no stop of it to link, so the
  offset move looks for the focus drive first and, without one, reads no
  PFS state (#158).
- **What "cannot be read" covers** (#158). The core's AutoFocus slot is
  checked first: a slot that names another device is a configuration error,
  not a transient read failure, and refuses the offset move with nothing
  sent (FM-87), as every other slot call does. Only a failed
  `isContinuousFocusEnabled()` counts as on, and it is logged at WARNING,
  since it is an assumption: `autofocus PFS: its state could not be read
  before <description> (<error>); counted as on, so a Z stop during it
  switches it off`.
- **Why unknown means "on" here and a refusal elsewhere (FM-19, #158).**
  `focus_by_um` and the Z interlock's `suspend` let the same failed read
  raise, with nothing sent: they choose by it *what moves* (Z or the
  offset; whether to engage PFS again after the Z move, which drives Z), so
  a guess either way can move the wrong axis. The offset move does the same
  thing whatever the read says; the read only decides whether the
  operator's Z stop also reaches it. Counted as on, the worst outcome is a
  switch-off that was not needed, after the operator asked for a stop.
- **A capability that outlived its `Microscope`** (FM-92, #158): `engage`,
  `set_offset_native` and `offset_by_native` look up the focus drive inside
  the compound before anything else, and refuse with the facade's
  `HardwareError`, with nothing read or sent, whatever PFS's state.
- **Outcome of a Z stop once the call has sent something.** PFS is switched
  off, and the offset device is stopped while its move is pending or under
  way, before Z's own stop and inside one `hold_log` (FM-62). The call raises
  `MotionStoppedError` naming Z, e.g. `` `z ZDrive`: it was stopped during
  autofocus: engage, so autofocus PFS was left off; read the position before
  moving on ``, and the WARNING `autofocus PFS: left off, because autofocus:
  engage did not complete (MotionStoppedError)` is logged. `engage` keeps its
  contract: `False` means "no lock, and it is off". A Z stop is a raise, never
  a `False`.
- **The call waits for the Z stop's outcome before it reports (FM-97,
  #158).** A Z stop marks the link reached before the stopping thread sends
  the switch-off, so a call whose block ends meanwhile (its lock found, its
  offset move landed) would read "stopped, nothing failed" while the
  switch-off is still on its way. Once its block has ended, a call that a Z
  stop reached waits for that thread's linked stop to return, at most
  `Z_STOP_OUTCOME_WAIT_S` (5 s, assumed: MMCore's default device timeout),
  still under the microscope lock. This holds for every compound that links
  the Z stop: the relock and a locked `focus_by_um` as well. A halt cuts the
  wait: the stand's halted state is read at each poll, since the Z stop has
  already ended the block with a plain `MotionStoppedError` by the time the
  emergency stop comes; the call then reports the switch-off as not returned
  yet. A Ctrl-C during the wait goes on at once, with the WARNING of a Ctrl-C
  that ended the block (#166).
- **When the Z stop's switch-off failed, or had not returned in time**, no
  "left off" is claimed. The call raises `MotionStoppedError` naming Z:
  `` `z ZDrive`: it was stopped during <description>, but autofocus PFS could
  not be switched off (<failure>); it may still move Z: stop() and check the
  stand, then resume(); read the position before moving on ``, chained from
  whatever ended the block (its own motion stopped, as the offset device is
  by the same Z stop; a lock found; a move that landed). Three endings go on
  as they are: a give-up whose own error already says that PFS may still
  move Z (`DeviceTimeoutError`, which keeps the motion registered), a halt
  and an interrupt. The WARNING `autofocus PFS: the Z stop during
  <description> could not switch it off (<failure>); it may still move Z` is
  logged. A switch-off that had not returned names itself as the failure:
  `its switch-off had not returned after 5 s`, or `its switch-off had not
  returned yet` when a halt or a Ctrl-C cut the wait.
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
- **On a stand that refuses the Z or offset stop** (the Ti2-E), every
  stop is still sent in this order, and PFS off is what ends a Z move PFS
  drives (§11.9).
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
   The relock runs §12's check and Z guard, as `engage()` does.
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
- **A refused stop** (a stand that declares `[safety] stop_refused_by`)
  is reported as such, never as a fault or as a stop: §11.5.
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
  what was sent just before, in the same order, and is idempotent. That
  second switch-off is sent only while the relock still waits when Z's
  stop goes out (FM-96, #158). On a stand whose stage stops are slow (over
  a serial line), the relock may see PFS off at its next poll and end
  first: the log then has one switch-off, the link is never reached, and
  the caller gets the relock's own outcome, naming the autofocus, not Z.
  PFS is off all the same. In exchange, when the link is reached, the
  compound knows a Z stop ended it: its caller gets `MotionStoppedError`
  naming Z, and the "left off" WARNING is logged. A
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
- `hold_whole_units(label)` makes a `Stage` or `AutoFocus` device hold
  whole units, as the Ti2-E's `PFSOffset` does (session 1, #207: 7927.5
  reads 7927, 7925.75 lands on 7925): a `setPosition(label, v)` or, for the
  AutoFocus slot's device, a `setAutoFocusOffset(v)` stores `v` truncated
  towards zero (measured above zero only; whether the device truncates or
  rounds is unsettled, and 7927.5 fits both). `log` keeps `v` as sent; a
  value that is not finite is stored as given. It raises `ValueError` for a
  label that names no loaded `Stage` or `AutoFocus` device, so that a
  misspelled label fails the test instead of building a stand that holds
  fractions (FM-119).
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
  case. **In whole units** (#207, FM-119): the Ti2-E's `PFSOffset` holds
  nothing finer, so every offset a contract sends is the start plus a whole
  number, and no tolerance is left to absorb a fraction. The jog contract
  sets start + 1.0, then jogs by −2.0: above the limit of 1.0, so refused,
  and forced it lands at start − 1.0 (a jog clamped to the limit would land
  at the start, which the contract tells apart). The `fake` stand's
  autofocus holds whole units (`hold_whole_units`, §3.12) from an offset of
  7927, a positive whole number like the stand's reads, and the `demo`'s
  holds floats, so a contract that sends a fraction fails on `fake`. A
  stand whose offset has a step coarser than one unit would need a step in
  the profile; none is known. The teardown restores, in order: shutter closed, autofocus off
  (`disengage`), XY, Z (`keep_autofocus=False`), offset, exposure,
  auto-shutter, then the autofocus engaged if it was (no lock is a named
  teardown failure, FM-49), then the shutter. On `hardware`, a case that
  needs a lock skips when `engage()` returns `False` ("no lock on this
  sample"); on `fake` and `demo` it fails.
- **FakeCore** (`tests/unit/test_autofocus.py`, the Ti2-E layout above,
  through `Microscope.from_core`): the interlock with the quirk on
  (`test_moving_z_re_engages_pfs`), every refusal and failure path of §3.4
  and §3.5, the slot, the offset device, the emergency stop, threading. A
  `PFSOffset` that holds whole units gets a relative move that starts from
  the read-back (`test_a_relative_offset_move_starts_from_the_whole_unit_readback`).
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
| #14 | `capabilities.py`, `errors.py`, `channels.py` (new), `backends/mm.py`, `microscope.py`, `profile.py`, `testing/fakes.py`, `tests/contracts/conftest.py`, `profiles/demo.toml`, `profiles/README.md` (§8) | after #12; alone on `backends/mm.py`, `microscope.py` and `fakes.py` (#141, #163 and #166 wait); #124 (`roles.py`) runs beside it |
| #106, #111, #112 | `profile.py` | one at a time, not beside #13 |
| #12 | `backends/nikon.py`, `discovery/dll_probe.py`, `discovery/_dll_child.py`, `discovery/__init__.py`, `discovery/models.py`, `discovery/report.py`, `cli.py`, `profiles/nikon-*`, `profiles/README.md` (§7) | after #13; before #48 and #15, which touch the same discovery and CLI modules |
| #15 | `discovery/mm_inventory.py`, `discovery/_mm_child.py`, `discovery/models.py`, `discovery/report.py`, `discovery/config_build.py` (new), `hardware/core.py`, `cli.py`, `docs/hardware/microscope-pc-setup.md`, `profiles/README.md` (§10) | after #48; before #183 and #159 |

## 6. Not verified

- Everything in §4.
- That PFS corrects only through the Z drive on both Nikon stands (§2).
- That `nikon-control`'s 5 s lock wait is enough (§3.2 labels it assumed).
- The lock-wait motion on a real stand: that `isContinuousFocusLocked()`
  reads `False` while PFS searches and `True` once it holds (§3.3).

## 7. Nikon stand knowledge (#12)

Extends M1 §1 (module map), §9 (commands) and §10 (`smc discover`).
Sources: `nikon-control` (`scope/stand.py`, `scope/discover.py`,
`scope/config_build.py`, read 2026-10-19), `docs/hardware/inventory.md` and
session 0 (`nikon-ti2.md`). Every fact below is measured unless it says
otherwise.

### 7.1 What is known

- **Two adapters, two meanings of a device list.** `NikonTi2` asks Nikon's
  SDK what is on the stand, so an empty list means the SDK could not be
  reached. `NikonTI` publishes a fixed list of 18 devices, listed with no
  microscope attached, so a list proves nothing.
- **Where each driver is found.** `NikonTi2` loads `Ti2_Mic_Driver.dll` from
  its own folder. On our Ti2-E it was beside the adapter, with the SHA-256 of
  the copy in `C:\Program Files\Nikon\Ti2-SDK\bin` (session 0). `NikonTI`
  finds `NikonTi.dll` on the system path, where Nikon's installer puts it
  (`C:\Program Files\Nikon\Shared\Bin`; from `nikon-control`, not seen here).
- **One message, many causes (FM-09).** MMCore reports `Failed to load device
  adapter "<name>"` for a missing dependency, a wrong SDK, the wrong bitness
  and an interface mismatch. Windows' own error code tells them apart.
  `nikon-control` reads it by loading the DLL with `ctypes.WinDLL` in its
  own process, vendor DLL first, then the adapter. **That second probe can
  pass when the adapter's dependency is missing**: Windows resolves a module
  already loaded in the process by its base name (from the Windows loader's
  documented behaviour, not reproduced).
- **A failed load leaves nothing loaded.** Measured on the demo
  (pymmcore-plus 0.18.1, 2026-10-19): a `.cfg` that loads `DHub` and `DCam`,
  then fails on an unknown adapter, leaves only `Core` in
  `getLoadedDevices()`. MMCore's message is several lines: `Line 4:
  Device,…` and then `Failed to load device adapter "<name>" …`.
- **Why a hub does not open, in order of likelihood** (`nikon-control`,
  `config_build.py`): another program already holds the stand (the one
  people hit most and never guess, since it is usually their own); the stand
  or its controller is off, or was powered in the wrong order; the driver
  DLL is missing or the wrong version.
- **Stand facts that are causes.** Ti2-E: power the stand before the
  controller box, or only `Ti2-Simulator` answers; `Ti2Sample.exe` from the
  Ti2 SDK tests the SDK without Micro-Manager; Ti2 Control 2.10 and 2.20
  crash Micro-Manager when the nosepiece is added (mmCoreAndDevices#44),
  2.90.108 works on ours. Ti-E: USB 2, since USB 3 is a documented source of
  connection failures; `TIDiaLamp` has crashed Micro-Manager with some
  driver versions. (`nikon-control`, except 2.90.108.)
- **Roles.** The built-in exclusions of M1 §4 already keep the TIRF
  positioners out of `xy_stage` and the PFS offset and the TIRF drive out of
  `focus`. Measured on the session-0 list and on `nikon-control`'s Ti-E list
  (2026-10-19): every role resolves as §7.8 says, and a `.cfg` slot naming
  `PFSOffset`, `TIRF1` or `TITIRF` falls back to the right device with a
  warning. `light_source` offers the light path on both stands (#124).

### 7.2 Module — `smc/hardware/backends/nikon.py`

Pure Python and the file system: it never imports `pymmcore_plus` and never
opens a core. The DLL probe it calls runs in a child (§7.3).

```python
@dataclass(frozen=True, slots=True)
class Driver:
    dll: str  # "Ti2_Mic_Driver.dll"
    sdk_dir: str  # a Windows path, where the vendor installer puts it
    installer: str  # what to install to get it
    beside_adapter: bool  # True: the adapter loads it from its own folder
```

```python
@dataclass(frozen=True, slots=True)
class Stand:
    key: str  # "ti2" | "tie": the argument of `smc install-driver`
    adapter: str  # "NikonTi2" | "NikonTI"
    name: str  # "Nikon Ti2-E" | "Nikon Ti-E"
    hub: str  # "" when the SDK names it (Ti2-E__0 on ours) | "TIScope"
    dynamic: bool  # the device list comes from the SDK
    driver: Driver
    causes: tuple[str, ...]  # likely causes after ONE_CONNECTION, in order

    @property
    def adapter_file(self) -> str: ...  # f"mmgr_dal_{adapter}.dll"

    @property
    def empty_list_meaning(self) -> str: ...
```

```python
@dataclass(frozen=True, slots=True)
class DriverLocation:
    path: Path | None  # None: not found
    beside_adapter: bool  # in the adapter's own folder
    satisfied: bool  # where this stand's adapter finds it
```

```python
@dataclass(frozen=True, slots=True)
class StandCheck:
    stand: Stand
    adapter_path: Path | None  # the adapter MMCore would load, None: none
    searched: tuple[Path, ...]  # the adapter folders, in MMCore's order
    driver: DriverLocation
```

```python
TI2: Stand
TIE: Stand
STANDS: tuple[Stand, ...]  # (TI2, TIE)
ONE_CONNECTION: str  # the first likely cause, shared by both stands


def stand_for_adapter(adapter: str) -> Stand | None: ...
def stand_for_key(key: str) -> Stand | None: ...
def stands_in_config(path: Path) -> list[Stand]: ...
def find_driver(
    stand: Stand, adapter_dir: Path | None, *, sdk_dir: Path | None = None
) -> DriverLocation: ...
def check_stand(
    stand: Stand, adapter_dirs: Sequence[Path], *, sdk_dir: Path | None = None
) -> StandCheck: ...
def readiness(check: StandCheck) -> list[tuple[str, str]]: ...
def deep_check(
    check: StandCheck, *, extra_path: Sequence[Path] = ()
) -> list[DllProbe]: ...
def diagnosis(check: StandCheck, probes: Sequence[DllProbe]) -> list[str]: ...
def install_driver(
    stand: Stand, adapter_dir: Path, *, source: Path | None = None
) -> tuple[bool, str]: ...
```

**The registry** (decided; texts may be reworded, their facts may not):

| | `TI2` | `TIE` |
|---|---|---|
| `key`, `adapter`, `name` | `ti2`, `NikonTi2`, `Nikon Ti2-E` | `tie`, `NikonTI`, `Nikon Ti-E` |
| `hub`, `dynamic` | `""`, `True` | `TIScope`, `False` |
| `driver.dll`, `sdk_dir` | `Ti2_Mic_Driver.dll`, `C:\Program Files\Nikon\Ti2-SDK\bin` | `NikonTi.dll`, `C:\Program Files\Nikon\Shared\Bin` |
| `driver.installer` | `Ti2 Control (the Ti2 SDK)` | `Nikon's Ti Setup Tool (or the TiSDKRedist package)` |
| `driver.beside_adapter` | `True` | `False` |
| `causes`, in order | power (stand before controller box; else only `Ti2-Simulator` answers); the driver (missing beside the adapter, or not the version the adapter was built for; `Ti2Sample.exe` tests the SDK alone); Ti2 Control 2.10/2.20 (2.00 reported to work, 2.90.108 works on ours) | power and USB (a USB 2 port; USB 3 fails on this generation); the driver (`NikonTi.dll` not installed, or not on the system path); `TIDiaLamp` (leave it out of the configuration unless it is needed) |

`ONE_CONNECTION`: another program holds the stand, which accepts one
connection at a time; close NIS-Elements, Ti2 Control, Micro-Manager and any
other smc or pymmcore session, then run this again.

`empty_list_meaning`: dynamic, "the adapter asks Nikon's SDK what is on the
stand, so an empty device list means the SDK could not be reached, not that
the adapter is missing"; fixed, "the adapter publishes a fixed device list,
so a device list proves nothing about what is connected; only a probe does".
`smc discover` uses it in #48.

`nikon-control`'s `SHARED_NOTES` (moving Z switches PFS off) is not ported
as a note: the Z interlock encodes it (§3.4).

### 7.3 The loader probe — `smc/discovery/dll_probe.py`, `smc/discovery/_dll_child.py`

Generic: any adapter, not only Nikon's (FM-09). In `smc.discovery`, which
already owns the throwaway children; `nikon.py` imports it, and nothing in it
imports `smc.hardware`.

```python
DLL_PROBE_TIMEOUT_S = 20.0  # M1 §10: every OS call has a 20 s timeout
CHILD_COMMAND: list[str]  # [sys.executable, "-m", "smc.discovery._dll_child"]
WINERROR_HINTS: Mapping[int, str]  # 126, 127, 193, 1114 (below)


@dataclass(frozen=True, slots=True)
class DllProbe:
    path: Path
    loads: bool | None  # None: not tried, or no verdict
    winerror: int | None
    detail: str  # one line, ready to print


def describe_winerror(code: int | None, strerror: str) -> str: ...
def probe_dll(
    path: Path,
    *,
    extra_path: Sequence[Path] = (),
    timeout_s: float = DLL_PROBE_TIMEOUT_S,
) -> DllProbe: ...
```

`WINERROR_HINTS` (from `nikon-control`, the first one made generic):
- 126: a DLL it depends on could not be found; for a vendor adapter, usually
  the vendor's driver or runtime DLL, not beside the adapter or not
  installed;
- 127: an entry point is missing; the vendor DLL is not the version this
  adapter was built against;
- 193: wrong architecture; a 32-bit DLL under a 64-bit Micro-Manager, or the
  reverse;
- 1114: the DLL loaded but its initialisation routine failed; usually a
  vendor SDK mismatch, or hardware that is off.

`describe_winerror(126, "The specified module could not be found.")` is
`WinError 126: The specified module could not be found. — <hint>`; an
unknown code has no hint; `None` gives `strerror` alone.

**One child per DLL.** `probe_dll` runs `CHILD_COMMAND <path>` and reads its
verdict, so a probe never shares a process with another DLL (§7.1), a vendor
`DllMain` that crashes or hangs costs that probe only (FM-02), and the
parent never holds a vendor DLL. Its rules:
- not Windows: `DllProbe(path, None, None, "not Windows: a .dll cannot be
  loaded here")`, and no child is started; a missing file: `loads=None`,
  `"<path> does not exist"`, no child;
- the child runs as `mm_section`'s does: a temporary folder as working
  directory (FM-33), stdin from `DEVNULL`, stdout and stderr to files in that
  folder, never pipes (FM-04), `CREATE_NO_WINDOW` (FM-22), waited in slices
  (`mm_inventory._wait_in_slices`, FM-83), killed with its tree at the
  deadline (`_kill_child_tree`, FM-35) — reused, never copied;
- `extra_path` goes first on the child's `PATH`, in order, as `open_core`
  puts the profile's adapter folders there (an entry containing
  `os.pathsep` is left out, FM-81);
- deadline: `loads=None`, `"did not return within 20 s: its initialisation
  hung"`; a non-zero exit or output that is not the protocol: `loads=None`,
  `"the probe child ended with exit code <exit_detail(…)> and no verdict"`.

**The child** (`_dll_child.py`, `main(argv) -> int`): first
`SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX)`, its
`argtypes` and `restype` declared (FM-03, FM-26); then
`ctypes.WinDLL(str(path), winmode=0x8)` (`LOAD_WITH_ALTERED_SEARCH_PATH`:
the DLL's own folder first, then the standard order, `PATH` included, which
is where each Nikon driver is found; that MMCore searches the same way is
assumed). It writes one line of ASCII JSON to stdout (FM-23):
`{"loads": true}` or `{"loads": false, "winerror": 126, "strerror": "…"}`,
then exits 0. It imports nothing from `pymmcore_plus` and never frees the
DLL: the process ends.

### 7.4 `smc doctor` (extends M1 §9)

With a profile, and with `--config`, `doctor` finds the Nikon stands in the
configuration it is about to load (`stands_in_config`: the library column of
its `Device,<label>,<library>,<name>` lines; comments skipped; each stand
once, in order of first appearance; an unreadable file gives `[]`). The
adapter folders are the profile's `adapter_search_paths`, then the install
(MMCore's order, first wins, FM-76); with `--config`, the install alone.

1. **Before opening**, for each stand, `readiness(check)` as aligned rows
   under a heading `Nikon Ti2-E (NikonTi2)`:
   `adapter` (its path, or `not found in <folders>`) and `driver`
   (`Ti2_Mic_Driver.dll beside the adapter`, or where it is and that it must
   be copied, `smc install-driver ti2`, or `not found`; for the Ti-E, whether
   `NikonTi.dll` is in its `sdk_dir`). These rows never change the exit code.
2. **If the open raises `CoreError`**, the `✗` line is printed as today
   (`--debug` unchanged), then, for each stand, `deep_check(check)` and the
   `diagnosis` lines, under `Nikon Ti2-E: what to check`, and the exit code
   is 1. `deep_check` probes the adapter (`adapter_path`), then the driver
   (when found), each in its own child, with the adapter folders on `PATH`.
   MMCore has unloaded everything by then (§7.1). Any other error is reported
   as today, without a diagnosis.
3. **`diagnosis`** lists, in this order:
   - `found: …`, one line per certain finding: the adapter is in none of the
     folders (naming them); the Ti2 driver is not beside the adapter (where
     it is, and `smc install-driver ti2`); it is nowhere (install the
     installer); the Ti-E driver is not in its `sdk_dir`;
   - `probe: <file> loads`, `probe: <file> will not load: <detail>` or
     `probe: <file> not probed: <detail>`, one per probe;
   - `in order of likelihood:`, then `  1. <ONE_CONNECTION>` and the stand's
     `causes`, numbered on.

Without a Nikon stand in the configuration, `doctor` prints exactly what it
prints today.

### 7.5 `smc install-driver STAND` (extends M1 §9)

`smc install-driver {ti2|tie} [--from FILE] [--to DIR]` copies the stand's
driver to where its adapter loads it. `--to` defaults to the folder of the
stand's adapter in the install (`core.find_install()`); with no adapter
there, `✗` and "pass --to". `--from` defaults to `sdk_dir/dll`. Exit 0 when
the driver is in place (copied, or an identical copy already there), 1
otherwise, 2 for usage. `install_driver`'s outcomes, each `(ok, message)`:

- a stand with `beside_adapter = False` (`tie`): `False`, the adapter finds
  it on the system path, install `<installer>` instead;
- no source: `False`, `<src> does not exist`, install `<installer>` or pass
  `--from`;
- a destination with the source's SHA-256: `True`, already there;
- a destination that differs: `False`, it differs (both SHA-256 prefixes),
  and smc never overwrites a driver;
- otherwise `shutil.copy2` to `<dll>.partial` in the destination folder,
  then `os.replace` (FM-38); `True`, copied, with the SHA-256 prefix. An
  `OSError` gives `False`, `could not copy to <dest>: <exc>` (the folder may
  need an elevated prompt), and removes the partial file.

**Copy, never move**: the vendor's own software still needs its copy.

### 7.6 `smc discover` (extends M1 §10)

`AdapterInfo` gains `load_diagnosis: str | None = None`. After the adapter
child returns, on Windows, each listed adapter whose `error` contains
`Failed to load device adapter` gets `probe_dll(<install>/mmgr_dal_<name>.dll)
.detail`, at most `MAX_DLL_PROBES = 10` of them; the rest get none and one
note says how many were not probed (FM-74). The report prints it on a row of
its own under the `ERROR` row, as text, never markup (FM-36). Hub peripherals,
pre-init ports and the empty-list meaning are #48's.

### 7.7 Profiles

- **`profiles/nikon-ti2.toml`** points at **`profiles/nikon-ti2.cfg`**, a
  committed copy of the lab's `MMConfig.cfg` (nikon-control, 2026-09-16),
  which session 0 loaded with every core role answering. It holds no COM
  port, serial number, host or account name (FM-50), so it is committed,
  with a comment that says where it comes from. The profile sets what
  session 0 measured or the owner decided: `device_timeout_ms = 60000`,
  `[roles.assign] filter_turret = "FilterTurret2"`, `xy_soft_limits_um =
  [[-49000.0, 49000.0], [-29000.0, 29000.0]]`, `z_soft_limits_um = [10.0,
  4350.0]`, `max_z_jog_um = 100.0` (chosen, not measured; the owner revises
  it at #16). Every `[autofocus]` key, the offset safety keys and every pixel
  size stay unknown until #16 and #22 measure them.
- **`profiles/nikon-tie.toml`** points at **`profiles/nikon-tie.cfg`**,
  **assumed**: written for #12 from the adapter's fixed list, never loaded on
  the stand, without `TIDiaLamp`. The first Ti-E session replaces it with
  `smc config build` (#15). Its profile sets no limit beyond the layer's
  defaults, so the Z jog default is logged as assumed.
- Every profile under `profiles/` loads (a test), and each `.cfg` lists the
  devices its stand's measured list holds (a test).

### 7.8 Roles on the two stands

Measured with `resolve()` (2026-10-19). On the Ti2-E list, with and without
the `.cfg`'s core roles: `camera` Camera-1, `xy_stage` XYStage (the only
candidate), `focus` ZDrive (the only candidate), `autofocus` PFS,
`autofocus_offset` PFSOffset, `objective_turret` Nosepiece (the only
candidate), `shutter` DiaLamp (Turret1Shutter, Turret2Shutter as runners-up),
`light_path` LightPath, `filter_turret` FilterTurret1 or FilterTurret2 (the
profile assigns FilterTurret2); IntermediateMagnification fills no role. On
the Ti-E list: no `camera`, `xy_stage` TIXYDrive, `focus` TIZDrive (the only
candidate), `autofocus` TIPFSStatus, `autofocus_offset` TIPFSOffset,
`objective_turret` TINosePiece, `shutter` TIEpiShutter (TIDiaShutter,
TIDiaLamp, TIAuxShutter), `light_path` TILightPath, `filter_turret` four
candidates; TIAnalyzer fills no role. `light_source` is #124's.

### 7.9 Not verified

- Everything in §7.3 and §7.4 that only Windows shows: a real WinError, the
  child's `winmode`, `SetErrorMode`, and that MMCore's own search matches it.
- `nikon-tie.cfg` and every Ti-E fact from `nikon-control`.
- That a Ti2 hub that will not initialise fails `loadSystemConfiguration`
  with a `CoreError` (only an adapter that does not load was reproduced).

## 8. ObjectiveTurret, LightSource and Channels (#14)

Extends M1 §3 (capabilities), §5 (profiles), §6 (backend), §7 (facade), §8
(FakeCore, contracts) and §13 (kinds of call). Sources: the demo, the Ti2-E
(`nikon-ti2.md`, `profiles/nikon-ti2.cfg`) and `nikon-control`
(`scope/control.py`, `scope/channels.py`, read 2026-10-19).

### 8.1 What is known

Measured on the demo on 2026-10-19 (pymmcore-plus 0.18.1, pymmcore
12.5.0.75.0), unless a line names another source.

- **Roles on the demo.** `objective_turret` is `Objective` (heuristic).
  `light_source` is `LED`: a `State` device (`Closed`, `385nm` … `635nm`)
  that has no property naming an intensity. `light_path` and `filter_turret`
  are not filled; the demo's `Path` matches neither rule.
- **Turret labels.** On the demo, `Objective` has six positions:
  `Nikon 40X Plan Fluor ELWD`, `Nikon 10X S Fluor`, `Objective-2`,
  `Nikon 20X Plan Fluor ELWD`, `Objective-4`, `Objective-5`. An empty
  position carries a default label. On the Ti2-E, `Nosepiece` reads
  `1-Plan Fluor 4x`, `2-Plan Fluor 10x`, `3-Plan Apo LmbdD0.25 40x`, `4-`,
  `5-` and `6-` (the configuration's labels). Nothing tells smc which
  positions are empty.
- `setStateLabel` with an unknown label raises `RuntimeError('Error in
  device "Objective": Label not defined (10).  Available Labels: (…)')`. A
  demo turret move returns at once, and `deviceBusy` then reads `False`.
- **Config groups on the demo**: `Camera`, `Channel`, `Channel-Multiband`,
  `LightPath`, `Objective` and `System`. The core's channel group is
  `Channel`. `getCurrentConfig("Channel")` reads `DAPI` after the open, and
  `''` once a device no longer matches any preset. `getConfigData` returns a
  pymmcore-plus `Configuration`, which iterates as `(device, property,
  value)` tuples.
- `setConfig` with an unknown preset raises `ValueError('Preset "Nope" of
  configuration group "Channel" does not exist')`. On the demo it returns
  at once, with no device busy. A preset that sets `Core,Shutter` moves the
  core's shutter slot: `getShutterDevice()` reads the preset's shutter
  afterwards.
- **`defineConfig` merges into an existing preset.** It changes or adds one
  setting and keeps the others, so replacing a preset means `deleteConfig`
  first. `defineConfig` accepts the label of a device that is not loaded
  and says nothing. It creates a missing group. `deleteConfig` on a missing
  preset raises `ValueError('Configuration group "Channel" does not
  exist')`, which names the group and not the preset. A group keeps
  existing once its last preset is deleted, and `getChannelGroup()` does
  not change.
- **MMCore matches presets by string.** A preset `Camera,Exposure,10`,
  once applied, reads `10.0000`, and `getCurrentConfig` then reads `''`.
  The Ti2-E's `BF` sets `Exposure` 10 and `Intensity` 210, so MMCore would
  report no current channel after selecting it.
- A `.cfg` with CRLF line endings and a preset inserted among another
  preset's lines loads, and the inserted preset reads back whole.
- `setProperty` outside a property's limits raises `RuntimeError('Cannot
  set property "Photon Flux" to "9999.000000"')`.
- **`saveSystemConfiguration` rewrites the whole file** from the core's
  state. It drops comments and writes every label, every pre-initialisation
  property (a serial `Port` among them, FM-50) and the pixel-size
  configurations. `capture()` never calls it.
- `getMagnificationFactor()` reads 1.0 on the demo. The Ti2-E's
  `IntermediateMagnification` (`Magnifier`, read-only, `1.0x` or `1.5x`)
  divides the pixel size by 1.5; §8.10 says how the profile's sizes account
  for it (#167).
- **The Ti2-E's light** (session 0): `DiaLamp` is a `Shutter` whose
  `Intensity` is a free number (no limits; 501 was read), and it fills both
  the `shutter` role (the configuration's slot) and `light_source`. The
  lab's one group is `Channel`, with one preset `BF`: `CondenserTurret`
  `7-N2`, `FilterTurret1` `1-`, `Turret1Shutter` 0, `FilterTurret2`
  `1-ANALY`, `Turret2Shutter` 0, `LightPath` `4-L100`, `DiaLamp` `State` 0
  and `Intensity` 210, and `Camera-1` `Exposure` 10. The file has no
  `Core,ChannelGroup` line. The epi light (Lumencor Spectra III) is in no
  configuration.
- **`nikon-control`** (read): `set_objective` needs `confirm=True`, checks
  neither Z nor PFS, and waits with `waitForDevice`. The intensity knob is
  the first writable property whose name holds `intensity`, `brightness`,
  `power`, `level` or `voltage`, searched on the shutter and light-path
  roles and then on every device named like a light. A value outside the
  limits is clamped. The channel group is `Channel`, `Channels` or
  `channel`, else an explicit name. A capture records every `Shutter` and
  `State` device except the nosepiece, stages and autofocus, by `Label`
  (else `State`), plus the knob. It redefines the core's preset without
  deleting it first, so stale settings stay. It edits the `.cfg` as text,
  with no encoding, not atomically. Its tests: confirmation, "intensity is
  found on the lamp, not the camera", "a config group that is not channels
  is not treated as one", "capture never includes the objective turret".

### 8.2 Protocols and value types — `capabilities.py`

```python
@dataclass(frozen=True, slots=True)
class ChannelPreset:
    group: str
    name: str
    # (device, property, value), in the order MMCore applies them
    settings: tuple[tuple[str, str, str], ...]
    # the configuration file capture() wrote; None: the running core only
    saved_to: Path | None = None
```

```python
@runtime_checkable
class ObjectiveTurret(Protocol):
    def labels(self) -> list[str]: ...  # every position, in position order
    def current(self) -> str: ...
    def select(self, label: str, *, confirm: bool = False) -> str: ...
    def pixel_size_um(self, label: str | None = None) -> float: ...  # 0.0 = unknown
    def offset(self, label: str) -> ObjectiveOffset: ...  # §13.4, the profile's
```

```python
@runtime_checkable
class LightSource(Protocol):
    def device(self) -> str: ...
    def intensity_knob(self) -> PropertyInfo | None: ...  # None: no knob
    def intensity_native(self) -> float | None: ...  # None: no knob
    def set_intensity_native(self, value: float) -> float: ...
    def is_on(self) -> bool | None: ...  # None: no switch that smc knows
    def set_on(self, on: bool) -> bool: ...
```

```python
@runtime_checkable
class Channels(Protocol):
    def group(self) -> str: ...
    def names(self) -> list[str]: ...
    def current(self) -> str | None: ...  # None: the stand matches no preset
    def preset(self, name: str) -> ChannelPreset: ...
    def select(self, name: str) -> str | None: ...
    def capture(self, name: str) -> ChannelPreset: ...
```

The four names join `__all__`. **Names.** ADR-0003's `intensity`,
`set_intensity()` and `on/off` become `intensity_native()`,
`set_intensity_native()`, `is_on()` and `set_on()`. They are methods (M1
§2), and the intensity is in the device's own units: the `DiaLamp`'s
`Intensity` has no unit that smc knows (ADR-0003: named for what it is). The
ADR writes `select(label, *, confirm=True)` as the call a tool makes, so the
default is `False`.

### 8.3 ObjectiveTurret — `MMObjectiveTurret`

The `objective_turret` role; faults through `as_hardware_errors(core,
"objective_turret")` (FM-79).

- `labels()` is `list(getStateLabels(label))`, in position order, empty
  positions included as the configuration names them. `current()` is
  `getStateLabel(label)`.
- **`select(label, *, confirm=False)`**, in this order:
  1. **Confirmation (FM-14).** When `[safety] turret_requires_confirm` is
     true (the default) and `confirm` is false, it raises
     `SafetyRefusedError(reason="turning the objective turret can drive an
     objective into a loaded plate (objective_turret 'Nosepiece')",
     how_to_force="pass confirm=True once the stage is clear")`. Nothing is
     read or sent, no lock is taken, and the refusal also happens in dry-run.
     With the key false, `select` runs without `confirm` and logs at INFO
     `objective_turret: no confirmation required ([safety]
     turret_requires_confirm = false)`.
  2. A `label` that is not in `labels()` (a lock-free read) raises
     `ValueError` naming it and listing the labels. Nothing is sent.
  3. **One compound**, `objective_turret: select <label>` (§3.7):
     1. If `current()` already equals `label`, nothing is sent and the
        autofocus is left alone. It logs `objective_turret: already at
        <label>` at INFO and returns `label`.
     2. Unless in dry-run, the autofocus is switched off first when it is
        engaged (`disengage()`, a safe call). The autofocus is the facade's
        `Autofocus`, reached through the facade's weak reference (FM-92). A
        state that cannot be read counts as engaged, with a WARNING that
        says so (FM-19). It is **never engaged again** (FM-88). The WARNING
        reads `objective_turret: autofocus 'PFS' switched off before the
        turret turns; it is not engaged again (the new objective's offset
        is not known); engage() it once in focus`.
     3. `executor.do` with `Step(send=setStateLabel(turret, label),
        readback=getStateLabel(turret), dry_result=label)`, the motion
        `_motion(core, turret, f"objective_turret {turret}",
        stoppable=False)` and `timeout_s=_core_timeout(core)`. The readback
        is re-read until it is `label`, within a bound (§8.11 D3).
     4. Unless in dry-run, and when the stand has a focus drive: Z is read
        before the send and, once the label is read back, again until it
        has been still for a moment, then logged (§8.11 D2). The stand
        moves Z during a turn (§8.9).

     In dry-run the compound runs and its nested action logs and skips, so
     nothing is sent, the switch-off included, and no Z is read.
  4. It returns the readback.

  **Why the switch-off.** A PFS that follows through a turret rotation
  drives Z while no action holds the lock (§3.2). The new objective's
  offset differs (`offset_per_um` is per objective; on the Ti-E, the valid
  range depends on the objective, §1), so engaging again could drive Z
  anywhere. The Ti2-E switches PFS off by itself on a turn (§8.9, session
  2: PFS on and searching, no sample; a locked PFS was not tried). The
  layer's switch-off **stays**: what a stand does by itself is not the
  layer's guarantee, the layer's own read of the autofocus after a turn
  must not depend on it, and a locked PFS may behave otherwise.
- **A turret cannot be stopped** (a `State` device, M1 §13). One still busy
  at the deadline stays registered as moving until it reads idle or
  `resume()` is called. Neither the emergency stop nor `close()` sends it
  anything.
- `pixel_size_um(label=None)` returns the profile's `[camera]
  pixel_size_um` for `label` (the current objective when `None`) at binning
  1, divided by the core's magnification factor (§8.10, #167), and 0.0 when
  the profile has none or the factor is not a positive number.
  `Camera.pixel_size_um()` (M1 §3) follows the same rule.

### 8.4 LightSource — `MMLightSource`

The `light_source` role; faults through `as_hardware_errors(core,
"light_source")`.

- `device()` is the role's label.
- **The knob** is resolved on the first call that needs it and is kept from
  then on (its property name, or that there is none). Every call reads the
  value again.
  1. **The profile's `[light_source] intensity_property`**: used when the
     device has that property and it is writable. Otherwise a WARNING names
     the key, the device and its writable properties, and the search falls
     through.
  2. **Hints**: the device's writable properties whose normalised name
     (`[a-z0-9]`) contains one of `INTENSITY_HINTS = ("intensity",
     "brightness", "power", "level", "voltage")`. They are ranked by hint,
     then by property order. The first one that has limits wins, else the
     first whose value reads as a finite float. A property with no numeric
     value is never a knob.
  3. **None**: logged once at INFO, `light_source 'LED': no intensity
     property (searched for intensity, brightness, power, level, voltage);
     set [light_source] intensity_property if the device has one`.

  Only the role's device is searched: a camera's `BeadBrightness` is not a
  lamp (`nikon-control`'s test).
- `intensity_knob()` is the knob's `PropertyInfo`, read now, or `None`.
  `intensity_native()` is `float(getProperty(device, knob))`, or `None`
  without a knob. A value that is not a number raises `HardwareError`
  naming the device, the property and the value.
- **`set_intensity_native(value)`** raises before anything is sent:
  - `HardwareError` when there is no knob: `light_source 'LED' has no
    intensity property that smc knows; set [light_source] intensity_property
    in the profile, or use Properties`;
  - `ValueError` for a value that is not finite;
  - `ValueError` for a value outside the knob's limits, when it has limits.
    The value is never clamped: `nikon-control` clamped, which reads as
    success.

  Otherwise it is an action, `light_source: intensity <value>`, through
  `executor.do`: `Step(send=setProperty(device, knob, value),
  readback=float(getProperty(...)), dry_result=value)`. It is not a motion.
  The readback is **re-read until it is the value that was set, within a
  bound**: the Ti2-E's `DiaLamp` reads the previous value for a while after
  a set (§8.9). §8.11 D1 fixes the rule, the bound and the words.
- **`is_on()` / `set_on(on)`**, when the device's type is `Shutter`:
  `getShutterOpen(device)` and `setShutterOpen(device, on)`, with the same
  kinds of call as `MMShutter` (M1 §13). `set_on(False)` is a safe call,
  `SafeCall(device=<label>, log="light_source: off", …)`. `set_on(True)` is
  an action overridden by that safe call. On the Ti2-E, `DiaLamp` is both
  the shutter and the light source. Both capabilities key their safe calls
  by its label, so a close from either one cancels an open from either one.
  For any other type, `is_on()` returns `None`, and `set_on()` raises
  `HardwareError`: `light_source 'LED' (a State device) has no on/off that
  smc knows; switch the light with its shutter (Shutter)`.
- The emergency stop does not cut the light: it stops what moves (M1 §13).
  `Shutter.set_open(False)` and `set_on(False)` are never refused.

### 8.5 Channels — `MMChannels` and `smc/hardware/channels.py`

A config group, not a role. The pure parts live in a new module,
`smc/hardware/channels.py`, which imports no `pymmcore_plus` (ADR-0004's
split): the group's choice, the preset checks, the capture's choice of
settings, and the `.cfg` edit.

```python
CHANNEL_WORD = "channel"
NOT_CHANNEL_WORDS = ("camera", "objective", "system")
ALLOWED_CORE_SETTINGS = ("Shutter", "AutoShutter")
FORBIDDEN_TYPES = ("Stage", "XYStage", "AutoFocus")


def named_for_channels(group: str) -> bool: ...
def choose_group(
    groups: Sequence[str], *, core_group: str, profile_group: str
) -> tuple[str | None, list[str]]: ...  # (the group, warnings)
def check_preset_name(name: str) -> None: ...  # ValueError
def forbidden_settings(
    settings: Iterable[tuple[str, str, str]],
    *,
    device_types: Mapping[str, str],
    objective_turret: str | None,
) -> list[str]: ...  # one reason per refused setting, empty: allowed
def check_value(device: str, prop: str, value: str) -> None: ...  # HardwareError
def setting_matches(expected: str, read: str) -> bool: ...
def replace_preset(
    text: str, group: str, name: str, settings: Sequence[tuple[str, str, str]]
) -> str: ...
```

**The group**, `choose_group`, takes the first source that gives one:
1. the profile's `[channels] group`, when it is not empty, whether the
   group exists yet or not (`capture()` creates it);
2. the core's channel group (`getChannelGroup()`), when it exists and is
   named for channels;
3. the one existing group named for channels. With several, there is none,
   and a warning names them and `[channels] group`.

A group is **named for channels** when its normalised name contains
`channel` and none of `camera`, `objective` or `system`. Measured: the demo
gives `Channel` (from the core), and the Ti2-E gives `Channel` (the only
group). `Channel-Multiband` on the demo is named for channels too; the core
decides between the two. With no group, `Channels` is not available (§8.6).

- `group()` returns the chosen group. `names()` is
  `list(getAvailableConfigs(group))`, or `[]` while the group does not
  exist. `preset(name)` reads `getConfigData`. A name that is not in
  `names()` raises `ValueError` listing them.
- **`current()`** is the first preset, in `names()` order, whose every
  setting matches what the stand reads now, else `None`. It does not use
  `getCurrentConfig`, which compares strings (§8.1). `setting_matches`
  holds when the two strings are equal, or when both read as finite floats
  that are equal. A `Core,Shutter` setting is compared with
  `getShutterDevice()`, and `Core,AutoShutter` with `getAutoShutter()` as
  `1` or `0`. A read that fails propagates as a `HardwareError`.
- **`select(name)`**:
  1. A name that is not in `names()` raises `ValueError`. Nothing is sent.
  2. One action, `channels: select <group> <name>`, through `executor.do`
     with a step **built under the lock**: it reads the preset's settings
     and the type of each device. If `forbidden_settings` returns any, it
     raises `SafetyRefusedError` with an empty `how_to_force`, which cannot
     be forced: the preset would move a stage, an autofocus or the
     objective turret around their guards, or change a core slot. The
     message names each setting and says to edit the preset in the
     configuration, or to use the capability (`ZStage`, `ObjectiveTurret`).
     Refused: a device of type `Stage`, `XYStage` or `AutoFocus`; the
     `objective_turret` role's device; a `Core` property other than
     `Shutter` and `AutoShutter`; a device that is not loaded.
  3. The command is `setConfig(group, name)`. The motion is
     `Motion(device=f"config group {group}", name=f"channels {group}
     {name}", is_busy=<any of the preset's non-Core devices reads busy>,
     stop=None)`, with `timeout_s=_core_timeout(core)`. The readback is
     `name` when every setting of the preset matches what the stand reads
     (two presets may hold the same settings), else `current()`; the match
     is re-read until it holds, within a bound (§8.11 D1, D4), since a
     setting of the preset may read its previous value right after the
     set; `dry_result` is `name`. One command moves several
     devices, and no stop reaches a `State` device, so the motion has a
     key of its own. A device labelled `config group <name>` would share
     it; no stand has one.
  4. When the readback is not `name`, a WARNING names each setting that
     reads otherwise. It returns the readback: `None` when
     the stand matches no preset.
- **`capture(name)`**:
  1. `check_preset_name(name)`: the name must not be empty, must be
     printable ASCII, must not contain a comma, and must not start or end
     with whitespace. Otherwise `ValueError`.
  2. One action, `channels: capture <group> <name>`, through `executor.do`
     with no motion. Its step is built under the lock:
     1. **The settings**, in this order:
        - for each loaded device of type `State`, in `getLoadedDevices()`
          order, except the `objective_turret` role's device and the
          profile's `[channels] exclude` labels: `(device, "Label",
          getStateLabel(device))`. A device whose `Label` is read-only or
          absent is skipped;
        - the light source's knob, when `LightSource` is available and has
          one, with its value as `getProperty` returns it;
        - `("Core", "Shutter", getShutterDevice())`, when the slot is not
          empty.

        A capture never records a shutter's open state, a `Stage`,
        `XYStage`, `AutoFocus`, `Camera`, `Hub` or `Magnifier` device, or
        the exposure. A channel says which light path and how bright, not
        whether the light is on (auto-shutter or `Shutter` opens it). The
        lab's `BF` preset sets shutters closed and an exposure: `select()`
        accepts such presets, and `capture()` does not write them.
     2. `check_value` for each setting: the value must be printable ASCII
        with no comma (FM-81: the `.cfg` is comma-separated; FM-05: an
        adapter's bytes). Otherwise it raises `HardwareError` naming the
        device, the property and the value, and nothing is written.
     3. With no setting, it raises `HardwareError`, `nothing to capture`.

     `dry_result` is the `ChannelPreset` with `saved_to=None`.
  3. **The send**:
     1. **The file first**, when `profile.config_path()` is not `None`. The
        file is read as bytes and decoded as UTF-8; a decoding failure
        raises `HardwareError` naming the file, and nothing is written. The
        new text, `replace_preset(…)`, is encoded as UTF-8 and written as
        bytes, atomically (FM-38: a temporary file in the same folder,
        `O_CREAT | O_EXCL` with mode `0o666`, `os.replace`, one `try`; it is
        removed on failure, except when only the rename failed, which keeps
        it and names it).
     2. **Then the running core**: `deleteConfig(group, name)` when `name`
        is in `names()`, then one `defineConfig` per setting, in order.

     The file is the record. A failed write leaves the core unchanged. A
     core failure after the write raises `HardwareError`, which says the
     file holds the preset and the next open loads it. Without a file (the
     demo's configuration), only the core changes, and an INFO line says
     `channels: <name> not saved: the profile loads the demo configuration,
     which has no file`.
  4. The readback is `preset(name)` with `saved_to`.

**`replace_preset(text, group, name, settings)`**: a line belongs to the
preset when its first three comma-separated fields, stripped, are
`ConfigGroup`, `group` and `name`. Those lines are removed. The new lines
(`ConfigGroup,<group>,<name>,<device>,<property>,<value>`) go where the
first of them was. When there was none, they go after the group's last
line. When the group has no line, they go at the end of the file. Every
other line is kept byte for byte, comments included. The line ending is
the file's own: `\r\n` when the text contains one, else `\n`. The text
ends with a line ending. A preset `BF2` is untouched when `BF` is replaced.

### 8.6 Facade, errors and profile

- **`CAPABILITY_ROLES`** gains `ObjectiveTurret: (Role.objective_turret,)`,
  `LightSource: (Role.light_source,)` and `Channels: ()`, after `Autofocus`
  and before `Properties`. `available()` on the demo becomes `[XYStage,
  ZStage, Camera, Shutter, Autofocus, ObjectiveTurret, LightSource,
  Channels, Properties]`.
- **`_build`**:

  ```python
  MMObjectiveTurret(
      core,
      label,
      executor,
      requires_confirm=profile.safety.turret_requires_confirm,
      pixel_sizes_um=profile.camera.pixel_size_um,
      autofocus=lambda: _owned(owner, "ObjectiveTurret")._lookup(Autofocus),
  )
  ```

  ```python
  MMLightSource(
      core, label, executor, intensity_property=profile.light_source.intensity_property
  )
  ```

  ```python
  MMChannels(
      core,
      group,  # choose_group(...); None: no Channels, and nothing is built
      executor,
      objective_turret=roles.get(Role.objective_turret),
      exclude=profile.channels.exclude,
      config_path=profile.config_path(),
      light=lambda: _owned(owner, "Channels")._lookup(LightSource),
  )
  ```

  `choose_group`'s warnings are logged when `Channels` is built. Nothing a
  capability stores is a bound method of itself (FM-92).
- **`require(Channels)`** with no group raises `CapabilityMissingError(
  "Channels", None, available=<the core's groups>, hint="no config group is
  named for channels; name one in the profile's [channels] group")`.
  `CapabilityMissingError` gains the keyword `hint: str = ""`. When it is
  set, it replaces the `[roles.assign]` advice.
- **`state()`** gains `objective: str | None`, `channel: str | None` and
  `light_intensity_native: float | None`. Each is `None`, with an `errors`
  entry when its read fails, and `None` without the capability.
  `describe()` and `cli._status_line` are unchanged.
- **No `_STOPPED` entry**: nothing the three capabilities move can be
  stopped.
- **Profile**:

  ```toml
  [light_source]
  intensity_property = ""   # the light_source device's property that sets how bright; "" = found by hints (m2 §8.4)

  [channels]
  group = ""                # the config group whose presets are channels; "" = chosen (m2 §8.5)
  exclude = []              # device labels that capture() never records
  ```

  ```python
  class LightSourceSection(_Section):
      intensity_property: str = ""
  ```

  ```python
  class ChannelsSection(_Section):
      group: str = ""
      exclude: list[str] = []
  ```

  ```python
  class Profile(BaseModel):  # gains
      light_source: LightSourceSection = LightSourceSection()
      channels: ChannelsSection = ChannelsSection()
  ```

  Validation, each a `ProfileError` naming the key: `channels.group` must
  be printable ASCII, with no comma and no leading or trailing whitespace,
  since it is written into the `.cfg`. Each `channels.exclude` entry must
  not be empty. `Profile.demo()` and `profiles/demo.toml` stay equal: the
  demo sets neither section, and `demo.toml` carries the lines above,
  commented. `profiles/README.md` documents the three keys and the
  `turret_requires_confirm` behaviour of §8.3. `nikon-ti2.toml` sets none
  of them: its one group is chosen, and `DiaLamp`'s `Intensity` is found by
  the hints.

### 8.7 Kinds of call (extends M1 §13)

| Kind | Calls |
|---|---|
| read | `ObjectiveTurret.labels`, `current`, `pixel_size_um`; `LightSource.device`, `intensity_knob`, `intensity_native`, `is_on`; `Channels.group`, `names`, `current`, `preset` |
| action | `ObjectiveTurret.select` (a compound: the switch-off, then a motion that cannot be stopped); `LightSource.set_intensity_native`, `set_on(True)`; `Channels.select` (a motion that cannot be stopped), `capture` |
| safe | `LightSource.set_on(False)` |

### 8.8 FakeCore and tests

**FakeCore** (extends M1 §8; each behaviour as measured in §8.1):
- `setStateLabel(label, text)`. An unknown text raises the measured
  `RuntimeError`.
- **Config groups.** The attribute `config_groups: dict[str, dict[str,
  list[tuple[str, str, str]]]]`. The calls:
  - `getAvailableConfigGroups()` and `getAvailableConfigs(group)`, sorted
    as the demo returns them;
  - `getConfigData(group, name)`, a tuple of tuples;
  - `defineConfig(group, name, device, prop, value)`, which merges and
    creates the group;
  - `deleteConfig(group, name)`;
  - `setConfig(group, name)`, which applies each setting in order through
    the property store, a `Core,Shutter` setting through the shutter slot;
  - `getChannelGroup()` and `setChannelGroup(group)`.

  The errors are the measured `ValueError`s. `defineConfig`, `deleteConfig`,
  `setConfig` and `setChannelGroup` are logged.
- `demo_like()` gains the demo's six groups with the measured settings, and
  the channel group `Channel`. `tests/test_fake_core_demo.py` compares
  groups, presets and settings with the demo.
- **Two quirks of the Ti2-E (session 2, §8.9, #330)**, each set through a
  method that validates its labels (FM-119), off by default, and kept
  *assumed* in their shape: the Ti2-E showed them once. §8.11 D6 fixes the
  signatures and what they change. The demo shows neither (measured: 0 of
  200 `Exposure` sets read back otherwise; its turret turns at once and
  does not move Z), so `tests/test_fake_core_demo.py` has nothing to
  compare them with.
  - `make_reads_stale(device, prop, *, reads)`: the next `reads` reads of
    that property after each set return the value it held before the set.
  - `move_z_on_turn(label, delta_um, *, escape_um=None, escape_reads=0)`: a
    turn of any `State` device to `label` moves the focus slot's Z by
    `delta_um`, and the next `escape_reads` reads of that Z return
    `escape_um`.

**Ti2-E layout for unit tests.** Use the session-0 list
(`tests/data/nikon-ti2-devices.json`), with `Nosepiece`'s six labels,
`DiaLamp` as a `Shutter` with a writable `Intensity` `"501"` and no limits,
the filter turrets' labels, and the `Channel` group with `BF` copied from
`profiles/nikon-ti2.cfg`.

**Contracts** (`tests/contracts/`, through `Microscope` only):
- **The envelope** also records the objective and the channel at the
  start. The put-back restores the objective (`select(…, confirm=True)`)
  after Z, and the channel after the exposure, each only when it differs.
- **On `hardware`, the turret never turns, and no capture is ever run.**
  `select` of the current label (which sends nothing) and of the current
  preset are the only selections. A capture writes the stand's `.cfg`.
- `test_objective_turret.py`:
  - `current()` is in `labels()`;
  - without `confirm`, `select` raises `SafetyRefusedError` whose
    `how_to_force` names `confirm=True`, and on `fake` nothing is in the
    log;
  - an unknown label raises `ValueError`;
  - on `fake` and `demo`, a select of another label returns it and reads
    back;
  - in dry-run, nothing turns;
  - `pixel_size_um()` is 0.0 or the profile's value divided by the
    magnification factor (0.65 on `fake` for `Nikon 10X S Fluor`, whose
    factor is 1.0).
- `test_light_source.py`: `intensity_knob()` is `None` or names the role's
  device. Without a knob, `intensity_native()` is `None` and the set raises
  `HardwareError`. With a knob, setting the current value reads back. When
  `is_on()` is `None`, `set_on` raises. The demo's `LED` has neither.
- `test_channels.py`:
  - the group is named for channels;
  - `current()` is in `names()` or is `None`;
  - selecting the current preset reads back;
  - on `fake` and `demo`, `FITC` then back;
  - an unknown name raises `ValueError`;
  - in dry-run, nothing changes;
  - on `fake` and `demo`, `capture("smc-contract")` gives a preset that has
    no `Objective` setting, has `Core,Shutter`, and has `saved_to=None`.

**Unit tests** (the Ti2-E layout and the pure module):
- `tests/unit/test_mm_objective_turret.py`;
- `tests/unit/test_mm_light_source.py`;
- `tests/unit/test_mm_channels.py`;
- `tests/unit/test_channel_presets.py`, the pure module, with no core.

**Demo**: `tests/test_turret_light_channels_demo.py`.

The plan of #14 names each test.

### 8.9 Measured on the Ti2-E, and not verified (#330)

Session 2 (2026-11-03, report #334, `docs/hardware/nikon-ti2.md`) read what
this section used to list, through pymmcore directly (a read every
10–50 ms), with no sample, the 10× in place, and PFS off except where the
row says. Each row says what the layer does about it.

| Read | Result | The layer |
|---|---|---|
| A turn of `Nosepiece`, 10× → 4× → 10× | `setState` returns after about 1.0 s (0.99 s, 1.07 s); `deviceBusy("Nosepiece")` never reads true. | The call is the wait: the turn's busy poll returns at the first read, as the XY and Z moves do (#221). |
| PFS during a turn | PFS on and searching (no sample): **the stand switched it off by itself**; `PFS` enabled read false at the call's return. A locked PFS was not tried. | §8.3's switch-off stays (§8.3, "Why the switch-off"). |
| Z during a turn | Z read 561.5 µm before and **500.0 µm when the call returned**; 0.02 s later 570.26 µm, settled at 570.22 µm with the 4× (+8.7 µm); back with the 10× it read 561.48, then 561.44 µm. The operator heard a different mechanical noise and a beep. | A read of Z at the return can catch the stand mid-move: §8.11 D2 reads Z until still and reports before and after. |
| `PFSOffset` across a turn, PFS off (session 3, #405 F3/F5) | **The stand puts an offset of its own in place.** From 0 (the floor), the turn to the 10× read **9370** and the turn to the 40× **17794**; the turn to the 4× left 0 as it was. The 10× read 9370 at power-on too, although session 2 had left 8141; the contracts' turns left 9370 behind. smc wrote nothing across these turns, and with PFS off an offset write moves no Z (§12.10.1). Each read came seconds after the turn, between commands: **when the offset lands against the call's return was not read**. The reading "an offset per objective, put back on a turn, none for the 4×" is inferred from these reads (`nikon-ti2.md`, *Session 3*). | §8.11 D8: the drive's text names the change; §12.10 D7: `--floor` comes after the last turn. |
| `DiaLamp` `Intensity` | Limits [0, 2100]; 501 at the start. Set to 210, read 0.3 s later: 210. Set back to 501, read **9 ms later: 210 (the previous value)**; read 14 s later: 501. | §8.11 D1: the readback is re-read until fresh, within a bound. |
| `setConfig("Channel", "BF")` | Returns after **7.1 s**; no device of the preset read busy at any point. Before it, `LightPath` read `1-EYE` and `CondenserTurret` `1-OPEN` (the preset sets `4-L100` and `7-N2`). Afterwards every value matched the preset, `Camera-1` `Exposure` reading `10.0000` for the preset's `10`. Which device takes the time is unknown. | §8.11 D5: a 7 s select is acceptable on the page; the page's other actions are refused meanwhile. |
| `getCurrentConfig("Channel")` after that select | Read `""` (FM-100, confirmed on the stand). | `Channels.current()` compares numbers as numbers (§8.5); not run through smc on the stand. |
| `getMagnificationFactor()` | 1.0 at the lever's 1.0× and 1.5 at 1.5× (the property is `Magnification`, `1.0x` and `1.5x`); the pixel size reads 0 (no preset). | §8.10's fallbacks are right at both positions. |

**Not verified**, for the next visit (`/hardware-session` copies this list
into its checklist):
- the **dip to 500.0 µm**: whether it is a fixed position (an escape) or
  relative to Z, and what a turn does at the heights the stand will be at.
  With no sample and the objective clear, turn 10× → 4× → 10× at Z about
  561, 1500 and 3000 µm, reading Z every 10 ms from the call until 1 s
  after it. **Before the first turn under a plate** (Z about 4195 µm at
  the 10×), the same read at the stage's own height without the plate,
  and PFS off. A turn under a plate is not run until those reads are in;
- the same turn **with a locked PFS**, no sample, to see whether it ends
  locked, off or searching, and what Z does;
- how long `Intensity` lags a set (a read every 10 ms for 2 s after a set,
  up and down), and whether `getShutterOpen` lags `setShutterOpen`: §8.11
  D1 does not cover a shutter's state until it is measured;
- whether `getStateLabel("Nosepiece")` reads the new label at the return
  of a turn (§8.11 D3);
- which device of `BF` takes the 7.1 s: set `CondenserTurret` `7-N2` and
  `LightPath` `4-L100` one at a time from `1-OPEN` and `1-EYE`, timing each;
- that `Channels.select("BF")` through smc returns `BF` and `current()`
  reads `BF` (the contracts skipped it: the stand matched no preset).

### 8.10 The magnification factor in the pixel size (#167)

Extends §8.3 and M1 §3, §6, §8. It applies to both pixel sizes that the
profile feeds: `Camera.pixel_size_um()` (its fallback) and
`ObjectiveTurret.pixel_size_um()`.

**Measured** on the demo on 2026-10-20 (pymmcore-plus 0.18.1), in throwaway
cores:

- The demo has no `Magnifier`, and `getMagnificationFactor()` reads 1.0.
  `DemoCamera`'s `DOptovar` is a `Magnifier`: `Position` takes `1x` and
  `1.6x`. At `1.6x` the factor reads 1.6 and `getPixelSizeUm()` reads 0.625,
  the pixel-size preset's 1.0 divided by 1.6. Binning 2 doubles it (1.25).
  Two optovars at `1.6x` read 2.56: the factor is the product of every
  `Magnifier` device (FM-13). With no pixel-size preset, `getPixelSizeUm()`
  reads 0.0 whatever the factor.
- **MMCore passes an invalid factor on.** An optovar whose magnification is
  0, -1.5, NaN or infinity gives a factor of that value, and
  `getPixelSizeUm()` then reads infinity, -0.667, NaN or 0.0. Infinity is
  greater than 0, so a check of `getPixelSizeUm() > 0` alone reports it as
  measured.
- The Ti2-E's `IntermediateMagnification` is a `Magnifier` (session 0). Its
  factor reads **1.0 at `1.0x` and 1.5 at `1.5x`** (session 2, §8.9), so
  the rule below holds at both lever positions.

**The rule.** The profile's `[camera] pixel_size_um` is the size at binning 1
and a magnification factor of 1.0, the convention of MMCore's own pixel-size
presets. Wherever the layer uses it, it divides by the factor that MMCore
reports, as MMCore does for its own value:

- `Camera.pixel_size_um()`: MMCore's value when it is finite and above 0 (it
  already includes the binning and the factor, so it is never divided
  again); else the profile's value for the current objective times the
  binning, divided by the factor.
- `ObjectiveTurret.pixel_size_um(label)`: the profile's value for `label`
  divided by the factor, for any label (the lever does not depend on the
  objective). The factor is read only when the profile has a size for the
  label, so an objective the profile lacks is 0.0 without a device read.
- A factor that is not finite and above 0 gives 0.0 (unknown) and one WARNING
  per call, like a bad profile entry. A factor that cannot be read raises
  (`HardwareError` naming the role and `getMagnificationFactor()`, FM-79);
  `Microscope.state()` is the tolerant layer.
- **No profile key.** A stand whose changer MMCore cannot read (a manual tube
  lens that is no `Magnifier` device) reads 1.0 and is wrong, and the layer
  cannot know it: its sizes must be measured with the changer in place. A
  size measured (#22) with the lever at 1.5× is entered multiplied by 1.5.
  A key that names the factor of the calibration can come with #22.

**Code.** A module function `_magnification_factor(core, for_what) -> float
| None` in `backends/mm.py` (the factor, or `None` with the WARNING), called
from both methods through the role's fault-translating core. In the
turret it runs inside `Executor.read`. `FakeCore` gains
`getMagnificationFactor()`, which multiplies `magnification[label]` (default
1.0) over its `Magnifier` devices and raises a device's `failing` entry, and
`demo_like(extra_devices=())`, which appends devices after the demo's own;
`tests/test_fake_core_demo.py` compares the product with the demo's.

**Not verified.** That a size calibrated with the lever at 1.5× is entered
multiplied by 1.5 (the factor now reads 1.5 there, §8.9, but no calibration
was run at 1.5×). A stand whose changer MMCore cannot read still gets a
pixel-size preset keyed to the lever in its configuration, not code.

### 8.11 A read after a set, a turn that moves Z, a slow select (#330)

Extends §8.3–§8.5 with what session 2 measured (§8.9). **Status**: design
for #330 (2026-11-05). Measured on `FakeCore` and the demo on 2026-11-05
(pymmcore-plus 0.18.1, pymmcore 12.5.0.75.0), `main` at 5dd657e, unless a
line names another source.

#### 8.11.1 What is known

- **Today a stale read is the answer.** `MMLightSource.set_intensity_native`
  reads back once, at once. On the Ti2-E layout (`ti2_core()`) with a
  `DiaLamp` whose first three reads after a set return the previous value,
  `set_intensity_native(210.0)` from 501 returns **501.0**, the old value,
  in 0.000 s (a prototype quirk on `FakeCore`). `drive.switch_light`
  compares `is_on()` with what it sent, so a `State` that lags the same way
  would print `light on, although off was sent`; whether it lags is not
  measured (§8.9).
- **A bounded re-read finds the value.** The same quirk read every
  `POLL_INTERVAL_S` (10 ms) returns 210.0 on the fourth read, 0.038 s after
  the set; one that never turns fresh returns its last read after the
  window, which is the previous value.
- **The demo never reads stale and formats short.** 0 of 200 `Exposure`
  sets read back otherwise. It reads a number with four decimals
  (`'50.0000'`), while MMCore writes a double with six
  (`setProperty(camera, "TestProperty1", 0.1234567)` raises `Cannot set
  property "TestProperty1" to "0.123457"`, outside its limits [-0.1, 0.1],
  so the text shows the format). A readback compared with the value that
  was set needs a tolerance of that order.
- **During a slow select, other actions are refused and reads go
  through.** With `Channels.select` inside a `setConfig` of 0.6 s on
  `FakeCore.demo_like()`, a Z jog from another thread was refused after
  about 0.06 s with `MotionInProgressError` (`channels Channel Cy5 is still
  moving…`, naming the select); `ZStage.position_um()` and a shutter close
  returned at once (FM-16: reads and the safe calls never take the lock).
- **The demo's turret returns at once and reads not busy**, and does not
  move Z, so the demo cannot show D2. The Ti2-E layout's `FakeCore` can,
  once it has the quirk of D6.
- `ZStage.position_um()` is a read through the facade, with no lock, in
  dry-run and while halted (M1 §13).

#### 8.11.2 Decisions

**D1. A readback that stands for what the device holds is re-read until it
is fresh, within a bound (FM-188).** Fresh is: the read is what was set. A
bound passes: the last read is returned, and a WARNING says so.

```python
#: How long a readback that is not what was set is re-read before it is
#: believed (m2 §8.11 D1). Assumed: the Ti2-E's DiaLamp read the previous
#: value 9 ms after a set and the new one 0.3 s after, and nothing longer
#: was measured. The visit that measures the lag may set another value.
FRESH_READ_WINDOW_S = 1.0
```

- A module function in `backends/mm.py`, `_reread(read, is_fresh) ->
  _Reread[T]`, with `_Reread` a frozen dataclass `(value, fresh, waited_s)`.
  It reads once; when that is fresh, it returns at once, so a device that
  reads fresh costs nothing. Otherwise it reads again every
  `POLL_INTERVAL_S` until a read is fresh or `FRESH_READ_WINDOW_S` (read at
  the call, so a test can shorten it) has passed since the first read, and
  returns the last read. A read that raises propagates. It logs nothing: its
  callers do, so that a caller that already warns does not warn twice.
- **Where it applies.** `MMLightSource.set_intensity_native` (fresh: the
  read is the value set within `math.isclose(read, value, rel_tol=1e-6,
  abs_tol=1e-4)`, which covers the demo's four decimals and MMCore's six),
  `MMChannels.select`'s match (D4) and the turret's label (D3).
- **Where it does not.** `LightSource.set_on`, `Shutter.set_open`: a
  shutter's state was not measured to lag (§8.9), and a re-read after an
  off, a safe call, would wait the bound for a light the operator was told
  is off. The helper is written so that one line makes either use it when
  the visit shows a lag. `Properties.set` is the escape hatch and returns
  its readback as one read: a lamp's `Intensity` set through it meets §8.9
  as it is, which this issue leaves alone.
- **The cost.** None when the read is fresh. When it is not, the action
  holds the lock for at most the bound more. A device that holds another
  value than it was given (a quantised intensity, a set that is ignored)
  pays the bound on every set, then the WARNING: that is the price of not
  returning an old value as the new one, and the answer is a profile that
  sets values the device holds.
- **The return stays the last read.** `LightSource.set_intensity_native ->
  float` is unchanged (nothing outside the layer calls it): a caller that
  compares it with what it sent sees the doubt, as before.
- **Words** (`label` the role's device, `knob` the property, numbers `:g`):
  - fresh after a wait, INFO: `light_source 'DiaLamp': Intensity read 501
    right after the set of 210, then 210: a read right after a set may be the
    previous value (FM-188)`;
  - not fresh, WARNING: `light_source 'DiaLamp': Intensity still reads 501
    after 1 s, although 210 was set; the read may predate the set, or the
    device did not take it`.

**D2. A turn reads Z before and, once Z is still, after, and says what the
stand did.** `MMObjectiveTurret.__init__` gains the keyword `z_stage:
Callable[[], ZStage | None] = lambda: None`; `_build` passes `lambda:
cast("ZStage | None", _owned(owner, "ObjectiveTurret")._lookup(ZStage))`
(FM-92: nothing the capability stores is a bound method of itself, and the
lookup at the call raises once the facade is gone, as the autofocus's does).
In `_turn` the two lookups, the autofocus's and `z_stage()`, come right after
the already-at check and **before** the switch-off, so that a facade that is
gone raises before anything is switched off or sent. Then, unless in
dry-run, and only when `z_stage()` is not `None`:

1. Z is read before the send, with `ZStage.position_um()`.
2. The turn is the existing `executor.do`.
3. Once it returns, Z is read every `POLL_INTERVAL_S` until every read of
   the last `TURN_Z_SETTLE_S` lies within `TURN_Z_STILL_UM` of the first
   read of that stretch (a read outside it starts a new stretch), or the
   core timeout passes, or the stand is halted. The settled Z is the last
   read.
4. One line is logged (below). The turn's return value is unchanged.

```python
#: How long Z must have been still before a turn is reported (m2 §8.11 D2).
#: Assumed: Z was at its place 0.02 s after a turn returned (session 2, reads
#: 10-50 ms apart); the stand returned Z after the call did.
TURN_Z_SETTLE_S = 0.1
#: How far two Z reads may differ and count as still. Chosen: Z read 561.48
#: then 561.44 µm at rest (session 2).
TURN_Z_STILL_UM = 0.5
```

- **Why it waits.** The call returned while Z read 500.0 µm and came back
  within 0.02 s (§8.9). A report that read Z once at the return would give
  the stand's escape position as the result, and the next action would
  start while the stand was still moving Z. The wait is inside the
  compound, so other actions wait for the lock; a read, STOP and the safe
  calls go through.
- **The words** (positions `:.2f`, the delta `:+.2f`):
  - moved by more than `TURN_Z_STILL_UM`, INFO: `objective_turret: turned to
    '1-Plan Fluor 4x'; the stand moved Z from 561.52 µm to 570.22 µm
    (+8.70 µm)`;
  - moved by no more than that, INFO: `objective_turret: turned to
    '1-Plan Fluor 4x'; Z is 561.52 µm, as before`;
  - not still at the core timeout, WARNING: `objective_turret: Z was still
    changing 5 s after the turn (570.22 µm at the last read); the stand may
    still be moving it` (the timeout `:g`, in seconds);
  - halted during the wait, WARNING: `objective_turret: Z was not waited for:
    the stand was halted`;
  - a Z read that fails, before or after, WARNING: `objective_turret: Z could
    not be read around the turn (<_named(exc)>); the stand moves Z during a
    turn (m2 §8.9)`, and no INFO line. The turn is not failed by it: it
    was sent, and a read that only informs a report never turns a done turn
    into an error.
- **No Protocol change.** `ObjectiveTurret.select -> str` stays, and so does
  every stub that implements it. The report is the log line and the drive
  text of D7.
- **smc applies no Z correction** and checks no Z after a turn: the stand
  corrects Z itself (§13.2), and the Z soft limits guard the next move. A
  turn that ends outside them is seen in the line and refused at the next
  move.
- **The switch-off is untouched** (§8.3): it comes first, before Z is read.

**D3. The turret's label is re-read until it is the label asked for
(D1).** Whether `getStateLabel` reads the new label at the return of a turn
is not verified (§8.9): the turn's readback is one read today. Fresh: the
read equals `label`. Not fresh, WARNING: `objective_turret 'Nosepiece':
reads '2-Plan Fluor 10x' after 1 s, although '1-Plan Fluor 4x' was set; the
read may predate the turn, or the turret did not turn`; fresh after a wait,
INFO: `objective_turret 'Nosepiece': read '2-Plan Fluor 10x' right after
the turn, then '1-Plan Fluor 4x': a read right after a set may be the
previous value (FM-188)`. A turret that stays where it was costs the bound,
and `drive.set_objective` then says `although` as today (FM-155).

**D4. `Channels.select`'s match is re-read (D1).** The readback's
`self._mismatches(settings)` is read again until it is empty, within the
bound: a preset that holds a `DiaLamp` `Intensity` and returns quickly would
otherwise read its previous value and report `the stand reads otherwise`
for a select that worked. Fresh: no setting reads otherwise. The WARNING
`channels: <group> <name> selected, but the stand reads otherwise: …` is
unchanged and is the only one at the bound (the helper logs nothing); a wait
that ended fresh logs INFO `channels: <group> <name>: a setting read its
previous value right after the select (FM-188)`. A preset that names a
setting the stand never reads as set (a quantised value, a read-only
property) costs the bound, then the WARNING.

**D5. A 7 s `Channels.select` is acceptable on the page; no code.** The
7.1 s of `setConfig("Channel", "BF")` is the adapter blocking inside the
call (no device read busy), so there is no shorter wait to split it into,
and the layer cannot tell which device takes it. While it runs the lock is
held: the page's other actions are refused with the select's name
(`MotionInProgressError`, measured in §8.11.1), and reads (the status
strip), the light's off and STOP go through. A refusal that
names `channels Channel BF` and says to wait is the right answer to a
click during a light-path change, and 7 s is short against the 60 s lock
timeout. Which device takes it matters only if the owner wants the select
shorter: the visit times the two devices (§8.9), and the answer goes to
`docs/hardware/nikon-ti2.md`, not to the layer.

**D6. `FakeCore` gets the two quirks (§8.8).**

```python
def make_reads_stale(self, device: str, prop: str, *, reads: int) -> None: ...
def move_z_on_turn(
    self,
    label: str,
    delta_um: float,
    *,
    escape_um: float | None = None,
    escape_reads: int = 0,
) -> None: ...
```

- `make_reads_stale`: the next `reads` reads of `(device, prop)` after each
  write of a **different** value return the value held before that write;
  a write of the value it already holds leaves nothing stale; each write
  restarts the count. The reads it covers are `getProperty`, and the
  dedicated calls that read the same store (§8.1, "One store":
  `getStateLabel` for `Label`, `getState` for `State`, `getExposure` for
  `Exposure`). Every path that writes the property restarts it
  (`setProperty`, `setConfig`'s settings, `setState`, `setStateLabel`,
  `setExposure`). `reads` is at least 1, and `ValueError` for a device that
  is not loaded, a property it lacks, or a read-only one: a misspelt label
  would otherwise build a stand that reads fresh and a test that passes
  (FM-119). A shutter's open state is not a property in the fake and is not
  covered, as it is not measured.
- `move_z_on_turn`: a turn of **any `State` device** to the position named
  `label` (the labels of the loaded `State` devices, `ValueError` for one
  that is none) adds `delta_um` to the focus slot's Z, set on the store
  directly, so that the Nikon quirk (FM-12) does not trip; and the next
  `escape_reads` reads of the focus slot's position (`getPosition`) return
  `escape_um` before they return the stored Z. A turn to the position it is
  already at changes nothing. `escape_um` is given if and only if
  `escape_reads` is at least 1, else `ValueError`. Several labels may be
  registered. `ValueError` also with no focus slot filled.
- The Ti2-E session-2 values are the tests': `4×` +8.70 µm, `10×` −8.70 µm
  and an escape of 500.0 µm for a few reads.

**D7. The drive's text says what the stand did to Z.** `drive.set_objective`
reads `ZStage.position_um()` before `turret.select(...)` and after it, when
the stand has a focus drive and it is not a dry run; a read that raises
`HardwareError` counts as unknown, without keeping the exception (FM-98),
and the layer's WARNING has the cause. After the existing text, the
clause:

- `; the stand moved Z from 561.52 to 570.22 µm (+8.70 µm)` when the two
  reads differ by more than `_Z_MOVED_UM = 0.5` (chosen: the read noise at
  rest is 0.04 µm, session 2; the corrections measured are 8.7 µm);
- nothing when they do not, or when the stand has no focus drive, or in a
  dry run (nothing turned: the text is today's);
- `; Z could not be read around the turn` when either read failed.

The clause follows either of today's texts (`objective '4x'` and
`objective '10x', although '4x' was asked`), and `reached` is today's. It
is not a second mutating call: the two reads are reads (m5 §6.1).
`drive.set_objective`'s docstring changes from "reads nothing and sends
nothing" to "sends nothing" for a refused turn: Z is read first. The CLI
prints the text, the page shows it as the outcome; neither changes.

**D8. The drive's text also names a change of the autofocus offset (#407).**
Session 3 read an offset of the stand's own come back across a turn (§8.9:
0 → 9370 at the 10×, 0 → 17794 at the 40×), and smc's text said only what
Z did: the operator learned of it from `smc autofocus offset` or not at
all. `drive.set_objective` reads `Autofocus.offset_native()` where D7 reads Z,
and says what the two reads show. It is not a second mutating call (m5 §6.1).

- **When.** The stand has an `Autofocus` (`microscope.get`) and it is not a
  dry run (nothing turns). The first read comes just before
  `turret.select(...)` and the second just after `turret.current()`, beside
  D7's two Z reads; a turn that is refused or raises says nothing about the
  offset, as it says nothing about Z.
- **Compared exactly.** The offset is in the device's own units, no
  tolerance can be given without a measurement, and a device that holds an
  offset reads it back identically (`PFSOffset` read 9370 on every call of
  session 3; the seen offsets are compared the same way, §12.2 D1). Any
  difference counts.
- **The words**, a clause after D7's Z clause (or in its place), in the
  offset's own unit (`device units` when the profile names none); numbers
  `:g`:
  - changed: `; the autofocus offset went from 0 to 9370 PFSOffset units`;
  - either read failed: `; the autofocus offset could not be read around the
    turn (<cause>)`, with the cause as text on one line (the layer logs
    nothing for this read, so the cause goes here, FM-170; FM-98);
  - unchanged, no autofocus, or dry run: no clause: today's text.

  So `objective '2-Plan Fluor 10x'; the stand moved Z from 3006.96 to 2998.04
  µm (-8.92 µm); the autofocus offset went from 0 to 9370 PFSOffset units`.
  The clause follows either of today's texts (`although ... was asked`
  included); `reached` is today's.
- **It says what was read, not why.** "The stand's value for this
  objective" is an inference (§8.9), the 4× shows none, and the change is
  the stand's whatever its reason: smc wrote nothing across the turn.
- **Not chosen: a line in the turret's own INFO log.** Two places that
  disagree on what the offset was are worse than one. Nothing but the drive
  turns the turret for an operator, and `objective-offsets`' turns do not
  engage PFS.
- **Not chosen: waiting for the offset as D2 waits for Z.** When it lands
  against the call's return is not measured (§8.11.3); a wait would be a
  number made up. The two reads are one after the other, and FM-188 and FM-195
  are the risk (below).
- **The risk, named.** The second read comes after the turret's own wait for
  Z (D2), so on a stand with a focus drive it is at least `TURN_Z_SETTLE_S`
  (0.1 s) after the call returned (FM-195). A restore that lands after it is
  missed: the text is today's, which says nothing false. A read that
  catches the offset on its way would say a value that is neither: no stand
  has shown it, and the visit reads `PFSOffset` every 10 ms across a turn
  (§8.11.3).
- **`FakeCore` gets the stand's quirk** (D6's family), so that the text and
  the engage guard are tested against a stand whose turn changes the offset:

  ```python
  def set_offset_on_turn(
      self, label: str, offset: float, *, device: str | None = None
  ) -> None: ...
  ```

  A turn of **any `State` device** to the position named `label` (that moves
  it: a turn to the position it is at changes nothing) sets the autofocus
  offset to `offset`, held as the device holds it (`hold_whole_units`
  applies). `device` is the label of the loaded `Stage` that is the offset
  device (the Ti2-E's `PFSOffset`); `None` is the AutoFocus slot's own
  offset (the demo's). It is set on the store and not logged: the stand's
  own doing is no call of smc, and `log` stays what smc sent. Several labels
  may be registered, each with its offset, and the latest registration of a
  label wins. `ValueError` for a `label` that is no position of a loaded
  `State` device, a `device` that is no loaded `Stage`, and `device=None`
  with no device in the AutoFocus slot (FM-119: a misspelt label would build
  a stand that never restores, and a test that passes).

#### 8.11.3 Not verified

- D1's bound against the real lag (§8.9): 1 s is assumed, and the visit
  measures it. A shutter's state is covered by no re-read until it is
  measured.
- D2's settle against a turn at a plate's height or with a locked PFS: the
  escape at 500.0 µm may be fixed or relative (§8.9), and Z may need longer
  than the core timeout from Z about 4195 µm.
- D3: that the label is fresh at the return of a turn.
- D8: when `PFSOffset` reaches the stand's own value against the return of
  the call, and whether it passes through other values on the way. Session
  3 read it between commands. The visit reads `PFSOffset` with Z every 10 ms
  from the call until 1 s after it, PFS off, on 10× → 4× → 10× → 40× → 10×
  (`turn_reads.py` gains the read), and notes whether the first read after
  the return is the final one.
- That the Ti2-E reads the label, the lamp and Z fast enough for the
  windows: `POLL_INTERVAL_S` is 10 ms, and a read there costs 0.00–0.05 ms
  (session 2).

## 9. A probe that tells the truth (#48)

Extends M1 §10 and §7.6 of this document. **Status**: design for #48
(2026-10-20). Sources: the demo (measured 2026-10-20, pymmcore-plus 0.18.1,
pymmcore 12.5.0.75.0), session 0 on the Ti2-E (#126, the field note on #48)
and `nikon-control` (`scope/discover.py`, `scope/config_build.py`, read
2026-10-20). `smc discover --probe-adapter NAME` is the only command here
that initialises devices; nothing in this section moves a stage.

### 9.1 What is known

Measured on the demo unless a line names another source.

- **A hub lists its peripherals only after it initialises.** `SequenceTester`
  lists one device, `THub` (Hub). `getInstalledDevices("THub")` before
  `initializeDevice` raises `RuntimeError('Operation
  (GetInstalledPeripheralNames) not permitted on uninitialized device')`;
  after it, it returns a tuple of 14 names (`TCamera-0`, `TCamera-1`,
  `TShutter-0`, `TShutter-1`, `TXYStage-0` … `TSwitcher-1`).
  `getInstalledDeviceDescription` reads `N/A` for each, as on the Ti2-E.
- **A peripheral loads by its name although the adapter does not list it.**
  `loadDevice(name, "SequenceTester", "TCamera-0")` works, the loaded device
  already has its type (`getDeviceType`), and with `setParentLabel` to the
  hub all 14 initialise. The probe of #47 said `THub ok` and stopped.
- **The listing and the report agree on the demo's other hub.** `DemoCamera`
  lists 20 devices (`DHub` and 19); `DHub` reports the same 19 after it
  initialises. `Utilities` lists no hub.
- **Labels.** MMCore refuses a label that is empty, `Core`, or contains a
  comma (`Device label "a,b" contains reserved characters`); `XYStage:XY:31`,
  `a b` and `[x]` load. The probe uses the device name as the label, so a
  refused label is that device's failure.
- **A pre-init `Port`.** On a loaded, uninitialised device,
  `hasProperty(label, "Port")`, `isPropertyPreInit(label, "Port")` and
  `getProperty(label, "Port")` answer; `isPropertyPreInit` of a property the
  device lacks raises `RuntimeError`. The only device with one on the demo
  is `Utilities`' `Serial port DTR Shutter`: its `Port` reads `''`, not
  `Undefined`, and it initialises with no port, with `Undefined` and with
  `/dev/no-such-port` alike. `Undefined` is the issue's word for what the
  serial adapters publish; no installed adapter here shows it.
- **`argparse` and a port that starts with `-`.** `--port -x` is refused
  (`expected one argument`); `--port=-x` gives `-x`, and `--port=` gives `''`.
- **The Ti2-E** (session 0, #126): the hub `Ti2-E__0` reports 16 names, all
  with the description `N/A`; the 16, loaded under the hub without being
  initialised, gave all 16 types. The adapter lists two hubs, `Ti2-E__0` and
  `Ti2-Simulator`; with the stand connected the simulator fails `(Error
  message unavailable) (-10)`, which is expected. The probe of #47 said
  `Ti2-E__0 ok`, no peripherals, `Notes (0)`. `HamamatsuHam_DCAM` fails
  `Native module failed to load (6)` because DCAM-API is not installed; no
  Hamamatsu is fitted.
- **`nikon-control`** (read, not run): it initialises the hub, asks it
  (`getInstalledDevices`), loads each name under it and reports the ones
  that fail, and a hub that lists nothing falls back to the adapter's own
  list.

### 9.2 Models — `smc/discovery/models.py`

```python
ProbeStatus = Literal["ok", "failed", "not-probed"]
ProbeSource = Literal["adapter", "hub", "both"]
```

```python
class ProbedDevice(BaseModel):
    name: str  # also its label in the throwaway core
    type: str  # "Hub", "Camera"…; "Unknown" for a value MMCore gave and we lack
    status: ProbeStatus
    # who named it: the adapter's list, the hub's report, or both
    source: ProbeSource = "adapter"
    error: str | None = None  # failed: MMCore's own text, raw
    reason: str | None = None  # not-probed: why nothing was tried
    hint: str | None = None  # failed: what the error usually means (§9.4)
    port: str | None = None  # the Port set before it initialised (--port)

    @property
    def ok(self) -> bool: ...  # status == "ok"; a property, not a JSON key
```

```python
class ProbeResult(BaseModel):
    adapter: str
    devices: list[ProbedDevice] = Field(default_factory=list)
    # the adapter could not be probed, or nothing below its hub was
    error: str | None = None
    notes: list[str] = Field(default_factory=list)  # one line each, §9.3
    causes: list[str] = Field(default_factory=list)  # ordered, when a hub failed
    complete: bool = True  # False: the child stopped before the last device (§9.5)
```

`AdapterInfo` gains `list_meaning: str | None = None` (§9.4). In
`inventory.json`, `ProbedDevice` loses the key `ok` and gains `status`; no
code reads the old key, and the survey folders are never committed.

A `not-probed` row is not a failure and not a pass: nothing was tried, and
`reason` says why. That is the whole point of the issue: a row never says
`ok` for something nobody initialised.

### 9.3 The walk — `smc/discovery/mm_inventory.py`

```python
def walk_adapter(
    core: CMMCorePlus,
    name: str,
    *,
    port: str | None = None,
    progress: Callable[[str], None] | None = None,
    checkpoint: Callable[[ProbeResult], None] | None = None,
) -> ProbeResult: ...
```

```python
def probe_adapter(
    name: str,
    *,
    port: str | None = None,
    progress: Callable[[str], None] | None = None,
    checkpoint: Callable[[ProbeResult], None] | None = None,
    core: CMMCorePlus | None = None,
) -> ProbeResult: ...
```

`walk_adapter` works on a core the caller owns and **leaves every device it
loaded loaded**, so that #15 (`smc config build`) builds on it and keeps what
initialised. `probe_adapter` opens a `CMMCorePlus()` when `core` is `None`,
walks, and releases the core in a `finally` (`close_core`), as today. `core`
exists so that a test hands it a stub (FM-40: no test lists a vendor
adapter). A new public name, `walk_adapter`, goes in `__all__`.

The rules, in this order. Each device's row is appended as soon as the device
is done (§9.5).

1. **Nothing to probe.** Adapter not installed: `error="adapter not
   installed"`. Enumeration error: `error` is that text. Nothing listed:
   `error = f"{name} lists no devices: " + diagnosis.empty_list_meaning(name)`.
   No row in any of the three.
2. **Hubs first, one hub.** The listed devices of type `Hub`, in listing
   order. Each one: `progress(f"probing {name}/{hub}")`, `loadDevice(hub,
   name, hub)`, rule 3, `initializeDevice(hub)`. The first that initialises is
   *the hub*; the others are not tried: `not-probed`, reason `hub <the hub>
   of this adapter is already initialised; a probe connects one hub`. (The
   Ti2-E lists the stand first and the simulator second; on the stand the
   simulator is no longer tried, and when the stand is off and the simulator
   runs, the stand's row is `failed` and a note says `hub <failed> did not
   initialise but <hub> did: what follows is <hub>'s`.) A hub that raises is
   `failed`, with MMCore's text in `error` and `hint` from §9.4.
3. **A pre-init `Port`** (any device, hub or not), after `loadDevice` and
   before `initializeDevice`: the device *needs a port* when
   `core.hasProperty(label, "Port")` and `core.isPropertyPreInit(label,
   "Port")` and `str(core.getProperty(label, "Port")).strip().casefold()` is
   `""` or `"undefined"`. Without `port`: `not-probed`, reason `needs a port:
   pass --port COMx to probe it`, and `initializeDevice` is never called.
   With `port`: `core.setProperty(label, "Port", port)` first, the row's
   `port` is set, and an exception from the set is `failed`. An exception
   while *reading* the property counts as "needs no port": the device is
   initialised, and a failure is reported raw (unknown beats guessed). When
   `port` was given and no device needed it, a note: `--port <port> was not
   used: no device of <name> needs a port`.
4. **The hub's peripherals.** When a hub initialised:
   `progress(f"probing {name}/{hub}: listing its peripherals")`, then
   `reported = [str(n) for n in core.getInstalledDevices(hub)]`, without
   duplicates and without a name already loaded. If the call raises, a note
   `hub <hub> did not list its peripherals: <error>`, and `reported` is
   empty.
   - **`reported` is not empty**: the candidates are the reported names, in
     the hub's order, `source` `both` when the adapter also lists the name
     and `hub` when it does not. The listed non-hub devices the hub does not
     report get a `not-probed` row, after the candidates, in listing order:
     reason `listed by the adapter but not reported by the hub <hub>: not
     attached, or not detected`. They are not initialised.
   - **`reported` is empty and the adapter lists other devices**: those are
     the candidates (`source` `adapter`), and a note `hub <hub> reports no
     peripherals: the adapter's own list was probed`. Some hubs do not
     detect, and their devices still work.
   - **`reported` is empty and the adapter lists no other device**:
     `error = f"hub {hub} initialised but reports no peripherals, and {name}
     lists no other device: nothing below it was probed; " +
     diagnosis.empty_list_meaning(name)`. The hub's row stays `ok`.

   Each candidate: `progress(f"probing {name}/{device}")`, `loadDevice(device,
   name, device)`, `setParentLabel(device, hub)`, rule 3, `initializeDevice`.
   `type` is the listed one, or `device_type_name(core.getDeviceType(device))`
   read after the load (`"Unknown"` when the load failed). Any `Exception` is
   `failed`: `error` is `str(exc)`, `hint` is `error_hint(name, error)`.
5. **No hub listed.** Every listed device is a candidate, with no parent
   (`Utilities`; as today).
6. **Hubs listed, none initialised.** Every listed non-hub device is
   `not-probed`, reason `its hub <first listed hub> was not initialised`.
   Nothing is loaded for them.
7. **Causes.** `causes = diagnosis.hub_failure_causes(name)` when at least
   one hub row is `failed`. A hub that `needs a port` is not a failure.
8. **Order of the rows**: hubs (tried or not, in listing order), the
   candidates in order, then the listed devices the hub does not report.
9. `KeyboardInterrupt` and `SystemExit` pass through. Every other exception
   of a single device is that device's `failed`.

### 9.4 Diagnosis — `smc/discovery/diagnosis.py` (new)

Pure: no `pymmcore_plus`, no pydantic. The Nikon registry
(`smc.hardware.backends.nikon`) is imported **inside the functions**, never at
module level: `nikon.py` imports `smc.discovery` (§7.3), so a module-level
import closes a cycle.

```python
GENERIC_EMPTY_LIST: str
GENERIC_HUB_CAUSES: tuple[str, ...]


def empty_list_meaning(adapter: str) -> str: ...
def listing_meaning(adapter: str, device_count: int) -> str | None: ...
def hub_failure_causes(adapter: str) -> list[str]: ...
def error_hint(adapter: str, error: str) -> str | None: ...
```

- `empty_list_meaning`: for a Nikon adapter (`stand_for_adapter`), the stand's
  `empty_list_meaning` (§7.2); otherwise `GENERIC_EMPTY_LIST`: `the adapter
  lists no devices: if it asks a vendor SDK what is attached, the SDK could
  not be reached; otherwise this PC offers none`.
- `listing_meaning`: `empty_list_meaning(adapter)` when `device_count` is 0;
  the stand's `empty_list_meaning` for a Nikon stand whose list is fixed
  (`dynamic` is False: it says the list proves nothing about what is
  connected), whatever the count; otherwise `None`. `list_adapters` sets it
  on each `AdapterInfo` it builds, and the report prints it (§9.6).
- `hub_failure_causes`: for a Nikon adapter, `[ONE_CONNECTION, *stand.causes]`
  (§7.2); otherwise `GENERIC_HUB_CAUSES`, in this order: another program holds
  the device (vendor software, another smc or pymmcore session; close them and
  run this again); the device or its controller is off, was powered in the
  wrong order, or its cable, USB port or serial port is wrong; the vendor
  driver or runtime DLL is missing, or is not the version the adapter was
  built for (on Windows, `smc discover` shows what the loader says for an
  adapter that fails to load). The texts may be reworded; the order may not.
- `error_hint`: a table of `(fragment, hint)`, matched on `str.casefold`.
  One entry today: `native module failed to load` → `the vendor runtime the
  adapter wraps could not be loaded: it is not installed on this PC, or is
  not the version the adapter was built for`, and for `HamamatsuHam` the
  words `(DCAM-API)` are added. The only vendor runtime named is the one measured (#126);
  another adapter gets the sentence without a name. No match: `None`.

### 9.5 The child, the parent and `--port`

- `check_port(port: str) -> str` in `mm_inventory.py`: the stripped value;
  `ValueError` when it is empty, longer than 200 characters, or holds a
  character below U+0020 or U+007F (FM-32, FM-78).
- `mm_section(..., port: str | None = None)` adds `--port=<port>` to the
  child's command as **one** argument (§9.1); `inventory(..., probe_port:
  str | None = None)` passes it, and raises `ValueError` when `probe_port` is
  given without `probe_adapter`.
- `_mm_child.py` takes `--port` and calls `probe_adapter(args.probe,
  port=args.port, progress=_progress, checkpoint=…)`. The checkpoint sets
  `result.probe` to the snapshot it is given and writes the result file; an
  `OSError` there is swallowed after a `_progress` line, since Windows
  antivirus can hold a file just written (FM-24) and a failed intermediate
  write must not end a probe. The final write is not guarded, as today.
- **A partial result survives a crash (FM-02).** `walk_adapter` calls
  `checkpoint` with a deep copy of the result, `complete=False`, once after
  the listing and before the first device, and after every device row. The
  value it returns has `complete=True`. A child that dies mid-probe leaves the
  last checkpoint in the file: `mm_section` returns it (`probe` is `None` only
  when the child died before the first checkpoint), and the note it already
  adds names the device that was running. A Ti-E probe that crashes on
  `TIDiaLamp` (§7.1) then keeps the other seventeen rows.

### 9.6 Report — `smc/discovery/report.py`

The `Probe of <adapter>` table has the columns `device`, `type`, `from`
(`adapter`, `hub`, `both`), `result` and `detail`. `result` is `ok`, `FAILED`
or `not probed`. `detail` is: for `failed`, `error`, then a dash and the
`hint` when there is one; for `not-probed`, `reason`; for `ok` with a `port`,
`port <port>`. When `complete` is False the title ends with the words `INCOMPLETE: the
adapter child stopped before the last device`. After the table, in this
order: the `ERROR` row as today when `error` is set; one line per
note, `- <note>`; and, when `causes` is not empty, `In order of
likelihood:` then one numbered line per cause, `1. <cause>`. An `AdapterInfo` with a `list_meaning` gets one
row under its devices, `meaning: <text>`, in the `detail` column of the
listing. Every cell and line is built with `_t` (text, never markup, FM-36).

### 9.7 CLI — `smc discover`

`--port TEXT`: the serial port for devices that need one. Without
`--probe-adapter` it is a usage error (exit 2, `✗ --port needs
--probe-adapter`); a value `check_port` refuses is a usage error with its
message. The banner reads `Probing <adapter> (port <port>): the stand must be
powered and its vendor software closed.` with `(port <port>)` only when given.
The exit code of `discover` is unchanged: a probe that finds failures is a
finding, not an error.

### 9.8 Files and sequencing

`smc/discovery/models.py`, `mm_inventory.py`, `_mm_child.py`, `report.py`,
`__init__.py` (the `probe_port` argument), `diagnosis.py` (new), `cli.py` (the
`discover` command only) and `docs/hardware/microscope-pc-setup.md`. After #12
(merged). #159 edits `cli.py` too, in `doctor`, and #15 adds a command there:
the plan of #15 waits for this one, since it reuses `walk_adapter`.

### 9.9 Tests

The plan of #48 names each test. In outline: a `_ProbeCore` stub in
`tests/unit/test_probe_walk.py` carries every rule of §9.3 (a port that is
`Undefined`, a hub that reports less than the adapter lists, a hub that will
not initialise, two hubs, checkpoints); the demo shows the `SequenceTester`
hub's 14 peripherals and `Utilities`' serial shutter, which needs a port; the
pure module has its own unit test; `tests/test_discover.py`'s crash test now
expects a partial probe, not `None`.

### 9.10 Not verified

- Everything on a real hub: that a Ti2 hub reports its 16 names through
  `getInstalledDevices` *inside a probe* (session 0 did it in a script of its
  own); that 16 initialisations fit the child's 180 s (`--timeout-s` raises
  it); that the simulator is second in the real listing (§9.1 reads it from
  #126's table), which rule 2 does not depend on.
- The `Undefined` sentinel: no installed adapter here shows it (`Utilities`
  shows `''`). Rule 3 treats both as unset.
- That a serial adapter accepts the `Port` it is given: some adapters restrict
  the allowed values of the property.
- That Windows antivirus ever holds the result file between two checkpoints
  (§9.5's guard is a precaution, FM-24).
- What `TIDiaLamp` does inside a probe on the Ti-E (§7.1): the checkpoints
  keep the other rows, and the crash itself is a Ti-E session's to read.

## 10. A configuration from the devices that answer (#15)

Extends M1 §9 and §10, §7 and §9 of this document. **Status**: design for
#15 (2026-10-20). Sources: `nikon-control` (`scope/config_build.py` and the
`build` command of `scope/cli.py`, read 2026-10-20), the demo (measured
2026-10-20, pymmcore-plus 0.18.1, pymmcore 12.5.0.75.0) and
`profiles/nikon-ti2.cfg`, which `nikon-control`'s builder wrote and session 0
loaded. `smc config build` initialises every device of one adapter, as
`smc discover --probe-adapter` does; nothing in it moves a stage.

### 10.1 What is known

- **`nikon-control`'s builder** loads the hub, asks it for its peripherals,
  loads them one at a time keeping those that initialise, adds a camera,
  resolves the roles and writes a `.cfg` that carries over eight directives
  of the file it replaces. What is not ported, and why:
  - it ran in its own process, so a crash on a device (the Ti-E's
    `TIDiaLamp`, §7.1) lost the build and said nothing about which device;
  - it renamed devices (`safe_label`: the part after a `:`, commas and `#`
    removed); the walk of §9.3 loads each device under its own name, and a
    name MMCore refuses as a label is that device's failure;
  - it tried every installed camera adapter when none was named;
  - it carried `Delay` and `FocusDirection` lines and every preset without
    looking at the devices they name, and did not carry `Label` lines, so a
    preset that selects a cube by a label someone typed lost that label.
- **Measured on the demo.**
  - `walk_adapter(core, "DemoCamera")` initialises `DHub` and its 19
    peripherals (`source` `both`), and `getParentLabel` reads `DHub` for
    each. Four are State devices: `DWheel` and `DStateDevice` (10 labels,
    `State-0`…), `DObjective` (6, `Objective-A`…), `DLightPath` (3).
  - `walk_adapter(core, "SequenceTester")` on the same core afterwards
    initialises `THub` and its 14; no name collides with DemoCamera's. A
    label already in use is refused by `loadDevice` (`The specified device
    label "X" is already in use`).
  - A `.cfg` written from that walk (`Device`, `Parent`, `Initialize`, the
    camera slot, `Label` lines) loads back with 21 devices, with LF or CRLF
    line ends, and with a UTF-8 byte-order mark.
  - MMCore drops empty fields, then refuses a line whose field count is
    wrong for its directive, and the whole load fails: `Label,DWheel,0,a,b`
    (a label with a comma), `Label,DWheel,0,` (an empty label),
    `ConfigGroup,Channel,Empty` and `ImageSynchro,DShutter` (this MMCore no
    longer reads it). `ConfigGroup,G,P,DWheel,Label,` (an empty value) loads.
  - `defineStateLabel` refuses a comma (`contains reserved characters`) and
    accepts a line break: a label a device holds can still break a line.
  - A `Label` or `Delay` line that names a device the file does not load
    fails the whole load (`No device with label "NoDev"`). A preset or a
    pixel-size line that names one loads, and the preset fails when applied
    (`No device with label "NoDev"`); so does a preset that sets a `Label` the
    device lacks (`Label not defined (10)`).
  - The field counts this MMCore reads, directive included (pymmcore-plus's
    own parser; the three marked measured): `ConfigGroup` 2, 5 or 6;
    `ConfigPixelSize` 5; `PixelSize_um` 3; `PixelSizeAffine` 8;
    `PixelSizeAngle_dxdz`, `PixelSizeAngle_dydz`, `PixelSizeOptimalZ_Um` 3
    (measured); `Label` 4; `Delay` 3; `FocusDirection` 3.
  - Roles on the DemoCamera build (`resolve`, heuristics only): `camera`
    DCam, `xy_stage` DXYStage, `focus` DStage, `autofocus` DAutoFocus,
    `objective_turret` DObjective, `shutter` DShutter, `light_path`
    DLightPath; no `autofocus_offset` and no `filter_turret`.

### 10.2 The walk gains `skip` — `mm_inventory.py`

`walk_adapter` and `probe_adapter` gain `skip: Collection[str] = ()`
(keyword-only, after `port`). A device whose name is in `skip` is never
loaded: its row is `not-probed`, reason `left out: --skip <name>`, it counts
among the untried devices of the `--port` note, and a skipped hub is not a
failed one (no `causes`); the next listed hub is tried (rule 2). `mm_section`,
`inventory` and `smc discover` do not take it (#15 does not touch them).

`check_device_name(name: str) -> str`, beside `check_port` and with its
rules: the stripped value; `ValueError` when it is empty, longer than 200
characters, or holds a control or invisible character.

### 10.3 The build child — `_mm_child.py`, `mm_inventory.py`, `models.py`

The build runs in the adapter child, for the reasons of M1 §10 (FM-01 to
FM-04): a device that crashes or hangs costs the build, never the parent,
and the parent says which device it was.

```python
class BuiltDevice(BaseModel):  # models.py
    label: str  # the device name: the walk loads each device under it
    library: str
    name: str
    type: str
    parent: str | None = None  # getParentLabel, None when it reads ""
    state_labels: list[str] = Field(default_factory=list)  # State: position i at i
```

```python
class BuildScan(BaseModel):  # models.py
    stand: ProbeResult
    camera: ProbeResult | None = None  # None: no --camera-adapter
    devices: list[BuiltDevice] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    complete: bool = True
```

`ChildResult` gains `build: BuildScan | None = None`.

`python -m smc.discovery._mm_child --result FILE --build ADAPTER
[--camera=ADAPTER] [--skip=NAME]… [--search-path=DIR]…`: every value travels
as one `--opt=value` argument (§9.1). With `--build`, `--adapters` and `--all`
are ignored, nothing is listed, and the child:

1. opens its core with `core.new_core(search_paths)` (below); a `CoreError`
   is written as `BuildScan(stand=ProbeResult(adapter=ADAPTER,
   error=str(exc)))` and the child exits 0;
2. walks the stand (`walk_adapter(core, ADAPTER, skip=…, progress=…,
   checkpoint=…)`), each checkpoint writing `BuildScan(stand=<snapshot>,
   complete=False)`;
3. with `--camera`, walks it on the same core, each checkpoint writing
   `BuildScan(stand=<the stand's result>, camera=<snapshot>,
   complete=False)`. Both walks always run: a stand that failed still has its
   camera diagnosed;
4. for every `ok` row of both walks, stand first, in row order: progress
   `reading <adapter>/<device>`, then `parent` and, for a State device,
   `state_labels`. A read that raises leaves the field empty and adds a note
   `<device>: its <parent|state labels> could not be read: <error>`;
5. writes the whole `BuildScan` (`complete=True`) **before** `close_core`,
   in a `finally` as `_probe` does (FM-02).

`_PROGRESS_PREFIXES` gains `reading `.

`core.new_core(adapter_search_paths: Sequence[str | Path] = ()) ->
CMMCorePlus` (`smc/hardware/core.py`): what `open_core` does before it loads a
file (checks, merge, `PATH`, the INFO line), taken out of it; `open_core`
calls it. The build child then searches adapters as the profile's stand
will be opened (FM-76), and the folders reach `PATH` as `open_core` puts
them (FM-81).

```python
# Two adapters; a probe's 180 s is not verified on a hub (§9.10).
BUILD_TIMEOUT_S = 300.0


@dataclass(frozen=True, slots=True)
class BuildRun:
    scan: BuildScan | None  # None: the child wrote nothing usable
    notes: list[str]
    stopped_at: str | None  # the device the child was on when it died or was stopped


def build_section(
    stand: str,
    *,
    camera: str | None = None,
    skip: Sequence[str] = (),
    search_paths: Sequence[Path] = (),
    timeout_s: float = BUILD_TIMEOUT_S,
) -> BuildRun: ...
```

`build_section` and `mm_section` share one private runner (temporary
folder as working directory, files not pipes, `CREATE_NO_WINDOW`, waited in
slices, the tree killed at the deadline, the result file read whatever the
exit code); `mm_section`'s behaviour and texts do not change. The build's
notes read `build: the build child …` where `mm_section`'s read
`adapters: the adapter child …`. `stopped_at` is the device of the last
`probing <adapter>/<device>` or `reading <adapter>/<device>` line, set only
when the child did not exit 0 or was stopped.

### 10.4 The plan and the text — `smc/discovery/config_build.py` (new)

Pure: no `pymmcore_plus`. It imports `smc.hardware.roles` (pure at module
level).

```python
NOT_CARRIED = "# not carried: "
FIELD_COUNTS: Mapping[str, frozenset[int]]  # §10.1's table
```

```python
@dataclass(frozen=True, slots=True)
class Block:
    kind: Literal["preset", "pixel", "labels", "device", "property", "other"]
    key: str  # "Channel/DAPI", "Res10x", "Nosepiece", "Delay ZDrive", …
    lines: tuple[str, ...]  # verbatim, in file order
    devices: tuple[str, ...]  # the labels it names, Core excluded
```

```python
@dataclass(frozen=True, slots=True)
class PreviousConfig:
    path: Path | None
    blocks: tuple[Block, ...]  # in order of first appearance
    # Property,Core,<Camera|XYStage|Focus|AutoFocus|Shutter>,<label>
    core_slots: Mapping[Role, str]


def parse_previous(text: str, path: Path | None = None) -> PreviousConfig: ...
def read_previous(path: Path) -> PreviousConfig | None: ...  # None: no file
```

```python
@dataclass
class ConfigPlan:
    adapter: str
    camera_adapter: str | None
    devices: list[BuiltDevice]  # what the file loads, in the order written
    roles: RoleMap
    carried: list[Block]
    not_carried: list[tuple[Block, str]]  # the block, why
    left_out: list[tuple[str, str]]  # "<adapter>/<device>", why
    refusals: list[str]  # not empty: nothing may be written
    notes: list[str]

    @property
    def writable(self) -> bool: ...  # not self.refusals


def plan_config(
    scan: BuildScan,
    *,
    previous: PreviousConfig | None = None,
    overrides: Mapping[Role, str] | None = None,
    exclusions: Mapping[Role, Sequence[str]] | None = None,
    skip: Sequence[str] = (),
) -> ConfigPlan: ...
def render_config(plan: ConfigPlan, *, header: Sequence[str]) -> str: ...
```

**Reading the previous file.** `read_previous` reads with
`encoding="utf-8", errors="surrogateescape"`, so that the bytes of a line it
carries are written back unchanged (a cp1252 `µ` stays one byte); an
`OSError` propagates. `parse_previous`:
- drops a byte-order mark at the start (FM-80), blank lines and comments,
  except a line that starts with `NOT_CARRIED`: its rest is read as a line
  of the file, so a block set aside by an earlier build comes back once it
  applies again (§10.5);
- splits on `,` and counts the non-empty fields, as MMCore does;
- drops what the build writes itself: `Device`, `Parent`, `Property,Core,…`.
  `Property,Core,<slot>,<label>` for the five slots goes to `core_slots`;
- groups the rest into blocks: `ConfigGroup,<g>,<p>,…` by preset (`g/p`;
  `ConfigGroup,<g>` alone is a block of its own, naming nothing);
  `ConfigPixelSize`, `PixelSize_um`, `PixelSizeAffine`, `PixelSizeAngle_*`
  and `PixelSizeOptimalZ_Um` by resolution; `Label` by device; `Delay` and
  `FocusDirection` one block per line (`device`); `Property,<device>,…` one
  block per line (`property`); any other directive, or a line whose field
  count is not in `FIELD_COUNTS`, one block per line (`other`);
- the devices a block names: the device field of each line, except `Core`;
  and for a `Core` setting of `Camera`, `Shutter`, `Focus`, `XYStage`,
  `AutoFocus`, `Galvo`, `SLM` or `ImageProcessor`, its non-empty value.

**The plan**, in this order:
1. **Refusals** (nothing is written; each one line):
   - the scan is not `complete`: `the build child stopped before the end`;
   - `stand.error`: `<adapter>: <error>`;
   - a `failed` Hub row in the stand's walk, even when another hub then
     initialised (the Ti2-E's simulator answers when the stand is off, §9.3):
     `hub <hub> of <adapter> did not initialise`. A hub left out with `--skip`
     is not a failure: that is how a build of the simulator is asked for;
   - no `ok` row in the stand's walk other than its hubs: `<adapter>:
     nothing below the hub initialised` (or `no device initialised`);
   - with a camera adapter: its `error`; a `failed` Hub row in its walk; no
     `ok` Camera row in its walk: `<camera adapter>: no camera initialised`.
2. **Devices written**: the stand's `ok` devices, then the camera adapter's
   `ok` Camera devices and the hub that is the parent of one of them, each
   in row order. Left out, with the reason written at the end of the file:
   every `failed` and `not-probed` row of both walks (`error` with the hint
   after a dash, or `reason`; for a device that needs a port, `needs a
   serial port, which smc config build does not set`, since the walk's
   reason names `--port`, which this command lacks); the camera adapter's other `ok` devices (`not
   a camera: --camera-adapter adds cameras only`); a device whose name holds
   `,`, a line break or a control character (`its name cannot be written in
   a .cfg line`), and any device whose parent is left out.
3. **Roles**: `resolve` on the written devices
   (`DeviceInfo(label, type, library, name)`), with `core_roles` from the
   previous file's `core_slots`, the profile's `overrides` and
   `exclusions`. A slot the previous file set therefore keeps its device
   when the device is still there and fits the role.
4. **Labels**, per written State device: the previous file's `Label` block
   for that device when it has one (a human may have named the positions,
   and the presets use those names), otherwise the device's
   `state_labels`. A label that is empty after `strip`, or holds `,` or a
   line break, is not written; a note names the device and the position.
5. **Carried, or set aside.** A block is carried when every device it names
   is written and, for a `preset` or `pixel` block, every `<device>,Label,<x>`
   setting names a label that device has in this file (point 4). Otherwise
   it is set aside, whole, with its reason: `names <device>, which this
   configuration does not load`; `sets <device> to the label <x>, which it
   does not have here`. `property` and `other` blocks are always set aside:
   `a device setting: the build writes no Property line for a device; put it
   in a preset (System/Startup)` and `not a line this Micro-Manager reads`.
   A `labels` block for a device that is not written is set aside too.
6. **Notes**: no Camera device written: `no camera: <adapter> provides none;
   pass --camera-adapter with the camera's adapter (smc discover lists the
   installed ones)`; a `skip` name that matches no row of either walk:
   `--skip <name> matched no device of <adapter>[ or <camera adapter>]`.

**The text** (`render_config`), LF line ends; a section with nothing in it is
left out with its heading; every text that goes into a comment has its runs
of whitespace (line breaks included) collapsed to one space (FM-37's lesson
for files):

```text
# <each header line>
# Only the devices that initialised are loaded; the others are listed at the end.

# Reset
Property,Core,Initialize,0

# Devices
Device,<label>,<library>,<name>

# Hub (parent) references
Parent,<label>,<hub>

# Initialize
Property,Core,Initialize,1

# Roles
Property,Core,<Camera|Shutter|Focus|XYStage|AutoFocus>,<label>
Property,Core,AutoShutter,1

# Labels
# <device>  (or: # <device> (from the previous configuration))
Label,<device>,<position>,<label>

# Carried over from the previous configuration
<the carried blocks, line by line, in the previous file's order>

# Set aside: kept as comments, carried again by a later build once they apply
# <block key>: <why>
# not carried: <line>

# Did not connect, or left out
#   <adapter>/<device>: <why>
```

The role lines follow `Camera, Shutter, Focus, XYStage, AutoFocus` and only
for an assigned role; `AutoShutter,1` only with a shutter. Labels before the
presets: a preset that selects a label then finds it defined.

### 10.5 What is never lost

Every line of the previous file that the build does not write itself comes
back, carried or set aside as a comment that a later build reads again. So
a build with the camera switched off sets the camera's presets aside, and
the next build with the camera on carries them again. On `--force` the
previous file is also copied to `<out>.bak` before it is replaced.

### 10.6 CLI — `smc config build`

A `config` sub-app (`smc config …`) with one command:

```text
smc config build --out FILE [--profile P] [--adapter NAME]
                 [--camera-adapter NAME] [--skip NAME]... [--dry-run]
                 [--force] [--timeout-s S]
```

- `--out` is required. `--adapter` defaults to the one Nikon stand in the
  profile's configuration (`nikon.stands_in_config`); none or several, or a
  profile with no configuration file (the demo), is a usage error: `✗ name
  the stand's adapter with --adapter (…)`, exit 2. `--camera-adapter` equal
  to `--adapter`: exit 2. A `--skip` value that `check_device_name`
  refuses: exit 2 with its message. `--timeout-s` 1 to 3600 and finite, as
  `discover`'s; default `BUILD_TIMEOUT_S`.
- **Before anything is initialised** (FM-30): `--out` is a folder, or its
  folder does not exist: exit 2. `--out` exists without `--force` or
  `--dry-run`: exit 2, `✗ <out> exists: pass --force to replace it (its
  presets, pixel sizes and labels are carried over, and it is saved as
  <out>.bak)`. `--out` exists and cannot be read: exit 1, nothing built.
- The banner, as `discover`'s: `Building from <adapter>[ and <camera>]:
  every device is initialised, so the stand must be powered and its vendor
  software closed.` The build runs under `console.status`.
- `plan_config` with the profile's `roles.assign` and `roles.exclude`;
  `render_config` with the header lines `Generated by smc config build on
  <YYYY-MM-DD HH:MM> from <adapter>[ and <camera>]` and `Profile <name>`.
- **Writable and not `--dry-run`**: the file is written **before** the report
  is printed (FM-30): an existing `--out` is first copied to `<out>.bak`, then
  the text replaces `--out` through `cli._replace` (FM-38), with
  `encoding="utf-8", errors="surrogateescape", newline="\n"`. An `OSError`
  prints the report all the same, then `✗ could not write <out>: …`, exit 1
  (FM-34).
- **The report**, every line as text, never markup (FM-36): the stand's
  probe table and the camera's, with their notes and causes (`report.py`'s
  `_probe` becomes the public `probe_renderables(probe)`, unchanged); the
  scan's and the plan's notes, `- <note>`; the child's notes; the role table
  (`RoleMap.describe()`) and its warnings; one line per carried block kind
  (`carried over from <out>: 3 presets, 2 pixel sizes, labels of 4 devices`)
  and one per block set aside, `! set aside: <key>: <why>`.
- Then exactly one of:
  - **refusals**: each as `✗ <refusal>`; when `stopped_at` is set and the
    scan is missing or not complete, `re-run with --skip <device> to leave
    it out`; exit 1, nothing written;
  - **`--dry-run`**: `--- would write <out> ---` and the text, line by line;
    exit 0;
  - **written**: `✓ wrote <out>` (`, previous saved as <out>.bak`) and
    `check it with: smc doctor --config <out>`; exit 0.

A device that failed, a camera missing or a block set aside is a finding:
the exit code stays 0 as long as the file is written.

### 10.7 Files and sequencing

`smc/discovery/mm_inventory.py`, `_mm_child.py`, `models.py`, `report.py`,
`config_build.py` (new), `smc/hardware/core.py` (`new_core`), `smc/cli.py`
(the `config` sub-app only), `docs/hardware/microscope-pc-setup.md`, and
`profiles/README.md` (how a profile's `.cfg` is rebuilt). After #48
(merged). #183 (`mm_inventory.py`, `_mm_child.py`) and #159 (`cli.py`) wait
for it.

### 10.8 Not verified

- Everything on a real stand: that the Ti2-E's 16 peripherals and a camera
  fit in 300 s; what the Ti-E's `TIDiaLamp` does in a build; that a Nikon
  device that initialises in a build also initialises from the written file
  in a fresh core.
- That an MMStudio-written `.cfg` (the format of the lab's other stands)
  holds no directive outside §10.1's table that matters; any other is set
  aside, never dropped.
- That a carried line with bytes that are not UTF-8 means to MMCore what it
  meant before: the bytes are kept, and MMCore reads bytes.

## 11. A stand that refuses `stop()` (#204, #206)

Extends M1 §13 (*Stop and halt*), §3.3.1 and §3.11. **Merging this section
decides it**: §11.2 puts the four questions of #204 with the options not
chosen and why. #206 (the Ctrl-C line) is built with it, since both change
the same stop paths. Segmenting a long move is #221.

### 11.1 What is known

Measured in Ti2-E session 1 (#208; `nikon-ti2.md`, *Session 1*), unless
labelled:
- `core.stop()` raises `RuntimeError('Error in device "XYStage":
  Unsupported device command (11)')` for `XYStage`, `ZDrive` and
  `PFSOffset`, still or moving. `PFS` answers `Cannot stop "PFS": not a
  stage`. Two refusals came back within 1 ms (B1: 6.737 s, 6.738 s).
- **A move that was sent runs to its target.** In stop drill 2 (`smc stage
  jog 20000 0 --force`, then Ctrl-C under way), the stage went the full
  20000 µm.
- **Switching PFS off stops a Z move that PFS drives**, at once (B5, one
  trial). With PFS off, an offset move moves no Z (§4.5: +10 and +50 units,
  Z unchanged).
- `deviceBusy("ZDrive")` and `deviceBusy("PFSOffset")` never read True
  (§4.9). Two things were not read, since the console of drill 2 was not
  kept: whether `deviceBusy("XYStage")` reads True during a move, and
  whether `setXYPosition` returns before the move ends.
- **What the layer does today**, reproduced on a `FakeCore` whose `stop()`
  raises that error (2026-10-28, `main` b09fe21):
  - A Ctrl-C during an XY move logs `xy_stage XY: stop after
    KeyboardInterrupt failed (HardwareError('xy_stage: stop('XY') failed
    (…Unsupported device command (11)); check the device: power, cable,
    another program holding it')); it still reads busy, so it counts as
    moving until it is idle`. The CLI then prints `interrupted — any move
    under way was stopped` (#206).
  - `close()` sends every stop again, and every one is refused. It logs
    `close: xy_stage XY still moving; the stages were stopped`, then one
    ERROR per refusal.
  - `Microscope.stop()` switches the autofocus off, sends every stop and
    raises `the emergency stop failed for XYStage 'XY' (…check the device:
    power, cable…); ZStage 'Z' (…)`.

  "Check the device: power, cable" is the wrong advice for this refusal:
  nothing is broken.

### 11.2 Decisions

**D1. How a stand declares it: a profile key, `[safety] stop_refused_by`.**
It lists the roles whose device refuses `stop()`. A hardware session fills
it, and a hardware contract checks it at each later session (§11.7).
- **The stop is still sent** to a declared role. It costs one call that
  returns at once (§11.1), and if the declaration is wrong (a firmware that
  accepts the stop), the stop works. The declaration changes what the layer
  promises and says. It never changes what the layer sends.
- An **undeclared** refusal is still a failure, as today, but its message
  says the move may still be running.
- Not chosen: *a probe at open* (`stop()` on each stage as the stand opens).
  Every `smc` command would then send a command to every stage. A transient
  fault would be read as a refusal. On a stand whose stop works, the probe
  would cancel a move started from the stand's own controls.
- Not chosen: *learning from the first refusal*. By then the first move,
  the one that matters, is already running, and nothing could be said
  before it.

**D2. Long moves: not segmented for now** (#221). Every message says that
a sent move runs to its target. Only today's guards bound such a move:
- the jog limits;
- the soft limits;
- a plugin's stop event, which m3's tools check between moves. A STOP
  during a scan therefore ends it after the current field move (m3 §15.1).

Why that is enough for now:
- On an inverted stand, the axis that crashes is Z. A Z move sent by smc
  is bounded by `max_z_jog_um` and `z_soft_limits_um`, and a Z move that
  PFS drives stops when PFS is switched off (§11.9).
- An XY move ends inside `xy_soft_limits_um`.

Not chosen now:
- *Cutting a long move into segments* that can be abandoned between sends,
  with a profile cap on one segment. This is safe only if the layer knows
  that a segment has ended before it sends the next. The Ti2-E's Z drive
  never reads busy (§4.9), and neither the XY busy state nor whether
  `setXYPosition` blocks has been read. Until session 2 makes the two reads
  that #221 lists, the end of a segment would be assumed.
- *Refusing a long move on such a stand.* A plate traverse is legitimate,
  and the jog guard already refuses a long relative move without `--force`.

**D3. Contracts.** For a declared role:
- `test_stop_on_an_idle_stage_is_harmless` is skipped, citing the
  declaration, as the turret's skips cite §8.8.
- A new contract checks the declaration on the stand (§11.7). A firmware
  that starts to accept the stop fails it, and the profile is then
  corrected.

**D4. Z under PFS: §3.3.1's table holds** (§11.9). Every row in which PFS
drives Z switches PFS off first, and that works on this stand. Only the
plain Z move loses its stop.

### 11.3 Profile — `profile.py`, `safety.py`

```toml
[safety]
# stop_refused_by = []  # roles whose device refuses stop(): "xy_stage", "focus", "autofocus_offset"; measured at a stand (m2 §11)
```

```python
class SafetySection(_Section):  # gains, after its M2 keys
    stop_refused_by: tuple[Role, ...] = ()
```

Validation, each a `ProfileError` naming `safety.stop_refused_by`: every
value is one of `xy_stage`, `focus` and `autofocus_offset`, the roles with a
stop (the message lists them), and no value appears twice. `Profile.demo()`
and `profiles/demo.toml` stay equal: the demo declares nothing, and the
commented line above goes into `demo.toml`. `profiles/README.md` documents
the key. `profiles/nikon-ti2.toml` sets `stop_refused_by = ["xy_stage",
"focus", "autofocus_offset"]`, measured in session 1 (#208).

`Safety(..., stop_refused_by: frozenset[str] = frozenset())` takes the
roles' values, since `safety.py` reads no profile model. The facade passes
`frozenset(role.value for role in safety.stop_refused_by)`.

### 11.4 Motion, the Executor and errors — `safety.py`, `errors.py`, `backends/mm.py`

```python
@dataclass(frozen=True, slots=True)
class Motion:  # gains, last
    # The profile declares that the device refuses stop() (m2 §11).
    stop_refused: bool = False
```

`mm._motion(core, label, name, *, stoppable: bool, stop_refused: bool =
False)` and `autofocus_offset_motion(core, label, *, stop_refused: bool =
False)`. Each sets the field from `safety.stop_refused_by`:
- `MMXYStage` from `"xy_stage"`;
- `MMZStage` from `"focus"`;
- the offset device's motion from `"autofocus_offset"`, in `MMAutofocus`
  and in the facade's `_stop_one` alike, so both build the same `Motion`.

The autofocus lock wait (its stop switches PFS off) and the `State`
devices are unchanged.

```python
class StopRefusedError(HardwareError):
    """A stop the device refused, as the profile declares ([safety] stop_refused_by).

    Not a fault: nothing is broken, and a move under way runs to its target.
    """

    def __init__(self, refused: tuple[str, ...], message: str = "") -> None:
        self.refused = refused  # motion names: ("xy_stage XYStage",)
        ...  # default message below; __reduce__ as MotionStoppedError's


class StopOutcome(str, Enum):  # extended by §11.11 E4, E6
    accepted = "accepted"  # the device took the stop
    refused = "refused"  # raised on a declared role: the move runs to its target
    failed = "failed"  # raised on an undeclared role: a fault
    impossible = "impossible"  # no stop from smc (a State device)
    interrupted = "interrupted"  # a Ctrl-C cut the stop: it may not have got through
    still_moving = "still_moving"  # tracked as moving when the stand was released


@dataclass(frozen=True, slots=True)
class StopNote:  # `declared` removed by §11.11 E4
    motion: str  # "xy_stage XYStage"
    outcome: StopOutcome


def note_stop(exc: BaseException, note: StopNote) -> None: ...
def stop_notes(exc: BaseException) -> tuple[StopNote, ...]: ...
```

- **Default message** of `StopRefusedError`: `` `xy_stage XYStage` refused
  the stop, as the profile declares ([safety] stop_refused_by): a move under
  way runs to its target; stop it with the stand's own controls if it must
  not finish ``. Several names are joined with `, ` and `and`.
- **Notes ride on the exception** that ended the wait (a
  `KeyboardInterrupt`, usually), in an attribute `_smc_stop_notes`. It is a
  tuple, and `note_stop` appends to it. A note holds only a string and an enum,
  so it can hold no traceback (FM-98). `stop_notes` collects the
  notes of `exc` and of its `__context__` chain, oldest first, and stops at
  an exception it has already seen, and at the first one that is not a
  `KeyboardInterrupt` (§11.11 E5). Measured on Python 3.13 (2026-10-28): a
  `KeyboardInterrupt` takes the attribute, and the attribute survives a
  re-raise. CI's 3.10 job runs the same tests.
- **`Executor.stop(motion)`.** When `motion.stop_refused` is set and the
  device's own stop raised, it raises `StopRefusedError((motion.name,))`,
  chained from the device's error. A failing linked call is unchanged, and
  two refusals are merged (§11.11 E1). The WARNING line keeps its form,
  `<name>: stop failed (<error>)`, and gains the note `; the profile declares
  this refusal ([safety] stop_refused_by), so a move under way runs to its
  target`.
- **The give-up** (`_give_up`, internal) returns its text and its
  `StopOutcome`. For a declared refusal, the text is `the stand refuses its
  stop, as the profile declares ([safety] stop_refused_by), so it runs to its
  target`. An undeclared refusal keeps `the stop failed (<error>), so it may
  still be moving`. Its outcome is `refused` on a declared role and `failed`
  on any other (§11.11 E4); its WARNING line carries the declaration's note,
  as `Executor.stop`'s does (§11.11 E2). `DeviceTimeoutError` and the
  timeout's `MotionStoppedError` carry that text, as today. The public
  `give_up()` still returns the text alone.
- **`_move`'s `except BaseException`** notes the give-up's outcome on the
  exception, once `_give_up` has returned (`StopNote(motion.name,
  outcome)`). If `_give_up` itself raises (a second Ctrl-C inside the stop
  call), that new exception gets `StopNote(motion.name, interrupted)` before
  it propagates, and no second stop is sent. The same holds for the give-up
  at the move's timeout (§11.11 E3). Its `__context__` is the first
  exception, which `stop_notes` also reads.
- **A stop requested during a move on a declared device** (the generation
  moved, then the device read idle) still raises `MotionStoppedError`: the
  caller asked for a stop, so the plan's next step must not run (M1 §13).
  Its detail becomes `a stop was requested during the move, but the profile
  declares that it refuses its stop ([safety] stop_refused_by), so it may
  have run to its target`.
- Nothing changes in what is sent, or in the order of the sends (FM-17,
  FM-62). Every stop is still sent before any line.

### 11.5 Facade — `microscope.py`

- `_stop_stages()` returns a private `_Stops(done: list[str], refused:
  list[str], failed: list[str])`, collected past every failure as today:
  - `done` gets `autofocus <label> off` for a `disengage()` that returned,
    and the motion name of each stop that was accepted;
  - `refused` gets the names of each `StopRefusedError.refused`, each once
    (§11.11 E1);
  - `failed` gets today's entries, unchanged.
- **`stop()`**:
  - Nothing refused or failed: it returns, as today.
  - Refusals only: it raises `StopRefusedError(refused, message)`:
    `the emergency stop halted the microscope[ and switched autofocus PFS
    off], but xy_stage XYStage, z ZDrive and autofocus_offset PFSOffset
    refused the stop, as the profile declares ([safety] stop_refused_by): a
    move under way runs to its target. The microscope stays halted: stop
    the stage with the stand's own controls if it must not finish its move,
    then call `resume()``.
  - Any failure: today's `HardwareError` text, `the emergency stop failed
    for <failed>`. When there were refusals too, it adds `; <names> refused
    the stop, as the profile declares ([safety] stop_refused_by), so a move
    under way runs to its target`, then today's ending.
- **`close()`** (`_stop_if_moving`) logs after the stops, as today (FM-62).
  It writes one WARNING, `close: <moving> still moving; <outcome>`. The
  outcome is made of these clauses, joined with `; `:
  - `stopped: <done>`, when `done` is not empty;
  - `refused, as the profile declares ([safety] stop_refused_by), so a move
    under way runs to its target: <refused>`;
  - `not stopped: <n> failed`.

  Then it writes one ERROR per failure, as today. It never writes "the
  stages were stopped" (#206). Unloading does not wait for a refused move
  to end: Ctrl-C must let smc exit, and the stand finishes the move on its
  own. What unloading does to a stage that is still moving is unverified
  (§11.10).
- **`__exit__`** notes what is still tracked as moving when a Ctrl-C leaves
  the `with` block (§11.11 E6): `moving()` is read before `close()`, and
  `StopNote(name, still_moving)` is added to the `KeyboardInterrupt`, for
  each name its notes do not carry.

### 11.6 CLI — `cli.py`

- **The Ctrl-C line** (`_report`, #206) is built from
  `stop_notes(exc)`. Its tail stays `read the position (smc stage get, smc z
  get) before the next move`, and the exit code stays 130.
  - No note: `interrupted — no move was under way; read the position …`.
    The layer sends a move only inside `_move`'s `try`, and every Ctrl-C
    inside it leaves a note; a move the registry still tracks, whoever sent
    it, is noted when the stand is released (§11.11 E6).
  - Notes: `interrupted — <clauses>; …`, one clause per note, joined with
    `; `:
    - accepted: `<name> was stopped`;
    - refused: `<name> refuses a stop on this stand ([safety]
      stop_refused_by), so it runs to its target`;
    - failed: `the stop of <name> failed, so it may still be moving`
      (§11.11 E4);
    - impossible: `<name> cannot be stopped from smc, so it may still be
      moving`;
    - interrupted: `the stop of <name> was cut by a Ctrl-C, so it may still
      be moving` (§11.11 E3);
    - still moving: `<name> was still tracked as moving when the stand was
      released; close()'s log line says what its stop did` (§11.11 E6).
  - When any note is not *accepted*, the tail starts with `let it finish
    or stop it with the stand's own controls, then read the position …`.
- **Before a move**, one line `! the profile declares that this stand
  refuses stop() on <roles> ([safety] stop_refused_by): once sent, a move
  runs to its target, and Ctrl-C ends smc, not the move`. It is printed
  after the stand opens and before anything is sent, never in dry-run:
  - by `stage move` and `stage jog`, when `xy_stage` is declared;
  - by `z move` and `z jog`, when `focus` is declared;
  - by `smc run`, when any role is declared, listing every declared role.

  `smc doctor` prints the same line after the role table, with every
  declared role. One helper builds the line.

### 11.7 Contracts — `tests/contracts/test_xy_stage.py`, `test_z_stage.py`

- `test_stop_on_an_idle_stage_is_harmless` skips when its role is declared:
  `the profile declares that this stand refuses a stop of xy_stage ([safety]
  stop_refused_by, m2 §11)`, and the same with `focus`.
- `test_a_declared_stop_refusal_still_holds`, in both files, skips when its
  role is not declared. Otherwise the stage's `stop()` must raise
  `StopRefusedError`, and the next move must still run and land (the same
  move and tolerance as the idle-stop contract). The envelope keeps
  `stop_refused_by`, since it copies every `[safety]` key it does not
  replace (`model_dump`).

### 11.8 FakeCore — `smc/testing/fakes.py`

`stop_refused: set[str]` is a new knob. It lists the labels whose `stop()`
is logged, then raises `RuntimeError('Error in device "<label>":
Unsupported device command (11)')`, the Ti2-E's text. `busy_devices` is
left alone, so a busy device stays busy: the move goes on. Through
`as_hardware_errors`, the error reaches the layer as `xy_stage: stop('XY')
failed (…)`, measured with a prototype (2026-10-28).

### 11.9 §3.3.1 on the Ti2-E

Every stop is still sent in §3.3.1's order. With `ZDrive` and `PFSOffset`
refusing theirs:

| Motion | What still stops it |
|---|---|
| Z move (`ZDrive`) | nothing from smc: it runs to its target, inside `max_z_jog_um` and `z_soft_limits_um` |
| Relock after an interlocked Z move | PFS off (`disengage()`, `ZStage.stop()`'s link, the emergency stop) |
| Locked `focus_by_um` | PFS off. The offset lens finishes its move, which with PFS off moves no Z (§4.5) |
| A direct `engage()` | PFS off |
| A direct offset move, PFS on | PFS off, then as above |
| The same move, PFS off | nothing, and nothing needs to: it moves no Z (§4.5, measured) |
| PFS holding focus | PFS off (the emergency stop) |

`ZStage.stop()` on a link still switches PFS off first. Then Z's own
refusal raises `StopRefusedError`, and the compound reports its
`MotionStoppedError` naming Z, "so autofocus PFS was left off", which is
true. Where `engage()` takes Z, and what bounds it, is §12 (#205): it is
about where the move goes, not how it is stopped.

### 11.10 Not verified

- Whether `deviceBusy("XYStage")` reads True for the whole of a move, and
  whether `setXYPosition` blocks (#221, session 2). Until then, a Ctrl-C
  during an XY move on the Ti2-E may arrive only once the move has ended.
- What `unloadAllDevices()` does to a Ti2-E stage that is still moving.
  Drill 2 suggests no harm, but its console was not kept.
- B5's "PFS off stops Z at once" rests on one trial.
- Whether a Z move still running on the Ti2-E is noted when the stand is
  released (§11.11 E6): the drive never reads busy (§4.9), so nothing tracks
  it, and the Ctrl-C line then says "no move was under way". The `!` line
  before the move (§11.6) is what warns.
- `test_a_declared_stop_refusal_still_holds` on the Ti2-E: the owner runs
  it at session 2.
- The wording of m3 §15's STOP on such a stand. The page shows `stop()`'s
  error text (§15.1), and its own label is m3's, after #214.

### 11.11 What the first build left open (#224)

The review of #223 and its adversarial review left seven cases (#224). Each
was reproduced on a `FakeCore` (2026-10-30, `main` cf79b98), and none changes
what is sent. **Merging this section decides them**: E1-E3 are the fixes the
issue names, E4-E7 answer the questions it left to the design, E8 is the
wording. The edits are in place in §11.4-§11.6.

**E1. Two refusals, one error (#224 item 1).** On a Ti2-E fake, `Z.stop()`
during an offset move with PFS on raised `StopRefusedError(("autofocus_offset
PFSOffset",))`: the link's refusal won over Z's own, so `.refused` missed
`z ZDrive`, and `close()` from another thread listed the offset twice and Z
not at all. In `Executor._stop_linked`, when the linked call and the device's
own stop both raise `StopRefusedError`, one is raised whose `refused` is the
link's names, then the device's, each once, with the default message and the
`__cause__` of the device's own refusal (the error of the device the caller
asked to stop). Every other pair is as today:

| linked call | device's own stop | `stop()` raises |
|---|---|---|
| refusal | refusal | one refusal naming both (new) |
| refusal | accepted | the link's refusal |
| accepted | refusal | the device's refusal |
| a fault | a refusal | the fault (FM-127) |
| a refusal | a fault | the device's fault |
| a fault | a fault | the device's fault |

`_Stops.refused` (§11.5) lists a name once, in the order of first
appearance: `close()` sends the offset's stop, then Z's, whose link stops the
offset again.

**E2. The give-up's line says what the profile declares (item 2).** On a
declared role, the WARNING of a failed stop in a give-up (at the timeout, or
after an exception) is `<name>: stop after <why> failed (<error>)`, then the
note `; the profile declares this refusal ([safety] stop_refused_by), so a
move under way runs to its target`, then the busy note (`; it still reads
busy, so it counts as moving until it is idle`). The CLI line was right; the
log contradicted it.

**E3. A give-up is not given up twice (item 4).** Measured: a Ctrl-C inside
the *timeout's* give-up made `_move`'s outer `except` give up again, so two
`stop()` were sent and the note said `accepted`, the second stop's. Now
`gave_up` goes up in the give-up's `registered` callback, right after the
motion is registered and before the stop is sent, and a `BaseException` that
escapes a give-up from that point, on either path, is noted `interrupted` and
propagates; no second stop is sent. Not before the give-up: set there, the
flag left a window (a Ctrl-C before the registration) in which no stop was sent
and nothing was tracked, so `close()` could not retry (FM-168, the adversarial
review of #224). A cut before the registration is not noted: the `except` gives
up again, as it always did, and notes what that did. The note is part of the
fix: setting the flag alone leaves a cut stop with no note, and the Ctrl-C
line would say "no move was under way". On the timeout path the cut Ctrl-C is
the first, so the clause no longer says "second" (§11.6).
- Not chosen: *keeping the second stop as a retry*. It runs under the same
  hazard as the first, and `close()`, which every command runs on leaving
  its `with`, sends every stop again to what is still tracked as moving. The
  give-up registers the motion before it sends the stop, so a cut stop
  leaves it tracked.

**E4. `failed` is an outcome, and `declared` goes (item 5).** The `Executor`
never parses a device's text (D1), so a stop that raised on a role the
profile does not declare is a failure: a lost connection reads the same as a
refusal. The Ctrl-C line said "refused the stop" for it. `StopOutcome.failed`
is that case, and `refused` is the declared one, the same two words
`_Stops.refused` and `_Stops.failed` already use. `_give_up` returns
`refused` iff `motion.stop_refused`, else `failed`. `StopNote.declared` then
says nothing the outcome does not, and has no value for `still_moving`: it
is removed, and a note is `(motion, outcome)`.

**E5. `stop_notes` stops at the first exception that is not a
`KeyboardInterrupt` (item 3).** A note is the interrupt's own, or belongs to
the interrupt it interrupted (a second Ctrl-C inside a give-up's stop). A
Ctrl-C raised inside the handler of an earlier, noted `HardwareError` (a
plugin that catches a failed move) reported that old stop as its own. The
walk still follows `__context__`, since `from None` leaves it, but only
through `KeyboardInterrupt`s; the first other exception ends it and is not
read.
- Not chosen: *keeping the walk and documenting it*. The line exists to say
  what this interrupt did.

**E6. What the registry still tracks is noted when the stand is released
(item 6).** A Ctrl-C inside `XYStage.wait()` on a stage that an earlier
refused give-up left tracked carried no note, so the line said "no move was
under way" while it travelled (measured: notes `()`, `moving()`
`('xy_stage XY',)`). `Microscope.__exit__`, when the exception that leaves
the block is a `KeyboardInterrupt`, reads `self._executor.moving()` before
`close()`, and in a `finally` after it adds `StopNote(name,
StopOutcome.still_moving)` for each name that `stop_notes(exc)` does not
carry. Nothing here takes a lock beyond what `close()` takes, or does I/O,
and a failing `close()` loses no note. It makes the line's "no note, no
move" true for what the registry knows.
- Not chosen: *`wait()` notes its own interrupt*: it covers one place, and a
  Ctrl-C between two calls on a stand with a leftover would still say "no
  move". *Reading `moving()` in `_report`*: the stand is released by then
  (`_Reported` wraps the `with`).
- What it cannot say: the Ti2-E's Z drive never reads busy (§4.9), so a Z
  move that is still running is not tracked (§11.10).

**E7. On a declared role, any raise is the declared refusal: kept (item 7).**
The declaration is the profile's statement that this stop does not work.
Whatever `stop()` raised, the refusal or a lost connection, the move under
way runs to its target, which is what every text says. The device's own error
stays in the WARNING line and in `__cause__`, and is not parsed (D1); a lost
connection shows on the next read, with that read's diagnosis.
- Not chosen: *the error's text in the give-up's message*. It carries the
  translator's "check the device: power, cable" (§11.1), the wrong advice for
  a refusal.

**E8. The words.** Ctrl-C *sends a stop* to any move under way, and the line
it prints says what that stop did; a stand may refuse it. Three texts said
"stops": M1 §9's exit-code table (corrected with this section),
`docs/plugins/README.md` (*Stopping*) and `smc run --help`.

**Tests** (`tests/unit/test_refused_stop.py`, `test_errors_and_roles.py`,
`test_cli.py`): one per decision, each on a `FakeCore`; the plan names them.

## 12. Engaging an autofocus that drives Z to its offset's plane (#205)

Extends §3.2 (`engage`), §3.4 (the relock), §3.6 and §3.9. **Merging this
section decides it**: §12.2 answers #205's three questions, with the
options not chosen and why. FM-118 is the checklist entry.

### 12.1 What is known

Measured in Ti2-E session 1 (#208; `nikon-ti2.md`, *Session 1*), unless
labelled:
- **Engaging PFS drives Z to the plane its stored offset names**, not to
  the operator's focus. Run 1 (10×, a dish focused by eye at 4269.44 µm, PFS
  off, offset 9370 left from an earlier use): `enableContinuousFocus(True)`,
  then `locked` read True at the first read (0.1 s), Z unchanged for 0.5 s,
  then Z at 4504.76 µm at the next 100 ms read: +235 µm, 155 µm above the
  4350 µm soft limit. The stand's PFS button did the same (about 200 µm).
- **Once the offset was set on the sample** (PFS on, the knob turned until
  the cells were sharp: 7927), engaging moved nothing, three times, and
  `enableContinuousFocus(True)` sent again while locked moved nothing.
- `deviceBusy("ZDrive")` never read True, in 928 reads, the jump included.
  `locked` is not "Z has arrived", and busy is not "PFS is moving Z".
- With no sample, PFS on for 10 s never locked and Z stayed within
  0.02 µm (§4.2): a search with no reference moves no Z.
- Under lock, an offset change moves Z: +50 units gave +6.68 µm, and Z
  settled in about 0.3 s, then crept by about 0.3 µm over 5 s (§4.4). With
  PFS off, an offset change moves no Z (§4.5).
- The layer polls a wait every 10 ms (`POLL_INTERVAL_S`). How long one Z
  read takes on the Ti2-E was not measured (§12.9).
- `PFS Status` read `0000001100001001` while Z was still to move and
  `0000001100001010` once it had, and flips between the two while the
  offset moves under lock. That a bit means "settled" is a hypothesis.
- Loading the configuration leaves PFS and its offset as the last user of
  the stand left them (7803, then 9370, then 7927).
- **What the layer does today** (`main` 6fc1b14): `engage()` waits for
  `isContinuousFocusLocked()` and reads nothing else. Nothing bounds Z while
  PFS drives it, and nothing checks where the offset came from.
- **The demo** (`DAutoFocus`) moves nothing when it engages (§1).

### 12.2 Decisions

**D1. When the layer may engage: only at an offset it has seen locked, on
a stand that declares the behaviour.** A profile key,
`[autofocus] engage_drives_z_to_offset = true`, declares that engaging
drives Z to the plane the stored offset names (measured on the Ti2-E). On
such a stand, `engage()` and the relock send nothing unless the offset now
set is one at which this autofocus has been **read locked, with the same
objective in place, since this `Microscope` was opened**: a *seen offset*.
Otherwise `engage()` is refused (`SafetyRefusedError`, not forceable), and
the relock raises `AutofocusLostError` (§12.5).
- A lock is seen by **any read the layer makes** that finds it locked:
  `is_locked()`, the lock wait's polls, `suspend()`, the implied-Z check,
  `focus_by_um`. So the operator's lock, taken at the stand before
  anything runs, becomes a seen offset at the first `is_locked()`, which
  m3's scan makes before anything moves (m3 §8). An offset the layer moves
  while PFS is locked becomes seen at the next locked read, since PFS
  follows it (§4.4).
- **Engaging a seen offset is the case m3 needs**: the scan's per-field
  `engage()` after a lost lock, at the operator's offset, on the same
  sample. The plane it drives to is that offset's on this sample, so the
  jump is bounded by the sample's own shape, which D2 bounds.
- **Not forceable.** A forced engage of an unknown offset at 40× is the
  crash itself, and D2 cannot stop a jump that ends between two reads. The
  route for a new sample or objective is a first lock from below (§12.10,
  D4-D7): the offset set to its floor with PFS off, an engage that drives Z
  away from the sample, then the knob until the sample is sharp. *Amended
  by #331*: this bullet first named the stand's button, which jumped
  +206.6 µm in session 2 at an unseen offset.
- **PFS on, searching, at an unseen offset** is switched off before the
  refusal (a safe call, `disengage()`): left searching, it would drive Z to
  that offset's plane the moment it found the reference. The refusal says
  so.
- **Undeclared stands** (the demo, the default) engage as today; D2 and D3
  apply to every stand.
- Not chosen: *`force=True` on `engage()`*, for the reasons above; it would
  also change the `Autofocus` Protocol for a call no tool should make.
- Not chosen: *offsets kept beyond the process* (in the profile or a run
  document). The layer cannot tell that the sample is the same: a kept
  offset is #205's stale offset by design. m3's `pfs_offset` per position
  stays a record for NIS-Elements, never something smc engages from.
- Not chosen: *predicting Z for an unseen offset from `offset_per_um`*. The
  factor varies (5.9 to 10.8 units/µm on one objective, §4.4) and converts
  a *change* between two locked planes; it says nothing about where an
  offset never locked on this sample lies.
- Not chosen: *never engaging from smc on the Ti2-E*. The scan then loses
  every field after a lost lock, which is what m3 §8 relies on PFS to avoid.

**D2. A Z guard around every engage, on every stand: it limits the damage,
it cannot prevent it.** `engage()`, the relock and `focus_by_um`'s relock
read Z before switching on, then at every poll of the lock wait and of D3's
settle window:
- **Before anything is sent**, Z outside `[safety] z_soft_limits_um`
  refuses the engage (`SafetyRefusedError`, not forceable): the guard
  could not tell a jump from where Z already is.
- **While it waits**, Z outside the soft limits, or outside the *band*,
  switches the autofocus off at once (the lock wait's own stop), forgets
  the offset (D1) and raises `AutofocusExcursionError` (§12.5). The band is
  `[min(z_start, z_held) − E, max(z_start, z_held) + E]`, with `E` the new
  `[safety] max_engage_excursion_um`, `z_start` the Z read before the
  switch-on, and `z_held` the Z at which PFS held its lock when `suspend()`
  switched it off for the move (the relock only; else `z_start`). A relock
  brings Z back to where PFS held it (§3.5: "a Z move would be undone by the
  relock"), so the way back is inside the band. Without `E`, only the soft
  limits bound it.
- **It is a limit, not a prevention**, and every message says so: Z is read
  every `POLL_INTERVAL_S` plus the read's own time, and run 1's jump took
  less than 100 ms. What the guard buys: PFS is not left holding Z past a
  limit, a jump is named instead of read as a lock, and on a declared stand
  the forgotten offset refuses every later engage until the operator locks
  again.
- **An excursion is an error, not a refusal** (FM-64): something moved.
  `AutofocusExcursionError` is a `HardwareError`. m3's scan records it as a
  failed field (m3 §8); its next engage is refused before anything is sent,
  so `max_failed_in_row` ends the scan with no second jump.
- **A Z read that fails** while the guard needs it switches the autofocus
  off and raises the read's `HardwareError` (unknown counts as the state
  that adds the safeguard, FM-19).
- **No focus drive to read** (no `ZStage` at all): no guard, and a WARNING
  per engage says so. An overridden `ZStage` is read like the built one:
  the guard only reads, unlike §3.3.1's links, which need `MMZStage`'s
  stop.
- The guard reads Z only when it has something to check (a soft limit,
  `E`, or D3's window); otherwise the lock wait reads what it reads today.
- Not chosen: *halting the microscope on an excursion*. The forgotten
  offset already refuses the next engage, and a halt would also refuse the
  reads that diagnose what happened.
- Not chosen: *a watchdog over PFS while it holds focus between actions*.
  Locked PFS following the sample is not a motion (§3.3); watching it is a
  thread of its own, for M5 if the sessions show a need.

**D3. What "locked" means to the layer.** `is_locked()` stays the device's
own flag, and its docstring says what it is not: on the Ti2-E it read True
before Z had moved. **`engage()` returning `True` means locked and
settled**: with `[autofocus] settle_s` > 0, after the first locked read
the wait goes on until every Z read of the last `settle_s` lies within
`[autofocus] settle_tolerance_um` of each other, and it still reads locked
(a lock lost meanwhile starts the window again at the next locked read).
`timeout_s` bounds the whole engage, the lock and the settling. A lock that
has not settled by then is given up like no lock: switched off, `False`, a
WARNING that names the spread. `settle_s = 0` (the default) keeps today's
behaviour: the first locked read ends the wait.
- The Ti2-E's profile sets `settle_s = 1.0` (chosen: Z moved 0.5 s after
  the lock, once) and `settle_tolerance_um = 0.5` (chosen: the creep is
  about 0.06 µm/s).
- Not chosen: *the `PFS Status` bit*. It is a hypothesis from one run;
  session 2 reads it (§12.9). If it holds, a later change adds a key, as
  `in_range_property` was added.
- Not chosen: *a fixed wait after the lock*. It ends on time, not on Z, and
  one observation does not bound the delay.

### 12.3 Profile — `profile.py`

```toml
[safety]
# max_engage_excursion_um = 50.0  # how far Z may move while the autofocus engages, beyond where it starts or held; omitted: only z_soft_limits_um bound it (m2 §12)

[autofocus]
# engage_drives_z_to_offset = false  # engaging drives Z to the plane the stored offset names; then the layer engages only at an offset it has seen locked (m2 §12)
# settle_s = 0.0                     # after the first locked read, engage() waits until Z has been still this long; 0: the lock is taken as read
# settle_tolerance_um = 0.5          # how far apart the Z reads of that window may lie
```

```python
class SafetySection(_Section):  # gains, after stop_refused_by
    max_engage_excursion_um: float | None = None


class AutofocusSection(_Section):  # gains, last
    engage_drives_z_to_offset: bool = False
    settle_s: float = 0.0
    settle_tolerance_um: float = 0.5
```

Validation, each a `ProfileError` naming the key:
- `max_engage_excursion_um`: positive and finite;
- `settle_s`: finite, in [0, 60], and below `lock_timeout_s` (below the
  default 5 s when that is unset; a model validator), since the lock
  timeout bounds the settling too;
- `settle_tolerance_um`: positive and finite;
- a TOML boolean is refused for the three numeric keys (#106).

`profile.py` mirrors `DEFAULT_LOCK_TIMEOUT_S` as it mirrors
`MAX_LOCK_TIMEOUT_S` (a pure module imports no backend), and a test pins
that the two agree.

`Profile.demo()` and `profiles/demo.toml` stay equal: the demo sets none of
them, and the commented lines go into `demo.toml`. `profiles/README.md`
documents them. `profiles/nikon-ti2.toml` sets, each with its source:
`engage_drives_z_to_offset = true` (measured, #208),
`settle_s = 1.0` and `settle_tolerance_um = 0.5` (chosen, D3),
`max_engage_excursion_um = 50.0` (chosen, for the owner to confirm at
session 2), and drops `# PFS-driven Z is not bounded by these (#205)` and
the `[quirks]` line on engaging, which the keys now carry.

### 12.4 Backend — `backends/mm.py`, `safety.py`

```python
class Safety:  # gains
    def __init__(self, *, ..., max_engage_excursion_um: float | None = None): ...
    max_engage_excursion_um: float | None


class MMAutofocus:  # gains three keywords, last
    def __init__(
        self,
        ...,
        engage_drives_z_to_offset: bool = False,
        settle_s: float = 0.0,
        settle_tolerance_um: float = 0.5,
    ) -> None: ...
```

- **Seen offsets.** `MMAutofocus` keeps `_seen: dict[str | None,
  frozenset[float]]`, keyed by the objective's label (`None` without a
  turret), replaced whole at each change (one store: no lock, FM-70). Only
  a declared stand records. `_read_locked()` is the one place a lock is
  read; when it reads `True`, it reads the objective and the offset and adds
  the offset. A failure of those two reads records nothing (a DEBUG line):
  the absence of a seen offset is the safe side. Offsets are compared as
  read back, exactly: the layer compares one readback of a device with
  another.
- **The engage check** runs inside the action, under the lock, before the
  switch-on, and in dry-run too (as the implied-Z check, FM-90). In order:
  1. Z read through the focus drive, then the soft limits (D2), when there
     is a focus drive;
  2. on a declared stand: `_read_locked()` (which records a lock), then the
     objective and the offset; a failed read propagates as its
     `HardwareError` with nothing sent (it decides whether Z moves, FM-19);
     an unseen offset refuses, after switching off an autofocus that reads
     engaged.
- **The guard** is an internal `_EngageGuard`, built per engage with the
  band and the settle window, and polled through the lock wait's
  `Motion.is_busy` (in the action's thread, under the lock: a Z read through
  `ZStage.position_um()` there was measured on the Ti2-E fake, 2026-10-28).
  It reads engaged, locked and Z, and returns busy while searching or
  settling. When Z leaves the band or the limits, or a read fails, it keeps
  the outcome as text (never an exception, FM-98) and reads idle, so the
  wait ends at once; the readback (`after_wait`) then gives up on the
  motion (`Executor.give_up`, which switches it off), forgets the offset
  and raises. A switch-off that fails there keeps the motion registered and
  says that PFS may still move Z, as `_LostLockError` does. A deadline
  reached while locked but not settled gives the D3 WARNING, not "no lock".
- **`suspend()`** reads `locked` (on a declared stand, so that the
  operator's lock is seen before the move switches it off) and, when it
  reads locked and the guard has a bound to keep, Z, kept as `z_held` for
  the relock that follows in the same compound; each `suspend()` overwrites
  it.
- **An effective lock timeout not above `settle_s`** (the argument, the
  profile's or the default, read at call time): `ValueError` before
  anything is sent, as for a timeout out of bounds.
- Unchanged: the `Autofocus` Protocol's signatures, `ZInterlock`, §3.3.1's
  links and stops, `disengage()`.

### 12.5 Errors and messages — `errors.py`

```python
class AutofocusExcursionError(HardwareError):  # not a SafetyRefusedError (FM-64)
    z_start_um: float  # Z read before the switch-on
    z_um: float  # the read that tripped the guard
    # __reduce__ as AutofocusLostError's
```

- **Excursion**: `` autofocus PFS: Z went from 4269.44 µm to 4504.76 µm
  while it engaged, <outside the soft limits [10, 4350] µm | more than
  [safety] max_engage_excursion_um (50 µm) beyond where it started>; it was
  switched off. The guard reads Z every 10 ms: it stops PFS once a jump is
  read, it does not prevent one. Check the sample and the objective; on
  this stand, lock PFS on the sample at the stand before engaging again ``
  (the last clause on a declared stand only). The WARNING `autofocus PFS:
  switched off: Z left its bound while it engaged (<z_um> µm)` precedes it.
- **Unseen offset** (`SafetyRefusedError`, `how_to_force=""`): `` autofocus
  PFS: not engaged: on this stand engaging drives Z to the plane the offset
  names ([autofocus] engage_drives_z_to_offset, m2 §12), and offset 9370
  PFSOffset units has not been seen locked with objective '40x' since the
  stand was opened (seen: 7927, 7977 | seen: none)[; it was on and
  searching, so it was switched off]. <the advice of §12.10.5> ``. The
  advice never names the stand's button (#331).
- **Z outside the soft limits before the switch-on** (`SafetyRefusedError`,
  `how_to_force=""`): `autofocus PFS: not engaged: Z is at 4400.00 µm,
  outside the soft limits [10, 4350] µm`.
- **The relock refused** (`AutofocusLostError`, `in_range` read as today):
  `z: moved to <z> µm, but autofocus PFS was not engaged again: <the
  unseen-offset or Z reason>; it is off`, then today's ending.
- **Not settled** (WARNING, then `False`): `autofocus PFS: locked, but Z
  did not settle within <timeout> s (it moved <spread> µm in the last
  <settle_s> s); switched off`.

### 12.6 Facade — `microscope.py`

`_build(Autofocus)` passes the three `[autofocus]` keys; `Safety(...)` gets
`max_engage_excursion_um`. Nothing else changes: the seen offsets live and
die with the `MMAutofocus`, so with the `Microscope`.

### 12.7 FakeCore — `smc/testing/fakes.py`

Two knobs reproduce run 1:
- `autofocus_target_um: Callable[[], float] | None = None`: where the
  autofocus drives the focus device once locked (a test's closure reads the
  offset device). `None`: it moves nothing, as today.
- `autofocus_drive_after_reads: int = 0`: how many `True` reads of
  `isContinuousFocusLocked()` come before that move, counted from each
  switch-on from off.

The move sets the focus device's position directly: it must not trip the
quirk that switches the autofocus off on a Z move (FM-12).

### 12.8 Tests

- **Unit** (`tests/unit/test_engage_guard.py`, new, the Ti2-E fake of
  `test_mm_autofocus.py`): each refusal and its empty send log; a seen
  offset engages; the jump of run 1 trips the guard inside the settle
  window, switches off, forgets the offset and refuses the next engage; the
  band of a relock; a failed Z read; no focus drive; the D3 window and its
  WARNING; an undeclared stand engages as today.
- **Profile**: each key's bounds, the boolean refusals, `settle_s` against
  the lock timeout, the demo's equality.
- **Contracts** (`tests/contracts/test_autofocus.py`): the `autofocus`
  fixture reads `is_locked()` once, so that an operator's lock is seen; a
  case that needs a lock skips on `hardware` when the engage is refused,
  with the refusal's text; and `test_engage_at_an_unseen_offset_is_refused`,
  which skips on an undeclared stand and otherwise moves the offset by 1.0
  with PFS off (which moves no Z, §4.5), then expects the refusal and PFS
  off.
- **Freeing** (FM-98): the excursion and the refused engage that switches
  off go into `test_error_exits_free_the_core.py`.

### 12.9 Not verified

For session 2 (#198), before any engage at 40×:
- the band: 50 µm is a choice; how much a plate's bottom varies from one
  field to the next under PFS;
- `settle_s`: whether the delay between the lock and the jump is always
  under 1 s, and how long PFS takes to settle after a lost lock at 40×;
- the `PFS Status` bit (D3), read every 10 ms through an engage of a seen
  offset and through an offset move under lock;
- a Z read's time on the Ti2-E, which sets how far a jump goes before the
  guard reads it.

Limits of the design, named:
- **A sample changed while one `Microscope` stays open** keeps the old
  sample's seen offsets. A CLI command or `smc run` opens its own; M5's
  page, which stays open, must forget them when the operator changes the
  sample (M5's design).
- **A stand with no objective turret resolved** cannot tell objectives
  apart: its seen offsets survive a change of objective by hand.
- **A lock read in the 0.5 s before a jump** that the operator started at
  the stand (the button) is seen like any lock. The jump is then the
  operator's own; the layer's later engages of that offset run under D2.

### 12.10 A first lock from below (#331)

Extends D1 and D2 (§12.2), §12.4, §12.5 and the drive commands of m5 §6.
**Merging this section decides it.** The owner chose this route on #331
(2026-11-05): smc sets the floor and engages under its guard, the direction
is seen at the stand before it is declared, and E stays 50 µm. D1's route
and §12.5's text are edited in place to point here.

#### 12.10.1 What is known

Measured in Ti2-E sessions 1 and 2 (#208, #334), unless labelled:
- **The stand's button jumped again.** Session 2, a 24-well plate, the 10×,
  PFS off, focused by eye at Z 4191.34 µm, stored offset 9370: the button
  drove Z to 4397.92 µm, +206.6 µm and 48 µm past the 4350 µm soft limit.
  Session 1 jumped +235 µm from the same 9370. D1's refusal then advised
  that same button.
- **PFS drives Z to the plane its offset names, wherever Z starts.** What
  makes a first lock safe is therefore the offset that is set, not where Z
  is: a lower start only makes the jump longer.
- **With PFS off, an offset write moves no Z** (+10 and +50 units, §4.5),
  and only software writes it: the knob changes the offset only with PFS
  on.
- **Under lock, a larger offset gives a larger Z.** Every factor measured
  at 10× is positive, 5.9-10.8 units/µm (#208, #334); 9370 → 8141 brought Z
  from 4397.92 to 4195.26 µm (6.1 units/µm).
- **At an offset seen locked, PFS off then on moved Z by 0.00 µm** (#334,
  P1).
- **A larger Z is closer to the sample: inferred, not seen** (#98). The
  knob fully down reads 5-6 µm and samples focus near 4200 µm; the 100 µm
  jog of session 2 was too small to see from the side.
- **Inferred from the optics, not measured.** The offset lens changes the
  vergence of the PFS beam ahead of the objective, so an offset change
  moves the focal plane by about f² times that change: the same sign with
  every objective, and about 16 times fewer µm per unit at 40× than at 10×
  (f = 5 mm against 20 mm behind a 200 mm tube lens). Nothing below rests
  on the factor; D4 rests on the sign, and D5 bounds the case where the
  sign is wrong.
- **The 40×** is `3-Plan Apo LmbdD0.25 40x`. Nikon's catalogue gives the CFI
  Plan Apochromat Lambda D 40X NA 0.95 and a working distance of
  0.17-0.25 mm (not read at the stand, P6). On session 2's plate, if the
  40× is parfocal with the 10×, contact lies near 4365-4445 µm: the 4350 µm
  soft limit is 15-95 µm short of it, a bound and not a procedure.
- **A turn moves Z by itself** (+8.7 µm with the 4×) and switches PFS off
  (#330).
- **A turn also changes the offset** (session 3, #405 F3/F5; §8.9): from 0,
  the turn to the 10× read 9370 and the turn to the 40× 17794, and the turn
  to the 4× left 0. The 10× read 9370 at power-on too, so 9370 was never the
  operator's: it is the offset of session 1's and 2's button jumps. The
  reading is "an offset per objective, put back on a turn" (inferred, the
  4× shows none). What follows from the reads alone: **an offset set before
  a turn is not the offset after it**, whichever objective is turned to.
- **`PFSOffset` exposes no property limits**; the values seen are
  7803-9420 (#208), and session 3 read 0 to 20000 back as written and
  24370 for a write of 30000. Which values the device accepts is unknown.

#### 12.10.2 Decisions

**D4. A first lock: `engage()` at an unseen offset at or below the floor.**
On a declared stand, `engage()` also engages an offset that has not been
seen locked when that offset, as read back, is at or below the *floor*: the
lower bound of `[safety] autofocus_offset_soft_limits`. Under lock a larger
offset gives a larger Z, so the floor names the lowest plane smc can set,
below the focus of a sample that rests on the vessel's bottom: engaging
there drives Z away from the sample. The operator then brings the focus up
with the knob, PFS on, under their own eyes; smc reads the lock, and the
offset becomes seen (D1).
- **Only when the guard can bound the case where the floor is not below**:
  the floor, a focus drive to read, `[safety] max_engage_excursion_um` and
  `[safety] z_away_from_sample` (D5). With any of them missing, the unseen
  offset is refused as today, and the refusal names what is missing
  (§12.10.5).
- **`engage()` only.** The relock and `focus_by_um`'s relock keep D1 as it
  is: they bring Z back to where PFS held it, so an unseen offset there is
  a lost lock, not a first lock.
- **Not forceable, and nothing new to force**: the signature of `engage()`
  and the `Autofocus` Protocol do not change.
- **One floor for every objective**: there is one offset lens, and the sign
  of its effect does not depend on the objective (inferred, §12.10.1). A
  floor per objective is a later change, if a visit shows that PFS cannot
  lock from one floor with every objective.
- Not chosen: *the stand's button from the floor* (smc writes the floor,
  the operator presses). It is the same procedure while the inference
  holds, and an unguarded jump when it does not; the first lock at 40× is
  exactly where the inference has not been checked.
- Not chosen: *a `first_lock()` call* that writes the floor and engages
  under one lock. It changes the Protocol for every backend, and with D4
  and D6 each of the two calls is safe on its own, whatever the operator
  does between them.
- Not chosen: *carrying the offset from the 10×* (#331, option 3). It needs
  a relation between the two objectives' offsets that is not measured; the
  offset it gives is still unseen at 40×, with an error of unknown sign;
  and a turn moves Z and switches PFS off (#330). The visit below records
  the 10× and 40× focus offsets on one plate, should that relation ever be
  wanted.
- Not chosen: *an operator procedure alone* (#331, option 1). Only software
  writes the offset with PFS off, so any safe procedure contains D4's
  write.
- Not chosen: *a key of its own for the first offset*. The floor is the
  lowest offset smc may set, so it is the lowest plane smc can name; another
  key could only be higher.

**D5. A first lock's band is one-sided: it bounds Z toward the sample
only.** `[safety] z_away_from_sample = "decreasing" | "increasing"` declares
which way Z moves away from the sample. A first lock's band is
`(−∞, z_start + E]` when it is `"decreasing"` and `[z_start − E, +∞)` when
it is `"increasing"`, with `E` = `max_engage_excursion_um`; the Z soft
limits still bound both sides, as in D2. D2's symmetric band would trip on
the drop that D4 asks for. *Amended by #402*: the bounded side uses
`[safety] first_lock_max_rise_um` when the profile sets it (§12.11 D8).
- **A trip is D2's**: switched off at once, `AutofocusExcursionError`, and
  the offset forgotten (it was never seen). It means that the floor's plane
  was not below where Z started: either the operator had not focused first
  (Z was below the focus), or §12.10.1's inference is wrong for this
  objective. The message names both (§12.10.5).
- **The value names the reading, not a gesture.** On an upright or a
  light-sheet stand, "down" is not a sign of Z. This is the key #98 asks
  for; whether the Z contracts read it stays #98's.
- **Unset is unknown**, and no first lock runs. The Ti2-E's profile
  declares it once it has been seen from the side (the owner, 2026-11-05:
  seen, not inferred).
- `E` stays 50 µm on the Ti2-E (the owner, 2026-11-05). The 40×'s working
  distance leaves 170-250 µm (catalogue), and the guard reads Z after a jump
  has started.

**D6. On a declared stand, an offset move with the autofocus on needs a
lock and a factor.** `set_offset_native` and `offset_by_native` are refused,
not forceable, while the autofocus reads on, unless it reads locked and the
profile has an `[autofocus.offset_per_um]` factor for the current
objective. In that case §3.6's implied-Z check bounds it, as today.
- **On and searching**, a new offset is an engage at an unseen offset by
  another route: PFS drives Z to its plane the moment it finds the
  reference.
- **Locked with no factor**, the Z it implies is unknown, which is why
  `focus_by_um` already refuses it (§3.2).
- **Off**, the offset moves no Z (§4.5), and the write is free.
- This closes the way around D4's guard: the floor written while PFS is on.
  Undeclared stands keep §3.6. At 40×, until its factor is in the profile,
  the knob is the way to move the focus under lock, under the operator's
  eyes.

**D7. The operator's route, and the advice.** A first lock is two commands,
each a single mutating capability call (m5 §6.1):
1. PFS off, focus on the sample by eye (with PFS off, the knob moves Z);
2. `smc autofocus offset --floor`: the offset to its floor, Z does not move;
3. `smc autofocus engage`: Z drops to the floor's plane, away from the
   sample;
4. the knob, PFS on, until the sample is sharp. PFS holds that lock on the
   stand; the first read of the next `Microscope` (`smc autofocus get`, the
   scan's `is_locked()`, m3 §8) sees it, and the offset is seen for as long
   as that `Microscope` is open (D1).

D1's refusal of an unseen offset gives exactly this advice, and never the
stand's button (§12.10.5). The page gains no control here: its autofocus
control engages as today and shows the same refusal. A control for the
floor on the page is M5's (m5 §6.2), if the operator asks for one.

*Amended by #407.* **`--floor` comes after the last turn.** A turn changes
the offset (§12.10.1), so a floor set before it is gone after it, and an
engage would then be at the stand's own offset for the new objective:
unseen, refused. In step 1's terms: turn to the objective first, focus by
eye with PFS off, then `--floor`, then `engage`. The turn's own text says
when the offset changed (§8.11 D8), so the operator sees it before the
refusal does.

**A turn forgets no seen offset (#407, decided).** The question (#364,
item 3, asked again by #407): after a turn, must an engage refuse an offset
because it is the one the stand put back? It already does when it matters,
and needs nothing new:

- **The seen offsets are keyed by objective**, and a turn changes the
  objective: an offset seen locked at the 10× is no offset seen at the 4×,
  even when the stand leaves the same number there.
  `tests/unit/test_engage_guard.py::test_a_seen_offset_with_another_objective_in_place_is_refused`
  already pins it, for a turn made at the stand.
- **A turn records nothing.** It switches PFS off first (§8.3), so no read
  after it finds a lock; and a restored offset that was never read locked
  with that objective is unseen, and is refused with the first-lock advice
  (measured on `FakeCore` with the Ti2-E's layout, 2026-11-10: the 40×'s
  restored 17794).
- **A restored offset that was seen locked is a seen one, rightly.** It is
  the plane PFS held with that objective on this sample, which is D1's
  premise, and D2's band bounds the jump. Forgetting it on every turn would
  make "turn away and back, then engage" a first lock again for no gain in
  safety, on the page that stays open. D9 (§12.11) applies on top: at or
  below the floor, `engage()` is a first lock whether or not the floor was
  seen.
- What #364's item 3 asks, forgetting at a *sample* change, stays M5's: a
  turn is not one.

#### 12.10.3 Profile — `profile.py`

```toml
[safety]
# z_away_from_sample = "decreasing"  # which way Z moves away from the sample, "decreasing" or "increasing"; seen at the stand, never guessed (m2 §12.10, #98)
```

```python
class SafetySection(_Section):  # gains, after max_engage_excursion_um
    z_away_from_sample: Literal["decreasing", "increasing"] | None = None
```

- Any other value, a boolean included, is a `ProfileError` that names the
  key and its two values.
- `Profile.demo()` and `profiles/demo.toml` stay equal: the commented line
  goes into `demo.toml`, and `profiles/README.md` documents the key.
- `profiles/nikon-ti2.toml` gets the line commented, labelled *inferred, not
  seen (#98): declared at the visit once seen from the side*.
  `autofocus_offset_soft_limits` stays commented until the visit measures
  it (§12.10.7). Until both are set, the Ti2-E refuses a first lock and
  says why.

#### 12.10.4 Backend — `backends/mm.py`, `safety.py`

- **`Safety`** gains `z_away_from_sample: Literal["decreasing",
  "increasing"] | None = None` (keyword, last); the facade passes it
  (§12.6).
- **The engage check** (`_check_engage`, §12.4) gains a keyword that only
  `engage()` sets (the relocks do not). On a declared stand, when the
  offset read is unseen and that keyword is set and the offset is at or
  below the floor, it checks the first lock's conditions in this order: the
  floor, a focus drive, `max_engage_excursion_um`, `z_away_from_sample`.
  With all of them, the guard is built one-sided (D5) and the engage goes
  on; with one missing, it gives today's refusal with §12.10.5's advice. As
  D1's check, it runs inside the action, before the switch-on, in dry-run
  too.
- **`_EngageGuard`** gains `open_side: Literal["below", "above"] | None =
  None`: the band has no bound on that side (`"below"` when
  `z_away_from_sample` is `"decreasing"`). `what_happened()` names a first
  lock (§12.10.5).
- **D6** runs in the steps of `set_offset_native` and `offset_by_native`,
  beside `_check_implied_z`, before anything is sent, in dry-run too: on a
  declared stand, it reads whether the autofocus is on, then locked, then
  the factor. A read that fails refuses with its cause (unknown counts as
  on, FM-19). `focus_by_um` does not change: it already refuses when the
  autofocus is on without a factor.

#### 12.10.5 Errors and messages — `errors.py`

- **Unseen offset, a first lock possible** (§12.5's advice): `` … (seen:
  none)[; it was on and searching, so it was switched off]. Lock it from
  below (m2 §12.10): with PFS off, focus on the sample by eye, set the
  offset to its floor (5000 PFSOffset units), engage, then turn the knob
  with PFS on until the sample is sharp; smc then reads the lock ``.
- **Unseen offset, no first lock** (§12.5's advice): `` … (seen: none). A
  first lock from below (m2 §12.10) needs <[safety]
  autofocus_offset_soft_limits | a focus drive to read | [safety]
  max_engage_excursion_um | [safety] z_away_from_sample>, which this stand
  does not have; until then smc engages only at an offset it has read
  locked ``.
- **An unseen offset above the floor** gets the first of the two, with the
  floor's value.
- **Excursion during a first lock** (`AutofocusExcursionError`): ``
  autofocus PFS: Z went from 4195.00 µm to 4260.00 µm while it engaged from
  the offset's floor, more than [safety] max_engage_excursion_um (50 µm)
  toward the sample; it was switched off. The floor's plane is not below
  where Z started: focus on the sample by eye before a first lock; if Z was
  already there, the floor is not below this objective's focus (m2 §12.10),
  so do not try a first lock with it again. The guard reads Z every 10 ms:
  it stops PFS once a jump is read, it does not prevent one ``.
  *Amended by #402*: the bound and the advice of this text are §12.11
  D10's.
- **D6** (`SafetyRefusedError`, `how_to_force=""`): `` autofocus PFS:
  offset not moved: it is <on and not locked | locked, with no
  [autofocus.offset_per_um] factor for objective '3-Plan Apo LmbdD0.25
  40x'>, and on this stand it drives Z to the plane the offset names (m2
  §12.10). Switch it off to set the offset (that moves no Z), or turn the
  knob while it is locked ``.

#### 12.10.6 Drive and CLI — `drive.py`, `cli.py`

```python
def read_autofocus_offset(microscope: Microscope) -> DriveResult: ...
def set_autofocus_offset(microscope: Microscope, value: float | None) -> DriveResult:
    """``None``: the floor, the lower bound of the offset's soft limits."""
```

- `smc autofocus offset` reads only: `offset 8141 PFSOffset units, soft
  limits [5000, 12000]`, or `…, no soft limits in the profile (no floor:
  no first lock)`.
  *Amended by #407*: **an offset outside the soft limits says so.** Session
  3 read `offset 24370 PFSOffset units, soft limits [0, 20000]` and nothing
  said that 24370 lies outside what smc may set. The limits bound smc's
  writes, never the stand's (a turn restores, the knob under lock moves it),
  so a value outside them is the stand's. With a finite offset below the
  lower bound or above the upper one (the bounds themselves are inside), a
  clause follows the limits, a colon first, and `reached` stays true: the
  read worked, and a finding is not a failure of it.
  - Below: `offset -5 PFSOffset units, soft limits [0, 20000]: below them;
    smc would not set an offset there`.
  - Above, on a stand that declares `engage_drives_z_to_offset`: `offset
    24370 PFSOffset units, soft limits [0, 20000]: above them; smc would not
    set an offset there, and here it engages only at an offset it has read
    locked with this objective (m2 §12.2 D1), and the stand's own button
    would drive Z to the plane this offset names, as it did from 9370 (m2
    §12.1)`.
  - Above, on a stand that does not declare it: the first clause only,
    `…: above them; smc would not set an offset there`. The consequences of
    the last clause are the declared stand's (D1, §12.1).
  - **Why the two sides differ.** Measured on `FakeCore` with the Ti2-E's
    layout (2026-11-10): above the limits an engage at an offset not read
    locked is refused, and one at an offset read locked engages; below the
    floor an unseen engage is a first lock (D4), not a refusal. What a
    value below the floor means on the stand is #364's first item, open, and
    the Ti2-E has never been written below 0 (§12.11.4): the text claims
    only what is certain there.
  - An offset that is not finite gets no clause: it lies on no side.
- `smc autofocus offset --to VALUE` or `--floor` (one of the two;
  `--dry-run` as the other drive commands) calls `set_offset_native`, then
  prints the readback: `offset 5000 PFSOffset units`. A readback that
  differs from the value asked says so, `reached` false (FM-155).
- `--floor` with no soft limits exits 1 before anything is sent: `no floor:
  [safety] autofocus_offset_soft_limits is not set (m2 §12.10)`.
- Reading the floor (`offset_limits_native()`) is a read, so
  `set_autofocus_offset` stays one mutating call (m5 §6.1).

#### 12.10.7 The next visit to the Ti2-E (#198)

In this order, every command with `--profile nikon-ti2`. Nothing before
step 4 brings an objective toward a sample at 40×.
1. **Which way is away (#98).** No plate on the stage, the 10×: `smc z move
   500`, then `smc z move 4000`, watching the nosepiece from the side
   (3.5 mm). Write `z_away_from_sample` in the profile.
2. **The floor.** The 10×, PFS off (`smc autofocus disengage`): `smc
   autofocus offset --to V` for falling values (7000, 5000, 3000, 1000, 0),
   each read back, with `smc z get` before and after (no move expected,
   §4.5); then rising values the same way, to find the top. Write
   `autofocus_offset_soft_limits` from the lowest and highest values read
   back as written.
3. **A rehearsal at 10× on the plate.** The working distance is 15.2 mm, so
   a wrong move is harmless. Focus by eye with PFS off, then `smc autofocus
   offset --floor` and `smc autofocus engage`. Expected: Z goes down, then
   locks. `smc autofocus get` must then read locked: if it reads off, the
   end of the `engage` command switched PFS off (§12.10.9), and the engage
   of steps 3 and 4 is made from the page, which stays open. Then the knob,
   PFS on, until sharp; then `smc z get` and `smc autofocus offset`. If it
   does not lock from the floor, raise the floor
   by steps until it does, and write that floor. If Z goes up: stop, and
   do not take the 40×.
4. **The 40×.**
   - `smc autofocus disengage`, `smc z get`, `smc objective set "3-Plan Apo
     LmbdD0.25 40x" --confirm`, `smc z get` (#330). The turn's text names
     the offset when the stand changed it (§8.11 D8), and `--floor` below
     comes after the turn for that reason (D7, #407).
   - Focus by eye with PFS off; `smc z get` gives the by-eye focus.
   - `smc autofocus offset --floor`, `smc autofocus engage`, and `smc z
     get` at once: Z must be below the by-eye focus.
   - The knob, PFS on, until sharp; then `smc z get` and `smc autofocus
     offset`, read against the by-eye focus.
   - Two or three knob positions under lock, each with both reads, give
     the 40×'s factor for `[autofocus.offset_per_um]`.

#### 12.10.8 FakeCore and tests

No new knob: `autofocus_target_um` (§12.7) returns a plane below or above
where Z starts, from the offset the test writes.
- **Unit** (`tests/unit/test_engage_guard.py`, the Ti2-E fake):
  - a first lock at the floor drops Z more than E below where it started
    and returns `True`, with the offset seen; the mutant that keeps D2's
    symmetric band is killed;
  - a first lock whose plane lies above `z_start + E` trips: switched off,
    `AutofocusExcursionError` with the first lock's text;
  - a drop past the lower Z soft limit trips;
  - each missing condition (the floor, a focus drive, E, the direction)
    refuses with an empty send log and names what is missing;
  - an unseen offset one unit above the floor is refused, and one at the
    floor engages: the mutants `<` and `≤` both die;
  - the relock at an unseen offset at the floor raises
    `AutofocusLostError`;
  - `"increasing"` mirrors the band;
  - an undeclared stand engages as today.
- **D6** (`tests/unit/test_mm_autofocus.py`): on and searching, and locked
  with no factor, are refused with nothing sent, in dry-run too; off is
  sent and moves no Z; locked with a factor gets today's implied-Z check;
  an undeclared stand is unchanged; a failed read refuses.
- **Profile**: the two values, any other string, a boolean, the demo's
  equality.
- **Drive and CLI**: the read; `--floor` with and without soft limits (exit
  1, nothing sent); `--to`; both options together; dry run; a readback
  that differs.
- **Freeing** (FM-98): the first lock's trip and D6's refusal go into
  `test_error_exits_free_the_core.py`.
- **No contract runs a first lock**: it moves Z by what the stand decides.
  The visit runs it by hand (§12.10.7).

#### 12.10.9 Not verified

- That PFS locks from the floor at all, at 10× and at 40× (steps 3 and 4).
- That the floor's plane lies below the focus at 40×: the sign is inferred,
  and D5 bounds the other case.
- The 40×'s factor: about 16 times the 10×'s if §12.10.1's inference holds.
- How long a long drop takes, against the 5 s lock timeout and the 1 s
  settle window.
- Which way the knob moves Z with PFS on (#208); the operator reads the
  image either way.
- Whether switching PFS off in the middle of a jump stops Z at once (#205).
- Whether a lock taken by `smc autofocus engage` holds once the command has
  ended and unloaded the stand. Loading leaves PFS as the last user left it
  (§12.1); unloading was never read. The floor is written with PFS off, so
  only the engage depends on it (§12.10.7, step 3).

Limits of the design, named:
- **The stand's button stays outside smc.** A press at an unseen offset
  jumps whatever smc says; D7 only stops smc from advising it.
- **A first lock rests on a sign and on the operator's by-eye focus.** The
  guard limits a wrong sign to E (`R` since §12.11 D8) plus what Z covers
  between two reads; it does not prevent it.

### 12.11 A first lock whose floor lies above the focus (#402)

Extends §12.10 (D4, D5, §12.10.4, §12.10.5). **Merging this section
decides it**: it answers #402's three questions, with the options not
chosen and why, and fixes what the 40× does on 2026-11-11. No Protocol and
no signature changes.

#### 12.11.1 What is known

Measured in Ti2-E session 3 (2026-11-10, #405; `nikon-ti2.md`, *Session
3*), the 40× `3-Plan Apo LmbdD0.25 40x`, a Greiner 24-well plate with a
standard plastic bottom (thickness not read) and debris, no cells:
- **C4** (XY (−9545, 1932), near its −Y wall): focused by eye at Z
  4005.56 µm, `--floor` (0), `engage`: locked at 3958.68 µm, **46.9 µm
  away**, at the first try. Sharp under lock at offset 5388, Z 4004.58 µm;
  the 40×'s factor is about 115 units/µm.
- **C3** (XY (10031, 9375)), reached with PFS off and Z 200 µm down: focused
  by eye at Z 4017.28 µm, `--floor`, `engage`: the guard tripped at
  4074.08 µm, **56.8 µm toward the sample**, 6.8 µm past its 50 µm bound, one
  read late. Z read 4074.14 µm next: the switch-off stopped Z at once.
- So the plane the floor names lay 46.9 µm below the focus in C4 and at
  least 56.8 µm above it in C3: relative to the focus, **one offset's plane
  moved by at least 104 µm between two wells**, while the two by-eye foci
  lie 11.7 µm apart.
- At the 40×, PFS stayed locked over the plastic between B4 and C4 (the
  `C4-edge` grid, Z within 1.6 µm); inside C4, Z varied by 1.7 µm over
  1.4 mm, and an engage at a seen offset moved Z by −0.04 µm.
- **Why the plane moved is not known.** Hypotheses, inferred only: (a) PFS
  takes another interface of the thick bottom in one well than in the other
  (the outer air-plastic surface or the inner one); a lock held over the
  plastic between wells fits the outer one at C4; (b) the bottom's
  thickness differs by about 100 µm between C3 and C4; (c) one by-eye focus
  was on another plane, such as debris under the bottom, which the 11.7 µm
  between the two foci makes unlikely.
- **Measured on the Ti2-E fake** (`main` caaa60e, planning #402): a first
  lock whose plane lies 8 µm above where Z started locks and returns
  `True`, since its band is `z_start + 50`; and once the floor has been
  seen locked, an engage at the floor from another Z, whose plane lies
  40 µm above it, is a seen offset under D2's symmetric band and locks
  40 µm toward the sample with no word. In one `Microscope` (the page), the
  second well of F11 would not have been a first lock at all.

#### 12.11.2 Decisions

**D8. A first lock's bound toward the sample is a tolerance, not E.**
`[safety] first_lock_max_rise_um` (`R`) bounds how far Z may move toward
the sample during a first lock: the band is `(−∞, z_start + R]` when
`z_away_from_sample` is `"decreasing"` and `[z_start − R, +∞)` when it is
`"increasing"`. Unset, `R` is `E`: today's band.
- **Why not E.** E bounds the plane of a seen offset, which lies where the
  sample's own shape puts it (D2). A first lock asks for a drop, and any
  rise means that its premise failed (D5): its bound is the noise of a Z
  read, not the shape of a plate.
- **What it buys.** F11's trip would have been read at about `R` + 6.8 µm
  (the travel of one read in F11, one observation) instead of 56.8 µm.
- **The Ti2-E sets `R = 5` µm** (chosen: Z reads repeat within 0.02 µm;
  PFS off then on at a seen offset moved Z by 0.00 µm, a re-engage by
  −0.04 µm). Z was not traced during a first lock that worked (§12.11.4).
- **Validation**: positive, finite, not a TOML boolean (#106), and not above
  `max_engage_excursion_um` when both are set (a `ProfileError` that names
  both keys): a first lock looser than an engage at a seen offset is a
  mistake. `Safety` checks the same with a `ValueError`.
- **Not a condition of a first lock**: §12.10.4's four stay as they are.
- **One value for every objective**: a rise means the same at every
  magnification.
- Not chosen: *a smaller E on the Ti2-E*. E also bounds the scan's relocks
  at the 10×, across a plate that no session has traced; a trip there fails
  the scan for no danger (working distance 15.2 mm).
- Not chosen: *E per objective* (#402's third question). `R` covers the
  first lock, where F11 happened; inside a well at the 40× a seen offset
  moved Z by less than 2 µm. A 40× scan that relocks across walls is #416's.

**D9. At or below the floor, `engage()` is always a first lock.** On a
declared stand with the first lock's four conditions met and the autofocus
off, `engage()` at an offset at or below the floor builds D5's one-sided
band with `R`, whether or not that offset has been seen locked.
- **Why.** F11: one offset with one objective named planes at least 104 µm
  apart, relative to the focus, in two wells. A lock seen at the floor in
  one well says nothing of the floor's plane in the next, and the floor is
  set exactly when the operator starts at a new place. D1's path would
  bound it by E both ways, and it locked 40 µm up with no word (§12.11.1).
- **The autofocus on** at a seen floor (locked, or searching): D1's path,
  as today. On at an unseen one: refused and switched off, as today.
- **The relocks** (`first_lock=False`) keep D1: they bring Z back to where
  PFS held it.
- A first lock that locks leaves the floor seen; one that trips forgets
  it, as today.
- Not chosen: *seen offsets kept per XY*. A field, a well and a plate are
  m3's notions, and the layer would have to guess a radius. A seen offset
  above the floor engaged at another place stays D1's and D2's (§12.11.5).

**D10. The trip names the place, not the objective.** §12.10.5's text for
a first lock's trip toward the sample becomes, with `R` set:

`` autofocus PFS: Z went from 4017.28 µm to 4074.08 µm while it engaged
from the offset's floor, more than [safety] first_lock_max_rise_um (5 µm)
toward the sample; it was switched off. The floor's plane is not below
where Z started: focus on the sample by eye before a first lock; if Z was
already there, the floor's plane lies above the focus at this place (m2
§12.11), and no offset smc may set names a lower one: do not engage PFS
here with this objective. The guard reads Z every 10 ms: it stops PFS once
a jump is read, it does not prevent one ``

Without `R`, the bound clause stays `more than [safety]
max_engage_excursion_um (50 µm) toward the sample`, and the advice is the
new one. The INFO line before the switch-on names the bound in force:
`` … Z may move away from the sample, and at most 5 µm toward it ``.
- **Why**: "the floor is not below this objective's focus … do not try a
  first lock with it again" read as a verdict on the 40×, which had locked
  from that floor in C4 ten minutes earlier.

**#402's questions, answered.**
- *Reuse an offset seen locked in another well instead of the floor*: no.
  Under lock a larger offset names a higher plane (§12.10.1), and the floor
  is the lowest offset smc may set: in C3 the offsets seen in C4
  (5388-5698) name planes about 47-50 µm above the floor's, which already
  lay at least 56.8 µm above the focus.
- *A floor per objective*: no; D4 stands. A floor of its own could only lie
  above the lowest value the device accepts, and so name a higher plane. A
  lower floor needs offsets below 0, which no session has written
  (§12.11.4).
- *An excursion bound per objective*: not now (D8).

#### 12.11.3 The 40× on 2026-11-11

The owner scans at the 10× first, then at the 40× (#402). Every command
with `--profile nikon-ti2`:
1. **The F11 probe, at the 10×** (working distance 15.2 mm: harmless), in
   C3 then in C4: `smc autofocus disengage`; `smc z move <Z − 200>`; `smc
   stage move X Y`; focus by eye, PFS off, then `smc z get` (`Z_eye`);
   `smc autofocus offset --floor`; `smc autofocus engage`; `smc z get`
   (`Z_floor`); the knob under lock until sharp, `smc z get` and `smc
   autofocus offset`. If `Z_eye − Z_floor` agrees within about 20 µm
   between the two wells, the plane moves at the 40× only (hypothesis a,
   at NA 0.95); if it differs by 100 µm or more, it moves from well to well
   at every objective (a or b). The procedure below holds either way.
2. **The bound.** With #402's PR merged and pulled, the profile has
   `first_lock_max_rise_um = 5.0`. Without it, the stand's copy sets
   `max_engage_excursion_um = 10.0` for the 40× part only (it bounds a
   first lock's rise and every engage; at the 40× a seen offset moved Z by
   less than 2 µm), and 50.0 again before any 10× scan.
3. **One well at a time, PFS off across every wall.** PFS locked and
   following its reference over a wall drives Z with no guard (D2: no
   watchdog between actions), and F11 says the next well's plane can lie
   104 µm elsewhere (#416). Per well: `smc autofocus disengage`; `smc z
   move <Z − 200>`; `smc stage move X Y`; focus by eye with the knob, PFS
   off; `smc autofocus offset --floor`; `smc autofocus engage` (the CLI,
   not the page, until D9 is merged); `smc z get` at once, below the
   by-eye focus; the knob under lock until sharp; then that well's scan
   only.
4. **A trip** (`AutofocusExcursionError` from the floor): that well is not
   imaged with PFS at the 40×. `smc z move <Z − 200>`, note it, next well;
   never the stand's button, never a second engage there.
5. **Recorded per well**: `Z_eye`, `Z_floor` or the trip's Z, the sharp Z
   and its offset: the data that settles F11.

#### 12.11.4 Not verified

- Which hypothesis of §12.11.1 holds; §12.11.3's probe separates "the 40×
  only" from "every objective".
- Whether PFS moves Z toward the sample before it drops on a first lock
  that works: Z was not traced during one. If `R` trips first locks where
  the by-eye focus says the floor lies below, `R` is raised in the profile,
  never above E.
- How fast a PFS jump travels: 6.8 µm in one read in F11, once.
- Whether `PFSOffset` accepts values below 0, which would lower the floor:
  never written.
- Whether F11's jump touched the plate (not answered at the stand).

#### 12.11.5 Limits of the design, named

- **A seen offset above the floor, engaged at another place** (the page
  after an XY move, or the scan after a lost lock in a new well) is D1's
  and D2's: bounded by E, 50 µm on the Ti2-E, while F11 says its plane can
  lie 104 µm away. At the 40×, §12.11.3 avoids it; the scan is #416's.
- **PFS locked and following over a wall** is outside every guard (D2's
  *Not chosen*), at every objective; #416.

## 13. Offsets between objectives (#340)

A plate registered with one objective (the 4× or the 10×: a wide field,
walls easy to see) is scanned with another (the 40×). The same feature is
not centred at the same stage position under two objectives
(parcentricity), nor in focus at the same Z (parfocality). This section
fixes where the stand's offsets are written, how the layer hands them out,
and how session 3 measures them. Applying them to a plate is m3 §18.

### 13.1 What is known

- The Ti2-E's turret holds `1-Plan Fluor 4x`, `2-Plan Fluor 10x` and
  `3-Plan Apo LmbdD0.25 40x` (`docs/hardware/nikon-ti2.md`).
- **A nosepiece turn moves Z by itself** (session 2, #330): 10× → 4× → 10×
  read 500.0 µm at the call's return (from 561.5 µm), 570.2 µm with the 4×
  (+8.7 µm) and 561.4 µm back with the 10×. The stand lowers Z for the turn
  and puts it back with a correction of its own per objective. A Z offset
  that smc applied on top would count it twice. A turn now reads Z before
  and once it is still after, and the drive's text says what the stand did
  (§8.11 D2, D7); session 3's reading of Z after a turn can start from it.
- The stand switches PFS off on a turn (session 2), which m2 §8.3 already
  assumes.
- No XY offset between the three objectives has been read. Nothing in the
  layer or in a plugin carries one today; `route` says so in its `assumed`
  (m3 §11).
- A read right after a set returns the previous value on this stand
  (#330); the turret's `current()` right after a turn is not verified, and
  the turn re-reads it until it is the label asked for (§8.11 D3).

### 13.2 Definitions (decided)

The offsets are relative to one **reference objective**, named by the
profile. For an objective `O`, with the same point-like feature centred in
the frame under `O` and under the reference, each after a focus:

- `xy_offset_um(O) = (x_O − x_ref, y_O − y_ref)`, the stage positions as
  `XYStage.position_um()` reads them;
- `z_offset_um(O) = z_O − z_ref`, the focus stage's readings
  (`ZStage.position_um()`) after the operator focused, that is **after**
  the stand's own correction of §13.1.

The reference's own offset is `(0, 0)` and `0` by definition. A position
`p_A` found with objective `A` is the position `p_B = p_A + xy_offset_um(B)
− xy_offset_um(A)` with objective `B` (m3 §18). The shift is a translation:
the same for every position on the stage, which session 3 checks (§13.6).

**The layer applies no Z offset** (in this section or in m3 §18): the
stand corrects Z itself on a turn, and no plugin sets an absolute Z from a
record made with another objective. `z_offset_um` is stored and shown so
that session 3's reading is not lost, and so that a later issue (the route
to another objective, m3 §11) can use it once its meaning on this stand is
settled.

### 13.3 Profile — `profile.py`

```toml
[objectives]
reference = ""   # the objective every offset is measured from, by its turret label; "" = no offsets (m2 §13)

# [objectives.offsets."1-Plan Fluor 4x"]
# xy_offset_um = [0.0, 0.0]   # stage XY under this objective minus under the reference, the same feature centred
# z_offset_um = 0.0           # focused Z under this objective minus under the reference, after the stand's own correction
# measured = "YYYY-MM-DD, session N (#issue)"
```

```python
#: An offset beyond these is a unit or sign mistake, not a measurement
#: (FM-121): parcentric objectives differ by tens of µm. Chosen, not measured.
MAX_XY_OFFSET_UM = 1000.0
MAX_Z_OFFSET_UM = 500.0


class ObjectiveOffsetEntry(_Section):
    xy_offset_um: tuple[float, float] | None = None
    z_offset_um: float | None = None
    measured: str = ""


class ObjectivesSection(_Section):
    reference: str = ""
    offsets: dict[str, ObjectiveOffsetEntry] = Field(default_factory=dict)

    def offset(self, label: str) -> ObjectiveOffset: ...


class Profile(_Section):  # gains, after `channels`
    objectives: ObjectivesSection = Field(default_factory=ObjectivesSection)
```

Validation, each a `ProfileError` naming the key:

- each number of `xy_offset_um` and `z_offset_um` is finite, not a boolean
  (`_no_boolean`), and within `±MAX_XY_OFFSET_UM` per axis or
  `±MAX_Z_OFFSET_UM`; `xy_offset_um` has exactly two numbers;
- an entry that sets either value has a `measured` that is not blank: an
  offset with no date and session is a guess;
- `offsets` that is not empty needs a `reference` that is not blank;
- the reference has no entry of its own: its offset is zero by definition,
  and an entry would be a second, contradicting statement;
- `reference` and the labels are not checked against the turret here (the
  profile cannot know it); `smc objective offset` shows a label that is not
  on the turret (§13.5).

`ObjectivesSection.offset(label)` returns, for:

- `label == reference` (not blank): `xy_um=(0.0, 0.0)`, `z_um=0.0`,
  `measured=""`;
- a label with an entry: the entry's values (either may be `None`);
- anything else, `""` included, or any label when `reference` is blank:
  `xy_um=None`, `z_um=None`, `measured=""`.

`reference` in the result is `None` when the profile names none. It never
raises and reads nothing from the stand.

`Profile.demo()` and `profiles/demo.toml` stay equal: the demo sets no
offsets, and `demo.toml` carries the block above, commented.
`nikon-ti2.toml` gets the same commented block with its three labels and
`reference = "2-Plan Fluor 10x"` commented too: nothing is entered before
session 3 has measured it. `profiles/README.md` documents the section, the
sign convention of §13.2 and the procedure of §13.6.

### 13.4 Capability — `capabilities.py`, `backends/mm.py`, `microscope.py`

```python
@dataclass(frozen=True, slots=True)
class ObjectiveOffset:
    objective: str
    reference: str | None  # None: the profile names no reference objective
    xy_um: tuple[float, float] | None  # None: not measured
    z_um: float | None  # None: not measured
    measured: str  # when and in which session; "" for the reference and when unknown
```

`ObjectiveOffset` joins `__all__`. `ObjectiveTurret` (§8.2) gains:

```python
class ObjectiveTurret(Protocol):  # gains
    def offset(self, label: str) -> ObjectiveOffset: ...  # the profile's
```

- `MMObjectiveTurret.__init__` gains the keyword `offset_of:
  Callable[[str], ObjectiveOffset]`; `_build` passes
  `offset_of=profile.objectives.offset` (a bound method of the profile's
  section, not of the capability: FM-92 holds). `offset(label)` returns
  `offset_of(label)`.
- **Kind of call** (extends §8.7): none. It reads the profile only: no
  lock, no executor, allowed while halted, in a dry run and while the stand
  moves.
- Every class in `src/` and `tests/` that implements `ObjectiveTurret` for
  a test gains the method (`runtime_checkable` checks that it exists).

### 13.5 CLI — `smc objective offset`

```text
smc objective offset [--profile NAME]
```

Opens the stand as `smc objective list` does and prints one line per
turret label in position order, empty labels skipped, the current one
marked as `list` marks it:

```text
'1-Plan Fluor 4x'  xy (+12.50, -4.00) µm  z -8.70 µm  measured 2026-11-12, session 3 (#NNN)
'2-Plan Fluor 10x'  (current)  reference
'3-Plan Apo LmbdD0.25 40x'  xy unknown  z unknown
```

A known value is printed with its sign and two decimals; an unknown one
as `unknown`, never `0`. Then, one line per profile entry whose label is
not on the turret, and one if the reference is not on it: `'<label>' is in
the profile's [objectives] but not on the turret: check the label`. With
no reference: one line, `the profile names no reference objective
([objectives] reference): every offset is unknown`. It exits 0 in every
case above: it is a read-out. Its help says what the offsets are for and
where they are written.

**Writing them (#380).** `smc objective offset --from-run-doc PATH
[--session TEXT] [--write]` opens no stand: it prints the entries that a run
of `objective-offsets` measured and what they replace, and with `--write`
writes them into the profile's `[objectives]`, inside a block between two
marker lines that it alone edits. The rules are m3 §20.7.

### 13.6 What session 3 measures (for `/hardware-session`)

**Since #380 the tool measures the XY offsets from the plate** (m3 §20): a
registration with `plate-walls` under one objective, the same wells
re-measured under each other one. The procedure below is run **once** at the
Ti2-E to compare with the tool's values (the same reference, the difference
per objective against the bound of step 6), and stays the way to read a
focused Z, which the tool does not measure.

The procedure for the session's checklist, with PFS off (a turn switches
it off anyway), the magnifier lever at 1.0× (the offsets are for it,
FM-103) and the operator focusing by hand or with `smc z jog`:

1. **A point, not a line.** A small, sharp feature visible with all three
   objectives: a stage micrometer's cross, the corner of a printed grid, a
   fine pen dot on the underside of a plate. A wall is a line: it centres
   one axis only.
2. With the **10×** (`smc objective set "2-Plan Fluor 10x" --confirm`),
   focus, centre the feature in the frame (within 2 px of the centre of a
   snap), and read `smc stage get` and `smc z get`.
3. Turn to the **4×** with the stage clear (objective clearance as the
   checklist says). Read `smc z get` **before** focusing (the stand's own
   correction, §13.1), then focus, centre, read XY and Z.
4. The same with the **40×**: Z after the turn, then focus, centre, XY, Z.
5. Back to the **10×**: centre again and read. The difference with step 2
   is the 10×'s own repeatability, and bounds what the offsets can claim.
6. Repeat steps 2 to 5 once more, at another feature far across the plate
   if one is available: the offset is a translation only if both places
   agree within that repeatability.

Each offset is the mean of the two cycles, `p_O − p_10x` within the same
cycle. It goes into `nikon-ti2.toml` with `reference = "2-Plan Fluor
10x"` and `measured = "<date>, session 3 (#<report>)"` only if the two
cycles agree within the 10×'s repeatability plus 2 px of the coarser
objective (5.5 µm at 4×); otherwise the readings go into the report and
nothing is entered.

### 13.7 Tests

- `tests/unit/test_profile.py`: the section loads from TOML with quoted
  labels; each refusal of §13.3 (a value with no `measured`, a blank
  `measured`, a value beyond each bound, NaN, a boolean, three numbers,
  offsets with no reference, an entry for the reference); `offset()` for
  the reference, an entry, an unknown label, `""` and a blank reference.
- `tests/contracts/test_objective_turret.py`:
  `test_offset_is_unknown_or_the_profiles` (every label; on `fake` it
  sends nothing, the call log unchanged).
- `tests/test_cli_drive.py`: the read-out's lines on a demo profile with
  offsets (an entry, the reference, an unknown label, a profile label not
  on the turret) and on the plain demo (no reference).

### 13.8 Not verified

- The offsets themselves, and that they are a translation (§13.6).
- Whether the Ti2-E's own Z correction on a turn (§13.1) is the whole
  parfocality, leaving `z_offset_um` near 0.
- Whether `current()` reads the new objective right after a turn (§8.11 D3
  re-reads it, #330).
- Whether the magnifier lever at 1.5× moves the image's centre: the
  offsets are keyed by objective only and measured at 1.0× (FM-103).
- Whether NIS-Elements applies an offset of its own when its objective
  changes, which decides how the route's positions for another objective
  are handed on (m3 §11, §18).
