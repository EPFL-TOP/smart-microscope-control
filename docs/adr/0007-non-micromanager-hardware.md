# 0007 — Hardware without a Micro-Manager adapter, and vendor-controlled systems

- **Status**: **accepted** 2026-11-16 by the owner (in chat, in the design
  session of #25), as revised that day with what M1–M3 taught (first
  revision 2026-09-22, after the owner's direction on the LS1). The design
  for the Zeiss is [`docs/design/m4-zeiss.md`](../design/m4-zeiss.md)
- **Date**: 2026-11-16 (first written 2026-09-22)
- **Issue**: #4

## Context

ADR-0002 makes Micro-Manager the bus. Two of the five microscopes are not
reachable through an existing adapter out of the box. This record was first
written before any of the layer existed; M1–M3 have since built it on the
Nikon Ti2-E, and the section *What M1–M3 taught* below is what that changes.

### Zeiss Axio Observer 7

Micro-Manager's `ZeissCAN29` adapter speaks CAN29 over a serial port. This
stand carries CAN29 **over USB into Zeiss's `CZCanSrv`** (every device in
its MTB configuration says `PortType="USB"`), so the serial adapter finds
nothing at any baud. `lightsheet-live-tracking-tool` drives the stand
through **MTB 2011** (Zeiss's hardware abstraction service) via
`pythonnet`: stage axes, focus, piezo, objective changer, reflector,
sideport, condenser, TL shutter, Colibri LEDs and Definite Focus 2 all work
(`tracking_tools/microscope_interface/mtb.py`). The camera (Prime 95B)
works through Micro-Manager's `PVCAM` adapter. Two hard constraints: **one
MTB login per process** (logout kills the realtime subsystem for good) and
**one PVCAM handle per camera**.

The survey of 2026-09-24 (`docs/hardware/zeiss-observer7.md`) confirmed this
first-hand: the stand enumerates as a USB `ZeissCanNode`, the `MicoIf` FPGA
card is present with no adapter, `PVCAM` sees the camera, and the PC runs
Python 3.14 on Windows 10 LTSB 2015. `pythonnet` 3.2 supports Python 3.11
to 3.15.

### Viventis LS1 (two stands)

The LS1 ships with Viventis's control software and its scripting API,
**PyMCS** (`Microscope`, `StageXYZ`, `Camera`, `TimeLapseController`,
`AcquisitionController`). Everything the lab has done on the LS1 so far
runs *inside* that software: the tracking tool reads frames from the
acquisition folder and nudges positions; `viventis_control` draws ablation
point patterns. That model has the same ceiling ZEN had — the vendor owns
the hardware and the acquisition loop, and we bolt on from outside.

**Owner's direction (2026-09-22): do not build on PyMCS. Drive the LS1's
components directly, as we do for every other stand.** A light sheet is a
set of components — an XYZ stage, one or two cameras, lasers, scan
mirrors, filter changers, a piezo, and a controller (DAQ/FPGA) that does
the hardware timing between galvo sweep, camera exposure and laser
blanking. Many of these are standard parts with Micro-Manager adapters;
which ones the LS1s actually contain, and what performs the timing, has
still never been inventoried (#29 is open; `smc discover`, which did the
Zeiss survey, now exists for it).

### The tracking tool is a tool

`lightsheet-live-tracking-tool` is, first of all, a live tracking
application. In this architecture that makes it a **plugin** (ADR-0004),
not a set of microscope backends: its trackers and its closed-loop logic
survive; its five `MicroscopeInterface_*` backends are superseded by the
capability layer.

### What pymmcore-plus offers

`pymmcore_plus.experimental.unicore` provides a `UniMMCore` — a subclass
of `CMMCorePlus` — that loads **devices written in Python** alongside C++
adapters, declared in an ordinary `.cfg` by lines prefixed `#py`. Base
classes: `XYStageDevice`, `XYStepperStageDevice`, `StageDevice`,
`StateDevice`, `ShutterDevice`, `CameraDevice`, `HubDevice`,
`GenericDevice`, `SLMDevice` — **still no `AutoFocusDevice`** (upstream
`main`, 2026-11). unicore holds **no per-device lock**: concurrent calls to
a Python device are that device's own business, where MMCore serialises
calls to a C++ one.

The latest release, **0.18.1 (2026-04-02), lacks two upstream fixes this
stand needs**: pymmcore-plus#629 (a Python `StateDevice` that moves leaves
the cached `State`/`Label` stale, so the pixel size reads wrong after an
objective change) and pymmcore-plus#635 (`getProperty("Core", <role>)`
does not see Python devices). Both are merged on `main`, unreleased.

### What M1–M3 taught

1. **The layer is a Micro-Manager layer, and that is now an asset.** The
   nine capabilities are implemented once, as the `MM*` classes over a
   `CMMCorePlus` (`backends/mm.py`, ~4 200 lines); the safety guards —
   jog limit, Z soft limits, the Z↔autofocus interlock, confirmed turret
   moves, dry-run, the `Executor`'s stop contract (FM-17, FM-18, FM-19),
   segmented moves for a stand that refuses `stop()` (#204, #221) — are
   called from inside those classes. A device that answers the core's API
   inherits all of it; a backend that does not would re-implement it.
2. **There is no backend selection.** `Microscope._build` builds the `MM*`
   classes for whatever the core holds; roles come from the profile, the
   core's slots and type heuristics. `backends/nikon.py` is not a backend
   but driver facts for `smc doctor`. A profile has no backend key and no
   Python-device declaration; it names a `.cfg`.
3. **The autofocus is PFS-shaped where it touches the layer.** The
   `Autofocus` protocol separates engaged / locked / in range and keeps the
   offset in device units (`m2 §3`); it was written not to preclude DF2,
   and a one-shot `stabilize_now` was deferred (`m2-nikon.md`, #13). But
   `MMAutofocus` acts on the core's AutoFocus slot, `Role.autofocus`
   accepts only an `AutoFocus` device type, and `Microscope._interlock`
   returns only an `MMAutofocus` — any other `Autofocus` gets no Z
   suspend/relock today.
4. **The web UI opens and closes the facade within one process.** `smc ui`
   owns one `Microscope`; *Connect* and *Disconnect* open and close it
   repeatedly (`m5 §7`, so that the vendor software can take the stand),
   and `close()` unloads every device. With "one MTB login per process,
   logout is final", an MTB session tied to the facade's lifetime would
   break the second *Connect*.
5. **Reads must never wait behind a move.** STOP and the page's tick read
   `state()` on the server loop while a worker thread drives the stand
   (FM-70, FM-184; #333 was a read blocked 464 ms behind a snap). MMCore
   gives that for C++ devices; a Python device must give it itself.
6. **Vendor-independent device lessons recur on any stand**: `stop()`
   refused while the move runs on (FM-117), busy never True and the set
   call is the wait (FM-19), a read right after a set returns the old value
   (FM-188), the stand moves itself during a turret turn (FM-195), values
   reformatted and quantised (FM-100, FM-119), a following autofocus drives
   Z to its own reference on engage and its reference differs between
   wells and objectives (FM-118, FM-215, FM-223).
7. **Nothing in M4 or M6 needs the Nikon.** Both wait on this decision only.

## Options

### For a stand whose vendor API is the only route (Zeiss)

- **A — Wrap the vendor API as unicore Python devices** loaded in the same
  `UniMMCore` as the PVCAM camera. Roles, capabilities, every guard of the
  layer, `useq` MDA and plugins work unchanged. M1–M3 make this stronger
  than it was: the layer the devices plug into now exists and is proven.
- **B — A direct backend implementing the capability protocols**, camera in
  a separate core. Simpler per device; two buses, no `useq` MDA over the
  stage, and every `MM*` class with its guards implemented a second time.
- **C — A C++ Micro-Manager adapter for MTB.** The upstream route; months
  of work, a .NET/C++ build chain, licensing questions.

### For the MTB session's lifetime

- **S1 — Process-lifetime session.** Logged in at the first *Connect*,
  logged out only at process exit; *Disconnect* unloads the devices and
  keeps the login.
- **S2 — MTB in a child device-host process**, killed at *Disconnect*.

### For Definite Focus 2

- **F1 — A local `Autofocus` implementation** over a `GenericDevice` (B for
  this one device), the interlock generalised to any `Autofocus`, and an
  `AutoFocusDevice` proposed upstream without waiting for it.
- **F2 — Upstream first**: contribute `AutoFocusDevice`, wait for its
  release, then use `MMAutofocus` unchanged.

### For a system built from standard components (LS1)

- **A′ — Components through Micro-Manager**, exactly like the Nikon stands:
  an inventory, a `.cfg` built from the devices that answer, a profile, and
  unicore Python devices for any component without an adapter. The vendor
  software is not running while we drive the hardware.
- **B′ — PyMCS as a backend.** Rejected by the owner: it keeps the vendor
  loop, offers relative moves and file watching only, and requires our
  code to run inside the vendor's process.

The risk in A′ is not the stage or the cameras — it is the **light-sheet
timing**. If the sweep/exposure/laser synchronisation lives in a vendor
controller with no public protocol, Micro-Manager can still drive
hardware-triggered sequences (camera external trigger, `NIDAQ`, Arduino or
TriggerScope adapters), but re-implementing that synchronisation is real
work and must be scoped before anything else on the LS1 is promised.

## Decision (proposed)

1. **Zeiss Axio Observer 7: option A.** The MTB components are unicore
   Python devices in the same `UniMMCore` as the `PVCAM` camera, declared
   by `#py` lines in `profiles/zeiss-observer7.cfg`; the profile gains no
   key. B is the fallback for a device unicore cannot express, and today
   that is DF2 only (point 4).
2. **The core is a `UniMMCore` only when the `.cfg` declares Python
   devices.** The core opener reads the `.cfg` before loading it and picks
   the class; the Nikon stands stay on `CMMCorePlus` and carry none of
   unicore's experimental code.
3. **The MTB session lives as long as the process (S1).** One
   process-wide session owner logs in at the first *Connect* and logs out
   at process exit (atexit), never on `Microscope.close()`; *Disconnect*
   unloads the Python devices, darkens the light, and keeps the login.
   Whether ZEN can take the stand while our process holds the login is an
   **unknown**, measured at the first session (#27) and reported, not
   assumed.
4. **Definite Focus 2: F1.** DF2 is a `GenericDevice` with its own
   `Autofocus` implementation; `Microscope._interlock` accepts any
   `Autofocus` that implements the interlock, not only `MMAutofocus`; the
   protocol gains the one-shot `stabilize_now` that #13 deferred, and the
   per-position reference blobs live in the run document (#26). An
   `AutoFocusDevice` is proposed to pymmcore-plus in parallel; when a
   release carries it, DF2 may move onto it.
5. **Version: wait for a release.** M4 development runs on a fake MTB API on
   any OS. The `[zeiss]` extra and the first stand session (#27) require a
   pymmcore-plus release newer than 0.18.1 that carries #629 and #635; we
   ask upstream for one. No git-commit pin is installed on a stand PC.
6. **Every Python device is safe for concurrent calls**: it holds its own
   lock, a read never waits behind a move, `stop()` is honoured or declared
   refused in the profile (`[safety] stop_refused_by`), busy and readbacks
   are truthful (FM-19, FM-188). The contract suite runs over the devices
   on the fake API, as it does over the demo devices.
7. **Viventis LS1: option A′, inventory first** — unchanged. The first M6
   deliverable is an on-site inventory of **both** LS1s with `smc discover`
   (#29) — controllers, cameras, lasers, scan mirrors, filter changers,
   piezo, timing hardware, as seen by the OS and by the vendor's
   configuration files — mapped to Micro-Manager adapters or unicore
   devices, with the timing controller identified. A short follow-up ADR
   then fixes the scope (full control, or stage+camera first). PyMCS is a
   reference for *how the vendor sequences its hardware*, never a
   dependency.
8. **Predecessor tools become plugins.** The tracking tool → the
   closed-loop tracking plugin; `viventis_control`'s ablation patterns → a
   plugin candidate once the LS1's components are reachable. No
   vendor-specific application survives the migration.

## Alternatives considered

- **B** for the whole Zeiss: lost because M3 put the guards inside the
  `MM*` classes — a second implementation of nine capabilities is a second
  place for every safety rule to drift.
- **C**: months of C++/.NET work for no gain over A.
- **S2** (MTB in a child process): it frees the stand for ZEN for certain,
  but costs an IPC hop on every call and a long-lived device subprocess on
  Windows, of which the repository has no precedent; kept as the fallback
  if #27 shows ZEN cannot coexist with our login and the owner needs both.
- **F2** (upstream autofocus first): puts M4 behind an external release
  cycle for a device that one local class covers.
- **`UniMMCore` for every stand**: one code path, but the Nikon stands
  would carry experimental code they do not need.
- **Pinning an unreleased pymmcore-plus commit**: unblocks the stand
  sooner, but installs unreleased code on a microscope PC.
- **B′** (PyMCS), and keeping the LS1 on the vendor's loop with "position
  nudge" support only — rejected because it perpetuates the "different
  software per microscope" problem that ADR-0006 exists to end.

## Consequences (if accepted)

- `smc.hardware.backends.zeiss_mtb` holds the process-wide MTB session and
  the Python devices; `core.py`'s opener picks `UniMMCore` on `#py` lines.
  `docs/architecture.md`'s core box stops naming PyMCS.
- A design document for M4 (`docs/design/m4-zeiss.md`) fixes the interfaces
  before the plans: the session owner, the device classes, the generalised
  interlock, `stabilize_now`, the fake MTB API. #25, #26 and #27 are
  planned against it.
- The `Autofocus` protocol changes (one method added) and the interlock is
  generalised: a change to the layer every stand uses, tested on the demo
  devices and on `FakeCore` before any Zeiss code depends on it.
- #27 is blocked on a pymmcore-plus release carrying #629 and #635, and
  measures whether ZEN can take the stand while `smc ui` holds the login.
- The NSSM service question (`m5 §9`: does a vendor SDK open in session 0)
  extends to MTB and is answered at #27.
- The LS1 carries an explicit unknown until its inventory session; no LS1
  feature is scheduled before that session has answered the timing
  question.
- The tracking tool's algorithms are ported as plugin code; its backends
  and its Panel/Bokeh apps are not ported.
