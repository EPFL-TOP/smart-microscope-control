# 0007 — Hardware without a Micro-Manager adapter, and vendor-controlled systems

- **Status**: **proposed** — revised 2026-09-22 after the owner's direction on the LS1; decision requested
- **Date**: 2026-09-22
- **Issue**: #4

## Context

ADR-0002 makes Micro-Manager the bus. Two of the five microscopes are not
reachable through an existing adapter out of the box:

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
never been inventoried.

### The tracking tool is a tool

`lightsheet-live-tracking-tool` is, first of all, a live tracking
application. In this architecture that makes it a **plugin** (ADR-0004),
not a set of microscope backends: its trackers and its closed-loop logic
survive; its five `MicroscopeInterface_*` backends are superseded by the
capability layer.

### What pymmcore-plus offers

`pymmcore_plus.experimental.unicore` provides a `UniMMCore` that loads
**devices written in Python** alongside C++ adapters. Base classes today:
`XYStageDevice`, `StageDevice`, `StateDevice`, `ShutterDevice`,
`CameraDevice`, `HubDevice`, `GenericDevice`, `SLMDevice` — **no
`AutoFocusDevice` yet**.

## Options

### For a stand whose vendor API is the only route (Zeiss)

- **A — Wrap the vendor API as unicore Python devices.** `MTBXYStage
  (XYStageDevice)`, `MTBFocus`/`MTBPiezo (StageDevice)`, changers as
  `StateDevice`, `MTBTLShutter (ShutterDevice)`, `MTBColibri
  (GenericDevice)` loaded into the same `UniMMCore` as the PVCAM camera.
  Roles, capabilities, `useq` MDA and plugins work unchanged. DF2 is a
  `GenericDevice` until unicore has an autofocus type (we can contribute
  it), and the Zeiss `Autofocus` capability talks to it.
- **B — A direct backend implementing the capability protocols**, camera
  in a separate core. Simpler per device; two buses, no `useq` MDA over the
  stage, every capability implemented twice.
- **C — A C++ Micro-Manager adapter for MTB.** The upstream route; months
  of work, a .NET/C++ build chain, licensing questions.

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

1. **Zeiss Axio Observer 7: option A**, with **B** as the fallback for any
   device unicore cannot express. One bus is the point of ADR-0002; the
   "experimental" label is a known risk mitigated by contract tests
   (ADR-0005) and by pinning pymmcore-plus.
2. **Viventis LS1: option A′, inventory first.** The first M6 deliverable is
   an on-site inventory of **both** LS1s — controllers, cameras, lasers,
   scan mirrors, filter changers, piezo, timing hardware, as seen by the
   OS (PnP devices, serial VID/PID, PCI cards) and by the vendor's
   configuration files — mapped to Micro-Manager adapters or unicore
   devices, with the timing controller identified. A short follow-up ADR
   then fixes the scope (full control, or stage+camera first). PyMCS is a
   reference for *how the vendor sequences its hardware*, never a
   dependency.
3. **Predecessor tools become plugins.** The tracking tool → the
   closed-loop tracking plugin; `viventis_control`'s ablation patterns → a
   plugin candidate once the LS1's components are reachable. No
   vendor-specific application survives the migration.

## Alternatives considered

Options B, C and B′ above. Also: keeping the LS1 on the vendor's loop and
adding "position nudge" support only — rejected because it perpetuates the
"different software per microscope" problem that ADR-0006 exists to end.

## Consequences (if accepted)

- `smc.hardware.backends.zeiss_mtb` owns the single MTB session; the
  `UniMMCore` is created by the facade when a profile declares Python
  devices.
- `smc discover` (OS-level inventory cross-referenced with adapters) moves
  up the roadmap: it is the tool for the LS1 inventory and useful on every
  stand.
- The LS1 carries an explicit unknown until its inventory session; no LS1
  feature is scheduled before that session has answered the timing
  question.
- The tracking tool's algorithms are ported as plugin code; its backends
  and its Panel/Bokeh apps are not ported.
