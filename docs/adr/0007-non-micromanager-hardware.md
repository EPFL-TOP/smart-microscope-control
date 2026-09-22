# 0007 — Hardware without a Micro-Manager adapter (Zeiss MTB, Viventis LS1)

- **Status**: **proposed** — decision requested from the project owner
- **Date**: 2026-09-22

## Context

ADR-0002 makes Micro-Manager the bus. Two of the five microscopes cannot be
reached through an existing adapter:

- **Zeiss Axio Observer 7.** Micro-Manager's `ZeissCAN29` adapter speaks
  CAN29 over a serial port. This stand carries CAN29 **over USB into
  Zeiss's `CZCanSrv`** (every device in its MTB configuration says
  `PortType="USB"`), so the serial adapter finds nothing at any baud.
  `lightsheet-live-tracking-tool` drives the stand through **MTB 2011**
  (Zeiss's hardware abstraction service) via `pythonnet`: stage axes, focus,
  piezo, objective changer, reflector, sideport, condenser, TL shutter,
  Colibri LEDs and Definite Focus 2 all work and are wrapped in
  `tracking_tools/microscope_interface/mtb.py`. The camera (Prime 95B) works
  through Micro-Manager's `PVCAM` adapter. Two hard constraints: **one MTB
  login per process** (logout kills the realtime subsystem for good) and
  **one PVCAM handle per camera**.
- **Viventis LS1.** Controlled by the vendor's software with a Python API
  (**PyMCS**), not by Micro-Manager. The tracking tool runs *inside* PyMCS
  scripts, reads frames from disk and nudges positions. What PyMCS exposes
  for direct stage/camera control has not been inventoried.
- pymmcore-plus ≥ 0.13 ships `pymmcore_plus.experimental.unicore`: a
  `UniMMCore` that loads **devices written in Python** alongside C++
  adapters. Available base classes today: `XYStageDevice`, `StageDevice`,
  `StateDevice`, `ShutterDevice`, `CameraDevice`, `HubDevice`,
  `GenericDevice`, `SLMDevice` — **no `AutoFocusDevice` yet**.

## Options

- **A — Wrap the vendor API as unicore Python devices.** The Zeiss stand
  becomes `MTBXYStage(XYStageDevice)`, `MTBFocus(StageDevice)`,
  `MTBPiezo(StageDevice)`, `MTBChanger(StateDevice)` ×4,
  `MTBTLShutter(ShutterDevice)`, `MTBColibri(GenericDevice)` loaded into the
  same `UniMMCore` as the PVCAM camera. Everything above the core — roles,
  capabilities, `useq` MDA, plugins — works unchanged. DF2 is exposed as a
  `GenericDevice` with properties until unicore grows an autofocus type
  (we can contribute it upstream), and the `Autofocus` capability for Zeiss
  talks to that device.
- **B — A direct backend implementing the capability protocols** without
  Micro-Manager in the path (what `MicroscopeInterface_MTB` is today), with
  the camera still in a separate `CMMCorePlus`. Simpler per device; but two
  cores/buses per microscope, no `useq` MDA over the stage, and every
  capability implemented twice.
- **C — Write a C++ Micro-Manager adapter for MTB.** The "proper" upstream
  route; requires a C++/.NET build chain and Zeiss SDK licensing — months.

## Recommendation

**A, with B as the fallback for any device unicore cannot express.** The
whole point of ADR-0002 is one bus; unicore keeps it. Its "experimental"
label is a risk we accept knowingly and mitigate with contract tests
(ADR-0005) and by pinning pymmcore-plus.

For the **LS1**, the first step is an inventory of the PyMCS API (issue);
the decision between A, B and "read files + nudge positions only" follows
from what it exposes.

## Consequences (if accepted)

- `smc.hardware.backends.zeiss_mtb` owns the single MTB session; the
  `UniMMCore` is created by the facade when a profile declares Python
  devices.
- Contract tests for XYStage/ZStage/State run against the MTB devices in
  hardware sessions.
- We track unicore's roadmap for an autofocus device type and offer the
  DF2 wrapper upstream when stable.
