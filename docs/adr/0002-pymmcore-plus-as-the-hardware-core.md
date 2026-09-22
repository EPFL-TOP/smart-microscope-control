# 0002 — Python, Micro-Manager and pymmcore-plus as the hardware core

- **Status**: accepted
- **Date**: 2026-09-22

## Context

The lab operates five microscopes from three vendors: two Nikon inverted
stands (Ti2-E, Ti-E), a Zeiss Axio Observer 7 and two Viventis LS1 light
sheets. Each ships with its own control software and API (NIS-Elements JOBS
and macros; ZEN and its gRPC gateway plus the MTB 2011 service; PyMCS). The
goal is one control layer so that a tool written once — find the edges of a
well plate, locate objects in wells, drive a time-lapse — runs on any of them.

Both predecessor projects converged on the same stack independently:

- `nikon-control` drives the Ti2/Ti-E through **Micro-Manager** device
  adapters via **pymmcore-plus**, with `useq-schema` describing plates and
  acquisitions.
- `lightsheet-live-tracking-tool` moved its Zeiss backend off ZEN (which
  cannot change positions during a running experiment) to a Micro-Manager +
  pymmcore-plus MDA loop, and uses `useq.MDAEvent` per frame.

Micro-Manager has adapters for the Nikon stands (`NikonTi2`, `NikonTI`),
most cameras (Hamamatsu, Photometrics `PVCAM`, Andor) and hundreds of other
devices. It also ships a **simulated microscope** (`DemoCamera`: camera, XY
stage, Z, turret, shutter, autofocus) that installs on any OS with one
command. pymmcore-plus adds a Pythonic API, an event system, an acquisition
engine driven by `useq`, and — experimentally — the ability to write device
adapters in pure Python (`UniMMCore`, see ADR-0007).

## Decision

- The project is written in **Python ≥ 3.10** (the microscope PCs run 3.11).
- **Micro-Manager's MMCore, through pymmcore-plus, is the hardware bus.**
  A microscope is a Micro-Manager configuration; a device is a Micro-Manager
  device. Vendor SDKs are reached either through an existing Micro-Manager
  adapter or by wrapping them as devices (ADR-0007) — never called directly
  from tools.
- **`useq-schema` is the language for acquisitions and plate geometry**
  (`MDASequence`, `MDAEvent`, `WellPlatePlan`, grid plans). We do not write
  our own acquisition engine; we produce `useq` objects and let
  `CMMCorePlus.run_mda` execute them.
- Device adapters are installed with `mmcore install` (full nightly on
  Windows / Intel macOS; demo adapters elsewhere). The **device interface
  version** (DIV) of pymmcore-plus and of any separately installed
  Micro-Manager GUI must match — `smc doctor` reports it.
- Exactly **one core per process** talks to a stand. Nikon and Zeiss stands
  accept a single connection; a second core (another dashboard, NIS, ZEN)
  fails to initialise with no hint that the cause is the first.

## Alternatives considered

- **Vendor APIs directly** (NIS JOBS Python, ZEN API, PyMCS): no shared
  device model, one implementation per vendor per tool, and two of the three
  cannot be driven while the vendor GUI owns the hardware. ZEN specifically
  cannot update positions during a run (confirmed in writing by Zeiss).
- **pycro-manager**: requires the Java MMStudio to be running and bridges to
  it; heavier, slower for tight loops, and adds a Java dependency on every
  PC. pymmcore-plus talks to MMCore in-process.
- **python-microscope (Micron)** or a home-grown device layer: far fewer
  adapters, no plate/acquisition ecosystem, and we would own every driver.
- **A Micro-Manager Java plugin**: wrong language for the lab's analysis
  code (all Python, PyTorch, scikit-image).

## Consequences

- Everything above the hardware layer can be developed and tested on a
  laptop with the demo devices (ADR-0005).
- We inherit Micro-Manager's constraints: the `.cfg` format, DIV matching,
  `waitForDevice` semantics, the 5 s default device timeout (too short for a
  plate traverse — set it to 60 s), config-group "channels".
- Hardware with no Micro-Manager adapter (the Observer 7's USB-CAN bus, the
  LS1) needs a bridging decision: ADR-0007.
- We stay compatible with the pymmcore-plus ecosystem (pymmcore-widgets,
  napari-micromanager, Faro) should we want any of it later.
