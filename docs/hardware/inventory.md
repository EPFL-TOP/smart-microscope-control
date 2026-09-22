# Hardware inventory — what we know about each stand

Institutional knowledge carried over from `nikon-control` and
`lightsheet-live-tracking-tool`, condensed. Each line here cost somebody
hours; when it becomes code (a profile entry, a heuristic, a test), link it.
New findings arrive as *Hardware session report* issues and land here.

## Micro-Manager demo devices (the simulator)

- Adapter `DemoCamera`: `DCam`, `DXYStage`, `DStage`, `DObjective`,
  `DAutoFocus`, `DShutter`, `DWheel`, `DHub`. Installed by
  `mmcore install --test-adapters` on any OS; the default
  `MMConfig_demo.cfg` loads with `loadSystemConfiguration()`.
- Has no autofocus *offset* device and no vendor quirks — which is why the
  PFS interlock is tested with a `FakeCore` (ADR-0005).

## Nikon Ti2-E

- Adapter **`NikonTi2`**, hub enumerated **dynamically** from Nikon's SDK:
  an empty device list means *the SDK could not be reached*, almost always
  because `Ti2_Mic_Driver.dll` (from `C:\Program Files\Nikon\Ti2-SDK\bin`)
  is not **copied beside** the adapter in the Micro-Manager folder.
- Power on the **stand before the controller box**, or only the simulator
  device appears. `Ti2Sample.exe` tests the SDK without Micro-Manager.
- Ti2 Control 2.10/2.20 crash Micro-Manager when the nosepiece device is
  added (mmCoreAndDevices #44); 2.00 works.
- Four devices are typed `XYStage`: the stage and **three TIRF illuminator
  positioners**. Driving a TIRF drive reads exactly like a dead stage
  (0,0, never moves). Exclude by name.
- The PFS offset is a `Stage`-typed device; the Z drive is another. Type-
  first resolution with name tie-breaks is mandatory.
- The stand adapter never provides the camera (Hamamatsu `HamamatsuHam`
  here).
- `DiaLamp` is the brightfield shutter/lamp; `Turret1Shutter`/
  `Turret2Shutter` are epi shutters. Intensity is a device *property*, not
  an MMCore API — found by property-name hints on light-like devices.
- Camera pixels 6.5 µm; at 40× → 0.1625 µm/px, 2048² → 333 µm field.

## Nikon Ti-E (older stand)

- Adapter **`NikonTI`**, hub `TIScope`, **fixed** device list: eighteen
  devices appear with no microscope attached, so a device list proves
  nothing — only a probe does.
- Vendor DLL `NikonTi.dll` in `C:\Program Files\Nikon\Shared\Bin`, found on
  the system path (no copy needed).
- Connect over **USB 2**; USB 3 causes connection failures. `TIDiaLamp`
  crashed Micro-Manager with some driver versions — skippable.
- PFS offset positions are 1/40 of the raw steps; valid range depends on
  the objective.
- Three devices typed `Stage`: Z drive, PFS offset, TIRF drive.

## Both Nikon stands

- **Setting Z programmatically disables PFS** (micro-manager #1815). Any
  Z move during a time-lapse must re-engage PFS or focus drifts overnight.
- With PFS engaged, focus is changed with the **PFS offset**, not Z; the
  offset unit is not a micron.
- PFS *engaged* ≠ *in range* ≠ *locked*; a lock at the wrong offset holds
  a sharp plane on the glass, not the cells.
- Only **one connection** to a stand at a time (NIS-Elements, Ti2 Control,
  a second core).
- Rotating the turret under a loaded plate can drive a dry 40× into glass:
  turret moves are confirmed events.
- MMCore's 5 s device timeout is shorter than a plate traverse; use 60 s.

## Zeiss Axio Observer 7 (EPFL LiveScope rig)

- Camera **Photometrics Prime 95B** (PCIe; 1200×1200, 16-bit, 11 µm
  pixels) through Micro-Manager **`PVCAM`** — works. One PVCAM handle per
  camera; **ZEN must be closed** (it holds the frame grabber).
- Stand devices are **not reachable by `ZeissCAN29`**: CAN29 runs over USB
  into `CZCanSrv` (MTB config: `PortType="USB" PortNo="4100"`); COM probing
  returns error 10015 at every baud.
- Working route: **MTB 2011** (`C:\Program Files\Carl Zeiss\MTB 2011 -
  2.12.0.7\MTBApi\MTBApi.dll`) via `pythonnet`. Component IDs:
  `MTBStageAxisX/Y`, `MTBFocus`, `MTBPiezoFocusCan`, `MTBFocusStabilizer2`
  (DF2), `MTBFLLEDController` (Colibri), `MTBObjectiveChanger`,
  `MTBReflectorChanger`, `MTBTLShutter`, `MTBTLHalogenLamp`,
  `MTBSideportChanger`, `MTBCondenserContrastChanger`.
- **One MTB login per process**; logout destroys the realtime subsystem.
  Components must be cast to `IMTBContinual`; `Position` is 1-based,
  `GetElement` 0-based; pythonnet 3 refuses int→Enum, use the enum member.
- Axes: X/Y ±350 mm (0.25 µm step, 1.875 µm repeatability), focus ±14 mm
  (0.01 µm), piezo 0–500 µm (0.01 µm step, ~0.055 µm deviation; z-steps
  below ~0.06 µm duplicate planes).
- **Definite Focus 2** holds a distance from the coverslip, not the sample.
  It corrects through the **focus drive**, not the piezo, on this stand
  (`GetPiezoFocusUsage` = Off) — measure *total* focus. `SignalQuality`
  reads 0 even when it works. Per-position reference blobs from
  `InitOnCurrentFocusPosition` (9 bytes) enable multi-position focus.
- Light path: sideport position **2 = left = camera**; a phase ring on the
  condenser starves brightfield. Flat frames at mean≈99/std≈2.2 mean no
  light reaches the sensor.
- Objectives: pos 2 `Fluar 10x/0.50`, pos 4 `Plan-Apochromat 40x/0.95`;
  pixel pitch 1.1 µm at 10×, 0.275 µm at 40×. LEDs 385/430/475/511/567/630.
- Phototoxicity: light must be gated on **every** exit path (end of live,
  start/stop of run, atexit, browser session end); TL field diaphragm is
  manual and lights a whole well.
- PC: HP Z840, Windows 10 LTSB 2015 (unsupported), Zeiss `MicoIf` FPGA
  card with proprietary `MicoIfDrv` — replacement risk documented in the
  tracking tool's `docs/pc_replacement.md`.

## Viventis LS1 (×2)

- Vendor control software with a Python API, **PyMCS**; scripts run inside
  it. The tracking tool's `MicroscopeInterface_LS1` reads frames from the
  acquisition folder (`t{NNNN}_{channel}.tif` per position) and applies
  relative moves through PyMCS.
- Not on PyPI; Windows PC only. **Unknown**: what PyMCS exposes for direct
  stage/camera control outside a running acquisition → first issue for M6.
- One timepoint is ~1 GB; OME companion files (never rewriting pixels) are
  the way to present a run as a 5-D series.
