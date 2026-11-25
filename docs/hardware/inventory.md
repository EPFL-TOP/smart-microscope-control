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
  added (mmCoreAndDevices #44); 2.00 works. **2.90.108 works**: the hub and
  all 16 peripherals, nosepiece included, initialise through pymmcore
  (session 0, #126).
- The hub lists **16 peripherals**: `ZDrive`, `XYStage`, `Nosepiece`,
  `CondenserTurret`, `FilterTurret1` + `Turret1Shutter`, `FilterTurret2` +
  `Turret2Shutter`, `LightPath` (EYE/R100/AUX/L100), `PFS`, `PFSOffset`,
  `IntermediateMagnification`, `DiaLamp`, `TIRF1`–`TIRF3`. Each loads
  without being initialised, which is enough to read its type. Every
  description is empty: roles come from names and types only.
- **Two filter turrets**, each with its own shutter, so "the" filter
  turret is a profile choice. The `IntermediateMagnification` (1.0×/1.5×)
  is read-only, a manual lever, and it changes the pixel size.
- Four devices are typed `XYStage`: the stage and **three TIRF illuminator
  positioners**. Driving a TIRF drive reads exactly like a dead stage
  (0,0, never moves). Exclude by name.
- The PFS offset is a `Stage`-typed device; the Z drive is another. Type-
  first resolution with name tie-breaks is mandatory.
- **Neither `ZDrive` nor `PFSOffset` reports a focus direction** (0,
  unknown): the profile must say which way is away from the sample (#98).
  Seen from the side in session 3: **Z decreasing moves the nosepiece
  down, away from the sample**; under lock, the knob turned clockwise
  lowers Z (#208).
- PFS state is in properties: `FocusMaintenance` (On/Off), `PFS in Range`
  and `PFS Status` (read-only), `DichroicMirrorInserted`, `LEDIntensity`.
  Speeds are properties as well: `ZDrive` 0.5–2.5 mm/s, `XYStage`
  0.25–51 mm/s.
- **No axis accepts `stop()`**: `XYStage`, `ZDrive` and `PFSOffset`
  refuse it (`Unsupported device command (11)`), so a sent move runs to
  its target. Switching PFS off stops a Z move that PFS drives (session 1,
  #208; #204).
- **Engaging PFS drives Z to the stored offset's plane**, hundreds of µm
  if the offset was left by another sample, with `locked` already True
  and `ZDrive` never busy (#205). `PFS in Range` means a reflection is
  seen, PFS on or off. A Z move with PFS locked switches PFS off (#1815,
  confirmed). `PFSOffset` holds whole units, has no property limits, and
  moves Z only while PFS is on; about 6–11 units per µm at 10× (6.34
  from the floor) and about 115 at 40× (session 3, #405). It reads back
  0–20000 as written and clamps 30000 to 24370. The
  stand's **PFS button ignores smc's soft limits**: it jumped +206.6 µm
  on a new plate (session 2, #334; #331). The first lock from the floor
  works at the 10× in a well with water (6.3 units/µm, Z about 1240 µm
  away from the sample), and did **not** lock at the 40× in water:
  `in range: no` for 5 s (session 5, #550; #539). It worked again at the
  10× in session 6 (6.31 units/µm, 436 of 436 fields of a 10× scan
  locked). **At the 40× the Ti2-E reads PFSOffset back the write + 1 at 0
  and at 1** (0 → 1, 1 → 2), and as written from 1000 to 17794; at the 10×
  the same day 0 read back 0. `in range` was yes at 17794, the stand's own
  offset, and no at 14000 down to 1000, so no first lock from the floor
  can be tried at the 40× (session 6, #620; #621).
- **The stand keeps a PFS offset per objective and restores it on a turn**
  (inferred from session 3's reads): 9370 for the 10×, also at power-on,
  and 17794 for the 40×; the 4× leaves it as it was. An offset written
  before a turn is gone after it (#407). Session 5 saw it on all 21 turns,
  whatever offset was locked before (#550); session 6 saw 17794 and 9370
  again, after offsets of 7987 and 2.
- **XY and Z moves return only once they have ended**, and neither reads
  busy: 10 mm of XY in about 0.6 s, 50 µm of Z in 0.2–0.35 s (#221). A read
  takes under 0.05 ms (a cache in the adapter, inferred).
- **A nosepiece turn** returns after 1.0–4.4 s, more from higher up, and
  never reads busy; the stand switches PFS off by itself and moves Z by
  itself: down to a 500 µm escape height and back **inside the call**
  (the 500 µm read at the return is stale), then +8.7 µm with the 4× and
  −47.5 µm with the 40× against the 10× (session 3, #405; −47.32 to
  −47.52 µm over 10 turns in session 5; −47.58 µm and +47.14 µm back in
  session 6). The 40×'s focus lies about 60 µm under the 10×'s, so a turn
  from the 10×'s focus lands the 40× about
  12 µm short of its focus (session 5, #550). A turret left
  between positions fails `getStateLabel` ("Cannot get current position
  label"), and seating it from the stand finishes the restore: Z rose
  485 → 3657 µm by itself (#406). `setConfig` of the lab's `BF` preset takes 7.1 s.
  `DiaLamp`'s `Intensity` (0–2100) reads its previous value for a moment
  after a set (session 2, #330).
- `IntermediateMagnification`'s `Magnification` reads `1.0x`/`1.5x`, and
  `getMagnificationFactor()` 1.0/1.5 (session 2).
- The stand adapter never provides the camera. On our Ti2-E it is a
  **Photometrics Prime 95B** through `PVCAM` (`Camera-1`, 1412×1412, 16-bit;
  PVCAM despeckle filters on by default). nikon-control's notes named a
  Hamamatsu, which is not fitted (session 0). An Andor Zyla is present but
  unused. The epi light, a **Lumencor Spectra III**, is not on the hub; nor
  is the Mad City Labs Nano-Z100 piezo insert that was bought with the
  stand.
- Our Ti2-E has no TIRF illuminator (none was bought; **inferred** from the
  purchase quote), yet the hub lists `TIRF1`–`TIRF3` at (0, 0): it lists
  them whether or not anything is behind them.
- `DiaLamp` is the brightfield shutter/lamp; `Turret1Shutter`/
  `Turret2Shutter` are epi shutters. Intensity is a device *property*, not
  an MMCore API — found by property-name hints on light-like devices.
- Camera pixels: 11 µm is the Prime 95B's datasheet figure, not a
  measurement, and the lab's configuration defines no pixel size, so
  smc reports it as unknown. The old "6.5 µm, 2048²" figures were a
  Hamamatsu's. Measured by `camera-calibration` in session 3 (lever 1.0×):
  **2.73909 µm at 4×, 1.12447 µm at 10×, 0.276136 µm at 40×**, and the
  camera sits **2.56° off the stage's axes** (rotation 177.44° at all
  three). Session 5: 1.12494 µm at 10×, 0.275748 µm at 40× (#550).
  **Session 6: 1.12432 µm at 10×, 0.277555 µm at 40×, and the rotation
  reads 178.33° and 178.32°**: the camera turned by about +0.9° since
  session 5, the turret unchanged (**inferred:** the camera was turned on
  its port, after work on the microscope; a calibration older than
  2026-11-24 is suspect for rotation; #620). The Prime 95B is on the left
  port (`LightPath` `L100`); a frame with the light path elsewhere is dark
  (#408). Cells at the 40× in a 24-well plate fill about 2 % of the
  camera's range at 10 ms (session 6).
- A load fails now and then (`Ti2-E__0` (-10), or PVCAM
  `PL_ERR_DDI_DEVICE_OPEN_FAILED`) when commands follow each other, and the
  next command loads; nothing is sent (session 3, #411). Session 5 saw
  two `Ti2-E__0` (-10) in a row at the page's *Connect*, right after a
  console command; the third loaded. Session 6 saw six in the day, at
  console commands and once at the *Connect*; the next command loaded each
  time (#620).
- **The stage frame is turned 180° from useq's for a plate with A1 at the
  back left**: the back of the stand is −Y and the operator's right −X
  (session 5, #540); a drive to B2 landed under B2 in session 6.
- **The condenser module follows the objective**: `2-Ph1` for the 10×,
  `7-N2` (dry DIC) for the 40× (session 6, #628). No role takes a condenser
  turret, so smc cannot switch it.
- **PVCAM and the hub load from an SSH session** (OpenSSH Server, key
  login): `smc doctor` exits 0 there; prefix it with `set PYTHONUTF8=1&&`
  (session 5, #550). It needs `SMC_PROFILE` set, or `-p` after the command
  (session 6).
- The lab's `MMConfig.cfg` (nikon-control, 2026-09-16) has no Startup
  preset and no `Core,TimeoutMs`. Its state labels are its own, and
  FilterTurret2's are truncated fragments, unusable as channel names.
  Details: `docs/hardware/nikon-ti2.md`.

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
- smc profile: [`profiles/zeiss-observer7.toml`](../../profiles/zeiss-observer7.toml)
  and its `.cfg`, the MTB components as unicore Python devices
  (`smc.hardware.backends.zeiss_mtb`, design [m4](../design/m4-zeiss.md));
  built and tested on `FakeMTB` only, never loaded at the stand (#27).

## Viventis LS1 (×2)

- Vendor control software with a scripting API, **PyMCS** (`Microscope`,
  `StageXYZ`, `Camera`, `TimeLapseController`, `AcquisitionController`;
  installed at `C:\Viventis\PyMCS\v2.0.0.2`). The tracking tool and
  `viventis_control` run *inside* it. **Not our route** (ADR-0007): the
  LS1s are to be driven through their components, like every other stand.
- **Components: unknown until inventoried on site**, for both stands
  (they may differ). Expected classes: XYZ stage controller, one or two
  sCMOS cameras, laser combiner, scan mirrors (galvo), filter changers,
  piezo, and a DAQ/FPGA doing sweep/exposure/laser timing. The timing
  controller is the feasibility question.
- Inventory sources: Windows Device Manager / `Get-PnpDevice`, COM ports
  with VID/PID, PCI cards (the tracking tool's
  `tools/audit_windows_hardware.ps1` is a starting point), the vendor's
  configuration files under `C:\Viventis\`, and the acquisition metadata
  it writes.
- Frame layout written by the vendor software: `t{NNNN}_{channel}.tif` per
  position; one timepoint ≈ 1 GB; OME companion files present a run as a
  5-D series without rewriting pixels.
