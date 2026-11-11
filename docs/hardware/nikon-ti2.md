# Nikon Ti2-E — PC survey, sessions 0, 1 and 2

Four visits are curated here. The **PC survey** (2026-09-25, below down to
*Surprises*) read the PC without initialising anything. **Session 0**
(2026-10-07, state re-read 2026-10-14; [its section](#session-0--the-stand-through-pymmcore))
initialised the stand through pymmcore and moved nothing. Session 0
answers most of the survey's surprises; each one says so. **Session 1**
(2026-10-21 and 2026-10-26, #208; [its section](#session-1--first-moves-contracts-pfs))
moved the stand through the layer, ran the contracts and read PFS.
**Session 2** (2026-11-03, #334; [its section](#session-2--a-24-well-plate-at-10-the-reads-the-page))
put a 24-well plate under the 10× and used the plate page for the first
time.

Curated by the design session from the raw survey in `local/surveys/nikon-ti2/` (private, never committed)
(issue #16), collected 2026-09-25 with the hardware connected and the
vendor software closed. `smc discover` was run without `--out`, so there is
no `inventory.json`; the four console captures (`smc-discover`,
`smc-doctor`, `pytest`, `pre-commit`) are all there is. Facts below come
only from those four files and from `docs/hardware/inventory.md`, marked
where used. The host name and the account name in file paths are redacted
throughout; nothing else in the raw files needed it.

## System

| | |
|---|---|
| OS | Windows-11-10.0.26200-SP0 (build 10.0.26200) |
| Python | 3.14.7 |
| smc | 0.0.1 |
| pymmcore-plus | 0.18.1 |
| Micro-Manager | 2.0.3, nightly build 20260914 |
| Device API | version 75, Module API version 10 |
| MMCore | 12.5.0 (from `smc doctor`) |
| Adapters installed | 265 |
| Other MM install present | `Micro-Manager-75.20260319` (an older nightly; **inferred** same device-interface generation from the name only — not stated by either tool) |

265 installed adapters matches "a full Micro-Manager nightly with every
device adapter" from `mmcore install` (`docs/hardware/microscope-pc-setup.md`
§1). The install path itself contains the Windows account name and is
redacted; in the shape `docs/hardware/microscope-pc-setup.md` §1 already
uses, it is `%LOCALAPPDATA%\pymmcore-plus\pymmcore-plus\mm\Micro-Manager_2.0.3_20260914`.

## Micro-Manager adapters and devices

`smc discover` lists the device set for "the adapters this lab is likely to
use" (§2 of the setup doc) without initialising anything. That means most
of the rows below are the adapter DLL's **static declared device list**,
not proof that matching hardware is attached — the same caveat
`docs/hardware/inventory.md` already gives for the Ti-E ("eighteen devices
appear with no microscope attached, so a device list proves nothing — only
a probe does"). This survey did not run `--probe-adapter` (correctly — it's
disallowed on a first visit).

Devices that plausibly belong to this stand:

| adapter | device label | type | description |
|---|---|---|---|
| NikonTi2 | Ti2-E__0 | Hub | Nikon Ti2 microscope |
| NikonTi2 | Ti2-Simulator | Hub | Nikon Ti2 simulator (SDK required) |
| HamamatsuHam | HamamatsuHam_DCAM | Camera | Universal adapter using Hamamatsu DCAM interface |

The Hamamatsu camera confirms `docs/hardware/inventory.md`'s note that "the
stand adapter never provides the camera." The `NikonTi2` row is only its two
top-level hub entries; `smc discover` does not initialise the hub (§2), so
this does **not** confirm the Nikon SDK/`Ti2_Mic_Driver.dll` is reachable —
see Surprises #4.

The sibling `NikonTI` (Ti-E) adapter is also installed (full nightly) and
listed exactly 18 fixed devices with no microscope attached, matching
`docs/hardware/inventory.md`'s Ti-E note precisely — expected, not evidence
of a second stand on this PC.

Also queried, with no serial/USB/PCI evidence tying them to this PC
specifically (the lab-wide "likely to use" set, §2): `ASIStage`, `ASITiger`,
`Marzhauser`, `MarzhauserLStep`, `Prior`, `SutterLambda`, `TriggerScopeMM`,
`Omicron`, `Cobolt`, `CoherentOBIS`, `PI_GCS_2`, `ThorlabsFilterWheel`,
`Arduino`, `ZeissCAN29`, `DemoCamera`. Two exceptions with real hardware
evidence — `PVCAM` and `NIDAQ` — are in Surprises #2.

`AndorSDK3` failed to load ("Failed to load device adapter … from
…\mmgr_dal_AndorSDK3.dll"); nothing in the survey or in
`docs/hardware/inventory.md` suggests Andor hardware on this stand, so this
reads as an unrelated full-nightly artifact, not a Ti2 problem.

## Serial ports

One serial port: **COM3**, FTDI USB-serial bridge, VID:PID `0403:6001`.
`smc discover`'s own hint says this bridge could carry Marzhauser, ASI
(Stage/Tiger), Sutter (Lambda), Prior, CoherentOBIS, Cobolt or Omicron
protocol — "the label on the device tells which." No photo was taken this
visit to settle it (Surprises #3). The port's serial number is redacted.

## USB / PnP devices that matter

154 USB/PnP devices total; almost all are standard PC peripherals. Two
matter:

- **Nikon USB Microscope**, VID:PID `04B0:7836`, manufacturer Nikon
  Corporation — hinted to `NikonTi2`/`NikonTI`; this is the Ti2 stand's own
  USB link.
- **Sentinel/Gemalto dongles** ("Sentinel HASP Key", "Sentinel USB Key"
  VID:PID `0529:0001`, "Sentinel HL Key") — vendor-software licence dongles,
  hinted as "no adapter." Almost certainly NIS-Elements'; no key material
  was printed, only device names.

## PCI devices that matter

111 PCI devices total; the rest is the workstation's own chipset, CPU,
storage and networking (an HP Z4 G5 workstation, Intel Xeon w3-2435, NVIDIA
RTX 4000 Ada, Samsung/Seagate storage, Intel/Marvell networking —
**inferred** from device names only, not confirmed by notes or photos, and
not microscope-relevant). Two entries matter and are unexpected for this
stand — see Surprises #2:

- **National Instruments PCIe-6323**, VID:PID `1093:C4C4`, hinted to
  `NIDAQ`.
- **Teledyne Photometrics PVCam PCIe camera**, VID:PID `1B6B:0001`, hinted
  to `PVCAM`.

## What `smc doctor` says

`smc doctor` confirms the same install (Device API 75 / Module API 10,
MMCore 12.5.0, 265 adapters, demo devices "yes"). No `profiles/nikon-ti2.toml`
existed at survey time, so `smc doctor` validated only the **demo
configuration**: core roles camera=Camera, xy stage=XY, focus=Z,
autofocus=Autofocus, shutter=White Light Shutter, xy position (-0.0, -0.0)
µm, ending "✓ demo configuration loads and answers." None of this exercised
the real Ti2 hardware, the `NikonTi2` adapter, PFS, or the Hamamatsu camera
— see Surprises #5.

## pytest

245 collected, **242 passed, 3 skipped, 23.14 s** — Python 3.14.7,
pytest-9.1.1, on this Windows machine with the full 265-adapter install.
Skips: an `--all-adapters` test (would load vendor adapters on this
install), one `NotificationTester` case ("did not crash the child on this
install"), and the macOS-only crash-guard test (skipped by platform, as
expected on Windows). No failures. The default suite kept off vendor
adapters and hardware, per `docs/design/failure-modes.md` FM-40.

## pre-commit

`pre-commit installed at .git\hooks\pre-commit` — installed; no run output
was captured this visit.

## Surprises / open questions

1. **Python 3.14.7 was used**, not the Python 3.11 that
   `docs/hardware/microscope-pc-setup.md` §1 asks for ("Python 3.11 for all
   users"). The full suite still passed. Open question: relax/update the
   setup doc, or re-provision the PC with 3.11? (**inferred**: likely a
   pre-existing system Python or a newer "for all users" install; the
   survey does not say which.)
   *Settled (owner, 2026-10-14)*: Python 3.14 stays; the setup guide
   already says 3.14.
2. **A Photometrics PVCAM PCIe camera and an NI PCIe-6323 DAQ card are
   physically present** on this Ti2 PC (real PCI evidence, not just a
   static adapter list), but `docs/hardware/inventory.md`'s Ti2-E entry
   names only the Hamamatsu camera; PVCAM/Photometrics is attributed there
   to the Zeiss stand, and a PCIe DAQ to the Viventis LS1 as "a candidate
   light-sheet timing controller." Is this PC shared or repurposed, is
   `docs/hardware/inventory.md` missing something about the Ti2, or is
   another instrument colocated here? Nothing in this survey (no photos, no
   notes.txt) can settle it.
   *Session 0*: the PVCAM card carries **the camera this stand uses**, a
   Photometrics Prime 95B (`Camera-1` in the lab's configuration). No
   Hamamatsu is fitted. The DAQ's wiring is still unknown (part P of #16).
3. §2 of `docs/hardware/microscope-pc-setup.md` asks every visit for
   `photos/` and `notes.txt`, and, for Nikon PCs with `nikon-control`
   installed, `nikon-control-adapters.txt` / `nikon-control-stand.txt`.
   None of these are in `local/surveys/nikon-ti2/` — only `smc-discover`,
   `smc-doctor`, `pytest`, `pre-commit`. Without them, what is actually
   connected to COM3, and whether `nikon-control` is even installed on this
   PC, is unknown.
   *Session 0*: `notes.txt` came back. `nikon-control-scope` is not on the
   PC's `PATH`, but a `nikon-control` checkout is there, with the
   `MMConfig.cfg` it generated on 2026-09-16. COM3 is an FTDI FT232R; what
   it is cabled to is still unknown (part P). Photos: none yet.
4. Whether **`Ti2_Mic_Driver.dll` was copied** into the Micro-Manager
   adapter folder (`docs/hardware/microscope-pc-setup.md` §1, "Nikon Ti2
   only") cannot be confirmed from this survey: `smc discover` does not
   initialise the `NikonTi2` hub, so the two entries it lists (`Ti2-E__0`,
   `Ti2-Simulator`) are the adapter's static declared device types, not
   proof the SDK was reached. A future visit needs `--probe-adapter` (vendor
   software closed, stand powered before the controller box, per
   `docs/hardware/inventory.md`) to settle it.
   *Session 0*: settled. `Ti2_Mic_Driver.dll` 2.1.0.159 is beside the
   adapter, with the same SHA-256 as the copy in `Ti2-SDK\bin`, and the hub
   initialises.
5. `smc doctor` only validated the demo configuration this visit (no
   profile/`.cfg` existed yet), so none of the Ti2-specific facts already in
   `docs/hardware/inventory.md` (four `XYStage` devices, PFS as a `Stage`
   device, `DiaLamp`/`Turret1Shutter`/`Turret2Shutter`) were re-confirmed —
   they still rest on the earlier `nikon-control`-derived institutional
   knowledge, not on this survey.
   *Session 0*: all re-confirmed on the stand (below).
6. A second Micro-Manager install, `Micro-Manager-75.20260319`, sits
   alongside the fresh `Micro-Manager_2.0.3_20260914` nightly. Both look
   like device-interface 75 by name (**inferred**, not confirmed by either
   tool), so no DIV mismatch is apparent, but why a second install exists is
   not stated.
   *Session 0*: `mmcore list` gives device interface 75 for both and marks
   `Micro-Manager-75.20260319` "incompatible". *Owner, 2026-10-14*: it was
   installed for nikon-control's standalone control of the stand; smc
   ignores it.

## Session 0 — the stand through pymmcore

Run on 2026-10-07 from a remote connection, with nobody at the stand,
following the plan on #16. Nothing was commanded: the hub and its
peripherals were initialised and read, and released. NIS-Elements 6.20.02
and Ti2 Control 2.90.108 were closed, and the stand was powered before the
controller. Commit `f92bf85` (smc 0.0.1, pymmcore-plus 0.18.1, Python
3.14.7). On 2026-10-14, a second remote read added a property dump of every
device through the lab's configuration, the Windows driver store, and the
corrected property check of that configuration. Report: #126. The account
name, host name and serial numbers are redacted.

### The hub's peripherals

The `NikonTi2` hub `Ti2-E__0` initialises, and `getInstalledDevices` lists
16 peripherals. Each one loads without being initialised, which is enough
to read its type. All of them then initialise with the hub through the
lab's configuration. **Ti2 Control 2.90.108 does not crash pymmcore**,
nosepiece included (2.10/2.20 do, per `inventory.md`).

| label | type | what it holds (read 2026-10-14) |
|---|---|---|
| `ZDrive` | Stage | `Speed` 0.50–2.50 mm/s (2.50 set), `Tolerance` 0.000–0.160 µm or `Open` (0.000 set); **focus direction 0, unknown** |
| `XYStage` | XYStage | `Speed` 0.25–51 mm/s (25 set), `Tolerance` 0–0.80 µm or `Open`, `Invert-X/Y` No |
| `Nosepiece` | State | 6 positions |
| `CondenserTurret` | State | 7 positions: `1-OPEN`, `2-Ph1`, `6-Shutter`, `7-N2`; 3–5 unlabelled |
| `FilterTurret1` | State | 6 positions, all unlabelled |
| `Turret1Shutter` | Shutter | |
| `FilterTurret2` | State | 6 positions, labels below |
| `Turret2Shutter` | Shutter | |
| `LightPath` | State | `1-EYE`, `2-R100`, `3-AUX`, `4-L100` |
| `PFS` | AutoFocus | see *PFS* |
| `PFSOffset` | Stage | position in device units (7803.0 read), focus direction 0 |
| `IntermediateMagnification` | Magnifier | `1.0x` or `1.5x`, **read-only**: a manual lever the stand reports |
| `DiaLamp` | Shutter | `Intensity`, a free number (501 read), and `State` |
| `TIRF1`–`TIRF3` | XYStage | (0, 0); `XPosition`/`YPosition` 0 |

- Every peripheral's installed description is `N/A`, and its description
  is empty once loaded. Role resolution has names and types only.
- `Ti2-Simulator` fails with `(Error message unavailable) (-10)` when the
  real stand is connected.
- `smc discover --probe-adapter NikonTi2` initialises `Ti2-E__0` and
  reports no peripherals and no note: that is #48. The
  load-without-initialising loop above is the evidence that #48's design
  works on this stand.

### The lab's configuration

`MMConfig.cfg`, generated by nikon-control on 2026-09-16, in a nikon-control
checkout under the account's `software\` folder. `nikon-control-scope` itself
is not on `PATH`.

- It loads the hub, the same 16 peripherals and `Camera-1` (`PVCAM`). Core
  roles: camera `Camera-1`, XY `XYStage`, focus `ZDrive`, autofocus `PFS`,
  shutter `DiaLamp`, auto-shutter on. `smc doctor --config` loads it and
  every core role answers.
- **No `System`/`Startup` preset, and no device property set during
  loading**: only `Core` properties. The checks are in the
  hardware-session skill.
- **No pixel-size configuration**, so smc reports the pixel size as 0,
  unknown, whatever the lever reads. A size the profile gives later is divided
  by the core's magnification factor, which includes
  `IntermediateMagnification` (#167). **No `Core,TimeoutMs`** either: MMCore's
  5 s default applies unless the profile sets `device_timeout_ms`.
- One config group, `Channel`, with one preset, `BF`: `CondenserTurret`
  `7-N2`, `FilterTurret1` `1-`, `Turret1Shutter` 0, `FilterTurret2`
  `1-ANALY`, `Turret2Shutter` 0, `LightPath` `4-L100`, `DiaLamp` state 0 and
  intensity 210, exposure 10 ms.
- **The state labels are the configuration's** (`Label,…` lines), written by
  nikon-control from what the stand reported on 2026-09-16. The adapter's
  own labels were not read. Nosepiece: `1-Plan Fluor 4x`,
  `2-Plan Fluor 10x`, `3-Plan Apo LmbdD0.25 40x`, `4-` to `6-` empty. NA and
  immersion are not known yet (part P6). FilterTurret2: `1-ANALY`,
  `2-MXR00700 -96372 - GF`, `3-96372 - GF96373 - YF`,
  `4-96373 - YF96375 - mC`, `5-96375 - mCMXR00712 -`, `6-MXR00712 -`. Each
  of these is two 10-character fragments, and the second fragment of one
  position is the first of the next. They look like truncated part names
  (GFP, YFP and mCherry cubes, **inferred**), so they cannot be used as
  channel names. Whether the adapter or nikon-control cut them is unknown.
- MMCore offers `PFSOffset` as a `Core,Focus` device too, and `TIRF1`–`3` as
  `Core,XYStage` devices.
- A comment in the file records that `HamamatsuHam_DCAM` did not connect
  (`Native module failed to load (6)`), and the smc probe says the same.
  **No Hamamatsu is fitted.**

### PFS

Read on 2026-10-14 with continuous focus off: `FocusMaintenance` `Off`
(`Off`/`On`, writable), `PFS in Range` `Out of Range` (read-only),
`PFS Status` `0000000000000000` (read-only, 16 characters),
`DichroicMirrorInserted` `Yes` (writable), `LEDIntensity` 40. Which of
these reads *in range* and *locked* while PFS works is unknown until a
session with PFS on.

### Camera: Photometrics Prime 95B

- `Camera-1`, PVCAM 3.10.2, adapter 1.4.5, firmware 19.35, chip `GS400BSI`,
  **1412×1412**, 16-bit (`Mono16`), port `CMOS`, readout `100MHz 16bit` (or
  `200MHz 12bit`), gain `1-HDR` only, binning 1×1 or 2×2, cooled to −15 °C.
- It is the **22 mm variant** of the Prime 95B (purchase quote: 1412×1412,
  11 µm pixels, PCIe or USB 3.0). The 11 µm figure comes from the
  datasheet; it has not been measured.
- At a 10 ms exposure the camera reports a readout of 29.03 ms and a clear
  of 58.06 ms (`ClearMode` `Pre-Exposure`, 2 cycles). A snap costs much more
  than its exposure (#24).
- **Despeckle is on.** Post-processing filters PP1–PP4 (despeckle bright
  low/high, dark low/high) are enabled; denoising and QuantView are off.
  They change pixel values. *Owner, 2026-10-15*: despeckle is the
  camera's default and the lab keeps it; denoising and denoising.ai are
  not used. smc leaves them as the camera has them until it offers the
  same options as NIS-Elements (#134).
- Trigger `Internal Trigger`, `ExposeOutMode` `First Row`; a circular buffer
  of 20 frames; metadata off.

### The rest of the PC

- **Andor Zyla**: present, switched off during the session, and not in
  the configuration. **Not used on this stand** (owner, 2026-10-14), so smc
  ignores it and nobody needs to fix its adapter. Its interface is
  unknown: the driver store holds Andor WinUSB/libusb drivers and a BitFlow
  (CameraLink frame grabber) driver. The `AndorSDK3` adapter does not load.
  The adapter folder holds `atmcd64d.dll` (the SDK2 runtime) and no SDK3
  `at*.dll`, although "Andor Driver Pack 3" is installed in
  `Program Files`. The likely cause is that the SDK3 runtime DLLs are not
  beside the adapter (**inferred**).
- **The epi light is a Lumencor Spectra III** (owner, 2026-10-15, and the
  purchase quote). It is on neither the hub nor the configuration. Its
  **USB** and **TTL** (DB15) ports are cabled; LAN and RS-232 are not
  (photos). **COM3 (FTDI FT232R) is the Spectra's USB link**: it disappears
  from Device Manager when that cable is unplugged, and comes back when it
  is plugged in again (owner, 2026-10-16). The TTL lines come from the NI
  breakout box (below).
- **NI PCIe-6323**, with its trigger breakout box. The owner traced each
  BNC (2026-10-15):

  | NI line | connected to | carries (**inferred** from the wiring) |
  |---|---|---|
  | `AO 0` | MCL Nano-Drive `input` | the piezo's position, as a voltage, DAQ to piezo |
  | `P0.0`–`P0.7` | Spectra III TTL breakout | one line per light channel, DAQ to light |
  | `PFI 1` | Prime 95B `EXPOSE OUT 1` (green lead) | the camera's exposure signal, camera to DAQ |
  | `PFI 0` | the Andor Zyla | unknown direction (the Zyla is unused) |

  The thick cable of the Prime 95B's I/O breakout (the "multiplex") goes to
  the camera. The breakout also has a red `TRIGGER IN` lead, which goes to
  the TI2-CTRE controller. So the
  **camera's exposure clocks the DAQ**, and **the Nikon controller can
  trigger the camera** (**inferred** from the wiring; the timing itself
  is NIS-Elements' configuration).
- **The Mad City Labs piezo is installed and used for Z scans** (owner,
  2026-10-15), although no MCL device appears on USB. Its Nano-Drive
  controller's `input` takes `AO 0` from the NI breakout box, and its DB15
  output drives the stage insert. Its `sensor` output (the measured
  position) and its own USB port are not connected, so nothing reads the
  piezo's position back. The piezo is driven by the DAQ's analog output,
  not by a USB command. One Micro-Manager route is the
  `NIDAQ` adapter's analog output, wrapped as a Z stage by `Utilities`' DA Z
  Stage. That is an option for M2; it has **not been tried**.

### What was bought (purchase quotes, 2024-11 and 2025-05)

The owner provided the two purchase quotes; they stay private in `local/`.
They say what was ordered, not what is installed. Each line below is either
matched to a session 0 measurement or marked "not seen".

- **Stand**: Ti2-E with an encoded motorised Z drive (10 nm steps open
  loop, 20 nm closed loop), the TI2-CTRE controller, a PFS unit with a
  motorised nosepiece, and a motorised encoded XY stage with joystick,
  well-plate holder and stage-up kit. The light path has four positions
  (left 100 %, eyepiece 100 %, right 100 %, left/eye 80/20). The stand
  reports them as `1-EYE`, `2-R100`, `3-AUX`, `4-L100`; how `AUX` maps to
  80/20 is unknown.
- **No TIRF illuminator.** So `TIRF1`–`TIRF3`, listed by the hub at
  (0, 0), are devices with nothing behind them (**inferred**).
- **Condenser**: a motorised condenser turret (long working distance,
  NA 0.52, WD 30 mm) with Ph1 and dry DIC (N2) modules, a 40× DIC slider,
  an intelligent polarizer, and an **analyzer cube**. These match
  `CondenserTurret` `2-Ph1` and `7-N2`, and `FilterTurret2` `1-ANALY`.
- **Epi**: an EPI-FL module (25 mm field), **two motorised epi filter
  turrets**, the Spectra III, a 2 m liquid light guide, and a DB15-to-9-BNC
  trigger cable. The Spectra III has 8 channels, with filters 390/22,
  440/20, 475/28, 510/25, 555/28, 575/25, 635/22 and 740/20.
- **Cubes** (Semrock, Ti2 format, zero pixel shift): DAPI-1160B, GFP-4050B
  (466/40 → 525/50), YFP-2427B (500/24 → 542/27), mCherry-C
  (562/40 → 641/75) and Cy5-4040C (628/40 → 692/40). FilterTurret2's
  truncated labels start with `MXR00700` (DAPI), `96372 - GF`,
  `96373 - YF`, `96375 - mC` and `MXR00712` (Cy5). FilterTurret2 therefore
  probably holds, from 1 to 6: analyzer, DAPI, GFP, YFP, mCherry, Cy5. Six
  items fill one turret, so FilterTurret1 is probably empty (both
  **inferred**; part P6 confirms).
- **Dia**: an LED lamp house (`DiaLamp`), a dia filter slider and a
  rectangular field-stop slider.
- **Objectives**: CFI Plan Fluor DL 4XF (NA 0.13, WD 16.4 mm, PhL),
  CFI Plan Fluor DL 10XF (NA 0.30, WD 15.2 mm, Ph1), and CFI Plan
  Apochromat Lambda D 40X (the quote gives no NA). They match nosepiece
  positions 1–3. Nothing was bought for positions 4–6.
- **Piezo**: a **Mad City Labs Nano-Z100-N2** sample Z insert (100 µm
  travel, closed loop, Nano-Drive controller), with well-plate and
  universal holders. Installed and used for Z scans (see above): a second
  focus device that M2 must plan for (#13).
- **Incubation**: an Okolab BL3 cage incubator with a heated chamber for
  the piezo stage. Not seen by session 0.
- **Software**: NIS-Elements AR, with modules for fast triggering, 6D, a
  wavelength switcher and Z drive control.

### Positions, for the record

| | 2026-10-07 | 2026-10-14 |
|---|---|---|
| XY (µm) | (4125.0, −7214.6) | (−4465.2, −16603.3) |
| Z (µm) | 1792.18 | 4199.8 |
| Nosepiece | — | `2-Plan Fluor 10x` |

The stand was used between the two reads. Neither reading is a travel
limit: the Z and XY soft limits are still unmeasured (part P2).

### The console on this PC (cp850, conhost)

Rehearsed on the demo devices.

- Redirected without `PYTHONUTF8`, `smc doctor` writes `?` for `✓` and a
  wrong byte for `µ`, and does not crash (FM-20). With `PYTHONUTF8=1`, the
  files are correct UTF-8; only `type` shows them wrong.
- Redirected tables wrap at 80 columns. The install path ends in
  `Micro-Man…`, and the `--config` table title is cut (#94, FM-37).
- In a 9 s snap, a Ctrl-C after about 2 s prints `✗ interrupted` only when
  the exposure ends. The exit code is 130 and no file is written. The
  line speaks of a move (#94).
- A QuickEdit selection freezes output until Esc. With a selection up,
  the first Ctrl-C only copies it; the second interrupts.
- A Ctrl-C during the contracts stops cleanly: no later test starts, and
  nothing needs putting back. A Ctrl-C while pytest is still starting (its
  first run under 3.14 is slow) prints pytest's own `KeyboardInterrupt`
  traceback.
- The DemoCamera ignores an exposure above 10 000 ms, and `smc snap` does
  not say so (#125).

### Profile entries, measured

For the Ti2 profile. Its form, and where its configuration file lives, are
M2's design (#13).

- `[micromanager] config`: the lab's `MMConfig.cfg`, with which every role
  fills. `device_timeout_ms = 60000`, because the file leaves MMCore at 5 s.
- Roles: the configuration's core roles plus the heuristics fill all ten.
  `light_source` wrongly offers `LightPath` (#124). `filter_turret` has two
  real candidates. It should be assigned to **`FilterTurret2`**, NIS's
  "up" turret: the only one with six cubes. `FilterTurret1`, NIS's "low"
  turret, is empty. Matching up/low to 2/1 rests on which turret has
  labels (**inferred**).
- Light path to the camera: `4-L100`. The Prime 95B is on the left port.
- Z direction: the adapter gives none, so the profile must say (#98).
- `[camera] pixel_size_um`: nothing measured yet. The nominal values,
  11 µm divided by the magnification, are 2.75 µm (4×), 1.1 µm (10×) and
  0.275 µm (40×) at 1.0×. They are assumed, not measured; camera
  calibration (#22) measures them. The 1.5× magnifier divides them by 1.5:
  the profile's sizes are for a factor of 1.0, and the layer divides them by
  the core's `getMagnificationFactor()` (#167), so the lever is accounted for
  **if** MMCore reads it. What the factor reads at 1.0× and 1.5× is not
  verified (#16); if it reads 1.0 at both, the fix is a pixel-size preset
  keyed to the lever in the stand's configuration, not code.
- `[safety]` soft limits, **set by the owner** (2026-10-15) for plate work,
  not measured end stops: `xy_soft_limits_um = [[-49000, 49000],
  [-29000, 29000]]`. Z must never go above **4350 µm**, where the 40× is
  close to the sample. The bottom of the travel reads about 5–6 µm, so a
  lower bound just above it completes `z_soft_limits_um`. Set `max_z_jog_um` too: without it, every command
  warns that the 100 µm default is assumed.

### Part P, at the stand (2026-10-15)

With the owner at the stand; smc only read.

- **P1**: `smc z get` gave Z 4256.48 µm, and `smc stage get` gave XY
  (23070.30, −8804.70) µm.
- **P2**: the owner set the limits above, from plate work rather than from
  the end stops. With the focus knob turned fully down, Z reads about
  5–6 µm: that is the bottom of the travel.
- **P3**: the Spectra III's USB and TTL ports are cabled; LAN and RS-232
  are not (photos).
- **P4**: the NI wiring is in the table above, and the piezo is described
  above (photos). The owner traced the camera's leads: `EXPOSE OUT 1` (green)
  goes to `PFI 1`; `TRIGGER IN` (red) goes to the TI2-CTRE.
- **P5**: the Prime 95B is on the left port (`4-L100`).
- **P6**: NIS-Elements shows two turrets, "up" and "low". Only "up" is
  filled: `ANALY`, then DAPI, GFP, YFP, mCherry, Cy5, as inferred from the
  labels.

Photos (private) show the Spectra III's back panel, the Nano-Drive with
the NI breakout box, the piezo insert, the TI2-CTRE, and the
`TRIGGER IN` lead.

Nothing is left open from session 0. COM3 was confirmed as the Spectra's
USB link on 2026-10-16.

## Session 1 — first moves, contracts, PFS

2026-10-21 (no sample, 10×, PFS off) and 2026-10-26 (a glass-bottom dish
with cells, 10×), the owner at the stand, commit `a5cfa51`. Report: #208,
with the raw excerpts. The PFS reads went through pymmcore directly (a
throwaway script, posted on #16), every 100 ms.

### Stopping

- **`stop()` is refused by every axis**: `XYStage`, `ZDrive` and
  `PFSOffset` answer `Unsupported device command (11)`, idle or moving;
  `PFS` answers `Cannot stop "PFS": not a stage`. A move that was sent
  runs to its target: a Ctrl-C during `smc stage jog 20000 0 --force`
  left the stage 20000 µm on (#204). The CLI still printed "any move under
  way was stopped" (#206).
- **Switching PFS off stops a Z move that PFS drives**, at once (one
  trial: Z stopped after +4.2 µm of about +6.7 µm). It is the only stop
  measured to work on this stand.
- The stand's own controls stop everything: the joystick, the focus knob
  and the PFS button.

### PFS (m2 §4)

| § | Read | Result |
|---|---|---|
| 4.1 | Lock | `locked` True at the first read (0.1 s) in all three engagements, and once 0.5 s **before** Z moved (below). `PFS in Range` reads `In Range` whenever a sample reflects, PFS on or off. |
| 4.2 | No reference | No sample, PFS on for 10 s: never locked, Z within 0.02 µm, `Out of Range`. |
| 4.3 | #1815 | `setPosition(ZDrive, z − 5)` with PFS locked switches PFS off. |
| 4.4 | Offset under lock | A larger offset gives a larger Z. +50 units: +6.68 µm (7.5 units/µm); −50: −4.64 µm (10.8); a manual re-set of −1443 units: −243 µm (5.9). Each round trip leaves Z higher. Z settles in about 0.3 s, then creeps by about 0.3 µm over 5 s. |
| 4.5 | Offset, PFS off | Z does not move (+10, +50). |
| 4.6 | Limits, stop | `PFSOffset` exposes no property, so no limits. `stop` is refused. Values seen: 7803, 7927, 9370. |
| 4.7 | Slot API | `getAutoFocusOffset()` equals `PFSOffset`'s position; `setAutoFocusOffset()` moves it. |
| 4.8 | Switch-off under motion | Z stops at once. |
| 4.9 | Z busy | `deviceBusy("ZDrive")` never read True, in 928 reads, during PFS-driven moves included. `PFSOffset` neither; it reads its target at once. |

- **Engaging PFS drives Z to the plane of the stored offset**, not to
  the operator's focus. With an offset of 9370 left from earlier use,
  Z went from 4269.44 to 4504.76 µm between two reads, 155 µm above the
  4350 µm limit. The stand's PFS button did the same (about 200 µm).
  Once the offset was set with the knob (PFS on, the knob turns the
  offset), engaging moved nothing. At 40× such a jump reaches the sample;
  the 40× reads were skipped for that reason (#205).
- **`PFSOffset` holds whole units**: 7927.5 reads 7927, 7925.75 lands on
  7925 (#207).
- `PFS Status` (16 characters): `0000000000000000` off with no sample,
  `0000000100000000` off with a sample, `…1000001001` / `…1000011001`
  searching, `0000001100001001` / `0000001100001010` locked. It read
  `…1001` while Z was still to move and `…1010` once it had. That a bit
  means "settled" is a hypothesis.
- Loading the configuration leaves PFS as the stand had it: on and locked
  if the button left it on. The offset is whatever the last user left.

### Contracts

Without autofocus, no sample: 40 passed, 10 skipped (the expected skips),
2 failed: `test_stop_on_an_idle_stage_is_harmless` on XY and Z, the
refused stop. Autofocus, the dish in focus: 13 passed, 1 skipped (no
`offset_per_um` then), 1 failed: the offset jog contract, on the
whole-unit offset.

### Positions and the profile

- A glass-bottom dish at 10× focuses at Z 4261–4269 µm, 80–90 µm under
  the 4350 µm limit.
- The profile now has `[autofocus]` `offset_unit`, `in_range_property`,
  `in_range_values`, `offset_per_um` for the 10× (5.9, the smallest of
  the three factors), and `[safety] max_autofocus_offset_jog = 50`
  (chosen). `lock_timeout_s` stays assumed; `autofocus_offset_soft_limits`
  stays unknown.

### Not settled by session 1

- Which way Z goes, seen from the side (#98). The reads say that a larger
  Z is closer to the sample on this inverted stand (a larger offset under
  lock gives a larger Z, and the knob fully down reads 5–6 µm); nobody
  watched the objective.
- `smc snap` on the Prime 95B (#125): the snap ran without `-p` and used
  the demo camera.
- The 40× (factor, lock), what `IntermediateMagnification` reads at 1.0×
  and 1.5×, and the turret block of m2 §8.9.

## Session 2 — a 24-well plate at 10×, the reads, the page

2026-11-03, the owner at the stand, commit `ef866eb`. Report: #334, with
the raw excerpts. The sample was a 24-well plate with cells, imaged
through its bottom with the 10× only. The reads went through pymmcore
directly (a throwaway script, every 10–50 ms). The session ended before
the 40×, the chain on the plate and NIS-Elements (#198).

### Moves and reads

- **XY and Z moves return only once they have ended**, and busy never
  reads True: `setXYPosition` 10 mm in 0.59–0.67 s, 1 mm in 0.22–0.25 s;
  `setPosition(ZDrive)` 50 µm in 0.23–0.35 s. The position reads the
  target at the first read after the call; landing within 0.1 µm (#221).
- **Reads cost nothing**: Z, XY, the offset, the lock, `PFS Status` and
  busy each take 0.00–0.05 ms (a cache in the adapter, inferred).
- **`Microscope.state()`** takes 0.4 ms idle and 133 ms median, 464 ms
  at most, while snaps run (a 10 ms exposure; a snap takes 166 ms). The
  waiting is probably a camera read behind the snap (#333).
- **The Ctrl-C drill** printed the lines of #206 and #223 (errorlevel
  130): a Ctrl-C after about 0.5 s let a 20 mm jog run to its end, and one
  at once moved nothing.
- **The camera** applied 20 ms as 20 and 12.345 ms as 12.35 (#125).

### The turret block of m2 §8.9

| Read | Result |
|---|---|
| Magnifier | `getMagnificationFactor()` reads 1.0 at the lever's 1.0× and 1.5 at 1.5×; the property is `Magnification` (`1.0x`, `1.5x`). The pixel size reads 0 (no preset). |
| `DiaLamp` `Intensity` | Limits [0, 2100]. A read 0.3 s after a set returns the new value; a read 9 ms after a set returns the **previous** one (#330). |
| `setConfig("Channel", "BF")` | Returns after **7.1 s**, no device of the preset ever busy. Every value then matches the preset, yet `getCurrentConfig` reads `""`: `Exposure` reads `10.0000` for the preset's `10` (FM-100). |
| Nosepiece 10× → 4× → 10× | Each `setState` returns after about 1.0 s, never busy. **The stand switched PFS off by itself** (it was on and searching). **Z moved by itself**: 500.0 µm at the call's return (from 561.5), 570.2 µm with the 4× (+8.7 µm), 561.4 µm back with the 10×. A different mechanical noise and a beep (#330). |

### PFS on the plate

- **The stand's button jumped +206.6 µm** (Z 4191.34 → 4397.92 µm, 48 µm
  past the 4350 µm limit), to the plane of the stored offset 9370: the
  value session 1 also started from. Nobody noticed until a later command
  refused. smc's engage refused twice, as designed: outside the soft
  limits at 4398.14 µm, and at an offset not seen locked at about 4202 µm.
  The refusal's advice, "switch PFS on with the stand's own controls", is
  the step that jumped (#331).
- **The way out**: the button again with Z already in the offset's plane
  (no jump), then the knob with PFS on until sharp. The offset went
  9370 → 8141 and Z came back to 4195.26 µm: 6.0 units/µm.
- **At an offset seen locked**, PFS off for 1 s then on locked after
  0.016 s and moved Z by 0.00 µm.
- **Offset moves under lock**: +10 units moved Z 0.00 µm; +50 moved it
  +5.22 µm (9.6 units/µm), starting 0.47 s after the write and within
  0.1 µm of its end after 1.18 s; −50 moved it −4.78 µm (10.5), starting
  after 0.15 s. The round trip left Z 0.44 µm higher (#233).
- `PFS Status` alternated `…1001` and `…1010` at every read while Z
  moved, so neither bit means "settled".
- The 24-well plate focuses at Z 4191–4195 µm at 10×, 155 µm under the
  limit.

### Contracts

Without autofocus, no sample: 42 passed, 12 skipped (the declared
refusals and the expected skips), none failed. Autofocus: the first run
started at Z 4398 and every test failed in its set-up (the stand too
close to its soft limit); after the knob, 16 passed and 1 skipped (the
fake-only whole-unit test).

### The page

- Connect, Live (5.9 frames/s at full frame), the drive and *Light off*
  worked. `↑` (stage +Y) moves the image content up, `→` (+X) right; the
  owner keeps this convention.
- **Wall sides are named in the stage frame while the operator reads them
  on the image.** A one-wall-per-axis seed put B3's centre 15.5 mm off on
  both axes; snaps of B3's four walls put it at (10267.7, −10038.1) µm,
  15599 µm wide in X and 15401 µm in Y against a nominal 15600 (#327).
- A second attempt recorded B3's right wall under B4, and failed. The
  owner found the page very hard to get to grips with (#332, #328).

### Not settled by session 2

- Which way Z goes, seen from the side (#98: "not visible"), and which way
  the knob moves Z with PFS on (#208).
- How a first lock is reached at 40× without the jump (#331), and what a
  turn does to Z under a plate (#330).
- The 40×, the chain on the plate, the NIS export (#198); the plate's brand
  (a marking on one long edge, to read next time).

## Session 3 — first locks from the floor, three calibrations, a 40× jump

2026-11-10, the owner at the stand, commit `caaa60e`. Report: #405, with
the raw excerpts; the folder stays in `local/surveys/nikon-ti2/session3/`
and `local/scans/2026-11-10/`. The sample was session 2's 24-well plate,
a **Greiner 24 Well SensoPlate, glass bottom** (items 662892 / 662896;
the maker's customer drawing, revision 9), holding debris and no cells. The 4×, 10× and 40× were used, the lever at 1.0×,
the Prime 95B on the left port (`L100`).

### Z, the offset and the turns

- **Z decreasing moves the nosepiece down, away from the sample**, seen
  from the side (#98): `z_away_from_sample = "decreasing"`.
- **Under lock, turning the knob clockwise lowers Z** (#208).
- **`PFSOffset` with PFS off** reads back every value from 0 to 20000 as
  written and clamps 30000 to 24370; no write moved Z (2999.26 →
  2999.22 µm over nine writes). Hence
  `autofocus_offset_soft_limits = [0.0, 20000.0]` (#401).
- **The stand keeps a PFS offset per objective and restores it on a
  turn** (F3/F5, inferred from these reads): 9370 for the 10× (also at
  power-on, although session 2 left 8141: the plane of the button jumps of
  sessions 1 and 2), 17794 for the 40×; a turn to the 4× leaves the
  offset as it was. A `--floor` written before a turn is gone after it
  (#407).
- **Every turn drops Z to 500 µm, whatever the start, and brings it back
  inside the call** (F4). Without a plate, PFS off, 10× → 4× → 10×:

  | Z before | 10×→4× call | 4× Z max / last | 4×→10× call | Z after | round trip |
  |---|---|---|---|---|---|
  | 561 µm | 1.00 s | 569.74 / 569.54 | 2.82 s | 560.72 | −0.28 µm |
  | 1500 µm | 1.86 s | 1508.70 / 1508.42 | 2.05 s | 1499.34 | −0.62 µm |
  | 3000 µm | 3.31 s | 3008.70 / 3008.16 | 3.39 s | 2999.22 | −0.78 µm |
  | 4195 µm | 4.30 s | 4203.70 / 4203.02 | 4.38 s | 4193.80 | −1.20 µm |

  The Z read at the return of 10× → 4× is 500 µm, and the next one, 10 ms
  later, 4203.70: the first is **stale** (FM-188), since Z cannot travel
  3.7 mm in 10 ms. Busy never read True. smc's turn line reports the
  settled Z. The stand's parfocal steps from the 10× are **+8.7 µm for
  the 4×** and **−47.5 µm for the 40×** (−47.48 there, +47.20 back).
- **A turret left between positions** gives `getStateLabel('Nosepiece')
  failed (Cannot get current position label)`; when the stand seated it,
  Z rose by itself from 485.38 to 3656.86 µm: the unfinished turn's escape
  height, then its restore (F1, F2, inferred; #406).

### First locks from the floor (m2 §12.10.7)

| Objective, well | By-eye focus | Z after `engage` from offset 0 | Z moved |
|---|---|---|---|
| 10×, B3 | 4132.38 µm | 3006.14 µm, locked | 1126 µm away |
| 40×, C4 | 4005.56 µm | 3958.68 µm, locked | 46.9 µm away |
| 40×, C3 | 4017.28 µm | guard tripped at 4074.08 µm | **+56.8 µm toward** |

- At C3 the floor's plane lay above the by-eye focus. The excursion guard
  (`max_engage_excursion_um = 50`) switched PFS off one 10 ms read past
  its bound. Why the same floor sits 47 µm below the focus at C4 and at
  least 57 µm above it at C3 is unknown (#402). Whether the 40× touched
  the plate is not known.
- **Factors**: 10×, **6.34 units/µm** (floor to 7061; sessions 1 and 2:
  5.9, 6.0); 40×, **about 115 units/µm** from four points under lock,
  (0, 3958.68), (4048, 3993.04), (5388, 4004.58), (6973, 4018.66): pairwise
  112.6–117.8. Ratio 18, near (40/10)² = 16.
- **At the 40× PFS held** over two 5×5 grids of 350 µm steps (Z within
  1.7 µm over 1.4 mm) and through a 7.5 mm move inside a well (+2.1 µm).
  An engage at an offset already seen locked moved Z by −0.04 µm (D1).
- The 40× (Plan Apo λD, working distance 0.17–0.25 mm) focused through the
  plate's glass bottom at Z 4005–4011 µm, 344 µm under the 4350 µm
  limit. The lab uses this plate at the 40×. (The bottom was called
  "standard plastic" at the stand; the drawing and the owner correct it.)

### The camera

`camera-calibration`, lever 1.0×, binning 1:

| Objective | Pixel size | Step | Residual | Rotation |
|---|---|---|---|---|
| 4× | 2.73909 µm | 900 µm | 0.894 µm | 177.437° |
| 10× | 1.12447 µm | 350 µm | 0.642 µm | 177.443° |
| 40× | 0.276136 µm | 100 µm | 0.535 µm | 177.437° |

- **The camera sits 2.56° off the stage's axes** at all three objectives
  (the "180°" of earlier notes was a rough value). Effective
  magnifications 4.02×, 9.78×, 39.8× for 11 µm pixels; the 40× field is
  389.9 µm.
- With no pixel size known, the step falls back to 50 µm: at the 4× that
  is about 18 px, and the fit gave a 10.7 µm residual (F6, #408).
- **A frame with the light path on the eyepiece** makes the calibration
  refuse with "parallel shifts: did the stage move?" (F8, #408). Live on
  the page was dark for the same reason, with no hint. The lab calls the
  left port "the Zyla"; it is the Prime 95B (1412 × 1412 px).

### The plate

- **Column +1 lies at stage −X** (#327).
- Hand registration from wall reads at the 10×: B3 (10003, −10051),
  B5 (−28982, −9919), C3 (10031, 9375) µm; **pitch 19.49 mm between
  columns, 19.43 mm between rows**; the plate sits square to 0.1–0.2°.
  B3 is 265 µm off session 2's in X: the plate was put back between them.
- **The maker's numbers** (Greiner's drawing): pitch **19.5 mm** on both
  axes, well Ø 15.66 mm at the bottom and 16.28 mm at the top, depth
  17.7 mm; A1's centre 15.13 mm and 13.49 mm from the left and top edges;
  footprint 127.76 × 85.48 mm (±0.2), height 19 mm (±0.15); flatness
  "curvature 0.2" on top, +0.3/−0.1 mm at the bottom. The measured pitch
  is 0.05 % (columns) and 0.36 % (rows) short of it.
- useq's `24-well` has a 19.0 mm spacing, so `plate-walls` refused the
  plate at +2.7 % (F9, #400). At the 10× it found the walls of all three
  wells; at the 4× it took an edge 1.6 mm inside B5's wall (F7, #409).

### Loads

Three of about ten `smc` commands in a row failed to load and the next
one loaded: `Error in device "Ti2-E__0": (Error message unavailable)
(-10)` twice, and once PVCAM's `pl_cam_open failed
(PL_ERR_DDI_DEVICE_OPEN_FAILED)` (F10, #411). A failed load sends
nothing.

### Not settled by session 3

- Why the 40× first lock from the floor went toward the sample at C3, and
  whether it touched the plate (#402).
- Which definition the plate gets: the measured 19.45 mm or the maker's
  19.5 mm (#400), and the plate's 180° on the stage (#414).
- The page's STOP during a drive and during Live (#333), the chain on a
  registered plate (scan, cells, route), the objective offsets (#380),
  the NIS export (#198); a plate with cells at the 40× (#342).
