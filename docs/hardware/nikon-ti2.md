# Nikon Ti2-E — PC survey and session 0

Two visits are curated here. The **PC survey** (2026-09-25, below down to
*Surprises*) read the PC without initialising anything. **Session 0**
(2026-10-07, state re-read 2026-10-14; [its section](#session-0--the-stand-through-pymmcore))
initialised the stand through pymmcore and moved nothing. Session 0
answers most of the survey's surprises; each one says so.

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
  unknown. **No `Core,TimeoutMs`** either: MMCore's 5 s default applies unless
  the profile sets `device_timeout_ms`.
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
  (photos). COM3 (FTDI FT232R) is the PC's only serial port, so it is
  probably the Spectra's USB link (**inferred**). The TTL lines come from
  the NI breakout box (below).
- **NI PCIe-6323**, with its trigger breakout box. The owner traced each
  BNC (2026-10-15):

  | NI line | connected to | carries (**inferred** from the wiring) |
  |---|---|---|
  | `AO 0` | MCL Nano-Drive `input` | the piezo's position, as a voltage, DAQ to piezo |
  | `P0.0`–`P0.7` | Spectra III TTL breakout | one line per light channel, DAQ to light |
  | `PFI 1` | Prime 95B `EXPOSE OUT 1` (green lead) | the camera's exposure signal, camera to DAQ |
  | `PFI 0` | the Andor Zyla | unknown direction (the Zyla is unused) |

  The Prime 95B's I/O breakout cable (the "multiplex") also has a red
  `TRIGGER IN` lead, which goes to the TI2-CTRE controller. So the
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
  calibration (#22) measures them. The 1.5× magnifier divides them by 1.5,
  so a table keyed by objective alone is not enough.
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

Still open, minor: confirm that COM3 is the Spectra's USB link (it
disappears from Device Manager when that cable is unplugged).
