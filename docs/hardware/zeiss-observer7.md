# Zeiss Axio Observer 7 — microscope PC survey

Curated by the design session from the raw survey in `local/surveys/zeiss-observer7/` (private, never committed):
(`smc-discover`, `smc-doctor`, `pytest`, `pre-commit`; console text only, no
`inventory.json` — `smc discover` was run without `--out`). Collected
2026-09-24 (UTC), stand issue #27. Host name and the
Windows user name embedded in installation paths are redacted throughout;
see the private survey notes for what was removed (repo is public, FM-50).

Facts below come from the four survey files unless marked **inferred**.

## System

| | |
|---|---|
| Host | redacted — see the private survey notes |
| OS | Windows-10-10.0.10240-SP0 (build 10.0.10240) |
| Python | 3.14.7 |
| smc | 0.0.1 |
| pymmcore-plus | 0.18.1 |
| Micro-Manager (active) | `Micro-Manager_2.0.3_20260828` |
| Micro-Manager install path | `D:\Users\<user>\AppData\Local\pymmcore-plus\pymmcore-plus\mm\Micro-Manager_2.0.3_20260828` (user name redacted) |
| Device API / Module API | version 75 / version 10 |
| MMCore | 12.5.0 |
| Adapters installed | 265 (full nightly, `mmcore install` default) |
| Other Micro-Manager install found | `Micro-Manager-75.20260319` |

Build 10.0.10240 is Windows 10's original 1507/LTSB 2015 release, matching
`docs/hardware/inventory.md`'s existing note that this PC runs "Windows 10
LTSB 2015 (unsupported)".

The Micro-Manager install directory is on `D:`, not `C:`, unlike the plain
`%LOCALAPPDATA%` layout `docs/hardware/microscope-pc-setup.md` §1 describes.
**Inferred**: either this PC's user profile is redirected to a data drive,
or `mmcore install` was pointed at `D:` explicitly — worth one line in
`notes.txt` on the next visit.

## Relevant adapters and devices

`smc discover` lists devices for the 20 adapters "this lab is likely to
use" (per `docs/hardware/microscope-pc-setup.md` §2); it does not probe or
initialise anything, so this is each adapter's own declared device
catalogue, not a confirmation that a device is physically present.

- **`PVCAM`**: `Camera-1`..`Camera-4` (`Camera` type, "Universal PVCAM
  interface - camera slot N"). Backed by physical evidence: PCI device
  `PVCam PCIe Camera` (`1B6B:0001`, Photometrics and QImaging, class
  `PMPCIEWDF`, status OK), vendor-hinted to adapter `PVCAM`. Matches
  `docs/hardware/inventory.md`'s documented camera route (Photometrics
  Prime 95B via `PVCAM`).
- **`ZeissCAN29`**: `ZeissScope` (`Hub` type, "Zeiss AxioObserver
  controlled through serial interface") — the adapter's own description,
  not a live connection.
- Adapters in the same "likely to use" list that **failed to load** on
  this PC: `AndorSDK3`, `HamamatsuHam`, `NIDAQ`, `NikonTi2` (all "Failed to
  load device adapter ... .dll"). None of these correspond to hardware
  seen elsewhere in this survey (no Andor or Hamamatsu camera, no NI-DAQ
  card in the PCI list, no Nikon stand) — read as expected absence of
  unrelated vendor DLLs/dependencies on a full "every adapter" nightly
  install, not a fault in this stand's own route.
- Remaining adapters in the list (`ASIStage`, `ASITiger`, `Arduino`,
  `Cobolt`, `CoherentOBIS`, `Marzhauser`, `MarzhauserLStep`, `NikonTI`,
  `Omicron`, `PI_GCS_2`, `SutterLambda`, `ThorlabsFilterWheel`,
  `TriggerScopeMM`, `DemoCamera`) loaded their device catalogues normally;
  none has a matching serial port, USB/PnP or PCI entry elsewhere in the
  survey, so none of this hardware is present on this stand.
- The 265-adapter list includes `ZeissCAN`, `ZeissCAN29` and
  `ZeissAxioZoom`, but **no adapter for MTB** — there is none to have,
  which is itself first-hand confirmation of ADR-0007's premise (see
  below).

## Serial ports (1)

Only `COM1` ("Communications Port (COM1)"), a standard Windows port type
with no manufacturer, VID:PID, serial number or vendor hint. No ASI,
Sutter, PI, Marzhauser or Thorlabs serial device is physically connected.
Consistent with `docs/hardware/inventory.md`'s note that `ZeissCAN29`'s
"COM probing returns error 10015 at every baud" on this stand: there is no
working legacy-serial route into it.

## USB / PnP devices (105 total; instrument-relevant ones below)

- `Axio Observer Stand` — `0758:1004`, Carl Zeiss Microscopy GmbH, class
  `ZeissCanNode`, status OK. This is the CAN29-over-USB node into
  `CZCanSrv` that `docs/hardware/inventory.md` and ADR-0007 describe: the
  stand's CAN bus is wired over USB, not a physical serial port, which is
  why the empty serial-port list above is expected rather than a fault.
- Three licence dongles (Gemalto): `Sentinel USB Key`, `Sentinel HL Key`,
  `Sentinel HASP Key` — hinted by `smc discover` as "licence dongle of a
  vendor software; no adapter". **Inferred**: likely required by ZEN and/or
  MTB; not confirmed which software from this survey alone.
- The remaining ~100 entries are ordinary workstation peripherals (an HP
  Z27n display, an HP LaserJet 4200/4300 print queue listed twice, and
  standard Microsoft/ACPI/system/software devices) — not instrument
  hardware, omitted here.

## PCI devices (153 total; instrument-relevant ones below)

- **`MicoIf (CZMI)`** — `10EE:0303`, Carl Zeiss Microscopy GmbH, class
  `PCMCIA`, status OK, hinted "Zeiss realtime card (Xilinx FPGA); reached
  through MTB, no adapter". This is the proprietary FPGA card
  `docs/hardware/inventory.md` already documents ("Zeiss `MicoIf` FPGA
  card with proprietary `MicoIfDrv`"); the driver loads cleanly, and the
  hint states directly that it has no Micro-Manager adapter — first-hand
  confirmation of ADR-0007's premise for this stand (see Surprises /
  ADR-0007 note below).
- `PVCam PCIe Camera` — `1B6B:0001`, Photometrics and QImaging, class
  `PMPCIEWDF`, status OK, hinted "Photometrics" → adapter `PVCAM`.
  Confirms the camera route.
- CPU: twelve logical `Intel(R) Xeon(R) CPU E5-2623 v3 @ 3.00GHz` entries
  (a dual-socket Haswell-EP part, c. 2015). GPU: NVIDIA Quadro M4000.
  **Inferred**: consistent with `docs/hardware/inventory.md`'s "PC: HP
  Z840" (a dual-Xeon-E5-v3/v4 workstation class), though no HP system-model
  string appears in this capture to confirm the chassis directly.
- One PCI device reported in an **error** state: `Base System Device`
  (`10B5:8609`), status Error — the same VID:PID appears twice elsewhere as
  a healthy `PCI-to-PCI Bridge`. Not obviously instrument-related; flagged
  below as unexplained.
- The remaining ~148 entries are Xeon/chipset/QPI/memory-controller
  housekeeping devices, network adapters (Intel I218-LM, two ×X520-2,
  I210), audio and SATA/USB host controllers — ordinary workstation
  hardware, not instrument-relevant.

## Vendor hints (3)

| vendor | matched | adapter | note |
|---|---|---|---|
| Sentinel | usb `Sentinel USB Key` | — | licence dongle of a vendor software; no adapter |
| Zeiss | pci `MicoIf (CZMI)` | — | Zeiss realtime card (Xilinx FPGA); reached through MTB, no adapter |
| Photometrics | pci `PVCam PCIe Camera` | `PVCAM` | — |

## `smc doctor`

Ran against the **demo** configuration — no Zeiss-specific `.cfg` exists
yet: Device API version 75, Module API version 10, MMCore 12.5.0, 265
adapters, demo devices yes. Core roles are the demo config's own (`Camera`,
`XY`, `Z`, `Autofocus`, `White Light Shutter`); xy position `(-0.0, -0.0)
µm`. Result: "✓ demo configuration loads and answers." This proves the
install itself is sound; it says nothing about this stand's own hardware,
which has no `.cfg` to load yet (MTB route, ADR-0007).

## `pytest`

245 collected, **242 passed, 3 skipped**, 30.62 s, Python 3.14.7, pytest
9.1.1. All three skips are environment-appropriate, not failures:
`--all-adapters` was not requested, `NotificationTester` did not crash the
child on this install, and the macOS-only crash guard test is skipped on
Windows. No failures.

## `pre-commit`

Only the install step was captured: `pre-commit installed at
.git\hooks\pre-commit`. No `pre-commit run` output is in this survey, so
the hooks' actual behaviour on this PC is unverified.

## Surprises / open questions

1. **Python 3.14.7** was used, not the Python 3.11
   `docs/hardware/microscope-pc-setup.md` §1 instructs (`py -3.11 -m venv
   .venv`). It is within `requires-python = ">=3.10"`, so nothing failed,
   but it deviates from the documented procedure — worth confirming
   whether 3.14 is now the intended version for new PCs, or whether this
   PC's operator used whatever `py` resolved to by default.
2. The Micro-Manager install lives on `D:`, not `C:` (see System, above).
3. **`notes.txt` is empty (0 notes) and no `mtb-config` folder or photos
   were captured.** `docs/hardware/microscope-pc-setup.md` §2 specifically
   asks the Zeiss PC for both: a `robocopy` of
   `C:\ProgramData\Carl Zeiss\MTB2011` and the MMStudio GUI's *Help → About
   Micro-Manager* device-interface version in `notes.txt`. Without them we
   cannot confirm the standalone MMStudio GUI's DIV matches pymmcore-plus's
   DIV 75 (the mismatch §1 warns about), nor confirm the MTB component
   list, ports or version against `docs/hardware/inventory.md`'s "MTB 2011
   - 2.12.0.7".
4. The second on-disk install, `Micro-Manager-75.20260319`, is **not
   confirmed** to be the vendor MMStudio GUI that
   `docs/hardware/microscope-pc-setup.md` §1 says is already on this PC
   ("The Zeiss PC has one") — it may equally be a second
   pymmcore-plus-managed nightly. Only the GUI's own About dialog (item 3,
   above) would settle this.
5. One PCI device (`Base System Device`, `10B5:8609`) is in an Error
   state, while the same VID:PID is healthy elsewhere as a `PCI-to-PCI
   Bridge`. Not obviously instrument-related, but unexplained.
6. `smc doctor` only exercised the demo configuration; nothing here
   confirms end-to-end camera or stage control on real hardware — that
   needs the MTB backend (ADR-0007 option A) to exist first, plus a
   hardware session with the vendor software closed.
7. The survey folder was first named `zeiss-axio-observer`; it now follows the setup guide's `zeiss-observer7`.

## Does this support or contradict ADR-0007?

**Supports it, directly.** ADR-0007 already assumed the MTB route from
`lightsheet-live-tracking-tool`'s field experience; this survey adds
first-hand OS-level evidence for the same conclusion on this PC:

- The `MicoIf` FPGA card is present and its driver loads, but
  `smc discover`'s own vendor hint says in as many words that it has "no
  adapter" and is "reached through MTB" — there is no Micro-Manager
  adapter for it in the 265 installed, and none could be, since MTB is a
  Windows service/API, not a device with a serial or USB device protocol
  a C++ adapter could speak.
- The stand's CAN bus enumerates as a USB device (`ZeissCanNode`), not a
  serial port; the only COM port present is a standard, unrelated one.
  This is exactly the "CAN29 over USB into `CZCanSrv`" picture ADR-0007
  describes, and it is why `ZeissCAN29` (which speaks CAN29 over a serial
  port) cannot reach this stand.
- The camera route through Micro-Manager's `PVCAM` adapter is confirmed
  independently (PCI device present, driver OK, correctly hinted).

Nothing in the survey contradicts ADR-0007's direction for this stand.
