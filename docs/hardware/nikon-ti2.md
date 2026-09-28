# Nikon Ti2-E — PC survey

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
2. **A Photometrics PVCAM PCIe camera and an NI PCIe-6323 DAQ card are
   physically present** on this Ti2 PC (real PCI evidence, not just a
   static adapter list), but `docs/hardware/inventory.md`'s Ti2-E entry
   names only the Hamamatsu camera; PVCAM/Photometrics is attributed there
   to the Zeiss stand, and a PCIe DAQ to the Viventis LS1 as "a candidate
   light-sheet timing controller." Is this PC shared or repurposed, is
   `docs/hardware/inventory.md` missing something about the Ti2, or is
   another instrument colocated here? Nothing in this survey (no photos, no
   notes.txt) can settle it.
3. §2 of `docs/hardware/microscope-pc-setup.md` asks every visit for
   `photos/` and `notes.txt`, and, for Nikon PCs with `nikon-control`
   installed, `nikon-control-adapters.txt` / `nikon-control-stand.txt`.
   None of these are in `local/surveys/nikon-ti2/` — only `smc-discover`,
   `smc-doctor`, `pytest`, `pre-commit`. Without them, what is actually
   connected to COM3, and whether `nikon-control` is even installed on this
   PC, is unknown.
4. Whether **`Ti2_Mic_Driver.dll` was copied** into the Micro-Manager
   adapter folder (`docs/hardware/microscope-pc-setup.md` §1, "Nikon Ti2
   only") cannot be confirmed from this survey: `smc discover` does not
   initialise the `NikonTi2` hub, so the two entries it lists (`Ti2-E__0`,
   `Ti2-Simulator`) are the adapter's static declared device types, not
   proof the SDK was reached. A future visit needs `--probe-adapter` (vendor
   software closed, stand powered before the controller box, per
   `docs/hardware/inventory.md`) to settle it.
5. `smc doctor` only validated the demo configuration this visit (no
   profile/`.cfg` existed yet), so none of the Ti2-specific facts already in
   `docs/hardware/inventory.md` (four `XYStage` devices, PFS as a `Stage`
   device, `DiaLamp`/`Turret1Shutter`/`Turret2Shutter`) were re-confirmed —
   they still rest on the earlier `nikon-control`-derived institutional
   knowledge, not on this survey.
6. A second Micro-Manager install, `Micro-Manager-75.20260319`, sits
   alongside the fresh `Micro-Manager_2.0.3_20260914` nightly. Both look
   like device-interface 75 by name (**inferred**, not confirmed by either
   tool), so no DIV mismatch is apparent, but why a second install exists is
   not stated.
