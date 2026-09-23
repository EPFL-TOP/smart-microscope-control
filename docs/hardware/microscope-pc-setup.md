# Setting up a microscope PC, and what to send back

For the Windows PCs attached to the stands, **one PC at a time**: install
the project, survey the machine into **one folder per PC**, bring the
folder back. Nothing here moves hardware.

## 0. One folder per PC

| Stand | Folder | Issue |
|---|---|---|
| Nikon Ti2-E | `nikon-ti2` | #16 |
| Nikon Ti-E | `nikon-tie` | #17 |
| Zeiss Axio Observer 7 | `zeiss-observer7` | #27 |
| Viventis LS1, first PC | `viventis-ls1-a` | #29 |
| Viventis LS1, second PC | `viventis-ls1-b` | #29 |

Put a sticker saying *A* or *B* on the two LS1 PCs so the names stay with
the machines. The folder name is also the future profile name
(`profiles/<folder>.toml`).

Suggested order: the **Ti2** first — the best-known stand, so it validates
the procedure, and it feeds milestone M2 — then the **two LS1** (the
biggest unknown: which controller performs the light-sheet timing), then
the **Ti-E** and the **Zeiss**, which are well documented already.

**The repository is public.** A raw survey contains serial numbers and
host names. It is never pasted into an issue or committed: it lives in
`local/surveys/<folder>/`, which is git-ignored, on the PC and on the Mac.
The design session publishes a curated, redacted version.

## 1. Install (once per PC, ~15 min)

Prerequisites: Python 3.11 for all users (Miniconda "for all users" is
fine), Git for Windows, the Microsoft Visual C++ x64 redistributable.

```bat
cd C:\Tools
git clone https://github.com/EPFL-TOP/smart-microscope-control.git
cd smart-microscope-control
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
mmcore install            :: full Micro-Manager nightly with every device adapter (Windows)
smc doctor
```

`mmcore install` writes to `%LOCALAPPDATA%\pymmcore-plus\pymmcore-plus\mm\`
(platformdirs repeats the name on Windows; from pymmcore-plus 0.18.1's
source) and needs no administrator rights. `mmcore list` prints the folder
it really uses. If the download is blocked by the corporate proxy,
set `HTTPS_PROXY` or run once from a network with direct access.

### Every terminal you open for this project

```bat
cd C:\Tools\smart-microscope-control
.venv\Scripts\activate
set PYTHONUTF8=1
```

`PYTHONUTF8=1` makes Python write UTF-8 when its output is redirected to a
file; without it a redirected `smc doctor` crashes on Windows (#42). Set it
per terminal, not machine-wide: other Python tools on a shared PC may rely
on the old behaviour.

### If a Micro-Manager GUI (MMStudio) is already installed

The Zeiss PC has one. Its **device-interface version (DIV)** must match
pymmcore-plus's, or a `.cfg` saved by the GUI fails silently in Python.
`smc doctor` prints pymmcore-plus's DIV (`Device API version NN`); the
GUI's is under *Help → About Micro-Manager*. They must be equal. If not,
install a nightly from the date range of the right DIV
(<https://micro-manager.org/Device_change_log>) or point pymmcore-plus at
the GUI's adapter folder in the profile (`adapter_search_paths`).

### Nikon Ti2 only

The `NikonTi2` adapter loads `Ti2_Mic_Driver.dll` from **its own folder**.
Copy (never move) it from `C:\Program Files\Nikon\Ti2-SDK\bin` into the
Micro-Manager folder that `smc doctor` reports. Without it the adapter
lists no devices and it looks as if the adapter were missing.

## 2. Survey the machine

**When**: nobody is using the microscope and the vendor software
(NIS-Elements, ZEN, the Viventis software) is closed. Listing the devices of
some adapters asks the vendor SDK, which talks to the stand — the Nikon Ti2
adapter does — and a stand accepts one connection at a time.

In a terminal prepared as in §1, name the folder once (the table in §0):

```bat
git pull
set SURVEY=C:\Tools\smart-microscope-control\local\surveys\nikon-ti2
smc discover --out %SURVEY%
```

`smc discover` writes `inventory.json` and
`inventory.txt` into the folder: the Micro-Manager install and its device
interface version, the installed adapters and the devices of the ones this
lab is likely to use, serial ports with their USB IDs, USB/PnP devices, PCI
cards, and hints that map known vendor IDs to adapters. It initialises
nothing. Do not use `--probe-adapter` on a first visit.

Then add to the same folder:

- **Every PC**: `photos\` — the back panel, every controller's front
  panel, every label with a model or part number (labels beat software) —
  and `notes.txt`: the vendor software and its version, what is physically
  connected, anything unexpected.
- **Nikon PCs**, if `nikon-control` is installed: in a terminal of its own
  environment, `set PYTHONUTF8=1` and `set SURVEY=…` again, then

  ```bat
  nikon-control-scope adapters > %SURVEY%\nikon-control-adapters.txt
  nikon-control-scope stand --notes --deep > %SURVEY%\nikon-control-stand.txt
  ```

- **Zeiss PC**: MTB's configuration lists every device and the port it
  uses; add the MMStudio device interface version (*Help → About*) to
  `notes.txt`.

  ```bat
  robocopy "C:\ProgramData\Carl Zeiss\MTB2011" %SURVEY%\mtb-config *.xml /S /MAX:5000000 /R:0 /W:0
  ```

- **LS1 PCs**: the vendor's configuration and logs, size-capped, never
  data.

  ```bat
  robocopy C:\Viventis %SURVEY%\vendor-config *.ini *.xml *.json *.cfg *.config *.yaml *.yml *.log /S /MAX:5000000 /R:0 /W:0
  ```

- **Any PC where `lightsheet-live-tracking-tool` is installed** (the Zeiss,
  probably the LS1s): its hardware audit — PCI cards, slots, drivers.

  ```bat
  powershell -NoProfile -ExecutionPolicy Bypass -File <tracking-tool>\tools\audit_windows_hardware.ps1 > %SURVEY%\audit.txt
  ```

## 3. Bring it back

Copy the folder to the Mac — network share, USB stick, e-mail to yourself —
into `smart-microscope-control/local/surveys/<folder>/` (git-ignored there
too), and tell the design session "survey `<folder>` is in". It turns the
folder into one pull request: `docs/hardware/<folder>.md` (what the stand is
made of), a draft `profiles/<folder>.toml`, the stand's device list as a
redacted test fixture for role resolution, and a summary on the stand's
issue.

## 4. What must not happen during a first visit

- No survey while someone acquires, or while NIS-Elements, ZEN or the
  Viventis software is running.
- No `--probe-adapter`, no `.cfg` built, no device initialised, nothing
  moved.
- Nothing from `local/surveys/` pasted into an issue or committed: the
  repository is public.
