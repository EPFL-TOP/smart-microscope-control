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

Prerequisites: Python 3.14 for all users (what the Ti2-E and Observer 7 PCs
run; any Python from 3.10 works, and CI tests 3.11 and 3.14 on Windows) (Miniconda "for all users" is
fine), Git for Windows, the Microsoft Visual C++ x64 redistributable.

```bat
cd C:\Tools
git clone https://github.com/EPFL-TOP/smart-microscope-control.git
cd smart-microscope-control
py -3.14 -m venv .venv
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
(<https://micro-manager.org/Device_change_log>).

To use the GUI's own adapters (one that only the GUI install has), list
the GUI's folder in the profile under `[micromanager]
adapter_search_paths`. `Microscope.open` searches it before pymmcore-plus's
folder and logs the full list at INFO. This does **not** fix a DIV
mismatch: an adapter whose DIV differs from pymmcore-plus's does not load.
The two DIVs must be equal first. MMCore takes each adapter from the first
folder that has it, so the GUI's folder supplies **every** adapter it holds,
not only the one you were missing: with a DIV mismatch the whole stand fails
to load, not just that adapter. The error then lists the folders in the order
they were searched.

### Nikon Ti2 only

The `NikonTi2` adapter loads `Ti2_Mic_Driver.dll` from **its own folder**.
Copy (never move) it from `C:\Program Files\Nikon\Ti2-SDK\bin` into the
Micro-Manager folder that `smc doctor` reports. Without it the adapter
lists no devices and it looks as if the adapter were missing.
`smc install-driver ti2` does the copy: it never moves the file and never
overwrites a different one (`--from` and `--to` name another source or folder).

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

### Reading a probe

Not part of a first visit (§4). When the owner asks for it, on a stand that
is powered with its vendor software closed,
`smc discover --probe-adapter NikonTi2 --out %SURVEY%` loads and initialises
the devices of that one adapter. The `Probe of …` table has one row per
device, and its `result` column has three words:

| result | what it means |
| --- | --- |
| `ok` | The device was loaded and initialised. `from` says who named it: `adapter` (the adapter's own list), `hub` (the initialised hub reports it, the adapter does not list it) or `both`. |
| `FAILED` | MMCore refused. `detail` is its own text, then a dash and what that text usually means when it is known. |
| `not probed` | Nothing was tried, and `detail` says why: the device needs a port (below), its hub did not initialise or does not report it, or another hub of the adapter was already connected. It is neither a pass nor a fault: a stand that was not tested says so. |

- **One hub per probe.** The Ti2-E lists the stand and then `Ti2-Simulator`;
  once the stand's hub is up the simulator is not tried, so its expected
  `-10` is not shown as a fault of the stand.
- **`--port COM3`** (needs `--probe-adapter`) gives that serial port to every
  probed device whose `Port` is empty or `Undefined` before it initialises,
  and the row says `port COM3`. A device that already has a port keeps it. A
  device that needs a port and was not given one is `not probed`, with the
  words `needs a port`. If no device needed it, a note says the port was not
  used.
- **A hub that will not initialise** is followed by `In order of likelihood:`
  and numbered causes, the first being another program holding the stand.
  Close NIS-Elements, Ti2 Control and any other smc or Micro-Manager session
  and run it again before reading the others.
- **`Native module failed to load`** means the vendor runtime the adapter
  wraps is not installed or not the right version; for `HamamatsuHam` that is
  DCAM-API. Where the camera is not fitted, that failure is expected.
- **`INCOMPLETE`** in the title means the adapter child stopped (a crash or
  the `--timeout-s` limit) before the last device: the rows are the ones it had
  finished, and the note under *Notes* names the device that was running.
- Under *Adapter devices*, a `meaning:` row says what an empty list, or a list
  the adapter fixes whatever is connected, tells about the stand.

### Build a configuration

Not part of a first visit (§4) either: it initialises every device of the
stand's adapter, as a probe does, so the stand must be powered and its vendor
software closed. `smc config build` writes a Micro-Manager `.cfg` from the
devices that answer, without the Java wizard. Look first, and write second:

```bat
smc config build --profile nikon-ti2 --out %SURVEY%\nikon-ti2.cfg --dry-run
smc config build --profile nikon-ti2 --out %SURVEY%\nikon-ti2.cfg
```

`--dry-run` prints the probe table, the roles it resolved, and the file it
would write, and writes nothing. `--out` has no default. The adapter is the one
Nikon stand the profile's configuration loads; `--adapter NikonTi2` names it
when the profile has none (the demo's does not), and the camera is only what the
stand's adapter provides, or the cameras of `--camera-adapter` (`smc discover`
lists the installed adapters). No camera adapter is tried unasked.

- **Nothing is written** when the stand's hub did not initialise (the output
  then lists the likely causes in order: another program holds the stand, power,
  the driver), when nothing below the hub initialised, when a named camera
  adapter gave no camera, or when the build child stopped before the end. Each
  is one `✗` line, with exit code 1.
- **`--skip DEVICE`** (repeatable) leaves a device out without loading it. When a
  device crashes the build (the Ti-E's `TIDiaLamp` may), the output says
  `re-run with --skip <device> to leave it out`; the file then lists it under
  *Did not connect, or left out*. On the Ti2-E, `--skip Ti2-E__0` builds from the
  simulator when the stand is off.
- **`--force`** replaces an existing `--out`, which is saved first as
  `<out>.bak`. Without it, an existing file is refused before anything is
  initialised.
- **A rebuild never loses a line it does not generate.** The presets, pixel
  sizes, a device's typed state labels and `Delay` and `FocusDirection` lines of
  the file it replaces are carried over; one that names a device the new file does
  not load (or selects a label the device does not have there) is set aside
  whole, as `# not carried:` comments under *Set aside*, and the output has a
  line `! set aside: <block>: <why>` for each. A later build reads those comments
  as lines again, so a preset comes back once its device does. Device `Property`
  lines are always set aside: put a device setting in a preset (`System/Startup`).
- **A device that needs a serial port is left out**, with its reason: this
  command sets no port.
- **Check what it wrote**: `smc doctor --config %SURVEY%\nikon-ti2.cfg`, as the
  last line says. A device that failed, a camera that is missing or a block that
  was set aside does not change the exit code: they are findings, and the file
  says them.

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
