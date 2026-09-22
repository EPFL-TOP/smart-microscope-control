# Setting up a microscope PC, and what to send back

For the Windows PCs attached to the stands. Goal of a first visit: install
the project, prove Micro-Manager sees the machine, and bring back an
inventory that turns into a profile. **Nothing here moves hardware.**

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

`mmcore install` writes to `%LOCALAPPDATA%\pymmcore-plus\mm\` and needs no
administrator rights. If the download is blocked by the corporate proxy,
set `HTTPS_PROXY` or run once from a network with direct access.

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

Once `smc discover` exists (issue #30):

```bat
smc discover --json inventory.json > inventory.txt
```

Until then, this collects the same facts by hand — run it from the
activated venv and keep the output:

```bat
python -c "from pymmcore_plus import CMMCorePlus as C; c=C(); print(c.getAPIVersionInfo()); print(c.getVersionInfo()); print(sorted(c.getDeviceAdapterNames()))"
python -c "from pymmcore_plus import CMMCorePlus as C; c=C(); [print(a, list(c.getAvailableDevices(a))) for a in ('NikonTi2','NikonTI','HamamatsuHam','PVCAM','ZeissCAN29','Marzhauser','MarzhauserLStep','ASIStage','ASITiger','NIDAQ','PI','PI_GCS_2','ThorlabsFilterWheel','Arduino') if a in c.getDeviceAdapterNames()]"
powershell -NoProfile -Command "Get-PnpDevice -PresentOnly | Where-Object {$_.Class -in 'Ports','USB','Camera','Image','System','Unknown'} | Select-Object Status,Class,FriendlyName,Manufacturer,InstanceId | Format-Table -AutoSize -Wrap"
powershell -NoProfile -Command "Get-PnpDevice -PresentOnly | Where-Object {$_.InstanceId -like 'PCI\*'} | Select-Object Status,Class,FriendlyName,Manufacturer,InstanceId | Format-Table -AutoSize -Wrap"
```

On the Nikon PCs, `nikon-control-scope adapters` and
`nikon-control-scope stand --notes` (from the `nikon-control` checkout)
add the driver-DLL diagnosis. On any Windows PC, the tracking tool's
`tools\audit_windows_hardware.ps1` produces the fuller PCI/slot report.

## 3. Send it back

Paste `smc version`, the `smc doctor` output and the survey into the
stand's issue: Nikon Ti2 → #16, Nikon Ti-E → #17, Zeiss Axio Observer 7 →
#27, Viventis LS1 (each stand separately) → #29. Add photos of the back
panel and of any controller front panel; model numbers on labels beat
guesses from software.

For the LS1s also copy (read-only) the vendor's configuration folder
listing: `dir /s C:\Viventis > viventis-files.txt`, and the contents of any
`*.ini`, `*.xml`, `*.json` under it that name components, ports or timing.

## 4. What must not happen during a first visit

- No `.cfg` is built and no device initialised while the vendor software
  (NIS-Elements, ZEN, the Viventis software) is running: the stands accept
  one connection.
- Nothing is moved. `smc doctor` and the survey only read.
