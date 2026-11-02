# The plate page as a service

On a microscope PC the [plate page](plate-page.md) can run as a Windows service, so
that it is there after a reboot and a logoff, at `http://localhost:5006/`, without a
console left open. The service is the plain `smc ui` command, run by
[NSSM](https://nssm.cc/) as the lab already runs `nikon-control`'s dashboard. The
design is [`m5-web-ui.md`](../design/m5-web-ui.md) §7.

The page **starts with the stand closed** (see
[Connect and Disconnect](plate-page.md#connect-and-disconnect)): a service that starts
at boot, or restarts after a crash, never takes the stand from NIS-Elements or ZEN.
The stand is held only between **Connect** and **Disconnect** on the page.

The page listens on `localhost` only and has no login, as a service too. From a desk,
use the PC's remote desktop. Serving it from a lab server needs an ADR on
authentication first: it is not allowed until then.

## Install

From an administrator's command prompt, with `nssm.exe` on the `PATH`. The paths are
an example: the checkout in `C:\smc\smart-microscope-control`, its venv in `.venv`,
the site's profiles in `C:\smc\profiles`, the run document in `C:\smc\runs`, the log
in `C:\smc\logs`.

```bat
mkdir C:\smc\logs
nssm install smc-ui "C:\smc\smart-microscope-control\.venv\Scripts\python.exe" -m smc.cli ui -p nikon-ti2 --run-doc C:\smc\runs\run.json --port 5006
nssm set smc-ui AppDirectory C:\smc\smart-microscope-control
nssm set smc-ui AppEnvironmentExtra SMC_PROFILES=C:\smc\profiles PYTHONUTF8=1
nssm set smc-ui ObjectName ".\lab-user" "the account password"
nssm set smc-ui AppStdout C:\smc\logs\smc-ui.log
nssm set smc-ui AppStderr C:\smc\logs\smc-ui.log
nssm set smc-ui AppRotateFiles 1
nssm set smc-ui AppRotateOnline 1
nssm set smc-ui AppRotateBytes 10485760
nssm set smc-ui AppStopMethodConsole 30000
nssm set smc-ui AppNoConsole 0
nssm set smc-ui Start SERVICE_AUTO_START
```

Replace `lab-user` and the password, and keep the double quotes: `cmd.exe` reads an
unquoted `&`, `|`, `<`, `>` or `^` as its own. Quotes do not protect a `%`, which
must be doubled (`%%`) in a `.bat` file, and a password with a `"` cannot be quoted
at all: set the account with `nssm edit smc-ui` instead (its *Log on* tab), which no
shell reads.

`AppExit` is left at its default, `Restart`, with NSSM's throttle: a page that ends
alone is started again, closed.

## Start, stop, restart, remove

```bat
nssm start smc-ui
nssm status smc-ui
nssm stop smc-ui
nssm restart smc-ui
nssm stop smc-ui
nssm remove smc-ui confirm
```

Another profile, run document or port is a change of its parameters, then a restart:

```bat
nssm set smc-ui AppParameters "-m smc.cli ui -p nikon-ti2 --run-doc C:\smc\runs\other.json --port 5006"
nssm restart smc-ui
```

## What each setting is for

- **What it runs**: the venv's `python.exe -m smc.cli ui ...`. On Windows a venv's
  `Scripts\python.exe` is itself a launcher (a copy of `venvlauncher.exe`) that runs
  the base interpreter as a child process, as the `smc.exe` console script does: the
  interpreter that holds the stand is that child. It shares NSSM's console, so the
  Ctrl-C reaches it; that is to be checked at a stand (below). `--run-doc` is an
  absolute path.
- **The account** (`ObjectName`): the Windows account for which Micro-Manager, the
  vendor drivers and the venv were set up on that PC. Not `LocalSystem`, which has
  none of that account's environment.
- **The working directory and the environment**: `AppDirectory` is the checkout,
  whose `profiles\` is on the profile search path, and `SMC_PROFILES` names the
  site's profiles. `PYTHONUTF8=1`, because the log is a file, which Windows would
  otherwise write in cp1252 (`µm` would be lost).
- **The log**: stdout and stderr both go to `C:\smc\logs\smc-ui.log`, rotated at
  10 MB while the service runs. It holds what the command prints (the `!` line of a
  stand that refuses its stop, the serving line, and the closing lines: what the
  stop did to a run that was going) and every warning and error. A Connect's outcome
  and a STOP's note are logged at INFO, which the command does not print without
  `-v`: they are on the page only. `-v` prints them, with the hardware layer's own
  INFO lines (not measured: how fast they fill the log).
- **The stop**: NSSM's first stop method is a Ctrl-C to the console it creates
  (`AppNoConsole 0`: without a console there is no Ctrl-C). That is the command's own
  Ctrl-C: a run that is going is told to stop, then the server stops, then the stand
  is released. NSSM waits only 1.5 s by default before its next method and ends with
  `TerminateProcess`, which releases nothing; `AppStopMethodConsole 30000` covers the
  page's 10 s wait for a run and the server's 10 s.
- **The start**: `SERVICE_AUTO_START`, at boot, before anyone logs on.

## What it changes for whoever uses the PC

- **The port is always held.** A `smc ui` started by hand on 5006 is refused with
  `port 5006 is in use`: use `http://localhost:5006/`, or give another `--port`.
- **The stand is held only between Connect and Disconnect.** While it is held,
  NIS-Elements, ZEN and the `smc` commands cannot open it: press Disconnect first.
  Disconnect switches nothing off.
- **Anyone logged on to the PC** (any account, at the PC or over RDP) reaches
  `localhost:5006` and can drive the stand. There is no login.
- **It survives a logoff and a reboot**, and starts with the stand closed.
- **Files**: the run document and what the tools write belong to the service's
  account, so their folder must be writable by it. A service does not see mapped
  drive letters (`Z:`): give local or UNC paths.
- **Restarts**: a changed profile, or another run document, needs `nssm set smc-ui
  AppParameters ...` or `nssm restart smc-ui`. A restart ends a run that is going,
  as Ctrl-C does.
- **What the command prints** goes to the log file, not to a console.
- **Stopping the PC**: press Disconnect first. How long Windows lets a service stop
  at shutdown is not known here.

## Not verified

None of this has run on a microscope PC yet. To check, at a stand:

- under NSSM, in session 0: whether the vendor SDK (Ti2 SDK, PVCAM, Zeiss MTB) and
  the camera driver open there, under the account chosen;
- Connect, then Disconnect, then NIS-Elements taking the stand, then Connect again;
- `nssm stop smc-ui` while a run goes: the Ctrl-C reaches the interpreter, the child
  of the venv's launcher, and the stand is released within `AppStopMethodConsole`;
- `nssm restart smc-ui` while a Connect is still opening a slow stand: the old
  process ends after its 10 s wait with the open still going, on a daemon thread;
- a reboot and a logoff;
- how long Windows lets the service stop at shutdown;
- whether the lab's NSSM version names its rotation parameter `AppRotateFiles`, and
  whether its `set AppParameters` takes the quoted string as written;
- an account set through `nssm edit`, and the *Log on as a service* right it needs;
- a stand opened on the page's worker thread, where the command used to open it on
  its main thread.
