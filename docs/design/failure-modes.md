# Failure modes the code must survive

The project's shared checklist. `/plan` names the entries an issue must
handle; the adversarial reviewers (`.claude/agents/`) go through the ones
that apply to a change. Every entry was learned on this project or its
predecessors. **Add an entry** whenever a review confirms a failure whose
category is not here: an id, what happens, and how to check.

## Vendor DLLs and Micro-Manager adapters

- **FM-01 Prints while loading.** An adapter writes to stdout or stderr, with or without a newline. Any protocol over stdout breaks. *Check*: results travel through a file or a dedicated channel, never "the last line of stdout".
- **FM-02 Crashes or hangs while loading, or at unload.** A result already produced is lost when the parent trusts the exit code. *Check*: adapter work runs in a child; the child writes its result before teardown and calls `os._exit(0)` after the final write; the parent reads the result whatever the exit code.
- **FM-03 Modal dialog on Windows.** A missing dependency or the crash reporter opens a dialog, which looks like a hang. *Check*: processes that load adapters call `SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX)` first.
- **FM-04 Helper processes inherit the pipes.** The pipe never reaches end-of-file; on Windows `subprocess.run(timeout=…)` then waits without limit after `kill()`. *Check*: child output goes to files; `Popen.wait(timeout)` then `kill()`.
- **FM-05 Non-UTF-8 strings.** pymmcore decodes with `surrogateescape`; one lone surrogate makes the JSON invalid. *Check*: strings from adapters are sanitised before they are serialised.
- **FM-06 Listing devices contacts the hardware.** `NikonTi2` asks the Nikon SDK, and a stand accepts one connection at a time (NIS-Elements, ZEN, a second core). *Check*: the default test suite never lists vendor adapters; surveys run with the vendor software closed.
- **FM-07 Hub-based adapters.** Peripherals appear only after the hub initialises (`getInstalledDevices`). *Check*: probe and configuration code asks the initialised hub.
- **FM-08 Pre-init properties.** Serial devices need `Port` before `initializeDevice`; failing without it is not a hardware fault. *Check*: detect it and report "needs a port".
- **FM-09 One message, many causes.** "Failed to load device adapter" hides missing DLL, wrong SDK, wrong bitness and interface mismatch. *Check*: ask the OS (`ctypes.WinDLL`, WinError code) when the diagnosis matters.

## Micro-Manager core

- **FM-10 Timeout shorter than the move.** MMCore's 5 s default is shorter than a plate traverse. *Check*: `setTimeoutMs` from the profile (60 s).
- **FM-11 Device-interface mismatch.** An MMStudio install and pymmcore-plus with different device-interface versions: a `.cfg` from one fails silently in the other. *Check*: report both versions.
- **FM-12 Moving Z switches PFS off** on the Nikon stands. *Check*: suspend and re-engage around Z moves.
- **FM-13 Several devices share a type.** Ti2: four `XYStage` devices, three of them TIRF positioners; Ti-E: three `Stage` devices. *Check*: type-first resolution with exclusions, and the candidates reported.
- **FM-14 Turret moves can crash an objective** under a loaded plate. *Check*: turret moves require confirmation.

## Windows

- **FM-20 Redirected output is cp1252 with strict errors.** `✓` crashes `smc doctor > f.txt` (#42). *Check*: the stream fix at process entry; files the tool writes are UTF-8.
- **FM-21 Text files without `encoding=`** are decoded as cp1252. *Check*: ruff `PLW1514`.
- **FM-22 A child sharing the console can change its code page** (`[Console]::OutputEncoding` in PowerShell). *Check*: `creationflags=CREATE_NO_WINDOW`.
- **FM-23 A child's stdout uses the ANSI code page.** *Check*: ASCII-only protocols, or explicit encodings on both sides.
- **FM-24 Paths.** Spaces, backslashes, platformdirs' layout (`%LOCALAPPDATA%\pymmcore-plus\pymmcore-plus\…`), files that cannot be replaced while open. *Check*: `pathlib`, no string surgery, no assumptions about the layout.
- **FM-25 PowerShell 5.1.** `ConvertTo-Json` prints an object, not a list, for one item. No BOM with `[Text.Encoding]::UTF8` on a pipe (checked on a Windows runner, 2026-09-23).

## Processes, CLI and files

- **FM-30 Artefacts after the console.** A broken pipe or Ctrl-C while printing loses files written afterwards. *Check*: check the destination before long work, write files first, one-line errors.
- **FM-31 External calls without a timeout.** *Check*: every OS tool and child has one, and its failure becomes a note ("failures are findings").
- **FM-32 Unbounded options.** `nan`, `inf`, negative or huge values reach `subprocess`. *Check*: bounds plus `math.isfinite`.
- **FM-33 A child's working directory is on `sys.path`.** *Check*: children run from a neutral directory.

## Tests

- **FM-40 The default suite touches vendor adapters or hardware.** A microscope PC has the full nightly installed. *Check*: tests pin adapter lists to the test adapters (`DemoCamera`, `Utilities`, `SequenceTester`, `NotificationTester`).
- **FM-41 Assertions that an adapter is absent** break on a full install.
- **FM-42 `CliRunner` hides encoding problems.** *Check*: a subprocess test with `PYTHONIOENCODING=cp1252` and stdout to a file.
- **FM-43 A test that cannot fail.** *Check*: break the line under test and watch the test fail.

## Data

- **FM-50 The repository is public.** Survey output (serial numbers, PnP instance IDs, host names) never goes into an issue or a commit.
