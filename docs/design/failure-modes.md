# Failure modes the code must survive

The project's shared checklist. `/plan` names the entries an issue must
handle; the adversarial reviewers (`.claude/agents/`) go through the ones
that apply to a change. Every entry was learned on this project or its
predecessors. **Add an entry** whenever a review confirms a failure whose
category is not here: an id, what happens, and how to check.

## Vendor DLLs and Micro-Manager adapters

- **FM-01 Prints while loading.** An adapter writes to stdout or stderr, with or without a newline. Any protocol over stdout breaks. *Check*: results travel through a file or a dedicated channel, never "the last line of stdout".
- **FM-02 Crashes or hangs while loading, or at unload.** A result already produced is lost when the parent trusts the exit code. *Check*: adapter work runs in a child; the child writes its result before teardown and calls `os._exit(0)` after the final write; the parent reads the result whatever the exit code.
- **FM-03 A crash dialog (Windows, macOS).** On Windows, a missing dependency or the crash reporter opens a modal dialog, which looks like a hang. On macOS, a crashing `Python.app` (Homebrew) makes ReportCrash show "Python quit unexpectedly" once per crash, `pytest` runs included (#47). *Check*: processes that load adapters guard themselves first: `SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX)` on Windows, and on macOS libc `_exit` as the raw handler of the crash signals (`_mm_child._no_error_dialogs`). A reproduction that crashes a native process on macOS goes through the same guard, or says in its evidence that it will open a dialog.
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
- **FM-15 A blocking device call under the microscope lock.** A vendor call with no timeout of its own (`snapImage`, a driver deadlock) holds the one `Executor` lock, so every other capability on that stand hangs with no diagnosis (#5 review). *Check*: waits poll `deviceBusy` against a deadline; callers of a lock held too long fail with `MicroscopeBusyError`, which names the holder (design §13); a call with no timeout of its own says so in its docstring.
- **FM-16 A wait that holds the lock.** Polling a moving device while holding the microscope lock blocks every reader and every stop for the length of the move, which is 60 s for a plate traverse (#54). *Check*: the command is sent under the lock and the wait takes it only for each poll; a test reads from another thread while a wait polls.
- **FM-17 A wait that gives up leaves the device moving.** A timeout, a Ctrl-C or any other exception during the wait raises, but the stage keeps going while the caller believes the move failed (#52 review). *Check*: every exit from a wait other than "idle" sends the device's stop first; so does a command that raises.
- **FM-18 A stop that races a command.** A stop that does not wait for the lock can reach the device just before a move command already under way, which then starts the stage after the stop (#54). *Check*: a stop that arrives while a command is being sent is sent again after it, and the move raises `MotionStoppedError`.
- **FM-19 A busy state that cannot be read.** `deviceBusy` raises. Reading that as idle lets the next action run during a move; reading it as moving forever locks the stand. *Check*: unreadable counts as moving, the refusal names the error, and a stop or `resume()` releases it; a motion that cannot be stopped (a `State` device) must still have a way out (#59 review).

## Windows

- **FM-20 Redirected output is cp1252 with strict errors.** `✓` crashes `smc doctor > f.txt` (#42). *Check*: the stream fix at process entry; files the tool writes are UTF-8.
- **FM-21 Text files without `encoding=`** are decoded as cp1252. *Check*: ruff `PLW1514`.
- **FM-22 A child sharing the console can change its code page** (`[Console]::OutputEncoding` in PowerShell). *Check*: `creationflags=CREATE_NO_WINDOW`.
- **FM-23 A child's stdout uses the ANSI code page.** *Check*: ASCII-only protocols, or explicit encodings on both sides.
- **FM-24 Paths.** Spaces, backslashes, platformdirs' layout (`%LOCALAPPDATA%\pymmcore-plus\pymmcore-plus\…`), files that cannot be replaced while open. *Check*: `pathlib`, no string surgery, no assumptions about the layout.
- **FM-25 PowerShell 5.1.** `ConvertTo-Json` prints an object, not a list, for one item. No BOM with `[Text.Encoding]::UTF8` on a pipe (checked on a Windows runner, 2026-09-23).
- **FM-26 An undeclared `ctypes` Win32 return type.** A `ctypes.windll`/`WinDLL` function called without setting `.argtypes`/`.restype` gets ctypes' default `restype` of `c_int` (32-bit); a `HANDLE` or other pointer-sized return value can be silently truncated on 64-bit Windows (adversarial review of #64). *Check*: every Win32 function reached through `ctypes.windll` declares `.argtypes` and `.restype` from `ctypes.wintypes` before the first call.

## Processes, CLI and files

- **FM-30 Artefacts after the console.** A broken pipe or Ctrl-C while printing loses files written afterwards. *Check*: check the destination before long work, write files first, one-line errors.
- **FM-31 External calls without a timeout.** *Check*: every OS tool and child has one, and its failure becomes a note ("failures are findings").
- **FM-32 Unbounded options.** `nan`, `inf`, negative or huge values reach `subprocess`. *Check*: bounds plus `math.isfinite`.
- **FM-33 A child's working directory is on `sys.path`.** *Check*: children run from a neutral directory.
- **FM-34 A failed write hides the report.** Writing the files first (FM-30) and exiting on an `OSError` loses the on-screen result of minutes of work (review of #47). *Check*: on a write error, still print what was collected, then the one-line error and the exit code.
- **FM-35 Killed is not gone.** `kill()` stops only the direct child: a helper process it started survives and may keep holding the hardware, and a process stuck in a driver call may not die, so an unbounded `wait()` after `kill()` hangs (review of #47, reproduced on macOS). *Check*: the wait after `kill()` has a timeout and its failure is a note; kill the process tree where helpers are expected. A tree kill that itself fails (`taskkill` denied or missing) but leaves the direct child dead is not a clean stop: the note says so — "could not be stopped" — instead of reading as resolved with the failure buried as a trailing detail (two reviewers, #50).

## Tests

- **FM-40 The default suite touches vendor adapters or hardware.** A microscope PC has the full nightly installed. *Check*: tests pin adapter lists to the test adapters (`DemoCamera`, `Utilities`, `SequenceTester`, `NotificationTester`).
- **FM-41 Assertions that an adapter is absent** break on a full install.
- **FM-42 `CliRunner` hides encoding problems.** *Check*: a subprocess test with `PYTHONIOENCODING=cp1252` and stdout to a file.
- **FM-43 A test that cannot fail.** *Check*: break the line under test and watch the test fail. For a rule table held as data, break each row's fields in turn: a line-by-line sweep misses rows (#6). For a test merging a real captured fixture into a synthetic sample (e.g. two `system_profiler` keys), the fixture can happen to contribute nothing usable, so the test passes whether or not the new key is actually read; check by removing the code path under test and confirming the assertion changes, not just that nothing crashes (#50).
- **FM-44 A threading test that hangs instead of failing.** A deadlock in the code under test freezes the suite until CI kills it, with no diagnosis. *Check*: threads meet on `threading.Event`; every `join(timeout=...)` is followed by `assert not t.is_alive()`; no assertion on timing tighter than about 0.5 s (Windows sleeps in steps of about 15 ms); interrupts are raised from stubs, not sent as signals.
- **FM-45 An assertion matched by the fixture, not the code.** A bare-word substring check on an error message (`"finite" in msg`) is satisfied by text the test infrastructure supplied, not the code under test: `tmp_path` is named after the test id, so `test_non_finite_soft_limits_are_rejected` puts `"finite"` in every path it writes, and the assertion passes even with the validator's check removed (#55 review). *Check*: assert the exact phrase the code raises, not a word that could also appear in the path, the test id, or another fixture's output.

## Data

- **FM-50 The repository is public.** Survey output (serial numbers, PnP instance IDs, host names) never goes into an issue or a commit.

## Concurrency (the layer's own locks and threads)

Learned from the design review of #59, where `/code-review` reproduced 15
races in the first motion guard.

- **FM-60 A motion found by name.** A waiter or a stop looks up "the" motion of a device by its label, and finds another caller's or none. A stop is lost, a stopped move reports an arrival, a stale waiter stops a new move. *Check*: only the action that sent a command waits for it, holding the lock; a stop is a per-device generation the action compares, not a flag on whatever registration exists (design §13).
- **FM-61 A give-up by a bystander.** A thread that only waits for someone else's motion stops it when its own short deadline passes. *Check*: only the thread that started a motion stops it when giving up.
- **FM-62 Log before stop.** A stop path that writes its log line first is delayed by a blocked console (a QuickEdit selection on Windows), or skipped by a second Ctrl-C. *Check*: send the stop, then log.
- **FM-63 A readback in its own lock section.** Another caller acts between the end of the wait and the readback, so the move returns another move's position and a set returns another thread's value. *Check*: command, wait and readback in one lock section.
- **FM-64 A halt that can be caught as a refusal.** `except SafetyRefusedError: skip` swallows the emergency stop, and the loop carries on. *Check*: the halt error derives from no error that a tool is expected to catch and carry on after.
- **FM-65 An interrupt between `acquire()` and `try`.** A Python-level lock wrapper (a generator context manager) leaves a window in which a Ctrl-C keeps the lock held for good; `with lock:` has none. *Check*: nothing between `acquire()` returning and the `try`; a test interrupts the main thread (`_thread.interrupt_main()`) while it waits for the lock.
- **FM-66 A check that is not the last one.** A halt (or any flag set without the lock) checked before a slow step, such as the guard polling a serial device, lets the action run after the flag was set. *Check*: check the flag again immediately before the command.
