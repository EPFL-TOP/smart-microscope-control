---
name: hardware-session
description: Prepare a session at a real microscope (pre-flight checklist, exact commands, safety) and afterwards turn the raw log into a Hardware session report issue plus follow-up issues, tests and profile entries. Use before and after any visit to a stand.
---

# /hardware-session <microscope> [before|after]

Hardware is only ever driven by a **human present at the stand**. From a
development machine you prepare and you write up; you never run
`pytest -m hardware` or move an axis.

## Before

Produce a one-page checklist for the operator:

1. Goal and the issues it unblocks.
2. Power-on order and vendor-software state (NIS-Elements / ZEN / Ti2
   Control closed; Ti2: stand before controller; Observer 7: restart PC,
   camera, then stand).
3. `smc doctor --config <cfg>` (or `--profile`), expected output. Load a
   `.cfg` only if both checks print nothing: a `System`/`Startup` preset
   runs on every load, `--dry-run` included (#78), and so does a device
   property set during loading.
   ```bat
   findstr /n /i "ConfigGroup,System,Startup" <cfg>
   findstr /n /i /r "^Property," <cfg> | findstr /v /i /c:":Property,Core,"
   ```
   Not `findstr /i /r "^Property,[^C]"`: with `/i` the class ignores case
   too, so it skips every device whose label starts with C or c (a
   `CondenserTurret`).
4. Contract tests: `pytest -m hardware --profile <name> tests/contracts -x`.
5. The specific checks, each as a command with expected result and the
   safety note (clearance before any Z or turret move; jog guard on).
6. What to paste back: raw outputs, `smc version`, commit hash.
7. The console set-up of every step sets `SMC_PROFILE=<name>`: a command
   without `-p` otherwise opens the demo and says nothing (Ti2-E session
   1, #208: a snap meant for the Prime 95B came from the demo camera).
   The console of every step that is not redirected to a file is saved
   before it is closed (title bar, *Edit → Select All*, Enter, then paste
   into `terminal.log`), so the console comes back as well as the notes.
8. Every item left over from earlier notes on the session issue goes
   into the checklist, or is marked as left for later.

## After

From the raw log the operator pastes:

1. File a **Hardware session report** issue (template) — raw log included,
   as excerpts with the account name, host name and serial numbers
   redacted: the repository is public, and the whole folder stays in
   `local/surveys/<stand>/`. Say which excerpts are summarised or
   translated.
2. Extract every finding into: a profile entry, a `FakeCore` test, a
   heuristic with a test, or a line in `docs/hardware/inventory.md`.
3. Open follow-up issues with the right `scope:` label; link them from the
   report.
