---
name: hardware-session
description: Prepare a session at a real microscope (pre-flight checklist, exact commands, safety) and afterwards turn the raw log into a Hardware session report issue plus follow-up issues, tests and profile entries. Use before and after any visit to a stand.
---

# /hardware-session <microscope> [before|during|after]

Hardware is only ever driven by a **human present at the stand**. From a
development machine you prepare and you write up; you never run
`pytest -m hardware` or move an axis.

## Before

Produce a one-page checklist for the operator:

1. Goal and the issues it unblocks. Order the steps by what they unblock
   and say which can be dropped: a visit cut short then still gives the
   most (Ti2-E session 2, #334, ended before steps 6–8).
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
9. Ask the owner what is on the stage before writing wells, coverage or
   wall positions: the plate type (and its brand, if known), or a dish.
   Ti2-E session 2 met a 24-well plate where the checklist assumed a
   96-well one.
10. Right after any switch-on of an autofocus that drives Z to its
    offset's plane (the Ti2-E's PFS, from smc or from the stand's own
    button), the next line of the checklist reads Z and compares it with
    the focus found by eye, before anything else runs. It says what to do
    if Z moved: switch the autofocus off with the stand's button, then
    write down both values. The jump went unnoticed in Ti2-E sessions 1
    and 2 (+235 µm, +206.6 µm; #205, #331).

## During

A session that sits beside the operator at the stand (on a laptop, never on
the stand's PC) and walks the checklist with them. Used for the first time at
Ti2-E session 3 (2026-11-10): it caught in the moment what sessions 1 and 2
had missed (a Z read against the by-eye focus right after a PFS switch-on,
an objective not seated, a plate whose pitch is not useq's).

1. **The operator types; the session reads.** It never runs a command that
   reaches the stand, never widens a bound, never works around a refusal of
   the layer, and never decides a profile key outside the checklist's own
   procedure.
2. Read the checklist and the notes on the session issue (`gh issue view N
   --comments`) and `docs/hardware/<stand>.md` first. For each step: the
   exact command, the expected result, the safety reads it needs (Z against
   the by-eye focus, the soft limits, the objective in place, the PFS offset
   after a turn: the Ti2-E restores one per objective). Check what the
   operator pastes against them and say **go** or **stop** before the next
   step.
3. On a blocking finding: record it with the raw lines, give the
   checklist's fallback if it has one, and move on; what has no fallback is
   left for the write-up, not improvised.
4. Keep `local/surveys/<stand>/session<N>/notes.md` as a running log (step,
   time, command, outcome, finding `F<n>`), remind the operator to save each
   console into `terminal<k>.log` before it is closed, and name every file
   the operator saves (run documents, frames, Z-stacks) with its path.
5. At the end, write a **Summary for `/hardware-session <stand> after`** at
   the bottom of the notes: done, not done, the findings in order, what the
   findings unblock offline and what needs the stand again. File only the
   issues the owner asks for that evening (p0, the next visit's critical
   path), named in the notes so that the write-up does not file them twice.

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
4. Post a `## Note (hardware session write-up, <date>)` on every open
   issue the session measured something for, and on the session issue
   what is left for the next visit.
