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
   96-well one. Ask too what is **in** the wells, well by well: cells,
   beads or debris, medium or dry. Step 1 of the checklist has the operator
   confirm it at the stand. Ti2-E sessions 3 and 5 assumed cells and had
   none (#550, F5), and a wet well and a dry one do not behave alike (#539,
   #548).
10. Right after any switch-on of an autofocus that drives Z to its
    offset's plane (the Ti2-E's PFS, from smc or from the stand's own
    button), the next line of the checklist reads Z and compares it with
    the focus found by eye, before anything else runs. It says what to do
    if Z moved: switch the autofocus off with the stand's button, then
    write down both values. The jump went unnoticed in Ti2-E sessions 1
    and 2 (+235 µm, +206.6 µm; #205, #331).
11. **Commands are typed in Windows `cmd`, as written** (Ti2-E session 5,
    #550):
    - Every value is a number, never a placeholder: `<Z − 200>` was typed
      as it stood, and `<` and `>` are redirections (F9). Where a value
      comes from a read at the stand, the line says so and the guide gives
      the number.
    - A duration is timed with delayed expansion,
      `cmd /v:on /c "echo !time! & <command> & echo !time!"`. A plain
      `echo %time% & … & echo %time%` prints the same time twice (F1).
    - The drop that clears a turn sits on the turn's own line, so that the
      turn runs only after it: `smc z move 3720 && smc objective set
      "3-Plan Apo LmbdD0.25 40x" --confirm` (F16).
12. **SSH, read-only, where the stand has it** (the Ti2-E since session 5;
    owner decision 2026-11-15):
    - **At the start**, the guide gives the operator the whole line with
      its laptop's current address written in (`ipconfig getifaddr en0`),
      in the chat and never on GitHub. The operator runs `Start-Service
      sshd` and `Set-NetFirewallRule -Name OpenSSH-Server-In-TCP -Enabled
      True -RemoteAddress <the address>`.
    - **At the end**, the operator runs `Stop-Service sshd` and
      `Set-NetFirewallRule -Name OpenSSH-Server-In-TCP -Enabled False`,
      and the guide checks that port 22 no longer answers.
    - **Over SSH the guide may only:**
      - read files, `git log` and `git status`;
      - run `smc doctor` while nothing else holds the stand;
      - `scp -r` folders into **new** folders on the laptop.
    - smc over SSH is prefixed with `set PYTHONUTF8=1&&`, or `✓` and `µ`
      arrive in the console's code page (F18).
    - Every command that moves anything is still typed by the operator.

## During

A session that sits beside the operator at the stand (on a laptop, never on
the stand's PC) and walks the checklist with them. Used for the first time at
Ti2-E session 3 (2026-11-10): it caught in the moment what sessions 1 and 2
had missed (a Z read against the by-eye focus right after a PFS switch-on,
an objective not seated, a plate whose pitch is not useq's).

1. **The operator types; the session reads.** It never runs a command that
   reaches the stand, except the SSH reads of *Before* §12. It never widens
   a bound, never works around a refusal of the layer, and never decides a
   profile key outside the checklist's own procedure.
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
4. Keep `local/surveys/<stand>/session<N>-guide/notes.md` as a running log
   (step, time, command, outcome, finding `F<n>`). It sits outside
   `session<N>/`, which the copy from the stand brings back: session 3's
   copy erased its log. Remind the operator to save each console into
   `terminal<k>.log` before it is closed, the page's server console
   included. Name every file the operator saves (run documents, frames,
   Z-stacks) with its path. The laptop's clock and the stand's differ: the
   notes take their times from the run documents' UTC stamps where they
   exist.
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
5. Read the saved consoles and the run documents themselves, not only the
   guide's notes. The page's server console of Ti2-E session 5 held 16
   objective turns and a whole scan that the notes never saw (#550, F19
   and step 14).
6. Findings about the checklist or the commands go into this skill by a
   small PR.
