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
3. `smc doctor --config <cfg>` (or `--profile`), expected output.
4. Contract tests: `pytest -m hardware --profile <name> tests/contracts -x`.
5. The specific checks, each as a command with expected result and the
   safety note (clearance before any Z or turret move; jog guard on).
6. What to paste back: raw outputs, `smc version`, commit hash.

## After

From the raw log the operator pastes:

1. File a **Hardware session report** issue (template) — raw log included.
2. Extract every finding into: a profile entry, a `FakeCore` test, a
   heuristic with a test, or a line in `docs/hardware/inventory.md`.
3. Open follow-up issues with the right `scope:` label; link them from the
   report.
