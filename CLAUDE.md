# CLAUDE.md — how work is done in this repository

You are a contributor to a long-lived scientific software project: one
control layer over several microscopes, with interchangeable tools. Read
`docs/architecture.md` once; read the ADR that covers the layer you touch
before changing it (`docs/adr/README.md`). The rules below are the project's
rules for everyone; you follow them without being reminded.

## Two kinds of session

Work is split between a **design session** and **execution sessions**, and
the GitHub issue is the contract between them:

- The **design session** owns architecture: ADRs, `docs/design/*.md`, and
  the plan of every issue (`/plan`). It reviews every PR (`/review`). It
  does not implement.
- An **execution session** (a fresh Claude Code session, any model) takes
  **one** issue labelled `status: ready`, implements exactly its plan,
  verifies, opens the PR and reports back on the issue (`/develop`). It
  never redesigns: a plan that does not survive contact with reality comes
  back as a `## Plan deviation` comment.
- The **owner** decides ADRs, merges PRs and runs hardware sessions.

Every execution session works in its **own worktree** (Claude Code's
`EnterWorktree`, under `.claude/worktrees/`) with its **own `.venv`**
(`python scripts/dev/worktree.py setup`) — the main checkout's venv is an
editable install of the main checkout's code. Parallel sessions never touch
the same modules; the design doc fixes the interfaces between them. The
design session stays in the main checkout.

## Workflow: issue → plan → branch → implement → verify → PR

1. **Start from an issue.** If none exists, create one with the matching
   template (`gh issue create`). Acceptance criteria before code.
2. **Plan in the issue** (a comment or the *Plan* field): steps, files,
   which tests prove it, whether hardware is needed. Keep it short. For a
   change of direction, write an ADR (`/adr`) instead of arguing in a PR.
3. **Branch** from `main`: `feat/<issue>-<slug>`, `fix/…`, `docs/…`,
   `chore/…`, `adr/…`. Never commit to `main`.
4. **Implement test-first where the behaviour is checkable**; the demo
   devices (`demo_core` fixture) are the default test bed. Port existing
   algorithms from the sibling repos *with their tests*.
5. **Verify before every push**: `ruff check . && ruff format --check . &&
   mypy && pytest`. Fix root causes, never silence a check to pass.
6. **Open the PR** with the template filled honestly (what was *not*
   verified is stated). Conventional-commit title. Link the issue with
   `Closes #N`. Request review; **never merge** — that is the owner's call.
7. Report outcomes faithfully. If a test fails or a step was skipped, say
   so with the output.

## Architecture rules (non-negotiable without an ADR)

- **Hardware is reached only through `smc.hardware`.** Nothing else imports
  `pymmcore_plus` for control, names a Micro-Manager device label, or calls
  a vendor SDK. Plugins depend on capability `Protocol`s (ADR-0003).
- **Roles, not device names.** New stand-specific knowledge goes into a
  profile (`profiles/*.toml`) or into role heuristics with tests — not into
  a plugin.
- **Pure algorithm ⇄ hardware loop split** in every plugin (ADR-0004). The
  algorithm is unit-tested on synthetic data; the loop on the demo devices.
- **Safety guards live in the layer**: jog limit, Z↔autofocus interlock,
  confirmed turret moves, dry-run. A tool cannot opt out; it can only ask
  for `force=True` at a call site that reads as deliberate.
- **Units in names**: `_um`, `_ms`, `_deg`, `_px`. Device-native units are
  named for what they are (`pfs_offset`) and never silently converted.
- **Measured beats assumed; unknown beats guessed.** A pixel size of 0 is
  reported, not defaulted. Anything assumed is labelled as such in the
  result.
- **Failures are findings.** A device that will not connect is a result
  with a diagnosis, not an exception that ends the inventory.

## Code style

- Python ≥ 3.10, `src/` layout, fully typed (`mypy --strict` on `src/`).
- `ruff` is the formatter and linter (config in `pyproject.toml`); Google
  docstrings. A docstring explains **why** — what hardware behaviour or
  decision the code encodes — not what the next line does.
- No `print` in library code; `logging`. The CLI talks through `rich`.
- Results and parameters are `pydantic` models; geometry uses `useq` types
  rather than re-deriving plate arithmetic.
- Windows is a first-class target: `pathlib`, no shell-isms, ASCII-only in
  `.ps1`/`.bat`, and **always `encoding="utf-8"`** when reading or writing
  text (Windows decodes with cp1252 by default; ruff `PLW1514` enforces it).
- Tests are named for the behaviour or the failure they prevent
  (`test_moving_z_re_engages_pfs`).

## Testing (ADR-0005)

- `pytest` runs unit + simulator tests; `-m hardware --profile <name>` is
  opt-in and **only run by a human at the microscope**. Never run hardware
  tests, never move a real stage, never touch a vendor driver from this
  machine unless the user explicitly says the hardware is connected and
  they are present.
- Every new capability gets a contract test; every backend runs the
  contracts. Vendor quirks that the simulator cannot show get a `FakeCore`
  test.
- CI must be green on Linux, macOS and Windows.

## Hardware sessions

Prepared with `/hardware-session`. Pre-flight: vendor software closed,
`smc doctor --config <cfg>` passes, objective clearance checked. Everything
learned is filed as a *Hardware session report* issue the same day, then
turned into code, a profile entry, a test or a doc line.

## Where things are

- `src/smc/hardware/` bus, capabilities, roles, facade, backends
- `src/smc/plugins/` registry + built-in tools · `src/smc/run/` run document
- `profiles/` one TOML per instrument · `docs/hardware/` what we know
- `docs/adr/` decisions · `docs/roadmap.md` milestones
- `scripts/github/` labels, milestones, backlog bootstrap
- Sibling repos to port from: `../nikon-control` (`src/nikon_control/scope/`)
  and `../lightsheet-live-tracking-tool` (`tracking_tools/microscope_interface/`)

## Skills

Design session: `/plan <issue>` write the executable plan · `/review <pr>`
review against plan, architecture and tests · `/adr` propose a decision ·
`/hardware-session` prepare and write up a session at a stand.
Execution session: `/develop <issue>` implement one planned issue to a PR.
Scaffolds: `/new-plugin`, `/new-backend`. Shortcut: `/feature` (plan +
develop in one session, small issues only).

## Language

Code, comments, docs, issues and PRs are in **English**. Conversation with
the owner may be in French.
