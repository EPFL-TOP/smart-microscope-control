# 0008 — Repository workflow: issues, branches, PRs, CI

- **Status**: accepted
- **Date**: 2026-09-22

## Context

The project is developed by lab members together with an AI coding agent
(Claude Code) and should read like a well-run open-source project: every
change traceable to a reason, reviewable, and verified before it lands.
`nikon-control` grew as a long series of "add new tools" commits on `main`,
which made its history useless as documentation.

## Decision

- **Everything starts as an issue** (feature, bug, ADR, hardware session).
  Issues carry acceptance criteria and, before coding, a short plan.
- **`main` is protected**: no direct pushes, PR required, CI green required,
  squash merge, linear history. (GitHub settings — configured by the owner.)
- **Branches**: `feat/<issue>-<slug>`, `fix/…`, `docs/…`, `chore/…`, `adr/…`.
- **Commits** follow Conventional Commits (`feat(core): …`, `fix(zeiss): …`,
  `docs(adr): …`, `test: …`, `chore(ci): …`). The squash-merge title is the
  PR title, so PR titles follow the same rule.
- **Pull requests** use the template: what, why, how verified, safety, docs.
  A PR that moves real hardware says so and links its session report.
- **CI gates**: `ruff check`, `ruff format --check`, `mypy --strict`, tests
  on Linux/macOS/Windows against the demo devices.
- **Labels** (created by `scripts/github/bootstrap.py`): `type: *`,
  `area: *`, `scope: <microscope>`, `status: *`, `priority: p0–p2`,
  `good first issue`. **Milestones** track the roadmap (`docs/roadmap.md`).
- **Decisions** are ADRs (ADR-0001). **Hardware knowledge** goes into
  session-report issues first, then into profiles, tests and
  `docs/hardware/`.
- **The agent** follows `CLAUDE.md`: it plans in the issue, works on a
  branch, verifies locally, opens the PR, and never merges.

## Alternatives considered

- **Trunk-based commits by the owner**: fastest for one person; loses
  review, CI gating and the paper trail the project exists to build.
- **Git flow with `develop`**: unnecessary ceremony for this team size.

## Consequences

- Slightly slower for trivial changes; far better history and onboarding.
- `scripts/github/` is the source of truth for labels, milestones and the
  initial backlog; re-running the bootstrap is idempotent.
