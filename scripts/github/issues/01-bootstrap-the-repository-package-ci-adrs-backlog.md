---
title: chore: bootstrap the repository (package, CI, ADRs, backlog)
labels: [type: chore, area: ci, priority: p0]
milestone: M0 — Bootstrap
---
## Goal
A repository that installs, lints, type-checks and tests on Linux, macOS and Windows against Micro-Manager's simulated microscope, with the project's decisions and backlog written down.

## Delivered by the bootstrap PR
- `pyproject.toml` (hatchling, `smc` package, `smc` CLI), `ruff`, `mypy --strict`, `pytest` config, `pre-commit`.
- `smc doctor` / `smc version`; `smc.hardware.core` (open a core, demo config, install status).
- Simulator tests (`demo_core` fixture), CI matrix, issue/PR templates, dependabot.
- ADRs 0001–0008, `docs/architecture.md`, `docs/roadmap.md`, `docs/hardware/inventory.md`.
- `CLAUDE.md`, `CONTRIBUTING.md`, project skills, `scripts/github/` bootstrap of labels, milestones and this backlog.

## Acceptance criteria
- [ ] CI green on the 4-job matrix
- [ ] `smc doctor` passes on a fresh clone after `mmcore install --test-adapters`
- [ ] Labels, milestones and backlog created with `python scripts/github/bootstrap.py`
