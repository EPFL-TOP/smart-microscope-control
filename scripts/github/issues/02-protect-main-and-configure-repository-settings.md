---
title: chore: protect main and configure repository settings
labels: [type: chore, area: ci, priority: p0]
milestone: M0 — Bootstrap
---
## Goal
The workflow in ADR-0008 is enforced by GitHub, not by convention.

## Owner actions (repository admin)
- [ ] Branch protection on `main`: require a pull request, require the `lint & types` and `tests (*)` checks, require linear history, block force pushes and deletion.
- [ ] Merge settings: squash only, default squash message = PR title + body, delete head branches on merge.
- [ ] Repository description and topics (`microscopy`, `micro-manager`, `pymmcore-plus`, `smart-microscopy`).
- [ ] Allow auto-merge (optional) and Dependabot security updates.
- [ ] Decide whether Discussions are wanted for design chatter (issues are fine too).
