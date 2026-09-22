---
title: adr: decide the user interface (ADR-0006)
labels: [type: adr, status: needs-decision, area: ui, priority: p0]
milestone: M0 — Bootstrap
---
## Decision to make
Which interface users of the microscopes see. Full analysis in `docs/adr/0006-user-interface.md` (status **proposed**).

## Options considered
- **A** Qt desktop composed from `pymmcore-widgets` — many ready widgets; desktop-only; fragile on GPU-less RDP sessions.
- **B** Web app in **Panel** (Bokeh) — remote & multi-user; what the lab already deploys; we write the microscope widgets.
- **C** Web app with a custom frontend (FastAPI + React/Vue, or NiceGUI) — most flexible; highest cost, lowest familiarity.
- **D** CLI only — always present; not an end-user interface.

## Recommendation
CLI as the reference interface (accepted regardless), **B** for the graphical interface, `pymmcore-widgets` as an optional `[qt]` engineering-console extra.

## To close this issue
Edit the ADR status to `accepted` (or `rejected` + a new ADR), merge, and unblock the M5 issue.
