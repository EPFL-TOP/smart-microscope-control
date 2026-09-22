---
title: spike(viventis): inventory the PyMCS API and decide the integration route
labels: [type: spike, area: backend, scope: viventis-ls1, status: needs-hardware, priority: p1]
milestone: M6 — Viventis LS1
---
## Goal
Know what the Viventis LS1's PyMCS exposes — stage/position read & write outside a running acquisition, camera access, acquisition start/stop, events — so the route (unicore devices / direct backend / file-ingest + nudge as today) can be decided in a short ADR.

## Context
The tracking tool's `MicroscopeInterface_LS1` runs inside PyMCS scripts and only uses `relative_move` + file watching. `viventis_control` (bokeh GUI) has more calls to mine.

## Output
- [ ] `docs/hardware/viventis-pymcs-api.md`: objects, methods, units, threading model, licensing
- [ ] A decision issue / ADR for the backend route
