---
title: adr: decide the route to hardware without a Micro-Manager adapter (ADR-0007)
labels: [type: adr, status: needs-decision, area: backend, scope: zeiss-observer7, scope: viventis-ls1, priority: p0]
milestone: M0 — Bootstrap
---
## Decision to make
How the Zeiss Axio Observer 7 (MTB 2011 over `pythonnet`; `ZeissCAN29` cannot reach its USB-CAN bus) and the Viventis LS1 (PyMCS) join a Micro-Manager-centred stack. Full analysis in `docs/adr/0007-non-micromanager-hardware.md` (status **proposed**).

## Options considered
- **A** Wrap the vendor API as **pymmcore-plus unicore Python devices** loaded in the same core as the camera — one bus, plugins and `useq` MDA unchanged; unicore is experimental and has no autofocus device type yet.
- **B** A direct backend implementing the capability protocols, camera in a separate core — simpler per device, two buses, everything implemented twice.
- **C** A C++ Micro-Manager adapter for MTB — the upstream way; months of work and licensing questions.

## Recommendation
**A**, with **B** as the fallback for what unicore cannot express (DF2 until an autofocus device type exists). LS1: inventory PyMCS first (M6 spike), then decide.
