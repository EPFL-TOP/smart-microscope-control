---
title: feat(zeiss): Definite Focus 2 as a device and Autofocus implementation
labels: [type: feature, area: backend, scope: zeiss-observer7, status: blocked, priority: p1]
milestone: M4 — Zeiss Axio Observer 7
---
## Goal
DF2 through the `Autofocus` capability: learn a plane (opaque reference blob, base64 in the run document), restore, stabilize once per visit, operation modes, `piezo_focus_usage` reported (this stand corrects via the focus drive), `SignalQuality` ignored (reads 0 even when working).

## Context
`MTBDefiniteFocus` in the tracking tool + `tools/df2_test.py` findings (2026-09-11). unicore has no autofocus device type: expose as `GenericDevice` properties and implement the capability over it; propose an autofocus device type upstream.

## Acceptance criteria
- [ ] Per-position references stored in the run document; "who owns Z" policy (`df2_owns_z`) as a profile/param
- [ ] Fake-API tests; hardware check script equivalent to `df2_test.py` as a hardware test
