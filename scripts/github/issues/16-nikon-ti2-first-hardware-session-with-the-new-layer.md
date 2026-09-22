---
title: hw: Nikon Ti2 — first hardware session with the new layer
labels: [type: hardware-session, status: needs-hardware, scope: nikon-ti2, priority: p0]
milestone: M2 — Nikon on hardware
---
## Goal
Prove the M1/M2 layer on the Ti2-E: `smc doctor --profile nikon-ti2`, contract suite, stage moves, PFS engage/lock/offset, snap through the Hamamatsu camera.

## Pre-flight
- [ ] NIS-Elements and Ti2 Control closed; stand on before controller
- [ ] `Ti2_Mic_Driver.dll` beside the adapter (`smc doctor` says so)
- [ ] Objective clearance checked; jog guard on

## Checks
1. `smc doctor --profile nikon-ti2` — roles resolved, candidates reviewed
2. `pytest -m hardware --profile nikon-ti2 tests/contracts -x`
3. `smc stage jog 100 0` then back; `smc snap`
4. PFS: engage → in range → locked; `focus_by` moves the offset while locked
5. Time a 9 mm move and a 200 µm move (feeds the timing plugin later)

## Output
A *Hardware session report* (this issue, filled), then follow-ups.
