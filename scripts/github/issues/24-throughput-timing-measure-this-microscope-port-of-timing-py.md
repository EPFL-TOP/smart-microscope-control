---
title: feat(plugins): throughput timing — measure this microscope (port of timing.py)
labels: [type: feature, area: plugins, priority: p2]
milestone: M3 — Plate tools as plugins
---
## Goal
Measure stage settling (overhead + ms/mm from two distances), autofocus lock, channel switch and exposure/readout on the actual microscope, then answer "how many positions fit in this interval?" — with everything assumed rather than measured labelled as such.

## Context
Port `nikon-control/src/nikon_control/scope/timing.py` (+ 19 tests).

## Acceptance criteria
- [ ] Pure `Timings`/`plan()` with tests; `measure()` over capabilities, stoppable
- [ ] Results stored in the run document and reused by the scan plugin's estimate
