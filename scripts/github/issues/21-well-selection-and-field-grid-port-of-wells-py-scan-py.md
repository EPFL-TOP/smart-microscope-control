---
title: feat(plugins): well selection and field grid (port of wells.py + scan.py)
labels: [type: feature, area: plugins, priority: p1]
milestone: M3 — Plate tools as plugins
---
## Goal
Choose wells (`A1`, `B2-B5`, `A1:D6`, `C*`, `*3`, `all`; serpentine order), lay a field grid per well from the camera's real field of view (`useq.GridRowsColumns`), estimate frames and time, and scan: frames + a manifest with per-frame stage coordinates so a detection converts back to a stage position.

## Context
Port `nikon-control/src/nikon_control/scope/{wells,scan}.py` with their tests; `stage_of_pixel` uses the camera calibration when present. Output format decision (TIFF folder vs OME-Zarr) tracked in the backlog I/O issue — start with TIFF + manifest.

## Acceptance criteria
- [ ] Selection parsing + ordering pure and tested; `to_plan()` → `useq.WellPlatePlan`
- [ ] `scan` plugin: steppable, failed field recorded and skipped, manifest rewritten per well
- [ ] Refuses to plan without a pixel size
