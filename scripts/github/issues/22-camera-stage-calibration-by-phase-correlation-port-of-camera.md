---
title: feat(plugins): camera↔stage calibration by phase correlation (port of camera_cal.py)
labels: [type: feature, area: plugins, priority: p1]
milestone: M3 — Plate tools as plugins
---
## Goal
Move the stage a known distance in X and Y, correlate frames, and measure the full 2×2 pixel→stage mapping: pixel size, rotation, handedness. Reject featureless fields (weak correlation) instead of returning a confident wrong answer.

## Context
Port `nikon-control/src/nikon_control/scope/camera_cal.py` (+ tests). Uses `skimage.registration.phase_cross_correlation` — adds `scikit-image` to the plugin's dependencies (extra `[plugins]` or core dependency: decide in the PR).

## Acceptance criteria
- [ ] Pure `measure_shift`/`fit` with tests; `CameraCalibration.trustworthy`
- [ ] Plugin writes the matrix into the run document; `Camera.pixel_size_um` can be set from it per objective
- [ ] Demo test with the synthetic textured sample
