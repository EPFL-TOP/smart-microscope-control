---
title: test: simulated sample — stage-aware synthetic images for the demo camera
labels: [type: test, area: core, scope: demo, priority: p1]
milestone: M1 — Stage on the simulator
---
## Goal
The demo camera's frames do not depend on where the stage is, so plugins that *look* at the sample (plate finding, camera↔stage calibration, object detection) cannot be tested end to end. Provide a synthetic sample: a callable `(x_um, y_um, z_um) -> frame` rendering a well plate (bright wells, dark plastic) with objects, textured enough for phase correlation, and a hook that substitutes it for the demo camera's image.

## Context
`lightsheet-live-tracking-tool/tracking_tools/microscope_interface/synthetic_source.py` (`DriftingGaussianEmbryo`, `ReplayFromFolder`) is the precedent.

## Acceptance criteria
- [ ] `smc.testing.synthetic.PlateSample(plate="96-well", a1_center_xy, rotation_deg, objects=…)` renders frames from `useq` plate geometry
- [ ] `demo_microscope(sample=…)` fixture returns frames from the sample for the current stage position
- [ ] A test shows the frame brightness changes when the stage crosses a well wall
