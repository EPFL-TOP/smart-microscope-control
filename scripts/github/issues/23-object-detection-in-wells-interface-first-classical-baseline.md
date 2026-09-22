---
title: feat(plugins): object detection in wells — interface first, classical baseline second
labels: [type: feature, area: plugins, priority: p1]
milestone: M3 — Plate tools as plugins
---
## Goal
Turn a scan into a list of objects with stage coordinates, class and score — the input to position selection and to any tracked acquisition. The *interface* (a `Detector` protocol over frames + manifest) matters more than the first detector.

## Context
`nikon-control` has a trained Faster R-CNN for cells (single/doublet/debris at 40×) and a placeholder BF detector; the tracking tool has a tail detector for embryos. Neither belongs in the core dependency set (torch).

## Acceptance criteria
- [ ] `Detector` protocol; `Detection` model (bbox px, stage xy µm, class, score, frame ref)
- [ ] Classical baseline (high-pass + Otsu + connected components) with tests on synthetic frames
- [ ] `detect` plugin reads a scan manifest, writes `detections` to the run document
- [ ] Torch-based detectors live in an optional extra / separate package, discovered as plugins
