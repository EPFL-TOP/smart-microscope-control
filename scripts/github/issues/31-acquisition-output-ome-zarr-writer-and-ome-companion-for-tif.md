---
title: feat(io): acquisition output — OME-Zarr writer and OME companion for TIFF runs
labels: [type: feature, area: plugins, priority: p2]
---
## Goal
Decide and implement where frames go: OME-Zarr for new acquisitions (ecosystem default; pymmcore-plus has writers), and an OME companion file for TIFF-per-frame runs so Fiji/Bio-Formats/tifffile see a 5-D series without rewriting a pixel.

## Context
`lightsheet-live-tracking-tool/tracking_tools/ome_companion/` (companion writer, tested against Bio-Formats 8.0.1; filename `ome-tiff.companion.ome` is load-bearing). `nikon-control` scan manifest (`scan.json`).
