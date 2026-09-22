---
title: feat(plugins): closed-loop tracking plugin over the acquisition loop (port of the tracking runner)
labels: [type: feature, area: plugins, priority: p2]
---
## Goal
The "smart" in smart microscopy: a plugin that watches frames as they arrive, tracks an object per position, and edits the next `MDAEvent`s' stage targets — the queue-driven MDA of `MicroscopeInterface_Micromanager` re-expressed over the capability layer.

**This is what `lightsheet-live-tracking-tool` becomes** (ADR-0007): it is first of all a live-tracking tool, so in this architecture it is a plugin. Its trackers and closed-loop logic are ported; its five `MicroscopeInterface_*` backends (LS1/PyMCS, ZEN, Files, MTB, Micro-Manager) are superseded by the capability layer, and its Panel/Bokeh apps are design input for the shell (ADR-0006), not code to port.

## Context
`TrackingRunner`, `MicroscopeInterface_Micromanager` (three regression invariants to keep: unconditional cycle advance; drift updated under lock; baselines from config, never from the stage), CoTracker via serverkit, per-position ROI files, `df2_owns_z` policy. Faro (pertzlab) is the ecosystem framework for this class of plugin.

## Acceptance criteria
- [ ] Plugin runs on the demo devices with the synthetic drifting sample and recovers a known drift (the existing smoke test, as a pytest)
- [ ] Trackers are pluggable (`Tracker` protocol); the torch/serverkit tracker is an optional extra
- [ ] Per-position ROIs and corrections live in the run document
