---
title: feat(plugins): closed-loop tracking plugin over the acquisition loop (port of the tracking runner)
labels: [type: feature, area: plugins, priority: p2]
---
## Goal
The "smart" in smart microscopy: a plugin that watches frames as they arrive, tracks an object per position, and edits the next `MDAEvent`s' stage targets — the queue-driven MDA of `MicroscopeInterface_Micromanager` re-expressed over the capability layer.

## Context
`lightsheet-live-tracking-tool`: `TrackingRunner`, `MicroscopeInterface_Micromanager` (three regression invariants: unconditional cycle advance; drift updated under lock; baselines from config, never from the stage), CoTracker via serverkit. Faro (pertzlab) is the ecosystem framework for this.
