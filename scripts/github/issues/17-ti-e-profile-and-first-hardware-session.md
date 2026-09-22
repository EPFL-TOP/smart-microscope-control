---
title: feat(nikon): Ti-E profile and first hardware session
labels: [type: feature, area: backend, scope: nikon-tie, status: needs-hardware, priority: p1]
milestone: M2 — Nikon on hardware
---
## Goal
The older Ti-E driven through the same layer with only a profile change: `NikonTI` adapter, hub `TIScope`, fixed device list, `NikonTi.dll` on the system path, USB 2, PFS offset = raw steps / 40 with objective-dependent range, `TIDiaLamp` optionally skipped.

## Acceptance criteria
- [ ] `profiles/nikon-tie.toml` with quirks encoded
- [ ] Resolver picks `TIZDrive` over `TIPFSOffset`/TIRF for `focus` (test)
- [ ] Hardware session report filed; contract suite passed
