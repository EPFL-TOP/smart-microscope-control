---
title: feat(nikon): port stand knowledge — adapters, driver DLL checks, exclusions, diagnostics
labels: [type: feature, area: backend, scope: nikon-ti2, scope: nikon-tie, priority: p0]
milestone: M2 — Nikon on hardware
---
## Goal
Everything `nikon-control` learned about the two Nikon stands lives in this repository as profiles, heuristics, diagnostics and tests.

## Context
Sources: `nikon-control/src/nikon_control/scope/{stand,discover}.py` and `docs/scope-bringup.md`. Facts: `NikonTi2` enumerates dynamically (empty list = SDK unreachable, usually `Ti2_Mic_Driver.dll` not copied beside the adapter); `NikonTI` publishes a fixed list (a list proves nothing, only a probe does); TIRF positioners typed `XYStage`; PFS offset typed `Stage`; power-on order; Ti2 Control 2.10/2.20 crash; USB 2 for the Ti-E; `WinError` decoding for adapter loads.

## Acceptance criteria
- [ ] `profiles/nikon-ti2.toml`, `profiles/nikon-tie.toml`
- [ ] `smc.hardware.backends.nikon`: stand registry, `find_driver`, `deep_check`, `install_driver` (copy, never move), readiness notes
- [ ] `smc doctor --profile nikon-ti2` prints the diagnosis in order of likelihood
- [ ] Role exclusions (TIRF, PFS-as-focus) covered by resolver tests
