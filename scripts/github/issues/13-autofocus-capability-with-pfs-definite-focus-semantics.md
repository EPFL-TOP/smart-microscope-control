---
title: feat(core): Autofocus capability with PFS / Definite Focus semantics
labels: [type: feature, area: core, scope: nikon-ti2, scope: zeiss-observer7, priority: p0]
milestone: M2 — Nikon on hardware
---
## Goal
`Autofocus`: `engaged`, `in_range`, `locked`, `engage(timeout) -> bool`, `disengage()`, `offset` (device units, named as such), `focus_by(delta)` dispatching to the offset when locked and to Z otherwise. `ZStage.move_to_um` suspends and re-engages an engaged autofocus (Nikon: setting Z disables PFS, micro-manager #1815).

## Context
`nikon-control` `Scope.{engage_pfs,pfs_in_range,focus_by,move_z}`; Zeiss DF2 in the tracking tool (`MTBDefiniteFocus`: learn/restore/stabilize_now, corrects via focus drive or piezo). Offset units differ per stand and are never converted implicitly.

## Acceptance criteria
- [ ] Protocol + Micro-Manager implementation (continuous-focus API + offset device role)
- [ ] FakeCore test: `test_moving_z_re_engages_pfs`; `engage()` returns False on no lock instead of raising
- [ ] Demo-devices test with `DAutoFocus`
- [ ] Profile field for offset units / limits
