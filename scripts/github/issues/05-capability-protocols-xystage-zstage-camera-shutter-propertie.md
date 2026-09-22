---
title: feat(core): capability protocols — XYStage, ZStage, Camera, Shutter, Properties
labels: [type: feature, area: core, priority: p0]
milestone: M1 — Stage on the simulator
---
## Goal
The typed interfaces every tool programs against (ADR-0003), with a Micro-Manager implementation over `CMMCorePlus` that runs on the demo devices.

## Context
Generalises `nikon-control/src/nikon_control/scope/control.py` (`Scope`). Autofocus, ObjectiveTurret, LightSource and Channels follow in M2 because the demo devices cannot exercise their semantics.

## Acceptance criteria
- [ ] `smc.hardware.capabilities` defines `XYStage`, `ZStage`, `Camera`, `Shutter`, `Properties` as `typing.Protocol`s with units in method names (`position_um`, `move_to_um`, `exposure_ms`)
- [ ] `smc.hardware.backends.mm` implements each over a core; `Camera.pixel_size_um()` returns 0.0 when unknown and never guesses
- [ ] `XYStage.move_by_um` refuses jogs above the configured limit unless `force=True`
- [ ] Each capability has a simulator test and a contract test (see the contract-suite issue)
- [ ] `mypy --strict` clean; docstrings explain the hardware behaviour encoded
