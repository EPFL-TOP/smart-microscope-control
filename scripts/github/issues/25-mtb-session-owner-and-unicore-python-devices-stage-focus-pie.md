---
title: feat(zeiss): MTB session owner and unicore Python devices (stage, focus, piezo, changers, TL shutter, Colibri)
labels: [type: feature, area: backend, scope: zeiss-observer7, status: blocked, priority: p0]
milestone: M4 — Zeiss Axio Observer 7
---
## Goal
Per ADR-0007 (blocked until accepted): the Axio Observer 7's MTB components become pymmcore-plus unicore devices in the same `UniMMCore` as the PVCAM camera, so roles, capabilities, `useq` MDA and every plugin work unchanged.

## Context
Port from `lightsheet-live-tracking-tool/tracking_tools/microscope_interface/mtb.py` (`MTBSession.shared()` — one login per process; `MTBAxis`, `MTBChangerDevice` (1-based positions, 0-based `GetElement`), `MTBObjective`, `MTBColibri`) with its fakes and tests. `pythonnet` ≥ 3 refuses int→Enum.

## Acceptance criteria
- [ ] `smc.hardware.backends.zeiss_mtb.session` owns the single login; closed at exit
- [ ] Devices: `MTBXYStage(XYStageDevice)`, `MTBFocus`/`MTBPiezo(StageDevice)`, changers as `StateDevice`, `MTBTLShutter(ShutterDevice)`, `MTBColibri(GenericDevice)` with per-LED properties
- [ ] Tests against a fake MTB API on any OS; `[zeiss]` extra
- [ ] `profiles/zeiss-observer7.toml`; light gated on every exit path (phototoxicity lesson)
