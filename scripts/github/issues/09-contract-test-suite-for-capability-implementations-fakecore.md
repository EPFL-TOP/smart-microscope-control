---
title: test: contract test suite for capability implementations + FakeCore
labels: [type: test, area: core, priority: p0]
milestone: M1 — Stage on the simulator
---
## Goal
One parametrised suite per capability that every implementation must pass — Micro-Manager over the demo devices, fakes, and (in a hardware session) real stands — plus a `FakeCore` that records calls and models vendor quirks the simulator lacks (ADR-0005).

## Acceptance criteria
- [ ] `tests/contracts/test_xy_stage.py`, `test_z_stage.py`, `test_camera.py`, `test_shutter.py` parametrised over a `capability_backends` fixture
- [ ] `tests/fakes.py::FakeCore` with a call log; first quirk modelled: setting Z disables continuous focus
- [ ] `pytest -m hardware --profile X` selects the same contracts against a real profile (skipped without `--profile`)
- [ ] Documented in `CONTRIBUTING.md`
