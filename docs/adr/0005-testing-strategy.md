# 0005 — Testing strategy: simulator first, hardware by session

- **Status**: accepted
- **Date**: 2026-09-22

## Context

Microscope time is scarce, hardware sessions are risky (a wrong Z move
crashes an objective), and most bugs in the predecessor projects were logic
bugs — PFS left off after a Z move, a survey that scored 0.99 while
millimetres wrong, a channel group that rotated the turret — which a
simulator would have caught. Micro-Manager ships a simulated microscope
that installs on every OS in seconds. Some behaviours (PFS interlock,
Definite Focus 2's actuator choice, MTB's one-login-per-process) do not
exist in the simulator and can only be pinned down with fakes.

## Decision

Four layers, cheapest first, and every PR states which it exercised:

1. **Unit tests** for pure algorithms and geometry (`tests/unit/`): fast,
   deterministic, with `hypothesis` for the arithmetic that has invariants
   (a calibration round-trips through `useq`'s forward model; a serpentine
   visits every well once).
2. **Simulator tests** against Micro-Manager's demo devices
   (`@pytest.mark.demo`, fixture `demo_core` / `demo_microscope`): the
   regression bed. They run in CI on Linux, macOS and Windows. In CI a
   missing install **fails** (`SMC_REQUIRE_MM=1`); locally it skips.
3. **Fake-core tests** for behaviours the simulator cannot show: a
   `FakeCore` that records calls and models the quirk (Z move → PFS off).
   Each such test is named for the failure it prevents.
4. **Contract tests** (`tests/contracts/`): one parametrised suite per
   capability protocol, run against every implementation available on the
   machine (Micro-Manager over demo devices, fakes, and — in a hardware
   session — the real stand). A new backend passes the contracts or it is
   not a backend.

**Hardware tests** carry `@pytest.mark.hardware`, are deselected by default
(`addopts = -m "not hardware"`), take `--profile <name>`, and are run only by
a person at the microscope. Every hardware session ends with a *Hardware
session report* issue; anything learned becomes a test, a profile entry or a
doc line before the next session.

Tooling: `pytest`, `pytest-cov` (reported, not yet gated), `ruff`, `mypy
--strict` on `src/`, `pre-commit`. CI runs lint → tests on a 4-job matrix.

## Alternatives considered

- **Hardware-in-the-loop CI** (a runner on a microscope PC): tempting later
  for one Nikon; unsafe unattended today and it would block on the
  microscope schedule.
- **Mocking `pymmcore_plus` wholesale**: brittle and tests our mocks. The
  demo devices are a *real* MMCore with real config-group, timeout and
  wait semantics.

## Consequences

- A plugin is *done* when it runs on the demo devices in CI, not when it
  ran once at the microscope.
- The `FakeCore` grows a catalogue of vendor quirks that doubles as
  documentation.
- Hardware sessions become short and scripted: `smc doctor`, the contract
  suite, the specific check — then a report.
