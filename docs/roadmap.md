# Roadmap

Milestones are GitHub milestones; the issues under each are created by
`scripts/github/bootstrap.py` from `scripts/github/issues/`. Order within
a milestone is roughly dependency order. Nothing here is a date: the lab
gets microscope time when it gets it.

## M0 — Bootstrap *(this repository as it stands)*

- Package, CLI (`smc doctor`), simulator tests, CI on three OSes,
  pre-commit, ADRs 0001–0008, backlog. → **PR #1**
- Owner: protect `main`, decide ADR-0006 (UI) and ADR-0007 (non-MM hardware).

## M1 — The stage, on the simulator

The abstraction layer exists and is proven on the demo devices.

- Capability protocols: `XYStage`, `ZStage`, `Camera`, `Shutter`,
  `Properties` (Autofocus/Turret/LightSource/Channels follow in M2).
- Role resolution (profile > core slots > heuristics) with candidate
  reporting.
- `Microscope` facade over `CMMCorePlus`, safety policy (jog guard, soft
  limits, dry-run).
- Profiles: TOML schema, `profiles/demo.toml`, loader, validation.
- Contract test suite; `FakeCore`; synthetic image source.
- CLI: `smc devices`, `smc stage get|move|jog`, `smc snap`.

## M2 — Nikon Ti2 (then Ti-E) on hardware

- Port the Nikon stand knowledge from `nikon-control`: adapters, driver
  DLL checks, dynamic vs fixed device lists, TIRF exclusions, PFS
  interlock, turret confirmation.
- `Autofocus` capability with PFS semantics (engaged / in range / locked /
  offset units); `ObjectiveTurret`, `LightSource`, `Channels`.
- Config builder (`smc config build`) from what actually answers.
- Hardware session 1: `smc doctor`, contract tests, stage + PFS on the Ti2.
- Ti-E profile and session.

## M3 — Plate tools as plugins

- Plugin API: `Plugin` protocol, entry-point registry, `Params`/`Result`
  models, run document, `smc run`.
- Plugins ported from `nikon-control`: plate registration (wells, wall
  touches), automatic plate finding (lattice fit), well selection + field
  grid, camera↔stage calibration, throughput timing.
- New plugin: object detection in wells — interface first, a classical
  detector second, the trained detector from `nikon-control` third.

## M4 — Zeiss Axio Observer 7

- Per ADR-0007: MTB session owner; unicore Python devices for stage,
  focus, piezo, changers, TL shutter, Colibri; DF2 as a device +
  `Autofocus` implementation; PVCAM camera in the same core.
- Profile, contract tests in a hardware session, port of the phototoxicity
  guards (darken on every exit path).

## M5 — Web UI

- Per ADR-0006: connect to a profile, drive, snap/live, run plugins from
  generated forms, show the run document. Deployable as a service.

## M6 — Viventis LS1

- PyMCS API inventory → decision → backend (unicore devices, direct, or
  file-ingest + position nudge as today).

## Backlog (any time)

- `smc discover`: OS-level inventory (serial VID/PID, USB, PCI) cross-
  referenced with installed adapters → suggested profile skeleton.
- Documentation site (mkdocs-material) from `docs/`.
- Data output: OME-Zarr writer for acquisitions; OME companion for TIFF
  runs (port from the tracking tool).
- Closed-loop tracking plugin over the run loop (port of the tracking
  runner) — the "smart" in smart microscopy.
