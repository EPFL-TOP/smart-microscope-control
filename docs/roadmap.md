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

The abstraction layer exists and is proven on the demo devices. Interfaces
between the issues are fixed in
[`docs/design/m1-hardware-layer.md`](design/m1-hardware-layer.md).

- `smc discover` (pulled forward from the backlog): read-only inventory of
  a microscope PC — adapters and their devices, serial VID/PID, USB/PnP,
  PCI — so the owner can survey every stand while M1 is built.

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

- Framework spike per ADR-0006: the same minimal shell in Panel and in
  NiceGUI on the demo devices; the owner decides by looking.
- The shell: connect, status strip with Stop, live view, drive, tools rail
  grouped by workflow stage, generated plugin forms, run document as a
  checklist. Identical on every microscope; deployable as a service.

## M6 — Viventis LS1 (×2)

- On-site inventory of both stands' **components** — controllers, cameras,
  lasers, scan mirrors, filter changers, piezo, timing hardware — as seen by
  the OS and by the vendor's configuration; map each to a Micro-Manager
  adapter or a unicore device; identify what performs the light-sheet
  timing (ADR-0007). PyMCS is a reference, not a route.
- Short follow-up ADR fixing the scope (full control vs stage+camera
  first), then `.cfg`, profile, contract tests, hardware session.

## Backlog (any time)

- `smc discover` follow-ups: suggested profile skeleton from an inventory;
  probing every adapter with per-device timeouts.
- Documentation site (mkdocs-material) from `docs/`.
- Data output: OME-Zarr writer for acquisitions; OME companion for TIFF
  runs (port from the tracking tool).
- Closed-loop tracking plugin over the run loop — `lightsheet-live-
  tracking-tool` becomes this plugin; its microscope backends are
  superseded by the capability layer.
- Laser ablation patterns plugin (port of `viventis_control`'s point
  patterns) once the LS1's components are reachable.
