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

Interfaces between the issues are fixed in
[`docs/design/m2-nikon.md`](design/m2-nikon.md), one section per issue as
its plan is written.

- Port the Nikon stand knowledge from `nikon-control`: adapters, driver
  DLL checks, dynamic vs fixed device lists, TIRF exclusions, PFS
  interlock, turret confirmation.
- `Autofocus` capability with PFS semantics (engaged / in range / locked /
  offset units); `ObjectiveTurret`, `LightSource`, `Channels`.
- Config builder (`smc config build`) from what actually answers.
- Hardware session 1: `smc doctor`, contract tests, stage + PFS on the Ti2.
- Ti-E profile and session.

## M3 — Plate scan to NIS-Elements on the Ti2-E

Re-scoped 2026-10-26 by the owner: one workflow, end to end, on the
Ti2-E in brightfield, before anything else. Interfaces between the issues
are fixed in [`docs/design/m3-plugins.md`](design/m3-plugins.md), written
for the whole workflow at once (§2 onwards) so that the plans can then run
in parallel. Ports are made from `nikon-control@1aad7c9`
(branch `wip/plate-tools-2026-09-18`), with their tests.

1. Plugin API: `Plugin` protocol, registry, `Params`/`Result`, run
   document, `smc run` (done, #18).
2. Calibrate: camera↔stage (phase correlation), plate registration (wells,
   wall touches) and automatic plate finding (lattice fit).
3. Choose wells and scan them at 40x: field grid per well, frames with
   their stage position.
4. Detect single and doublet cells on every frame with the trained
   Faster R-CNN of `nikon-control` (single/doublet/debris), as cells in
   stage µm, de-duplicated across overlapping fields.
5. Count cells per well and type (density).
6. Plan positions: fields of view that hold the most cells (well centre
   optionally excluded), then the shortest route between them.
7. Show the density and the route: a static report first, then the plate
   page of the UI (Panel, ADR-0006), where wells are clicked.
8. Export the positions as a list NIS-Elements reads.
9. Hardware session: a real plate on the Ti2-E, from calibration to NIS.

Paused until this works: M4, M5 (except the plate page), M6, the Ti-E
(#17), the piezo (#135), and the p2 backlog.

## M4 — Zeiss Axio Observer 7

- Per ADR-0007: MTB session owner; unicore Python devices for stage,
  focus, piezo, changers, TL shutter, Colibri; DF2 as a device +
  `Autofocus` implementation; PVCAM camera in the same core.
- Profile, contract tests in a hardware session, port of the phototoxicity
  guards (darken on every exit path).

## M5 — Web UI

- Panel, per ADR-0006 (accepted 2026-10-26 without the spike). The M3
  plate page is the shell's first version; M5 grows it (design:
  `docs/design/m5-web-ui.md`).
- The shell: connect, status strip with Stop, live view, drive, tools rail
  grouped by workflow stage, generated plugin forms, run document as a
  checklist. Identical on every microscope; deployable as a service.
  Issues: #28 (`smc ui`, every plugin, result cards, Snap), #262 (live),
  #263 (drive, with a command per action), #264 (service, Connect).

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
