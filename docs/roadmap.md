# Roadmap

Milestones are GitHub milestones; the issues under each are created by
`scripts/github/bootstrap.py` from `scripts/github/issues/`. Order within
a milestone is roughly dependency order. Nothing here is a date: the lab
gets microscope time when it gets it.

## Where the project stands

As of 2026-11-19. What was measured at a stand is in
[`docs/hardware/`](hardware/nikon-ti2.md) with its session; the sessions
curated there are 0, 1, 2, 3 and 5. A row says what is built and what is
not, and the issues that are still open on it.

| Milestone | State | Still open |
|---|---|---|
| M0 Bootstrap | Done. ADR-0006 (accepted 2026-10-26) and ADR-0007 (accepted 2026-11-16) are decided. | Protecting `main` (#2, the owner's). |
| M1 The stage, on the simulator | Done: capabilities, roles, the `Microscope` facade, profiles, contracts, `FakeCore`. | Leftovers of its issues only (`fix` and `chore`). |
| M2 Nikon Ti2 | **Ti2-E**: driven at the stand in sessions 0, 1, 2, 3 and 5 (stage, PFS, turret, camera, contracts). **Ti-E**: a profile written from the adapter's fixed device list, never loaded on a stand (its header says *assumed*). | The Ti-E and its first session (#17); the 40× lock from the floor in a well with water (#539). |
| M3 Plate scan to NIS-Elements on the Ti2-E | Built, and run **end to end at the Ti2-E in session 5**: calibration, registration from the walls, a 10× scan with PFS, the route, a position list that NIS-Elements loads. The plate held beads, not cells. | Cells on a real scan, and the cell model on 40× frames (#342, #343); the chain on a real plate with cells (#198); the offset between objectives (#548). |
| M4 Zeiss Axio Observer 7 | The MTB session owner, the devices and Definite Focus 2 are built and tested on a fake MTB (#25, #26). **Never loaded at the stand.** | The first session at the stand (#27); Zeiss follow-ups (#559, #583, #584). |
| M5 Web UI | The plate page exists (`smc ui`, Panel): connect, drive, live, every tool, its results. It has been used at the stand since session 2. | The page as one guided scan (#431) and the leftovers of the sessions (#592). |
| M6 Viventis LS1 | Not started. The components of the two stands are not inventoried. | The inventory (#29). |

The collaborator's kit for selection algorithms (a dataset, a loader with no
hardware, the contract of a selector) is designed in
[`docs/design/algorithms.md`](design/algorithms.md). Its loader is built
(`smc.data`, #564); the rest is built by issues #565 to #568. The way from
these pieces to smc running the experiment itself is the section
[*From NIS-Elements to smc for the experiment*](#from-nis-elements-to-smc-for-the-experiment).

## M0 — Bootstrap *(this repository as it stands)*

- Package, CLI (`smc doctor`), simulator tests, CI on three OSes,
  pre-commit, ADRs 0001–0008, backlog. → **PR #1**
- Owner: protect `main` (#2); ADR-0006 (UI) and ADR-0007 (non-MM hardware)
  are decided.

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

Paused while this was built: M4, M5 (except the plate page), M6, the Ti-E
(#17), the piezo (#135), and the p2 backlog. M3's chain has run end to end at
the stand (session 5); M4's code was built on a fake MTB in the meantime
(#25, #26) and the rest stayed paused. What the chain leads to next is the
section *From NIS-Elements to smc for the experiment*.

## M4 — Zeiss Axio Observer 7

- Per ADR-0007: MTB session owner; unicore Python devices for stage,
  focus, piezo, changers, TL shutter, Colibri; DF2 as a device +
  `Autofocus` implementation; PVCAM camera in the same core.
- Profile, contract tests in a hardware session, port of the phototoxicity
  guards (darken on every exit path).
- Design: [`docs/design/m4-zeiss.md`](design/m4-zeiss.md) (#25's interfaces;
  #26 adds Definite Focus 2).

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

## From NIS-Elements to smc for the experiment

On the Ti2-E the lab's experiment runs in NIS-Elements. smc already writes a
list of positions that NIS-Elements loads (session 5). The aim, set by the
owner on 2026-11-18, is to take the experiment over **in four stages**, each
one usable by itself: smc chooses the fields, then NIS-Elements runs from
smc's list, then the functions the experiment needs move from NIS-Elements
into smc one at a time, then NIS-Elements is no longer needed for it. A stage
is done when its proof can be shown at a stand or on a laptop; there is no
date. The owner decides that a stage is done.

Throughout, the rules of the project hold for every function that moves into
smc: the stand is reached only through `smc.hardware`, its guards (STOP, the
jog limit, the Z and autofocus interlock, the confirmed turret) apply to every
move, and only one program holds the stand at a time, so the page's
*Connect* and *Disconnect* hand it from smc to NIS-Elements and back
([`docs/hardware/inventory.md`](hardware/inventory.md), *Both Nikon stands*).

### Stage 1 — A 40× scan, and a collaborator chooses the fields offline

- **What it is.** A 40× scan of a plate with cells, whose every frame carries
  its stage position in its record (the readback at the frame's centre, its
  well, its place in the well, Z, the focus state and the PFS offset). A
  second developer, on a laptop with no microscope, works on that scan and
  writes the algorithm that chooses the fields of view.
- **Built.** The scan and its records: at the Ti2-E in session 5 a 40× scan
  of one well with PFS off made 25 of 25 fields in 10.2 s, and its records
  carry `well_dx_um` and `well_dy_um`
  ([session 5](hardware/nikon-ti2.md#scans-and-the-page)). The kit the
  collaborator works with is designed in
  [`docs/design/algorithms.md`](design/algorithms.md); its loader is on `main`
  (#564): `smc.data` reads a scan's run folder with no Micro-Manager
  (`load_scan`, `Scan`, `cells_table`) and `smc dataset show` prints what it
  holds.
- **Open.** The rest of the kit: the offline runs (#566), the dataset's export
  (#565), the contract of a selector (#567), the drawing of what it chose
  (#568), then the guide (#554). The data: the scans of the stand session of
  2026-11-24 as three datasets, the 10× scan of four wells and the whole-well
  40× tilings of C4 and C3 (#595; [the list](algorithms/datasets.md)), on the
  share once the owner has copied them.
  The first task, the algorithm itself (#570). One question is the owner's:
  whether the selection reads the 10× scan and the 40× scan only judges it
  (as #570 says), or reads the 40× scan itself.
- **Done when**, both:
  1. the collaborator loads the dataset of a 40× scan with cells on a laptop
     with no Micro-Manager, following the guide only, and sees every frame
     with its stage position and its detections;
  2. their selector runs on it offline and the fields it chose, drawn well by
     well, have been looked at with the owner.

### Stage 2 — smc makes the list of fields, NIS-Elements runs the experiment

- **What it is.** The fields come from smc: `route` takes a selector by name
  (the collaborator's, or today's rule), `nis-export` writes the positions
  as the list NIS-Elements 6.20 loads, and NIS-Elements visits them and runs
  the experiment with its own tools.
- **Built.** `route` with today's rule (the fields with the most cells, then
  the shortest path) and `nis-export` ([`route`](plugins/route.md),
  [`nis-export`](plugins/nis-export.md)). At session 5 the export's positions
  load in NIS-Elements; whether beads were in view at those points was not
  settled ([session 5](hardware/nikon-ti2.md#scans-and-the-page)).
- **Open.** A selector chosen in `route` by a parameter (#567) and on the
  plate page (#569) are built: the page lists the installed selectors and
  orderers, shows a chosen one's parameters as fields, keeps them in a preset
  and shows a stop's score on the map and in the report; what is open is the
  choice made at a stand, and the owner's wiring issue that carries a selector
  such as #570's there. When the fields are chosen on a 10× scan and
  imaged at 40×, the offset between the two objectives decides where they
  land: `objective-offsets` read three wells with a spread of 10.89 µm
  against a bound of 2.25 µm, so it wrote nothing (#548,
  [session 5](hardware/nikon-ti2.md#scans-and-the-page)).
- **Done when**, at a stand, all of:
  1. a list made by smc with a selector other than the default is loaded by
     NIS-Elements;
  2. NIS-Elements, sent to the positions of the list, shows the chosen
     object in the field at the positions the owner checks by eye, and the
     geometry of the list (its axes and its Z) is checked at every visit;
  3. an experiment runs to its end in NIS-Elements from that list.

### Stage 3 — The functions the experiment needs are built in smc, one at a time

- **What it is.** Each function that the experiment takes from
  NIS-Elements is built in smc, tested first on the demo stand with a sample
  (which needs no microscope), then run at the stand against NIS-Elements
  doing the same thing. The functions move one at a time, so that
  NIS-Elements stays the fallback for the others.
- **Built.** The pieces such a function stands on: `scan` (a loop over fields
  with the autofocus handling, STOP, a checkpoint and a resume),
  `route`, `timing`, and the guards of the layer.
- **Open.** The functions named so far:
  - **an acquisition loop over the positions**, which goes to each, focuses,
    acquires, writes and comes back for the next time point (#556). Its
    engine, smc's own loop or pymmcore-plus's MDA behind `smc.hardware`, is
    an architecture decision still to be taken in an ADR;
  - **its output**, where the frames and their records land and in what
    format (#31);
  - **decisions during the acquisition**, which a selection or a tracker
    takes between two time points (#32).

  Two more open issues belong here only if the experiment uses them: the
  Prime 95B's post-processing as NIS-Elements offers it (#134) and a Z-stack
  with the Ti2-E's piezo (#135).
- **Done when**, for each function, all of:
  1. it passes its tests on the demo stand with a sample, and every move it
     makes goes through the layer's guards;
  2. at the stand, on the same sample, it does what NIS-Elements does for the
     same positions (the same visits, in the same order, the frames on the
     same objects), and the owner has compared the two and filed the
     difference in a hardware session report;
  3. its issue says what NIS-Elements still does that smc does not.

  The stage is done when every function the experiment uses is done.

### Stage 4 — smc runs the experiment

- **What it is.** The lab's experiment runs from smc alone, from the plate's
  registration to the saved frames; NIS-Elements is not opened for it.
- **Open.** Everything in stages 1 to 3 that the experiment uses, then one
  run of the whole experiment.
- **Done when**, at the Ti2-E, both:
  1. one whole experiment of the lab runs from smc alone, with the layer's
     guards on;
  2. its frames and records go through the collaborator's analysis as those
     of the experiment run in NIS-Elements do, and the two are compared.

  This stage ends the experiment's need for NIS-Elements. It does not say
  that NIS-Elements leaves the PC.

## Backlog (any time)

- `smc discover` follow-ups: suggested profile skeleton from an inventory;
  probing every adapter with per-device timeouts.
- Documentation site (mkdocs-material) from `docs/`.
- Data output: OME-Zarr writer for acquisitions; OME companion for TIFF
  runs (port from the tracking tool); #31, stage 3 above.
- Closed-loop tracking plugin over the run loop — `lightsheet-live-
  tracking-tool` becomes this plugin; its microscope backends are
  superseded by the capability layer; #32, stage 3 above.
- Laser ablation patterns plugin (port of `viventis_control`'s point
  patterns) once the LS1's components are reachable.
