# smart-microscope-control

One control layer for several microscopes, and tools that work on all of
them.

The lab runs two Nikon inverted stands (Ti2-E, Ti-E), a Zeiss Axio
Observer 7 and two Viventis LS1 light sheets. Each has its own software.
This project puts one Python layer over all of them — built on
[Micro-Manager](https://micro-manager.org) through
[pymmcore-plus](https://pymmcore-plus.github.io/pymmcore-plus/) and
[useq-schema](https://pymmcore-plus.github.io/useq-schema/) — so that a
*tool* written once (find the edges of a well plate, locate objects in
wells, run a tracked time-lapse) runs on any stand that provides the
capabilities it needs.

## Where the project stands

As of 2026-11-19. What was measured at a stand is in
[`docs/hardware/`](docs/hardware/nikon-ti2.md), each fact with its session;
the milestones and what is open are in [`docs/roadmap.md`](docs/roadmap.md).

- **Nikon Ti2-E: driven at the stand.** Five sessions (0, 1, 2, 3 and 5)
  brought the stand up through the layer, moved the stage, read and locked
  the PFS, turned the turret, imaged with the Prime 95B and ran the
  contracts. The **plate chain** (camera calibration, registration of the
  plate from its walls, the scan plan and the scan, cells, density, the route,
  the report) is built, and it ran end to end on the plate page in session 5,
  on a plate with beads and no cells: a 10× scan with PFS took 218 of 218
  fields in 116.3 s, and the export's **position list loads in NIS-Elements**.
  What is not done: a 40× lock from the floor in a well with water (#539), and
  the chain on a plate with cells.
- **Zeiss Axio Observer 7: built, not run at the stand.** The Zeiss MTB
  components and Definite Focus 2 are Python devices in the layer, tested on a
  fake MTB; they have never been loaded on the stand.
- **Nikon Ti-E: not started.** Its profile is written from the adapter's
  device list and says it is assumed; no session has been held.
- **Viventis LS1: not started.** The components of the two stands are not
  inventoried.
- **Where it goes.** smc is to take over the lab's experiment from
  NIS-Elements in four stages, from "a collaborator's algorithm chooses the
  fields" to "smc runs the experiment"
  ([`docs/roadmap.md`](docs/roadmap.md#from-nis-elements-to-smc-for-the-experiment)).
  The collaborator's kit (a dataset exported from a scan, a loader that needs
  no microscope, the contract of a selection algorithm) is designed in
  [`docs/design/algorithms.md`](docs/design/algorithms.md); its guide comes
  with issue #554 and does not exist yet.

## How it fits together

```
interfaces      CLI `smc`  ·  the plate page `smc ui` (web, ADR-0006)
plugins         plate registration · find plate · wells · scan · cells · density · route · NIS export · …
capabilities    XYStage · ZStage · Camera · Autofocus · Turret · Shutter · LightSource
                Microscope facade: roles + safety guards + dry-run
hardware bus    Micro-Manager core via pymmcore-plus
                C++ adapters (Nikon, cameras) · Python devices (Zeiss MTB; Viventis planned) · demo
profiles        one TOML per instrument: config, role overrides, limits, quirks
```

Details and the reasoning behind each layer: [`docs/architecture.md`](docs/architecture.md).

## Quick start (development, no microscope)

```sh
git clone git@github.com:EPFL-TOP/smart-microscope-control.git
cd smart-microscope-control
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
mmcore install --test-adapters      # Micro-Manager's simulated microscope (any OS)
smc doctor                          # can this machine drive the simulator?
smc devices                         # the demo devices and the role each one fills
smc stage jog 100 0 --dry-run       # what a jog would send; the stage stays put
smc objective list                  # the turret's objectives; `smc objective set <label> --confirm` turns it
smc objective offset                # each objective's XY and Z offset from the profile's reference (m2 §13)
smc autofocus engage                # engage, disengage or get the autofocus
smc autofocus offset --floor        # the offset to its floor, before a first lock (m2 §12.10)
smc light get                       # the light source: get, on, off
smc snap --out frame.tif            # one frame from the simulated camera
smc run plate-finding -p demo-sample --run-doc run.json   # the plate-scan tools, on a synthetic plate (docs/plugins/README.md)
smc ui -p demo-sample --run-doc run.json                  # the plate page in a browser (needs the ui extra: pip install -e ".[ui]")
smc discover                        # read-only survey of this PC (docs/hardware/microscope-pc-setup.md)
pytest                              # the suite runs against the simulator
pre-commit install                  # lint + format on every commit
```

`mmcore install` without `--test-adapters` fetches a full Micro-Manager
nightly build with every device adapter (Windows and Intel macOS only) —
that is what a microscope PC needs.

## The microscopes

| Microscope | Route | Status |
|---|---|---|
| Micro-Manager demo devices | `DemoCamera` adapter | **the regression bed** — every tool must run here |
| Nikon Ti2-E | Micro-Manager `NikonTi2` + the Prime 95B on `PVCAM` | **driven at the stand** in five sessions; the plate chain ran end to end in session 5 ([Ti2-E notes](docs/hardware/nikon-ti2.md)) |
| Nikon Ti-E | Micro-Manager `NikonTI` | not started: an assumed profile, never loaded on a stand |
| Zeiss Axio Observer 7 | Micro-Manager `PVCAM` camera + Zeiss MTB 2011 as Python devices ([ADR-0007](docs/adr/0007-non-micromanager-hardware.md)) | **built on a fake MTB, not yet run at the stand** ([Zeiss notes](docs/hardware/zeiss-observer7.md)) |
| Viventis LS1 (×2) | Direct control of its components (Micro-Manager adapters or Python devices), never through the vendor's PyMCS; on-site inventory first ([ADR-0007](docs/adr/0007-non-micromanager-hardware.md)) | not started: the components are not inventoried |

What is known about each stand (adapters, DLLs, power-on order, quirks that
cost days) is collected in [`docs/hardware/inventory.md`](docs/hardware/inventory.md).

## Working on the project

- Every change starts as an issue and lands through a pull request with
  green CI — see [`CONTRIBUTING.md`](CONTRIBUTING.md) and
  [ADR-0008](docs/adr/0008-repository-workflow.md).
- Architecture decisions are records in [`docs/adr/`](docs/adr/README.md).
  All nine are **accepted**, among them the user interface
  ([0006](docs/adr/0006-user-interface.md), accepted 2026-10-26) and the route
  to hardware without a Micro-Manager adapter
  ([0007](docs/adr/0007-non-micromanager-hardware.md), accepted 2026-11-16);
  0005 and 0008 are amended in part by
  [0009](docs/adr/0009-light-ci-full-nightly.md), the light CI.
- The AI coding agent follows [`CLAUDE.md`](CLAUDE.md); the same rules
  apply to humans.

## Lineage

This project consolidates two earlier repositories of the lab:
[`nikon-control`](https://github.com/EPFL-TOP/nikon-control) (role-based
Micro-Manager control of the Nikon stands, plate registration and
automatic plate finding, well selection, field scanning, camera↔stage
calibration, throughput measurement) and
[`lightsheet-live-tracking-tool`](https://github.com/EPFL-TOP/lightsheet-live-tracking-tool)
(closed-loop tracking with Viventis, Zeiss MTB and Micro-Manager
backends). Their algorithms are ported here as plugins, with their tests, and
their hardware knowledge as profiles and tests. The plate chain is ported; the
tracking runner (#32) and the laser patterns (#36) still wait.

## License

Apache-2.0 — see [`LICENSE`](LICENSE).
