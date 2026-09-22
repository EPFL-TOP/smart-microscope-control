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

> **Status: bootstrap.** The package installs, the CLI answers, the test
> suite runs against Micro-Manager's simulated microscope on Linux, macOS
> and Windows. No real stand is driven yet. The plan is in
> [`docs/roadmap.md`](docs/roadmap.md); the decisions are in
> [`docs/adr/`](docs/adr/README.md).

## How it fits together

```
interfaces      CLI `smc`  ·  web UI (proposed)
plugins         plate registration · find plate · wells · field grid · detection · …
capabilities    XYStage · ZStage · Camera · Autofocus · Turret · Shutter · LightSource
                Microscope facade: roles + safety guards + dry-run
hardware bus    Micro-Manager core via pymmcore-plus
                C++ adapters (Nikon, cameras) · Python devices (Zeiss MTB, Viventis) · demo
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
| Nikon Ti2-E | Micro-Manager `NikonTi2` + camera adapter | next (knowledge carried over from `nikon-control`) |
| Nikon Ti-E | Micro-Manager `NikonTI` | after Ti2; same roles, different adapter |
| Zeiss Axio Observer 7 | Micro-Manager `PVCAM` camera + Zeiss MTB 2011 as Python devices ([ADR-0007](docs/adr/0007-non-micromanager-hardware.md)) | proposed |
| Viventis LS1 (×2) | PyMCS — to be inventoried | later |

What is known about each stand (adapters, DLLs, power-on order, quirks that
cost days) is collected in [`docs/hardware/inventory.md`](docs/hardware/inventory.md).

## Working on the project

- Every change starts as an issue and lands through a pull request with
  green CI — see [`CONTRIBUTING.md`](CONTRIBUTING.md) and
  [ADR-0008](docs/adr/0008-repository-workflow.md).
- Architecture decisions are records in [`docs/adr/`](docs/adr/README.md).
  Two are **proposed** and waiting for a decision: the user interface
  ([0006](docs/adr/0006-user-interface.md)) and the route to hardware
  without a Micro-Manager adapter
  ([0007](docs/adr/0007-non-micromanager-hardware.md)).
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
backends). Their algorithms are being ported here as plugins; their
hardware knowledge as profiles and tests.

## License

Apache-2.0 — see [`LICENSE`](LICENSE).
