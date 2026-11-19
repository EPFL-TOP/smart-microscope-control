# Architecture

One control layer, several microscopes, interchangeable tools. The layers
below are the ADRs in one picture; each box names the decision it comes
from.

```
 ┌────────────────────────────────────────────────────────────────────────────┐
 │  Interfaces   CLI `smc` (reference)  ·  the plate page `smc ui` (ADR-0006) │
 └──────────────┬─────────────────────────────────────────────────────────────┘
                │ derives forms/commands from plugin Params
 ┌──────────────▼─────────────────────────────────────────────────────────────┐
 │  Plugins (ADR-0004)   camera calibration · plate registration · find plate │
 │                       scan plan · scan · cells · density · route · timing  │
 │                       report · NIS export · objective offsets · …          │
 │                       pure algorithm  ⇄  hardware loop     →  run document │
 └──────────────┬─────────────────────────────────────────────────────────────┘
                │ requires capabilities only
 ┌──────────────▼─────────────────────────────────────────────────────────────┐
 │  Capabilities (ADR-0003)  XYStage · ZStage · Camera · Autofocus · Turret   │
 │                           Shutter · LightSource · Channels · Properties    │
 │  Microscope facade: roles (profile > core slots > heuristics), safety      │
 │  guards (jog limit, Z↔autofocus interlock, confirmed turret), dry-run      │
 └──────────────┬─────────────────────────────────────────────────────────────┘
                │ one core per process
 ┌──────────────▼─────────────────────────────────────────────────────────────┐
 │  Micro-Manager core via pymmcore-plus (ADR-0002)                           │
 │   CMMCorePlus  ── C++ adapters: NikonTi2 · NikonTI · PVCAM · Hamamatsu …   │
 │   UniMMCore    ── Python devices (ADR-0007): Zeiss MTB (built, on a fake   │
 │                   MTB only) · Viventis LS1 (planned)                       │
 │   DemoCamera   ── the simulator: regression bed for everything above       │
 └────────────────────────────────────────────────────────────────────────────┘
        profiles/<microscope>.toml : config, role overrides, limits, quirks
```

## Data flow of a typical session

1. `smc` loads a **profile** → opens the core with its `.cfg` → resolves
   **roles** → exposes **capabilities**.
2. A plugin runs (`smc run plate-finding --profile nikon-ti2 --run-doc
   run.json`, or `plate-walls`, which registers the plate from images of
   its walls): it moves the stage through `XYStage` + `Camera`, fits the plate
   (a pure algorithm), and writes the slot `plate` into the **run document**.
3. The next plugin reads `plate`. `scan-plan` takes the wells to scan (a
   `useq.WellPlatePlan`) and plans the fields in each; `scan` visits them and
   writes one frame and one record, with its stage position, for each.
4. `cells`, `density` and `route` read the scan and choose the fields to image
   next, in an order; `report` draws the result and `nis-export` writes the
   positions as the list NIS-Elements loads. NIS-Elements runs the experiment
   from that list today.
5. Not built yet: smc's own loop over a position list (#556), its output
   format (#31) and decisions taken during an acquisition (#32). Its engine,
   smc's own loop or pymmcore-plus's MDA (`useq.MDASequence`,
   `CMMCorePlus.run_mda`) behind `smc.hardware`, is to be decided in an ADR.
   The steps by which these take over the experiment from NIS-Elements are in
   [`docs/roadmap.md`](roadmap.md#from-nis-elements-to-smc-for-the-experiment).

## Glossary

- **Role**: what a device is *for* (`xy_stage`, `focus`, `autofocus`,
  `autofocus_offset`, `objective_turret`, `camera`, `shutter`,
  `light_source`, `light_path`, `filter_turret`).
- **Capability**: the typed interface a tool programs against; implemented
  over one or more roles.
- **Profile**: the per-instrument TOML that names the config and the
  exceptions.
- **Run document**: the per-experiment JSON that plugins read and write.
- **DIV**: Micro-Manager device interface version; adapters and core must
  match.

## Source map

```
src/smc/
  cli.py                  the reference interface (devices, stage, z, objective,
                          autofocus, light, snap, run, ui, presets, config, doctor)
  drive.py                the drive actions, written once for the CLI and the page
  plates.py, plates.toml  the lab's own plate definitions, registered with useq
  discovery/              the read-only inventory of a PC, in throwaway cores
  hardware/
    core.py               open/close a core; demo config; install status
    capabilities.py       the Protocols                         (M1, M2)
    roles.py              role resolution                       (M1)
    microscope.py         the facade                            (M1)
    safety.py             the guards, and the executor every call goes through
    profile.py            TOML profiles (profile_write.py: the [objectives] block)
    channels.py           channel presets
    backends/             mm.py (over CMMCorePlus; UniMMCore when the .cfg has
                          #py lines, ADR-0007), nikon.py (what is known about
                          the two Nikon stands), mm_stabilizer.py (a one-shot
                          focus stabilizer as an Autofocus), zeiss_mtb/ (the MTB
                          components as unicore devices, over a fake MTB; M4)
  plugins/                api.py, registry.py, params.py, runner.py   (M3)
                          one package per tool: camera_calibration, plate_registration,
                          plate_walls, plate_finding, plate_check, objective_offsets,
                          scan_plan, scan, cell_focus, cells, density, route, timing,
                          report, nis_export
                          goto_well.py, plate_orientation.py, stage_report.py
                          geometry.py   the camera matrix: pixels <-> stage um
                          slots/        the models of the plate-scan results
  run/
    document.py           the run document: stages, slots, staleness  (M3)
    files.py              atomic writes, tokens and run-folder paths  (M3)
  testing/                FakeCore, the fake MTB, the demo-sample stand, fixtures
  ui/                     the plate page, a view over plugins: controller.py (no Panel),
                          plate_page.py (Panel; the `ui` extra), guide.py (what to
                          do next), frames.py                         (M3, ADR-0006)
profiles/                 one TOML per instrument
tests/                    unit/ · simulator (demo) · contracts/ · hardware/
docs/adr/                 decisions
docs/design/              the contracts between issues, one document per milestone
docs/hardware/            what we know about each stand
docs/plugins/, docs/ui/   how to use each tool, and the plate page
scripts/github/           labels, milestones, backlog bootstrap
```

Not on `main` when this was written: `src/smc/data/` and `src/smc/selection/`,
the collaborator's kit (a loader with no hardware, selectors chosen by name;
designed in [`docs/design/algorithms.md`](design/algorithms.md), built by the
issues #564 to #568), and a Viventis backend (M6, not started).
