# Architecture

One control layer, several microscopes, interchangeable tools. The layers
below are the ADRs in one picture; each box names the decision it comes
from.

```
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  Interfaces                       CLI `smc` (reference)   ·   web UI (ADR-0006, proposed)  │
 └──────────────┬───────────────────────────────────────────────────────────┘
                │ derives forms/commands from plugin Params
 ┌──────────────▼───────────────────────────────────────────────────────────┐
 │  Plugins (ADR-0004)   plate registration · find plate · wells · field grid │
 │                       camera↔stage calibration · object detection · timing │
 │                       pure algorithm  ⇄  hardware loop     →  run document │
 └──────────────┬───────────────────────────────────────────────────────────┘
                │ requires capabilities only
 ┌──────────────▼───────────────────────────────────────────────────────────┐
 │  Capabilities (ADR-0003)  XYStage · ZStage · Camera · Autofocus · Turret  │
 │                           Shutter · LightSource · Channels · Properties   │
 │  Microscope facade: roles (profile > core slots > heuristics), safety      │
 │  guards (jog limit, Z↔autofocus interlock, confirmed turret), dry-run      │
 └──────────────┬───────────────────────────────────────────────────────────┘
                │ one core per process
 ┌──────────────▼───────────────────────────────────────────────────────────┐
 │  Micro-Manager core via pymmcore-plus (ADR-0002)                          │
 │   CMMCorePlus  ── C++ adapters: NikonTi2 · NikonTI · PVCAM · Hamamatsu …  │
 │   UniMMCore    ── Python devices (ADR-0007): Zeiss MTB · Viventis PyMCS   │
 │   DemoCamera   ── the simulator: regression bed for everything above      │
 └──────────────────────────────────────────────────────────────────────────┘
        profiles/<microscope>.toml : config, role overrides, limits, quirks
```

## Data flow of a typical session

1. `smc` loads a **profile** → opens the core with its `.cfg` → resolves
   **roles** → exposes **capabilities**.
2. A plugin runs (`smc run find-plate --profile nikon-ti2`): it surveys the
   stage through `XYStage` + `Camera`, fits the plate lattice (pure
   algorithm), and writes `plate` into the **run document**.
3. The next plugin reads `plate`, lets the user choose wells, writes
   `wells` — as a `useq.WellPlatePlan`.
4. An acquisition plugin turns positions into a `useq.MDASequence` and
   `CMMCorePlus.run_mda` executes it; events feed the UI and any
   closed-loop plugin (tracking, status trigger).

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
  cli.py                  the reference interface
  hardware/
    core.py               open/close a core; demo config; install status
    capabilities.py       the Protocols                         (M1)
    roles.py              role resolution                       (M1)
    microscope.py         the facade + safety                   (M1)
    profile.py            TOML profiles                         (M1)
    backends/             mm.py (over CMMCorePlus), zeiss_mtb/, viventis/
  plugins/                registry + built-in plugins           (M3)
  run/                    run document                          (M3)
profiles/                 one TOML per instrument
tests/                    unit/ · simulator (demo) · contracts/ · hardware/
docs/adr/                 decisions
docs/hardware/            what we know about each stand
scripts/github/           labels, milestones, backlog bootstrap
```
