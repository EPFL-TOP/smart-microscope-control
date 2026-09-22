# 0003 — The abstraction layer: capabilities, roles and profiles

- **Status**: accepted
- **Date**: 2026-09-22

## Context

A tool such as "scan the plate to find the well edges" needs an XY stage
and a camera. It does not care whether the stage is a Nikon `XYStage`, a
`TIXYDrive`, a Märzhäuser Tango behind Micro-Manager, or a Zeiss
`MTBStageAxisX/Y` pair behind a Python bridge. Two things stand between the
tool and that indifference:

1. **Naming.** Micro-Manager identifies devices by *label*, chosen per
   configuration. MMCore has slots for five *roles* (Camera, XYStage, Focus,
   AutoFocus, Shutter) and none for the others a tool needs: the objective
   turret, the autofocus *offset*, the light path, the light source.
   `nikon-control` solved this with type-first role resolution — the Ti-E
   reports three devices typed `Stage`, and only one is the focus drive.
2. **Semantics.** The same operation means different things per stand.
   Moving Z on a Nikon disables PFS (micro-manager #1815); on a Zeiss,
   Definite Focus corrects through the focus drive *or* the piezo; a PFS
   offset unit is not a micron; a Ti2 turret rotation under a plate can
   crash a 40× into glass. A tool must not need to know any of this — and
   must not be able to bypass the guard.

## Decision

Three concepts, three modules under `smc.hardware`:

### Capabilities (`smc.hardware.capabilities`)

Small, typed `Protocol`s — the *only* things a tool may depend on:

| Capability | Core operations | Notes |
|---|---|---|
| `XYStage` | `position_um()`, `move_to_um(x, y)`, `move_by_um(dx, dy)`, `wait()`, `limits_um()` | jog guard built in |
| `ZStage` | `position_um()`, `move_to_um(z)`, `move_by_um(dz)` | interlocks with `Autofocus` |
| `Camera` | `snap()`, `exposure_ms`, `image_shape()`, `pixel_size_um()` | pixel size may be *unknown* (0), never guessed |
| `Autofocus` | `engaged`, `locked`, `in_range`, `engage()`, `disengage()`, `offset`, `focus_by(delta)` | PFS, Definite Focus 2, demo AF |
| `ObjectiveTurret` | `labels()`, `current()`, `select(label, *, confirm=True)` | selection is an exceptional, confirmed event |
| `Shutter` | `is_open`, `set_open()` | |
| `LightSource` | `intensity`, `set_intensity()`, `on/off` | the "how bright" knob, found by property hints |
| `Channels` | `names()`, `current()`, `select(name)` | Micro-Manager config-group presets |
| `Properties` | `get/set(device, prop)`, `describe()` | the escape hatch for everything else |

Every method takes and returns **explicit physical units in the name**
(`_um`, `_ms`, `_deg`). Anything in device-native units (a PFS offset) is
named for what it is and never converted implicitly.

### Roles (`smc.hardware.roles`)

A `Microscope` facade resolves which loaded device fills each role, in this
order of authority:

1. **The profile's explicit mapping** (below) — a human said so.
2. **MMCore's own role slots** (`getXYStageDevice()` …) — the config said so.
3. **Type-first heuristics with name tie-breaks and exclusion lists**
   (generalised from `nikon-control/scope/stand.py`) — with the candidates
   *reported*, because a silent wrong pick reads exactly like broken
   hardware.

`Microscope.require(XYStage)` returns the capability or raises a
`CapabilityMissing` error naming the role and what *is* available.

### Profiles (`profiles/*.toml`)

One TOML file per instrument, committed to the repository: the Micro-Manager
config to load (or how to build one), role overrides, **safety limits**
(maximum jog, soft Z limits, turret confirmation), per-objective pixel
sizes, vendor quirks (PFS offset units, power-on order), and defaults such
as the plate holder. The profile is the single place where "this stand is
special" is allowed to be written.

### Behavioural guarantees live in the layer, not in the tool

- Moving Z while an autofocus is engaged suspends and re-engages it.
- A relative XY jog larger than the profile's limit is refused unless forced.
- Selecting an objective requires `confirm=True`.
- Reads never raise for a missing optional device; they return `None` and
  the facade's `state()` reports every problem in one pass.
- A **dry-run mode** makes every mutating call log instead of move.

## Alternatives considered

- **Tools call `CMMCorePlus` directly.** Fast to start, and exactly how
  device names, unit mistakes and PFS-off-overnight leak into every tool.
- **Our own device classes, one per vendor, no Micro-Manager underneath.**
  Loses the adapter library and the simulator (ADR-0002).
- **pymmcore-plus `Device` objects as the abstraction.** They are per
  *device*, not per *capability*, and carry no safety semantics.

## Consequences

- A tool that works on the demo devices works on any stand whose profile
  resolves the roles it requires — that is the definition of "portable".
- Every new stand costs a profile and, if its devices are unusual, new
  heuristics with tests. Every new capability costs a protocol, a
  Micro-Manager implementation, a demo test and a contract test (ADR-0005).
- The facade is the *only* code allowed to import `pymmcore_plus` for
  control purposes (`ruff` can enforce this with an import ban later).
