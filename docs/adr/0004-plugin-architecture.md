# 0004 — Tools are plugins over capabilities

- **Status**: accepted
- **Date**: 2026-09-22

## Context

The point of the abstraction layer is interchangeable *tools*: register a
well plate against the stage, find the plate automatically, select wells,
lay a field grid, calibrate camera-to-stage, detect objects in wells,
measure timing, run a tracked time-lapse. `nikon-control` already has
working versions of most of these as modules, and a design note describing
them as seven stages of a pipeline over one *run document*. They must
become units that can be added, replaced and combined without editing the
core — including by other labs.

## Decision

- A **plugin** is a Python object implementing the `smc.plugins.Plugin`
  protocol: a `name`, a `Params` pydantic model, a `requires` tuple of
  capability protocols (ADR-0003), and `run(microscope, params, ctx) ->
  Result`. Results are pydantic models too, so they serialise to JSON.
- Plugins are discovered through the **entry-point group `smc.plugins`**.
  Built-in plugins live in `smc.plugins.*` and register the same way, so
  nothing distinguishes a first-party plugin from a third-party one.
- **Pure algorithms are separated from hardware loops** in every plugin
  (as `plate_find.fit()` is separate from `plate_find.survey_scope()`): the
  algorithm takes arrays and returns numbers, the loop takes a `Microscope`
  and produces arrays. The algorithm is unit-tested with synthetic data; the
  loop is tested on the demo devices.
- Plugins **read and write a run document** (one JSON per experiment,
  `smc.run.RunDocument`) rather than calling each other. Each stage is
  re-runnable alone, inspectable, and a crashed run resumes.
- Plugins **emit progress and accept a stop request** through `ctx`, never
  by printing or by blocking a UI thread. Long loops are generators or check
  `ctx.should_stop()` between steps.
- The CLI exposes every registered plugin as `smc run <name> [--param …]`
  with parameters derived from the `Params` model; a future UI derives its
  forms the same way.
- Plugin outputs that describe *where to image* are `useq` objects
  (`WellPlatePlan`, positions, `MDASequence`) so acquisition needs no glue.

## Alternatives considered

- **A module per tool, imported by name** (the `nikon-control` state):
  works for one author; gives no discovery, no parameter introspection, no
  third-party path.
- **napari plugins / pymmcore-widgets** as the plugin system: ties every
  tool to Qt and a viewer; the lab runs headless microscope PCs and web
  dashboards.
- **A workflow engine** (Snakemake, Prefect): overkill for six stages, and
  the interactive, hardware-in-the-loop steps do not fit a DAG.

## Consequences

- The first plugins are ports of `nikon-control/scope/{plate,plate_find,
  wells,scan,camera_cal,timing}.py`, each in its own PR with its algorithm
  tests carried over.
- A plugin's `requires` is checked before it runs; a stand lacking the
  capability fails fast with a readable message instead of mid-scan.
- Third parties can ship `their-package` with an `smc.plugins` entry point
  and appear in `smc run --list`.
