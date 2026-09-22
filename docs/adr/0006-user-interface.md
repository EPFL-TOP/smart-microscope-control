# 0006 — User interface: CLI first, then a web UI

- **Status**: **proposed** — decision requested from the project owner
- **Date**: 2026-09-22

## Context

The interface shown to a microscope user shapes everything downstream, so it
must be chosen early. Constraints observed in the lab:

- Microscope PCs are **Windows** machines, often reached by **RDP**, some
  without a GPU in the session (Windows Server), where Qt/OpenGL has
  already been painful (napari on the annotation server).
- Users want to **start a run at the microscope and check on it from
  their desk**; several people share one instrument.
- Both predecessor projects converged on **browser dashboards** — Bokeh
  (`nikon-control`) and Panel (`lightsheet-live-tracking-tool`) — and lab
  members know how to operate and deploy them (NSSM service, port,
  `--allow-websocket-origin`).
- The pymmcore-plus ecosystem offers **Qt** building blocks:
  `pymmcore-widgets` 0.12 (stage, snap/live, property browser, MDA, an HCS
  plate-calibration wizard), `napari-micromanager` (maintenance mode),
  `pymmcore-gui` (pre-release). None runs in a browser.
- Every plugin already needs a scriptable, headless entry point for
  unattended overnight runs and for tests.

## Options

| | A — Qt desktop (`pymmcore-widgets`) | B — Web app (**Panel**, on Bokeh) | C — Web app, custom frontend (FastAPI + React/Vue or NiceGUI) | D — CLI only |
|---|---|---|---|---|
| Ready widgets for stage/camera/MDA/plate | **many** | none (we write them) | none | n/a |
| Remote / multi-user | no | **yes** | **yes** | via SSH |
| Runs on GPU-less RDP session | fragile | **yes** | **yes** | yes |
| Team familiarity | low | **high** | low–medium | high |
| Auto-forms from plugin `Params` | via magicgui | via `param`/pydantic → widgets | custom | via typer |
| Live image at 5–10 fps | native | ok (websocket, JPEG/PNG) | ok | n/a |
| Cost to first useful screen | low | medium | high | very low |

## Recommendation

1. **The CLI is the reference interface** (accepted now, independent of the
   rest): every capability and plugin is reachable from `smc …` first. It is
   what tests, scripts and overnight runs use.
2. **The graphical interface is a web application built with Panel**, one
   tab per plugin, forms generated from the plugin's `Params`, live view via
   websocket, run document shown as the pipeline state. It deploys on the
   microscope PC as a service and is used from any browser. This matches the
   lab's deployment reality and skills, and keeps Qt out of the microscope
   PCs.
3. `pymmcore-widgets` stays available as an **optional `[qt]` extra** for a
   local "engineering console" (property browser, hardware wizard) when
   bringing up a new stand — not as the user interface.

Reversibility: because the UI is a thin view over the capability layer and
plugins (ADR-0003/0004), switching B → C later costs the view code only.

## Consequences (if accepted)

- Adds `panel` to a `[ui]` extra; no Qt in the default install.
- Plugins expose progress/stop via `ctx`, never via UI callbacks.
- First UI milestone: connect to a profile, drive the stage, snap/live,
  run the plate-registration plugin — the `/scope` dashboard of
  `nikon-control` re-expressed over the new layer.
