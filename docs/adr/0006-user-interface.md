# 0006 — User interface: one simple interface for every microscope

- **Status**: **proposed** — revised 2026-09-22 with the owner's principles; framework decision pending a spike
- **Date**: 2026-09-22
- **Issue**: #3

## Context

What disturbs users most today is not any single program — it is that
**every microscope has a different one**: ZEN on the Zeiss, NIS-Elements on
the Nikons, the vendor's software plus home-made scripts on the LS1s. A
user trained on one stand starts over on the next, and every tool the lab
writes exists once per system. The owner's direction (2026-09-22): the
interface must be **as simple as possible**, **identical on every
microscope**, and must **scale to many plugins without becoming
complicated**. We must not build another complex application.

Constraints observed in the lab:

- Microscope PCs are **Windows** machines, often reached by **RDP**, some
  without a GPU in the session, where Qt/OpenGL has already been painful.
- Users start a run at the microscope and check on it from their desk;
  several people share one instrument.
- Both predecessor projects converged on browser dashboards (Bokeh, Panel)
  and the lab knows how to deploy them as a Windows service.
- The pymmcore-plus ecosystem offers Qt building blocks
  (`pymmcore-widgets`, `napari-micromanager` in maintenance mode,
  `pymmcore-gui` pre-release). None runs in a browser.
- Every plugin already needs a scriptable, headless entry point for
  unattended runs and for tests.

## Decision (proposed)

### Principles — these are the decision

1. **One interface, every microscope.** The same screens, controls,
   vocabulary and plugins on the Ti2-E, Ti-E, Observer 7 and both LS1. The
   vendor appears exactly twice: when choosing the microscope, and inside
   its profile. A control whose capability a stand lacks is shown disabled
   with the reason — never replaced by a different control.
2. **Radically simple.** The common tasks — connect, look, move, register a
   plate, run a tool, start an acquisition, stop — are done without
   training. One screen does one thing. Defaults are safe. Advanced
   settings exist but sit behind progressive disclosure. A feature that
   needs a manual is not finished.
3. **Scales with plugins without growing.** Plugins plug into a fixed
   **shell** with one presentation contract: a name, a one-line purpose, a
   form generated from its `Params`, Run/Stop, progress, a result card and
   an entry in the run document. A plugin cannot add a new *kind* of
   screen; the shell owns layout and navigation. The twentieth plugin adds
   a list entry, not a menu tree.
4. **Safety is visible.** What is moving, what light is on, the autofocus
   state and a Stop button are always on screen, on every page.
5. **The UI is a view.** Every UI action has a CLI equivalent and goes
   through the same capability layer (ADR-0003). Nothing exists only in the
   UI, so the CLI stays the reference interface and tests need no browser.

### The shell

```
┌ Connect ─────────────────────────────────────────────────────────────────┐
│  [ Ti2-E ] [ Ti-E ] [ Observer 7 ] [ LS1 A ] [ LS1 B ] [ Demo ]           │
└──────────────────────────────────────────────────────────────────────────┘
┌ Status: Ti2-E · XY 12 340, −4 210 µm · Z 1 203.4 · 40× · PFS locked · light OFF · [ STOP ] ┐
├────────────┬──────────────────────────────────────┬──────────────────────┤
│ Tools      │                                      │ Drive                │
│ Set up     │                                      │  jog pad  Z ± AF ⟳   │
│  · Plate   │            live view                 │  objective ▾ light ▾ │
│  · Wells   │      (scale bar, crosshair)          ├──────────────────────┤
│  · Camera  │                                      │ Tool: Find plate     │
│ Find       │                                      │  plate type   96-well│
│  · Objects │                                      │  step (µm)   [auto] │
│ Acquire    │                                      │  ▸ advanced          │
│  · Scan    ├──────────────────────────────────────┤  [ Run ]  progress   │
│  · Movie   │ Run: plate ✓  wells ✓  scan ·  detect ·│  result card         │
│ Watch      │                                      │                      │
└────────────┴──────────────────────────────────────┴──────────────────────┘
```

Tools are grouped by workflow stage (*Set up · Find · Acquire · Watch*),
the run document is the checklist at the bottom of the view, and every
tool opens in the same right-hand panel. This is the whole UI; it does not
grow new regions.

### Technology

- **A web application**, served from the microscope PC (or a lab server)
  and used from any browser — accepted: it is the only route that gives the
  same interface on every PC and from the desk, and keeps Qt off the
  microscope PCs.
- **The framework is chosen by a time-boxed spike, not by argument.** Two
  candidates are credible for a *simple, polished* shell in pure Python:
  **Panel** (Bokeh; what the lab already knows and deploys) and **NiceGUI**
  (FastAPI + Vue/Quasar components; modern look out of the box, pytest
  fixtures that drive the UI without a browser). Build the same minimal
  shell in each on the demo devices — connect, status strip, live view,
  jog, one generated plugin form with Run/Stop — and compare on: look and
  feel with zero styling, lines of code, `pydantic → form` generation, live
  view at 5–10 fps, testability without a browser, Windows-service
  deployment. The owner decides by looking at both.
- `pymmcore-widgets` remains available as an **optional `[qt]` extra** for
  an engineering console when bringing up a stand (property browser,
  hardware wizard) — never the user-facing interface.

## Alternatives considered

- **Qt desktop composed from `pymmcore-widgets`**: many ready widgets, but
  desktop-only, fragile on GPU-less RDP sessions, and a Qt install per
  microscope PC that drifts — the fragmentation we are trying to end.
- **Plugins inside the vendor software** (NIS macros, ZEN extensions,
  PyMCS scripts): three implementations of every tool and three different
  user experiences. Rejected on principle 1.
- **napari plugin**: viewer-centric, heavy, and its Micro-Manager plugin is
  in maintenance mode.
- **A custom JavaScript frontend** (React/Vue over FastAPI): maximum control
  at maximum cost and lowest team familiarity; revisit only if the spike
  shows both Python frameworks failing the simplicity bar.

Reversibility: the UI is a thin view over capabilities and plugins, so
changing the framework later costs the view code only — which is why the
principles, not the framework, are the durable part of this record.

## Consequences (if accepted)

- Every UI change is reviewed against the five principles; "does this add a
  new kind of screen?" is a blocking question in PR review.
- Plugins expose progress/stop through `ctx`, never through UI callbacks,
  and their `Params` models carry the metadata the form generator needs
  (labels, units, which fields are advanced).
- The spike is an M5 issue with priority p0; the M5 UI skeleton is blocked
  on it.
- The Bokeh/Panel dashboards of `nikon-control` and the tracking tool are
  design input for the shell, not code to port.
