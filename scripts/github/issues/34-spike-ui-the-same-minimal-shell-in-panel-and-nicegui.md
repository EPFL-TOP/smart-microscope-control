---
title: spike(ui): the same minimal shell in Panel and in NiceGUI on the demo devices — decide the framework
labels: [type: spike, area: ui, status: needs-decision, priority: p0]
milestone: M5 — Web UI
---
## Goal
Choose the web framework for the shell by looking at two working prototypes rather than by argument (ADR-0006). Time-box: two days per framework, on the demo devices, no styling beyond defaults.

## The minimal shell to build in each
Connect to `demo` → status strip (XY, Z, objective, light, Stop) → live view at 5–10 fps → jog pad → one plugin form generated from a pydantic `Params` model with Run/Stop and progress.

## Compare on
| Criterion | Panel | NiceGUI |
|---|---|---|
| Look and feel with zero styling | | |
| Lines of code for the shell | | |
| `pydantic → form` generation (labels, units, advanced fields) | | |
| Live view fps and CPU on the demo camera | | |
| Tests without a browser | | |
| Windows service deployment; RDP session without GPU | | |
| Team familiarity / maintenance | | |

## Acceptance criteria
- [ ] Both prototypes in `spikes/ui-<framework>/` (not packaged; deleted after the decision)
- [ ] Table above filled with measurements and screenshots in this issue
- [ ] Owner's decision recorded by editing ADR-0006 to `accepted` with the chosen framework
