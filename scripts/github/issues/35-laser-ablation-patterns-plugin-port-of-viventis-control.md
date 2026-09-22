---
title: feat(plugins): laser ablation patterns (port of viventis_control) — once LS1 components are reachable
labels: [type: feature, area: plugins, scope: viventis-ls1, status: blocked, priority: p2]
---
## Goal
`viventis_control` (PySimpleGUI over PyMCS) generates ablation point patterns — lines and circles with point count, spacing, pulse count, laser diameter — and drives them. As a plugin it becomes available on any stand with a steerable ablation/photomanipulation device, with the same form and run document as every other tool.

## Context
Blocked on the LS1 component inventory (#29) and on a capability for the ablation device (galvo/SLM/point scanner) that does not exist yet.

## Acceptance criteria
- [ ] Pattern geometry as a pure, tested algorithm (`useq`-style positions)
- [ ] Plugin form generated from `Params`; dry-run draws the pattern on the live view before firing
