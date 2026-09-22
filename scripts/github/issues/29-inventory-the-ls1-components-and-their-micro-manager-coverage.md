---
title: spike(viventis): inventory the LS1 components and their Micro-Manager coverage (both stands)
labels: [type: spike, area: backend, scope: viventis-ls1, status: needs-hardware, priority: p1]
milestone: M6 — Viventis LS1
---
## Goal
Know what the two LS1s are made of, so they can be driven through their **components** like every other stand — not through the vendor's PyMCS (owner's direction, ADR-0007). Output is knowledge, not code.

## What to inventory, on site, for each LS1
- Controllers and ports: Device Manager / `Get-PnpDevice`, COM ports with VID/PID, PCI/PCIe cards (start from the tracking tool's `tools/audit_windows_hardware.ps1`).
- Components: XYZ stage controller, camera(s) model and interface, laser combiner, scan mirrors / galvo driver, filter changers, piezo, and **what performs the light-sheet timing** (sweep ↔ exposure ↔ laser blanking): DAQ, FPGA, vendor controller?
- The vendor's configuration files under `C:\Viventis\` (component models, ports, timing parameters) and the metadata the acquisition writes.
- For each component: a Micro-Manager adapter that drives it, or "needs a unicore Python device", or "no public protocol".

## Acceptance criteria
- [ ] `docs/hardware/viventis-ls1-components.md`: one table per stand (component · model · interface · adapter or gap), plus the timing-controller finding
- [ ] Differences between the two stands recorded
- [ ] A short follow-up ADR proposing the scope: full control, or stage + camera first
- [ ] PyMCS used only as a reference for how the vendor sequences the hardware
