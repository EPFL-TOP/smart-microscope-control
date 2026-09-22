---
title: feat(discovery): smc discover — OS-level hardware inventory cross-referenced with Micro-Manager adapters
labels: [type: feature, area: core, priority: p2]
---
## Goal
Find out what is plugged into this PC without tracing cables: serial ports with VID/PID (pyserial), USB devices (Windows PnP / `system_profiler` / `lsusb`), PCI cards (the Zeiss `MicoIf` FPGA, Photometrics PCIe), cross-referenced with a small vendor table and the installed adapters → an inventory JSON and a suggested profile skeleton, with probes (load one device in a throwaway core) to confirm.

## Context
`nikon-control/scope/discover.py` (adapter scan, probe, `deep_check` with WinError decoding); the tracking tool's `tools/probe_serial_devices.py` and `audit_windows_hardware.ps1`.

## Acceptance criteria
- [ ] `smc discover [--probe]` works on macOS/Linux/Windows (degrading gracefully)
- [ ] Vendor table: FTDI, Märzhäuser, Photometrics (`1B6B`), Hamamatsu, Nikon, Zeiss, Sentinel HASP
- [ ] Output feeds `smc config build`
