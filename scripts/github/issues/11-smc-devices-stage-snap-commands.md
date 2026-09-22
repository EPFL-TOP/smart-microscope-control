---
title: feat(cli): smc devices / stage / snap commands
labels: [type: feature, area: cli, priority: p1, good first issue]
milestone: M1 — Stage on the simulator
---
## Goal
The CLI is the reference interface (ADR-0006): everything the facade can do is reachable from `smc`.

## Acceptance criteria
- [ ] `smc devices [--profile]` lists loaded devices, types and the role table with candidates
- [ ] `smc stage get|move X Y|jog DX DY [--force]` with the jog guard, `--dry-run`
- [ ] `smc snap [--out frame.tif] [--exposure-ms]` writes a 16-bit TIFF
- [ ] Exit codes are meaningful; errors are one readable line, not tracebacks
- [ ] `typer.testing.CliRunner` tests on the demo devices
