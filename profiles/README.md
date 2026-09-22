# Microscope profiles

One TOML file per instrument. A profile is the **only** place where "this
stand is special" may be written: which Micro-Manager configuration to
load, which device fills which role when the heuristics would guess wrong,
the safety limits, per-objective pixel sizes and the vendor quirks the
facade must honour (ADR-0003).

The loader and schema land in milestone M1 (issue *feat(core): microscope
profiles*). `demo.toml` shows the intended shape and is the first profile
that loader must accept.

```sh
smc doctor --profile demo
smc stage get --profile nikon-ti2
```

Site-specific `.cfg` files that embed COM ports or serial numbers are not
committed; a profile points at them by path and `smc config build` can
regenerate them.
