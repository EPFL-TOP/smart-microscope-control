# Microscope profiles

One TOML file per instrument. A profile is the **only** place where "this
stand is special" may be written: which Micro-Manager configuration to
load, which device fills which role when the heuristics would guess wrong,
the safety limits, per-objective pixel sizes and the vendor quirks the
facade must honour (ADR-0003). `demo.toml` is the simulator's profile and
the loader's reference fixture.

```python
from smc.hardware.profile import Profile, list_profiles

profile = Profile.load("demo")  # by name, over the search paths
profile = Profile.load("profiles/demo.toml")  # or by path
profile = Profile.demo()  # built in code, no file needed
```

## Where profiles are found

`Profile.load(name)` looks for `<name>.toml` in, in order:

1. every directory in `$SMC_PROFILES` (separated by `:` on macOS/Linux,
   `;` on Windows) — keep site profiles outside the repository here;
2. `./profiles` (relative to the working directory);
3. the working directory itself.

The first match wins. An argument that is a `Path`, contains a `/` (or `\`
on Windows) or ends in `.toml` is read as given instead. `"demo"` falls back
to the built-in `Profile.demo()` when no `demo.toml` is found.
`list_profiles()` returns every `(name, path)` a bare name would resolve to.

## Schema

Every section rejects unknown keys, so a misspelt safety limit is an error
at load time rather than a limit that silently does not apply. `[quirks]` is
the one free-form table. Only `[microscope].name` is required.

```toml
[microscope]
name = "demo"
vendor = "Micro-Manager"
description = "DemoCamera adapter: simulated camera, XY, Z, turret, shutter, autofocus."

[micromanager]
config = ""                    # "" = demo configuration; relative paths resolve against this file
device_timeout_ms = 60000
adapter_search_paths = []      # extra adapter directories, searched before pymmcore-plus's; must exist

[roles.assign]                 # overrides only; keys are Role values
# xy_stage = "XY"

[roles.exclude]                # extra name substrings per role
# focus = ["piezo"]

[safety]
max_jog_um = 5000.0
# max_z_jog_um = 100.0         # omitted: 100 µm, assumed and logged
# z_soft_limits_um = [-1000.0, 1000.0]
# xy_soft_limits_um = [[-60000.0, 60000.0], [-40000.0, 40000.0]]
turret_requires_confirm = true
# max_autofocus_offset_jog = 50.0                # in [autofocus] offset_unit
# autofocus_offset_soft_limits = [0.0, 15000.0]  # in [autofocus] offset_unit

[camera]
pixel_size_um = { "Nikon 10X S Fluor" = 0.65, "Nikon 40X Plan Fluor ELWD" = 0.1625 }

[autofocus]                    # every key optional; omitted = unknown
offset_unit = ""               # "" = unknown, shown as "device units"
# lock_timeout_s = 5.0         # omitted: 5 s, assumed and labelled
offset_per_um = {}             # objective label -> offset change per µm, measured
in_range_property = ""
in_range_values = []

[quirks]                       # free-form, consumed by backends; documented per key in M2
```

| Key | Meaning |
| --- | --- |
| `micromanager.config` | Must exist when the profile is loaded. Relative paths resolve against the profile file's directory. |
| `micromanager.device_timeout_ms` | Applied to MMCore; a plate traverse exceeds its 5 s default. |
| `roles.assign` / `roles.exclude` | Keys are role names: `camera`, `xy_stage`, `focus`, `autofocus`, `autofocus_offset`, `objective_turret`, `shutter`, `light_source`, `light_path`, `filter_turret`. |
| `safety.max_jog_um` | A relative XY move larger than this is refused unless forced. |
| `safety.max_z_jog_um` | A relative Z move larger than this is refused unless forced; omitted means 100 µm, assumed and logged. Set it per stand and objective, and set `z_soft_limits_um` too, since they bound a forced jog. |
| `safety.*_soft_limits_um` | `[low, high]` with `low < high`; omit for no soft limit on that axis. |
| `safety.max_autofocus_offset_jog` | A relative change of the autofocus offset larger than this is refused unless forced, in `autofocus.offset_unit`. Omitted, **every** non-zero relative change needs `force=True`: the unit is the stand's own, so no default is assumed. Positive and finite. |
| `safety.autofocus_offset_soft_limits` | `[low, high]` for the autofocus offset, in its own units; a target outside is refused and cannot be forced. Omit for no soft limit. |
| `camera.pixel_size_um` | µm per pixel per objective label. A missing objective is *unknown* (0.0), never guessed. |
| `autofocus.offset_unit` | What the offset's numbers are (`"PFS offset steps"`), for messages; `""` is unknown and shows as "device units". The offset is never converted to µm implicitly. |
| `autofocus.lock_timeout_s` | How long `engage()` and the Z interlock wait for a lock, in (0, 600] s. Omitted: 5 s, a figure from `nikon-control` that was never measured, labelled as assumed wherever it shows. A Z move under an engaged autofocus that finds no lock within it raises `AutofocusLostError`, and the autofocus is then off. |
| `autofocus.offset_per_um` | Objective label (as the objective turret reports it) -> offset change per µm of focus, **measured** with the autofocus locked, sign included; finite and non-zero. `focus_by_um()` while locked refuses for an objective that has none. |
| `autofocus.in_range_property` / `in_range_values` | The autofocus device's property that says whether its reference is in range, and the values of it that mean so. Both or neither, and no empty value. Without them, `is_in_range()` is `None`: the layer never guesses which property means "in range". |

The new numeric keys (`lock_timeout_s`, `offset_per_um`'s values,
`max_autofocus_offset_jog`, the offset soft limits) refuse a TOML boolean
(`true` would otherwise load as `1.0`) and accept an integer.

Errors name the file, the dotted key and what to change, e.g.
`profiles/x.toml: … safety.max_jog: unknown key in [safety] — fix the
spelling or remove it; allowed keys: max_jog_um, …`.

## Shipped profiles

| Profile | Stand | Configuration | State |
| --- | --- | --- | --- |
| `demo` | Micro-Manager's demo devices | the one shipped with the adapters | the regression bed of every test |
| `nikon-ti2` | Nikon Ti2-E, Prime 95B on PVCAM | `nikon-ti2.cfg` | measured in session 0 (#126), with the owner's limits |
| `nikon-tie` | Nikon Ti-E | `nikon-tie.cfg` | **assumed**: written from the adapter's fixed device list, never loaded on the stand |

A test loads every `*.toml` in this folder. A key a profile leaves out is
unknown, not defaulted silently: `nikon-ti2` has no pixel size, no autofocus
offset factor and no autofocus soft limit until a session at the stand
measures them. Each value in `nikon-ti2.toml` says in a comment whether it
was measured, set by the owner or chosen; `max_z_jog_um` is chosen, not
measured, and the owner revises it when the PFS session (#16) has run.

## Windows paths

Write paths with forward slashes (`config = "C:/Micro-Manager/scope.cfg"`),
or in single-quoted TOML literal strings (`'C:\Micro-Manager\scope.cfg'`).
In a double-quoted TOML string a backslash starts an escape sequence.

Site-specific `.cfg` files that embed COM ports or serial numbers are not
committed; a profile points at them by path and `smc config build` can
regenerate them. `nikon-ti2.cfg` is committed because it holds no COM port,
serial number, host or account name (the repository is public, FM-50); a
configuration that does hold one stays outside it.
