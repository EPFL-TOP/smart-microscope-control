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
adapter_search_paths = []      # extra Micro-Manager directories, e.g. a separate MMStudio install

[roles.assign]                 # overrides only; keys are Role values
# xy_stage = "XY"

[roles.exclude]                # extra name substrings per role
# focus = ["piezo"]

[safety]
max_jog_um = 5000.0
# z_soft_limits_um = [-1000.0, 1000.0]
# xy_soft_limits_um = [[-60000.0, 60000.0], [-40000.0, 40000.0]]
turret_requires_confirm = true

[camera]
pixel_size_um = { "Nikon 10X S Fluor" = 0.65, "Nikon 40X Plan Fluor ELWD" = 0.1625 }

[quirks]                       # free-form, consumed by backends; documented per key in M2
```

| Key | Meaning |
| --- | --- |
| `micromanager.config` | Must exist when the profile is loaded. Relative paths resolve against the profile file's directory. |
| `micromanager.device_timeout_ms` | Applied to MMCore; a plate traverse exceeds its 5 s default. |
| `roles.assign` / `roles.exclude` | Keys are role names: `camera`, `xy_stage`, `focus`, `autofocus`, `autofocus_offset`, `objective_turret`, `shutter`, `light_source`, `light_path`, `filter_turret`. |
| `safety.max_jog_um` | A relative move larger than this is refused unless forced. |
| `safety.*_soft_limits_um` | `[low, high]` with `low < high`; omit for no soft limit on that axis. |
| `camera.pixel_size_um` | µm per pixel per objective label. A missing objective is *unknown* (0.0), never guessed. |

Errors name the file, the dotted key and what to change, e.g.
`profiles/x.toml: … safety.max_jog: unknown key in [safety] — fix the
spelling or remove it; allowed keys: max_jog_um, …`.

## Windows paths

Write paths with forward slashes (`config = "C:/Micro-Manager/scope.cfg"`),
or in single-quoted TOML literal strings (`'C:\Micro-Manager\scope.cfg'`).
In a double-quoted TOML string a backslash starts an escape sequence.

Site-specific `.cfg` files that embed COM ports or serial numbers are not
committed; a profile points at them by path and `smc config build` can
regenerate them.
