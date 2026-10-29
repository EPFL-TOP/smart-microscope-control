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
# stop_refused_by = []         # roles whose device refuses stop(), measured at the stand
# max_engage_excursion_um = 50.0  # omitted: only z_soft_limits_um bound Z while the autofocus engages

[camera]
pixel_size_um = { "Nikon 10X S Fluor" = 0.65, "Nikon 40X Plan Fluor ELWD" = 0.1625 }

[autofocus]                    # every key optional; omitted = unknown
offset_unit = ""               # "" = unknown, shown as "device units"
# lock_timeout_s = 5.0         # omitted: 5 s, assumed and labelled
offset_per_um = {}             # objective label -> offset change per µm, measured
in_range_property = ""
in_range_values = []
# engage_drives_z_to_offset = false  # true: engage only at an offset seen locked
# settle_s = 0.0               # 0: the first locked read ends engage()
# settle_tolerance_um = 0.5

[light_source]
intensity_property = ""        # "" = found by hints on the light_source device

[channels]
group = ""                     # "" = the config group named for channels
exclude = []                   # device labels a capture never records

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
| `camera.pixel_size_um` | µm per pixel per objective label, at binning 1 and a magnification factor of 1.0 (the convention of MMCore's own pixel-size presets): the layer multiplies by the camera's binning and divides by the core's `getMagnificationFactor()`, the product of the stand's `Magnifier` devices. A stand whose changer MMCore cannot read needs sizes measured with it in place. A missing objective is *unknown* (0.0), never guessed, and so is any size while that factor is not a positive number. |
| `autofocus.offset_unit` | What the offset's numbers are (`"PFS offset steps"`), for messages; `""` is unknown and shows as "device units". The offset is never converted to µm implicitly. |
| `autofocus.lock_timeout_s` | How long `engage()` and the Z interlock wait for a lock, in (0, 600] s. Omitted: 5 s, a figure from `nikon-control` that was never measured, labelled as assumed wherever it shows. A Z move under an engaged autofocus that finds no lock within it raises `AutofocusLostError`, and the autofocus is then off. |
| `autofocus.offset_per_um` | Objective label (as the objective turret reports it) -> offset change per µm of focus, **measured** with the autofocus locked, sign included; finite and non-zero. `focus_by_um()` while locked refuses for an objective that has none. |
| `autofocus.in_range_property` / `in_range_values` | The autofocus device's property that says whether its reference is in range, and the values of it that mean so. Both or neither, and no empty value. Without them, `is_in_range()` is `None`: the layer never guesses which property means "in range". |
| `autofocus.engage_drives_z_to_offset` | `true` declares that engaging drives Z to the plane the stored offset names, whatever the operator focused (measured on the Ti2-E, #205). `engage()` and the Z interlock's relock then send nothing unless the offset now set is one this autofocus has been read locked at, with the same objective, since the stand was opened: lock it at the stand first, and smc sees the lock at its first read. Not forceable. `false` (the default): it engages as before (m2 §12). |
| `autofocus.settle_s` / `settle_tolerance_um` | With `settle_s` > 0, `engage()` returns `True` only once every Z read of the last `settle_s` seconds lies within `settle_tolerance_um` µm of each other, still locked; a lock that has not settled by the lock timeout is switched off and is `False`. `settle_s` is in [0, 60] s and below `lock_timeout_s` (5 s when unset), which bounds the lock and the settling together; the tolerance is positive and finite. `0` (the default): the first locked read ends the wait (m2 §12.2 D3). |
| `safety.max_engage_excursion_um` | How far Z may move while the autofocus engages, beyond where it started (or, for the relock after a Z move, where it held focus before). Beyond it, or past `z_soft_limits_um`, the autofocus is switched off and `AutofocusExcursionError` is raised. Z is read every 10 ms, so this limits a jump, it does not prevent one. Omitted: only the soft limits bound it. Positive and finite (m2 §12.2 D2). |
| `safety.stop_refused_by` | The roles whose device refuses `stop()`, among `xy_stage`, `focus` and `autofocus_offset`, each named once; empty (the default) declares nothing. A hardware session fills it from what the stand did, and `test_a_declared_stop_refusal_still_holds` checks it at each later session. The stop is still sent, so a firmware that accepts it still stops. What changes is what smc says: a move under way on a declared role runs to its target, `smc stage`/`smc z` moves and `smc run` print a `!` line saying so before anything is sent, and the Ctrl-C line, `close()` and the emergency stop name the refusal instead of "stopped" (m2 §11). |
| `safety.turret_requires_confirm` | `true` (the default): `ObjectiveTurret.select()` refuses unless the call passes `confirm=True`, because turning the nosepiece can drive an objective into a loaded plate. `false`: it turns without `confirm`, and an INFO line says so whenever a select passes none. Either way, a turn switches the autofocus off first and never engages it again. |
| `light_source.intensity_property` | The `light_source` device's property that sets how bright, in the device's own units. Used when the device has it and it is writable; otherwise a WARNING, and the hints decide. `""`: the first writable numeric property whose name holds `intensity`, `brightness`, `power`, `level` or `voltage`, one with limits first. With none, the light source has no knob, and `set_intensity_native()` refuses. |
| `channels.group` | The config group whose presets are channels; printable ASCII, no comma, no leading or trailing space. `""`: the core's channel group if its name holds `channel` and none of `camera`, `objective` or `system`, else the one group so named; with several, none (set this key). It may name a group that does not exist yet: the first `capture()` creates it. |
| `channels.exclude` | Device labels that `capture()` never records (a `State` device that is not part of a channel). The objective turret is never recorded, listed or not. |

**A capture writes the configuration file.** `Channels.capture(name)` replaces
the preset in the profile's `micromanager.config` file (an atomic text edit:
every other line, comment and line ending is kept), then in the running core.
`nikon-ti2.cfg` is committed, so a capture at the stand edits the repository's
copy; the owner decides whether to commit it.

The new numeric keys (`lock_timeout_s`, `offset_per_um`'s values,
`max_autofocus_offset_jog`, the offset soft limits, `settle_s`,
`settle_tolerance_um`, `max_engage_excursion_um`) refuse a TOML boolean
(`true` would otherwise load as `1.0`) and accept an integer.

Errors name the file, the dotted key and what to change, e.g.
`profiles/x.toml: … safety.max_jog: unknown key in [safety] — fix the
spelling or remove it; allowed keys: max_jog_um, …`.

## Shipped profiles

| Profile | Stand | Configuration | State |
| --- | --- | --- | --- |
| `demo` | Micro-Manager's demo devices | the one shipped with the adapters | the regression bed of every test |
| `demo-sample` | The demo devices, with a camera that images a synthetic 96-well plate | the one shipped with the adapters | for trying the plate tools without a microscope; the pixels are drawn, see [`[sample]`](#sample-the-synthetic-plate-of-demo-sample) |
| `nikon-ti2` | Nikon Ti2-E, Prime 95B on PVCAM | `nikon-ti2.cfg` | measured in session 0 (#126), with the owner's limits |
| `nikon-tie` | Nikon Ti-E | `nikon-tie.cfg` | **assumed**: written from the adapter's fixed device list, never loaded on the stand |

A test loads every `*.toml` in this folder. A key a profile leaves out is
unknown, not defaulted silently: `nikon-ti2` has no pixel size, no autofocus
offset factor and no autofocus soft limit until a session at the stand
measures them. Each value in `nikon-ti2.toml` says in a comment whether it
was measured, set by the owner or chosen; `max_z_jog_um` is chosen, not
measured, and the owner revises it when the PFS session (#16) has run.

## `[sample]`: the synthetic plate of `demo-sample`

An optional table, on the demo stand only: `Microscope.open` (so `smc run`, `smc ui
plate` and a notebook) then replaces the demo camera's pixels with a rendering of a
synthetic well plate at the stage's position, so the plate tools can be tried without a
microscope (`smc.testing.sample_stand`; design in `docs/design/m1-hardware-layer.md` §8).
Every key has a default, so `[sample]` alone is a 96-well plate centred on the stage
origin; `profiles/demo-sample.toml` writes each one out.

| Key | Default | Meaning |
| --- | --- | --- |
| `plate` | `"96-well"` | A `useq` plate name ending in `-well` (`6-well`, `24-well`, `384-well`, ...). |
| `a1_center_xy_um` | omitted | Where well A1's centre is on the stage, in µm, within ±1e6. Omitted: the plate centred on the stage origin, (−49 500, 31 500) for 96 wells, so a frame at the origin is plastic. |
| `rotation_deg` | `0.0` | The plate's rotation about A1, in [−45, 45]. |
| `camera_rotation_deg`, `mirrored` | `0.0`, `false` | How the camera sits on the stage: its rotation in [−180, 180], and whether the image is flipped left to right. |
| `cells_per_well` | `40` | The mean over the plate's columns, 0 to 500: the first column holds half of it and the last one and a half times, so density and route have something to tell apart. |
| `doublet_fraction` | `0.3` | The share of cells that are two spots side by side, in [0, 1]. |
| `cell_radius_um` | `5.0` | The radius of a cell's spot, in (0, 50] µm. |
| `seed` | `0` | The same seed gives the same cells, texture and noise. |

A `[sample]` is refused with a `micromanager.config` other than `""` (synthetic frames
over a real stage would look like a working stand), with a plate that is not a `useq`
well plate, with a number that is out of its bounds or is a TOML boolean, and when
`cells_per_well` times the plate's wells is above 50 000. `smc doctor -p demo-sample`
says what the sample is and what it is not; `Profile.demo()` and `demo.toml` have none.

## Rebuilding a profile's configuration

A profile's `.cfg` is rebuilt from the devices that answer with

```sh
smc config build --profile <name> --out <its .cfg> --force
```

(`--dry-run` first). The command carries the file's presets, pixel sizes, typed
state labels and delays over, sets aside, as `# not carried:` comments, what names a
device that is no longer loaded, and saves the previous file as `<its .cfg>.bak`.
It initialises the stand's devices: only at the stand, with the vendor software
closed. See `docs/hardware/microscope-pc-setup.md`.

## Windows paths

Write paths with forward slashes (`config = "C:/Micro-Manager/scope.cfg"`),
or in single-quoted TOML literal strings (`'C:\Micro-Manager\scope.cfg'`).
In a double-quoted TOML string a backslash starts an escape sequence.

Site-specific `.cfg` files that embed COM ports or serial numbers are not
committed; a profile points at them by path and `smc config build` can
regenerate them. `nikon-ti2.cfg` is committed because it holds no COM port,
serial number, host or account name (the repository is public, FM-50); a
configuration that does hold one stays outside it.
