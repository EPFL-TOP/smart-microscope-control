# `plate-check`

Checks that a registered plate **still lies where [`plate-walls`](plate-walls.md)
registered it**. It measures again, with `plate-walls`' own loop, the centres of two
wells the registration measured, under the objective in place, and compares them with
the registered centres. A plate that moved more than a bound is refused; one that did
not is recorded with **when** it was confirmed. The design is
[`m3-plugins.md`](../design/m3-plugins.md) §27; this page is how to use it.

Why: at Ti2-E session 5 (#550, F15) the plate was registered at the 10× and scanned,
then lifted and put back. It turned by about −0.056° and slid; nothing noticed, and the
40× scan made then placed its frames about (+19, −50) µm off in their wells. A stage
position stays true; what goes stale is its reading against the wells. Experimenters
lift and put back a plate without noticing.

**The scan makes the same check by itself** before its first field
([`scan`](scan.md#the-plate-check)). Run `plate-check` yourself after the plate was
handled and before what moves nothing or moves once: a drive to a well
([`goto-well`](plate-registration.md)), a [`route`](route.md), an
[NIS export](nis-export.md).

## Running it

```text
smc run plate-check -p nikon-ti2 --run-doc run.json
smc run plate-check -p nikon-ti2 --run-doc run.json --param calibration=cal-40x.json
smc run plate-check -p nikon-ti2 --run-doc run.json --param 'wells=["B2", "B5"]'
```

The commands are written for a POSIX shell. In `cmd.exe` and PowerShell 5.1, escape each
inner double quote with a backslash, as the [`smc run`](README.md#running-a-plugin-smc-run)
page says (`--param "wells=[\"B2\",\"B5\"]"`); that form was not tried on a Windows shell.

The first is under the objective the plate was registered with: the run document's own
camera calibration is that objective's. Under **another objective**, give a run document
whose `camera-calibration` was made with it in place (`calibration=`, as
[`objective-offsets`](objective-offsets.md) and the plate page make them, m5 §18.6), and
the profile must hold the offset between the two objectives (`[objectives]`, written by
`smc objective offset --write`): a check under another objective whose offset is not
measured reads the offset as a move (39.4 µm on the demo for a plate that did not move).

| Parameter | Default | Meaning |
|---|---|---|
| `wells` | the pair farthest apart | Two to six wells the registration measured. One well shows a slide and not a turn about itself; the pair farthest apart shows a turn at its longest lever. |
| `max_shift_um` | 25 | How far a checked well may lie from its registered centre, in µm (above 0, at most 500). |
| `calibration` | this document's | A run document holding a camera calibration of the objective in place. |
| `search_um` | min(1000, the well's radius) | How far each wall may lie from where the registered centre puts it. |
| `settle_ms` | 0 | A wait after each move, before the frame. |
| `save_frames` | true | The frame centred on each wall, as `plate-check/<token>/<well>_<side>.tif`. |

**The cost**: 8 frames per well (2 per side) while each wall lies within the first frame:
about **20 s at the 10×** for two wells 58 mm apart, estimated from session 5's
registration (not measured at the stand). A plate moved further takes up to 21 frames a
side at the 40×.

**The bound**, 25 µm, is chosen: it leaves room for the scatter of one objective's
offset across wells (up to 10.9 µm at session 5, wet wells against a dry one) and the
check's own repeatability (not yet measured at the stand). Session 5's move read 31.9
and 33.5 µm at its farthest pair. 25 µm is 6 % of the 40×'s field.

## What it says

A plate that did not move: the slot `plate-check` holds a `PlateCheckResult` whose
`check` has `checked_utc` (when the last well was measured: when the registration was
last confirmed), each well's registered, expected and measured centre, its difference
and distance, the worst distance and the bound, the turn and the slide, the frames and
the seconds. `assumed` ends with
`the plate was checked under <objective>: B2 and B5 within 0.4 µm of their registered
centres (bound 25 µm)`. The stage ends on the last well's fitted centre.

A plate that moved: the run fails, with every number in its reason:

```text
the plate moved since plate-walls registered it: B2 lies 31.9 µm from its registered centre (-14.5, -28.4) and B5 33.5 µm (-16.7, +29.0) under 2-Plan Fluor 10x, more than 25 µm; it turned about -0.056° and slid about (-15.6, +0.3) µm: register it again (plate-walls) before a scan, a drive to a well, a route or an NIS export
```

The turn and the slide are said, never judged: the verdict is how far a well moved.

## What is refused before anything moves

A dry run; a turret at an empty position; an offset between the plate's objective and
the one in place that is not measured (`the offset between <a> and <b> is not measured
(the profile's [objectives]): check the plate with <a> in place, or measure the offsets
(objective-offsets)`); no camera calibration (`no camera calibration of <objective>:
give calibration=<a run document whose camera-calibration was made with it in place> (m5
§18.6)`) or one of another objective or frame; a plate that `plate-walls` did not
register (a plate registered by eye or fitted has positions nobody measured); a well the
registration did not measure, one given twice, fewer than two; a walk that is too long
or leaves the stage's soft limits.

**The autofocus.** An engaged PFS is switched off once, before the first move, and read
back off: a lock in one well follows its reference over the wall into the next. The
result says so first; nothing engages it again, and a refusal after the switch-off ends
with `the autofocus was switched off before anything moved and was not engaged again`
(m3 §5.7).

## Not verified

At the Ti2-E, next visit (§27.9): `plate-check` twice in a row at the 10× and then at the
40× with the plate untouched (the repeatability: the floor of the bound); a plate lifted
and put back, refused, against a `plate-walls` recheck's numbers; the time of a check at
the 10× and the 40× against the 20 s above; the walls found at the 40× with PFS off at
the plane PFS held.
