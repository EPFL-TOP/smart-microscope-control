# `objective-offsets`

The offsets between the turret's objectives, measured **from the plate itself**. You
register the plate with one objective (the 4× on the Ti2-E); this tool turns the turret
to each other objective you name, finds again under it the wells the registration
measured, and says how far each objective's image of the same wells lies from the first
one's. `smc objective offset --from-run-doc` shows what that puts into the profile's
`[objectives]`, and `--write` writes it. The design is
[`m3-plugins.md`](../design/m3-plugins.md) §20 (the sign and the section are m2 §13);
this page is how to use it.

It replaces the stage micrometer centred by hand
([m2 §13.6](../design/m2-hardware-layer-completion.md)) as the way to get the numbers.
That procedure stays as the **cross-check**: once, at the stand, compare the two.

|  | by hand (m2 §13.6) | `objective-offsets` |
|---|---|---|
| The feature | a point on a stage micrometer, centred under each objective | a well's centre, measured from its four walls (a point, which one wall is not) |
| Who centres | you, with the joystick, under each objective | the tool, with `plate-walls`' own loop |
| Z | read, and written | **not measured** (a focused Z needs you) |
| Written to the profile | by hand | `smc objective offset --write`, in a block of its own |

**The tool turns the turret over a loaded plate.** It refuses to turn anything unless
you give `confirm_turns=true`, on every stand, whatever the profile's
`turret_requires_confirm` says: check that each objective clears the plate (its working
distance, the plate's height) first.

## At the stand

1. **Once per objective, in a run document of its own**, which later sessions reuse: turn
   to the objective, put the stage on a well's wall, calibrate the camera.

   ```text
   smc objective set "2-Plan Fluor 10x" --confirm
   smc run camera-calibration -p nikon-ti2 --run-doc cal-10x.json
   ```

   The same with the 40× into `cal-40x.json`. Each target needs its **own** calibration:
   the camera's turn and mirror decide which way every wall lies, and the scale changes
   with the objective. A calibration run in `run.json` would replace the one the plate was
   registered with and make the plate stale, hence a document each.
2. **The objective the plate is registered with in place** (the 4×): `camera-calibration`
   and then [`plate-walls`](plate-walls.md) into `run.json`. The plate must come from
   `plate-walls`: a plate registered by eye (`plate-registration`) or fitted
   (`plate-finding`) has positions nobody measured. **Do not touch the plate between
   this step and the next, and do not reload it.** The tool compares each well's centre now
   with the one `plate-walls` recorded, so a plate that moved as a whole (re-seated,
   bumped, drifted) shifts every well alike: the wells agree, the run ends `done`, and the
   shift is written as an offset between objectives. Nothing in the run can tell the two
   apart.
3. **Measure:**

   ```text
   smc run objective-offsets -p nikon-ti2 --run-doc run.json --param 'targets=["2-Plan Fluor 10x", "3-Plan Apo LmbdD0.25 40x"]' --param 'calibrations=["cal-10x.json", "cal-40x.json"]' --param confirm_turns=true
   ```

   The commands are written for a POSIX shell. In `cmd.exe` and PowerShell 5.1, escape
   each inner double quote with a backslash, as the
   [`smc run`](README.md#running-a-plugin-smc-run) page says; a Windows path's
   backslashes are doubled inside the JSON. That form was **not tried on a Windows shell**.
   The turret ends at the last target, and nothing turns it back.
4. **Look, then write:**

   ```text
   smc objective offset -p nikon-ti2 --from-run-doc run.json --session "session 3 (#NNN)"
   smc objective offset -p nikon-ti2 --from-run-doc run.json --session "session 3 (#NNN)" --write
   smc objective offset -p nikon-ti2
   ```

   The first prints what would be written and writes nothing, the second writes it, the
   third reads it back against the turret. The preview **is** the confirmation: the
   command line has no prompt anywhere, and the old values are printed before they are
   replaced.

## Parameters

```text
smc run objective-offsets --show-params
```

| Parameter | Default | Meaning |
|---|---|---|
| `targets` | required | The objectives to measure, by turret label (`smc objective list`), in the order the turret turns to them: one to five, none the plate's own, none twice. The last one is left in place. JSON: `["2-Plan Fluor 10x"]`. |
| `calibrations` | required | One run document per target holding a `camera-calibration` done with that target in place; JSON list of paths, as typed (relative to the working directory). They are matched to the targets by the objective each records, not by their place in the list. |
| `confirm_turns` | false | The stage is clear for the turret to turn to each target with the plate on it. Nothing turns without it. |
| `wells` | all the registration measured | At least two wells the registration measured. |
| `search_um` | the smaller of 1000 and the well's radius | How far each wall may lie from where the registration's centre puts it, in µm. It may not exceed the radius, and an offset larger than the profile's bound (±1000 µm) could not be written anyway. |
| `max_spread_um` | 2 pixels of the coarser of the two objectives | How far apart the wells' differences may lie. |
| `settle_ms` | 0 | Wait this long after each move before the frame is taken. |
| `save_frames` | true | Save the frame centred on each wall, see below. |

## What it does

Before anything turns, in this order: a dry run is refused; `confirm_turns`; the plate
(from `plate-walls`, with its objective recorded); the targets (on the turret, none the
plate's, none twice); the wells; one calibration per target, each fit for its objective
and frame; the profile's reference (if it names one, it must be the plate's objective or
a target); and each target's walk (its size and the stage's soft limits). A refusal at
this point turns and moves nothing.

**It switches the PFS off before anything turns or moves, and leaves it off.** After
those checks and a stop check, an autofocus that is on (or still searching) is switched
off once and read back off, with `plate-walls`' own function: a lock taken in one well
follows its reference over the wall into the next. The turret switches it off before a
turn too, but not when it already reads the first target (the 40× left in place by a
calibration, say), and then the first move to a wall would be made with the PFS engaged.
Nothing engages it again: lock it yourself on the sample afterwards. One that is already
off is left alone. If the switch-off fails, or the PFS still reads on after it, the run
ends before anything turns or moves: `the autofocus could not be switched off before
anything moved (<error>): nothing moved, and PFS may still be on: switch it off at the
stand, then run objective-offsets again`, or `the autofocus still reads on after it was
switched off: nothing moved: …`.

Then, for each target in order:

1. **The turn**, with the layer's own confirmation. The turret must **read the target
   after it**, and Z, read twice 0.1 s apart, must not have moved more than 0.5 µm: the
   Ti2-E moves Z by itself on a turn, and a frame is not taken while it still does.
2. **Each well, with `plate-walls`' own loop** and the target's calibration: the four
   walls, a circle to them, and the same refusals (a well of another size, a wall that is
   not a circle). The walk starts at the centre the registration measured.
3. **The well's difference**: its centre under the target minus its centre under the
   registration's objective. The plate's fit is not used: its prediction for a well
   carries the fit's residual, which two measured centres of one well do not.

The target's **shift** is the mean of its wells' differences, and the **spread** the
largest distance between two of them. If the shift is one translation over the plate, the
wells agree: the spread is the evidence, and it also carries the stage's repeatability.

After the last target the offsets are moved to the profile's reference (or, when the
profile names none, the plate's objective becomes the reference), and the stage goes to the
last target's last well's fitted centre, which you see on Live. **Nothing is turned or
moved back**; on a stop, a halt or a refusal a WARNING says where the stage and the turret
were left.

## What you get

The `objective-offsets` slot: per target, the calibration used and when it was measured,
the pixel size, Z after the turn, each well (where the registration put it, where the
target does, the difference, the fitted circle), the shift, the spread and the bound it was
held to, the frames and their folder; then the offsets to write, the reference and whether
the profile named it. `assumed` says what it rests on: that the stage reports no soft
limits when it does not; that the offsets are for the magnifier lever as it was, which the
profile cannot key (FM-103); that Z after a turn is where the stand left it, not a focused
Z; that the plate has not moved since `plate-walls` registered it (a plate that moved as a
whole is taken for an offset between objectives, and the wells agreeing does not show
otherwise); and, when the profile names no reference, that `--write` makes the plate's
objective the reference. When the PFS was on, **the first line** says it was switched off
before anything moved (a WARNING says it too).

**Frames.** With `save_frames` the centred frame of each wall is a TIFF under
`objective-offsets/<token>/<n>/<well>_<side>.tif` beside the run document, `n` the target's
place in `targets` from 1: no objective label is in a path.

## What `--write` writes

One block in the profile's file, between two marker lines:

```toml
# >>> [objectives], written by smc objective offset --write (m2 §13)
# The next write replaces everything up to the closing marker; edit values here only if no write follows.
[objectives]
reference = "2-Plan Fluor 10x"

[objectives.offsets."1-Plan Fluor 4x"]
xy_offset_um = [
    -5.0,
    2.5,
]
measured = "2026-11-12, session 3 (#NNN), objective-offsets in run.json"
# <<< [objectives], written by smc objective offset --write
```

- **No line outside the block is touched**: it is appended after one empty line when there
  is none and replaced in place when there is one. A profile saved with CRLF line ends
  keeps them; a byte-order mark is dropped, which the loader reads the same.
- **Entries are replaced, not merged.** An entry for a label the run measured is a new
  one, with `xy_offset_um` rounded to 0.01 µm and `measured` naming the date, your
  `--session` text and the run document's *name* (never its folder). A `z_offset_um` it
  held is dropped, and the preview says so: this tool measures XY only.
- **A damaged block, or an `[objectives]` table outside it, is refused**: a hand-written
  table (the profile as m2 §13.6 leaves it) must be deleted, its values are printed above,
  or kept by hand instead of writing.
- **The write needs the run to be current.** The result must be a usable
  `objective-offsets` record of this profile, and the profile's reference must still be the
  one the run measured from; otherwise the command says to run the tool again and writes
  nothing. A refused run has no `done` record, so there is nothing to write from.
- The new text is read back as the loader reads it before it is written, and written
  atomically. On Windows an editor that holds the profile open can make the replace fail:
  the command then says where the whole new text was kept (a `.<name>.<token>.smc-tmp` file
  beside the profile, which git ignores): close what holds the profile, write again, and
  delete that file. The profile is as it was.

## What each refusal means

A refusal is a result with a reason, and the run document records it `failed`. The first
group is checked **before anything turns or moves**.

| It says | It means, and what to do |
|---|---|
| `a dry run cannot turn the turret or image the walls; run it without --dry-run` | The tool turns and images; there is nothing to simulate. |
| `objective-offsets turns the turret to <t1>, <t2> with the plate on the stage: check that each objective clears the plate …, then give confirm_turns=true` | Nothing turns without your word, on any stand. |
| `the plate was registered with <method>: objective-offsets compares the centres plate-walls measured, so register it with plate-walls` | The plate came from `plate-registration` or `plate-finding`. |
| `the plate does not record the objective it was registered with: …` / `the plate's record names <a> and its wall images <b>: …` | Register the plate again with `plate-walls` on a stand with a turret. |
| `objective <t> is not on the turret: <labels>` / `<t> is the objective the plate was registered with: …` / `<t> is given twice in targets` | `smc objective list` shows the labels; the plate's own objective has no offset from itself. |
| `<w> is not a well the registration measured: choose among …` / `<w> is given twice in wells` / `give at least two wells: …` | Two wells at least, among those `plate-walls` measured. |
| `no camera calibration of <t> in calibrations: turn to it …` | Each target needs a calibration of its own; the message names the three commands. |
| `the calibration in <path> is of <objective>, which is not in targets: …` / `<a> and <b> are both calibrations of <t>: give one per target` | One calibration per target, no more. |
| `the camera calibration is not of this stand: …` | A calibration of another frame or an unusable matrix: run `camera-calibration` again at that objective. |
| `the profile's offsets are measured from <R> ([objectives] reference), which this run does not measure: add <R> to targets, or clear the reference` | The profile's reference is neither the plate's objective nor a target. |
| `search_um … is more than the well's radius …` / `up to <n> frames: …` / `<well>'s <side> wall would be looked for at … outside the stage's soft limits` | As in [`plate-walls`](plate-walls.md#what-each-refusal-means), per target. |

The others come **after the turret has turned**, which stays where it is:

| It says | It means, and what to do |
|---|---|
| `the turn to <t> was refused: …` | The layer refused the turn; its own text follows. |
| `the turret reads <x> after the turn to <t>: it did not turn, or its label lags …` | Nothing was measured with that target. Look at the turret. |
| `Z moved from <a> to <b> µm within 0.1 s after the turn to <t>: the stand was still moving it …` | Z had not settled; nothing was measured. Run it again, or give the stand time. |
| `Z reads <a> and <b> µm after the turn to <t>: a Z that is not a number is not a still Z …` | The focus drive answered `nan` or `inf`: its readback is not working. Nothing was imaged. |
| `no wall found on …` | As in `plate-walls`, but the hint is the target's: the light, whether the target images the wall as the registration's objective did, and whether `search_um` covers the offset. |
| `the wells do not agree on the offset, so it is not a translation or not measured well enough: <t>: <w1> (<dx>, <dy>), <w2> (…) µm, spread <s> µm, bound <b> µm; … look at the frames in <folder>, or give a larger max_spread_um if the stage's repeatability is larger` | Every number is in the message, for every target that failed, and nothing can be written. Look at the frames before raising the bound. |
| `the offset of <O> would be (<x>, <y>) µm, beyond the profile's ±1000 µm: a wrong well or a wrong calibration, not two parcentric objectives` | The offset could not be written, and it is not an offset. |

## What it does not do

- **No Z offset.** Z after a turn is read and recorded, never written: a focused Z needs
  the operator, which a plugin cannot ask, or a declared PFS lock.
- **Plates from `plate-registration` or `plate-finding`** are refused.
- **It does not turn back** to the registration's objective, and nothing moves back after a
  failure.
- **The magnifier lever** is not keyed: the offsets are for the lever as it was during the
  run, which the profile stores by objective alone.
- **The plate page** has no step for it yet (#382), nor a button that writes the profile.
  A preset never keeps `confirm_turns`: the turns are confirmed again at every run.

## Trying it without a microscope

The `demo-sample` stand can declare an offset per objective, so that the tool has
something to recover. In a copy of `profiles/demo-sample.toml`, set the plate and the
camera as the tests do and give each objective its shift:

```toml
[sample]
a1_center_xy_um = [0.0, 0.0]
rotation_deg = 1.2
camera_rotation_deg = 30.0
mirrored = true
objective_shift_um = { "Nikon 10X S Fluor" = [5.0, -2.5], "Nikon 20X Plan Fluor ELWD" = [0.0, 0.0], "Nikon 40X Plan Fluor ELWD" = [-32.5, 9.75] }
```

Under an objective a feature is centred at its stage position plus its shift, so the 10X
(in place at open) is the registration's objective and the offsets to recover are the
shifts' differences: the 40X from the 10X is `(-37.5, 12.25)` µm and the 20X `(-5.0,
2.5)` µm. Then follow the four steps above with `-p` that file, the turret labels of the
demo (`smc objective list`), `camera-calibration` under the 20X and 40X, and
`plate-walls` on A1 and B2.

Measured on the demo on 2026-11-09 (Python 3.13.0, pymmcore-plus 0.18.1), with the 40X as
the only target and A1 and B2 as the wells: the offset came back as (−37.48, +12.24) µm
for the declared (−37.5, +12.25), the two wells agreed to 0.03 µm against a bound of
2.00 µm, and the run took 5.6 s on the demo stage. That proves the loop, the arithmetic
and the write, not the stand: the camera is the demo's and the walls are the sample's.

## Not verified

At the Ti2-E, for session 3 or 4 ([#198](https://github.com/EPFL-TOP/smart-microscope-control/issues/198)):
the offsets 4× → 10× and 4× → 40× measured on a plate and compared once with m2 §13.6's
procedure by hand; whether the 40× images a well's wall in brightfield as the 4× and 10×
do (its depth of field, the meniscus), so that the wall is found; whether 2 pixels of the
4× (5.5 µm) hold the wells' spread, which also carries the stage's repeatability; how long
a run takes with the turns and the moves between wells; whether a turn at the plate's
working height settles within the turret's wait; that the shift is a translation over the
plate (two wells far apart that agree within the bound are the evidence the tool gives).
The JSON quoting on `cmd.exe` and PowerShell, and the write when an editor holds the profile
open on Windows, were not tried on a Windows machine.
