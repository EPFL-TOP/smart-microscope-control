# `plate-registration` and `goto-well`

Where the plate is on the stage, as three numbers: A1's centre and the plate's
rotation. The plate's own definition (`useq`: well spacing, well size) gives every
other well, so a registration needs only a few wells you centred, or a few walls
you touched. The result fills the run document's `plate` slot, which `scan-plan`
and everything after it read. The design is
[`m3-plugins.md`](../design/m3-plugins.md) §5 (the port of `nikon-control`'s
`plate.py` and its guided loop); this page is how to use it.

Two plugins make the loop, and each takes its inputs as **parameters**, so nothing
survives between two runs that the run document does not show:

| Plugin | Does | Moves the stage? |
|---|---|---|
| `plate-registration` | Fits the plate from `refs` (wells you centred) and `walls` (wall touches) and fills `plate`, with the objective it was seen with. | No; it only reads the turret. A dry run is the same computation. |
| `goto-well` | Drives to a well's centre or to the point on one of its walls, through the latest `plate`, in the frame of the objective in use; before any registration, to a well's centre where the stand's `[holder]` puts the plate ([Before registration](#before-registration-the-holders-estimate)). | Yes: **one absolute XY move**, after switching an engaged autofocus (PFS) off ([The autofocus](#the-autofocus-is-switched-off-before-the-move)). |

## The guided loop: never hunt for a well

Finding the first well is the only hard part, and it is done once. The well need
not be visible: put any well roughly under the objective, within a couple of
millimetres, and say which one it is.

```text
smc stage get -p <profile>
smc run plate-registration -p <profile> --run-doc run.json --param 'refs=[{"well": "A1", "x_um": 1000, "y_um": 2000}]'
smc run goto-well -p <profile> --run-doc run.json --param well=H12
smc stage get -p <profile>
smc run plate-registration -p <profile> --run-doc run.json --param 'refs=[ ...every reference so far... ]'
```

The commands are written for a POSIX shell, which keeps the double quotes of the JSON
inside the single quotes. In `cmd.exe` a single quote is not a quote, and the value is
split at each space; the form that the [`smc run`](README.md#running-a-plugin-smc-run)
page gives, double quotes outside and a backslash before each inner one, is the one to
try:

```bat
smc run plate-registration -p <profile> --run-doc run.json --param "refs=[{\"well\": \"A1\", \"x_um\": 1000, \"y_um\": 2000}]"
```

That form, and the one for PowerShell 5.1, which handles quotes in arguments to native
programs differently, were **not tried on a Windows shell**: check them at the stand. A
value that arrives broken is refused before anything is sent or written (`'refs' takes
JSON`, or an unexpected extra argument).

1. **Seed.** `smc stage get` prints the position (`XY (1000.00, 2000.00) µm`); give it
   as a reference. One reference is a *seed*: A1 is where it implies and the rotation
   is **assumed from the declared orientation** (0° for `not-turned`, 180° for
   `turned-180`, see [Which way the plate lies](#which-way-the-plate-lies)), which the
   result says in `assumed`. It is enough to drive to every other well to within a few
   millimetres. On a stand whose profile has a `[holder]`, `goto-well` with `plate=`
   reaches the first well before any reference, to the holder's estimate
   ([Before registration](#before-registration-the-holders-estimate)); centre it by eye
   and give that position as the seed.
2. **Go to the next well.** `goto-well` drives there through the latest `plate`; with
   PFS on, it switches it off before the move, and you engage it again in the well
   ([The autofocus](#the-autofocus-is-switched-off-before-the-move)). Jog until the
   well is where you want it (`smc stage jog`), and read `smc stage get`.
3. **Register again with every reference so far.** Each run replaces `plate`, and the
   run document's `params` keep every reference that made it: that is the audit
   trail. The fit replaces the seed, so the next well is found more accurately than
   the last, and by the third reference `goto-well` lands essentially on target.

Use **A1 and the two far corners** (`A1`, `A12`, `H1` on a 96-well plate; `A1`, `A24`,
`P1` on a 384-well one): adjacent wells barely constrain the rotation, so the extra
travel buys real accuracy. Two references are enough to fit it, and a third makes
the residual mean something.

The plate page ([`m3-plugins.md`](../design/m3-plugins.md) §15) collects the same
references by clicks and calls the same plugins with the same parameters. In the
words of the issue's steps, *seed* is `plate-registration` with one reference,
*goto-well* and *goto-wall* are `goto-well` (with `side=…`, a stage direction),
*capture-edge* is `smc stage get` followed by a wall touch in `walls`, and
*calibrate* is `plate-registration` with every reference and wall. One difference:
you name a wall on the command line by the **stage's** direction (`-X`, `+X`, `-Y`,
`+Y`), while the page lets you name it as the **image** shows it (`left`, `right`,
`top`, `bottom`) and maps it to the stage's with the
[camera calibration](camera-calibration.md#what-reads-it) (see
[Wall touches](#wall-touches)).

### A run on the demo stand

```text
$ smc run plate-registration -p demo --run-doc run.json --param 'refs=[{"well": "A1", "x_um": 1000, "y_um": 2000}]'
✓ plate-registration done in 0.0 s — run.json, stage plate-registration
{
  "assumed": ["one reference well (A1): rotation 0° assumed from the declared orientation (not-turned), not measured"],
  "plate": "96-well",
  "a1_center_xy_um": [1000.0, 2000.0],
  "rotation_deg": 0.0,
  "method": "seed",
  "residual_um": null,
  "scale_error": null,
  "n_refs": 1,
  "refs": [{"well": "A1", "x_um": 1000.0, "y_um": 2000.0}],
  "fit": null,
  "objective": "Nikon 10X S Fluor"
}

$ smc run goto-well -p demo --run-doc run.json --param well=B2 --param side=-X
✓ goto-well done in 1.0 s — run.json, stage goto-well
{
  "assumed": ["plate: one reference well (A1): rotation 0° assumed from the declared orientation (not-turned), not measured"],
  "well": "B2",
  "side": "-X",
  "target_x_um": 6800.0,
  "target_y_um": -7000.0,
  "x_um": 6799.995,
  "y_um": -7000.005,
  "objective": "Nikon 10X S Fluor",
  "plate_objective": "Nikon 10X S Fluor",
  "shift_um": [0.0, 0.0]
}
```

(Abridged: the real output has every list on separate lines.) The demo stage rounds a
command to 0.01 µm, so the readback is not the target; `goto-well` reports the
readback and keeps the target beside it.

## `plate-registration`

```text
smc run plate-registration -p <profile> --run-doc run.json --param plate=384-well --param 'refs=[...]' --param 'walls=[...]'
```

| Parameter | Default | Meaning |
|---|---|---|
| `plate` | `96-well` | A `useq` plate name (`6-well` to `1536-well`); the name is exact, `96-WELL` is refused. |
| `refs` | none | Wells centred under the objective, JSON: `[{"well": "A1", "x_um": 1000, "y_um": 2000}]`. A well name is as you type it: `a1`, `A 1` and `A01` are `A1`. |
| `walls` | none | Wall touches, JSON: `[{"well": "B2", "side": "-X", "x_um": 6800, "y_um": 0}]`. A `side` is a **stage direction** (`-X`, `+X`, `-Y`, `+Y`); `left`, `right`, `top` and `bottom` are refused. See [Wall touches](#wall-touches). |
| `max_rotation_deg` | 10 | The largest tilt of the plate from its declared orientation that the fit accepts (above 0, up to 45): see [What it refuses](#what-it-refuses). |
| `orientation` | `not-turned` | Which way the plate lies in its holder: `not-turned` (A1 at the stage's -X +Y corner) or `turned-180` (A1 at the +X -Y corner). See [Which way the plate lies](#which-way-the-plate-lies). |
| `objective` | the one in place | The objective the references were centred with, by its turret label (`smc objective list`): see [The objective the plate was registered with](#the-objective-the-plate-was-registered-with). Recorded, never turned to. |

Give `refs`, `walls` or both: a well given by walls becomes a reference like any
other, after the ones in `refs`. A well may be given one way only. The shell's
quoting of JSON is the one of [`smc run`](README.md#running-a-plugin-smc-run); on
PowerShell 5.1 or `cmd.exe` escape each inner quote with a backslash.

A fit of three references of a hand-centred plate, 100 µm out on A12 (abridged):

```text
$ smc run plate-registration -p demo --run-doc run.json --param 'refs=[{"well": "A1", "x_um": 1000, "y_um": 2000}, {"well": "A12", "x_um": 100000, "y_um": 2100}, {"well": "H1", "x_um": 1000, "y_um": -61000}]'
✓ plate-registration done in 0.0 s — run.json, stage plate-registration
{
  "assumed": [],
  "plate": "96-well",
  "a1_center_xy_um": [984.91, 2009.61],
  "rotation_deg": 0.0412,
  "method": "references",
  "residual_um": 31.69,
  "scale_error": 0.00046,
  "n_refs": 3,
  ...
}
```

`residual_um` is the largest distance between a reference and where the fit puts
that well; a few tens of µm is normal for hand centring. `scale_error` is the
measured spacing over the plate definition's, minus 1, on the longest baseline. Both
are `null` for a seed: nothing measured them.

A few hundred µm of registration error is **not** a problem for a scan: the error
shifts the scan grid and does not compound, and a well has margin. Rotation is the
part worth care, because it acts over the whole plate, which is why the references
should be far apart.

### Which way the plate lies

A plate can be put in its holder either way round. Turned 180°, every well lands where
another one was (B3 where C4 was, on a 24-well): the positions are the same and only the
names change, so nothing measured tells the two apart. The samples follow the names
printed on the plate, so **say which way it lies** with `orientation`:

| `orientation` | A1 is at the stage's | Rows run towards |
|---|---|---|
| `not-turned` (default) | -X +Y corner | -Y |
| `turned-180` | +X -Y corner | +Y |

Name the reference wells **from the plate** (the label printed on it). The fit is then
checked against the declaration: a fit that turns the plate about 180° from it is refused
with `the fit turns the plate by 179.8°: it lies turned 180° from the declared orientation
(not-turned), or two reference wells are swapped; ...`. Either the plate lies the other way
(declare `orientation=turned-180` and register again with the same references) or two
references were swapped (two wells exchanged fit at 180° exactly). The result keeps the
orientation, and every tool after it names the wells by it: `goto-well --param well=B3`
goes to the B3 printed on the plate, either way.

Found out after the fit that the plate lies the other way? Nothing needs measuring again:
[`plate-orientation`](plate-orientation.md) renames the registered plate's wells.

### Wall touches

At 40x a well is not visible (the field is about 333 µm, a 96-well well 6 400 µm),
so a centre is never judged by eye. Touch a wall instead: drive until the wall sits
on the crosshair, read `smc stage get`, and give it as a touch.

**Three frames meet at a plate, and a direction means something in one only.** The
**image** you look at (`left`, `right`, `top`, `bottom`), the **stage** (`+X`, `-X`,
`+Y`, `-Y`: the X and Y of `smc stage get`) and the **plate**, the stage's turned by
the plate's small rotation. Which stage direction an image side is depends on the
camera's turn and mirror: on the Ti2-E at 10x the image moves with the stage, so the
image's *left* is the stage's **+X** (Ti2-E session 2, where a wall named on the image
and read on the stage put a seed 15.5 mm from the well on both axes, #327).

- **`side` is a stage direction** on the command line: `-X`, `+X`, `-Y` or `+Y`, the
  wall that the stage reaches by going that way from the well's centre, along the
  plate's own axes. `+Y` is **towards row A** on a plate that is not turned; on either
  orientation `+X` is towards the stage's +X. `-X` and `+X` use only the touch's
  `x_um`, and `-Y` and `+Y` only its `y_um`: the other coordinate is never read.
  **`left`, `right`, `top` and `bottom` are refused**, with the reason (`side 'left'
  is a side on the image: a wall is named by a stage direction, -X, +X, -Y or +Y`):
  the same word in two frames was the defect, so an old word is not translated. A run
  document written before holds them in its `params` and is not migrated.
- **The plate page names walls on the image** and maps them with the latest usable
  [camera calibration](camera-calibration.md#what-reads-it), whatever its objective.
  Without one it refuses to capture a wall (run `camera-calibration`, *Set up*), and
  it says under the buttons what each side of the image is.
- **To read the stage's direction from the calibration** (command line): take its
  `rotation_deg` and `mirrored` and read the row. The page does the same with the
  matrix.

  | camera | image `left` | `right` | `top` | `bottom` |
  |---|---|---|---|---|
  | 0° | `-X` | `+X` | `+Y` | `-Y` |
  | 0°, mirrored | `+X` | `-X` | `+Y` | `-Y` |
  | 90° | `-Y` | `+Y` | `-X` | `+X` |
  | 90°, mirrored | `+Y` | `-Y` | `-X` | `+X` |
  | 180° (the Ti2-E at 10x) | `+X` | `-X` | `-Y` | `+Y` |
  | 180°, mirrored | `-X` | `+X` | `-Y` | `+Y` |
  | −90° | `+Y` | `-Y` | `+X` | `-X` |
  | −90°, mirrored | `-Y` | `+Y` | `+X` | `-X` |

  A camera more than 5° from a quarter turn names no side: the page refuses, and on the
  command line an image side is not a stage side there either.
- A well needs one `-X` or `+X` and one `-Y` or `+Y`. An **opposite pair**, `-X` and
  `+X` or `-Y` and `+Y`, gives the centre as the midpoint and measures the well's size,
  a free check that you touched the right wall: **prefer pairs**. **One wall** gives the
  centre only with the plate's nominal half-size, and the result says so in `assumed`
  (`B2: the Y centre is one wall plus the nominal half-size, 3200 µm`).
- `goto-well --param side=-X` drives to the point on that wall, along the plate's
  own axes, so it follows the plate's fitted tilt (its rotation from its declared
  orientation, never the 180° of a turn). The wall is then within a field or
  two. On a turned plate a touch along a stage axis errs by `r (1 − cos θ)` for one
  wall (8 µm at 4°, 49 µm at 10°, for a 96-well well, computed); an opposite pair of
  a round well cancels it.
- The nominal sizes are the plate's, width for X and height for Y.

### The objective the plate was registered with

Register with a wide objective (the 4x or the 10x: the walls are easy to see) and
scan with the 40x. Two objectives do not centre the same feature at the same stage
position, so the `plate` records the objective its positions were taken with, in
`objective`: the turret's label, `null` when the stand has no turret and in a run
document written before this field existed (unknown, not "the one in place now").

- **It is the objective in place when the registration runs.** Turn the turret
  first (`smc objective set <label> --confirm`); the registration never turns it.
- **`--param objective=<label>`** records another label, which must be on the
  turret, for references centred under an objective that is no longer in place.
- **A turret at an empty position is refused**: it is no objective. A stand with no
  turret records none, and refuses an `objective` that was asked for, since it cannot
  be checked. A turret that cannot be read fails the stage; it is not taken as "no
  objective".
- The plate page registers with the objective in place and offers no such field.
- `plate-finding` and `plate-walls` fill the same field, with the objective of their
  survey and of their frames.

`goto-well` and `scan-plan` use it: they move the plate by the offset between this
objective and theirs, which the profile holds (`[objectives]`, see
[`profiles/README.md`](../../profiles/README.md#objectives-offsets-between-objectives)).

### What it refuses

A refusal ends the stage `failed`, with no result: **no unusable `plate` fills the
slot**, and a stage that reads it stops with the reason.

| It says | Cause, and what to do |
|---|---|
| `no reference: give refs (a centred well) or walls (wall touches)` | Neither was given. |
| `A1 is given both as a reference and by walls: give it one way` | The same well in `refs` and in `walls`. |
| `well A1 given twice` | A well twice in `refs` (`A1` and `a1` are one). |
| `well 'Z99' is not on the 96-well plate` | A name the plate does not have. |
| `the measured well spacing is off the 384-well definition by +100.0%: wrong plate type, or a misidentified well` | The measured distance between the references is not the plate's, by more than 2 %: check the plate type and the well names. Without this check the fit would succeed and put every position further off across the plate. |
| `the fit turns the plate by 179.8°: it lies turned 180° from the declared orientation (not-turned), or two reference wells are swapped; if A1 is at the other corner, declare orientation=turned-180, else check the well names` | The fit is within `max_rotation_deg` of the *other* orientation. The plate lies the other way round (declare it), or two references were swapped, which fits at 180° exactly: see [Which way the plate lies](#which-way-the-plate-lies). |
| `the fit turns the plate by 90.0°, beyond ±10°: a reference well is misidentified` | The fitted tilt from the declared orientation is past `max_rotation_deg` (on `turned-180` the text says `by 17.5° from turned-180`). **Two references the same distance apart** (A1 and A2, or A1 and B1) fit a turn of 90° with no residual, so neither the scale check nor the residual can see a misnamed one: a bound on the rotation does. Raise `max_rotation_deg` only for a plate that really is tilted in its holder. |
| `H1 is 348 µm from where the others put it: centre it again` | A reference is further than 300 µm from the fit of the others: centre that well again. |
| `A1's -X and +X walls are 9200 µm apart, a 96-well well is 6400 µm (+43.8%): a touch of another well, or off the wall; touch them again` | An opposite pair whose size is off the plate's by more than 10 %: touch both walls again. |
| `B3's -X and +X walls are swapped: the +X wall is 15599 µm on the -X side of the -X wall, a 24-well well's width; touch them again, each named by its direction from the well's centre` | The pair has a well's size with the wrong sign: the two walls were named in the wrong frame (the image's `left` is not always the stage's `-X`, see [Wall touches](#wall-touches)). Name each by the stage direction that reaches it from the centre. `height` for the Y pair. |
| `side 'left' is a side on the image: a wall is named by a stage direction, -X, +X, -Y or +Y (the plate page names walls on the image and maps them with the camera calibration)` | An image word in `walls` (or in `goto-well`'s `side`), from an older document or a habit: it is refused, never translated. Use the stage direction; the table of [Wall touches](#wall-touches) reads it from the camera calibration. |
| `B3: the -X wall is given twice`, `a touch of B2 was given for A1`, `no X wall for A1: give -X and/or +X`, `no Y wall for A1: give -Y and/or +Y` | A mistake in `walls`; the text says which. |
| `objective 'Nikon 100X' is not on the turret: ...`, `no objective is in place: the turret is at an empty position; ...`, `the stand has no turret, so objective cannot be checked; leave it unset` | The objective is not one the turret can have: see [The objective the plate was registered with](#the-objective-the-plate-was-registered-with). |

### How good the centres must be

**Three wells far apart are enough, and they check each other.** A rotation and a translation
are three unknowns; three centres give six numbers, so three are left over to check the fit,
and a centre that disagrees with the other two is named. Two centres give four numbers for the
three unknowns: the one left over checks their spacing and nothing else. The numbers below
are for the lab's Greiner 24-well (`greiner-24-well`, 19.45 mm between wells) at A1
(-48 225, 28 850) µm, where the Ti2-E's holder puts it, with the centres of **B3, B5 and C3**
built from the plate's definition and moved by hand, as `tests/unit/test_plate_registration.py`
and `tests/test_plate_registration_demo.py` do:

| The centres | The fit |
|---|---|
| each off by at most 130 µm: (+100, -60), (-80, +90) and (+50, +110) µm | residual 121.6 µm, a rotation of 0.10° that the plate does not have, A1 off by 49.6 µm: accepted |
| B3 **900 µm** off in X, B5 and C3 exact | residuals B3 572.8, B5 351.5, C3 247.2 µm: refused, `B3 is 573 µm from where the others put it: centre it again` |
| each about 290 µm off (hand centring at the limit) | 300.5 µm: also refused |

- **The limit is 300 µm** (`GOOD_RESIDUAL_UM`): a fit whose worst residual is beyond it is
  refused. It is a limit on the *worst* centre, so three centres that are each 130 µm off
  give 121.6 µm, and a single bad one stands out.
- **By eye is only as good as the eye.** The tool needs centres good to a few hundred µm,
  which an operator cannot judge in the middle of a 15.6 mm well at the 10x (the field is
  about 1.4 mm). A small residual says the centres agree with each other, not that each is
  the well's centre: an error the three share (an eye that always centres on the same side of
  the field) moves A1 with it and shows no residual.
- **The tool that takes a rough position and measures** is [`plate-walls`](plate-walls.md):
  its rough position may be up to half the well's radius from the centre (`search_um`:
  3 900 µm on a 24-well), about ten times the error of the holder's estimate. The plate page offers it
  first once two or more centres are captured. Whether it registers the Greiner plate from one
  rough position in each of three wells, and the time it takes, is **verified at Ti2-E session 4**
  (#404), not here.
- **An estimate is not a centre.** The plate page draws a plate where the profile's
  [`[holder]`](../../profiles/README.md#holder-where-the-holder-puts-a-plate) says it lies,
  labelled *estimated*. That is for finding the way, and it fills no slot: this tool takes
  centres that were measured. Only a deliberate drive acts on it (`goto-well` with
  `plate=`, below).

## `goto-well`

```text
smc run goto-well -p <profile> --run-doc run.json --param well=H12
smc run goto-well -p <profile> --run-doc run.json --param well=B2 --param side=+Y
```

| Parameter | Default | Meaning |
|---|---|---|
| `well` | (required) | The well to go to, as `useq` names it (`H12`; `h12` and `H012` are read as `H12`). |
| `side` | `centre` | `centre`, or the point on a wall named by a **stage direction**: `-X`, `+X`, `-Y`, `+Y` (`+Y` is towards row A on a plate that is not turned; `+X` is the stage's +X on either orientation). `left`, `right`, `top` and `bottom` are sides on an image and are refused, see [Wall touches](#wall-touches). The result echoes the stage direction. |
| `plate` | none | Used only when the run document has no usable `plate`: the `useq` name of the plate in the holder (`96-well`, `greiner-24-well`), to drive to the holder's estimate. With a usable `plate`, it must name the registered plate, or the run is refused. |
| `orientation` | `not-turned` | Used only with the estimate: which way the plate lies (`not-turned`, `turned-180`). A registered plate's own orientation is used otherwise. |

It sends **one absolute move** and nothing else, and the layer's guards are the
guards: a target outside the profile's XY soft limits is refused and the stage does
not move (a soft limit cannot be forced), and a refusal, a halt or a fault of the
move ends the stage `failed` and reaches the caller as the error it is. A stop asked
for before the move (a UI's Stop, Ctrl-C) is recorded `stopped` and nothing is sent;
once the move is sent, only the layer's own handling of Ctrl-C can end it. A well the
plate does not have is refused before any move. The result keeps
the target (`target_x_um`, `target_y_um`) beside where the stage says it arrived
(`x_um`, `y_um`, the readback after the move), so a stage that stopped short shows.

**The target is in the frame of the objective in use.** `goto-well` reads the turret
(a turret at an empty position is refused before any move) and moves the plate from
the objective it was registered with to the one in place, by the offset between them
in the profile's `[objectives]`, before it works out the target: a plate registered
with the 10x and a `goto-well` at the 40x lands where the 40x sees the well, not
where the 10x did. Only XY moves; no Z offset is applied (the Ti2-E corrects Z on a
turn). The result says what happened:

| `objective`, `plate_objective`, `shift_um` | Case |
|---|---|
| the same objective twice, `[0.0, 0.0]` | The objective in place is the one the plate was registered with: nothing is shifted, and no line is added. |
| two objectives, `[-37.5, 12.25]` | The offset between them is measured: the target is the registration's plus the shift, and the log says at INFO what was moved and when each offset was measured. |
| `shift_um` is `null` | The plate does not record its objective, or the offset between the two is not measured (the profile's `[objectives]` has no entry or no reference), or the stand has no turret: the target is the registration's own, **not corrected**, and `assumed` says so after the plate's lines and a WARNING is logged. Unknown is never taken as zero. |

An offset is only as good as session 3's measurement of it (m2 §13.6); until then the
profile holds none and every change of objective is the third row.

**On the Ti2-E the soft limits are ±49 000 × ±29 000 µm**, a span of 98 × 58 mm,
while the corner wells of a 96-well plate are 99 × 63 mm apart. Wherever the plate
sits, some corner `goto-well`s will be refused by the layer. This was computed, not
measured at the stand (#198). A refused corner is the layer's soft limit and not a
fault of the registration: a well nearer the middle serves as a reference as well,
at a little less accuracy in the rotation.

**A profile without XY soft limits has nothing to bound the target.** Of the
profiles in `profiles/` only `nikon-ti2` declares them; `nikon-tie` and `demo` do
not. A seed given for the wrong well (A1 typed while the stage is under H1) puts every
well up to 99 × 63 mm from where it is, and `goto-well` sends the move as computed, up
to the stage's own end stops. Check the seed with `smc stage get` against the well you
named before the first `goto-well`, and add the limits to the profile (a hardware
session) before using a stand without them.

### The autofocus is switched off before the move

PFS locked in one well follows its reference over the wall into the next, whose
plane lay at least 104 µm away at the 40x (Ti2-E session 3), with nothing holding the
microscope (m2 §12.2 D2, FM-223). So `goto-well` switches an engaged autofocus off,
and reads it back, just before its move: on both paths, every time (the same well's
too), and whether it is locked or still searching. The move is then made with PFS off
and Z where PFS held it. **Engage it again in the well** (the page's *Autofocus on*,
or `smc autofocus engage`): `goto-well` never does, because at the 40x the way into a
new well is a first lock from below, after a focus by eye (m2 §12.11.3).

- It says so three ways: a WARNING, `autofocus_switched_off: true` in the result, and
  the **first** `assumed` line (`the autofocus was switched off before the move, so
  that no lock crossed a well's wall: engage it again in B2`), which the page's strip
  shows.
- An autofocus that is off is left off, and nothing is said. A stop asked before the
  move keeps the lock and sends nothing. In a dry run nothing is switched (a switch-off
  acts on the stand, even in a dry run).
- A switch-off that raises, a read of it that raises, or an autofocus that still reads
  on after it fails the run **before anything moves**: `the autofocus could not be
  switched off before the move to A2 (CoreError: ...): the stage did not move, and PFS
  may still be on: switch it off at the stand, then go to the well again`. A halted
  stand switches it off, then the move raises the halt as before.
- A scan run after it with `autofocus=if-engaged` runs without PFS unless you engage it
  again.
- `plate-walls` and `objective-offsets` switch it off once before anything moves, and
  `scan` with `autofocus=off` refuses an autofocus that is on (#478): see their pages.
  Not yet `smc stage move` and `smc stage jog`, which know no plate: switch PFS off
  before them at the 40x (m2 §12.11.3 step 3).

### Before registration: the holder's estimate

```text
smc run goto-well -p <profile> --run-doc run.json --param well=A1 --param plate=96-well
smc run goto-well -p <profile> --run-doc run.json --param well=B3 --param plate=greiner-24-well --param orientation=turned-180
```

With no usable `plate` in the run document, `goto-well` drives to the well's
**centre** where the profile's
[`[holder] plate_centre_xy_um`](../../profiles/README.md#holder-where-the-holder-puts-a-plate)
puts the plate you name: the centre of the plate's grid of wells on the holder's
point, turned by `orientation`. It is the pose the plate page draws as *estimated*,
from the same function, so the map and the drive never disagree
([`m3-plugins.md`](../design/m3-plugins.md) §5.5). It is a way to reach a well to centre
by eye, never a registration: it fills no slot any tool reads, and the record says so.

- **Only the centre.** The estimate can be a few hundred µm off, more than a wall is
  from the centre: `side=` other than `centre` is refused (`a wall is reached through
  a registered plate: ...`).
- **No objective offset.** The holder's point names no objective, so nothing is
  shifted.
- **What the record says**, in this order in `assumed`: `target estimated from the
  stand's holder, not registered: the 96-well centred at stage (400, -325) µm, not
  turned`; `plate not read: <why>` (the run document has no `plate`, or the one it has
  was a dry run, another stand's, stale ...); `no objective offset applied: the
  holder's estimate names no objective`; and, last, the dry-run line. `from_holder` is
  `true`, `plate_objective` and `shift_um` are `null`. A WARNING says the same.
- **A usable `plate` always wins.** `plate=` must then name it (`the registered plate
  is a 96-well, not a 384-well: leave plate out, or register the 384-well`), and
  `orientation` is not read.
- **Refused before any move**, and recorded `failed`: no `plate=` (`... or give
  plate=<a useq plate name> to drive to the estimate of the profile's [holder]`); no
  `[holder]` (`... and this stand's profile has no [holder] plate_centre_xy_um to
  estimate the plate from`); a plate name `useq` does not know (names are exact:
  `96-WELL` is refused); a well not on it; a turret at an empty position. A `plate`
  that is there and that its model refuses is refused as before, even with `plate=`.

On the Ti2-E, A1 of a 96-well plate estimated from the holder lies outside the soft
limits: the layer refuses it, and the page says so beside its *Drive* toggle (not
verified at the stand).

### Dry runs

`goto-well` in a dry run moves nothing, and its readback is the target: the result
says so in its last `assumed` line (`dry run: the stage did not move; x_um and y_um
are the target`). `plate-registration` moves no device, but a dry run records
`dry_run: true` like any other, so **a `plate` from a dry run cannot drive a real
`goto-well`**: it is refused with `plate-registration was a dry run; run it without
--dry-run`. A dry run may read a real `plate`.

The `plate` slot says what the registration assumed, and `goto-well` carries it into
its own result, each line prefixed with "plate: ".

## What is not here

- **No `plate.json`.** `nikon-control` kept the references in a file; here they are
  parameters, and the run document keeps them. A plate registered at the stand is
  reused by passing the same `refs` again.
- **Plate finding.** Finding the plate by itself from images is a separate plugin
  (#20), which fills the same `plate` slot.
- **Not verified at a stand.** No `goto-well` has been run on a real stage, and the
  Ti2-E's soft limits against a 96-well plate are computed from the profile, not
  measured (#198). Parcentricity of the objectives is a translation of tens of µm
  between two objectives (#340): `goto-well` and `scan-plan` apply the profile's
  offset when it is measured, and none has been measured yet (session 3, m2 §13.6).
  That it is a translation, and what the magnifier lever at 1.5x does to it, are not
  verified.
