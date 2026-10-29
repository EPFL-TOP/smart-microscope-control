# `scan-plan`

Which wells to image, and which fields of each, as numbers: a grid of fields in
every chosen well, sized from what the camera really sees, in the order a scan
will visit them. It **moves nothing**: it reads the camera's frame and pixel size
and the turret, so it runs in a dry run like at any other time. The scan, which
takes the frames, reads this plan and not the wells again. The design is
[`m3-plugins.md`](../design/m3-plugins.md) §7; this page is how to use it.

It needs the plate to have been registered: the `plate` slot of the run document
(filled by plate registration, #19, or plate finding, #20) says where the wells
are and how the plate is turned.

## Running it

```text
smc run scan-plan -p <profile> --run-doc run.json --param wells=A1:B6
smc run scan-plan -p <profile> --run-doc run.json --param wells=all --param coverage=0.33
smc run scan-plan -p <profile> --run-doc run.json --param wells="A1, C*, *12" --param rows=5 --param columns=5
smc run scan-plan -p <profile> --run-doc run.json --param wells=B2 --param objective="Nikon 40X Plan Fluor ELWD"
```

| Parameter | Default | Meaning |
|---|---|---|
| `wells` | (required) | The wells to image: see [Choosing wells](#choosing-wells). Put the whole `--param` item in quotes when the list has a comma or a space (`--param "wells=A1, C*"`): PowerShell 5.1 may read an unquoted comma as its array operator and split the argument in two. Not tried on a Windows shell: check it at the stand. |
| `serpentine` | true | Visit alternate plate rows in reverse: a raster drives back across the plate at the end of every row, a serpentine does not (for rows next to each other: see [Choosing wells](#choosing-wells)). |
| `objective` | the one in place | The objective to plan for, by its label on the turret. **The turret is never turned**: this is only whose pixel size the plan uses. |
| `overlap` | 0.1 | Fraction of a field shared with its neighbour, from 0 to under 0.9. A cell on a boundary then appears whole in one of two fields. |
| `coverage` | 1.0 | Fraction of the well's width the grid spans: 1 is edge to edge, 0.33 the middle third (up to 1.5, which runs past the well). Ignored when `rows` and `columns` are given. |
| `rows`, `columns` | derived | Fields along Y and along X, 1 to 100 each. Give both or neither. |
| `clip_to_well` | true | Keep only the fields whose frame meets the well: see [Clipping](#clipping-to-the-well). |
| `pixel_size_um` | none | Plan with this pixel size (rotation 0, no mirror) instead of a calibration's or the profile's. |

The demo has no plate, so the run document needs a `plate` stage first (here one
written for the occasion: a 96-well plate with A1 at `(1000, -500)`). The result of a
run on the demo, with the 10× in place and the camera's own 1.0 µm per pixel, looks
like this (abridged):

```text
$ smc run scan-plan -p demo --run-doc run.json --param wells=A1:B2 --param rows=2 --param columns=3
✓ scan-plan done in 0.0 s — run.json, stage scan-plan
{
  "assumed": [
    "no camera calibration for Nikon 10X S Fluor: rotation 0 and no mirror assumed"
  ],
  "plate_plan": { ... },
  "wells": ["A1", "A2", "B2", "B1"],
  "objective": "Nikon 10X S Fluor",
  "pixel_size_um": 1.0,
  "matrix_source": "profile",
  "image_shape_px": [512, 512],
  "fov_um": [512.0, 512.0],
  "overlap": 0.1,
  "rows": 2,
  "columns": 3,
  "fields": [
    { "well": "A1", "index": 0, "name": "A1_0000", "x_um": 539.2, "y_um": -269.6 },
    ...
  ],
  "estimate_s": null
}
```

## Choosing wells

Items are separated by commas or spaces and read in this order:

| Item | Means |
|---|---|
| `A1` | One well. `a01` is `A1`. |
| `A1:D6`, `B2-B5` | The rectangle between two corners, in any order. The dash is the port's spelling and reads the same: `B2-D2` is a run down a column, not along a row. |
| `C*` | Every well of row C. |
| `*3` | Every well of column 3 (`*03` too). |
| `all` | Every well (`ALL` too). |

Names are the plate's own, taken from `useq`: `AA3` is a well of a 1536-well plate,
and `A*` is the row `A`, not every name that starts with the letter. A well name
with a space in it (`A 1`) is two items, because a space separates items: write
`A1`.

**What cannot be read is an error, not a skip.** An item that is not in the table, or a
well that is not on the plate, fails the stage with one message that lists every such
item, and nothing is planned:

```text
$ smc run scan-plan -p demo --run-doc run.json --param "wells=A1, Q99"
✗ scan-plan: wells not understood on a 96-well plate: 'Q99'; give wells as A1, B2-B5, A1:D6, C*, *3 or all
```

(`nikon-control` skipped what it could not read and left the caller to compare the
count it asked for with the count it got; nobody did, and a wrong well set is only
found hours into an overnight run.) A text that selects nothing fails with `no wells
selected`. Duplicates are removed, keeping the first.

**The order you type is not the order of the scan.** The wells are put in plate order
(by rows), and with `serpentine` every other **plate row** is reversed: `B2, B1, A2, A1`
is visited as `A1, A2, B2, B1`. The result's `wells` and the `useq` plan's
`selected_wells` are in that order, and so is everything downstream (the scan, the
route). The reversal goes by the row's place on the plate (B, D, ...), not by its place
in the visit, so rows that are not next to each other all run left to right: `A*, C*`
visits `A1`..`A12` and then `C1`..`C12`, and the stage drives back across the plate
between them (298.6 mm of travel against 216.0 mm if `C` ran in reverse). Select
neighbouring rows, or one row at a time, to keep the serpentine's saving.

## Where the pixel size and the matrix come from

Every field position, and later every cell turned into a stage position, rests on the
camera matrix (rotation, mirror and pixel size). The plan takes it from the first of
these that exists, and **says which in `matrix_source` and what it assumed in
`assumed`**:

1. **`pixel_size_um` given**: `"parameter"`, an upright camera with no mirror. `assumed`:
   `pixel size given as a parameter; rotation 0 and no mirror assumed`.
2. **A `camera-calibration` of the same objective and frame shape**: `"camera-calibration"`,
   measured, rotation and mirror included. One of another objective or frame is not used,
   and `assumed` says why:
   `camera calibration not used: it is of <its objective>, the plan is for <this one>`,
   or `camera calibration not used: it is of a frame of (512, 300), the camera gives (300, 512)`
   (shapes are `(height, width)`; `an unknown objective` stands for a stand without a turret).
3. **The stand's configured pixel size**: `"profile"`, an upright camera with no mirror.
   For the objective in place (or a stand without a turret) it is the **camera's own**
   size, which includes the binning and the magnifier. For **another objective** it is
   the turret's, which is the profile's entry **for binning 1**, and `assumed` says so:
   `pixel size of <objective> from the profile, for binning 1: the camera's binning is not applied`.
   Either way: `no camera calibration for <objective>: rotation 0 and no mirror assumed`
   (`<objective>` reads `this stand` without a turret).
4. **None**: the plan is refused.

```text
$ smc run scan-plan -p demo --run-doc run.json --param wells=A1 --param objective=Objective-2 --param rows=2 --param columns=2
✗ scan-plan: no pixel size for Objective-2: run camera-calibration with it in place, or add it to the profile's [camera] pixel_size_um
```

A pixel size of 0 means "not measured" and is never defaulted; a size that is not
finite and positive (`nan`, a negative number, infinity, which is above 0) is refused
the same way (FM-104). `objective` must be a label of the turret
(`objective 'Nikon 60X' is not on the turret: ...`), and on a stand with no turret it
cannot be checked, so it must be left unset. A turret at an **empty position** has no
objective in place, so a plan for "the one in place" is refused (`no objective is in
place: the turret is at an empty position; ...`) and one for a named objective is not.

A calibration is used only when its matrix can be inverted, because the cells and the
route turn a stage position into a pixel through that inverse: a singular one (a run
document edited by hand, another tool filling the slot) is refused with
`the camera matrix cannot be inverted (determinant ...); run camera-calibration, or
give a pixel size`. Nothing else checks that a calibration agrees with itself: its
`rotation_deg` and `pixel_size_um` are copied as they are, and the field of view comes
from its matrix.

**What is not read.** Nothing in the camera interface reads the binning, and the Ti2-E's
intermediate-magnification lever (1.0× or 1.5×) is not read either; the profile's sizes
are for a fixed value of both (FM-103). A plan for an objective that is not in place does
not know the binning of the camera, and a **calibration measured with the lever or the
binning at another value is used as if it matched** when objective and frame shape
agree. Calibrate with the stand as it will be scanned.

### A turned camera leaves gaps

The grid is **axis-aligned in the stage frame** and its step is the extent of the
camera's footprint, so a camera turned by a few degrees leaves gaps between fields at
the corners. With a calibration, the plan checks the rotation folded to the nearest
multiple of 90° (a camera at 90° or 180° fills its box) and, when gaps open, says so
in `assumed` and what to do:

```text
the camera is turned 3.5° from the stage axes: at overlap 0.1 the grid leaves gaps between fields; use overlap=0.13
the camera is turned 30.0° from the stage axes: no overlap below 0.9 closes the gaps between fields; straighten the camera
```

Gaps are reported when `sin φ > overlap / 2`, which is from 2.9° at 10 % overlap, while
the design measured the first real gap on a square frame at 3.8°
([`m3-plugins.md`](../design/m3-plugins.md) §7.3). A matrix from the profile or a
parameter assumes no rotation and never says this.

**The rule is that of a roughly square frame.** It takes only the rotation, and on a wide
frame gaps open sooner than it says. Measured on a 512 × 2048 frame (a wide region of
interest) at 10 % overlap: 4.8 % of the area the grid spans is imaged by no field at
2.5°, where the check says nothing, and at 3.0° it suggests an overlap of 0.11 and 6.2 %
is still uncovered at that overlap. A 300 × 512 frame, the shape of the demo camera used
in the tests, is at 0 % at both 2.5° and 3.5°. With a wide region of interest, straighten
the camera as far as you can, and do not take the absence of a line for a proof that the
grid is closed. (The check should take the frame's shape: a follow-up, since its
signature is the design's.)

## The grid

`rows` and `columns` default to the number of fields that span `coverage` of the well,
from the field of view in stage axes (width for columns, height for rows) and the well
size from `useq`: per axis `ceil((well × coverage − field) / (field × (1 − overlap))) + 1`,
and one field for a well that a single field already covers. On a 96-well plate (a 6400 µm
well) at 40× with a 332.8 µm field, that is 20 × 20 at 0 % overlap and 7 × 7 for
`coverage=0.33`. The camera's frame is rarely square, so rows and columns differ.

A default that is **derived from the pixel size is derived from the very value that may
be wrong**, so a derived grid over 100 per axis is refused with the pixel size it came
from, instead of a plan of a million fields:

```text
$ smc run scan-plan -p demo --run-doc run.json --param wells=A1 --param pixel_size_um=0.001
✗ scan-plan: the grid would be 13889 × 13889 fields per well at 0.001 µm per pixel: check the pixel size, or give rows and columns
```

The 100 bounds **a well**, not the plan: the grid is built for every well before it is
clipped, at about 1.4 KB a field. Measured on a 96-well plate, `wells=all` with
`rows=100` and `columns=100` is 960 000 fields built in 8.6 s and 1.4 GB of memory (106 752
kept after clipping), and a 384-well plate would need about four times that. Nothing yet
bounds the total (a follow-up); give the wells and the grid you mean.

### Clipping to the well

A field that misses the well images only plastic. With `clip_to_well` (the default) a
field is kept when its **frame** meets the well, not when its centre is inside: for a
round well, when the distance from the well's centre to the frame is below the radius;
for a square well, when the frame and the well's rectangle (turned by the plate's
rotation) are not separated. The shape of the well is `WellGeometry`'s, the one place
that knows it. On a 96-well plate at 40× with the Ti2-E's assumed camera (1412 × 1412
pixels of 0.275 µm) and 10 % overlap, a 19 × 19 grid keeps 305 of its 361 fields.

`index` counts from 0 in each well over the fields that were kept, and `name` stays
`useq`'s (`A1_0003`), so a gap in the names is a clipped field. A well that no field
touches is refused (`no field of the grid touches well A2: ...`) rather than left out of
the scan without a word.

## The estimate

`estimate_s` is the time of the scan in seconds, **only when a `timings` stage is in the
run document**: each stage move between two fields (the first field costs none), a refocus
when the stand has an autofocus and the timings know its cost, and each snap, as
measured, and what the timings assumed (a refocus that was not timed, soft limits the
stage did not report) is named in `assumed`, one line each starting with `timings:`.
Without timings it is `null`: unknown, not guessed. A dry run is the same plan, and
gives the same estimate.

## The result

The slot `scan-plan` holds a `ScanPlanResult` (`smc.plugins.slots`): `plate_plan` (a
`useq.WellPlatePlan` of the registered plate with the wells in visiting order and the
`GridRowsColumns`), `wells`, `objective`, `pixel_size_um`, `matrix_um_per_px`,
`matrix_source`, `image_shape_px` (`(height, width)`), `fov_um` (`(width, height)` in
stage axes), `overlap`, `rows`, `columns`, `fields` (`well`, `index`, `name`, `x_um`,
`y_um`, in visiting order), `estimate_s` and `assumed`. A failed run fills no slot.

## Not verified here

Nothing in this page was measured on a microscope. The Ti2-E's pixel size at 40× (0.275 µm
per pixel is the datasheet's, assumed) and its camera's rotation, and what a plan costs
there, are for the first hardware session (#198).
