# `density`

Counts the **cells of each well, per class**, and divides them by the **area that
was imaged**: cells per mm² of single cells, of doublets, of whatever classes the
cells stage detected. The area is the union of the scan's fields clipped to the well,
so fields that overlap are not counted twice and a field that straddles the wall
counts only what is inside it.

It reads the `scan-plan`, `scan` and `cells` slots and moves nothing: no stand, no
frame, no file written. The design is [`m3-plugins.md`](../design/m3-plugins.md) §10;
this page is how to use it.

## Running it

```text
smc run density -p <profile> --run-doc run.json
smc run density -p <profile> --run-doc run.json --param area_grid_um=2
```

| Parameter | Default | Meaning |
|---|---|---|
| `area_grid_um` | 5.0 | The step of the grid on which the imaged area is counted, 0.5 to 100 µm. Smaller is finer and slower. |

A stop (Ctrl-C) ends the stage between two wells. Nothing is written and no
checkpoint is kept: the work is arithmetic on records that are already on disk, so a
re-run starts over and gives the same result.

## What it does, well by well

The wells it reports are those the scan **attempted**: it took a frame in them, or
failed a field in them, in the scan plan's order. A well the scan never reached is
not in the result, as the route leaves it out.

1. **The fields.** Each frame the scan took is a rectangle: the camera's footprint,
   turned and perhaps mirrored as the calibration measured it, at the position the
   stage **read back** after the move. The scan's own matrix and frame shape are used,
   not the plan's copy: a field is what the scan saw, and the demo stage rounds a
   command.
2. **The area.** A grid of cell centres covers the fields, cut to the well. A point
   counts when it is inside at least one field **and** inside the well's shape (a
   disc, or a rectangle turned by the plate's rotation, the edge included). The area
   is the number of points times the step squared.
3. **The counts.** Each cell of the well is tested against the same shape. A cell
   inside is counted in its label; a cell outside (a field's corner on the plastic)
   is `outside_well` and in no label. So a well's counts and `outside_well` always add
   up to its cells.
4. **The density.** `per_mm2` is each label's count divided by the area.

**The density is per imaged area.** A field that failed is in neither side of the
ratio: the cells it would have shown are not in the count, and its area is not in the
area. A well whose every field failed is reported with `fields` 0, its `failed_fields`,
an area of 0 and `per_mm2` `null`: with no frame the density is unknown, not zero.

## What it leaves

The slot `density` holds a `DensityResult` (`smc.plugins.slots`):

| Field | Meaning |
|---|---|
| `wells` | One entry per reported well, in the plan's order: `well`, `fields` (frames taken), `failed_fields`, `area_mm2`, `counts` (**every** label, zeros included), `outside_well` and `per_mm2` (the same labels, or `null` for an area of 0). |
| `labels` | The cells stage's classes, in its order. |
| `area_grid_um` | The step the area was counted on. |
| `assumed` | What the result rests on that was not measured (below). |

## What it assumes

`assumed` always says two things.

- **The area is rasterised.** An edge of it is known to within **half a step**. A
  single field has the error of its edges, up to `g / (2 w)` on each axis, where `g`
  is the step and `w` the field's side: +1.3 % for a field of 333 µm at 5 µm
  (measured). The area of a **well** is better, because the errors of many edges do
  not add: a 96-well disc covered by 305 fields gave 32.1703 mm² at 5 µm for a
  π r² of 32.1699 mm² (+0.001 %). Do not read the area of one field to better than
  a percent or two at the default step; lower `area_grid_um` for it. **Keep the step
  well under the field's shorter side, about a tenth of it**: the points are cell
  centres, so a field smaller than half a step contains none and counts as no area,
  and the well is then reported with `fields` 1, an area of 0 and `per_mm2` `null`,
  as if no frame had been taken. At the top step of 100 µm a 40 µm field gives 0.0 mm²
  (true 0.0016) and a 130 µm field 0.0100 (true 0.0169, 41 % low); at the default 5 µm
  both are exact, and the edges of a field of a 40× objective (about 388 µm) are known
  to `g / (2 w)`, 0.6 % per axis at 5 µm.
- **The wells' shapes are the plate registration's.** Their positions are the
  registration's and their sizes the plate's nominal ones: the wall was **not
  measured**, so the area and the cells near it are only as exact as the registration.
  A well off by a few tens of µm moves cells across its edge into `outside_well`.

## When it refuses

The stage ends `failed` (its record has no result) with a reason that says what to
run again, and never gives a density that rests on records that no longer fit:

- **A scan with no frame** (a dry run, or every field failed): no area was imaged.
  `run scan again`.
- **A camera matrix that cannot be inverted.** The scan's matrix validates field by
  field, so a singular one (a run document edited by hand) would otherwise be a field
  of no area, far from its cause.
- **Cells and scan that are not the same run.** The cells stage counted another
  number of frames than the scan holds, or other wells: the scan was run again after
  the cells, and both records still validate. `run cells again`.
- **A cells file that no longer matches its counts**, or that holds a label the
  classes do not name. `run cells again`.
- **A well the scan holds and the scan plan does not list**, or one that is not on the
  plate. `run scan-plan, scan and cells again`, or `run scan-plan again`.
- **A grid too fine for a well.** The grid over a well's imaged area is bounded at
  2¹⁷ points on a side and 2³⁰ in all, and the counting at 2³¹ tests, checked before
  anything is allocated; the reason names the well and says `raise area_grid_um`.
  A 96-well well is never too fine (at the finest step, 0.5 µm, it is 12 800 points
  on a side), but a 6-well well (34.8 mm) fully covered is refused below about 1.07 µm (more than 2³⁰ points in all).
- **A footprint that is not a field**: a readback beyond 1e12 µm (1000 km of stage),
  or a frame whose corners are not a parallelogram of some area. The reason names the
  well and the footprint by its **position among that well's frames in the scan's
  order, counting from 0** (`well A1: footprint 2 ...`), not by the frame's name: the
  `index` and `name` of a frame count the planned fields, gaps for the failed ones
  included, so the third frame taken can be `A1_0003`.
- **A scan plan that lists a well twice**: `run scan-plan again`.

A frame read back **far** from its well (a wrong readback of 1e9 µm, say) is not
refused: the grid is cut to the well, so such a frame touches nothing and adds no area
and no work.

## Limits

- **Speed.** Measured on an Apple M-series laptop, numpy 2.5.x on Python 3.13: a
  96-well well covered by 293 fields of 388 µm took 0.01 s at the default 5 µm and
  0.48 s at 0.5 µm, with about 40 MB above the baseline. A 6-well well (34.8 mm) with
  7845 fields took 1.1 s at 2 µm. The bounds refuse what would be worse; at the
  largest legal work (2³¹ tests) it is 3 to 5 s on that laptop, more on a slower PC.
  The microscope PC's CPU is not measured.
- **Stop.** A stop is looked at between two wells, not inside one: the raster of a
  well cannot be interrupted, so a stop waits for the well under way. Ctrl-C ends the
  stage once that well is done.
- **The run document's size.** An entry per well is small (the counts of a few labels):
  96 wells are well under 100 KB.

## Not verified here

Nothing in this page was run on a microscope.

- **The area and the density on a real scan** at the Ti2-E, against a count by eye on
  a few wells.
- **That the registration's wall is the real wall** (#198): the cells near the wall,
  and the area a field takes there, rest on it.
- **The time on the microscope PC's CPU.**
