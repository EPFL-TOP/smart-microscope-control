# `route`

Chooses **which fields to image** and **in what order**, from the cells that a scan
found: in each well, the fields of view that hold the most cells of the classes you
want, and then the shortest path through them, well after well. The result is the
list of stage positions that the second, high-magnification pass visits.

It reads the `scan-plan`, `scan` and `cells` slots (and `timings` when there is
one), and moves nothing. It asks the stand for nothing but the objective turret's
labels and the profile's pixel sizes, and only when you ask for another objective
than the scan's. The design is [`m3-plugins.md`](../design/m3-plugins.md) §11; this
page is how to use it.

## Running it

```text
smc run route -p <profile> --run-doc run.json
smc run route -p <profile> --run-doc run.json --param 'labels=["single","doublet"]'
smc run route -p <profile> --run-doc run.json --param fields_per_well=4 --param min_cells=3
smc run route -p <profile> --run-doc run.json --param wall_margin_um=300 --param exclude_centre_um=500
smc run route -p <profile> --run-doc run.json --param objective="Nikon 40X Plan Fluor ELWD"
```

| Parameter | Default | Meaning |
|---|---|---|
| `labels` | `["single"]` | The cell classes that count; JSON for more than one. Each must be one of the cells stage's classes, and none may repeat. |
| `fields_per_well` | 10 | The most fields to choose in a well (1 to 500). |
| `min_cells` | 1 | A field must hold at least this many cells that no earlier field holds. |
| `exclude_centre_um` | 0 | The **radius** of the disc at each well's centre that no field may touch. |
| `wall_margin_um` | 0 | The distance from the wall that every corner of a field keeps. |
| `objective` | the scan's | Image with this objective, by its label on the turret; see below. |
| `candidate_step_fraction` | 0.25 | The spacing of the candidate fields as a fraction of the field's shorter side, above 0 and up to 1. Smaller is finer and slower. |

A stop (Ctrl-C) ends the stage between two wells. Nothing is written and no
checkpoint is kept: the work is arithmetic on records that are already on disk, so a
re-run starts over and gives the same route (nothing in it is random).

## What it does, well by well

1. **Candidates.** A square lattice of `candidate_step_fraction` times the field's
   shorter side, centred on the well, over its bounding box. A candidate is kept when
   its **footprint** is allowed: every corner of the camera's frame is inside the well
   shrunk by `wall_margin_um`, and, with `exclude_centre_um`, the frame stays at least
   that far from the well's centre (a frame that covers the centre is at distance 0).
   The footprint is the camera's rectangle, turned and perhaps mirrored as the
   calibration measured it, in the well's own shape: a disc, or a rectangle turned by
   the plate's rotation.
2. **Greedy choice.** Each round takes the candidate that holds the most wanted cells
   **that no field already chosen holds**. Ties go to the lowest-numbered candidate,
   which is the one nearest the top of the well (+Y), then the leftmost (-X). It stops
   at `fields_per_well` fields, or when the best candidate would hold fewer than
   `min_cells` new cells. Every candidate whose frame would overlap the chosen one is
   removed, so **no cell is imaged twice**. Fields that only touch are allowed, and a
   cell exactly on the edge two of them share is counted once, in the first.
3. **Order.** Nearest neighbour from a starting field, then 2-opt on the open path
   with its first field fixed. The first well starts at its field nearest the first
   field the scan plan planned in it; each later well starts at its field nearest the
   previous well's last stop. The wells follow the scan plan's order, and the legs
   **between** wells are part of the route and of its length.
4. **Z and autofocus offset.** Each stop takes `z_um` and `pfs_offset` from the frame
   of its well that the stage reached nearest to it (by the frame's readback), or
   `null` for both when the well has no frame, or when you image with another
   objective (below).

A field holds the cells inside its frame, the edge included. A cell within a rounding
error of an edge (a millionth of a pixel) may fall on either side, and it is counted
once whichever it is: the count and the choice use the same arithmetic.

`min_cells` is the number of *new* cells, and `counts` of a stop are the cells it
holds that no earlier stop holds: the counts of all stops and what is uncovered add
up to the cells of the wanted labels, always.

## What it leaves

The slot `route` holds a `RouteResult` (`smc.plugins.slots`):

| Field | Meaning |
|---|---|
| `positions` | The stops in route order: `order` (in the whole route), `well`, `name` (`<well>_r<k>`, `k` from 0 along the well's route, three digits), `x_um`, `y_um`, `z_um`, `pfs_offset` and `counts` (**every** label asked for, zeros included). |
| `uncovered` | Well to the number of wanted cells that no stop holds. **Every well the cells stage scanned** is a key, zeros included, in the plan's order; a well it did not scan is absent. |
| `distance_um` | The length of the whole route, between wells included. |
| `travel_s` | The sum of `timings.move_ms` over the same legs; `null` when no `timing` stage ran. Unknown is not zero. |
| `matrix_um_per_px`, `image_shape_px`, `fov_um` | The camera the fields were chosen for: the scan's (scaled for another objective), its frame, and the extent of its footprint in stage axes. |
| `objective`, `labels` | The objective the route is for, and the labels as given. |
| `assumed` | What the result rests on that was not measured (below). |

`RouteResult.to_useq()` gives the stops as `useq` positions, for the NIS-Elements
export.

## Another objective

`objective` names an objective on the turret to image with. Its field is **the scan's
matrix times the ratio of the profile's pixel sizes** (`size(objective) / size(scan's
objective)`), the same frame of pixels, so a 40× pass from a 10× scan has a field a
quarter of the width and height. The turret is never turned: this only reads its
labels and sizes. It needs the scan to have recorded its objective, a turret, the
label on it, and a profile size for **both** objectives (`0.0` is "unknown", and is
refused, not defaulted). The scan's own objective given by name changes nothing and
needs no turret.

For another objective the result says, in `assumed`:

- `the field of <objective> is the scan's scaled by the profile's pixel sizes`;
- `no offset between the two objectives is applied`: two objectives on one stand do
  not image the same point at the same stage position, and nothing here knows by how
  much;
- `z_um and pfs_offset are unknown for <objective>: the scan's were measured with
  <scan objective>`, and they are `null` on every stop.

## What it assumes

`assumed` always starts with `the route minimises the Euclidean distance between
fields; whether the stage's move time follows it is not measured`. The timing model
is an overhead per move plus a rate per µm, so the shortest path is the fastest
under it, **if** the stage's time grows with the Euclidean distance. A stage whose X
and Y move at once (a time nearer the larger of the two) would prefer another
route. Which one the Ti2-E is has not been measured (#198). The 2-opt is a
heuristic: the route is never longer than nearest neighbour's and is not claimed to
be the shortest.

For a camera that is turned by the stage, the lattice's step and `fov_um` use the
bounding box of the turned frame, as `scan-plan` does, so the step is larger than a
quarter of the frame's shorter side. The overlap rule itself uses the frame's own
pixels, so the footprints that it keeps apart are the turned rectangles.

## When it refuses

The stage ends `failed` (its record has no result) with a reason that says what to
do, and never gives a route that is empty or that rests on a number that is not one:

- **A label that the cells do not have**: `label 'x' is not one of the cells' classes
  [...]; give labels among them`.
- **No cell of the wanted labels** (`the cells stage found no single cell: there is
  nothing to route; ...`), or **no field that holds `min_cells` of them** and keeps
  to the wells' margins (`no field of 400 by 300 µm holds 50 or more single cell ...;
  lower min_cells, wall_margin_um or exclude_centre_um, or check the scan's pixel
  size`).
- **An objective that cannot be scaled**: not on the turret (the labels are listed),
  no turret, no profile size for it or for the scan's objective, or a scan that did
  not record its objective.
- **A camera matrix that cannot be inverted**, or a cells file that no longer matches
  its counts (`...; run cells again`), or cells of a well the scan plan has no field
  for.
- **A lattice or a count that is too large.** The candidate lattice of a well is
  bounded at 250 000 points and the cell-by-candidate count at 2³¹ tests, checked
  before anything is allocated. The lattice is derived from the scan's pixel size, the
  value most likely to be wrong: a profile entry 1000 times too small would be a
  million candidates per well, and the reason names the **size of the field** it was
  derived from (`the field is 0.4 by 0.3 µm: check the scan's pixel size, or raise
  candidate_step_fraction`).

## Limits

- **Speed.** Counting is per cell over the few lattice points around it, so it grows
  with the cells and not with the well. Measured on 300 000 cells in one well, a
  40× field (1412² pixels at 0.275 µm), 10 fields, numpy 2.5.4 on Python 3.13 on an
  Apple M-series laptop: 0.29 s in a 96-well well, 0.27 s in a 24-well one and 0.45 s
  in a 6-well one at the default step. A finer `candidate_step_fraction` costs more: 0.05 took 1.5 s in the
  96-well well, and in the 24-well and 6-well wells it is refused, because their
  lattice would hold more than 250 000 candidates. The bounds refuse what would be
  worse; they do not make it fast. Raise the fraction rather than waiting. The
  microscope PC's CPU is not measured.
- **Stop.** A stop is looked at between two wells, not inside one. A well that the
  bound still lets through (760 000 cells with a window of 2 809 lattice points each,
  which is a 1331 µm field at a step of 0.02 of it) took 17 s and about 1 GB on the
  same laptop, so a stop can wait that long for the well under way; Ctrl-C ends it at
  once.
- **The run document's size.** The stops are in the run document, and the document is
  rewritten at every stage write. At the default of 10 fields per well, 96 wells are
  about 0.24 MB; at the largest `fields_per_well` (500) they are about 12 MB.

## Not verified here

Nothing in this page was run on a microscope.

- **The stage's move time against distance** on the Ti2-E: Euclidean, or both axes at
  once (#198). Until it is measured, `travel_s` is the model's estimate and the order
  is the shortest in µm.
- **The profile's pixel sizes of two objectives** at the stand, which a scaled field
  rests on, and **the offset between the two objectives**, which is not applied.
- **That NIS-Elements visits the positions as they are** (#196).
- **The time on the microscope PC's CPU.**
