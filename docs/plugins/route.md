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
4. **Z and autofocus offset.** Each stop takes `z_um` and `pfs_offset`, **both from
   one frame**, from the frame of its well that the stage reached nearest to it (by
   the frame's readback) **and on which the autofocus had not lost the sample**: a
   frame whose `focus` is `unlocked` is skipped, because its Z is wherever the
   search had got to and its offset is one nobody read locked on this sample. Frames
   whose autofocus was `off` or `none` count: their Z is what the focus stage read,
   and their `pfs_offset` is `null`, so the stop's is. Both are `null` when the well
   has no such frame (no frame at all, or only unlocked ones), or when you image
   with another objective (below).

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
| `positions` | The stops in route order: `order` (in the whole route), `well`, `name` (`<well>_r<k>`, `k` from 0 along the well's route, three digits), `x_um`, `y_um`, `z_um` and `pfs_offset` (both from the nearest frame that was not `unlocked`, step 4; `null` when there is none), and `counts` (**every** label asked for, zeros included). |
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

When the nearest frame of at least one stop was `unlocked` (the autofocus had not
locked on it), `assumed` ends with one more line, which counts them:

```text
<n> of <total> stops had an unlocked frame as their nearest frame (the autofocus had
not locked on it): <n - m> took z_um and pfs_offset from the nearest frame that was
not unlocked, <m> have none because every frame of their well was unlocked
```

It is one line (shown wrapped here), `m` may be 0 and it still says `0 have none`.
A stop whose nearest frame was not unlocked is not counted, whether or not a farther
frame was; a scan without unlocked frames adds no line; and for **another objective**
it never appears, because every stop has no Z and no offset already.

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

## Selection prototype

`scripts/dev/select_fields.py` is the first step towards choosing fields for what makes
them **good for the experiment**, and not only for how many wanted cells they hold: the
density of the cells, the distance between them (isolated cells, or a minimum
spacing), whether the field holds single cells and not doublets and debris, how far it
is from the wall, how clean its background is. It is a **prototype to tune on real 40×
scans**, offline and with the owner looking at the chosen fields, and it is **not the
plugin**: `route` is unchanged and still picks the fields that hold the most cells.
Once the weights are settled on two or three scans, `route.choose_fields` takes a
score and its parameters (an interface change, [`m3-plugins.md`](../design/m3-plugins.md)
§11, written by the follow-up issue), with the count-based rule above as its default.

### What it does

One command reads a run document that has `scan-plan`, `scan` and `cells` and, for each
well it scanned:

1. takes the scan's **own frames** as the candidate fields (every frame the autofocus
   had not lost the sample on) and computes the features of each, from the cells in its
   footprint and from its pixels;
2. scores them with the weights it is given, picks fields greedily by score (no two
   overlapping, as in `route`), and orders them with `route`'s own `order_route`;
3. prints the ranked fields with their features, draws one PNG per well, and writes
   `selection.json`;
4. writes a **copy** of the run document in which a record named `route-prototype`
   fills the `route` slot, so that `nis-export` exports the list unchanged.

**Why the scan's frames and not `route`'s lattice.** Every feature, and the Z and the
autofocus offset of a stop, are then measured on the very frame that will be imaged
again; no candidate lies on an area the scan did not see; and what you look at on the
picture is what was scored. The features are written for any field centre, so the
follow-up can move them to a lattice. Whether a cell is in a field, and whether two
fields overlap, are `route`'s own arithmetic (the script imports its helpers), so the
prototype and the plugin cannot disagree about either.

### Running the prototype

```text
python scripts/dev/select_fields.py run.json --labels single --weight density=1 --weight isolation=1
smc run nis-export -p <profile> --run-doc run-selected.json
```

| Option | Default | Meaning |
|---|---|---|
| `RUN_DOC` | | A run document with `scan-plan`, `scan` and `cells`, each `done`, not stale, run on the scan's profile. |
| `--wells A1,B2` | every scanned well | The wells to run, in the plan's order. |
| `--labels single` | `single` | The wanted classes, comma separated, among the cells stage's classes. |
| `--weight NAME=VALUE` | `density=1`, the rest 0 | One weight, repeated for several; `NAME` is one of the six below, `VALUE` a number from 0 to 1 000 000, at least one above 0. A weight you give replaces **only its own** default: `--weight isolation=1` leaves the density at 1, and `--weight density=0` removes it. |
| `--fields-per-well` | 10 | The most fields to choose in a well (1 to 500). |
| `--min-cells` | 1 | A candidate must hold at least this many wanted cells. |
| `--min-score` | 0 | The choice stops at the first score below this (0 to 1). |
| `--wall-margin-um`, `--exclude-centre-um` | 0 | As `route`'s `wall_margin_um` and `exclude_centre_um`: a candidate whose footprint breaks either is not a candidate. |
| `--min-spacing-um` | measured | Wanted cells closer than this to another cell are not isolated; see below. |
| `--no-pixels` | | Do not read the frames: no `sharp`, no `bg`, no mosaic. |
| `--no-png`, `--png-max-px` | 1600 | No pictures; the longer side of a picture (256 to 8000). |
| `--show` | 15 | Candidates listed per well in the table (the rest are in `selection.json`). |
| `--out-dir` | `selection/` next to `RUN_DOC` | Where `selection.json` and the pictures go. |
| `--run-doc-out` | `<stem>-selected.json` next to `RUN_DOC` | The copy of the run document. It is **never** the original, and it is replaced when it exists. |

Every option is bounded and refused with `select_fields: <option> must be ...` before
any file is read. The exit code is 0 when everything was done, 1 when a file could not
be written (the table was printed, and the other files were written), 2 for a refusal.
A folder for `--out-dir` that cannot be made costs `selection.json` and the pictures,
and not the copy, which does not go in it. If what reads the table goes away (`| more`,
then `q`), the table stops and the files are written all the same.

In **PowerShell** quote a comma list: `--wells "B2,B11"`, `--labels "single,doublet"`;
unquoted, PowerShell passes the two words as two arguments.

**Iterating on the weights.** Each run reads the wells' frames again, once each: about
0.02 s for a 1412² frame with a warm cache (measured while planning, on an Apple M-series
laptop), so 300 frames are about 7 s per well, and 0.28 s for the 72 frames of 512² of
the demo-sample run below (0.15 s with `--no-png`). `--wells` cuts the wells, `--no-png`
the mosaic and `--no-pixels` the reading altogether. The microscope PC's disk and CPU
are not measured. One 8-bit mosaic per well is held until the pictures are written
(2.6 MB at 1600 pixels: 250 MB for 96 wells): for a whole plate use `--png-max-px 800`
or `--wells`.

### Reading a picture

`selection/<well>.png` is the plate seen from above **in the stage's axes** (+X to the
right, +Y up), whatever the camera's turn:

- the scan's frames as a mosaic, stretched between the 1st and the 99.5th percentile of
  the well's pixels; dark grey where no frame reached;
- the well's wall in **blue**;
- every cell as a 3-pixel dot coloured by its class (the cells stage's order: gold,
  cyan, red, light green, magenta, white), with a **second ring** when the class is
  wanted;
- each chosen field as a closed **green** outline, turned or mirrored as the camera is,
  with its number in route order at its centre, and the route as a thin yellow line;
- a caption with the well, the number of fields and the weights above 0.

The mosaic is for **looking, not measuring**: a frame is reduced by block means to the
picture's pixel and placed on the nearest one, with a placement error of at most the
cropped remainder of an edge (under one picture pixel), and a later frame overwrites an
earlier one where they overlap.

### The six features

A candidate's score is the weighted mean of six parts, each in [0, 1], over the
candidates of its well (`sum(w × part) / sum(w)`, so scaling every weight changes
nothing). The **pool** a rank or a maximum is taken over is the well's candidates before
`--min-cells`, so that a rank does not depend on how many cells a field holds.

| Weight | The part, in [0, 1] | What you may want it for | Default |
|---|---|---|---|
| `density` | the wanted cells per mm² over the pool's largest | most cells per image: what `route` prefers | 1 |
| `isolation` | the share of the field's wanted cells whose nearest other cell, **of any class and anywhere in the well**, is at least `min_spacing_um` away | cells that can be segmented one by one; a cell alone in its well is isolated | 0 |
| `purity` | wanted cells over all cells of the field | fields without doublets and debris | 0 |
| `wall` | the distance from the footprint to the wall over the pool's largest | fields away from the meniscus | 0 |
| `sharpness` | the rank of `sharp` among the pool: (candidates below + half the ties) / candidates, so 1/6, 3/6, 5/6 for three | the frame that is crisp | 0 |
| `background` | one less the rank of `bg` among the pool | a dark, clean background | 0 |

Every default but `density=1` is **0**, so with no `--weight` the prototype prefers what
`route` prefers, and every other criterion is a deliberate step of the owner. The table's
columns are `n` (wanted cells), `dens/mm2`, `iso` (the isolated count), `purity`,
`nn_um` (the median nearest-neighbour distance of the field's wanted cells), `wall_um`,
`sharp`, `bg`, `focus` and `stop` (the route number of a chosen field, `-` otherwise).

- **`min_spacing_um`** is the value given, else **three times the median length of the
  wanted cells** (the longer side of a cell's box, times the pixel's size), over every
  well run, so that wells compare. The factor 3 is **chosen, not measured**; the table
  and the result say which of the two the value is.
- **`sharp` is not a focus measure.** It is the variance of the 4-neighbour Laplacian of
  the frame divided by its median squared (so it does not depend on the exposure), and
  cells add contrast: a frame full of cells reads sharper than an empty one in focus.
  That is why the column is called `sharp`. It is not corrected.
- **`bg`** is the frame's median.

### Things to know

- **The candidates overlap, so a chosen field excludes its neighbours.** The scan's
  frames overlap by the scan plan's `overlap` (10 % by default), and two fields whose
  frames overlap are never both chosen, as in `route`. Adjacent candidates are therefore
  never chosen together unless the scan was planned with an overlap of 0, and a cell in
  the strip a rejected frame shared with a chosen one can be left in no field:
  `uncovered` counts those cells. On the demo-sample run below, 5 of the 14 wanted
  cells of B11 were left in no field with 7 of 36 candidates chosen and 10 allowed.
  `route`'s lattice (a step of a quarter of the field) has no such constraint.
- **`--min-cells` counts the candidate's own wanted cells**, not only those that no
  earlier field holds (`route`'s rule). The two differ only for a cell on an edge that
  two touching fields share, which the first chosen holds.
- A stop is the frame's **readback** position, and its Z and autofocus offset come
  together from that frame. An `unlocked` frame is never a candidate (its Z is wherever
  the autofocus search had got to), and the result says how many were left out. A frame
  that cannot be read (missing, corrupt, of another shape than the scan's) is counted,
  skipped and said; it is not a candidate for a pixel weight and stays one otherwise.
  The table gives the first reason in the system's words (`the file is missing`,
  `Permission denied`) and never a path of this PC: it is pasted into reports.
- The last line printed gives the command that exports the copy, with the scan's
  profile (`smc run` defaults to the demo, and `nis-export` refuses a route run on
  another profile). A path that is not ASCII is not printed as an escape: the line says
  to take it from the one above.
- The score and the copy are the **owner's** to judge; nothing here is tuned, and
  `assumed` in the copy's record names the weights, the spacing and that the positions
  are the scan's own frames and not a lattice.

### What it reads and leaves

- It reads the run document, the cells file and the scan's frames, and never writes
  any of them.
- `selection.json`: every candidate of every well with its features, score, parts and
  `stop` (`null` or the route number), the weights and the options. For a notebook;
  nothing in smc reads it. The run document is named by its file name, not its path.
- `<well>.png`: the pictures above. Existing ones are replaced, they are derived.
- The **copy** of the run document: every stage of the original, plus the
  `route-prototype` record that provides `route`. It is named so and not `route` so that
  the document says who chose the fields, an older `route` record (if any) stays in the
  copy, and the later start wins. The record reads the current `scan-plan`, `scan` and
  `cells`, so it goes stale when one runs again, as `route`'s does. **Keep the copy next
  to its original**: a run document's file names are relative to its folder, and a copy
  moved elsewhere loses its frames and cells (`nis-export` reads only the route, and
  still works).

### What was tried, and what the owner kept

| Scan | Weights | Spacing | What the owner saw | Kept? |
|---|---|---|---|---|
| nothing is judged yet: the first 40× scan has not been taken | | | | |

### What was run here

On the `demo-sample` stand (a synthetic plate, simulated devices; the plate registered
from its two corners, then `scan-plan --param wells=B2,B11 --param coverage=0.4`,
`scan`, `cells --param detector=blobs`): 72 frames of 512² at 1 µm, 14 single cells and 2
doublets in B11 and 3 and 4 in B2. `--labels single --weight density=1 --weight
isolation=1` ran in 0.28 s, chose 7 fields in B11 and 1 in B2, and `nis-export` of the
copy exported those 8 positions. On the pictures the numbers are on the fields the table
lists, and the mosaic is the right way up: the grey level in the mosaic around a cell's
position was 197 and 200 on average against 124 over the well (15 and 7 cells in B11
and B2), and 122 and 129 at the same cells mirrored through the well's centre (13 and 6
that fall on the mosaic). The demo's frames are
noise, so `sharp` and `bg` rank nothing here.

## Not verified here

Nothing in this page was run on a microscope.

- **The selection prototype on a real 40× scan**: every number above is from the
  synthetic sample, and the weights have not been judged on real frames (part 2 of the
  issue, with the owner, fills the table above).
- **The prototype's cost on the microscope PC** (disk and CPU), and that its numbers
  hold on Python 3.10's numpy, scipy and Pillow (the full matrix runs it).
- **That NIS-Elements loads the copy's export** (the same open point as below).

- **The stage's move time against distance** on the Ti2-E: Euclidean, or both axes at
  once (#198). Until it is measured, `travel_s` is the model's estimate and the order
  is the shortest in µm.
- **The profile's pixel sizes of two objectives** at the stand, which a scaled field
  rests on, and **the offset between the two objectives**, which is not applied.
- **That NIS-Elements visits the positions as they are** (#196).
- **The time on the microscope PC's CPU.**
