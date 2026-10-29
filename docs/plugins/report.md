# `report`

Writes **one HTML file** for a scan: the density of the plate per cell class, the
cells and fields of each well, and the planned route. The file is static: no
script, no link to the network, nothing to install. Open it by double-click on the
microscope PC, offline, or attach it to an issue.

It reads the `scan-plan`, `scan`, `cells` and `density` slots, and the `route` slot
when there is one, and moves nothing: no stand, no frame, so a dry run writes the same
report as any other. The design is [`m3-plugins.md`](../design/m3-plugins.md) §13;
this page is how to use it.

## Running it

```text
smc run report -p <profile> --run-doc run.json
smc run report -p <profile> --run-doc run.json --param max_cells_drawn_per_well=500
```

| Parameter | Default | Meaning |
|---|---|---|
| `max_cells_drawn_per_well` | 2000 | The most cells drawn in one well's panel, 0 to 20 000 (a regular subsample). The whole page draws at most 300 000, whatever this is. |

The result names the file, relative to the run folder: `report/<12 hex>.html`. **Every
run writes a new file** and an old one is never rewritten, so a link to an earlier
report stays valid. They pile up: delete the ones you no longer need. A stop
(Ctrl-C) ends the stage before the write and leaves no file; nothing is checkpointed,
so a re-run starts over.

## What is on the page

In this order:

1. **The header.** The title, then a table of facts: the plate, the objective of the
   scan (and, when the route was planned for another one, the objective it is for),
   the frame as the **scan** took it (`60 × 80 px at 5 µm per pixel`), the wells
   scanned, the fields taken and failed, the cells found per class, the run
   document's **file name** (never its path), the profile, the time the report was
   made (**not** the date of the experiment: no record carries it), and the detector
   (weights file, the first 12 hex digits of its SHA-256, score threshold, merge
   radius, device). A scan with failed fields says so in words, in a coloured box:
   *N fields failed in M wells: their area is not in the density*.
2. **What was assumed.** Every `assumed` line of every stage, each starting with its
   stage (`density: the imaged area is the union of the scan's fields…`). This is
   where the area grid and its half-step error are stated.
3. **A plate map per class.** Every well of the plate in stage µm with A1 at the top
   left, whatever the plate's rotation. A scanned well is filled on **one scale per
   class**, white to the class's colour, linear from 0 to the largest density of
   that class on the plate, with the value written in it (not on a plate of more than
   384 wells: the tooltip and the table hold it). A well that was not scanned is
   outlined only. Click a well to go to its panel. The legend is five swatches.
4. **The density table.** One row per scanned well: fields, failed fields, imaged area,
   and per class the count and the density per mm², the detections outside the well's
   shape and, with a route, the cells of the wanted classes that no chosen field holds.
5. **The route**, with one. The wells outlined, the stops joined in order from the
   green start to the red end, the rank of each well (not on a plate of more than 384
   wells), then the number of stops and wells, the length in mm and the travel time. Without
   one the page says exactly *no route: run route to add the fields to image*.
6. **A panel per scanned well.** The well's outline; the footprints of the scan's frames
   (blue); the fields that failed (red, dashed); the cells as round dots, one colour per
   class (the colours of the maps); and with a route its chosen fields as numbered
   rectangles (the number is the stop's place in the route, from 1) joined by a dashed
   line. A caption says how many fields, how many cells are drawn of how many, and how
   many detections lie outside the well's shape and are not drawn.

### Reading a density

- A well with **no frame** (every field failed) is `no frame: unknown`: grey, `n/a`.
- A well whose fields are **smaller than the area grid's resolution** is `area below the
  grid's resolution (<g> µm)`: grey as well. Density samples the area on a grid, and a
  field under half a step contains no point, so its area is 0 and its density unknown,
  not zero (see [`density`](density.md)).
- A **measured 0** is `0` and is the palest step of the scale, paler than any other
  well but not white: white is the plate, and an unscanned well has no fill.

### Frames, cells and the cap

The footprints are the **scan's**: its camera matrix and frame shape at the position the
stage *read back*, as density counted them, not the plan's copy. The route's fields are
the **route's** own matrix and shape: another objective is another field of view.

Only the cells **inside the well's shape** are drawn, the ones density counted; the
caption gives the number on the plastic. When a well has more cells than the cap, a
**regular subsample** is drawn: the cells at `(i × n) // cap` of the cells file's order, so
the same records give the same page and the frames are thinned evenly. A cap of 0 draws
none, and says so.

The cap is also **shared**: the whole page draws at most 300 000 cells, so each of
96 wells draws at most 3 125 whatever the parameter. The dots are one path per class
and well, never an element per cell: 2 000 cells in each of 96 wells is under 5 MB as
paths against 12 MB as circles.

## When it refuses

The stage ends `failed` (its record has no result) with a reason that says what to run
again, and never draws records that do not describe one scan:

- **A density that is not the cells'.** Other labels, another number of fields or of
  failed fields in a well than the scan holds, or another number of cells than the cells
  file holds for it: a record was edited or replaced behind the runner's back.
  `run density again`.
- **A cell in a well density does not report.** `run density again`.
- **A cells file that no longer matches its result.** `run cells again`.
- **A route with a label the cells did not detect, or a stop in a well density does not
  report.** `run route again`.
- **A coordinate that is not a finite number, or is beyond 1e12 µm**, in the plate's
  registration (`run scan-plan again`), a frame or a camera matrix (`run scan again`), a
  cell (`run cells again`) or a route stop (`run route again`). A `nan` would vanish
  from the page's attributes without a word and a coordinate of 1e300 would print 303
  characters.
- **A well that is not on the plate.** `run scan-plan again`.
- **A file that cannot be written.** The reason names the file. When only the final
  rename failed (on Windows, a program holds the file) the whole page is kept in a hidden
  file next to it and the reason names it; when the disk is full it says to free some
  space. Either way, run `report` again.

## Limits

- **Size and time.** Measured on an Apple M-series laptop, Python 3.13, with the
  largest page the caps allow short of the 300 000 total: 96 wells, 29 280 fields (19 × 19
  clipped to the disc, 1412 px at 0.275 µm) and 2 000 cells in each (192 000 in all) gave
  a **5.2 MB** page built in **0.22 s**. The footprints are not capped: each is a frame
  the scan took, a TIFF on disk. The microscope PC's CPU is not measured.
- **A plate of 1 536 wells** has no value written in its wells and no rank in the route
  map: the tooltips and the table hold them.
- **A static page.** No zoom, no filter, no interactivity: the live plate page
  (Panel, #197) is for that. The hooks it can rely on are the page's `data-` attributes:
  `table.density tr[data-well]`, `svg.plate-map[data-label] g.well[data-well][data-state]`,
  `section.well-detail[data-well]` and, in its `svg`, `path.frames[data-n]`,
  `path.failed[data-n]`, `path.cells[data-label][data-drawn][data-total]`,
  `path.field[data-order][data-name]`, `polyline.route-well`; `svg.route-map` with
  `polyline.route[data-stops]`.
- **Escaping.** Every text from a record (the plate and objective names, the cell
  classes, a route stop's name, the assumptions, the title and the facts) goes through
  `html.escape`, and nothing from a record is an `id`, a class or a style.

## Not verified here

Nothing in this page was opened on the microscope PC.

- **Edge, Firefox and Safari.** The design's prototype was seen in Chrome only. The page
  uses plain inline SVG (no `foreignObject`, no gradient, no pattern) and a
  `vector-effect: non-scaling-stroke` rule so that outlines keep one pixel's width at any
  zoom. That is the one thing that is not plain SVG 1.1, and it has been seen in none of
  the three: if the lines look too thick or too thin there, that rule is the place to look.
- **Printing** the page.
- **The size and time on a real scan**, and **the cells of a real detector** in it.
