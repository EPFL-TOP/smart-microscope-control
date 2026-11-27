# `low-mag-survey`: the 40x fields worth scanning, from the 10x scan

A selector (#636). A whole well at 40x is about 1500 fields and a quarter of an hour; at 10x
it is 109 fields and under a minute. This selector reads the fast 10x scan's **frames**
and returns, per well, the 40x fields that are worth scanning: away from the
rough band at the wall, away from bubbles and clumps, where the 10x scan looked, and
where there are cells. The most cells come first.

```sh
smc run route --offline --run-doc <10x run.json> --param selector=low-mag-survey \
    --param "objective=3-Plan Apo LmbdD0.25 40x" --param wall_margin_um=800 \
    --param fields_per_well=500
smc dataset draw <10x run.json>
```

## Why the frames, and not the cells

The cell model was trained on 40x frames. On the 10x scan of 2026-11-24 it found 2 to
10 single cells a well, where the 40x tilings of the same wells hold thousands. At 10x
a cell is a **dark round blob of about 9 um** (its width at half its depth) with a
bright phase halo, on a bright, vignetted, grainy background, and contrast finds it:

1. the frame divided by a wide blur of itself (sigma 34 um, the blur taken on the frame
   extended by an odd reflection, so that the vignetted edges are not read as dark);
2. smoothed (sigma 1.7 um), and thresholded at 5 times the frame's own robust noise and
   at least 6 % darker than the background;
3. each object measured (its diameter at half its own depth), placed on the stage
   through the scan's matrix, and counted once where the frames overlap (two sightings
   from two frames within 5 um are one object).

`route`'s `labels` do not matter to this selector, and `data.cells` is not read; `route`
still requires a `cells` stage in the run document (the 10x scans have one).

## What it decides

Candidate fields of the **imaging objective** (402 x 404 um at the 40x) are laid on a
lattice from the well's centre, inside `route`'s `wall_margin_um`; with the default
`placement="greedy"`, on 16 copies of that lattice shifted by quarters of a field (they
overlap). Each candidate gets:

| Measure | What |
| --- | --- |
| `n_cells` | dark objects of 8 to 25 um in the field (the score) |
| `density_per_mm2` | `n_cells` over the field's area |
| `n_large` | dark objects of 30 um or more: clumps, bubbles, debris |
| `rough_fraction` | the share of the field's 72 um tiles whose texture is above `max_texture` |
| `coverage` | the share of the field that a read 10x frame covers |
| `wall_distance_um` | from the field's farthest corner to the wall (the plate's nominal circle) |

A field is banned when `n_large > max_large` (0), `rough_fraction > max_rough_fraction`
(0.1), `coverage < min_coverage` (0.9), `n_cells < min_cells` (1), or its density is above
`max_density_per_mm2` (no bound by default). The rest are ranked by `n_cells`, then by
`wall_distance_um`. With `placement="greedy"` they are taken in that order when they
overlap none taken before (a field can sit anywhere, centred on a group of cells that the
lattice would cut in two); with `placement="lattice"` the touching lattice is kept as it
is. The first `fields_per_well` are returned. A frame whose autofocus
was `unlocked` is not read: the fields over it lose coverage. `route`'s default orderer
(`nearest`) then orders them for the stage.

**Texture** is the spread of the contrast's structure between two blurs (1.7 and
6.7 um: the scale of dried droplets) over the spread of the tile's own pixel noise. The
plain spread of the contrast also grows where a frame is dim, and called 44 to 83 % of
the clean glass 1.0 to 1.5 mm from the wall rough.

## Parameters: which defaults were measured, which were chosen

Every number the algorithm decides with is a parameter of `LowMagSurveyParams`
(`--param 'selector_params={...}'`), none is a constant of the code. Each one says in
its description and its JSON schema (`basis`) what its default rests on:

- **measured**: set on the session of 2026-11-24 against the 40x tilings (below);
- **chosen**: a reasonable value, **not optimised on data**. These are the first to
  question on another sample, another plate or another illumination.

A test fails when a parameter says neither.

| Parameter | Default | Basis | What, and why this default |
| --- | --- | --- | --- |
| `background_sigma_um` | 34.0 | **chosen** | the blur that estimates a frame's background (30 px at 10x): well above a cell (~10 um), well below the vignetting's slope (hundreds of um). |
| `smooth_sigma_um` | 1.7 | **chosen** | the blur against the shot noise (1.5 px at 10x): small next to a cell. |
| `k_noise` | 5.0 | **chosen** | how many noise levels (the frame's robust spread) dark an object's pixel is. |
| `min_contrast` | 0.06 | **chosen** | how much darker than its background an object's darkest point is; checked only through the per-field match with the 40x counts. |
| `extent_fraction` | 0.5 | **chosen** | an object's pixels are at least this fraction of min_contrast dark, so its extent reaches past its darkest core. |
| `diameter_depth_fraction` | 0.5 | **chosen** | the diameter is that of the pixels at least this fraction of the object's own darkest contrast (0.5: the width at half depth). |
| `min_area_um2` | 15.0 | **chosen** | the smallest object, in um2 (12 px at 10x): below is noise. |
| `merge_radius_um` | 5.0 | **chosen** | two sightings of one object by two overlapping frames are this close or closer; two touching cells are ~10 um apart, and the cells stage merges at 5. |
| `cell_min_diameter_um` | 8.0 | **measured** | the smallest object counted as a cell; 6 um read 1.18 of the 40x count in C3, 8 um read 0.96 (and 0.89 in C4). |
| `cell_max_diameter_um` | 25.0 | **chosen** | the largest object counted as a cell (a doublet still counts). |
| `large_min_diameter_um` | 30.0 | **chosen** | an object this large or larger is a clump, a bubble or debris. |
| `tile_um` | 72.0 | **chosen** | the side of the squares whose texture is read (64 px at 10x; ~30 a 40x field). |
| `texture_low_um` | 1.7 | **chosen** | the finer blur of the texture's band-pass: the scale of dried droplets. |
| `texture_high_um` | 6.7 | **chosen** | the coarser blur of the texture's band-pass: the scale of dried droplets. |
| `texture_noise_sigma_px` | 1.0 | **chosen** | the blur whose residual is a tile's pixel noise, which the band-pass is divided by. |
| `halo_um` | 3.4 | **chosen** | how far an object's halo reaches past its dark pixels (3 px at 10x); left out of the texture. |
| `min_tile_kept_fraction` | 0.25 | **chosen** | a tile with less than this share of its pixels left after the objects has no texture (nan). |
| `max_texture` | 0.32 | **measured** | a tile rougher than this is rough: in each well >= 90 % of the tiles within 600 um of the wall and <= 2 % beyond 1.5 mm. None: no texture ban. |
| `max_rough_fraction` | 0.1 | **chosen** | a field whose share of rough tiles is above this is banned; banning on one rough tile (0) emptied 22 % of C4's clean fields, 0.1 bans 2 % of them. |
| `min_cells` | 1 | **chosen** | a field must hold at least this many cells: not an empty field. |
| `max_density_per_mm2` | None | **chosen** | a field denser than this is banned (overgrown). None: no bound, as no overgrown well was seen. |
| `max_large` | 0 | **chosen** | a field may hold at most this many large objects. |
| `min_coverage` | 0.9 | **chosen** | the share of a field that a frame that was read covers; tolerates a seam, not a missing frame. |
| `coverage_probes_per_side` | 8 | **chosen** | the coverage is read on this many by this many points of a field. |
| `coverage_reach_tiles` | 0.75 | **chosen** | a probe is covered by a tile whose centre is within this many tile sides (between the half side 0.5 and the half diagonal 0.71, and a little more). |
| `lattice_offset_x_um` | 0.0 | **chosen** | the lattice of fields is laid from the well's centre moved by this, in x. |
| `lattice_offset_y_um` | 0.0 | **chosen** | the lattice of fields is laid from the well's centre moved by this, in y. |
| `placement` | `greedy` | **measured** | how the fields are placed. 'lattice': the touching lattice, each cell in one field. 'greedy': candidates on a lattice finer by greedy_subdivisions, and the best field that overlaps none taken before, again and again. Greedy put 16 to 26 % more 40x cells in the first 100 to 200 fields of C3 and C4, 4 % more in C4's first 500, and needed a quarter to a third fewer fields for 90 % of the lattice's cells in B3, B4 and C3; it packs a dense well worse without a cap (C4: 486 fields where the lattice keeps 823). |
| `greedy_subdivisions` | 4 | **chosen** | greedy placement's candidates step by a field's extent divided by this (4: a quarter field, 16 shifted lattices). |

Not parameters, because they decide nothing: the robust spread's factor 1.4826 (the
median absolute deviation of a normal sample to its standard deviation), the blur's
extension of four sigmas (where a Gaussian is spent), and a background floor of one count
(so that the black outside a well is never divided by zero).

## Measured on the session of 2026-11-24

The 10x scan of B4, B3, C3 and C4 (`2026-11-24-10x-4wells`, 109 frames a well, PFS
locked) and the whole-well 40x tilings of C4 and C3 (`2026-11-24-40x-C4`, `-C3`, 1544
frames each) with the cell model's detections at its defaults (single and doublet are
counted as cells). A laptop CPU: about 80 s a well for the survey.

### The wall band

Dark objects of 8 to 25 um per mm2, by distance to the wall:

| Well | 0-200 um | 200-400 | 400-600 | 600-800 | 800-1000 | 1000-1500 | 1500-3000 | beyond |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B4 | 24.6 | 98.4 | 38.5 | 18.0 | 13.1 | 16.1 | 15.4 | 17.9 |
| B3 | 30.1 | 38.5 | 24.1 | 4.5 | 4.5 | 4.0 | 3.3 | 3.6 |
| C3 | 26.4 | 32.7 | 28.6 | 17.7 | 2.9 | 3.2 | 3.3 | 2.8 |
| C4 | 17.5 | 49.1 | 47.0 | 40.3 | 31.2 | 33.0 | 32.4 | 34.5 |

Within 600 um of the wall the counts are dried droplets, meniscus ridges and the
well's dark edge (seen on random samples of 10x crops beside the 40x crop of the same
spot); beyond 800 um they are flat. Hence `wall_margin_um=800`. The distance is to the
plate's nominal circle (radius 7830 um); `plate-walls` measured 7723 um on B2.

### The texture threshold

`max_texture` = 0.32 is the threshold at which, in each well, at least 90 % of the tiles
0-600 um from the wall are rough and at most 2 % of those beyond 1.5 mm are:

| Well | rough within 600 um of the wall | rough beyond 1.5 mm |
| --- | --- | --- |
| B4 | 94.8 % | 0.4 % |
| B3 | 91.1 % | 0.3 % |
| C3 | 90.0 % | 0.1 % |
| C4 | 94.1 % | 1.6 % |

A field holds about 30 tiles, so one rough tile is not enough to ban it: "any rough tile"
banned 22 % of C4's fields beyond 1.5 mm. A share above 0.1 bans 2 % of those, and 16 %
of the fields 0.8 to 1.5 mm from the wall.

### The 10x count against the 40x count

On every allowed 40x field of the lattice, the 10x count of objects of 8 to 25 um against
the cell model's count on the 40x tiling (single and doublet):

| Well | 10x total / 40x total | per-field correlation |
| --- | --- | --- |
| C4 | 0.89 | 0.92 |
| C3 | 0.96 | 0.84 |

With 6 to 25 um, C3 (sparse, with much debris) read 1.18 of the 40x count; with 8 to 25,
C4 reads 11 % low. The diameters beyond 1 mm from the wall: 10th percentile 7-8 um,
median 9 um, 90th percentile 11 um, in every well.

### What the plan keeps

The defaults, `wall_margin_um=800`, against the 40x tilings (single and doublet, each
cell counted once, at its 40x position minus the shift below). A field is allowed when
its whole footprint is beyond the margin: the allowed fields of the lattice hold 78 % of
C4's 6571 cells and 79 % of C3's 548 (the disc beyond 800 um holds 89 and 90 %; the
fields cannot reach its edge).

| Well | placement | fields kept | of the 40x tiling's 1544 frames | 40x cells in the kept fields |
| --- | --- | --- | --- | --- |
| C4, no cap | lattice | 823 | 53 % | 4823: 94 % of the allowed fields', 73 % of the well's |
| C4, no cap | greedy | 486 | 31 % | 3838: 75 % of the allowed fields', 58 % of the well's |
| C4, `fields_per_well=500` | lattice | 500 | 32 % | 3687: 72 % of the allowed fields', 56 % of the well's |
| C4, `fields_per_well=500` | greedy | 486 | 31 % | 3838: 75 % of the allowed fields', 58 % of the well's |
| C3 | lattice | 319 | 21 % | 392: 90 % of the allowed fields', 72 % of the well's |
| C3 | greedy | 263 | 17 % | 398: 91 % of the allowed fields', 73 % of the well's |

Some of the cells near the wall are the same droplets at 40x. C3, a sparse well, is
where the plan saves most: 83 % of the 40x frames for a loss of a tenth of the allowed
fields' cells. `route` stops at 500 fields a well, which a dense well like C4 exceeds on
the lattice.

Through `route` itself (the command above, `fields_per_well=500`) the stops were, with
`placement="lattice"`, B4 500, B3 359, C3 313 and C4 500 (1672), and with the default
`placement="greedy"`, B4 479, B3 302, C3 263 and C4 477 (1521). `route` lays the 40x
field from the scan's camera scaled by the profile's pixel sizes, a few um off the 40x
scan's own camera that the evaluation used, so a few fields at the margin differ.

### Greedy placement against the lattice

Nothing asks for a lattice: `route` checks only that the fields do not overlap, are
allowed, and are at most `fields_per_well`. The two placements on the same survey and the
same bans, 40x cells in the first fields chosen:

| Well | first 100 | first 200 | first 500 | all kept |
| --- | --- | --- | --- | --- |
| C3, lattice / greedy | 179 / **226** | 275 / **335** | 392 / 398 | 392 (319 fields) / 398 (263) |
| C4, lattice / greedy | 1051 / **1218** | 1834 / **2197** | 3687 / **3838** | **4823** (823 fields) / 3838 (486) |

Fields for 90 % of the lattice's 10x count: B3 314 / 211, B4 602 / 479, C3 278 / 204; C4
greedy never reaches it. Greedy takes the best spot first, so its first fields hold more
cells (16 to 26 % more 40x cells in the first 100 to 200, 4 % more at `route`'s cap of
500); but it packs worse, leaving gaps narrower than a field between its fields, so with
no cap a dense well ends with fewer fields and fewer cells (C4: 486 against 823). Under
`route`'s cap it is ahead in every well, hence the default; `placement="lattice"` is for
a whole-well tiling or tiles to stitch. Greedy is the greedy order, not the best packing:
filling its gaps with lattice fields is a follow-up.

## What it is not

- **It is in the 10x scan's stage frame.** The same object was at a different stage
  position in the 40x scans of that session: **40x = 10x + (+64.9, +167.3) um**, from
  301 matched objects in C4 and 89 in C3, agreeing within 0.3 um. That is the 40x scan
  minus the 10x scan: the 40x objective's offset, a plate moved between the two scans,
  or both (FM-207); the 40x scan skipped its plate check because the profile has no 40x
  offset. The evaluation above added it to find the fields in the 40x tilings; the
  selector does not, and a stop's position must be corrected by the stand before the
  40x visits it.
- **A stop's Z and autofocus offset** are the nearest 10x frame's (`route`): a 10x Z,
  not to be used at the 40x.
- **The counts are not a reference.** The 40x counts are the cell model's, not an
  annotation, and the model was not measured on these frames (#342).
- It was tuned on one session, one plate, one illumination; another sample or a
  brighter field may move the thresholds.
