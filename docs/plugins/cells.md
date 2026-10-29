# `cells`

Finds the single cells, the doublets and the debris on the frames of a scan, and
says where each is **in stage µm**, so that the density, the route and the report
work with positions on the plate and not with pixels of a frame. A cell that two
overlapping fields both saw is counted once.

It reads the `scan` slot, runs the trained detector on every frame, and for each
well: drops the second label the model gives to one object (suppress), turns each
box into a stage position through the scan's camera matrix (project), and removes
the views that two fields have of one cell (merge). Nothing moves and nothing is
read from the stand: it works on the frames in the run folder. The design is
[`m3-plugins.md`](../design/m3-plugins.md) §9; this page is how to use it.

## Before the first run

**Install the detector's dependencies.** torch is the `detect` extra, so a PC that
only drives the stage does not carry it:

```text
pip install -e .[detect]
```

(in `zsh`, quote the brackets: `pip install -e ".[detect]"`). Without it `cells` is
still listed by `smc run --list`, and a run says `torch is not installed ...:
install the detect extra`. A torch that is installed and does not load (a DLL, on
Windows) is told apart: `torch is installed but did not load (...); reinstall the
detect extra`.

**Put the weights where it looks.** The trained model, `cell_classes_model-cat-v2.pth`
(166 MB), is **not in git**. It is looked for in `local/models/` of the smc folder
(ignored by git), or in the folder `SMC_MODELS` names. A run without it says where
it looked:

```text
cells: no model at <path>; put cell_classes_model-cat-v2.pth there, set SMC_MODELS to its folder, or give model_path
```

An empty `SMC_MODELS` is no setting. In a worktree, the smc folder is the
worktree: set `SMC_MODELS` to the main folder's `local/models`.

## Running it

```text
smc run cells -p <profile> --run-doc run.json
smc run cells -p <profile> --run-doc run.json --param device=cpu
smc run cells -p <profile> --run-doc run.json --param merge_radius_um=3
smc run cells -p <profile> --run-doc run.json --resume
```

| Parameter | Default | Meaning |
|---|---|---|
| `detector` | `model` | `model` is the trained network. `blobs` counts bright round spots and is for the synthetic plate of the `demo-sample` profile only ([below](#the-blob-detector-for-the-synthetic-plate-only)). |
| `model_path` | `local/models/cell_classes_model-cat-v2.pth`, or that file in `$SMC_MODELS` | The weights. Refused with `detector=blobs`. |
| `device` | `auto` | `auto` is `cuda` when the PC has one and `cpu` otherwise: never `mps` unless asked. `cuda` or `mps` asked for and not there is refused (`device cuda is not available on this PC; use device=auto or cpu`), not replaced by the CPU. With `detector=blobs` only `auto` and `cpu` are accepted. |
| `score_threshold` | 0.5 | Detections below it are dropped (0.05 to 1). The blob detector scores every detection 1.0, so it has no effect there. |
| `max_overlap` | 0.7 | Two boxes of one frame sharing more than this fraction of the smaller one are one object; the more confident label stays. |
| `merge_radius_um` | 5 | Cells of two fields within this distance are one cell. 0 merges nothing. |

It needs a scan with frames: a dry-run scan, or one whose every field failed, has
none and is refused (`the scan has no frame ...; run scan again`). `--dry-run`
changes nothing else here: the stage only reads frames and writes its own files,
so it runs as a real run does.

Progress is one report per frame, `<well> <frame name>`, out of the scan's frames.
A stop is looked at before each frame: Ctrl-C ends the stage `stopped`, with the
checkpoint of the wells done (none, if the first well was not finished).

## The blob detector, for the synthetic plate only

```text
smc run cells -p demo-sample --run-doc run.json --param detector=blobs
```

The trained model does not see the cells of the `demo-sample` stand (on frames of 7
single and 7 paired synthetic spots it found at most 2 `single`), so the workflow
could not be followed past the scan on that stand. `blobs` is a second detector that
finds what that sample draws: **bright round spots**, as `single` (round) or `doublet`
(two side by side). It needs neither torch nor the weights, and it runs on the CPU.

**Why its counts are not cells.** It knows nothing of cells. It finds the pixels that
stand out of the frame's background (its 75th percentile plus 8 times the noise of
neighbouring pixels), keeps the connected spots between 9 pixels and 5 % of the frame
whose shape is not a strip, and calls a spot a `doublet` when it is longer than wide
(its minor axis is under 0.75 of its major one). Run on a real sample it would count
bright debris, a well's rim and bubbles the same way, so the result says so in
`assumed`: `cells were found by the blob detector (detector=blobs), which counts
bright round spots for the synthetic plate of the demo-sample profile; on a real
sample its counts are not cells`. It never replaces the model on a stand.

**Its limits on the synthetic plate.** It is tuned for the sample at the camera's
default exposure, 10 ms. The sample's texture grows with the exposure and its noise
does not, and the detector measures only the noise: from 50 ms up it counts speckle
in empty wells (tens to hundreds per frame, measured), and nothing on the page or in
`scan` sets an exposure, so only a deliberate `set_exposure_ms` or `smc snap
--exposure-ms` gets there. And it can count a wedge of well: a frame whose corner
just enters a well (by up to about 50 µm at 0.5 µm per pixel; a scan of the whole well,
`coverage` 1, keeps such frames) shows that wedge, which is under 5 % of the frame, as a
`doublet` boxed at the corner. Scans of the middle of a well, as the `coverage=0.4` of
the commands above, do not have such frames.

In the result, `model` is `blobs`, `model_sha256` is the SHA-256 of a string that
names the detector's version and rules (so a resume after a change of those rules
is refused, like a change of weights), `classes` are `single` and `doublet` (no
`debris`) and `device` is `cpu`. The page's Cells tool shows `detector` as a key
parameter, next to `score_threshold`. Measured on the synthetic plate, it finds exactly
the 5 single and 3 paired spots of a frame at 0.1625, 0.25, 0.5, 1 and 2 µm per
pixel, and nothing on a frame that crosses a well wall.

## Stop and resume

The unit of work is a well. After each well the cells found so far are written to
`cells/<token>.partial.json`, and then the checkpoint names the well. `--resume`
keeps the token, keeps the cells of the wells done and detects only the others.
Three refusals and one repair:

- **Other weights**: the SHA-256 of the weights is in the checkpoint, and a
  resume with a file that hashes differently is refused (`the weights changed
  since the run being resumed; run cells without --resume`): cells of two models
  are not one result. With `detector=blobs` the same check hashes the detector's
  identity string, so a resume after a release that changed its rules is refused too.
- **Other parameters or another profile**: the runner refuses it, as for every
  stage. Another `detector` is a parameter that differs.
- **A scan produced again**: the runner refuses a resume over inputs that changed.
- **A partial file that is missing or does not validate** is a WARNING, and every
  well is detected again under the same token, after the checkpoint has been
  reset to "no well done": a checkpoint that still named wells the new partial
  file does not hold would make a later resume skip them. Slower, never wrong.
  What a valid partial file cannot show is a well with no cell in it, so a copy
  of the run folder made while a run was going, with an older partial file than
  its checkpoint, is taken at its word: resume in the folder the run was in.

At the end `cells/<token>.json` is written, then the partial file is removed. A
partial file that cannot be removed (Windows, a file another program holds) is a
WARNING that names it; the stage is `done` and the result is whole.

## What it leaves

The slot `cells` holds a `CellsResult` (`smc.plugins.slots`):

| Field | Meaning |
|---|---|
| `file` | `cells/<token>.json`, relative to the run folder, with forward slashes. |
| `model`, `model_sha256` | The weights' **file name** (never its path: run documents are attached to issues), or `blobs`, and its SHA-256 (of the blob detector's identity string for `blobs`). The parameter `model_path` is recorded as you typed it, in the stage's `params`, like every parameter: when the path names your account, give the folder through `SMC_MODELS`, which is recorded nowhere. |
| `classes`, `device` | The detector's labels (`single`, `doublet`, `debris`) and the device it ran on, never `auto`. |
| `score_threshold`, `merge_radius_um` | The parameters. |
| `frames` | The number of frames of the scan. |
| `counts` | Well to label to count, every well of the scan and every label, **zeros included**. A well whose every field failed is absent, as in the scan. |
| `merged` | How many views were removed as the second sight of a cell, over every well, a resumed run's included. |
| `seconds_per_frame` | The mean time of `detect` in this attempt; `null` when it detected no frame. |
| `assumed` | `cells within 5 µm seen by two fields are counted once: a chosen radius, not measured`; empty when `merge_radius_um` is 0. With `detector=blobs`, first the line that says its counts are of bright round spots and not cells. |

The cells are in the file, not in the run document: a 96-well plate at 40× is
about 300 frames per well and thousands of cells per well. `load_cells` reads the
file back and refuses one that no longer matches the counts. A cell is
`well`, `x_um`, `y_um` (the centre of its box, in stage µm through the scan's
camera matrix), `label`, `score`, `frame` (its position in the scan's frames) and
`box_px`, `(row0, col0, row1, col1)` in that frame's pixels, for the report's
overlays.

## How a cell is counted once

Two overlapping fields can both show a cell. Cells of the **same well** and of
**different** frames, at most `merge_radius_um` apart, are views of one object.
The cells are taken from the least cut view (the box centre farthest from its
frame's edge, then the higher score, then the lower frame). Each one that is still
there is kept and removes **at most one view from each other frame**: the nearest.
Two cells of one frame never remove each other.

It is not "everything within the radius is one cell". That chains: one box of the
second field between two cells of the first would leave one cell for two, and two
neighbours seen by both fields with a 0.8 µm calibration error between them would
lose one of the pair. A neighbour that only one field found stays.

## The model's limit of boxes per frame

torchvision's Faster R-CNN keeps the best 100 boxes of a frame and says nothing
about the rest (`box_detections_per_img`, 100 by default; `nikon-control` used it
so too). A frame with more cells than that is **undercounted**,
silently as far as the result goes: a WARNING (`the model returned 100 boxes for a
frame, its limit per frame: a denser frame is undercounted`) names it, once per
such frame. The warning cannot tell a frame of exactly 100 cells from a larger one.
The limit is not in the result and was not raised: that changes what the model
returns, and is for the owner to decide.

## What a bad frame looks like

Each is a `failed` record whose reason names the frame's file and ends with `run
scan again`: the file is missing, cannot be read (a corrupt TIFF), is not 2-D, is
not of the scan's frame shape, or is a float frame with a value that is not
finite. The shape is checked because the scan's matrix is only right on it. A
label that the detector does not declare is refused too: the counts would not add
up.

## Not verified here

Nothing in this page was run on a microscope, and CI has neither torch nor the
weights: the tests of the plugin use a stub detector, and the tests of the model
are skipped without them.

- The model on **Prime 95B frames**: it was trained on other frames (1024², of a
  pixel size the checkpoint does not record) and resizes everything to 800
  pixels. Detections are checked by eye on the first real scan (#198).
- Its **speed on the microscope PC's CPU**: 0.34 s per frame on an Apple M5 Pro,
  which would make a well of 305 frames about two minutes.
- **`merge_radius_um = 5`** against real cells: it is half the size of a yeast
  cell, well above the calibration's error, well below two cells side by side, and
  chosen, not measured. Look at the merged pairs on the first scan.
- **`cuda`**: no GPU was available to the tests, and a card that cannot run the
  model (`no kernel image`) is only simulated: the message says to use
  `device=cpu`.
- **torch next to the stand's DLLs.** `smc run` opens the stand around the stage,
  although this stage needs none, so torch may be imported in a process that already
  holds the Micro-Manager and vendor DLLs, and Windows resolves a module that is
  already loaded by its base name (FM-95). Not tried: if `python -c "import torch"`
  works in a new console and `smc run cells` says `torch is installed but did not
  load`, that is the likely cause: file it (#198).
- The box convention: the detector's box is taken as given and its midpoint is the
  centre; whether the training put a pixel's centre at its index or half a pixel
  further is not recorded (at most 0.14 µm at 40×).
