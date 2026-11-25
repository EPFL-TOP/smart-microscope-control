# `camera-calibration`

How the camera is mounted, measured by the stage: the pixel size, the rotation
between the camera and the stage axes, and whether the image is mirrored. Every
later stage that turns a pixel into a stage position (the scan plan's grid, a
detected cell's position, the route) reads this and not an assumption.

It moves the stage a known distance in X and in Y, takes a frame after each
move, and measures how far the image moved by phase correlation. The stage is
the ruler, so the answer cannot disagree with the hardware. The design is
[`m3-plugins.md`](../design/m3-plugins.md) §4; this page is how to use it.

## When to run it

**Once per objective and camera setting** (binning, region of interest): the
calibration is of one objective and one frame shape, and the stages that read it
refuse another. Run it again after the camera has been turned or replaced.

**On a field with texture**: cells, the edge of a well, dirt on the plastic.
Correlation on a blank field returns a confident answer that means nothing, so
the plugin measures how well the frames agree *after* aligning them and refuses a
weak one (`the frames barely correlate: find a field with texture`). An
out-of-focus field fails the same way. A field that holds only a few specks of
sample (a plate with sparse cells, two beads in water) is read by a **second look**
before it is refused (below): it needs the specks, not a crowd of them, and a field
that shows nothing is still refused.

**With the light on and the light path on the camera**: a frame with no light on
it is the camera's black level and its fixed pattern, and that pattern correlates
well with itself, so the stage would be blamed for an image that never changed.
The plugin looks at the reference frame first, **before anything moves**, and
refuses one that is dark or flat (below).

It moves the XY stage by a quarter of the smaller side of the frame and back
(about 100 µm on the Ti2-E at 40×, from the datasheet's pixel size, which is
assumed). Check that the plate or the dish can move that far, and that the
objective has clearance, as for any stage move; the soft limits refuse a target
outside them.

## Running it

```text
smc run camera-calibration -p <profile> --run-doc run.json
smc run camera-calibration -p <profile> --run-doc run.json --param step_um=80
smc run camera-calibration -p <profile> --run-doc run.json --param from_run_doc=earlier.json
```

| Parameter | Default | Meaning |
|---|---|---|
| `step_um` | a quarter of the smaller side of the frame, from the camera's pixel size; 50 µm when the camera reports none | The probe distance, at most 2000 µm (a larger default is capped to it). Far enough for a precise shift, and well inside half the frame (below). On a **periodic** target, a grid slide or a stage micrometer, give a step under half the period (below). |
| `settle_ms` | 0 | A wait after each move, before the frame: for a stage that rings. |
| `from_run_doc` | none | Reuse the calibration in another run document, never the one `--run-doc` names. Nothing moves. |

The sequence is a reference frame, a move in +X, a frame, a move in +Y, a frame,
the return to the start, and a last frame that checks the return. Progress is
`1/3 +X probe`, `2/3 +Y probe`, `3/3 return`. The reference frame is checked
before anything moves: it must be one 2-D grey frame of the size the camera
reports, with no `nan` in it, and with light on it. A colour mode (`(height,
width, 3)`) or another region than the camera reports is refused there, with
`nothing was moved`, and so is a dark or flat frame (below).

The demo devices cannot be calibrated: the demo camera ignores the stage, so its
probes show no motion of the stage's making and the plugin says so (the tests use
the synthetic plate of `smc.testing.synthetic`, whose frames follow the stage):

```text
$ smc run camera-calibration -p demo --run-doc run.json
camera-calibration: 1/3 +X probe
✗ camera-calibration: the two moves gave parallel shifts: the frames probably hold too little texture (a sparse field, a dark or flat frame): try a field with more structure, then the light path, then the stage; the probe frames span 5138 and 5138 counts (1st to 99th percentile); step 128 µm; if this camera was calibrated at this objective before, reuse that calibration with from_run_doc=PATH_TO_ITS_RUN_DOCUMENT (nothing moves)
```

The demo's frames carry plenty of contrast (5138 counts), which tells a stand's
reader that this refusal is not about the light: the image simply did not follow
the stage. The demo camera's test pattern moves by itself, 32 px a frame (measured
2026-11-25), so the two probes read shifts that are the same and **not** the
camera's own pattern at zero shift: the second look (below) is not asked, and the
refusal has no sentence about it.

## What it leaves

A stage that ran and could not produce a usable result ends the record `failed`
with the reason, and **fills no slot**: a reader never finds a matrix it should
not trust. The reason says which of six things went wrong (seven texts: the first
is told two ways). The first ones are about the light and come before any
judgement of the shifts:

- `the camera's frame is dark or flat (the 1st to 99th percentile spans 11
  counts, under 50): is the light on and the light path on the camera? nothing
  was moved`: the reference frame has no light on it (the light path on the
  eyepiece, the lamp or its shutter off). It is judged first, so the stage did
  not move. The number is how many counts the middle 98 % of the frame spans; the
  Prime 95B's dark frame spans 11 and the dimmest lit frame measured at the stand
  177 (the floor, 50 at 16 bits, is in between; it is smaller for a camera in a
  shallower mode, 3.125 at 12 bits, which reads fewer counts for the same light).
  The 1st to 99th percentile and not the range, because one hot pixel makes a
  dark frame's range as wide as a lit one's. **A sparse field passes**: a few
  beads or cells on a dark background fill under 1 % of the frame, so the
  percentiles read it as dark, and a frame also carries light when at least 0.01 %
  of its pixels (20 of a 384 x 512 frame, 199 of 1412 x 1412) lie the floor or
  more from its median; a handful of hot pixels is far below that. The measure
  (`signal_spread`, `signal_floor_counts`, `carries_signal` and their two constants)
  lives in `smc.plugins.frame_signal`, which holds only numpy: the plate page asks the
  same question of each frame of Snap and Live (m5 §17.12 D22) and says `The frame is
  dark` with the same floor. `camera_calibration/algorithm.py` imports the names back,
  so the plugin and its tests are unchanged. `frame_signal.saturated` (at least 1 % of
  the pixels at the camera's full scale, an **assumed** fraction) is the page's; the
  calibration's own saturation refusal below is its own.
- `the camera's frame is saturated (...): lower the exposure or the light; nothing
  was moved`: the same refusal for a frame at full scale (65535 at 16 bits): the
  remedy is the opposite one.
- `the frame after the +X probe is dark or flat (...): did the light go off, or
  did the step of 96 µm leave the lit sample? the stage was sent back to its
  start`: the reference had light and a probe frame did not. The stage did move
  and the return was sent (it runs before the frames are judged; where it landed
  is not read), so the cause is the light, or a probe that left the lit sample
  (the black plastic between wells): give a smaller `step_um`. The first such probe
  is named.

The other three are what the shifts of frames with light say:

- `the frames barely correlate: find a field with texture`: nothing to lock onto.
  A field with light and no texture (a blank well) is this one, not a dark frame.
- `the two moves gave parallel shifts: the frames probably hold too little texture
  (a sparse field, a dark or flat frame): try a field with more structure, then
  the light path, then the stage`, then `the probe frames span <a> and <b> counts
  (1st to 99th percentile)`: the image did not follow the stage on one axis. The
  frames come first because they are the likelier cause: at the 40x in water, on a
  field of two 10 µm beads, the stage moved and the dust and shading fixed to the
  camera won the correlation, so both probes read a shift of zero (Ti2-E session
  5, F12, #545); the same field with more beads calibrated, and so does that field
  now, by the second look below. The next suspects are
  the light path and then the stage (a stage that ignored the command, the wrong
  camera; when the stage's own readback says it did not move, the refusal says so
  first, below). When a probe read zero, the second look ran and the reason says what it
  saw: `away from the camera's own pattern the best peaks are <a> and <b> times the
  next (1.5 asked)`, after the two figures and before the step. The two figures are the contrast of the two probe frames, as the
  dark-frame refusal above quotes it: a dark frame spans a few tens of counts, and a
  frame whose contrast is a still background can span as much as one with a lit
  sample, so the figure rules the light out and not the texture. It follows frames
  that carry light: a dark one is named above.
- `the shifts do not fit a rigid camera: did each axis move its full command?`:
  the frames agree, but the matrix they give is not a camera, because one axis
  executed a fraction of its command (about a quarter or less of it: a quarter
  when the camera is aligned with the stage, between a fifth and a third as it
  is turned) or none plus a
  little noise (a limit that clamped it, a stage that stopped short). Check the
  stage's travel on that axis, then run it again.

**The stage's own readback comes first when it says a move did not arrive** (#614).
`move_to_um` returns where the stage read it was once the move was done, and the plugin
keeps it for each probe move. A stage that accepts a command and goes nowhere gives two
shifts of zero, which the fit can only call a texture problem. When the fit refuses and
a probe's readback is within **1 µm** of the start (over both axes; 5 % of the step for a
step under 20 µm), the reason leads with that and the texture advice goes:

```text
the stage read its start after the +X probe (0.0 µm from it) and after the +Y probe (0.0 µm from it): the moves did not arrive, so the frames are not the cause; check the stage (its controller on, nothing else holding it); step 96 µm
```

With one probe it names that probe (`the stage read its start after the +Y probe (0.0 µm
from it): that move did not arrive, ...`): an axis that goes nowhere makes one column of the
shifts zero, so the shifts are parallel whatever the frames hold. The figure is the distance
the stage read from the start, to a tenth of a µm, so that a surprise at the stand is a
number to bring back. What goes with the texture advice: `PARALLEL_SHIFTS_NOTE` (it ranks
the frames first), the probe frames' spread, the second look's ratios, the wrap hint and
the reuse sentence (a reuse is no way out of a stage that did not obey). The fit's other
notes stay after the line (a weak response is a fact of the field), then `step <n> µm`. The
line says what was read, and its checks are suggestions, not a diagnosis.

Readbacks that left the start change **nothing**: the refusal is the texture one above,
byte for byte (the reason has no room to quote them: 572 of the card's 600 characters
at the longest). A fit that is trustworthy is never refused or relabelled for the
readbacks: a stage that moves and reads back its start (a stale readback) calibrates
from the images, which followed it. The tolerance is **assumed**: the demo stage lands
within 0.005 µm of a command and the fake exactly (measured 2026-11-25), and a stage that
goes nowhere reads its start with the same read; a real stage's idle readback was not
measured.

Each of the three texts above is followed by `step <n> µm`, and by `a step over 40 % of the field may
wrap: give a smaller step_um` when the step was the 50 µm fallback or more than
40 % of the smaller side. Phase correlation reports each axis's shift in
`[-N/2, N/2)`, so content moved by 200 rows of a 384-row frame reads -184: a step
beyond half the frame cannot be told from a wrong one, and the weak response
refuses it.

**A refusal for texture offers a way out that needs none.** `parallel shifts` and
`barely correlate` both end, after the step and the wrap hint, with `; if this
camera was calibrated at this objective before, reuse that calibration with
from_run_doc=PATH_TO_ITS_RUN_DOCUMENT (nothing moves)`. The tool reads no other
document, so it cannot name the file: `PATH_TO_ITS_RUN_DOCUMENT` stands for the run document of
an earlier calibration at this objective (the plate page keeps one per turret
position, below). A refusal for another reason (a dark frame, the rigid-camera
note, the stage's own readback) does not carry it: a reuse is no way out of a stage
that did not obey. The
whole reason stays within the 600 characters the plate page's card keeps of a
failure (572 at the longest: five-digit spreads, a 2000 µm step and the wrap hint).

### A sparse field: the second look

The camera's own pattern (a vignette, the sensor's column stripes, dust) does not
move with the stage. On a field that holds almost nothing of the sample it is the
highest peak of the phase correlation, at zero shift, and a stage that moved a
quarter field reads as one that did not. At the Ti2-E (session 6, F14, #625) the
10x in phase contrast on a plate with cells was refused on every field the owner
tried, the probe frames spanning 6674 and 6574 counts: the fields held a smooth
vignette, the stripes, dust and a few dark specks, and no cell (the contrast the
refusal quoted is the vignette's, not the sample's). The morning's plate of beads
had calibrated at once, and so would F12's field with more beads.

When the fit says `parallel shifts` **and** at least one probe read a shift within
2 px of zero, the plugin correlates the three frames it already holds **again**,
with the cells within 12 px of zero shift left out (the zero peak's skirt reaches 4
to 5 px), takes the strongest peak that remains for each probe, and says how many
times it stands above the highest peak more than 10 px from it. It believes the two
shifts when **both stand out 1.5 times the next** and **they are a camera's** (as
long as each other to 10 %, at a right angle to 10°, mirrored or not). The fit then
judges them as it judges any shifts (the rigid-camera check, the response, the
step), and the result is a calibration like any other, with one line of `assumed`
(below) that says it was read this way. Otherwise the refusal stands and says what
the second look saw. `measure_shift` is not changed: it is the answer for every
field that calibrates without a second look, and the return check's.

It moves nothing and takes no frame: it is arithmetic on the three frames in
memory (12 MB at 1412 x 1412 and 16 bits), after the return.

What it was measured on is **emulated**: the refused probes were never saved, so
each was rebuilt from the real 10x scan of that afternoon (the fixed pattern is the
median of 24 frames of the well; the sample's content is a frame minus it, moved by
the morning's matrix). On 148 fields of the four wells `measure_shift` read zero for
both probes on every one; the strongest peak away from zero was the true shift on
124 of them, and at 1.5 times the next the second look accepts 123, none at another
peak. On the morning's beads plate (74 fields) the figures are 67 and 60. A stage
that did not move gave at most 1.20 times the next peak (20 pairs), an empty field
1.19 (28 pairs). The numbers and the method are in
[`m3-plugins.md`](../design/m3-plugins.md) §4.4.1. **No calibration of a sparse
field was run at a stand yet**, so the two thresholds (1.5 and 12 px) are those of
the emulation and nothing more.

**The stage is returned to where it started** when the measurement ran to the
end, whether or not the fit is usable, so an empty field still gets the stage
back. After **a stop, a halt or a fault the stage is left where it is** and a
WARNING says where (`the stage is left at (x, y) µm, not returned to (x, y) µm`,
or `at an unknown position` when the position cannot be read either): a return
after the operator's stop would be another move nobody asked for. A stop before
the first move, or after the return has arrived, leaves the stage at its start,
and says nothing.

A dry run cannot measure (`a dry run cannot measure the camera; run it without
--dry-run`), but it may reuse.

## The result

The slot `camera-calibration` holds a `CameraCalibration`
(`smc.plugins.slots`):

| Field | Meaning |
|---|---|
| `objective`, `image_shape_px` | What it was measured with: the turret's current objective (`null` without a turret) and the frame as `(height, width)`. |
| `matrix_um_per_px` | `M`: a pixel offset `(d_col, d_row)` from the frame centre to a stage offset `(dx, dy)` in µm. Rows grow downwards and stage Y upwards, so an upright camera without mirror is `ps · R(θ) · diag(1, -1)` ([`geometry`](../design/m3-plugins.md), §3.1). |
| `pixel_size_um` | `sqrt(abs(det M))`. |
| `rotation_deg` | The camera's rotation, in `(-180, 180]`. |
| `mirrored` | Whether the image is flipped (`det M > 0`). An upright camera is **not** mirrored. |
| `residual_um` | How far the frame's edge lies from where a rigid camera of that size and rotation would put it: near 0 for a good camera, large for a stretched or sheared image or a bad measurement. |
| `response` | The smaller of the two probes' agreement after alignment, 0 to 1. A good field is well above the refusal threshold of 0.05. When the second look read the probes (`assumed` says so), it is the smaller of the two second-look responses, which counts the camera's own pattern too: it is not evidence of the sample's texture. |
| `return_error_um` | How far the stage came back from where it started, measured on the last frame; `null` (unknown, not zero) when that frame did not correlate, when the second look read the probes (the check is blind there, below) or when the calibration was reused. |
| `step_um` | The probe distance used. |
| `reused_from` | The run document this was copied from, as typed but written with forward slashes (`as_posix()`), and never resolved; `null` for a measurement. |
| `assumed` | What it rests on that was not measured: see below. |

`assumed` can hold: `step of 50 µm: no pixel size was known` and `step capped at
2000 µm: ...` (probe distances, not results); `sparse field: the camera's own
pattern won at zero shift, so both shifts are the strongest peaks away from it (4.7
and 4.9 times the next, 1.5 asked)` (the second look read the probes, after the step
lines and before the next one; at most 160 characters, the length the page's strip
keeps of the first line); `step of 50 µm fits poorly (18 px,
under a tenth of the 1412 px frame; residual 10.7 µm, 21 % of the step): run again
with step_um=970` (the next paragraph); `the return could not be checked:
the frame after it did not correlate (<r>)`, and `the return could not be checked: on
this field the camera's own pattern wins the correlation at zero shift` (after a second
look: the check cannot see a drift there); `the stage came back <n> µm off: its
repeatability limits the scan's stitching` (reported, not refused: above the
larger of 2 µm and four pixels); `camera calibration reused from <path>, measured
<time>`.

**A probe that fits poorly is named, not refused** (F6, Ti2-E session 3). At the
4x, with no pixel size in the profile, the 50 µm fallback is 18 px of a 1412 px
frame, and the fit gave 2.745 µm per pixel, 177.8° and a residual of 10.7 µm
(21 % of the step); with `step_um=900` the same stand gave 2.739 µm, 177.4° and
0.9 µm. The line is added when the step, at the **measured** pixel size, is under
a tenth of the frame's smaller side (141 px on the Prime 95B), or when the
residual is over 5 % of the step (the three good fits of that session: 0.1 to
0.5 %). The step to run again with is a quarter of the smaller side at the
measured pixel size, to two figures and at most 2000 µm, and is left out when it
is not larger than the step used (a residual that is large at a quarter of the
field is not cured by the same probe). A stand whose profile has the pixel sizes
(#401) never takes the fallback; the line also names a `step_um` you gave that is
a few pixels.

### A `done` calibration is not a proof

The plugin refuses a field with nothing to lock onto, a stage that did not move
at all, and a matrix that is not a camera because an axis did not execute its
command (about a quarter or less of it). It does **not** refuse a calibration that
is wrong in a way the correlation cannot see, and a `done` slot is read by the scan
plan as measured. Read the numbers before you trust them:

- **A stage that moves only part of the command on one axis, but more than about
  a quarter of it** (an axis at half its command, say) is not refused: the two
  axes then differ by a factor under 4 in pixel size (for a camera aligned with
  the stage; the limit moves between a fifth and a third as it is turned, measured
  on the fit). The fit is still not a
  camera, and `residual_um` is far above a good fit's (a fraction of a µm),
  though `response` is high: a large `residual_um` means the calibration is
  wrong.
- **A stage that moves a fraction of the command on both axes**, and a **periodic
  target** (a grid slide, a stage micrometer), give a clean rigid fit of the wrong
  size, with a residual of about 0: the correlation reports the shift modulo the
  period (a 96 µm step on a 80 µm grid reads as 16 µm). Compare `pixel_size_um`
  with what you expect (the objective's and the camera's nominal size) and with
  the pixel-size WARNING below; on a periodic target give a `step_um` under half
  the period.
- **A step of a few pixels** gives a result of poor precision, not an error: it
  is named in `assumed`, with the step to run again with (above).
- **A calibration the second look read rests on a few specks** (a line of
  `assumed` says `sparse field: ...`). Its `response` counts the camera's own
  pattern, so a response that looks fine says nothing about the sample; its
  pixel size, rotation and mirror are what to read, against what you expect (the
  Ti2-E's 10x measured 1.12432 µm per pixel and 178.33° on a plate of beads). And
  the return check would read the camera's own pattern as well: on such a field the
  frame after the return reads a shift of zero whether or not the stage came back
  (a return 70 µm off read 0.0 in the review of #625), so the result leaves
  `return_error_um` **unknown** (`null`) and says so in `assumed` (`the return could
  not be checked: on this field the camera's own pattern wins the correlation at zero
  shift`). Check where the stage is after the run. The result is as good as the specks are.

These checks are not made for you yet: whether a uniform scale error or a
periodic target should be refused, and against what, waits for the data of the
first calibration at the stand (#198, and #212).

## The pixel size is not written anywhere

When the pixel size the camera reports (the `pixel_size_um` entry of the
profile's `[camera]` section for this objective, scaled by the binning and the
magnifier factor) is finite and positive and differs from the measured one by
more than 2 %, a WARNING says so. For example:

```text
the camera reports 0.5 µm per pixel for Nikon 40X Plan Fluor ELWD, measured 0.55: update the profile's pixel size
```

The profile's entry is for binning 1 and a magnifier factor of 1.0, and the layer
scales it by both, so the entry to change is the one for that objective,
**multiplied by measured ÷ reported**, not replaced by the measured number when
the camera is binned or the magnifier is not at 1.0. When the camera reports no
pixel size (no entry for the objective: a missing one is unknown, never guessed)
there is nothing to contradict, and the measured size is what to enter.

The plugin does not change the profile or the layer: the `Camera` capability has
no setter, and a stage that rewrote the stand's configuration as a side effect
would be hidden state. The scan plan takes the matrix from this slot when its
objective and frame shape match, which is the pixel size per objective the issue
asked for; the owner decides whether the profile's value follows the measurement.

## Reusing a calibration

```text
smc run camera-calibration -p <profile> --run-doc run.json --param from_run_doc=earlier.json
```

Nothing moves, and a dry run may do it. The other document's calibration is copied
when it ended `done`, was run on this profile (and was not a dry run read by a
real one), and is of **the objective in place and the frame shape of the camera**.
Otherwise the run fails and names both values, for example `the calibration in
earlier.json is of Nikon 10X S Fluor, the objective in place is Nikon 40X Plan
Fluor ELWD; put Nikon 10X S Fluor in place or run camera-calibration with Nikon
40X Plan Fluor ELWD`. The copy says where it came from in `reused_from` and in
`assumed`, has `return_error_um` null, and keeps the path as typed, written with
forward slashes and never resolved: an absolute path names a user's folder, and a
Windows path is stored as `C:/Users/...` (the run document of the same run records
the parameter in its own spelling). A path with a space is the first
`--param` value that can need quoting on Windows: put the whole item in quotes,
`--param "from_run_doc=C:\runs\Lab Book\earlier.json"`, as the
[quoting note](README.md#running-a-plugin-smc-run) says for JSON values. Not tried
on a Windows shell: check it at the stand.

**The other document is not the one the run writes.** `--run-doc run.json --param
from_run_doc=run.json` is refused before anything is written, whichever way the one
file is spelt (relative or absolute, `./run.json`, a link):

```text
✗ parameters of camera-calibration: from_run_doc names the run document this run writes (run.json), where this run replaces the camera-calibration it would copy: name a copy of it or another run document (nothing was written). smc run camera-calibration --show-params lists the keys
```

The run marks the stage running in its own document before it reads its source,
which would be that same stage: with no refusal first, the document's calibration
ended `failed` and the morning's measurement was lost (Ti2-E session 6, F15, #626).
Copy the document (`copy run.json run-10x.json`) and name the copy, or name the
document of the calibration you mean. The check is the runner's (`check_run_doc`, [m3
§1.8](../design/m3-plugins.md)), so the plate page's expert view says the same line as
its outcome and writes nothing.

**A calibration is for the state of the stand it was measured in.** The objective
and the frame shape are compared; the Ti2-E's intermediate-magnification lever
(1.0× or 1.5×), a camera binned differently inside the same frame size, and any
other state the layer cannot read are not. The scan plan, which reads the slot,
places its tiles with the copied matrix. Reuse a calibration only for the same
objective with the lever and the camera setting as they were, and measure again
after changing either (the profile's pixel sizes are for a fixed value of both:
FM-103).

## The page's own documents, one per turret position

The plate page's offsets block (*Measure the offsets between objectives*, [the plate
page](../ui/plate-page.md#measure-the-offsets-between-objectives), #501) makes one calibration
document per objective, **each in a run document of its own** and never in `run.json`: a run
document holds one `camera-calibration`, the latest attempt, and the plate registered with the
10X was registered with the 10X's. Written into `run.json`, a calibration at the 40X replaces
it and the plate goes out of date (FM-216). The Stand screen's *Calibrate the camera* writes into
the same documents instead of `run.json` when the plate was registered from its walls under
**another** objective than the one in place, and into `run.json` otherwise (the first calibration
of a stand, a plate registered by eye, the plate's own objective: the run document's calibration is
the one `plate-walls` reads); the expert view's *Run camera-calibration* still writes `run.json`,
and its `from_run_doc` reuses a position's document into it (the way to register the plate afresh
at another objective). The page's button runs this
tool as `smc run camera-calibration --run-doc calibrations/position-<n>.json` would, with the
objective in place and the form's `step_um` and `settle_ms`, in a `calibrations` folder it makes
beside the run document. `<n>` is the objective's turret position, its label's index plus one,
and the file's name is that number and never the label (a label is free text, and the document
holds the objective). A new calibration at the same position replaces its document, and a
failed or stopped one leaves it with no usable calibration until one succeeds: the page says
so before the click. That includes a calibration the tool **refuses for a dark frame**: the page
waits while the light reads off (`LightSource.is_on()`, read again at the click; an unknown light
refuses nothing, and neither does a light that reads off while the core's auto-shutter is on, as
on the Ti2-E, whose lamp opens for the snap), but a frame that is dark with the light on is still
the tool's refusal and still replaces the document. A document names an objective and its file a position: the page checks the
record it wrote against the objective of the button (a turn at the handset during the run says
`it measured <other>, not <objective> as the button said ...`), and lists a document that is not
its position's with a line of its own and leaves it out of the empty field's default.
`objective-offsets` reads these documents with `calibration_from`, one per
objective, as it reads any typed path; a terminal makes the same documents with any name.

## What reads it

The scan plan, the cells stage and the route read the matrix as the sections above
say. **The plate page reads its rotation and mirror for the image's sides and the
arrows' direction** (#327): a wall is named on the image (`left`, `right`, `top`,
`bottom`) and the layer takes stage directions (`-X`, `+X`, `-Y`, `+Y`), so the page
maps one to the other with the latest usable calibration, whatever its objective,
since the turn belongs to the camera's port and mirrors and not to the objective. It
also says under the Drive pad what each arrow does to the image. Without a usable
calibration the page refuses to capture a wall and says to run this plugin; a camera
more than 5° from a quarter turn names no side. The table that reads a camera's
sides is in [`plate-registration.md`](plate-registration.md#wall-touches).

**Run it again after a change of camera port or light path**: a calibration is of the
path it was measured in, and the page does not know that the path changed.

## Not verified here

Nothing in this page was measured on a microscope. The pixel size at 40× on the
Ti2-E (0.275 µm per pixel is the datasheet's, assumed), the rotation and mirror of
its camera, and the stage's repeatability are for the first hardware session
(#198).

The two checks of #408 were built from frames and fits the Ti2-E gave in sessions
2 and 3 (the Prime 95B only), and run on the synthetic plate: the dark-or-flat
floor of 50 counts, the tenth of the frame and the 5 % of the step. **Not run at a
stand**: the refusal with the light path on the eyepiece, the line at the 4x with
no pixel size in the profile (it should suggest about 970 µm), and the floor on
the camera's other modes (binning, gain, a 12-bit mode: the floor follows the
bit depth the camera reports, and no 12-bit frame was measured). A dark frame
whose spread is above the floor, or a lit one under it, is a number to bring back.

The second look (#625) was built from probes **emulated** out of the stand's own 10x
scan, and built on a seeded synthetic sparse field. **Not run at a stand**: a
calibration at the 10x in phase contrast (Ph1) on the plate with cells, or on any
sparse field, which should read the morning's 1.12432 µm per pixel and 178.33°; and
one on a field the tool must still refuse (an empty well). The two thresholds, 1.5
times the next peak and 12 px around zero shift, are the emulation's. A ratio or a
shift that surprises is a number to bring back.

The stage's own line (#614) was built on a stage double that goes nowhere and on the demo
stage. **Not run at a stand**: a calibration with the stage disabled or held by the vendor
software, which should read the start after each probe (the distance read then is the
number to bring back, against the 1 µm tolerance, which is assumed), and a calibration on
a field the stage can move but the frames cannot show (it should keep the texture text).
