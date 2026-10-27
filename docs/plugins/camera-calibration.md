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
out-of-focus field fails the same way.

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
| `step_um` | a quarter of the smaller side of the frame, from the camera's pixel size; 50 µm when the camera reports none | The probe distance. Far enough for a precise shift, and well inside half the frame (below). |
| `settle_ms` | 0 | A wait after each move, before the frame: for a stage that rings. |
| `from_run_doc` | none | Reuse the calibration in another run document. Nothing moves. |

The sequence is a reference frame, a move in +X, a frame, a move in +Y, a frame,
the return to the start, and a last frame that checks the return. Progress is
`1/3 +X probe`, `2/3 +Y probe`, `3/3 return`.

The demo devices cannot be calibrated: the demo camera ignores the stage, so both
probes see no motion and the plugin says so (the tests use the synthetic plate of
`smc.testing.synthetic`, whose frames follow the stage):

```text
$ smc run camera-calibration -p demo --run-doc run.json
camera-calibration: 1/3 +X probe
✗ camera-calibration: the two moves gave parallel shifts: did the stage move?; step 128 µm
```

## What it leaves

A stage that ran and could not produce a usable result ends the record `failed`
with the reason, and **fills no slot**: a reader never finds a matrix it should
not trust. The reason says which of two things went wrong:

- `the frames barely correlate: find a field with texture`: nothing to lock onto.
- `the two moves gave parallel shifts: did the stage move?`: the image did not
  follow the stage on one axis (a stage that ignored the command, the wrong
  camera, a blocked light path).

Either is followed by `step <n> µm`, and by `a step over 40 % of the field may
wrap: give a smaller step_um` when the step was the 50 µm fallback or more than
40 % of the smaller side. Phase correlation reports each axis's shift in
`[-N/2, N/2)`, so content moved by 200 rows of a 384-row frame reads -184: a step
beyond half the frame cannot be told from a wrong one, and the weak response
refuses it.

**The stage is returned to where it started** when the measurement ran to the
end, whether or not the fit is usable, so an empty field still gets the stage
back. After **a stop, a halt or a fault the stage is left where it is** and a
WARNING says where (`the stage is left at (x, y) µm, not returned to (x, y) µm`):
a return after the operator's stop would be another move nobody asked for.

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
| `response` | The smaller of the two probes' agreement after alignment, 0 to 1. A good field is well above the refusal threshold of 0.05. |
| `return_error_um` | How far the stage came back from where it started, measured on the last frame; `null` (unknown, not zero) when that frame did not correlate or the calibration was reused. |
| `step_um` | The probe distance used. |
| `reused_from` | The run document this was copied from, as typed; `null` for a measurement. |
| `assumed` | What it rests on that was not measured: see below. |

`assumed` can hold: `step of 50 µm: no pixel size was known` (a probe distance,
not a result); `the return could not be checked: the frame after it did not
correlate (<r>)`; `the stage came back <n> µm off: its repeatability limits the
scan's stitching` (reported, not refused: above the larger of 2 µm and four
pixels); `camera calibration reused from <path>, measured <time>`.

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
`assumed`, has `return_error_um` null, and keeps the path as typed, never resolved:
an absolute path names a user's folder.

## Not verified here

Nothing in this page was measured on a microscope. The pixel size at 40× on the
Ti2-E (0.275 µm per pixel is the datasheet's, assumed), the rotation and mirror of
its camera, and the stage's repeatability are for the first hardware session
(#198).
