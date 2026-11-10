# `plate-walls`

Where the plate is on the stage, from **images of the walls of two or three wells
you centred by eye**. You drive to a well with the joystick and read the position;
the tool does the rest: around each position it images the well's four walls, fits
a circle to the edges (the well's centre and size), then fits the plate to the
centres. The result fills the run document's `plate` slot with `method="wall-images"`,
the slot that [`plate-registration`](plate-registration.md) and
[`plate-finding`](plate-finding.md) fill too, so `scan-plan` and everything after it
read it as before. The design is [`m3-plugins.md`](../design/m3-plugins.md) §17; this
page is how to use it.

|  | `plate-registration` | `plate-walls` |
|---|---|---|
| You give | wells centred under the objective, or wall touches you made | wells centred **roughly**, by eye |
| It moves the stage | no (a dry run is the same computation) | **yes**, XY only, around each recorded position |
| It images | nothing | each well's four walls, a few frames a side |
| Needs | nothing | the camera's calibration at the objective in place |

It touches no Z, turret, autofocus or light. The PFS stays as it is; the tool reads
no Z and moves none, and whether the PFS holds its lock while the stage looks at the
walls is **not verified** (see the end).

## At the stand

1. **Brightfield, with the light on**, the objective you will scan with (the 10× is the
   one measured on the Ti2-E). The inside of a well must be brighter than the plastic
   around it: that is how the wall is told from everything else.
2. **Calibrate the camera at that objective**, on a field with texture; a well's wall
   will do:

   ```text
   smc run camera-calibration -p <profile> --run-doc run.json
   ```

   The tool needs the camera's turn: the Ti2-E's 10× camera is turned 180°, so
   assuming an upright camera would send every search into the plastic. With no
   calibration, or one of another objective or frame, it refuses before it moves.
3. **Centre two or three wells by eye with the joystick**, far apart (A1 and the far
   corners; two wells fix the rotation, a third checks the fit). Centre each one
   roughly: it may be up to **half the well's radius** off, 3 900 µm on a 24-well
   plate and 1 600 µm on a 96-well one. Read each position:

   ```text
   smc stage get -p <profile>
   ```

4. **Register the plate:**

   ```text
   smc run plate-walls -p <profile> --run-doc run.json --param plate=24-well --param 'refs=[{"well": "A1", "x_um": 1000, "y_um": -2000}, {"well": "D6", "x_um": 97500, "y_um": -59900}]'
   ```

   The Ti2-E's Greiner plate is `plate=greiner-24-well` (19.45 mm pitch), not
   `24-well` (19.0 mm). A refusal of the spacing by a few percent (`the measured
   well spacing is off the 24-well definition by +2.4%`) is the wrong plate
   definition, not a bad measurement: see [Plate names](README.md#plate-names).

   The commands are written for a POSIX shell. In `cmd.exe` and PowerShell 5.1,
   escape each inner double quote with a backslash, as the
   [`smc run`](README.md#running-a-plugin-smc-run) page says:

   ```bat
   smc run plate-walls -p <profile> --run-doc run.json --param plate=24-well --param "refs=[{\"well\": \"A1\", \"x_um\": 1000, \"y_um\": -2000}, {\"well\": \"D6\", \"x_um\": 97500, \"y_um\": -59900}]"
   ```

   That form was **not tried on a Windows shell**: check it at the stand. A value that
   arrives broken is refused before anything moves (`'refs' takes JSON`).

   It takes about a minute. The stage ends in the middle of the last well, which is
   what you see on Live. The plate page shows the same tool under *More* with its own
   form; the one-click path from the page is a later issue.

The stage is only ever moved around the positions you gave. If a stop (Ctrl-C), a halt
or a fault ends the run, the stage stays where it is and a WARNING says where, in
which well and on which wall, and where it started: nothing moves to put it back.

## Parameters

```text
smc run plate-walls --show-params
```

| Parameter | Default | Meaning |
|---|---|---|
| `plate` | `96-well` | A `useq` plate name; the name is exact. The wells must be round: `384-well` and `1536-well` have square wells and are registered with `plate-registration`. |
| `refs` | none | Two to six wells, each with the stage position where you centred it, JSON: `[{"well": "B3", "x_um": 10200, "y_um": -9700}]`. A well name is as you type it (`b3`, `B03`). |
| `search_um` | half the well's radius | How far, on each axis, the wall may lie from where your position puts it. It may not exceed the radius. |
| `max_residual_um` | 100 | The largest distance, in µm, between a well's measured centre and where the plate's fit puts it. Revisit it at the stand once a plate has been registered. |
| `max_rotation_deg` | 10 | The largest rotation of the plate that the fit accepts (above 0, up to 45). |
| `settle_ms` | 0 | Wait this long after each move before the frame is taken. |
| `save_frames` | true | Save the frame centred on each wall, see below. |

## What it does

For each well, in the order given, and each of its four walls in the order `+X`, `+Y`,
`-X`, `-Y` (the **stage's** directions: no side is ever named on the image):

1. It looks for the wall at the well's nominal radius from your position, then a step
   nearer and a step further, two steps nearer and two further, and so on, up to
   `search_um`. The step is three quarters of the frame's smaller side. In each frame
   the wall is the first fall from bright to dark that a line meets going outwards from
   the inside of a well. It stops at the first frame whose wall lies within `search_um`
   of where the nominal radius puts it.
2. It moves once more, to put the wall at the frame's centre, and takes the frame whose
   points it keeps. Every side is framed the same way, so that the bias of the
   threshold (the level falls towards the wall: the meniscus) is the same on opposite
   walls and cancels in the centre.
3. After the four walls, it fits a circle to the points: the well's centre and
   diameter.

Then it fits the plate to the centres, as `plate-registration` does, and checks that
no well is further than `max_residual_um` from where the others put it. The stage
moves a last time, to the last well's fitted centre.

A frame is placed where the stage says it was, never where it was sent.

## What you get

The `plate` slot, as ever (A1's centre, the rotation, the residual, the scale error),
with:

- `method`: `wall-images`; `refs`: the fitted centres, in the order you gave the wells;
- `walls`: per well, where you put the stage (`recorded_xy_um`), the fitted
  `center_xy_um`, `diameter_um`, `size_error` (the diameter over the plate's, minus 1),
  `rms_um` (how far the points lie from the circle) and, per wall, where the frame was
  taken, its edge points and how many frames the side took; the pixel size and frame of
  the calibration, the search distance, and the seconds the run took.
- `assumed`: that the stage reports no soft limits, when it does not; that **two wells**
  check their distance only, a third checks the fit; and the frames that could not be
  saved.

**Frames.** With `save_frames` (the default) the centred frame of each wall is a TIFF,
`plate-walls/<token>/<well>_<side>.tif` beside the run document (`B3_+X.tif`), with the
well, the side and the stage's readback as its metadata. Look at them when a well is
refused: a bubble, a scratch or a label shows. A frame that cannot be written is a line
in `assumed`, never a failure of the registration.

## What each refusal means

A refusal is a result with a reason, and the run document records it `failed`. The
first group is checked **before anything moves**.

| It says | It means, and what to do |
|---|---|
| `a dry run cannot image the walls; run it without --dry-run` | The tool moves and images; there is nothing to simulate. |
| `plate-walls fits round wells, and the <plate> plate's are square: register it with plate-registration` | `384-well` and `1536-well`. |
| `give at least two wells in refs, …` / `give at most 6 wells in refs, got <n>` | Two to six wells. |
| `<well> is not a well of the <plate> plate` / `<well> is given twice in refs` | A typo, or the wrong `plate`. |
| `<well>'s recorded position (<x>, <y>) µm is beyond ±1e+06 µm: is it in µm?` | A unit slip: positions are µm. |
| `the camera calibration is not of this stand: it is of <a>, the camera is at <b>` (or `of a frame of <shape>, the camera gives <shape>`) | The calibration was made at another objective or frame. Run `camera-calibration` with this objective in place. |
| `the camera calibration's matrix cannot be inverted …` | A calibration edited by hand, or a camera that imaged nothing. Run it again. |
| `no objective is in place: the turret is at an empty position …` | Turn the nosepiece to an objective. |
| `a frame of <h> × <w> px is too small to find a wall in` | Frames under 128 px on a side hold too little wall. |
| `the field of <f> µm holds a whole well of <d> µm: use a higher magnification` | The wall is then not a line across a frame. |
| `search_um <s> is more than the well's radius <r> µm` | The search may not pass the well's centre. |
| `up to <n> frames: use a lower magnification or a smaller search_um` | More than 600 frames could be needed. |
| `<well>'s <side> wall would be looked for at (<x>, <y>) µm, outside the stage's soft limits` | A frame of the walk would leave the travel: put the well further inside, or give a smaller `search_um`. |

The others come **after the stage has moved**, which stays where it is and says where in
a WARNING:

| It says | It means, and what to do |
|---|---|
| `no wall found on <well>'s <side> side in <k> frames, from <a> to <b> µm out of the recorded centre …` | No frame showed a wall. Check that the light is on (the inside of a well is brighter than the plastic in brightfield), that the stage was in that well when you recorded it, then record it again. |
| `the wall of <well>'s <side> side was lost in the frame centred on it …` | The wall was found, then not in the frame put on it: the sample moved, or the exposure changed. Run it again. |
| `<well>'s <side> wall: the move to (<x>, <y>) µm was refused: …` | The layer refused a move (the soft limits, or a move already under way). |
| `the frame has shape <s>, the calibration <c>: check the camera's binning and region` | The camera is not as it was when it was calibrated. |
| `the stage's readback is not a finite position` | The stage did not answer a position. |
| `<well>'s walls do not fit a circle: …` | The edge points do not lie on a circle: look at that well's frames. |
| `<well> measures <D> µm across and the <plate> plate's wells <D0> µm: is it the right plate type?` | More than 10 % off the plate's well size: another plate, or the wrong `plate`. |
| `<well>'s wall points lie <rms> µm from a circle on average, more than <bound> µm: a frame may show something else than the wall …` | A bubble, a scratch or a label took the place of a wall (the bound is 2 % of the radius). Look at the frames. |
| `the fit turns the plate by <r>°, beyond ±<m>°: a reference well is misidentified …` | See the next paragraph. |
| `<well> is <n> µm from where the other wells put it (bound <b> µm): record it again, or check the plate type` | A well that the others do not put where it is, or a plate whose pitch is not the one named (a 96-well plate of pitch 9.03 mm fitted as 9.0 mm leaves two wells 196 µm off). |

**A rotation near 180°.** If the stage runs the plate's columns towards −X, two wells
fit a rotation near 180°, and the fit refuses it. That is the safe answer, and
`max_rotation_deg` is not to be raised for it (it stops at 45°): which way the
Ti2-E's plate runs is [#327](https://github.com/EPFL-TOP/smart-microscope-control/issues/327)'s question.

## Trying it without a microscope

On the demo stand with a synthetic plate, with the sample's own centres moved by a few
hundred µm (A1 and H12 of the centred 96-well plate are at (−49 500, 31 500) and
(49 500, −31 500) µm):

```text
smc run camera-calibration -p demo-sample --run-doc run.json
smc run plate-walls -p demo-sample --run-doc run.json --param plate=96-well --param 'refs=[{"well": "A1", "x_um": -49200, "y_um": 31300}, {"well": "H12", "x_um": 49250, "y_um": -31350}]'
```

(Written for a POSIX shell; in `cmd.exe` escape the inner double quotes of `refs` as in
step 4 of *At the stand*.)

It takes about 20 seconds on the demo stage (21 frames), puts A1 within a micrometre of
(−49 500, 31 500) and the rotation at 0°. The camera is the demo's, the walls are the
sample's: that proves the loop, not the stand.

## Not verified

At the Ti2-E: a registration of three wells (its residual and its time), and `goto-well`
to a fourth well (record that well where `goto-well` left the stage and run `plate-walls`
on it with one of the first three: its `center_xy_um − recorded_xy_um` is how far
`goto-well` landed from its centre, to be under 100 µm); the 4× objective; the light and
exposure the walls need; plates other than the 24-well; which way the plate's columns
run on the stage; whether the PFS holds its lock while the stage looks at the walls.
`max_residual_um = 100` was set before any plate was registered from images.
