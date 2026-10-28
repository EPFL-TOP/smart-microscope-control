# `scan`

Moves the XY stage to every field of the scan plan, takes one frame at each and
writes it as a TIFF under the run folder, with where the stage really was and what
the autofocus was doing. It is the stage that **moves**; everything it will need is
checked before the first move. The design is
[`m3-plugins.md`](../design/m3-plugins.md) §8; this page is how to use it.

It reads the `scan-plan` slot ([`scan-plan`](scan-plan.md)): the wells in visiting
order, the fields of each, the objective, the frame shape and the pixel size the plan
was made for. It does not read the wells or the plate again.

## Before you start

- **The objective the plan names is in place.** The scan never turns the turret: a
  turret turned under a loaded plate can drive the objective into it (FM-14).
- **The camera is as it was for the plan**: the same binning, region of interest and
  magnifier lever. The plan's matrix is only right on that frame.
- **The autofocus (PFS) is either locked on the sample, or off.** The scan never
  switches it on. With the default `autofocus=if-engaged` it is used when it is on and
  locked when the scan starts, and a scan with it on but not locked is refused:
  engaging it drives Z to the plane of whatever offset is stored, and only a lock you
  took on this sample shows that offset is yours (#205).
- **The disk has room** for every frame left, plus a tenth.

## Running it

```text
smc run scan -p <profile> --run-doc run.json
smc run scan -p <profile> --run-doc run.json --param autofocus=required
smc run scan -p <profile> --run-doc run.json --resume
smc run scan -p <profile> --run-doc run.json --dry-run
```

| Parameter | Default | Meaning |
|---|---|---|
| `autofocus` | `if-engaged` | `if-engaged`: use it when it is on and locked at the start. `required`: refuse to start unless it is. `off`: never touch it, not even to read it. |
| `max_failed_in_row` | 3 | This many fields failing in a row end the scan (1 to 100). |
| `max_unlocked_in_row` | 5 | This many fields in a row on which the autofocus would not lock end the scan (1 to 1000). |

## What is checked before anything moves

In this order; the first that fails ends the run with nothing moved and nothing
written, and the message says what to do:

1. **The checkpoint** (with `--resume`): its wells are the plan's, and every frame it
   names is still in the run folder (`1 of the checkpoint's 4 frame files are missing
   from the run folder, the first scan/…/A1/A1_0001.tif: run scan without --resume`).
2. **Every frame's path**: a well or a field name that is not letters, digits, `_`
   and `-` cannot become a file name, and is refused.
3. **The objective**: when the plan names one and the stand has a turret, the
   objective in place must be it:
   `the objective in place is 'Nikon 40X Plan Fluor ELWD'; the scan was planned for
   'Nikon 10X S Fluor'. Turn the turret to 'Nikon 10X S Fluor' (smc never turns it
   during a scan), or run scan-plan again with objective=Nikon 40X Plan Fluor ELWD`.
4. **The camera's frame shape** equals the plan's (a binning or a region of interest).
5. **The camera's pixel size** against the plan's: within 1 % for a plan from the
   profile, within 10 % for a plan from a camera calibration (a measured size is a few
   per cent from the profile's). The magnifier lever at 1.5× is 33 % and is refused. A
   plan whose pixel size was a parameter is not compared, and `assumed` says so. A
   camera that reports no pixel size (0, or a value that is not finite) is not
   compared either, and `assumed` says that neither the plan's size nor a change
   during the scan was checked.
6. **The autofocus**, as `autofocus` says (above).
7. **The disk**: the free space of the run folder's disk must exceed 1.1 × the
   frames left (about 4.4 MB per field at 1412 × 1412 pixels of 16 bits, 1.3 GB for
   the 305 fields of a 96-well well at 40×).

## During the scan

**At every well**, before its first field, the objective, the frame shape and the
pixel size are read again. A turret turned by hand, a binning changed or the lever
moved (a pixel size more than 1 % from the one at the start) ends the scan; the
checkpoint keeps the wells done, so set it back and `--resume`.

**At every field**: a stop asked for is honoured, the stage moves, and then (not in a
dry run) the autofocus is checked when it is used: if it lost its lock, it is engaged
once (`assumed` counts these), the focus drive's Z and the autofocus offset are read,
and the frame is taken and written whole. A frame whose shape is not the plan's ends
the scan and is not written.

**A field that fails** (the stage refuses the target or does not answer, the camera
times out) is recorded in `failed` with its **target** and one line of the error, and
the scan goes on. `max_failed_in_row` failures in a row end it: a stage that refuses
every move must not read as a scan of failed fields. **A halt is not a failed field**:
the emergency stop, or a move that was stopped, ends the scan where it is.

**Writing a frame that fails** (a full disk, a lost network share) ends the scan as it
is: it is not a field's fault, and every next frame would fail the same way.

The scan **never** moves Z, turns the turret, switches the autofocus on, changes its
offset, or returns to a start position.

## Resuming

The checkpoint is saved after each well. `--resume` continues with the same folder
(the same token), keeps the frames of the wells done and scans the others; the well
that was interrupted is scanned whole again and its files are rewritten. A resume is
refused when the plan was made again since (`scan read its inputs from an earlier run,
so its checkpoint no longer matches them; run it without --resume`), or when a frame
of the checkpoint is missing. Every count (failures, unlocked fields in a row) starts
at 0 in each attempt.

## Dry run

`--dry-run` walks the fields through the layer's dry run: the moves are logged and not
sent, no frame is taken, the autofocus is not touched beyond the checks, and **no file
is written** (the run document aside). The result has no frames and says
`dry run: <n> fields walked, no frame taken`.

## The result

The slot `scan` holds a `ScanResult` (`smc.plugins.slots`):

- `folder`: `scan/<token>`, relative to the run folder. Each frame is
  `scan/<token>/<well>/<name>.tif`, POSIX, so the run folder can be moved or copied
  to another PC as a whole.
- `frames`: per frame, `well`, `index`, `name`, `file`; `x_um`, `y_um`, the stage's
  **readback** after the move (the point at the frame's centre, not the target);
  `z_um` (`null` without a focus drive); `focus` (`locked`, `unlocked`: used and not
  locked, `off`: the stand has an autofocus the scan did not use, `none`: it has
  none); `pfs_offset` (in the autofocus's own unit, `null` when not used); `t_s` from
  the start of the attempt that took it.
- `failed`: `well`, `index`, `name`, the target `x_um`, `y_um`, and `error`.
- `objective`, `pixel_size_um`, `matrix_um_per_px`, `image_shape_px`: the plan's,
  copied so that the cells stage needs only `scan`; `exposure_ms` as read at the start.
- `assumed`.

Each TIFF is the frame in the camera's own dtype, with the frame's record as its JSON
metadata. `load_frame(run_dir, frame)` reads one back and refuses a path that leaves
the run folder.

## Not verified here

Nothing in this page was measured on a microscope. For the first hardware session
(#198): the scan on the Ti2-E; PFS over a 96-well well at 40×; an `engage()` on a
field at 40×, after which Z may not yet have arrived when the frame is taken (#205,
which blocks scanning at 40× until it is decided); a move that timed out on a stand
that refuses `stop()` (m2 §11, the next moves are refused and end the scan through
`max_failed_in_row`); the write speed and the antivirus on the microscope PC.
