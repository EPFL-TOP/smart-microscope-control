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
| `landing_tolerance` | 0.05 | How far a field may land from its target, as a fraction of the smaller side of the plan's field of view (above 0, at most 0.5). A field that landed farther is failed, not photographed. 0.05 is 4.8 µm for the demo's 10× (a field of 128 × 96 µm). Read the stand's real agreement at the first hardware session before trusting a fraction. |

## What is checked before anything moves

In this order; the first that fails ends the run with nothing moved and nothing
written, and the message says what to do:

1. **The checkpoint** (with `--resume`): its wells are the plan's, and every frame it
   names is still in the run folder (`1 of the checkpoint's 4 frame files are missing
   from the run folder, the first scan/…/A1/A1_0001.tif: run scan without --resume`).
2. **The plan's field of view** is positive on both sides (a run document that was
   edited is refused: `the plan's field of view (0.0, 96.0) µm is not positive: run
   scan-plan again with the stand as it is now`).
3. **Every frame's path**, each resolved once here and not again at the frames: a well
   or a field name that is not letters, digits, `_` and `-` cannot become a file name,
   and is refused; nor can a name Windows keeps for a device (`CON`, `PRN`, `AUX`,
   `NUL`, `COM0` to `COM9`, `LPT0` to `LPT9`, in any case); two fields whose names
   differ only in case would write one file on NTFS and APFS, and are refused
   (`fields 'A1_0000' of well 'A1' and 'a1_0000' of well 'A1' would write one file…`).
   **On Windows**, a frame's path must be shorter than 260 characters *while it is
   written*: the temporary file beside it is 18 characters longer than the frame's own
   name, and that is the length checked (`a frame's path would reach 262 characters
   while it is written, and Windows takes 259 at most unless long paths are enabled
   (not assumed): move the run document to a shorter folder, then run scan again`). Long
   paths are not read, so a PC that has them enabled is refused too: the fix is a
   shorter folder. A run folder that cannot be read at that moment is the message of
   the checkpoint's check (`the run folder cannot be read (…): … then run scan --resume`).
4. **The objective**: when the plan names one and the stand has a turret, the
   objective in place must be it:
   `the objective in place is 'Nikon 40X Plan Fluor ELWD'; the scan was planned for
   'Nikon 10X S Fluor'. Turn the turret to 'Nikon 10X S Fluor' (smc never turns it
   during a scan), or run scan-plan again with objective=Nikon 40X Plan Fluor ELWD`.
5. **The camera's frame shape** equals the plan's (a binning or a region of interest).
6. **The camera's pixel size** against the plan's: within 1 % for a plan from the
   profile, within 10 % for a plan from a camera calibration (a measured size is a few
   per cent from the profile's). The magnifier lever at 1.5× is 33 % and is refused. A
   plan whose pixel size was a parameter is not compared, and `assumed` says so. A
   camera that reports no pixel size (0, or a value that is not finite) is not
   compared either, and `assumed` says that neither the plan's size nor a change
   during the scan was checked.
7. **The autofocus**, as `autofocus` says (above).
8. **The disk**: the free space of the run folder's disk must exceed 1.1 × the
   frames left (about 4.4 MB per field at 1412 × 1412 pixels of 16 bits, 1.3 GB for
   the 305 fields of a 96-well well at 40×).

## During the scan

**At both ends of every well**, before its first field and after its last, the
objective, the frame shape and the pixel size are read again. A turret turned by
hand, a binning changed or the lever moved (a pixel size more than 1 % from the one at
the start) ends the scan. A well is marked done only when the stand was the same at
both ends, so a change in the middle of a well never leaves frames of another scale in
the checkpoint: set it back and `--resume`, and that well is scanned again.

**At every field**: a stop asked for is honoured, the stage moves, and then (not in a
dry run) **the readback is compared with the target**, before anything else is read or
moved. If it is farther than `landing_tolerance` of the smaller side of the field of
view, measured over both axes (or is not a number), the field is failed (below) and
nothing more is done for it: no autofocus call, which could drive Z, and no frame of
the wrong place. A dry run does not compare: the layer's dry-run readback is the target.
Then the autofocus is checked when it is used: if it lost its lock, it is engaged once;
the focus drive's Z and the autofocus offset are read, and the frame is taken and written
whole. **An autofocus you switch off during the scan stays off**: its fields are recorded
`unlocked`, and `max_unlocked_in_row` of them end the scan. An engage of the scan's own
that failed (it returned without a lock, or raised) is tried again at the next field, as
the protocol leaves PFS off; a lock you take yourself ends that, so a switch-off after it
is yours. `assumed` says `the autofocus was engaged again on <a> fields and locked on
<l> of them`. A frame whose shape is not the plan's ends the scan and is not written.

**A field that fails** (the stage refuses the target or does not answer, it lands off
target with `StageOffTarget: the readback is 6.0 µm from the target, more than 4.8 µm
(target (…), readback (…))`, the camera times out) is recorded in `failed` with its
**target** and one line of the error, and the scan goes on. `max_failed_in_row`
failures in a row end it: a stage that refuses every move, or is stuck, must not read as
a scan of failed fields. **A halt is not a failed field**: the emergency stop, or a move
that was stopped, ends the scan where it is.

**A scan with no frame is not `done`.** When every field failed, with fewer than
`max_failed_in_row` in a row (a bound of 100, say) and no frame from a checkpoint
either, the scan ends as `failed`: `every field failed (12), the first A1_0000: …:
check the stage and the camera, then run scan again without --resume (a resume skips
the wells already walked)`. The checkpoint keeps every well, so a `--resume` meets the
same refusal; run it again without `--resume`. A scan with one frame, from this attempt
or from the checkpoint, stays `done` with its failures listed. A dry run is held to the
same rule: its walked moves are the fields.

**Writing a frame that fails** ends the scan as it is: it is not a field's fault, and
every next frame would fail the same way. The error is a `PluginRunError` that names
the frame (`scan/<token>/A1/A1_0001.tif`), and the checkpoint keeps the wells done:

- A file another program holds open (a viewer, a backup or antivirus agent): `the frame
  … cannot be replaced: another program holds it open (a viewer, a backup or antivirus
  agent): close it, then run scan --resume`. The write tries five times, 0.05 s apart,
  which only rides out a momentary hold.
- Anything else (a full disk, a network share that dropped, a folder that is read-only):
  `the frame … could not be written (<the OS's reason>): free some space if the disk is
  full, or check the network share, then run scan --resume`.

The scan does not test beforehand whether a file is held open: opening a file does not
show the condition that blocks its replacement.

The scan **never** moves Z, turns the turret, switches the autofocus on, changes its
offset, or returns to a start position.

## Resuming

The checkpoint is saved after each well. `--resume` continues with the same folder
(the same token), keeps the frames of the wells done and scans the others; the well
that was interrupted is scanned whole again and its files are rewritten. A resume is
refused when the plan was made again since (`scan read its inputs from an earlier run,
so its checkpoint no longer matches them; run it without --resume`), or when a frame
of the checkpoint is missing. A run folder that cannot be read at that moment (a
network share that dropped) is not a missing frame: the message says so and `--resume`
stays the way on. Every count (failures, unlocked fields in a row) starts at 0 in each
attempt.

**Close any viewer that has the interrupted well's frames open** (Fiji, Explorer's
preview pane) before `--resume`: those files are rewritten, and Windows refuses to
replace a file another program holds open, which ends the scan at that field (with the
message above, and the file's name). Close the viewer and `--resume` again: the well is
scanned once more.

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
- `failed`: `well`, `index`, `name`, the target `x_um`, `y_um`, and `error` (a field
  that landed off target says `StageOffTarget: …`).
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
`max_failed_in_row`); the write speed and the antivirus on the microscope PC; whether
the Ti2-E's readback agrees with the commanded position within 5 % of a field (the first
session reads it before `landing_tolerance` is trusted, and sets it from what it sees);
a TIFF held open on Windows during `--resume` (the `PermissionError` path is tested with
a stand-in, never with a real viewer); and a run folder past 259 characters on a Windows
PC without long paths (the refusal is tested by patching the platform and the limit,
and the failure it prevents was not reproduced; whether long paths are enabled is not
read, so a PC that has them is refused too).
