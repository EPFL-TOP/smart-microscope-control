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
  switches on one that was off when it started. With the default `autofocus=if-engaged` it is used when it is on and
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
| `autofocus` | `if-engaged` | `if-engaged`: use it when it is on and locked at the start. `required`: refuse to start unless it is. `off`: never used, and **refused when it is on** (a lock would cross every well's wall): the one read `off` makes is whether it is on. |
| `check_plate` | true | Before the first field, re-measure two wells `plate-walls` measured and refuse a plate that moved more than `max_plate_shift_um` since ([below](#the-plate-check)); about 20 s at the 10×. |
| `max_plate_shift_um` | 25 | How far a checked well may lie from its registered centre, in µm (above 0, at most 500). |
| `max_failed_in_row` | 3 | This many fields failing in a row end the scan (1 to 100). |
| `max_unlocked_in_row` | 5 | This many fields in a row on which the autofocus would not lock end the scan (1 to 1000). |
| `landing_tolerance` | 0.05 | How far a field may land from its target, as a fraction of the smaller side of the plan's field of view (above 0, at most 0.5). A field that landed farther is failed, not photographed. 0.05 is 4.8 µm for the demo's 10× (a field of 128 × 96 µm). Read the stand's real agreement at the first hardware session before trusting a fraction. |
| `cell_focus` | `off` | The focus on the cells ([below](#the-focus-on-the-cells-cell_focus)): `per-field` sweeps Z at every field, `per-well` until one field of the well finds a peak; `map` takes each field's Z from a 10× scan taken under the lock, plus one offset per well ([below](#the-focus-from-the-10-lock-map-cell_focusmap)). |
| `focus_range_um` | 10 | The sweep's half-range, in µm (above 0, at most 50, and at most the stand's Z jog limit). Measured at the Ti2-E's 40×, m3 §21.11.6: the cells' focus curve is 9-11 µm wide at half its rise, and ±10 µm at 1 µm finds 17 of 24 cell peaks where ±5 µm finds 3. |
| `focus_step_um` | 1 | The distance between two planes, in µm (0.1 to 10, at most `focus_range_um`, at least 9 and at most 101 planes: the sweep measures its noise between neighbouring planes). |
| `focus_score` | `brenner` | `brenner`, `normalized-variance`, `gradient-energy` or `laplacian-variance`. `focus_step_um` and `focus_score` are chosen, not measured. |
| `focus_map` | none | `cell_focus=map` only, and required by it: a **copy** of the run document whose `scan` was taken with the autofocus locked on this registration of the plate (the 10× scan the fields were planned from). Never the run document this run writes: that one is refused before anything is written. |
| `map_check_every` | 0 | `cell_focus=map` only: a field whose index in its well is a multiple of this also sweeps around the map's Z, and its frame stays at the map's Z (0 to 100 000; 0 checks none). |

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
   Last, after every path, each well that has fields must be one the plan's plate holds,
   because every frame's offset from its well's centre is taken from that centre (an
   edited run document is refused, with nothing moved: `the plan has fields in well
   'Z9', which its plate does not hold: run scan-plan again with the stand as it is
   now`).
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
7. **The autofocus**, as `autofocus` says (above). With `autofocus=off`, an autofocus that
   is on or still searching is refused, and so is one whose state cannot be read (a dry
   run too, and on a resume): `autofocus=off, and the autofocus is on: the scan would carry
   its lock over every well's wall with no guard (m2 §12.2 D2): run with
   autofocus=if-engaged to use it, or switch it off first`, or `autofocus=off, and
   whether the autofocus is on could not be read (<error>): a lock would cross every
   well's wall: switch it off at the stand, then run the scan again`. The scan does not
   switch it off for you: `off` is what you asked, and you may have meant `if-engaged`.
   It is the one read `autofocus=off` makes of the autofocus. With `--resume`, a scan whose
   checkpoint holds frames taken with the autofocus locked is refused when it is not
   locked now (and `autofocus` is not `off`): `the scan being resumed used the
   autofocus, which is not locked now (the scan switches it off between wells, and a
   stop during the focus on the cells leaves it off): lock it on the sample, then run
   scan --resume`. Without it, `if-engaged` would
   resume the rest of the plate without PFS and say nothing.
8. **The disk**: the free space of the run folder's disk must exceed 1.1 × the
   frames left (about 4.4 MB per field at 1412 × 1412 pixels of 16 bits, 1.3 GB for
   the 305 fields of a 96-well well at 40×).
9. **The focus on the cells**, when `cell_focus` is not `off` (a dry run too): the
   stand has a focus drive (`cell_focus needs a focus drive, and this stand has none`),
   the profile sets `[safety] z_soft_limits_um` (`cell_focus moves Z: the profile must
   set [safety] z_soft_limits_um`), `focus_range_um` is not above the Z jog limit, and
   the sweep around the Z read now lies inside the soft limits (`the focus sweep around
   <z> µm would reach <plane> µm, outside the soft limits [<lo>, <hi>] µm`). With
   `autofocus=off`, an autofocus that is on is refused, as check 7 says, for every
   `cell_focus`: the sweep's Z moves would switch it off, and nothing would lock it again
   or say so. This is the same refusal and the same text, the one read `autofocus=off`
   makes of the autofocus. With `cell_focus=map`, the checks of the map follow, in
   [their own order](#before-anything-moves).
10. **The plate check's walk** ([below](#the-plate-check)), when the check is made: the
    frame must not hold a whole well, the walk must not be too long, and every frame of
    it must lie inside the stage's soft limits, as for
    [`plate-walls`](plate-walls.md#what-each-refusal-means). Such a refusal ends with
    `(the plate check before the first field; nothing moved: check_plate=false scans
    without it)`.

## The plate check

A plate lifted and put back since it was registered turns and slides, and every frame's
place in its well is then read against a pose the plate no longer has: at Ti2-E session 5
(#550, F15) a 40× scan made after the plate had been lifted placed its frames about
(+19, −50) µm off, and nothing said so. So **every attempt** of the scan (a fresh one and
every `--resume`: a stop is when a plate is lifted) measures again, in its **first
well**, the centres of the two wells the registration measured that lie farthest apart,
with [`plate-walls`](plate-walls.md)' own loop and the plan's camera matrix, and compares
them with the registered centres moved into the plan's objective's frame. It is
[`plate-check`](plate-check.md)'s check, made by the scan.

- **When**: after the first well's own checks and its switch-off of the autofocus
  (below), before its first field. No lock crosses the walls of the check's wells, and
  the first field engages PFS again as after any wall. The scan's progress says
  `checking the plate: A1 +X` and so on, at field 0.
- **A plate that moved** more than `max_plate_shift_um` ends the scan `failed` with **no
  frame** of this attempt and every number in the reason: `the plate moved since
  plate-walls registered it: B2 lies 31.9 µm from its registered centre (-14.5, -28.4)
  and B5 33.5 µm (-16.7, +29.0) under <objective>, more than 25 µm; it turned about
  -0.056° and slid about (-15.6, +0.3) µm: register it again (plate-walls), then run
  scan-plan and scan again, without --resume: frames already taken are placed against
  the old registration`. Under another objective than the registration's, the first
  well's clause names the offset its distance is from: `A1 lies 50.0 µm from its
  registered centre plus the offset (-37.5, +12.2) µm to <objective> (-50.0, +0.0)`, so
  that the numbers can be read against `plate-walls`' record; under the registration's
  own objective the text is the one above. When the scan had switched PFS off at that
  well, the reason ends with `; the autofocus was switched off before anything moved and
  was not engaged again: engage it again on the sample`. A stop, a halt or a fault
  during the check ends the scan as it would anywhere (`stopped` stays `stopped`), and
  after that switch-off a WARNING says that nothing engaged PFS again: lock it on the
  sample before the next scan, since no frame of the attempt records that it was used.
- **A check that could not measure** (no wall in view: a light path that shows none, a
  fluorescence one say, or a plate moved by more than the search of 1000 µm; a wall lost
  in the frame centred on it; walls that fit no circle) ends the scan `failed`, with no
  frame of the scan, and its reason ends with the parenthesis `(the plate check before the first field: check_plate=false
  scans without it)`. It does not say `nothing moved`: the stage moved. After the
  switch-off of the autofocus the reason ends with that parenthesis, then the
  autofocus's sentence. The verdict of a plate that moved never carries it: registering
  the plate again is its answer, and `check_plate=false` would scan the plate where it
  is not.
- **A plate that did not move**: the result's `plate_check` holds the check, with
  `checked_utc`, and the first line of `assumed` after the stand's says `the plate was
  checked before the first field: A1 and H12 lie within 0.4 µm of their registered
  centres under <objective> (bound 25 µm), at 2026-11-18T09:12:33+00:00`. The check's
  frames are kept as `plate-check/<token>/<well>_<side>.tif`. Next come the check walk's
  own lines (on a stage that reports no soft limits: `the stage reports no soft limits,
  so the frames' moves were not checked against any travel`), then, when frames could not
  be saved, `n frames of the plate check could not be saved (the log says why)`.
- **When it is not made**, the scan goes on, logs one WARNING and says why in `assumed`,
  first after the stand's lines, in this order:

  | when | the line |
  |---|---|
  | `check_plate=false` | `the plate was not checked against its registration (check_plate=false)` |
  | a dry run | `dry run: the plate was not checked against its registration` |
  | no well is left (a `--resume` of a scan whose wells are all done) | `the plate was not checked: every well of the plan was done by an earlier attempt, so this one takes no frame` |
  | the plate record could not be read | `the plate was not checked: plate not read: <why>` |
  | the plate was registered by `plate-registration`, `plate-finding` or the holder's estimate | `the plate was not checked: it was registered by <method>, which measures no well's walls (plate-walls does)` |
  | the plan does not record the objective of its registration (an older scan-plan) and the plate record is under one | `the plate was not checked: the plan does not record the objective of the registration it was made from (an older scan-plan), and the plate record is under <plate objective>: run scan-plan again to check it` |
  | the plan was made from another registration's objective | `the plate was not checked: the plan was made from a registration under <a>, the plate record is under <b>` |
  | the offset between the plate's objective and the plan's is not measured | `the plate was not checked: the offset between <a> and <b> is not measured (the profile's [objectives]), and a check would read it as a move` |
  | the plan's camera matrix is the profile's or a parameter | `the plate was not checked: the plan's camera matrix is the <source>'s, and a wall is looked for with a measured one only` |

  A plate record that exists but that this smc cannot read (a plate name it does not know,
  say) is not a skip: the scan ends `failed` before anything moves, with `scan reads plate,
  but the result of plate-walls is not what scan expects (…); run plate-walls again; or
  scan with check_plate=false, which does not read the plate`.

**The cost** is about 20 s at the 10× per attempt (two wells of 8 frames and the travel
between them, estimated from session 5's registration; not measured at the stand). **The
bound**, 25 µm, is chosen and revisited at the stand ([`plate-check`](plate-check.md)).

## During the scan

**At both ends of every well**, before its first field and after its last, the
objective, the frame shape and the pixel size are read again. A turret turned by
hand, a binning changed or the lever moved (a pixel size more than 1 % from the one at
the start) ends the scan. A well is marked done only when the stand was the same at
both ends, so a change in the middle of a well never leaves frames of another scale in
the checkpoint: set it back and `--resume`, and that well is scanned again.

**PFS never crosses a well's wall locked.** Locked over a wall, PFS follows its
reference to the plane its offset names in the next well, and nothing reads Z while it
does: on one 24-well plate at the 40×, that plane moved by at least 104 µm between two
wells whose foci by eye lie 11.7 µm apart (Ti2-E session 3, F11). So when the scan uses
the autofocus, before the first field of every well (the first well too, and not in a
dry run) it switches PFS off and reads it back off; the stage then moves to the well
with PFS off, and Z stays where PFS held it in the well before. The well's first field
engages it again at your offset, bounded by `[safety] max_engage_excursion_um` around
that Z (m2 §12.2 D2). It costs one engage per well. An autofocus that is already off at
a wall (you switched it off, or the focus on the cells did) is left as it is. A
switch-off that raises, or a PFS that still reads on after it, ends the scan before
anything moves to the well: `the autofocus could not be switched off before well
<w> (<error>): nothing moved to it, and PFS may still be on: switch it off at the
stand, then run scan --resume`, or `the autofocus still reads on after it was switched
off before well <w>: …`. `assumed` says `the autofocus was switched off before the first
field of <w> wells, so that no lock crossed a well's wall, and engaged again after it on
<a> fields, locked on <l> of them`, apart from the engages after a lost lock. With
`autofocus=off` the scan does not use PFS, so it cannot switch it off at a wall, and it
refuses to start while PFS is on (check 7): a lock left on would cross every wall.

**A trip ends the scan.** An engage that moves Z out of its bound (at a well's first
field, after a lost lock, or after the focus on the cells) switches PFS off and makes
the layer forget the offset, so every later engage would be refused. The scan ends,
with the wells done kept: `the autofocus moved Z out of its bound while it locked at
<field> (well <w>): …; the checkpoint keeps <n> wells. Its offset names another plane
here than where it held (m2 §12.11): …`. **What you do then**: with PFS off, focus on
the sample in that well by eye, lock PFS on it from below (m2 §12.10), and run
`scan --resume`; the well is scanned again whole. At the 40×, this replaces "one well
per run": lock PFS from below in the first well and scan the wells in one run; a trip
stops at the well whose plane moved.

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

The scan **never** turns the turret, changes the autofocus offset, or returns to a start
position. It moves Z itself only for the focus on the cells (below), which is off by
default. It switches the autofocus off and on again at every well's wall (above), for
the focus on the cells, and to engage a lock that was lost; never one that was off when
the scan started, or that you switched off.

## The focus on the cells (`cell_focus`)

PFS holds the plate's bottom; the cells sit a few micrometres above it, by an amount
that differs from well to well. With `cell_focus=per-field`, each field gets a short Z
sweep, and its frame is taken at the sharpest plane. It is the sweep of
[`focus-on-cells`](focus-on-cells.md), with the same checks and the same peak finder.

**At each field**, after the stage has landed and the lock is checked:

1. A field whose autofocus is used and reads **unlocked gets no step**: its Z is where a
   search left it. Its frame is taken as without the step, and `assumed` counts it.
2. The **centre** is the Z PFS holds at the field when the autofocus is used, else the Z
   read before the scan moved anything. It never follows the previous field's best plane,
   so nothing drifts from field to field. The PFS offset is read here.
3. The planes are checked against the soft limits; a sweep that would leave them fails
   the field with nothing moved (`SweepOutsideLimits: the focus sweep around …`).
4. **The autofocus is switched off** and read back off. A Z move under a lock is undone
   by the relock before the frame, and stepping the PFS offset needs the 40×'s offset
   factor, which is not measured.
5. The sweep, the peak, the move to the peak (or back to the centre when there is
   none: a flat sweep, a best at an end, two maxima), then the frame, **taken with the
   autofocus off**.
6. **The next field's re-engage locks it again**, and PFS takes Z back to its plane,
   which carries the plate's tilt to the next field. After the last field, one engage
   locks it again, on a normal end only; when no lock comes, `assumed` says `the
   autofocus did not lock again after the last field (<why>): it is off`.

With `cell_focus=per-well`, the fields of a well are swept in order until one finds a
peak; every later field of the well then moves to its own centre plus that field's
offset, with no sweep. An empty first field (a flat sweep) therefore does not decide a
well.

**A sweep that fails** (a readback more than a quarter step from its command that is
still off when read again once after 0.3 s, an autofocus that still reads on, a camera
that does not answer) is a failed field, and a
WARNING names the field and says the autofocus is left off. A halt or a stop ends the
scan as always, **with the autofocus off**: nothing restores it after a failure or a
stop, and every refusal of the scan then ends with `; the focus on the cells left the
autofocus off`. Lock it on the sample before `--resume`: the resume refuses otherwise
(check 7 above). A stop inside the first well leaves no checkpoint, so there is
nothing for that refusal to read: a scan with the step whose autofocus is present but
off when it starts says so, as a WARNING and in `assumed` (`the autofocus was off when
the scan started: the focus on the cells swept around the Z read then, and no frame
was taken with it (…)`).

**Each frame's `pfs_offset` is the lock's, not the cells' plane.** The frame's `z_um` is
the plane it was taken at (the cells'); its `pfs_offset` is the offset of the lock the
sweep was centred on, and `focus` stays the lock read at the field. An export that
engages PFS at that offset would image the plane PFS held, not the cells' plane: `assumed`
says how far apart they were. The scan records it so and does not derive anything: its
`pfs_offset` is a reading. [`route`](route.md) now turns it into the offset of the cells'
plane for its stops, from the frame's distance from the lock's plane and the profile's
factor for the scan's objective (or leaves the offset out and says why), and
[`nis-export`](nis-export.md) says when an offset it writes was derived. Carrying the
cells' plane into the PFS offset during the scan itself is a later change.

**What it costs** is measured and said: `assumed` gains `the focus on the cells added <a>
s per field (<n> fields: <b> s to sweep and move, <c> s for the lock after)`, how many
sweeps found a peak (`found a peak on <p> of <s> sweeps (flat <f>, at an edge <e>, two
peaks <t>): the others kept the centre`), and the fields skipped. A dry run plans it and
runs none of it: `dry run: the focus on the cells was planned (<n> planes per sweep), not
run`.

**A `--resume` of a scan stopped by an older smc** is refused as `the parameters differ`
(its parameters lack the four `focus_*` keys): run it again without `--resume`.

## The focus from the 10× lock map (`cell_focus=map`)

Until PFS locks at the Ti2-E's 40× (#539), a 40× scan runs with the autofocus off.
`per-field` centres every field's sweep on the Z the scan starts at, which cannot follow a
bottom that varies by 9 to 36 µm across a well, and costs 7.5–8.5 s a field. With
`cell_focus=map`, **each field's Z is the plate's bottom as the 10× scan under the lock
mapped it, plus one offset per well measured at the 40×** (m3 §21.12). Expected: 0.4–0.75
s a field plus 26–68 s a well for its anchor, estimated from measured parts; the next
visit measures it.

**The workflow** (m3 §21.12.7):

1. Scan the wells at the 10× with PFS locked, as usual, and **copy the run document right
   after the 10× scan, before planning the 40×** (`run-10x.json`): the 40× scan replaces
   the `scan` of the document it writes, and a copy taken after the 40× `scan-plan`
   replaced the 10× plan is refused (`the focus map in <path>: scan read its inputs from an
   earlier run; run it again`).
2. At the 40×, PFS off, focus by eye on cells near the first well's centre, then:

   ```text
   smc run scan -p nikon-ti2 --run-doc run.json --param cell_focus=map --param focus_map=run-10x.json --param map_check_every=25 --param autofocus=off
   ```

   One line: `cmd` and PowerShell do not read a trailing `\` as a continuation.

**The map** at a field is the Z of the 4 locked frames of the map's scan nearest the
field's **XY readback**, in the field's own well, weighted by 1/d (a frame within 1 µm
gives its own Z). Another well's frames are never read (a wall lies between), and a
frame that was not `locked` names no bottom and is not read either. Measured on session
6's 10× scan: 1.38 µm median error, leaving each frame out in turn.

**The anchors.** The operator's focus at the start gives the **start anchor**: the Z the
scan starts at minus the map at the stage's XY. Before each well's first field, the scan
sweeps (with `focus_range_um`, `focus_step_um` and `focus_score`) at the well's planned
fields **nearest its centre**, up to 8, each centred on the map there plus the start
anchor, until 3 find a peak; the well's anchor is the median of `peak - map` over them.
The centre, not the scan's order: the scan starts at the wall, where the map is least
sure. Every anchor is centred on the operator's, never on the previous well's, so
nothing drifts from well to well. A try that fails (a landing off target, a plane outside
the soft limits or beyond the Z jog limit of the Z the scan started at, a Z readback off
its command, a device error) is recorded with its error (`StageOffTarget`,
`SweepOutsideLimits`, `SweepBeyondJog`, `FocusDriveOffTarget`), a WARNING names it, and the
next field near the centre is tried: it is not a failed field. The anchors' frames are
neither written nor shown.

**A well where no sweep finds a peak** keeps the start anchor **only in the well the
operator focused in**: `start_well`, the well of the map's locked frame nearest the stage
at the start (check 5). The operator's focus is a measurement in that well alone, and the
10× → 40× difference is not a constant (it moved by 48 µm between two wells of one plate).
In any other well the scan has no anchor to give: the well is **not scanned**. Each of its
planned fields is a failed field with nothing moved for it (`MapNotAnchored: no sweep near
the centre of <well> found a peak (<t> tried), and the operator focused in <start_well>:
the well was not scanned`), they do not count toward `max_failed_in_row` (it is one
decision about one well, not a stage failing field after field), and the scan goes on with
the next well. A WARNING and `assumed` say so, for the operator's well too. **What to do
with such a well**: in a copy of the run document, plan that well alone (`scan-plan` with
`wells=<well>`), focus by eye on its cells there, and scan it with the same
`cell_focus=map` command. If every field of the scan fails this way, the scan ends
`failed` as any scan where every field failed.

**At each field**, after the stage has landed: the target is the map at the readback plus
the well's anchor; one Z move there. A target outside the soft limits fails the field with
nothing moved for it (`MapOutsideLimits: the focus from the map at <field> is <t> µm,
outside the soft limits [<lo>, <hi>] µm`), and so does a target beyond the Z jog limit of
the Z the scan started at (`MapBeyondJog: the focus from the map at <field> is <t> µm, <d>
µm from the Z the scan starts at, beyond the Z jog limit of <j> µm`): a peak found at an
anchor lies up to `focus_range_um` from the start anchor, and the map is read at the
readback, so a target can pass check 6's bound by about that. Then the frame. **The
autofocus is not touched**: the frame's `focus` is `off`, and [`route`](route.md) takes
its Z as recorded.
With `map_check_every=n`, a field whose index in its well is a multiple of `n` also
sweeps around the target, then moves back to it: **the check measures the map, it does
not correct it**, and the frame is still the map's. A check sweep that would leave the
soft limits or the Z jog limit is not made, and a WARNING and `assumed` say so.

### Before anything moves

After check 9's (a dry run too), in this order:

1. The scan uses the autofocus (on and locked): `cell_focus=map takes each field's Z from
   the map with the autofocus off, and the autofocus is locked: switch it off, or use
   per-field or per-well`.
2. `focus_map` is read as `camera-calibration`'s `from_run_doc` reads a calibration: its
   `scan` must be usable by this run (done, this profile, not a dry run read by a real
   run, not stale), and a result this smc reads. Its scan must have read **this run's
   plate registration**: `the focus map in <path> was scanned on another registration of
   the plate (<its attempt> against this run's <attempt>)` (`none` when either has none).
3. Every well left holds 4 locked frames of the map: `the focus map in <path> holds fewer
   than 4 locked frames in <wells>: scan them with the autofocus locked first`.
4. On `--resume`, the map is the one the scan being resumed used: `the focus map in <path>
   is not the one the scan being resumed used: run scan again without --resume`.
5. The stage's XY lies within the map scan's larger field side of a locked frame: `the
   stage at (<x>, <y>) µm is <d> µm from the map's nearest locked frame (<name> in
   <well>): focus by eye on the cells inside a mapped well, then start the scan`. That
   well is `start_well` (above). On `--resume` this check is not made: the start anchor
   and `start_well` are the ones the checkpoint holds, the operator's focus when the scan
   started, because a stop during a sweep leaves Z on a sweep plane up to `focus_range_um`
   away from it. **Refocusing by eye before a resume therefore changes nothing**: to
   anchor again, run the scan again without `--resume`.
6. Every planned field's `map + start anchor ± focus_range_um` lies inside the soft limits
   and within the Z jog limit of the Z the scan starts at (the moves are absolute, so the
   layer's jog guard does not reach them): `the focus from the map would reach <z> µm at
   <field>, outside the soft limits [<lo>, <hi>] µm`, or `…, <d> µm from the Z the scan
   starts at, beyond the Z jog limit of <j> µm`.

Then, **not a refusal**, the scan counts the fields far from every locked frame of their
well. The map is the Z of the 4 nearest locked frames weighted by 1/d, with no bound on
how far they lie, so a field in a band the 10× could not lock (near the wall, where the
bottom drops and PFS cannot lock either) gets a Z read from frames millimetres away. The
distance from each planned field to the nearest locked frame of its own well is measured
against the map scan's larger field side (1589 µm on session 6's 10× scan: every one of
C3's and C4's 1544 fields lay within 980 µm of a locked frame of its well, so a complete
map counts none; with the frames more than 6.5 mm from the centre taken out, 238 lay
beyond). The fields beyond it **are scanned like the others**: a refusal would leave the
whole well unscanned for its border, which the operator cannot fill in. A WARNING and a
line of `assumed` give their count, the farthest and its distance, and say that **the bound
is assumed, not measured**: the next visit says how far from a frame the map still holds
(m3 §21.12.7).

### What it records

Each frame's `cell_focus` has `source` `map`, `dz_um` 0.0, `map_z_um` (the map at its
readback) and `anchor_um` (its well's): its Z command was their sum. Its `sweep` is the
check's, or `null`. The result's `focus_map` holds the run document as typed, the map
scan's attempt and time, the `start_anchor_um` and the `start_well` of the attempt that
started the scan (a `--resume` keeps both; `start_well` is `null` when every well was done
before that attempt, and in a checkpoint of an older smc, where no well is the operator's
and a flat well is not scanned) and each well's anchor with its
tries (the field, the readback, the map there, the sweep or the error). A well not scanned
has `source` `none` and an `anchor_um` of `null`; the operator's flat well has `source`
`start`. The checkpoint
keeps the anchors of the wells done: a `--resume` keeps them and anchors only the others.
It is saved once before the first move too, with the start anchor, `start_well` and no
well, so that a stop inside the first well's sweeps leaves a `--resume` the operator's
anchor.
A scan without the map has neither `focus_map` nor the two keys of a frame, so it is
written as before; one with them is refused by an older smc. `assumed` says where the
map came from, the fields far from every locked frame (`<k> of <n> fields lie more than
<b> µm from every locked frame of their well, the farthest <field> at <d> µm: their Z is
read from frames that far (the bound, the map scan's larger field side, is assumed, not
measured)`, only when `k` is above 0), each well's anchor (`<well>: anchored <a> µm from
the map by <p> peaks of <t> sweeps near its centre`; or, with no peak, `<well>: no sweep
near its centre found a peak (<t> tried): the operator's focus anchors it, <a0> µm from
the map` in the operator's well and `<well>: no sweep near its centre found a peak (<t>
tried), and the operator focused in <start_well>: the well was not scanned` in another),
the time (`the focus from the map added <x> s per field (<n>
fields), and the anchors <y> s (<w> wells)`) and the checks (`the map's Z was checked on
<c> fields: a peak on <p>, within 1 µm of the map's Z on <q1> and within 2 µm on <q2>, the
median distance <m> µm`, or `the map's Z was not checked against a sweep
(map_check_every=0)`). A dry run makes every check above, then plans and moves no Z:
`dry run: the focus from the map was planned for <n> fields, Z <lo>…<hi> µm with the
operator's anchor <a0> µm, not run`, followed by the line of the fields far from every
locked frame when there are some. A dry run anchors no well, so it cannot say which one
would not be scanned.

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
  the first field of the attempt that took it (after the plate check, when the attempt
  made one: the check's time is the check's `elapsed_s`); and, only for a field that had
  the focus on
  the cells, `cell_focus`: its `source` (`swept`, or `well` for a field that reused its
  well's offset), its `sweep` (the centre, the Z readbacks, the scores, the outcome and
  the refined plane), `dz_um` (the frame's Z command minus the centre), `frame_score`
  and `duration_s`. A frame without the step has no `cell_focus` key at all, so a scan
  run with `cell_focus=off` is written exactly as before and still reads in an older
  smc. Every frame also records **where it is in its well**: `well_dx_um` and
  `well_dy_um`, the readback minus the well's centre, in **stage axes** and µm. `(0, 0)`
  is the well's centre and a positive `well_dx_um` is the stage's +X side of it. They are
  **not turned into the plate's axes**: on a plate turned by 13°, a frame 100 µm along a
  row of wells reads `(97.4, 22.5)`, which is what the stage did (the plate's own
  reading of the same two numbers is `WellGeometry.to_plate_frame`). They come from the
  readback, not the target, so a frame that landed 3 µm right of its field reads 3 µm
  more. A frame of a scan written before they existed has neither key, which is *not
  recorded* and never zero.
- A scan that records them has two keys that **an older smc refuses** (it refuses a key it
  does not know rather than drop it): upgrade the smc that reads it. A scan written
  before them still reads, and is written back without them.
- `failed`: `well`, `index`, `name`, the target `x_um`, `y_um`, and `error` (a field
  that landed off target says `StageOffTarget: …`).
- `objective`, `pixel_size_um`, `matrix_um_per_px`, `image_shape_px`: the plan's,
  copied so that the cells stage needs only `scan`; `exposure_ms` as read at the start.
- `plate_check`: the check of the plate made before this attempt's first field
  ([above](#the-plate-check)): `checked_utc`, the wells and their distances, the bound,
  the turn and the slide. A scan that made no check has no `plate_check` key at all, so
  it is written exactly as before and still reads in an older smc; one that made it has a
  key **an older smc refuses**.
- `assumed`. A scan of a plan whose fields a selection chose (`scan-plan`'s
  `fields_from`, [`scan-plan`](scan-plan.md#the-fields-a-selection-chose-at-another-objective))
  ends with `the plan's <n> fields in <k> wells were chosen by selector '<s>' in
  <run_doc>: the rest of each well was not scanned`: the frames are not the wells, and
  `cells`, `density` and a later selector read them as they are. Nothing else changes:
  the scan visits the plan's fields.

Each TIFF is the frame in the camera's own dtype, with the frame's record as its JSON
metadata, so the file alone says where the stage was and where that is in its well
(`tifffile`'s `shaped_metadata`). `load_frame(run_dir, frame)` reads one back and
refuses a path that leaves the run folder.

### What the offsets rest on

The centre of a well is the **plan's own**: `scan-plan`'s plate, in the frame of the
plan's objective (the plate moved between objectives included), computed once per well
before anything moves. The scan reads `scan-plan` and no other record, so it says what
that plate was and not how the plate was found (a seed, a lattice fit: that is the
plate record's to say). Written to `assumed` after every other line, when frames are
written (never in a dry run):

- **The plate was not moved to the objective's frame** (an older plan, an objective
  that is not known, a plate that records no objective, or an offset between the two
  that was not measured): `the frames' offsets from their well's centre (well_dx_um,
  well_dy_um) are against the plate as the plan holds it, registered with <objective>
  and not moved to the frame of <objective>: they carry any difference between the two
  (scan-plan's assumed says why)`. An objective that is not known reads `an unknown
  objective`, or `the objective in use`.
- **The plate was moved** by the profile's offsets between objectives: `the frames'
  offsets from their well's centre (well_dx_um, well_dy_um) are against the plate moved
  from <objective>'s frame to <objective>'s by (31.5, -22.5) µm, the profile's offsets
  between objectives` (the shift to one decimal).
- **The plate was registered with the objective in use**, or the stand has no turret:
  no line, the offsets are against the registration itself.

After a `--resume` across an older smc, the frames of the checkpoint carry no offsets
and the scan **does not fill them in** (their TIFFs hold what was written, and a record
that disagreed with its file would be worse than a gap): `<n> of the <m> frames kept
from the checkpoint were written before a frame's offset from its well's centre was
recorded: they carry no well_dx_um or well_dy_um`.

## Not verified here

**The focus from the map** (m3 §21.12.7, for the next visit at the Ti2-E, on a plate with
cells):

1. the 10× scan of two wells with PFS locked, as in session 6; copy the run document
   (`run-10x.json`);
2. at the 40×, PFS off, condenser N2, focus by eye on cells near the first well's centre,
   then the command of [the workflow](#the-focus-from-the-10-lock-map-cell_focusmap),
   with `map_check_every=25`, over one whole well (1544 fields, about 62 checks);
3. read `assumed`: the time a field and the anchors' (against D16's 0.4–0.75 s and 26–68
   s), and the check: the fraction of the checked fields whose peak lies within 1 µm and 2
   µm of the map's Z. The same records give what a fixed Z would have given (each check's
   peak against the well's anchor plane at its centre), computed offline;
4. look at the frames of a few checked fields whose peak lay 1, 2 and 3 µm from the map's
   Z: from which distance a cell is no longer readable at the 40× sets what "sharp" means
   here;
5. focus by eye in the second well and compare with its anchor: whether the 10× → 40×
   difference holds from one well to the next.

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
read, so a PC that has them is refused too). For the focus on the cells (m3 §21.10):
the sweep at 40× on cells with PFS locked, against the by-eye focus; PFS locking again
at the next field after the sweep, and how far Z moves in it; the time per plane and per
field against the `assumed` line; whether a Z readback lands within a quarter step of its
command; and the defaults of the score, the range and the step, which session 3 chooses
from the stacks of [`focus-on-cells`](focus-on-cells.md). For a frame's place in its
well (#430): at the Ti2-E, a 1 × 1 scan whose first frame was reached by `goto-well`
to a well's centre reads `(0, 0)` within the stage's readback; and a 40× scan with the
objective offsets applied reads the offsets that the by-eye position agrees with. For
PFS across walls (#416): the switch-off and the relock per well at the Ti2-E, a 10× then
a 40× scan of two wells with PFS locked in the first, Z read after each well's first
lock; and whether a plane moves between wells at the 10×. For the plate check (#542,
m3 §27.9): its repeatability with the plate untouched, at the 10× and the 40×; a lifted
plate refused; its time per attempt against the 20 s above.
