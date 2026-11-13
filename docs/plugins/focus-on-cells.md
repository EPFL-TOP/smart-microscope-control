# `focus-on-cells`

Sweeps Z a few micrometres around where the stage is, takes a frame at each plane,
scores each frame's sharpness and moves to the sharpest plane. Every frame of the sweep
is kept as a TIFF with its Z, so a run is also a **Z-stack** that
[`scripts/dev/focus_scores.py`](#scoring-the-stacks-offline) scores offline. The design
is [`m3-plugins.md`](../design/m3-plugins.md) §21; this page is how to use it.

PFS locks on the plate's bottom and holds the plane its offset names. The cells sit a
few micrometres above it, by an amount that differs from well to well. This tool finds
that plane at one place; [`scan`](scan.md#the-focus-on-the-cells-cell_focus) runs the
same sweep at every field (or once per well) with `cell_focus`.

## At the stand

1. Put the objective you will image with in place (40× for the cells), move to a field
   with cells, and **lock PFS on the cells** by eye (m2 §12.10.7). With PFS locked, the
   sweep is centred on the plane PFS holds; with PFS off, on the Z where the stage is.
2. Run it:

   ```text
   smc run focus-on-cells -p nikon-ti2 --run-doc run.json
   smc run focus-on-cells -p nikon-ti2 --run-doc run.json --param range_um=5 --param step_um=0.5
   ```

3. Copy the run's `cell-focus/<token>/` folder to `local/` and score it (below).

It never moves XY: it sweeps where the stage is.

| Parameter | Default | Meaning |
|---|---|---|
| `range_um` | 5 | The sweep's half-range, in µm (above 0, at most 50, and at most the stand's Z jog limit). |
| `step_um` | 1 | The distance between two planes, in µm (0.1 to 10, and at most `range_um`). At most 101 planes. |
| `score` | `brenner` | `brenner`, `normalized-variance`, `gradient-energy` or `laplacian-variance`. |
| `relock` | true | Lock PFS again after the sweep, when it was locked at the start. |

**Every default is chosen, not measured.** `brenner` is chosen from the literature; the
range and the step are a guess at a 40×'s depth of field. Session 3's stacks decide them
(m3 §21.10).

## What it does

1. **Checks, before anything moves** (a dry run too). PFS, when the stand has one and it
   is on, must be locked (`the autofocus is on and not locked: lock it on the sample, or
   switch it off to sweep around the Z where it is`); its offset is then read. The
   profile must set `[safety] z_soft_limits_um` (`focus-on-cells moves Z: the profile
   must set [safety] z_soft_limits_um`: unknown limits are not no limits). `range_um`
   must not exceed the Z jog limit (`[safety] max_z_jog_um`, else 100 µm): the sweep's
   moves are absolute, so the layer's jog guard does not see them, and the tool bounds
   their reach by the same number. Every plane must lie inside the soft limits (`the
   focus sweep around 4195.2 µm would reach 4201.2 µm, outside the soft limits …`). On
   Windows, the longest frame path must be shorter than 260 characters while it is
   written, as for [`scan`](scan.md#what-is-checked-before-anything-moves).
2. **Switches PFS off** for the sweep and reads it back off. A Z move under a lock is
   undone by the relock before the frame, and stepping the PFS offset needs the 40×'s
   offset factor, which is not measured (m2 §12.10.9).
3. **Sweeps**: for each plane in increasing Z, a move with the readback compared with
   the command (more than a quarter step away fails the sweep), a frame, its score, and
   the frame written as `cell-focus/<token>/<k>.tif`.
4. **Finds the peak**: the best score's plane, refined through a parabola on its two
   neighbours. Only a sweep with **one clear interior maximum** is taken. A flat sweep
   (an empty field: the scores vary by less than 5 %), a best at an end of the sweep
   (the focus is outside it), or two maxima (brightfield of unstained cells can lose
   contrast exactly at focus and show a maximum on either side) **keep the centre**,
   and the result says which.
5. **Moves** to the refined plane and takes `best.tif` there, or moves back to the
   centre.
6. **Locks PFS again** (`relock=true`): PFS takes Z back to the plane it holds. When the
   lock does not come, `assumed` says `the autofocus did not lock again: it is off`.
   With `relock=false`, `assumed` says `the autofocus was left off, at the best plane`
   (or `at the centre`), and Z stays there.

## What you get

The slot `cell-focus` holds a `CellFocusResult` (`smc.plugins.slots`):

- `folder`: `cell-focus/<token>`; `files`, one TIFF per plane in the sweep's order;
  `best_file`, the frame at the best plane (`null` without a peak).
- `sweep`: the `score`, the `centre_um`, the Z **readbacks** (`z_um`), the `scores`,
  the `outcome` (`peak`, `flat`, `at-edge`, `two-peaks`) and `best_z_um` (set only for a
  peak).
- `autofocus` at the start (`locked`, `off`, `none`), its `pfs_offset` when it was used,
  `relocked` (`null` when it was not used or not relocked), the `objective`, the XY
  readback, and `z_um`, read at the end.

Each TIFF is the frame in the camera's own dtype, with `{"z_um", "z_command_um",
"score", "score_name"}` as its JSON metadata.

## What each refusal and failure means

- **A refusal before anything moves** (above) moves nothing and writes nothing.
- **A failure during the sweep** keeps the frames taken and says where it ended: `the
  focus sweep ended at plane 3 of 11 (SafetyRefusedError: …); the 2 frames taken are kept
  in cell-focus/<token>/; the autofocus was left off; Z was left where it stopped`. A
  failure after the sweep names that step instead of a plane (`the move to the best
  plane`, `the snap at the best plane`, `the move back to the centre`), and a PFS that
  still reads on after its switch-off names `the switch-off of the autofocus`. Nothing
  moves to put anything back: Z stays where it stopped and PFS stays off. Lock PFS
  again yourself before you image.
- **A frame that cannot be written** (a full disk, a share that dropped, a file a viewer
  holds open) ends the run with the OS's reason; the frames before it are kept, and,
  when PFS had been switched off, the text ends `; the autofocus was left off; Z was
  left where it stopped`.
- **A halt or a stop** is not a failure: the run ends where it is. When PFS had been
  switched off, a WARNING says it was left off and Z where it stopped.

There is no checkpoint: a run is one sweep, and `--resume` starts it again.

## Scoring the stacks offline

```text
python scripts/dev/focus_scores.py local/cell-focus/* --csv scores.csv --plot plots
```

Each folder is one stack: the TIFFs whose metadata has `z_um` (`best.tif` is left out).
It prints one row per plane (Z, the four scores) and, per score, the outcome and the
refined Z of the same peak finder the tool uses. `--csv` writes every row with its
folder; `--plot` writes one PNG per stack (each score divided by its maximum, against Z)
when `matplotlib` is installed, and says so when it is not. The script expands a pattern
such as `local/cell-focus/*` itself, since cmd.exe and PowerShell do not. A folder with
fewer than 3 frames is reported and skipped; a TIFF without `z_um` is left out and
counted; a frame Windows refuses to read for a moment is read again, and one that stays
unreadable skips its stack (a curve with a hole in it would mislead). Exit 0 when one
stack was scored and every output asked for was written, else 1.

What to read from it at the stand: which score peaks within 1 µm of the focus you set by
eye, whether the curves show two maxima, and what range and step the sweep needs.

## What it does not do

- It does not change the PFS offset, and neither does the scan with it (m3 §21.2 D1): a
  frame keeps the lock's offset, a reading. The [`route`](route.md) derives, for its
  stops, the offset of the cells' plane from the frame's distance from the lock's plane
  and the profile's factor for the objective (the 40×'s is in the profile since session
  3), or leaves it out and says why (m3 §26); the NIS-Elements export says when an offset
  was derived. Carrying the cells' plane into the offset while scanning is a later
  change.
- It never moves XY, turns the turret, or returns to a start position.
- It does not choose the score or the defaults: session 3 does (§21.10).

## Trying it without a microscope

On the `demo-sample` stand, the frames blur by `|z| / 10` pixels around Z = 0, a
stand-in for defocus: a sweep from 20 µm with `range_um=25` and `step_um=5` finds Z = 0
within a few hundredths of a micrometre. The demo's autofocus is off when it opens.

```text
smc run focus-on-cells -p demo-sample --run-doc run.json --param range_um=25 --param step_um=5
```

## Not verified

Nothing here was run at a microscope. At the Ti2-E, 40× on a plate with cells, PFS locked
on them (m3 §21.10): the sweep on a few wells and `focus_scores.py` on their stacks (which
score peaks at the by-eye focus within 1 µm, whether the curves have two maxima, the
defaults of range and step); the PFS lock held after the relock (`smc autofocus get`) and
how far Z moved in it; the time per plane and per field; whether a Z readback lands within
a quarter step of its command. The 40×'s offset factor at the offsets the route uses, and
that NIS-Elements engages PFS at a derived offset on the cells' plane, are for the route's
own check (m3 §26.9).
