# `nis-export`

Writes the **route as an XY position list** (a *multipoint* list) that NIS-Elements 6.20
loads, one position per stop, in the route's order. The file is what turns the plate scan
into an acquisition on the Ti2-E: the positions the cells were found at, ready to be imaged
with NIS-Elements' own tools.

It reads the `route` slot and moves nothing: no stand, no frame, so a dry run writes the
same file as any other. The design is [`m3-plugins.md`](../design/m3-plugins.md) §14; this
page is how to use it.

## Running it

```text
smc run nis-export -p <profile> --run-doc run.json
smc run nis-export -p <profile> --run-doc run.json --param out="D:/NIS/plate 1.xml"
smc run nis-export -p <profile> --run-doc run.json --param out="D:/NIS/plate 1.xml" --param overwrite=true
```

| Parameter | Default | Meaning |
|---|---|---|
| `out` | none | The file to write, **ending in `.xml`** (any case). Without it the file is `nis-export/<12 hex>.xml` in the run folder. |
| `overwrite` | `false` | Replace `out` when it exists. Without it an existing file is refused and left untouched. |

The result names the file, the number of positions and how Z is carried (`z_mode`, below):

- **Without `out`** the file is `nis-export/<12 hex>.xml`, relative to the run folder. **Every
  run writes a new file** and an old one is never rewritten, so they pile up: delete the ones
  you no longer need.
- **With `out`** the file is written exactly there, relative to the **current folder** when the
  path is relative, and the result records its **absolute path**, which may name your account
  (an `out` is yours to choose). The folder of `out` must exist: a mistyped one is refused,
  never created. A folder named `x.xml` is refused, with or without `overwrite`.
- The write is all or nothing: the old file or the whole new one. Only when the final rename
  fails (on Windows, a program holds the file, NIS-Elements may) the whole list is kept in a
  temporary file next to it, named `.<name>.<hex>.smc-tmp`, and the reason names it; rename it
  to end in `.xml` to use it, or close the program and run `nis-export` again.
- A stop (Ctrl-C) ends the stage before the write and leaves no file. Nothing is
  checkpointed, so a re-run starts over.

## What the file is

NIS-Elements 6.20 exported a list of ten positions on the Ti2-E on 2026-10-29, and the file
is built from it, byte for byte: **UTF-16 with a byte-order mark, no line break anywhere**, an
`<variant version="1.0">` holding a list with two flags (`bIncludeZ`, `bPFSEnabled`) and one
`PointNNNNN` per position with its name, X, Y, Z, autofocus offset and an empty user-data
field. Doubles carry fifteen decimals, as NIS writes them (a value under 10 µm can come back
up to 1e-15 µm away, far under what a stage resolves). Every point is written checked
(`bChecked` true) and with empty user data, as in the sample.

The sample is kept as `tests/data/nis/nis-elements-6.20-multipoints.xml`, marked `binary` and
excluded from the text hooks of `pre-commit`: do not open it in an editor that saves it.

The name of a position is the route stop's (`<well>_r<k>`, for example `B03_r000`).

## Z: three modes

One flag covers the whole list, so the export carries Z in one way for **all** its positions,
decided by what the scan knew of every stop:

| `z_mode` | When | What the file holds |
|---|---|---|
| `pfs-offset` | every stop has a `z_um` **and** a `pfs_offset` | `bIncludeZ` and `bPFSEnabled` both true: Z and the autofocus offset of each stop |
| `z` | every stop has a `z_um`, at least one has no `pfs_offset` | `bIncludeZ` true, `bPFSEnabled` false: Z without the autofocus; `dPFSOffset` written as 0 |
| `none` | at least one stop has no `z_um` | both false: no Z and no offset, both written as 0; NIS-Elements keeps the focus it has |

A stop has no `z_um` when every frame of its well was `unlocked` (the autofocus had lost the
sample, [`route`](route.md)); one such stop takes the Z of all the others out of the list, and
the result says how many stops lacked it. A field whose flag is false is written as 0 and is
not read back.

## What is assumed

`assumed` always starts with the line that matters most at the stand:

- *`x_um`, `y_um` and `z_um` are Micro-Manager's readings in µm and `pfs_offset` is
  PFSOffset's own unit, written unconverted: that NIS-Elements shares their origin and
  direction is not measured.* There is **no sign flip and no offset** here. If NIS-Elements
  counts X or Y in the other direction, or from another origin, the export sends it to the
  wrong place with no error anywhere, until someone compares one stop with the stage
  (`smc stage get`).
- In `pfs-offset` mode: *`bIncludeZ` and `bPFSEnabled` are both set, which the NIS-Elements
  sample did not show.* The sample had `bPFSEnabled` false.
- In `z` mode: *N of M positions have no `pfs_offset`…*
- In `none` mode: *N of M positions have no `z_um`…*

## When it refuses

The stage ends `failed` (its record has no result) with a reason that says what to do:

- **The route has no positions**, or its stops are not numbered `0, 1, 2…` in order (a record
  edited behind the runner's back). `run route again`.
- **A position the list cannot carry**: a name with a control character, a lone surrogate or
  U+FFFE/U+FFFF, a coordinate that is not a finite number, or more than 100 000 positions.
  `run route again`.
- **`out` does not end in `.xml`.** This is refused with the parameters, before anything runs:
  `out must end with .xml, not 'list.txt'`. `out=` (empty) is refused the same way.
- **`out` cannot be resolved** (a network share that dropped), **is a folder**, or **exists
  and `overwrite` is false.**
- **A file that cannot be written**, with its folder missing, the disk full or the file held
  by another program. The reason names the file and what is kept (above).

## Limits

- **Only `encode` is for use.** The reading side of the format (`decode`) exists for the tests
  and the round trip; nothing reads a NIS-Elements list back into smc.
- **Units and frames are not converted**, in either direction (see *What is assumed*).
- **One flag per list.** A list where only some positions have a Z cannot be written, which
  is why a single stop without one changes the mode of all (*Z: three modes*).

## Not verified here

Nothing in this page was opened in NIS-Elements. For the owner at the Ti2-E (#198):

- **NIS-Elements 6.20 loads an export of each mode.** The sample shows `bIncludeZ` true and
  `bPFSEnabled` false only; the `pfs-offset` mode sets both, which has never been seen.
- **NIS shows the names, and its X, Y and Z for a stop equal `smc stage`'s readings at that
  stop** (`smc stage get`), with the same origin and direction. This is the check that the
  first assumed line is true.
- **With `bPFSEnabled` false and `dPFSOffset` 0** NIS neither complains nor moves the
  autofocus offset.
