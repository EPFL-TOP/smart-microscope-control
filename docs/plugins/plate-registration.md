# `plate-registration` and `goto-well`

Where the plate is on the stage, as three numbers: A1's centre and the plate's
rotation. The plate's own definition (`useq`: well spacing, well size) gives every
other well, so a registration needs only a few wells you centred, or a few walls
you touched. The result fills the run document's `plate` slot, which `scan-plan`
and everything after it read. The design is
[`m3-plugins.md`](../design/m3-plugins.md) §5 (the port of `nikon-control`'s
`plate.py` and its guided loop); this page is how to use it.

Two plugins make the loop, and each takes its inputs as **parameters**, so nothing
survives between two runs that the run document does not show:

| Plugin | Does | Moves the stage? |
|---|---|---|
| `plate-registration` | Fits the plate from `refs` (wells you centred) and `walls` (wall touches) and fills `plate`. | No. A dry run is the same computation. |
| `goto-well` | Drives to a well's centre or to the point on one of its walls, through the latest `plate`. | Yes: **one absolute XY move**. |

## The guided loop: never hunt for a well

Finding the first well is the only hard part, and it is done once. The well need
not be visible: put any well roughly under the objective, within a couple of
millimetres, and say which one it is.

```text
smc stage get -p <profile>
smc run plate-registration -p <profile> --run-doc run.json --param 'refs=[{"well": "A1", "x_um": 1000, "y_um": 2000}]'
smc run goto-well -p <profile> --run-doc run.json --param well=H12
smc stage get -p <profile>
smc run plate-registration -p <profile> --run-doc run.json --param 'refs=[ ...every reference so far... ]'
```

1. **Seed.** `smc stage get` prints the position (`XY (1000.00, 2000.00) µm`); give it
   as a reference. One reference is a *seed*: A1 is where it implies and the rotation
   is **assumed 0°**, which the result says in `assumed`. It is enough to drive to
   every other well to within a few millimetres.
2. **Go to the next well.** `goto-well` drives there through the latest `plate`. Jog
   until the well is where you want it (`smc stage jog`), and read `smc stage get`.
3. **Register again with every reference so far.** Each run replaces `plate`, and the
   run document's `params` keep every reference that made it: that is the audit
   trail. The fit replaces the seed, so the next well is found more accurately than
   the last, and by the third reference `goto-well` lands essentially on target.

Use **A1 and the two far corners** (`A1`, `A12`, `H1` on a 96-well plate; `A1`, `A24`,
`P1` on a 384-well one): adjacent wells barely constrain the rotation, so the extra
travel buys real accuracy. Two references are enough to fit it, and a third makes
the residual mean something.

The plate page ([`m3-plugins.md`](../design/m3-plugins.md) §15) collects the same
references by clicks and calls the same plugins with the same parameters. In the
words of the issue's steps, *seed* is `plate-registration` with one reference,
*goto-well* and *goto-wall* are `goto-well` (with `side=…`), *capture-edge* is
`smc stage get` followed by a wall touch in `walls`, and *calibrate* is
`plate-registration` with every reference and wall.

### A run on the demo stand

```text
$ smc run plate-registration -p demo --run-doc run.json --param 'refs=[{"well": "A1", "x_um": 1000, "y_um": 2000}]'
✓ plate-registration done in 0.0 s — run.json, stage plate-registration
{
  "assumed": ["one reference well (A1): rotation 0° assumed, not measured"],
  "plate": "96-well",
  "a1_center_xy_um": [1000.0, 2000.0],
  "rotation_deg": 0.0,
  "method": "seed",
  "residual_um": null,
  "scale_error": null,
  "n_refs": 1,
  "refs": [{"well": "A1", "x_um": 1000.0, "y_um": 2000.0}],
  "fit": null
}

$ smc run goto-well -p demo --run-doc run.json --param well=B2 --param side=left
✓ goto-well done in 1.0 s — run.json, stage goto-well
{
  "assumed": ["plate: one reference well (A1): rotation 0° assumed, not measured"],
  "well": "B2",
  "side": "left",
  "target_x_um": 6800.0,
  "target_y_um": -7000.0,
  "x_um": 6799.995,
  "y_um": -7000.005
}
```

(Abridged: the real output has every list on separate lines.) The demo stage rounds a
command to 0.01 µm, so the readback is not the target; `goto-well` reports the
readback and keeps the target beside it.

## `plate-registration`

```text
smc run plate-registration -p <profile> --run-doc run.json --param plate=384-well --param 'refs=[...]' --param 'walls=[...]'
```

| Parameter | Default | Meaning |
|---|---|---|
| `plate` | `96-well` | A `useq` plate name (`6-well` to `1536-well`); the name is exact, `96-WELL` is refused. |
| `refs` | none | Wells centred under the objective, JSON: `[{"well": "A1", "x_um": 1000, "y_um": 2000}]`. A well name is as you type it: `a1`, `A 1` and `A01` are `A1`. |
| `walls` | none | Wall touches, JSON: `[{"well": "B2", "side": "left", "x_um": 6800, "y_um": 0}]`. See [Wall touches](#wall-touches). |
| `max_rotation_deg` | 10 | The largest rotation of the plate that the fit accepts (above 0, up to 45): see [What it refuses](#what-it-refuses). |

Give `refs`, `walls` or both: a well given by walls becomes a reference like any
other, after the ones in `refs`. A well may be given one way only. The shell's
quoting of JSON is the one of [`smc run`](README.md#running-a-plugin-smc-run); on
PowerShell 5.1 or `cmd.exe` escape each inner quote with a backslash.

A fit of three references of a hand-centred plate, 100 µm out on A12 (abridged):

```text
$ smc run plate-registration -p demo --run-doc run.json --param 'refs=[{"well": "A1", "x_um": 1000, "y_um": 2000}, {"well": "A12", "x_um": 100000, "y_um": 2100}, {"well": "H1", "x_um": 1000, "y_um": -61000}]'
✓ plate-registration done in 0.0 s — run.json, stage plate-registration
{
  "assumed": [],
  "plate": "96-well",
  "a1_center_xy_um": [984.91, 2009.61],
  "rotation_deg": 0.0412,
  "method": "references",
  "residual_um": 31.69,
  "scale_error": 0.00046,
  "n_refs": 3,
  ...
}
```

`residual_um` is the largest distance between a reference and where the fit puts
that well; a few tens of µm is normal for hand centring. `scale_error` is the
measured spacing over the plate definition's, minus 1, on the longest baseline. Both
are `null` for a seed: nothing measured them.

A few hundred µm of registration error is **not** a problem for a scan: the error
shifts the scan grid and does not compound, and a well has margin. Rotation is the
part worth care, because it acts over the whole plate, which is why the references
should be far apart.

### Wall touches

At 40x a well is not visible (the field is about 333 µm, a 96-well well 6 400 µm),
so a centre is never judged by eye. Touch a wall instead: drive until the wall sits
on the crosshair, read `smc stage get`, and give it as a touch.

- `side` is `left`, `right`, `top` or `bottom`. `top` is **+Y, towards row A**.
  `left` and `right` use only the touch's `x_um`, and `top` and `bottom` only its
  `y_um`: the other coordinate is never read.
- A well needs one `left` or `right` and one `top` or `bottom`. An **opposite
  pair** gives the centre as the midpoint and measures the well's size, a free check
  that you touched the right wall. **One wall** gives the centre only with the
  plate's nominal half-size, and the result says so in `assumed` (`B2: the Y centre
  is one wall plus the nominal half-size, 3200 µm`).
- `goto-well --param side=left` drives to the point on that wall, along the plate's
  own axes, so it follows the fitted rotation. The wall is then within a field or
  two. On a turned plate a touch along a stage axis errs by `r (1 − cos θ)` for one
  wall (8 µm at 4°, 49 µm at 10°, for a 96-well well, computed); an opposite pair of
  a round well cancels it.
- The nominal sizes are the plate's, width for X and height for Y.

### What it refuses

A refusal ends the stage `failed`, with no result: **no unusable `plate` fills the
slot**, and a stage that reads it stops with the reason.

| It says | Cause, and what to do |
|---|---|
| `no reference: give refs (a centred well) or walls (wall touches)` | Neither was given. |
| `A1 is given both as a reference and by walls: give it one way` | The same well in `refs` and in `walls`. |
| `well A1 given twice` | A well twice in `refs` (`A1` and `a1` are one). |
| `well 'Z99' is not on the 96-well plate` | A name the plate does not have. |
| `the measured well spacing is off the 384-well definition by +100.0%: wrong plate type, or a misidentified well` | The measured distance between the references is not the plate's, by more than 2 %: check the plate type and the well names. Without this check the fit would succeed and put every position further off across the plate. |
| `the fit turns the plate by 90.0°, beyond ±10°: a reference well is misidentified` | The fitted rotation is past `max_rotation_deg`. **Two references the same distance apart** (A1 and A2, or A1 and B1) fit a turn of 90° with no residual, so neither the scale check nor the residual can see a misnamed one: a bound on the rotation does. Raise `max_rotation_deg` only for a plate that really is turned. |
| `H1 is 348 µm from where the others put it: centre it again` | A reference is further than 300 µm from the fit of the others: centre that well again. |
| `A1's left and right walls are 9200 µm apart, a 96-well well is 6400 µm (+43.8%): a touch of another well, or off the wall` | An opposite pair whose size is off the plate's by more than 10 % (a swapped pair has a negative size): touch both walls again. |
| `A1: the left wall is given twice`, `a touch of B2 was given for A1`, `no X wall for A1: give left and/or right` | A mistake in `walls`; the text says which. |

## `goto-well`

```text
smc run goto-well -p <profile> --run-doc run.json --param well=H12
smc run goto-well -p <profile> --run-doc run.json --param well=B2 --param side=top
```

| Parameter | Default | Meaning |
|---|---|---|
| `well` | (required) | The well to go to, as `useq` names it (`H12`; `h12` and `H012` are read as `H12`). |
| `side` | `centre` | `centre`, or the point on a wall: `left`, `right`, `top`, `bottom`. |

It sends **one absolute move** and nothing else, and the layer's guards are the
guards: a target outside the profile's XY soft limits is refused and the stage does
not move (a soft limit cannot be forced), and a stop or a fault ends the stage
`failed`. A well the plate does not have is refused before any move. The result keeps
the target (`target_x_um`, `target_y_um`) beside where the stage says it arrived
(`x_um`, `y_um`, the readback after the move), so a stage that stopped short shows.

**On the Ti2-E the soft limits are ±49 000 × ±29 000 µm**, a span of 98 × 58 mm,
while the corner wells of a 96-well plate are 99 × 63 mm apart. Wherever the plate
sits, some corner `goto-well`s will be refused by the layer. This was computed, not
measured at the stand (#198). A refused corner is the layer's soft limit and not a
fault of the registration: a well nearer the middle serves as a reference as well,
at a little less accuracy in the rotation.

### Dry runs

`goto-well` in a dry run moves nothing, and its readback is the target: the result
says so in its last `assumed` line (`dry run: the stage did not move; x_um and y_um
are the target`). `plate-registration` touches no device, but a dry run records
`dry_run: true` like any other, so **a `plate` from a dry run cannot drive a real
`goto-well`**: it is refused with `plate-registration was a dry run; run it without
--dry-run`. A dry run may read a real `plate`.

The `plate` slot says what the registration assumed, and `goto-well` carries it into
its own result, each line prefixed with "plate: ".

## What is not here

- **No `plate.json`.** `nikon-control` kept the references in a file; here they are
  parameters, and the run document keeps them. A plate registered at the stand is
  reused by passing the same `refs` again.
- **Plate finding.** Finding the plate by itself from images is a separate plugin
  (#20), which fills the same `plate` slot.
- **Not verified at a stand.** No `goto-well` has been run on a real stage, and the
  Ti2-E's soft limits against a 96-well plate are computed from the profile, not
  measured (#198). Parcentricity of the objectives is a constant translation that
  lands in A1 as a fixed bias of tens of µm, far inside the tolerance; it matters
  only for hitting one cell across a change of objective.
