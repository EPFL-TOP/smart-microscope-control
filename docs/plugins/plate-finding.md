# `plate-finding`

Finds the plate by looking at it: the stage visits a coarse grid of points over where
the plate should be, the camera takes one frame at each, and the plugin fits which
plate type it is and where A1 sits and how the plate is turned. The result fills the
run document's `plate` slot, the one [`plate-registration`](plate-registration.md)
fills, so `scan-plan` and everything after it read it unchanged. The design is
[`m3-plugins.md`](../design/m3-plugins.md) §6 (the port of `nikon-control`'s
`plate_find.py`); this page is how to use it.

It **moves the stage** (hundreds of absolute moves over the whole plate) and it
**switches the autofocus off**. Read the two sections on those before the first run.

## What it measures, and how well

One number per frame: the mean of the pixels (or their standard deviation, see
`statistic`). Light through a well and through the plastic between wells differ by far
more than focus does, so the objective in place is the one used and nothing is
focused or changed. The fit is the correlation between those numbers and "this point
is inside a well", for each candidate plate, so the type is identified by the same
computation that places it: the plate that explains the numbers best wins, and the gap
to the runner-up says whether it was a decision or a coin toss. Wells that read
darker than the plastic work as well as brighter ones (the sign is in the result).

On **synthetic** plates (eight types, rotations from -2.3° to +1.7°) the type was
identified every time and the well centres were located to 0.3 to 0.7 mm on a 96-well
plate, about a fifth of a well radius. On the demo devices, a synthetic 12-well plate
turned by 1.2° and surveyed 0.7 mm off its centre was identified with a score of 0.88
against 0.60 for the runner-up, and the worst well centre was 0.23 of a well radius
off. That is enough to place a scan grid, and coarser than the 50 µm or so a wall
touch gives: the natural use is automatic first, then one manual refinement with
`plate-registration` if it matters.

**Not measured:** whether a frame's mean separates a well from plastic at 40×, or
through a black-walled plate, on a real stand. Nothing here was run on a microscope.

## Running it

Put the plate on the stage with its centre about under the objective, with the objective
you will scan with. The survey is centred on where the stage is, unless you give the
centre.

```text
smc run plate-finding -p <profile> --run-doc run.json
smc run plate-finding -p <profile> --run-doc run.json --param plate=384-well
smc run plate-finding -p <profile> --run-doc run.json --param centre_x_um=12000 --param centre_y_um=-8000
smc run plate-finding -p <profile> --run-doc run.json --resume
```

A dry run is refused (`a dry run cannot survey the plate; run it without --dry-run`):
a survey of frames that were never taken has nothing to fit.

| Parameter | Default | Meaning |
|---|---|---|
| `plate` | `96-well` | The plate the survey is **sized for**: it reaches one pitch past the outer wells of this plate. A `useq` plate name ending in `-well`. |
| `plates` | all `-well` plates | The plates fitted against, JSON (`--param 'plates=["96-well","384-well"]'`). Say it when you know which plates it can be: a fit over seven plates takes about 2 s for a 96-well survey, and a close call between two is a note. |
| `centre_x_um`, `centre_y_um` | the stage's position | Where the plate's centre is, both or neither. |
| `step_um` | a third of the plate's pitch | Distance between survey points. A half was measured failing on three of eight plate types. At most 25 000 points are made. |
| `margin_pitches` | 1 | How far past the outer wells the survey reaches, 0 to 5. Under 1, the plugin refuses (see below). |
| `jitter` | 0.5 | Random nudge of each point, as a fraction of the step, 0 to 1. A regular raster beats against the plate's lattice. |
| `seed` | 0 | Seeds the nudge: the same seed gives the same survey. |
| `statistic` | `mean` | `mean` or `std` of each frame. Use `std` where wells and plastic differ by texture and not by brightness. |

The survey has, at the default step and one pitch of margin, 130 points on a 6-well
plate, 208 on 12-well, 352 on 24-well, 616 on 48-well, 1120 on 96-well, 3952 on
384-well and 14 800 on 1536-well. The stage walks the rows in snake order, so it does
not cross the plate between two rows. On the demo stage, at a raised speed, a point
takes 14 ms; how long a survey takes on a Ti2-E is **not measured**.

## What is checked before anything moves

In this order; the first that fails ends the run with nothing moved:

1. A dry run.
2. The checkpoint, with `--resume`: it must be this survey's (see *Resuming*).
3. The survey's size: more than 25 000 points is refused
   (`the survey would take 9477198001 points, more than 25000: give a larger step_um
   or a smaller margin_pitches`).
4. **The stage's soft limits.** Only the points inside the limits are visited. What is
   left must still reach one pitch past the outer wells of `plate`: **the margin is
   evidence, not padding.** Without samples beyond the last wells, nothing contradicts
   a lattice that slides along itself by a whole well, and the fit can score 0.99 while
   A1 is a well off. A survey that cannot show that is refused, not reported:

   ```text
   plate-finding: the survey spans 98x58 mm; this plate needs 108x72 mm — one pitch
   of margin past the outer wells — or which well is A1 cannot be pinned down; the
   stage's soft limits x [-49000, 49000], y [-29000, 29000] µm cut the survey: put
   the plate nearer the middle of the travel, or change [safety] if the limits are wrong
   ```

   (That is a 96-well survey centred on the origin of the Ti2-E's limits: 633 of its
   1120 points are inside.) The line about the limits is there only when the limits
   removed points. When they removed some but not enough to matter, the result says so
   in `assumed` (`3 of 208 survey points are outside the stage's soft limits and were
   not sampled`).
5. Then the autofocus is switched off, if it is on (below), and the survey starts.

### The Ti2-E's soft limits refuse every plate as profiled

`profiles/nikon-ti2.toml` has `xy_soft_limits_um = [[-49000, 49000], [-29000, 29000]]`:
98 × 58 mm of travel. The survey needs the wells' span plus one pitch: 108 × 72 mm for a
96- or 384-well plate and 114 × 76 mm for a 24-well plate. So, **as profiled, every
plate is refused before anything moves.** That is the intended answer, not a bug: the
limits are the owner's (#198), and the plugin never widens them. Either put the
plate's centre where the travel is widest and accept the refusal, or widen
`[safety] xy_soft_limits_um` for this stand once the true travel is known. These
numbers are computed from the profile and were not measured on the stand.

### Leaving the plate

At one pitch of margin a 96-well survey (117 × 81 mm, plus ±1.5 mm of jitter) stays
inside the plate's footprint of 127.8 × 85.5 mm. A 6-well survey (156 × 117 mm) and any
larger margin do **not**: the stage then moves the objective over the plate's holder
and past it. **Check the clearance first**, above all at 40×, where the working
distance is short.

## The autofocus is switched off, and stays off

The autofocus cannot lock over the plastic between wells, and a survey that waited for
a lock at every point would take longer than the scan it replaces. So when the
autofocus is **engaged**, the plugin switches it off before the first move (a safe
call; it moves nothing), logs a WARNING, and adds to the result's `assumed`:

```text
the autofocus was switched off for the survey; engage it again before the scan
```

It **never engages it again**, ever: engaging drives Z to the plane of whatever offset
is stored (#205). The scan, which uses the autofocus only as you left it, then finds it
off. Lock PFS on the sample, at the first well of the scan, before you run `scan`.
Nothing here moves Z or turns the turret, and nothing moves the stage afterwards: it
stays at the last survey point.

## During the survey

- **Stopping.** Ctrl-C ends the move under way and the run is `stopped`. A halt of the
  stand (the stop button of the layer) ends the survey and the run is `failed`; it is
  never counted as a point that failed (FM-64).
- **A point that fails is skipped and counted**: a move that times out, a stage that
  refuses a target. More than a tenth of the points failing (in whole numbers: 19 of
  180 is too many, 18 of 180 is not) ends the survey and keeps what was sampled:
  `21 of 208 survey points could not be sampled, the last at (…) µm: DeviceTimeoutError:
  …; check the stage, then run plate-finding --resume`. A frame whose mean is not a
  finite number is a skipped point too.
- A survey that ends with some points skipped says `<k> of <N> survey points could not
  be sampled and were skipped` in `assumed`.

## The result

A `PlateResult` with `method = "lattice-fit"`:

| Field | Meaning |
|---|---|
| `plate`, `a1_center_xy_um`, `rotation_deg` | The winner: which plate, where A1's centre is, how the plate is turned. |
| `residual_um`, `scale_error` | `None`: a lattice fit measured no reference well. |
| `n_refs`, `refs` | `0`, empty: nobody centred a well. |
| `fit.score` | 0 to 1: the absolute correlation with "inside a well". Under 0.25, or with 3 or fewer samples in wells, there is no result. |
| `fit.polarity` | `1`: wells read brighter than the plastic; `-1`: darker. |
| `fit.runner_up`, `fit.margin` | The second plate and the winner's score minus its score; both `None` when one plate was fitted. A margin under 0.08 is a note: `96-well (0.80) barely beats 384-well (0.79) — sample more finely, or say which plate it is`. |
| `fit.n_samples`, `fit.n_inside` | Points sampled, and how many the winner puts inside a well. |
| `fit.notes` | What is worth saying about how clear the win was. |

It fails, with the reason, in three cases: no plate scores 0.25 (`best score 0.12 is too
low to trust — the survey may not have crossed enough wells, or brightness does not
separate wells from plastic on this plate`), the winner is a larger plate than the
survey reaches (`6-well fits best, but the survey spans … : run plate-finding with
plate=6-well`), or too many points failed.

## Resuming

The survey is the checkpoint: the centre, how many points were handled, and what was
sampled, saved every 50 points and at the end. `--resume` continues from there, with
the same parameters (the runner refuses otherwise) and from the **first run's centre**,
not from where the stage stands now. A checkpoint that does not fit the survey (more
points handled than it has, counts that do not add up) is refused with
`run plate-finding without --resume`.

A fit that failed keeps its survey, and `--resume` fits it again **without moving the
stage or touching the autofocus**. With the same parameters it gives the same refusal;
what changes the outcome is another survey, with a larger `margin_pitches`, another
`plate` or another `statistic`.

## Not verified here

Nothing in this page was run on a microscope. For the first hardware session (#198):

- a survey at any stand, with the camera really looking at the plate;
- the Ti2-E's soft limits, which refuse every plate as profiled (computed from the
  profile, not measured), and what the true travel is;
- how long a survey takes on the Ti2-E, whose stage is not the demo's;
- whether a frame's mean separates well from plastic at 40×, and through a black-walled
  plate;
- the clearance of the objective over the plate's holder where the survey leaves the
  plate's footprint;
- numpy 1.26 on Python 3.10, which CI runs and the development machine did not.
