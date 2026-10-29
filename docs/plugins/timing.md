# `timing`

What a stage move, an autofocus lock, a channel switch and a snap cost **on this
stand**, measured, and then the arithmetic on them: how long a position takes, and how
many fit in an interval. A ten-hour movie at a five-minute interval is 120 timepoints;
whether it can also visit 200 positions in three channels depends on numbers nobody
can guess, so the plugin times them. The scan plan's estimate and the route's travel
time read this slot instead of an assumption.

The design is [`m3-plugins.md`](../design/m3-plugins.md) §12; this page is how to use
it. It is a port of `nikon-control`'s `timing.py` and its tests.

## Before you run it

It **moves the stage and switches the autofocus off and on**, so check three things
as for any stage move:

- **The stage travels up to `distances_um[1]` along X from where it stands** (9 mm by
  default), and comes back. The jog guard covers relative moves and not these absolute
  ones: the only check is the soft limits. Park the stage on the plate, **not at the
  edge of its travel**, and check that the objective has clearance. When the soft
  limits put `+X` outside the travel and `-X` inside it, the probe goes `-X`; when
  neither fits, the run is refused before anything moves, with the limits, the start
  and the distance in the text.
- **The autofocus must be on and locked on the sample.** Each lock is timed from
  disengaged: it is switched off and on again `repeats` times, and engaging drives Z
  on the Ti2-E. The plugin never switches on an autofocus that was off (that is the
  operator's call, and where it drives Z to), so a stand that is not locked gets an
  *assumed* refocus (below) and no Z motion from this plugin.
- **A snap is taken at the channel and exposure the stand has**, `repeats` times, and
  with `channels` the stand selects each channel in turn.

## Running it

```text
smc run timing -p <profile> --run-doc run.json
smc -v run timing -p <profile> --run-doc run.json --param repeats=5
smc run timing -p <profile> --run-doc run.json --param 'channels=["DAPI","FITC"]'
```

`-v` goes before `run`: it prints the measured model, one line each.

| Parameter | Default | Meaning |
|---|---|---|
| `repeats` | 3 | Samples per distance, autofocus lock, bare snap and channel (1 to 20). The median of each is kept, so one slow sample does not move it. |
| `distances_um` | `[200, 9000]` | The short and the long probe, in µm along X: a field-to-field hop and a well-to-well move. Each in (0, 50000] and the second longer; two equal distances fit no line. |
| `with_autofocus` | `true` | Time the autofocus lock. `false` touches nothing and assumes the refocus. |
| `channels` | none | Channels to time, by name, at most 32, none empty, none twice. Each must be a channel of this stand: a name it does not have is refused before anything moves, and the text lists the ones it has. |

On the demo (the demo stage moves at 10 mm/s, and its autofocus is off):

```text
$ smc -v run timing -p demo --run-doc run.json --param repeats=1 --param 'distances_um=[100,400]'
timing: 1/3 stage 100 µm
timing: 3/3 snap
INFO stage move   0 ms + 121.7 ms/mm
INFO refocus      400 ms
INFO channel swap not measured
INFO snap         11 ms
INFO assumed (not measured): the stage reports no soft limits: the probe moves were not checked against its travel; refocus 400 ms is assumed: the autofocus was off, and the plugin does not switch it on: engaging drives Z
✓ timing done in 0.1 s — run.json, stage timing
```

(The command line shows the first and the last progress line, as for any plugin; the
demo's travel time is the demo's, 10 mm/s, and its overhead fits as 0.) The steps are `stage <d> µm` for each distance, `autofocus lock` (only when it
is measured), `snap`, and `channel <name>` for each channel. A dry run is refused: the
calls that would be timed are skipped, so their durations are not the stand's. A
stopped timing is measured again; `--resume` re-measures, and nothing half-done is
stored.

## What it measures

- **The stage.** For each distance, `repeats` times: out along X and back, each move
  timed. The timed call is the facade's `move_to_um`, which returns once the stage has
  arrived: the cost a scan pays. A point of the fit is the **median of the distances
  the stage read back** and the median of its `2 · repeats` times, and the line is
  `ms = overhead + rate · mm` through the two points, each clamped at 0. The distance
  is the readback, never the command: a stage that stops short of a probe is not a
  measurement of that distance. A move whose readback differs from the one asked by
  more than `max(5 µm, 2 %)`, or is not a number, fails the run with how far the stage
  went, **after** the stage is back at the start.
- **The refocus.** `repeats` times: the autofocus is switched off, `engage()` is timed.
  The first lock that does not come (`engage()` returns `False`) or is refused ends
  the section: another sample would wait the whole lock timeout again, and a search
  drives Z. Anything else (a halt, an excursion) is not "it did not lock": it ends the
  run.
- **The snap.** `snap_ms` is the median of `repeats` bare snaps.
- **The channels.** With `channels`, each is selected (one switch sample) and snapped
  `repeats` times (`channel_ms[name]` is the median). The channel the stand was on is
  visited **last**, so that every timed switch changes channel: a switch to the channel
  already selected can cost nothing and would pull the median down. When the one channel
  given is the one the stand is on, nothing is selected and no switch is timed.

`samples` counts the timed operations whose medians are in the result: `4 · repeats` for
the stage, `repeats` for the autofocus when it was measured, `repeats` for the bare
snaps, and per channel `repeats` snaps plus its switch samples.

## What it leaves

When the run ends normally the **stage is back at the start**, the **autofocus is as it
was** (a measured run ends on a lock, so it is on and locked again) and the **channel the
stand was on is selected back**. After **a stop, a halt or a fault nothing is put back**: a return, an
engage or a select after the operator's stop would be another motion nobody asked
for, and an engage drives Z. A WARNING says what was left: where the stage is and where
it started, that the autofocus was switched off by this run and is left off, and which
channel the stand is on and which it was found on. A stop before the first move, or
between two steps that left nothing changed, says nothing.

The one exception on a normal end: a lock that did not come leaves the autofocus **off**,
and the result and a WARNING say so (below). Engage it again before the scan.

## The result

The slot `timings` holds a `TimingsResult` (`smc.plugins.slots`):

| Field | Meaning |
|---|---|
| `move_overhead_ms`, `move_ms_per_mm` | The line fitted to the stage's two distances. `move_ms(distance_um)` is `0` for no move and the overhead plus the rate otherwise. |
| `refocus_ms` | The median lock time, **or 400 ms assumed** (see `assumed`), or `null` on a stand that has no autofocus ("not applicable", never 0). |
| `snap_ms` | A bare snap, at the channel and exposure the stand has. |
| `channel_ms` | Per channel given: snaps in that channel. Empty without `channels`. |
| `channel_switch_ms` | The median switch, or `null`: no channel was given, or none was switched to. |
| `samples` | As above. |
| `assumed` | What the result rests on that was not measured. |

`assumed` can hold, each as a line of its own:

- `the stage reports no soft limits: the probe moves were not checked against its travel`.
- `refocus 400 ms is assumed: with_autofocus is false`, `...: the autofocus was off, and
  the plugin does not switch it on: engaging drives Z`, `...: the autofocus was on but
  not locked on the sample`.
- `refocus 400 ms is assumed: <why>; <n> of <repeats> samples locked, median <ms> ms`
  (or `; no sample locked`), where `<why>` is `the autofocus did not lock`, `the
  autofocus refused to engage (<its reason>)` or `something else switched the autofocus
  off during the measurement`. The median of the locks that came may be above 400.
- `the autofocus was switched off for the measurement and did not lock again; engage it
  again before the scan`.
- `no switch was timed: the one channel given is the one the stand was on`.

A refocus of 400 ms is an over-estimate on a stand that locks faster and an
under-estimate where it locks slower: the lines say which samples locked. `scan-plan`
copies every line into its own `assumed`, each starting with `timings:`, so that an
estimate whose refocus was assumed does not read as measured.

## How many positions fit: `plan()`

`smc.plugins.timing.algorithm` is pure: it takes a `TimingsResult` and no stand. There
is no command or page for it in M3 (the scan takes one channel per field and has no
interval), so a program calls it:

```python
doc = RunDocument.load(Path("run.json"))
timings = TimingsResult.model_validate(doc.stages["timing"].result)
budget = plan(timings, ["DAPI", "FITC"], interval_s=300, requested=96, duration_h=10)
print("\n".join(budget.describe()))
```

The model per timepoint is `per position = move + refocus + sum over channels(switch +
expose + read)`, and a request fits when `positions × per position` is under the usable
part of the interval. `headroom` (0.8) is the usable part: a timelapse scheduled to the
whole interval has no slack for a slow refocus or a retry, and drifts later at every
timepoint.

- A channel nobody measured costs the mean of the measured ones, else a bare snap.
- A switch the timings do not hold is 100 ms, assumed. With no channel, a position is one
  bare snap and no switch.
- A stand without autofocus (`refocus_ms` is `null`) adds no refocus.
- `Budget.assumed` is the timings' own `assumed`, plus a line for the switch and for each
  channel whose cost `plan` filled in.
- `plan` refuses (`ValueError`) an interval that is not finite and positive, a headroom
  outside (0, 1] (the port clamped it), a request, duration or move that is negative or
  not finite, and an interval and duration whose counts overflow.

## What is not known yet

Not run on a real stand. At the Ti2-E (#198): the stage's speed and settling, the PFS lock
time, whether `disengage()` then `engage()` at a seen offset is accepted and how far Z
moves, what a channel switch costs, and the stage's soft limits against `distances_um`.
