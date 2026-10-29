# M5 design — the web UI's shell

- **Status**: design for issue #28 (§1 to §4, 2026-10-29), as built by #270 and
  settled by #272 (§2.1, §2.4, §3: the stale card, the card's bounds,
  `card_text`'s signature); the directions of #262, #263 and #264 (§5 to §7),
  which their plans fix.
- **Owner**: the design session. **Executors**: `/develop` sessions, one per issue.
- **Rule**: as in M1 to M3, this document is the contract between issues.
  Names, module paths and signatures below are fixed; an executor who needs to
  change one stops and comments on its issue instead. M3 §15 (the plate page)
  holds everywhere this document does not say otherwise.

Read first: [ADR-0006](../adr/0006-user-interface.md) (the five principles and
the shell), M3 §15 (the plate page, which this milestone grows into the shell),
M1 §13 (the emergency stop).

## 0. Scope

ADR-0006 was accepted with Panel and without its spike (#35 not run). The UI
started in M3 as the plate page (#197): a status strip with STOP, the tools in
workflow order with forms generated from `Params`, the run document as a
checklist, and a plate map in stage µm. That page **is** the shell's first
version: M5 grows it, region by region, and never builds a second page.

What the shell lacks, against ADR-0006's picture, and where it is built:

| Region of ADR-0006 | Today (M3 §15) | Built in |
|---|---|---|
| Connect | the command's `--profile` | the stand named on the strip: #28; Connect / Disconnect on the page: #264 |
| Status strip, STOP | done | the stand's name: #28 |
| Live view | none | a **Snap** and its frame: #28; live: #262 |
| Drive | none (a drive click on a well) | #263, with a command for each action |
| Tools rail, tool panel | the tools `STAGES` names | **every** registered plugin, a result card: #28 |
| Run document | a checklist of M3's slots | every slot the document holds: #28 |
| Deployable as a service | no | #264 |

The page keeps its module names (`smc/ui/controller.py`, `smc/ui/plate_page.py`)
and its class names (`PlatePage`, `PlatePageView`) through M5: a rename is churn
while #251 and #252 edit the same code, and is done once, when the shell is
complete. Its texts keep "the plate page".

## 1. The command `smc ui` (#28)

```
smc ui [plate] [-p PROFILE] --run-doc PATH [--port 5006] [--dry-run]
```

One command, `ui_command` in `smc/cli.py`, replaces the `ui` group and its
`plate` command: `smc ui` and `smc ui plate` serve the same page with the same
options, so M3's notes and `docs/ui/plate-page.md` stay true. `plate` is an
optional hidden argument (`typer.Argument(hidden=True)`); any other word exits 2
with `unknown page <word>: smc ui serves one page`. Everything after the
argument is M3 §15.3's command, in its order and with its texts and exit codes.
`smc ui` with no option exits 2 for the missing `--run-doc`, as `smc ui plate`
does today (the group's `no_args_is_help` goes).

Measured 2026-10-29 with Typer 0.27.2 (`CliRunner`): `ui -p x --run-doc r.json`,
`ui plate -p x --run-doc r.json --port 5007` and `ui -p x plate` parse to the
same options; `ui other` reaches the function, which refuses it.

## 2. The controller (#28, `smc/ui/controller.py` and `smc/ui/frames.py`)

### 2.1 One activity at a time

The controller already runs one plugin at a time on one daemon worker, under
`_current_run` (M3 §15.3). In M5 every action that **reaches the stand through
a device call that may last** — a plugin run, a snap (#28), live (#262), a drive
action (#263) — is such an activity: it starts on the worker, holds
`_current_run` under its name, and a second one is refused with the existing
`RuntimeError("<name> is running")`. `stop()`, `resume()`, `close()` and
`join()` treat every activity as they treat a run. The activity names are
`snap`, `live` and the drive actions' (#263); they are texts on the strip, not
plugin names, and a plugin of the same name changes nothing.

`run()` and `snap()` share one private helper that holds the lock discipline of
M3 §15.3 (`_lock` guards `_current_run`, `_progress`, `_outcome` and the frame;
never held across a device call; `stop()` takes none, FM-70).

`PlatePage.activities_ended` counts the **activities** that have ended (a run,
a snap, a live session, a drive action), once each, after what the activity
wrote is on disk. It is part of the shared state, and a tab that sees it change
calls `refresh()` (FM-151): it reads the run document and redraws the plate, the
route and the checklist. The review of #270 measured about 0.34 s for that on a
1536-well plate. Measured here on `FakeCore`, 2026-10-29: the first `view()` of
that plate takes 591 ms and a later one 1 ms, and the redraw assigns the wells'
column data again, 11 columns of 1536 rows (122 kB as JSON), which the server
then sends to every browser (the send is not measured: no browser). That is
right for the end of a run and a cost to a snap, and it is wrong for anything
that happens *inside* an activity: **what an activity does while it goes is
never an activity**, so a live frame (§5) never counts.

### 2.2 A snap and its frame

```python
@dataclass(frozen=True, slots=True)
class Frame:
    number: int
    image: npt.NDArray[np.uint8] | npt.NDArray[np.uint16]
    bit_depth: int
    exposure_ms: float
    pixel_size_um: float
    xy_um: tuple[float, float] | None
    taken_utc: datetime
    summary: str
```

In `smc/ui/frames.py` (numpy only, no Panel, no device): the frame as the
camera gave it, never rescaled, made read-only (`image.flags.writeable =
False`), so that no session changes what the others draw. `number` counts the
controller's frames from 1; a view redraws when it changes, never on the tick
alone (FM-145, FM-151). `pixel_size_um` is `state().pixel_size_um` (`0.0` is
unknown, as `smc snap` writes it). `summary` is `smc snap`'s text:
`<h>x<w> <dtype>, exposure <e:g> ms, min <min>, max <max>`.

```python
def display_image(
    image: npt.NDArray[np.uint8] | npt.NDArray[np.uint16],
    *,
    bit_depth: int,
    max_side_px: int = 1024,
) -> npt.NDArray[np.uint8]: ...
```

The frame as 8-bit grey for the screen, in the camera's orientation (row 0 is the
top row): every `stride`-th pixel, `stride = max(1, ceil(max(h, w) /
max_side_px))`; then `lo, hi = np.percentile(that, (0.5, 99.5))` and `(v - lo) *
255 / (hi - lo)` clipped to 0..255. A **flat** frame (`hi <= lo`) is shown
against the camera's range, `v * 255 / (2**bit_depth - 1)`, so a saturated frame
is white and a dark one black, never a stretched black (`bit_depth` outside 1 to
the dtype's bits: the dtype's maximum). A frame that is not 2-D `uint8` or
`uint16` raises `ValueError` (`a <dtype> frame of shape <shape> is not shown`).

Measured 2026-10-29 (Python 3.13.0, numpy of the main venv; and numpy 1.26 on
Python 3.10.10): the demo camera gives 512×512 `uint16` frames of `bit_depth`
16, values 706 to 5846, a snap takes 10.8 ms at its default 10 ms exposure;
`display_image` takes 0.8 ms on it; 1200×1200 gives 600×600 and 2048×512 gives
1024×256; a frame of 65535 everywhere gives 255, of 0 gives 0. `mypy --strict`
types `np.clip(...).astype(np.uint8)` as `Any`: the result is wrapped in
`np.asarray(..., dtype=np.uint8)`.

**`PlatePage` gains** (additive):

```python
class PlatePage:
    def lacks(self, capability: type) -> str | None: ...
    def snap(self) -> None: ...
    def frame(self) -> Frame | None: ...
```

- **`lacks(capability)`** is `this stand has no <Capability.__name__>` when
  `microscope.has` refuses it, else `None`: the one text for a control that the
  stand cannot offer (principle 1). `_blocked` uses it for a tool's `requires`.
- **`snap()`** refuses as `run` does (`RuntimeError`: `<name> is running`, `the
  page is closed`). A stand without a `Camera` is an `outcome`, `snap: this
  stand has no Camera`, and no thread starts. Otherwise the worker reads the
  exposure, snaps, checks the frame as `smc snap` does (2-D `uint8` or `uint16`,
  else the outcome `snap: the camera returned a <dtype> frame of shape <shape>;
  the page shows 2-D uint8 or uint16 frames`), reads `microscope.state()` for
  the position and the pixel size, keeps the `Frame` and sets the outcome
  `snap: <summary>`. Any exception is the outcome `snap: <Type>: <message on one
  line>`, kept as text (FM-98); a snap while halted is `MicroscopeHaltedError`'s
  text. A snap writes nothing: `smc snap --out` is the command that writes a
  frame, and the page's equivalent of its acquisition.
- **`frame()`** is the latest `Frame`, `None` before the first.

### 2.3 Every plugin on the page

`tools()` shows **every** plugin the registry holds: those `STAGES` names in its
order, then the others under a last stage, `More`, by name. A tool's `key` rows
are `KEY_PARAMS[name]` in that order, then the **required** fields (default
`(required)` in `describe_params`) that `KEY_PARAMS` does not name, in
declaration order; `advanced` is the rest. A required field is never folded:
the twentieth plugin is usable without a line in `KEY_PARAMS`. `_blocked` covers
every registered plugin. A plugin that failed to load (`Registry.failures`) is a
`problems` line of `view()`, `plugin <name>: <reason>`, one per failure.

### 2.4 The result card

```python
@dataclass(frozen=True, slots=True)
class ResultCard:
    plugin: str
    status: str
    when_utc: str
    profile: str
    dry_run: bool
    lines: tuple[str, ...]
    assumed: tuple[str, ...]
    error: str | None
    stale: bool = False
```

`PlateView` gains `cards: dict[str, ResultCard]` (last field): one card per tool
that has a record in the run document (`doc.stages[name]`), none otherwise.
`status` is the record's `StageStatus` value; `when_utc` is `finished_utc`, else
`started_utc`, as `YYYY-MM-DD HH:MM:SS`; `profile` and `error` are the record's,
each on one line and cut as below. **`lines` come from the result only when the
status is `done`** (a stopped or failed record's values are not a result,
FM-154): one line per top-level key of `record.result` in its order, `assumed`
excepted — `<key>: <value>` for a string (newlines as spaces, cut to 80
characters and `…`), a bool, an int or `None`, `<key>: <value:g>` for a float,
`<key>: <n> items` for a list and `<key>: <n> entries` for an object; at most 8
lines, then `… <n> more`. `assumed` is the result's `assumed` list as strings (a
bare value is one entry; `()` when it has none), under the same rule as `lines`.
The card is built by `view()`, so it follows the run document, not the tick.

**A stale card says so** (#272). `stale` is `True` when the record's status is
`done` and `RunDocument.is_stale(plugin)`: the tool read an input that has been
produced again since, and the checklist beside it says `<plugin> read its inputs
from an earlier run; run it again`. The card **keeps** its `lines` and
`assumed`, and its header ends with `(stale)` (§3). `view()` passes
`doc.is_stale(name)` to the card's builder, `_card(record, *, stale=False)`,
which applies the status rule, so the rule is in one pure function.

- *A mark, not a card without values.* A stale `done` record's values are a
  result, of inputs that are no longer the document's: they are shown and
  marked. FM-154 hides the values of a record that never was a result (stopped,
  failed). The header already says every other reason the checklist gives for an
  unusable slot (a status that is not `done`, another profile, a dry run); stale
  is the one it did not.
- *Only a `done` record is marked.* `is_stale` ignores the status (measured, on
  `FakeCore`: a `failed` record that read an earlier input is stale), and a card
  that is not `done` has no values to mislead: its status already says `run it
  again`.
- `is_stale` is a pure read of the document that no cycle can make raise (the
  walk keeps its visited names), measured at 0.004 ms: `view()` calls it for
  each card.

**A card is bounded whatever its record holds** (#272). Measured on `FakeCore`,
2026-10-29, as the card is built by #270: 2000 assumptions of 300 characters
give a card of 620,035 characters, an `error` of 100 kB one of 100,038, and an
`assumed` that is one string of 1 MB one of 1,000,038. The bounds:

| What | Bound | When it is cut |
|---|---|---|
| `lines` | at most 8, `_CARD_LINES` | then `… <n> more` as the last line (as today) |
| `assumed` | at most 8, the same constant | then `… <n> more` as its **last entry**, so `card_text` shows `assumed: … 3 more` and needs no new code |
| a key, a string value, `profile` | 80 characters, `_CARD_CHARS` | the text and `…` (as today; `profile` is new) |
| an assumption, `error` | 600 characters, `_CARD_TEXT_CHARS` (new) | the text and `…` |

Each text is made one line first (`_flat`), then cut: `_cut(text, limit=_CARD_CHARS)`
takes the limit as a parameter, and a non-string item of `assumed` is `str()`-ed
before it, since it can print as long as it likes. 600 characters is more than
twice the longest assumption of a built-in plugin (236, the route's unlocked-frame
line for 1536 stops; measured), so no assumption a plugin writes today is cut,
and an error is a diagnosis whose end often says what to do, so it is cut late.
The worst card is about 7,000 characters (a prototype of the format, not the
code: a header with an 80-character profile, 8 lines of 166, 8 assumptions of
610, an error of 607), two orders of magnitude under the cases above. The run
document and the log hold the rest; a card never changes a record.
`profile` is cut so that the bound is the card's, not only two of its fields: a
profile longer than 80 characters then differs from `stand` and is named on
every card, which errs towards naming it.

**The code of this section's stale mark and bounds** (the `stale` field,
`_card`'s parameter, `_CARD_TEXT_CHARS`, the cut of `assumed`, `error` and
`profile`, and the line of `docs/ui/plate-page.md` that says what a card
shows) goes to the next issue that edits `controller.py`: **#262** (§8). Its
plan lists it as steps with the tests of #272's plan.

### 2.5 The strip and the checklist

`PageStatus` gains `stand: str`, its **first** field: the profile's
`microscope.name`, where ADR-0006 puts the stand's name. The checklist keeps
M3's slots in their order and appends every other slot that has a provider in
the document, by name, with `optional=True`: a third-party plugin's stage shows
without a line in `_CHECKLIST`.

## 3. The page (#28, `smc/ui/plate_page.py`)

- **The strip** starts with the stand: `status_line` gives `<stand> | XY ... |
  ...` (the rest as in M3).
- **The centre** is `pn.Tabs(("Plate", <the map's column>), ("Image", <the
  image's column>), dynamic=True)`: the plate map as today, and the image. Only
  the active tab is rendered (`dynamic=True`), so a figure is never laid out
  hidden. The checklist stays under the tabs, on both.
- **The image** is a `figure(match_aspect=True)` with one `image` glyph from an
  `image_source` (`ColumnDataSource` with the columns `image`, `dw`, `dh`),
  coloured by `LinearColorMapper(palette=Greys256, low=0, high=255)`. The data
  is `display_image(frame.image, bit_depth=frame.bit_depth)[::-1]`: Bokeh draws
  an image's first row at the bottom (its documented convention, not measured in
  a browser), so the flip puts the camera's top row on top. `dw` and `dh` are
  the **frame's** width and height in pixels, not the strided image's (FM-136),
  so the axes are camera pixels at any stride. The figure's title is
  `frame.summary`; before a frame, `no frame yet`.
- **Snap** is a button above the image, with a note beside it: `lacks(Camera)`
  when the stand has none (the button disabled), else empty. It is disabled,
  as every Run button, while an activity is going. A click calls `snap()` (an
  exception's message is the page's message, FM-98) and makes *Image* the active
  tab. The frame is drawn by the **tick** that first sees a new `frame().number`
  (within a second); a tick that sees the same number draws nothing.
- **The result card** of a tool is a `pn.pane.Str` under its Run button, never
  Markdown (a result's text is not markup, FM-139), built by a pure
  `card_text(card: ResultCard | None, *, stand: str | None = None) -> str`:
  `""` without a card, else the header `<status> <when_utc> UTC`, then
  ` on <profile>` when it is not `stand`, then ` (dry run)` when it was one,
  then ` (stale)` when `card.stale` (§2.4); then the `lines`, each indented by
  two spaces; then `assumed: <text>` per assumption; then `error: <error>`.
  `stand` is `PageStatus.stand`, keyword-only (the signature #270 applied, see
  its `## Plan deviation` on #28: the card alone cannot say what "this stand's"
  is). With `stand=None` (the view has not yet read the stand: its first
  `refresh()` draws before its first tick, which then draws the cards again) the
  profile is always told: a card never claims to be this stand's.
- **Tools under `More`** are sections like the others, in a last `Accordion`
  card.

Measured 2026-10-29 on a throwaway prototype (not committed), Panel 1.10.0 /
Bokeh 3.10.1 / Python 3.13.0 and Panel 1.9.4 / Bokeh 3.9.2 / Python 3.10.10,
under `warnings.simplefilter("error")`: `pn.Tabs(..., dynamic=True)` of a
`pane.Bokeh` and a `Column`, `Tabs.active = 1`, a `figure.image(image=, x=, y=,
dw=, dh=, source=, color_mapper=LinearColorMapper(palette=Greys256, low=0,
high=255))` with `uint8` data, a new `source.data` and `figure.title.text` all
build with `get_root()` and warn nothing; `button.clicks += 1` calls the
handler. Not measured: a real browser (the tab switch, the image's orientation
and look).

## 4. Tests and failure modes (#28)

Tests named in the plan of #28. **Failure modes**: FM-98, FM-70, FM-117,
FM-125, FM-139, FM-145, FM-146, FM-147, FM-151, FM-154, FM-136, FM-73 (a
non-square frame in every image test), FM-44, FM-43, FM-36, FM-37, FM-113 (a
third-party plugin's `Params` read by `describe_params`).

## 5. Live view (#262) — direction

- `live` is an activity (§2.1): `PlatePage.live(on: bool) -> None` starts a
  worker that snaps until its event is set and keeps each `Frame` (§2.2), as
  fast as the camera allows and at most 10 frames a second. STOP ends it as it
  ends a run (the event, then `microscope.stop()`).
- The page redraws the image on a second periodic callback of its own, faster
  than the 1 s tick, that does nothing but compare `frame().number` (FM-145);
  the status tick stays at 1 s.
- **A live frame is not an ended activity** (#272). `live` is **one** activity
  (§2.1): one `_start_activity("live", …)` and one worker whose loop keeps each
  `Frame` through the step `_snap` uses, and which never calls `snap()` or
  `_start_activity` again. A frame raises `frame().number` and nothing else; the
  activity's end (STOP, `live(False)` or an exception) raises `activities_ended`
  once, as any activity's does. Measured on `FakeCore`, 2026-10-29: five
  successive `snap()` calls raise the count from 0 to 5, so a live built as a
  loop of snaps would refresh every open tab at the frame rate: on a 1536-well
  plate, 122 kB of wells re-sent to each browser up to 10 times a second (§2.1).
  The plan names the test: the count is the same after several frames of a live
  that goes, and one higher after it ends
  (`test_live_frames_do_not_count_as_ended_activities`).
- **To decide in its plan**: how live gives way when the operator starts a tool,
  a snap or a drive action (refused with `live is running`, or live ended first
  and the action started after its worker has joined), and the redraw period.
- Measured here: the demo camera snaps in 10.8 ms (92 frames a second), a frame's
  conversion takes 0.8 ms and a new `source.data` 0.02 ms: nothing on the server
  limits 5 frames a second on the demo. What a browser shows, locally and over
  RDP, is the owner's to measure, and is ADR-0006's revisit criterion for Panel.

## 6. Drive (#263) — direction

- **The command first** (principle 5): `smc stage jog` and `smc z jog` exist;
  the objective, the light and the autofocus get commands in `smc/cli.py`, named
  in #263's plan, through the facade's guards (confirmed turret moves, the jog
  limit, the Z↔autofocus interlock, dry-run).
- **The region**: each control is one capability call on the worker (§2.1),
  refused while another activity is going, disabled with `lacks(...)` when the
  stand lacks the capability. An objective change asks for a confirmation on
  the page (FM-14). Risk: moves-hardware; its plan is written on Opus and its
  review includes `/code-review`.
- **To decide in its plan**: §2.1's rule holds, so each action's end refreshes
  every open tab. A control that repeats while it is held (a jog) is one
  activity per press, or one for the whole hold, as the plan decides, and says
  which.

## 7. The stand's lifetime and the service (#264) — direction

- Today the command opens the stand when it starts and holds it until Ctrl-C. A
  Windows service is never sent Ctrl-C, and a stand held by smc cannot be taken
  by NIS-Elements: **Connect / Disconnect on the page** come with the service.
- The lab's route to a service is NSSM (`nikon-control/docs/windows-deploy.md`).
- **To decide in its plan, with the owner**: whether the page stays on
  `localhost` only (no login, M3 §15.3) when it is a service; whether a vendor
  SDK and the camera driver run in a service's session 0 (measured at a stand
  only).

## 8. Sequencing

1. **#28** first and alone: it fixes the one-activity rule, `Frame` and the
   regions the others fill.
2. Then **#262** and **#263**, one after the other (both edit `controller.py`
   and `plate_page.py`), and **#264** after them. **#262 carries the code of
   §2.4's stale mark and bounds** (#272), being the first of the two to edit
   `controller.py`.
3. **#251** and **#252** edit the same modules: never at the same time as one of
   the above.

### 8.1 Not verified (for the owner)

- The page in a real browser: the *Image* tab, the frame's orientation, a snap
  on a stand's camera (Ti2-E: Prime 95B, 1412×1412 `uint16`, a stride of 2
  on the screen).
- Everything of M3 §16.1 about the page (RDP, Windows, a PC with no internet).
