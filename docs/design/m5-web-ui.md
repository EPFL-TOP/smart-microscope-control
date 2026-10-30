# M5 design — the web UI's shell

- **Status**: design for issue #28 (§1 to §4, 2026-10-29), as built by #270 and
  settled by #272 (§2.1, §2.4, §3: the stale card, the card's bounds,
  `card_text`'s signature); live fixed by #262's plan (§5, 2026-10-29); drive
  fixed by #263's plan (§6, 2026-10-30); the direction of #264 (§7), which its
  plan fixes.
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
| Drive | none (a drive click on a well) | #263, with a command for each action (§6) |
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
route and the checklist. Measured here on `FakeCore`, 2026-10-29, on a 1536-well
plate: the first `view()` takes 591 ms (the plate's geometry, FM-145's 339 ms,
which `view()` then keeps), a later `view()` 1 ms and a `refresh()` 1 to 2 ms;
what each one then costs is the redraw, which assigns the wells' column data
again, 11 columns of 1536 rows (122 kB as JSON) that the server sends to every
browser (not measured: no browser). That is right for the end of a run and a
small cost to a snap, and it is wrong for anything that happens *inside* an
activity: **what an activity does while it goes is never an activity**, so a
live frame (§5) never counts.

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
- **Every handler is registered through `PlatePageView._handler`** (#277): it
  runs inside `pn.io.unlocked()`, and what it raises is the page's message
  (`that action failed on the page: <Exc>: <text> (see the server's log)`, the
  text cut to 300 characters), never only a traceback on the console. Panel runs
  a handler without the session's document lock when an event flushes a change
  it had queued (the Enter of a text field; a Run click right after a typed
  field), and the page's own Bokeh models are then written outside the lock
  (FM-171). The periodic `tick` and `frame_tick` are Bokeh callbacks, run
  locked, and are not wrapped; a method called directly, as the tests do,
  still raises.

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

## 5. Live view (#262)

- `live` is an activity (§2.1): `PlatePage.live(on: bool) -> None` starts a
  worker that snaps until its event is set and keeps each `Frame` (§2.2), as
  fast as the camera allows and at most 10 frames a second. STOP ends it as it
  ends a run (the event, then `microscope.stop()`).
- The page redraws the image on a second periodic callback of its own, faster
  than the 1 s tick, that does nothing but compare `frame().number` (FM-145);
  the status tick stays at 1 s.

### 5.1 The controller

```python
#: smc/ui/controller.py
LIVE_PERIOD_S = 0.1


class PlatePage:
    def live(self, on: bool) -> None: ...
    @property
    def is_live(self) -> bool: ...
```

- **`live(True)`** refuses as `snap` does (`RuntimeError`: `<name> is running`,
  `the page is closed`; `live is running` when live already goes). A stand
  without a `Camera` is the outcome `live: this stand has no Camera`, and no
  thread starts. Otherwise one `_start_activity("live", self._live)`, whose
  event the page keeps (`_start_activity` returns it).
- **`live(False)`** sets the event of the live it started, and only that one
  (compared by identity, never by the name `live`, so a plugin named `live`
  is not stopped), and returns at once: it never joins and never calls
  `microscope.stop()` (ending a live view is not an emergency; STOP is). It
  does nothing when no live goes, after `close()` included, and never raises.
- **`is_live`** is `True` while the activity going is the live that `live(True)`
  started: what the page's button reads.
- **The worker** takes frames through `_take_frame()`, the step `_snap` uses:
  `_take_frame() -> Frame | str` keeps and returns the `Frame`, or returns
  the failure as text without a prefix (`the camera returned a ... frame of
  shape ...; the page shows 2-D uint8 or uint16 frames`, `<Type>: <message on
  one line>`), so that no exception is kept (FM-98). `_snap` is
  `snap: <summary>` or `snap: <text>`, as in #28. After each frame the worker
  puts `Progress(plugin="live", done=<frames>, total=None, message="frames")`
  on the progress queue, so the run line reads `live: <n> frames` while it goes,
  and waits on its event for what is left of `LIVE_PERIOD_S` since the frame
  began (`max(0.0, ...)`): the cap is a period, so a camera slower than 10
  frames a second runs at its own rate.
- **Its outcome**: ended by its event (`live(False)`, STOP, `close()`): `live
  ended: <n> frames in <s:.1f> s` (the rate the owner reads at a stand); a
  frame that fails ends it: `live ended after <n> frames: <text>`; the last
  good frame stays `frame()`. A live while the stand is halted ends at its
  first frame with the facade's refusal, as a snap does.

### 5.2 How live gives way

**A tool's Run and a Snap are refused while live goes** (`live is running`),
as every activity refuses another (§2.1), and their buttons are disabled; the
operator ends live with one click. Live is not ended for them:

- an action that silently ends what the operator is watching changes the
  screen with no word;
- ending live first means waiting for its worker (up to one exposure, seconds on
  a dim sample) either in a server callback, on the loop that handles STOP
  (FM-62, FM-148), or in a chained start with failure cases of its own;
- the rule costs no code: every button is already disabled while an activity
  goes (#28).

**A drive action is #263's** (§6): moving the stage and Z while watching is what
live is for (focusing, finding a field), so #263 decides how a drive action
reaches the stand while live goes without two threads on the devices (for
example, the live worker runs a queued drive action between two frames).
A drive click on the map (`goto`) stays refused with `live is running`: it is a
run of the `goto-well` plugin, which writes a record (§6.3).

### 5.3 The page

```python
#: smc/ui/plate_page.py
FRAME_TICK_MS = 100


class PlatePageView:
    live_button: pn.widgets.Button

    def on_live(self) -> None: ...
    def frame_tick(self) -> None: ...
```

- **Live** is a button beside Snap. Its label is `Live`, and `End live` while
  `page.is_live`; it is disabled with `lacks(Camera)`'s note on a stand without
  a camera, and while an activity other than this live goes. A click calls
  `page.live(not page.is_live)` (an exception's message is the page's message,
  FM-98); starting it makes *Image* the active tab; then `tick()`.
- **`frame_tick()`** calls `_draw_frame()` and nothing else: no `status()`, no
  `view()`. `build` registers it with `FRAME_TICK_MS` beside the tick
  (`TICK_MS`, unchanged), so a session has two periodic callbacks, both its
  own (FM-146). The tick keeps its `_draw_frame()`, which draws nothing when
  `frame_tick` already drew that number.
- `FRAME_TICK_MS` matches `LIVE_PERIOD_S`: a faster callback would find no new
  frame, a slower one would drop frames the camera took.
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
- How live gives way, and the redraw period, are §5.2 and §5.3.
- Measured here: the demo camera snaps in 10.8 ms (92 frames a second), a frame's
  conversion takes 0.8 ms and a new `source.data` 0.02 ms: nothing on the server
  limits 5 frames a second on the demo. What a browser shows, locally and over
  RDP, is the owner's to measure, and is ADR-0006's revisit criterion for Panel.
- Measured while planning #262 (2026-10-29, Python 3.13.0, the demo stand, a
  throwaway subclass of `PlatePage` with §5.1's loop): `_snap`'s step takes
  10.9 ms, of which `microscope.state()` 0.14 ms; over 2 s, 20 frames at the
  cap; `activities_ended` unchanged while live goes and one higher after
  `live(False)`; a snap while live goes refused with `live is running`; STOP and
  `close()` end it within one period (`close_lines`: the stop note, then `live
  ended: 3 frames in 0.2 s`). Not measured: `state()` on a real stand, read once
  per frame, which lowers the rate by what it costs.

## 6. Drive (#263)

Fixed by #263's plan (2026-10-30). The drive actions are written **once**, in a
module that both the command line and the page call (principle 5: the page's
control *is* the command's call, not a copy of it); the page runs them on its
worker (§2.1) and, while live goes, on the live worker between two frames.

Measured 2026-10-30 (Python 3.13.0, `FakeCore.demo_like()` and Micro-Manager's
demo devices, a throwaway script): the turret's labels are `Nikon 40X Plan Fluor
ELWD`, `Nikon 10X S Fluor`, `Objective-2`, `Nikon 20X Plan Fluor ELWD`,
`Objective-4`, `Objective-5`, at `Nikon 10X S Fluor` after an open; `select`
without `confirm` raises `SafetyRefusedError` whose `how_to_force` is `pass
confirm=True once the stage is clear`, and with it turns and switches an engaged
autofocus off (it stays off); `engage()` returns `True` and locks; a Z jog of 5
µm with the autofocus engaged returns 5.0 and leaves it engaged (the interlock);
an XY jog of (100, −10) lands at (100.005, −10.005) on the demo (12.5 ms) and
exactly on the fake. In dry run `select` returns the label and the turret stays,
and `engage()` returns `True` with the autofocus still off, while **switching
off is sent in dry run**: an engaged autofocus reads off after `disengage()`, and
a light that reads on reads off after `set_on(False)` (`set_on(True)` returns
`True` and the light stays off). **The demo's
`light_source` is `LED`, a `State` device**: `is_on()` is `None` and `set_on`
raises `HardwareError` (`light_source 'LED' (a State device) has no on/off that
smc knows; switch the light with its shutter (Shutter)`); a profile that assigns
`light_source = "White Light Shutter"` gives a light that switches and reads back,
on the fake and on the demo. On the Ti2-E the role is `DiaLamp`, a `Shutter`
(m2 §8). `Microscope.close()` stops what moves and unloads; it switches nothing
off, so a light or an autofocus that a command left on stays as the stand keeps
it.

### 6.1 The actions (`smc/drive.py`, new)

```python
@dataclass(frozen=True, slots=True)
class DriveResult:
    text: str
    reached: bool = True
```

```python
def jog_xy(
    microscope: Microscope, dx_um: float, dy_um: float, *, force: bool = False
) -> DriveResult: ...
def jog_z(
    microscope: Microscope, dz_um: float, *, force: bool = False
) -> DriveResult: ...
def set_objective(
    microscope: Microscope, label: str, *, confirm: bool
) -> DriveResult: ...
def engage_autofocus(microscope: Microscope) -> DriveResult: ...
def disengage_autofocus(microscope: Microscope) -> DriveResult: ...
def switch_light(microscope: Microscope, on: bool) -> DriveResult: ...
def um_text(value: float) -> str: ...
def xy_text(position: XY) -> str: ...
def z_text(z_um: float) -> str: ...
```

- Each one is `microscope.require(<capability>)` and **one** capability call, the
  one its command made or makes: `XYStage.move_by_um(dx, dy, force=force)`,
  `ZStage.move_by_um(dz, force=force)` (the interlock's default,
  `keep_autofocus=True`), `ObjectiveTurret.select(label, confirm=confirm)`,
  `Autofocus.engage()` (the profile's lock timeout; no `timeout_s`),
  `Autofocus.disengage()`, `LightSource.set_on(on)`. They raise what the
  capability raises (`CapabilityMissingError`, `SafetyRefusedError`,
  `MicroscopeHaltedError`, the facade's `ValueError` for an unknown label, …):
  the command maps it to its exit code, the page to its outcome text. No module
  but `smc.hardware` is reached; no guard is added or skipped here.
- `reached` is `False` when the call returned but the stand is not where it was
  asked to be: the command prints the text as an error and exits 1, the page
  shows it as the outcome. A dry-run text is always `reached=True`: nothing was
  asked of the stand but what dry run sends.
- **The texts** (`text`), each read back from the stand after the call, never
  from what was asked (FM-155, FM-167):

  | Action | Done | Dry run |
  |---|---|---|
  | `jog_xy` | `XY (x, y) µm`, the returned position | `[dry-run] would move to XY (x, y) µm; the stage is at XY (x, y) µm` |
  | `jog_z` | `Z z µm`, the returned position | `[dry-run] would move to Z z µm; the drive is at Z z µm` |
  | `set_objective` | `objective '<current()>'`; when `current()` is not `label`: `objective '<current()>', although '<label>' was asked`, `reached=False` | `[dry-run] would turn to '<label>'; the turret is at '<current()>'` |
  | `engage_autofocus` | `True`: `autofocus locked`; `False`: `autofocus did not lock; it is off`, `reached=False` | `[dry-run] would engage the autofocus; <describe_autofocus>` |
  | `disengage_autofocus` | `describe_autofocus(microscope.state())`; `reached=False` unless it is `autofocus off` | the same, then ` (dry run: switching off is sent all the same)` |
  | `switch_light(on=True)` | the readback: `light on`; `light off, although on was sent` with `reached=False` | `[dry-run] would switch the light on; it is <on, off or ?>` |
  | `switch_light(on=False)` | `light off`; `light on, although off was sent` with `reached=False` | `light off (dry run: switching off is sent all the same)` |

  Positions are `smc stage jog`'s: two decimals, `-0.00` shown as `0.00`
  (`um_text`, `xy_text` and `z_text`, which are `cli.py`'s `_um`, `_xy` and `_z`
  moved here; `cli.py` imports them). Labels are `repr`-quoted, so an empty
  position reads `''` (FM-125). `describe_autofocus(state: MicroscopeState) ->
  str` is `smc.hardware.microscope`'s `_describe_autofocus`, made public under
  that name (in the module's `__all__`; `Microscope.describe` calls it as
  before). The dry-run texts of `jog_xy` and `jog_z` are the ones `smc stage
  jog --dry-run` and `smc z jog --dry-run` print today, read again from the
  stage after the call, as `_move_xy` does.

### 6.2 The commands (`smc/cli.py`)

```
smc objective get  [-p PROFILE]
smc objective list [-p PROFILE]
smc objective set LABEL [--confirm] [--dry-run] [-p PROFILE]
smc autofocus get  [-p PROFILE]
smc autofocus engage    [--dry-run] [-p PROFILE]
smc autofocus disengage [--dry-run] [-p PROFILE]
smc light get [-p PROFILE]
smc light on  [--dry-run] [-p PROFILE]
smc light off [--dry-run] [-p PROFILE]
```

- Three groups beside `stage` and `z`: `objective_app` (help `The objective
  turret; a turn needs --confirm.`), `autofocus_app` (`The autofocus (Nikon
  PFS, Zeiss Definite Focus): engage, disengage, read.`) and `light_app` (`The
  light source: on, off, read.`), registered as `objective`, `autofocus` and
  `light`, each `no_args_is_help=True, rich_markup_mode="rich"` as `stage_app`.
- Each command opens the stand as `_move_xy` does (`_Reported`,
  `Microscope.open(profile, dry_run=dry_run)`), calls its action of §6.1 and
  prints `result.text` with `_say`, or with `_fail` and exit 1 when `not
  result.reached`. None prints `_say_stop_refused`'s line: none sends a move to
  `xy_stage`, `focus` or `autofocus_offset` that a refused stop leaves running
  (a turret cannot be stopped at all, and an engage's Z move is stopped by
  switching the autofocus off, which the stop does first; FM-117). A stand without the capability
  exits 1 with `CapabilityMissingError`'s text, as `smc stage get` does.
- `smc stage jog` and `smc z jog` call `jog_xy` and `jog_z`; their options,
  texts and exit codes do not change (their tests pin them).
- **`objective set`**: `--confirm` (help: `The stage is clear: turning the turret
  can drive an objective into a loaded plate.`) is passed as `confirm=`. A refusal
  for the missing confirmation exits 2 with the facade's reason and the hint
  `re-run with --confirm once the stage is clear` (`_CONFIRM_HINT`, passed as
  `force_hint=` to `_Reported`, as `_JOG_FORCE_HINT` is for a jog). A label that
  the turret does not have is the facade's `ValueError`, caught by the command
  and printed with `_fail`, exit 2: nothing was sent (the facade checks the
  confirmation first, then the label). `LABEL` is matched exactly as the turret
  names it (`smc objective list` prints them).
- **`objective get`** prints `objective '<label>'`; **`objective list`** one line
  per label in the turret's order, `'<label>'`, and `  (current)` after the
  current one. **`autofocus get`** prints `describe_autofocus(state)`. **`light
  get`** prints `light on`, `light off`, or, when `is_on()` is `None`, `light
  <device()>: no on/off that smc knows` (exit 0: a reading, not a failure).
- No option takes a free number here (FM-32), and none forces a guard: the
  confirmation is the turret's only guard that the operator answers.

### 6.3 The controller (`smc/ui/controller.py`)

```python
class PlatePage:
    def jog_xy(self, dx_um: float, dy_um: float) -> None: ...
    def jog_z(self, dz_um: float) -> None: ...
    def objective(self, label: str, *, confirm: bool) -> None: ...
    def autofocus(self, on: bool) -> None: ...
    def light(self, on: bool) -> None: ...
    def objective_labels(self) -> tuple[str, ...]: ...
    def light_lacks(self) -> str | None: ...
    def jog_limits_um(self) -> tuple[float, float]: ...
```

- **Names.** Each action is a drive action named as its command: `stage jog`,
  `z jog`, `objective set`, `autofocus engage`, `autofocus disengage`, `light
  on`, `light off`. The name is the activity's (§2.1: `stage jog is running`)
  and its outcome's prefix: `<name>: <result.text>`, or `<name>: <Type>:
  <message on one line>` for an exception, kept as text (FM-98), as a snap's.
  The page never passes `force=True`; `objective` passes the `confirm` it is
  given (the page's confirm button gives `True`, §6.4).
- **One private helper**, `_drive(name, capability, action)`, with `action:
  Callable[[], DriveResult]`, does what every action does, in this order:
  1. the page is closed: `RuntimeError("the page is closed")`;
  2. an activity goes that is not the live this page started (compared by
     identity, as `is_live`): `RuntimeError("<its name> is running")`, as
     `_refuse_if_busy`;
  3. `lacks(capability)` (and, for `light(True)`, `light_lacks()`): the outcome
     `<name>: <reason>`, and nothing starts or waits, as a snap without a camera;
  4. **live goes**: the action is handed to the live worker (below), and
     `_drive` returns;
  5. otherwise `_start_activity(name, body)`, whose body runs the action and
     returns its outcome text. The activity ends as every activity does, and
     raises `activities_ended` (§2.1's rule holds: every open tab refreshes).

  `light(False)` is never held back by `light_lacks()`: a light whose on/off smc
  does not know gets the facade's own error as its outcome, which says to use
  the shutter.
- **A drive action while live goes** runs on the live worker, **between two
  frames**: there is one thread on the devices, live goes on, and a frame
  taken after the action shows where it went (focusing and finding a field).
  The page keeps **one** waiting action, `_drive_waiting: tuple[str,
  Callable[[], str]] | None`, under `_lock`:
  - `_drive` takes `_lock`; if `_current_run` is no longer that live, or the
    live's event is set, it raises `RuntimeError("live is ending")`; if an
    action already waits,
    `RuntimeError("<its name> is waiting")` (a second click is refused, never
    queued: a queue would move the stage after the operator stopped clicking);
    otherwise it keeps the action and returns.
  - `_live`'s loop, at the top of each turn, before the frame: under `_lock`,
    takes the waiting action unless the event is set; outside the lock runs it,
    then puts its outcome text in `_outcome` (under `_lock`) and logs it at INFO
    (a drive action is logged; a frame never is, FM-148). A drive action that
    fails is its outcome and **live goes on**; only a frame that fails ends
    live (§5.1).
  - Whatever ends the loop (the event, a failed frame), a `finally` **sets the
    live's event**, then takes the action still waiting, under `_lock`, and
    drops it: it is **never sent**, and the live's outcome ends with `; <name>
    was not run`. STOP, End live, `close()` and a failed frame therefore never
    leave a move to be sent after them (FM-88's shape). The event is set first
    because a failed frame ends the loop without it: a `_drive` that took the
    lock after the drop would otherwise keep an action that no loop takes.
    Since the event is only ever set and `_drive` reads it under the lock, an
    action is either taken by the loop or refused, never lost.
  - The action waits for at most what is left of the frame in flight and of
    `LIVE_PERIOD_S`.
  - Measured on a throwaway subclass of `PlatePage` with this loop
    (2026-10-30, `FakeCore.demo_like()`, a 10 ms camera): a jog during live was
    taken 86 to 93 ms after the click, the next frames carried the new position,
    `is_live` stayed `True` and `activities_ended` did not change; a second jog
    while one waited was refused with `stage jog is waiting`; a jog left waiting
    at STOP was dropped and the stage did not move; after a frame that failed,
    a jog ran as its own activity and raised `activities_ended` by one.
  - It is **not** an activity: it does not raise `activities_ended` (§2.1, what
    an activity does while it goes is never an activity), so no tab redraws the
    plate's wells for it; the strip and the stage's marker follow on the next
    tick from `status()`.
- **A jog is one activity per click.** The page has no repeat while a button is
  held: Panel's button sends a click, not a press and a release, and a repeat
  that waits for a release the browser may never send (a lost focus, RDP) keeps
  moving. The step is the operator's choice (§6.4).
- **Halted**: a move, a turn, an engage and a light on are refused by the
  facade (`MicroscopeHaltedError`'s text is the outcome); `autofocus(False)` and
  `light(False)` run, since the facade never refuses switching off. The
  controls are not disabled while halted: the refusal says why.
- `objective_labels()` is `ObjectiveTurret.labels()` as a tuple, `()` when the
  stand lacks a turret or the read fails (logged at WARNING).
- `light_lacks()` is `lacks(LightSource)`, else, when `LightSource.is_on()`
  reads `None`, `the light source <device()> has no on/off that smc knows`,
  else `None`. A read that raises is `None` (the click then says what fails).
  The answer is kept after the first read that does not raise: a device's type
  does not change while the stand is held.
- `jog_limits_um()` is `(profile.safety.max_jog_um, profile.safety.max_z_jog_um
  or DEFAULT_MAX_Z_JOG_UM)` (`smc.hardware.safety`): what the page's step
  choices are bounded by (§6.4).
- `goto` (a click on the map in drive mode) is a run of the `goto-well` plugin,
  which writes a record: it stays refused while live goes (`live is running`).

### 6.4 The page (`smc/ui/plate_page.py`)

```python
#: smc/ui/plate_page.py
XY_STEPS_UM = (1.0, 10.0, 100.0, 1000.0)
Z_STEPS_UM = (0.5, 1.0, 5.0, 10.0, 50.0)
```

```python
class PlatePageView:
    xy_step_select: pn.widgets.Select
    z_step_select: pn.widgets.Select
    jog_buttons: dict[str, pn.widgets.Button]
    z_buttons: dict[str, pn.widgets.Button]
    autofocus_on_button: pn.widgets.Button
    autofocus_off_button: pn.widgets.Button
    objective_select: pn.widgets.Select
    objective_button: pn.widgets.Button
    objective_confirm: pn.Row
    objective_confirm_text: pn.pane.Str
    objective_confirm_button: pn.widgets.Button
    objective_cancel_button: pn.widgets.Button
    light_on_button: pn.widgets.Button
    light_off_button: pn.widgets.Button
    drive_note: pn.pane.Str

    def on_jog(self, direction: str) -> None: ...
    def on_z(self, direction: str) -> None: ...
    def on_autofocus(self, on: bool) -> None: ...
    def on_objective(self) -> None: ...
    def on_objective_confirm(self) -> None: ...
    def on_objective_cancel(self) -> None: ...
    def on_light(self, on: bool) -> None: ...
```

- **The region** is ADR-0006's *Drive*: at the top of the right-hand column,
  above the tools, under a bold `Drive` (as `Run document`). Top to bottom: the
  XY pad (`jog_buttons` keyed `up`, `left`, `right`, `down`, labelled `↑`, `←`,
  `→`, `↓`, with `xy_step_select`, `XY step`), the focus (`z_buttons` keyed
  `up`, `down`, labelled `Z +`, `Z −`, with `z_step_select`, `Z step`), the
  autofocus (`Autofocus on`, `Autofocus off`), the objective (`objective_select`,
  `Change objective`, and the confirmation row, hidden), the light (`Light on`,
  `Light off`), then `drive_note`.
- **Axes**: the pad is in the stage's axes, as `smc stage` (`X right, Y up`) and
  the plate map: `→` is `jog_xy(+step, 0)`, `↑` is `jog_xy(0, +step)`. `Z +` is
  `jog_z(+step)`. How the camera's image is turned against the stage is not
  known here: a stand where `↑` moves the image down is the owner's finding.
- **Steps**: the selects' options are `{f"{step:g} µm": step}` for the steps of
  `XY_STEPS_UM` and `Z_STEPS_UM` that do not exceed `jog_limits_um()`; defaults
  `100 µm` and `1 µm` (the largest offered step when the default is not
  offered). A profile whose limit is under every step offers the smallest one,
  and the facade's refusal says why.
- **The objective's confirmation** (FM-14). `objective_select` offers
  `objective_labels()`, an empty label shown as `(no label)` (`{label or "(no
  label)": label}`), and starts at the current objective; the tick never
  changes it (the strip says which objective is in place). `Change objective`
  **arms**: it keeps the chosen label, shows `objective_confirm` with the text
  `Turn the turret to '<label>'? A turn can drive an objective into a loaded
  plate: check that the stage is clear. The autofocus is switched off and not
  engaged again.` and its `Turn the turret` (`color="danger"`) and `Cancel`
  buttons. `Turn the turret` calls `page.objective(<the armed label>,
  confirm=True)` and disarms; `Cancel` disarms; a change of `objective_select`
  disarms; so does a tick that finds the drive controls disabled. A `Turn the
  turret` click with nothing armed (a stale browser) turns nothing and says
  `choose the objective again`.
- **Disabled**: every drive control is disabled while an activity other than
  this page's live goes (`status.running is not None and not page.is_live`),
  and **enabled while live goes**. A control is also disabled when its stand
  lacks its capability, and `drive_note` says why, one line per reason:
  `<control>: <lacks(...)>` for `XYStage` (the pad), `ZStage` (focus),
  `Autofocus`, `ObjectiveTurret` (or, with a turret, `objective: its positions
  could not be read` when `objective_labels()` is `()`), and `light:
  <light_lacks()>`. Measured on Panel 1.10.0 and 1.9.4: `clicks += 1` on a
  disabled button **still calls its handler** on the server (a browser holds a
  click on a button it shows disabled; one that has not yet drawn the change
  sends it), so the controller's refusals are the guard and the button's state
  only a courtesy; every handler goes through `_try` as the others do.
- **Each handler** calls its controller action through `_try` (a refusal is the
  page's message, FM-98) and then `tick()`, as `on_snap`.
- Widget spellings that do not warn on the lab's Panel (FM-147, measured on
  1.10.0 and 1.9.4 under `warnings.simplefilter("error")`): `label=`, not
  `name=`; `color="danger"`, not `button_type=`.

### 6.5 Tests and failure modes

Tests are named in #263's plan. **Failure modes**: FM-14, FM-12, FM-88, FM-89,
FM-118, FM-117, FM-64, FM-98, FM-70, FM-44, FM-96, FM-73, FM-125, FM-147,
FM-151, FM-155, FM-167, FM-36, FM-37, FM-32, FM-43, FM-45, FM-47, FM-161.

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
- Drive at a stand (§6): which way the image moves for `↑` and `→`; a jog, a
  focus step and an engage on the Ti2-E while live goes, and how long a click
  waits for the frame in flight on a real exposure; a turret turn under a
  plate; `smc light on` and `off` on `DiaLamp`; what the stand keeps on after
  a command ends (the light, the autofocus). With the autofocus locked on a
  stand that declares `engage_drives_z_to_offset`, `Z +` and `Z −` re-engage
  after the move (the interlock), which may drive Z back to the offset's plane
  (FM-118): the jog would not stay, and focusing by hand needs the autofocus
  off.
- A newcomer registers a plate on the `demo-sample` stand from the page, without
  a manual (#28's last criterion), with a lab member.
