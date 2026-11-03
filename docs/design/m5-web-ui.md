# M5 design — the web UI's shell

- **Status**: design for issue #28 (§1 to §4, 2026-10-29), as built by #270 and
  settled by #272 (§2.1, §2.4, §3: the stale card, the card's bounds,
  `card_text`'s signature); live fixed by #262's plan (§5, 2026-10-29); drive
  fixed by #263's plan (§6, 2026-10-30); the stand's lifetime and the service
  fixed by #264's plan with the owner (§7, 2026-10-30); the first run fixed by
  #293's plan with the owner (§8, 2026-11-02); the drive's follow-ups fixed by
  #295's plan (§6.1 to §6.4, 2026-11-02: a switch-off at the end of live, the
  outcome live keeps, the captures' refusal, the readbacks from the state, the
  objective select's sync); presets fixed by #319's plan (§10, 2026-11-03).
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
| Deployable as a service | no | #264: NSSM, `localhost` only (§7) |

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
shows) goes to the next issue that edits `controller.py`: **#262** (§9). Its
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
  image's column>), dynamic=False)`: the plate map as today, and the image. The
  tabs are **not dynamic** (#297): a dynamic tab destroys the figure's view when
  another tab is shown, and bokehjs's `DataRange1d` keeps one bounds entry per
  view and never drops a dead one, so a map redrawn while *Image* was shown came
  back framed on the union of every plate it had drawn, and Reset and Refresh
  recomputed from the same entries. Bokeh keeps the view of a hidden tab and its
  size, so no figure is laid out at zero size. The checklist stays under the
  tabs, on both.
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
handler. Not measured then: a real browser (the tab switch, the image's
orientation and look).

Measured 2026-11-02 in headless Chromium 1248 (Playwright), Panel 1.10.0 / Bokeh
3.10.1 and Panel 1.9.4 / Bokeh 3.9.2, reading `Bokeh.documents[0]` (#297): with
`dynamic=True` the map's `DataRange1d._plot_bounds` held the bounds of the
nominal map's dead view and of the registered plate's, after a registration, a
snap on *Image* and the return to *Plate*, and the frame was their union (x
−80996 to 130496 µm for a plate of ±52800); with `dynamic=False` it held one
entry and the frame was the plate's alone (x ±78521, the canvas's aspect, 2.057
at 1008 x 490 px), after the same steps, after Reset and after Refresh. A
resize made while a tab is hidden was matched when it was shown (1.037 at 508 x
490, 2.057 back at 1008 x 490), and an image drawn while *Plate* was shown was
framed on the first show of *Image*. Not measured: Windows, and a live view
drawn by another session into the figure of a tab nobody looks at (ten frames a
second, localhost only).

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
- **Its outcome** (§6.3 adds what the last drive action and a switch-off at the
  end say after it): ended by its event (`live(False)`, STOP, `close()`): `live
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
  a camera, and while an activity other than this live goes. **A click does
  what the label it shows says** (#287), not what the controller's state is at
  the click: the label is what the last tick drew, up to a second ago, and the
  click reads it, `starting = self.live_button.label != "End live"`, then calls
  `page.live(starting)` (an exception's message is the page's message, FM-98);
  starting it makes *Image* the active tab; then `tick()`, which draws the label
  again. The two labels are constants of the module, the one `_draw_buttons`
  draws and the one the click reads. So a **stale *Live*** (another tab started
  live less than a tick ago) is refused by the controller, `live is running`, and
  does not end that tab's live; and a **stale *End live*** (live ended in the
  last second: a frame that failed, another tab's *End live*) ends nothing and
  starts nothing, in particular no live on the camera that just failed: the
  outcome under the strip already says why it ended, and the redrawn label is
  the answer to the click. Both were the opposite of the label while the click
  read `page.is_live` (a `live ended after 0 frames: …` on a camera that had just
  failed; the first tab's live ended by the second's click). Residual: the label
  is the server's copy of what the browser draws, so a click made while a new
  label is on its way acts on the newer one; not removable without a token in
  the click.
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
  limits 5 frames a second on the demo. The demo test measures the frame step
  with the cap's wait taken out, and the cap is proved on an owned clock,
  because a timed wait returned late on the macOS runner (inferred from CI logs,
  not probed on it, #311). What a browser
  shows, locally and over RDP, is the owner's to measure, and is ADR-0006's
  revisit criterion for Panel.
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
def move_xy(microscope: Microscope, x_um: float, y_um: float) -> DriveResult: ...
def move_z(microscope: Microscope, z_um: float) -> DriveResult: ...
def read_autofocus(microscope: Microscope) -> AutofocusReading: ...
def um_text(value: float) -> str: ...
def xy_text(position: XY) -> str: ...
def z_text(z_um: float) -> str: ...
```

```python
@dataclass(frozen=True, slots=True)
class AutofocusReading:
    text: str
    engaged: bool | None
    known: bool
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
  | `disengage_autofocus` | `read_autofocus(...).text`; `reached=False` unless `engaged is False` | the same, then ` (dry run: switching off is sent all the same)` |
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
  stage after the call.
- **`move_xy` and `move_z`** (#295) are `smc stage move` and `smc z move`:
  `require` and **one** call, `XYStage.move_to_um(x, y)` or
  `ZStage.move_to_um(z)`, and the texts of the two jogs, which one private helper
  builds for all four (the dry-run tail "would move to ...; the stage is at ..."
  is written once; before #295 `cli.py`'s `_move_xy` and `_move_z` held a second
  copy, which go). The page has no absolute move: its own is `goto-well`.
- **A decision is read from a field, never from a text** (#295, FM-155).
  `read_autofocus` reads `state()` **once** and gives: `engaged`, the state's
  `autofocus_engaged`; `known`, `autofocus_known(state)`, a public function of
  `smc.hardware.microscope` that `describe_autofocus` itself calls (`False` exactly
  when the words are `autofocus ?`: a read that the answer needs failed); and
  `text`, `describe_autofocus(state)`, to which an unknown state adds its causes,
  once each, in brackets, **each on one line** (`" ".join(cause.split())`: a driver's
  multi-line message must not reach a one-line outcome). `disengage_autofocus` is
  `reached` when `engaged is False`; `engage_autofocus`, after a `False`, says `it is
  off` when `engaged is False` and `<text>` otherwise; `smc autofocus get` exits 1
  when `not known`. No decision compares a string, so rewording `describe_autofocus`
  cannot flip a `reached`. `autofocus_text` (a string) goes: it was `read_autofocus(...).text`.
  The controller's `status()` calls `describe_autofocus(state)` and no longer keeps
  its copy `_autofocus_words`.

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
- Each command goes through `_drive_command` (`_Reported`,
  `Microscope.open(profile, dry_run=dry_run)`), calls its action of §6.1 and
  prints `result.text` with `_say`, or with `_fail` and exit 1 when `not
  result.reached`. None of the three groups below prints `_say_stop_refused`'s
  line: none sends a move to `xy_stage`, `focus` or `autofocus_offset` that a
  refused stop leaves running (a turret cannot be stopped at all, and an
  engage's Z move is stopped by switching the autofocus off, which the stop does
  first, FM-117; `_drive_command`'s docstring says so). A stand without the
  capability exits 1 with `CapabilityMissingError`'s text, as `smc stage get`
  does.
- `smc stage jog` and `smc z jog` call `jog_xy` and `jog_z`, and (#295) `smc
  stage move` and `smc z move` call `move_xy` and `move_z`, all four through
  `_drive_command` with the line `_say_stop_refused` prints for `xy_stage` or
  `focus`; their options, texts and exit codes do not change (their tests pin
  them).
- **An empty turret position** (#295) is named by an empty `LABEL`: `smc objective
  set "" --confirm`. Windows PowerShell 5.1 drops an empty argument on its way to
  a program, so there it is `smc objective set '""' --confirm`; `cmd.exe` and the
  page take `""` and the choice `(no label)`. This is PowerShell's documented
  behaviour, not measured here (§9.1); `docs/ui/plate-page.md` says it.
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
  The page keeps **one** waiting action, `_drive_waiting: _Handed | None`, under
  `_lock`, where `_Handed` is a `NamedTuple` of `name: str`, `run: Callable[[],
  str]` and `switch_off: bool` (#295: `True` for `light off` and `autofocus
  disengage`, the two switch-offs; `_drive` takes it as a keyword, default
  `False`). While an action runs on the live worker, `_drive_running: str | None`
  holds its name, under `_lock` (#295: what a capture reads, below):
  - `_drive` takes `_lock`; if `_current_run` is no longer that live, or the
    live's event is set, it raises `RuntimeError("live is ending")`; if an
    action already waits,
    `RuntimeError("<its name> is waiting")` (a second click is refused, never
    queued: a queue would move the stage after the operator stopped clicking);
    otherwise it keeps the action and returns.
  - `_live`'s loop, at the top of each turn, before the frame: under `_lock`,
    takes the waiting action unless the event is set and sets `_drive_running`
    to its name; outside the lock runs it, then, under `_lock`, puts its outcome
    text in `_outcome`, clears `_drive_running`, and logs it at INFO (a drive
    action is logged; a frame never is, FM-148). A drive action that fails is its
    outcome and **live goes on**; only a frame that fails ends live (§5.1). The
    loop keeps the outcome text of the **last** action it ran (#295: below).
  - Whatever ends the loop (the event, a failed frame), a `finally` **sets the
    live's event**, then takes the action still waiting, under `_lock`:
    - **any action but a switch-off is dropped**: it is **never sent**, and the
      live's outcome ends with `; <name> was not run`. STOP, End live, `close()`
      and a failed frame therefore never leave a move, a turn or an engage to be
      sent after them (FM-88's shape);
    - **a switch-off is run** (#295), once, on the live worker, after the loop and
      outside the lock, as any drive action runs (`_drive_running` holds its name,
      its outcome is `<name>: <text>`, a failure is that text and never raised,
      FM-98), and its outcome ends the live's. Dropping it left the light on
      after a STOP (measured on `FakeCore.demo_like()` with the White Light
      Shutter: `light off` clicked behind a frame, then STOP: `live ended: 1
      frames in 0.0 s; light off was not run`, and the light still read on). It
      is the safe call the facade never refuses, also while halted (measured: on a
      halted facade `switch_light(off)` and `disengage_autofocus` run and read back
      `light off` and `autofocus off`, a jog is refused), and it sends **no move**,
      so FM-88's reason for dropping a restore step does not apply to it: a
      switch-off is not a restore and moves nothing. STOP does not wait for it
      (`stop()` takes no lock and joins nothing, FM-70); `close()` waits for it
      within `_CLOSE_WAIT_S`, as for a frame.

    The event is set first because a failed frame ends the loop without it: a
    `_drive` that took the lock after the drop would otherwise keep an action
    that no loop takes. Since the event is only ever set and `_drive` reads it
    under the lock, an action is either taken by the loop, run at the end or
    dropped, or refused, never lost.
  - **What live's outcome says** (#295). Before, `_start_activity` replaced the
    outcome of the last drive action with live's own at the end, and only the log
    kept it. Now live's outcome is `live ended: <n> frames in <s:.1f> s` (or `live
    ended after <n> frames: <text>`, §5.1), then, each only when it exists: `;
    <outcome of the last action the loop ran>` (it begins with `<name>: `), then
    `; <outcome of the switch-off run at the end>` or `; <name> was not run`.
    Example: `live ended: 2 frames in 0.1 s; stage jog: XY (100.00, -50.00) µm;
    light off: light off`. Only the last action's outcome is kept: the earlier
    ones were on the strip while live went, and the log has every one.
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
- **A capture reads the stage, so it is refused while the stage is about to move
  or moves** (#295). `capture_centre` and `capture_wall`, after their argument
  checks (a bad `side`, a well that is not on the plate: `ValueError`, as before)
  and before they read the stage, raise `RuntimeError("<name> is running")` when
  an activity other than this page's live goes (a run, a snap, a drive action of
  its own, `connect`, `disconnect`: `goto-well` and every plugin that moves the
  stage are runs), and, while live goes, `RuntimeError("<name> is waiting")` when
  an action waits and `RuntimeError("<name> is running")` when one runs on the
  live worker (`_drive_running`). Before, a capture clicked behind a jog recorded
  the stage *before* the jog (measured: a jog of 500 µm waiting behind a frame,
  `capture_centre` returned (0, 0), and the stage then went to (500, 0); the same
  with a jog that ran on the live worker, whose `_drive_waiting` was already
  empty), and the capture takes no busy check at all, so a `goto-well` run's
  stage is read mid-route by the same code. A live with no action waiting or
  running does not refuse: a frame moves nothing.
  The check takes `_lock` briefly and is not atomic with the read: the window is
  one `position_um()` call, for an operator clicking in two tabs. The view's
  capture buttons are disabled for the same states as a courtesy (§6.4); the
  controller is the guard (FM-169).
- `objective_labels()` is `ObjectiveTurret.labels()` as a tuple, `()` when the
  stand lacks a turret or the read fails. **A failure is logged at WARNING once**
  (#295) and not again until a read succeeds or the connection changes: the tick
  reads it again while the choices are empty (§6.4), and a line a second would
  fill the log and block the loop that handles STOP (FM-62, FM-148; measured: five
  failing reads logged five lines).
- `light_lacks()` is `lacks(LightSource)`, else, when `LightSource.is_on()`
  reads `None`, `the light source <device()> has no on/off that smc knows`,
  else `None`. A read that raises is `None` (the click then says what fails).
  The answer is kept after the first read that does not raise: a device's type
  does not change while the stand is held. **One read a tick** (#295): `status()`
  reads `is_on()` on every tick, so it hands its result to `light_lacks()` for the
  stand it read: a read that did not raise gives the answer (and keeps it), and
  one that raised gives `None` without a second read (measured: a failing
  `is_on()` was called twice a tick, by `status()` and by `light_lacks()`).
  `light_lacks()` called before any `status()` reads for itself.
- `jog_limits_um()` is `(guard.max_jog_um, guard.max_z_jog_um)` of a
  `smc.hardware.safety.Safety(max_jog_um=profile.safety.max_jog_um,
  max_z_jog_um=profile.safety.max_z_jog_um)` (#295): `Safety` is the class the
  facade enforces the jog with and the one that owns "no `max_z_jog_um` means
  `DEFAULT_MAX_Z_JOG_UM`", so the controller no longer repeats that rule and
  imports `DEFAULT_MAX_Z_JOG_UM` no more. It reads the profile and not the facade,
  because the page builds its steps while the stand may be closed (§7.3). What the
  page's step choices are bounded by (§6.4).
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
  label)": label}`), and starts at the current objective. **The select follows
  the turret until it has, once** (#295): each tick, while the select has not been
  set, sets it to `status.objective` when that is one of its options, and only
  then is it set; a read that failed gives `None`, and the next tick tries again
  (before #295 the first tick counted, and a first `current()` that raised left
  the select on its first option: *Change objective* then turned to that option
  and not to the objective in place; measured with a fake whose first `current()`
  raised: the select stayed on `Nikon 40X Plan Fluor ELWD` while `Nikon 10X S
  Fluor` was in place). After that the tick never changes it (the strip says which
  objective is in place); a change of connection starts it again (§7.3). Setting
  it disarms a confirmation armed meanwhile (the select's own watcher), so the
  turn that follows is always to a label the operator chose after the sync.
  **The choices are read again while there are none** (#295): a tick that finds
  `objective_select.options` empty on a connected stand calls
  `page.objective_labels()`; when it returns labels, the options are set and the
  sync above starts again. Before, one failed `labels()` at the build disabled
  the control for the life of the tab (measured: still `{}` and disabled after
  later ticks that would have read). The failing read is logged once (§6.3).
  `Change objective`
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
  only a courtesy; every handler goes through `_try` as the others do. **The
  capture buttons** (*Capture centre*, *Capture wall*, #295) are disabled with the
  stand closed, as before, and now also while an activity other than this page's
  live goes (`running is not None and not page.is_live`): the controller refuses a
  capture then (§6.3), and the button says so before the click. While live goes
  they stay enabled: only the controller knows whether an action waits or runs,
  and its refusal is the message.
- **The labels are the options' keys, so equal labels collapse** (#295, noted and
  not coded around): `{label or "(no label)": label}` has one option for two
  positions that share a label, or for two empty ones, and the layer addresses a
  turret position by its label too (`select(label)`), so the limit is the layer's
  and not the page's. Whether Micro-Manager allows such labels on a real stand is
  not known (§9.1).
- **Each handler** calls its controller action through `_try` (a refusal is the
  page's message, FM-98) and then `tick()`, as `on_snap`.
- Widget spellings that do not warn on the lab's Panel (FM-147, measured on
  1.10.0 and 1.9.4 under `warnings.simplefilter("error")`): `label=`, not
  `name=`; `color="danger"`, not `button_type=`.

### 6.5 Tests and failure modes

Tests are named in #263's plan, and in #295's for the follow-ups above.
**Failure modes**: FM-14, FM-12, FM-88, FM-89,
FM-118, FM-117, FM-64, FM-98, FM-70, FM-44, FM-96, FM-73, FM-125, FM-147,
FM-151, FM-155, FM-167, FM-36, FM-37, FM-32, FM-43, FM-45, FM-47, FM-161. #295
adds FM-180 (a drop at STOP that also drops the safe call) and FM-181 (a position
recorded while a move waits or runs).

## 7. The stand's lifetime and the service (#264)

Fixed by #264's plan (2026-10-30). Until #264, the command opens the stand when it
starts and holds it until Ctrl-C. A service runs for weeks, and a stand that smc
holds cannot be taken by NIS-Elements or ZEN. So the stand's lifetime becomes the
operator's: **Connect** opens it and **Disconnect** releases it, on the page. The
server's lifetime stays the command's.

**Decided with the owner (2026-10-30)**:

- **The page listens on `localhost` only**, as a service too, and the command
  has no option to change it. The page has no login (M3 §15.3), so whoever
  reaches the port can move the stage. From a desk, the route is the PC's remote
  desktop. ADR-0006's "served from a lab server" needs an ADR first, which
  decides authentication (who may drive which stand, over TLS). Until then it is
  not allowed.
- **The page always starts with the stand closed**, from a console as in a
  service. There is no `--connect`: the page behaves one way, and a service that
  starts at boot, or restarts after a crash, never takes the stand from the
  vendor's software.

**Also decided here**:

- The page connects to **the profile the command names**. ADR-0006's row of
  stands under *Connect* is not built: one PC serves one stand, and the demo is
  another `smc ui -p demo` on another port.
- The profile is read **once**, when the command starts, as today: the strip, the
  jog limits and the run document's checks use it while the stand is closed.
  A profile edited on disk takes effect when the command, or the service,
  restarts.

### 7.1 The command (`smc/cli.py`)

The options are §1's, unchanged. What changes is the order:

1. The checks are as they are today, and so are their texts and exit codes:
   - the page word;
   - `--run-doc`;
   - the `ui` extra;
   - the port;
   - `Profile.load` (a profile that does not load exits as today);
   - `RunDocument.load_or_new`.
2. Unless `--dry-run`, the line `_stop_refused_line(loaded.safety.stop_refused_by)`
   is printed from the profile, when there is one. `_say_stop_refused` needed the
   facade.
3. `controller = PlatePage.closed(loaded, path, dry_run=dry_run)`, then
   `plate_page.serve(controller, port=port)`.
4. Whatever ends `serve`, a `finally` calls `controller.close()` and keeps
   `close_lines` and `close_ended`. An inner `finally` then calls
   `controller.release()`. The stand is released whatever ends this, a second
   Ctrl-C during the wait included (FM-71), as the `with Microscope.open(...)`
   block did. An error that `release()` raises is reported by `_Reported` as the
   `with` block's was.
5. After the release, `close_lines` and the exit code are as today (§1, M3 §15.3).

The help says:

> Serve the plate page (needs the ui extra): connect the stand, click wells, run
> the tools, snap a frame.
>
> The page starts with the stand closed: Connect opens the profile's stand and
> Disconnect releases it, for NIS-Elements or ZEN. Every tool button is a plugin
> run with the parameters smc run takes, so each has a command-line equivalent,
> and each tool shows its last result. Ctrl-C sends a stop to any move under way
> and tells what is going to stop, then ends the server, then releases the
> stand. A stand that refuses stop() lets a sent move run to its target, and
> the line printed says what each stop did.

### 7.2 The controller (`smc/ui/controller.py`)

```python
#: smc/ui/controller.py
NOT_CONNECTED = "the stand is not connected"


class PageStatus:
    ...
    connected: bool = True  # the last field


class PlatePage:
    def __init__(
        self,
        microscope: Microscope,
        run_doc: Path,
        *,
        opener: Callable[[], Microscope] | None = None,
        entry_points: Iterable[EntryPoint] | None = None,
    ) -> None: ...
    @classmethod
    def closed(
        cls,
        profile: Profile,
        run_doc: Path,
        *,
        dry_run: bool = False,
        opener: Callable[[], Microscope] | None = None,
        entry_points: Iterable[EntryPoint] | None = None,
    ) -> PlatePage: ...
    @property
    def connected(self) -> bool: ...
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def release(self) -> None: ...
```

**Building the page**

- `PlatePage(microscope, run_doc)` starts **connected** to that facade, as every
  test and notebook builds it today.
- `PlatePage.closed(profile, run_doc, dry_run=...)` starts **closed**. It is
  what `smc ui` builds.
- The page keeps the profile (`_profile`) and the dry run (`_dry_run`), and takes
  them from the facade when it is given one. Every read that does not need the
  stand uses them instead of `self._microscope.profile` and `.dry_run`:
  - `status().stand`, `status().dry_run` and `status().stop_refused`;
  - the profile and dry run that `run()` hands `check_ready`;
  - `_unusable`;
  - `jog_limits_um()`.

  The checklist, the plate map, the selection and the result cards therefore
  work with the stand closed.
- `opener` is the connection: a call that returns an open facade. It defaults to
  `Microscope.open(profile, dry_run=dry_run)`, which is looked up **when Connect
  calls it**, so that a test's `monkeypatch.setattr(Microscope, "open", ...)` is
  the one used. The import of `Microscope` stays out of the module's top: it is
  inside the private function `_open_stand(profile, dry_run)`.
- `closed` builds the instance with `cls.__new__(cls)` and a private
  `_setup(...)`, which `__init__` also calls, so `__init__`'s signature does not
  change for its 79 callers.
- `connected` is `self._microscope is not None`.

**Connect and Disconnect are activities** (§2.1). Each one starts on the worker
under its name, `connect` or `disconnect`, raises `activities_ended` when it ends
(every open tab then calls `refresh()`), and is refused while another activity
goes with `<name> is running`. So **Disconnect is refused while a run, a snap,
live or a drive action goes**, and no activity can find the facade changed under
it.

- **`connect()`** checks, in this order:
  1. The page is closed: `RuntimeError("the page is closed")`.
  2. Another activity goes: `RuntimeError("<name> is running")`.
  3. The page is already connected: the outcome `connect: <stand> is already
     connected`, and no thread starts.
  4. Otherwise it starts the `connect` activity, whose body:
     - calls `opener()`. An exception becomes the outcome `connect: <Type>:
       <message on one line>`, kept as text (FM-98). That covers a
       `CoreError` from a stand that NIS-Elements holds, and a `ProfileError`
       for a `[sample]` stand.
     - then takes `_lock`. **When the page was closed meanwhile** (`close()`
       gave up waiting for a slow open), it does not keep the facade: it releases
       it outside the lock, and the outcome is `connect: <stand> was opened after
       the page closed, and released`. A stand is never left open with no owner.
     - otherwise keeps the facade, clears `_stop_note` (a new facade is not
       halted) and clears `_light_lacks_found`.
     - **when the activity's event is set** (STOP was pressed while the stand
       connected), it calls `self.stop()` on the new facade, so the stand is
       halted as the operator asked. The outcome then ends with `, and halted:
       STOP was pressed while it connected`.
     - The outcome is `connect: <stand> connected, <n> devices`, where `<n>` is
       `len(microscope.devices)`, then ` (dry run)` when it is a dry run.
- **`disconnect()`** checks, in this order:
  1. The page is closed: `RuntimeError("the page is closed")`.
  2. Another activity goes: `RuntimeError("<name> is running")`.
  3. The page is not connected: the outcome `disconnect: the stand is not
     connected`, and no thread starts.
  4. Otherwise it starts the `disconnect` activity, whose body:
     - under `_lock`, drops the facade (`_microscope = None`) and clears
       `_light_lacks_found`;
     - **outside** the lock, calls `Microscope.close()`. That stops what still
       moves, then unloads every device, and it switches nothing off: a light
       or an autofocus stays as the stand keeps it (§6);
     - has the outcome `disconnect: <stand> released`, or `disconnect: <Type>:
       <message on one line>` when `close()` raises. The page has let go of the
       facade either way. `close()` logs what to do when unloading fails.
  - The facade is dropped **before** it is closed, so no read that starts after
    the drop reaches a closed facade.
  - Measured (§7.5): a closed facade's `state()` reads `HardwareError ...
    another program holding it` for XY and Z, which would mislead, and its
    `has(XYStage)` still answers `True`.
- **`release()`** releases the facade the page holds, if any:
  - It drops it under `_lock`, then calls `Microscope.close()` outside it, and
    raises what that raises.
  - It is for the command, after `close()`. Before it, it raises
    `RuntimeError("release() comes after close()")`, because the facade could
    otherwise be pulled from under an activity.
  - It is idempotent, and it does nothing on a page that holds no facade.
  - It releases a facade handed to the constructor too. The tests that build a
    page over a fixture's facade never call it.
- **`close()`** keeps its contract: it stops the activity, waits up to
  `_CLOSE_WAIT_S`, never releases the stand, and never raises for a refused
  stop. Once it has run, `connect` and `disconnect` refuse with `the page is
  closed`. A `connect` still opening when the wait ends releases what it opens
  (above).

**What needs the stand, when the stand is closed** (principle 1: disabled with the
reason, never failing when pressed):

| Call | With the stand closed |
|---|---|
| `lacks(capability)` | `NOT_CONNECTED`, checked first. So a snap, live and every drive action have the outcome `<name>: the stand is not connected`, through the paths they already have (§2.2, §5.1, §6.3), and start nothing. `light_lacks()` gives the same. |
| `run(plugin, ...)` | after `_refuse_if_busy`, the outcome `<plugin>: the stand is not connected`, and no thread starts. `smc run` opens the stand for every plugin, so the page's Run needs it too. `goto` is a run of `goto-well`. |
| `_blocked(doc)` | every tool, `NOT_CONNECTED`, ahead of any other reason: what the tool's note says. |
| `capture_centre`, `capture_wall` | `RuntimeError(NOT_CONNECTED)`, through a private `_stand() -> Microscope` that raises it. |
| `objective_labels()` | `()` |
| `stop()` | sets the activity's event as before. The note is `<prefix>the stand is not connected: nothing to stop`, with the usual `<name> told to stop; ` prefix. It never raises. |
| `resume()` | `RuntimeError(NOT_CONNECTED)` |
| `status()` | `PageStatus(connected=False, ...)`: `stand`, `dry_run`, `running`, `progress`, `outcome`, `stop_note` and `stop_refused` as when connected; `xy_um`, `z_um`, `objective` and `light_on` are `None`; `autofocus` is `"?"`; `moving` and `problems` are `()`; `halted` is `False`. |
| `frame()` | the last frame, kept: it is a picture taken, with its time. |

- **`status()` and the facade.** `status()` reads the facade once, into a
  local, and reads the stand from that. If the page no longer holds that facade
  after the read, because a `disconnect` dropped it meanwhile, the status is the
  closed stand's. The stand's read is never reported as a fault of a stand that
  was released.
- **Every other method** that reads the facade on the server's loop
  (`lacks`, `objective_labels`, `light_lacks`, the captures) also takes it into
  a local first, never `self._microscope` twice.
- **A halt does not outlive the facade.** The note of a STOP stays after a
  Disconnect. A Connect clears it, because the new facade is not halted.
  `smc` commands each open a fresh facade too.

### 7.3 The page (`smc/ui/plate_page.py`)

```python
class PlatePageView:
    connect_button: pn.widgets.Button
    disconnect_button: pn.widgets.Button
    connection_text: pn.pane.Str

    def on_connect(self) -> None: ...
    def on_disconnect(self) -> None: ...
```

- **The region** is ADR-0006's *Connect*: a row **above** the strip, made of
  `Connect`, `Disconnect` and `connection_text`.
- `connection_text` depends on the state:
  - closed: `<stand> is not connected: Connect opens it, and the controls that
    need the stand are disabled until then`;
  - connected: `Disconnect releases <stand> for another program (NIS-Elements,
    ZEN)`;
  - in both cases, then ` (dry run)` when it is a dry run.

  This one line is the reason for every control that needs the stand.
- **Disabled**:
  - `Connect` while connected, or while an activity goes;
  - `Disconnect` while closed, or while an activity goes (live included);
  - the capture buttons while closed.

  Each controller method refuses on its own, so a click on a disabled button is
  refused all the same (FM-169).
- **The handlers** call `page.connect` and `page.disconnect` through `_try` (a
  refusal is the page's message, FM-98), then `tick()`, as `on_snap` does.
- **Closed, the regions say it**:
  - the strip: `status_line` gives `<stand> | not connected`, then `| DRY RUN`
    when it is a dry run, and nothing else;
  - the tools' notes: `_blocked`;
  - `snap_note`: `lacks(Camera)`;
  - `drive_note`: the one line `the stand is not connected`, not five.
- **The objective's choices follow the connection.** The tick keeps the
  `connected` it last saw. When `status.connected` changes, it reads
  `page.objective_labels()` again, sets `objective_select.options` (as
  `_build_drive` does), lets the next tick set the select to the objective in
  place, and disarms the confirmation. The drive region is built once per
  session, while the stand may be closed. `jog_limits_um()` comes from the
  profile and does not change.

### 7.4 The service (`docs/ui/service.md`, new)

NSSM, as the lab runs `nikon-control`'s dashboard
(`nikon-control/docs/windows-deploy.md`). The document gives the commands to
install, start, stop, restart and remove the service, named `smc-ui`, and says
what it changes for whoever uses the PC. It fixes the following:

- **What the service runs**: the venv's `python.exe -m smc.cli ui -p <profile>
  --run-doc <absolute path> --port 5006`. It is one process: a console-script
  `.exe` launcher would run Python as a child, which a stop must then also
  reach.
- **The account**: the Windows account for which Micro-Manager, the vendor
  drivers and the venv were set up on that PC, given as `ObjectName .\<account>
  <password>`. It is not `LocalSystem`, which has none of that account's
  environment.
- **The working directory and the environment**:
  - `AppDirectory` is the checkout, whose `profiles\` is on the search path;
  - `AppEnvironmentExtra` is `SMC_PROFILES=<site profiles>` and `PYTHONUTF8=1`.
    The log is a file, which Windows would otherwise write in cp1252 (FM-20).
- **The log**:
  - `AppStdout` and `AppStderr` are both `C:\smc\logs\smc-ui.log`;
  - rotation is on (`AppRotateFiles 1`, `AppRotateOnline 1`, `AppRotateBytes
    10485760`).
- **The stop**: NSSM's first stop method is a Ctrl-C to the console it creates.
  That is the command's own Ctrl-C (§7.1, FM-143): the activity is told to
  stop, then the server stops, then the stand is released. NSSM waits only 1.5 s
  by default before its next method, and ends with `TerminateProcess`. So
  `AppStopMethodConsole 30000` covers `close()`'s 10 s and the server's 10 s.
  `AppNoConsole` stays `0`: without a console, there is no Ctrl-C.
- **The start**: `Start SERVICE_AUTO_START`. `AppExit` stays at its default,
  `Restart`, with NSSM's throttle. A restart never takes the stand, since the
  page starts closed.
- **What it changes for whoever uses the PC**:
  - **The port is always held**. A `smc ui` started by hand on 5006 is refused
    with `port 5006 is in use`: use `http://localhost:5006/`, or give another
    `--port`.
  - **The stand is held only between Connect and Disconnect.** While it is
    held, NIS-Elements, ZEN and the `smc` commands cannot open it: Disconnect
    first. Disconnect switches nothing off.
  - **Anyone logged on to the PC** (any account, at the PC or over RDP) reaches
    `localhost:5006` and can drive the stand. There is no login.
  - **The service survives a logoff and a reboot**, and starts with the stand
    closed.
  - **Files**: the run document and what the tools write belong to the service's
    account, so their folder must be writable by it. A service does not see
    mapped drive letters (`Z:`): give local or UNC paths.
  - **Restarts**: a changed profile, or another run document, needs `nssm set
    smc-ui AppParameters ...` or `nssm restart smc-ui`. A restart ends a run
    that is going, as Ctrl-C does.
  - **What the page says** goes to the log file, not to a console.
- **Stopping the PC**: Disconnect first. How long Windows lets a service stop at
  shutdown is not known here.

### 7.5 Tests and failure modes

The tests are named in #264's plan.

**Failure modes**: FM-98, FM-70, FM-71, FM-92, FM-143, FM-146, FM-149, FM-151,
FM-169, FM-161, FM-117, FM-125, FM-44, FM-96, FM-43, FM-45, FM-36, FM-37, FM-20,
FM-24, FM-69.

**Measured 2026-10-30** (Python 3.13.0, on the demo, `demo-sample` and
`FakeCore.demo_like()`; throwaway scripts):

- **Opening and closing the demo stand**:
  - `Microscope.open("demo")` on a worker thread takes 6 ms, then 2 ms to open
    again after a close; `demo-sample` takes 17 ms, then 8 ms;
  - the facade is then used from the main thread (a snap, a jog), and closed
    from a third thread;
  - a closed facade's `state()` has `xy=None` and the errors `xy:
    HardwareError: ... No device with label "XY"; check the device: power,
    cable, another program holding it` (and Z's), and `has(XYStage)` is still
    `True`;
  - a `FakeCore` facade, once closed, lists only `Core` among its loaded
    devices, which is what a test checks to see a release. Two `close()` calls
    are harmless.
- **The activities**, on a throwaway subclass of `PlatePage` with
  `disconnect`/`connect` built on `_start_activity`:
  - a disconnect then a connect each raise `activities_ended` by one;
  - a connect whose opener was still waiting when `close()` gave up (wait
    shortened to 0.2 s) released, once the open ended, the facade it opened,
    with the outcome above;
  - with the facade dropped, today's `status()` and `stop()` raise
    `AttributeError`, which §7.2 replaces.

**Not measured**: a stand opened on a worker thread. Until #264, the command
opened the stand on the main thread. A vendor adapter that cares which thread
loaded it is the owner's finding (§9.1).

## 8. The first run (#293)

Fixed by #293's plan (2026-11-02). The owner followed the plate-scan workflow on
`demo-sample` from `docs/ui/plate-page.md` and found it hard. **Decided with the
owner (2026-11-02)**, the four points that cost the most, which are #293's
acceptance criteria:

- **registering the plate** needed a terminal command with a JSON `refs`, or the
  7 minutes of `plate-finding`;
- **which tool comes next, with which values** (`coverage` 0.4, `detector` =
  `blobs`) had to be known and typed;
- **where the files are** (the run document, the report, the export) was known
  only from the terminal;
- **choosing the wells**: a click selects a well, and the *Wells* box does not
  show it.

Also decided with the owner: **on a stand whose profile has a `[sample]`, the page
registers the plate in one click from the sample's known geometry**, and says
that those positions come from the profile and are not measured. Captures, when
the operator made any, win.

The page gains a **next step**: the tool the run document needs next, the values
this stand suggests, why each one, its command, and one Run button. Nothing moves
without that click, and the button is `run(plugin, pairs)` with the pairs the
step shows, the call its command makes (principle 5). The guide is computed from
the run document's checklist, the page's selection and captures, and the
profile; it holds no state of its own and adds no region of its own to
ADR-0006's picture (principle 3): it is the head of the tool panel.

### 8.1 What the profile suggests (`[suggest]`, `smc/hardware/profile.py`)

Stand-specific knowledge goes into the profile. A new optional table, one entry
per plugin:

```toml
[suggest.scan-plan]
params = ["coverage=0.4"]
why = "a scan covers 40 % of each well, so that a few wells take seconds on the demo stage"

[suggest.cells]
params = ["detector=blobs"]
why = "the synthetic cells are bright round spots that the trained model does not see; on a real sample blobs are not cells"
```

```python
class SuggestSection(_Section):
    params: tuple[str, ...]
    why: str


class Profile(_Section):
    ...
    suggest: dict[str, SuggestSection] = Field(default_factory=dict)
```

- `params` are `smc run --param` pairs, as written on the command line: the
  suggestion **is** the command's call. The model checks their shape only, since
  `smc.hardware` imports no plugin. Each item has an `=` and a key with no
  surrounding space, else `each item of params is key=value, as smc run --param
  takes it; got '<item>'`. A key is given at most once, else `'<key>' given
  twice`. At least one item is required.
- `why` is one non-empty line (the page shows it on one line), else `why is one
  line that the page shows; give a non-empty text without a newline`.
- The table's keys are plugin names. Whether such a plugin exists and takes those
  keys is checked by the page (§8.3), not by the profile.
- `_allowed_keys` names `SuggestSection`'s keys for an unknown key in any
  `[suggest.<name>]`, so the refusal lists `params, why`.
- `[suggest]` is allowed on every stand, unlike `[sample]`. In this issue,
  `profiles/demo-sample.toml` gains the two entries above, with these texts.
  `demo.toml`, `Profile.demo()` and the Nikon profiles gain none: what a real
  stand suggests is the owner's call after a session there.

### 8.2 The sample's references (`smc/testing/sample_stand.py`)

```python
def sample_refs(section: SampleSection) -> tuple[WellRef, ...]: ...
```

These are the wells of `suggested_refs(section.plate)` (A1 and the two far
corners, the plugin's own choice: A1, A12 and H1 on 96 wells), in that order.
Each centre comes from the `useq.WellPlatePlan(plate, a1_center_xy=_a1_of(section),
rotation=section.rotation_deg)` that `scatter_cells` draws the cells in, so the
references and the pixels share one source of truth.

The function lives beside the sample, not in the page: it is the sample's truth,
and the controller imports it inside the step's code, only when the profile
has a `[sample]`. `smc.cli` already imports this module for `smc doctor`.

Measured 2026-11-02 (Python 3.13.0, a throwaway script): `calibrate` fitted back
the sample's A1 and rotation exactly, residual 0 to 1.5e-11 µm, nothing
assumed, for four cases:

- the default 96-well plate, centred;
- A1 at (−40 000, 25 000) µm, turned 3°;
- a 384-well plate turned −2.5° (A1, A24, P1);
- a 6-well plate (A1, A3, B1).

### 8.3 The next step (`smc/ui/guide.py`, new; `smc/ui/controller.py`)

```python
#: smc/ui/guide.py: no Panel, no device
SELECT_WELLS = (
    "select the wells to scan: click them on the map, or type a range in Wells "
    "(A1:D6, C*, all)"
)
CAPTURE_WELLS = (
    "centre a well under the objective (Live and the Drive pad), click it on the "
    "map and press Capture centre; a second well, far from the first, fits the "
    "rotation too"
)


@dataclass(frozen=True, slots=True)
class NextStep:
    plugin: str | None
    params: tuple[str, ...]
    why: tuple[str, ...]
    command: str
    blocked: str | None


def next_slot(checklist: Sequence[CheckItem]) -> CheckItem | None: ...
def command_line(
    plugin: str,
    params: Sequence[str],
    *,
    profile: str,
    run_doc: Path,
    dry_run: bool,
) -> str: ...
```

```python
#: smc/ui/controller.py
class PlateView:
    ...
    next_step: NextStep | None = None  # the last field


class ToolView:
    ...
    suggested: tuple[tuple[str, str], ...] = ()  # (key, value)
    suggested_why: str | None = None


class PlatePage:
    @property
    def run_doc(self) -> Path: ...
```

`guide.py` imports `CheckItem` from the controller only for typing (`if
TYPE_CHECKING`), and the controller imports `guide.py`.

**`next_slot`** is the first item of the checklist that is not `optional` and
whose `state` is not `ok`, else `None`. The optional slots (`camera-calibration`,
`timings`, a third party's) are never a next step: the workflow runs without
them.

**`view()` builds `next_step`** from the document it has just read, in
`PlatePage._next_step(checklist, blocked, cards)`:

1. **No slot left**: `NextStep(plugin=None, params=(), command="", blocked=None,
   why=("every stage of the workflow is done; what it wrote:", *files))`.
   `files` are the `files` lines (§8.4) of every card, in tool order.
2. **The tool**: the first tool of `tools()` (the stage order, then `More`)
   whose plugin `provides` the slot. None is registered: `plugin=None`,
   `why=("no registered plugin provides <slot>; smc run --list shows the
   plugins",)`, and the walk stops there.
3. **Why this tool**: the first `why` line is the item's `reason` when the slot
   is `unusable` (the run document's own words: `run it again`, another
   profile, stale, a dry run), else `the run document has no <slot> yet`.
4. **The pairs and the lines that say where each value comes from**, in this
   order:
   - **`plate-registration`**:
     - the page has captures (`refs` or `walls` not empty): the pairs are
       `page_params("plate-registration")`, and the line is `plate, refs, walls:
       the <n> centres and <m> walls captured on this page`;
     - no capture, and the profile has a `[sample]` whose `plate` is `self.plate`:
       the pairs are `plate=<plate>`, `refs=<json of sample_refs(...)>` and
       `walls=[]`, dumped as `page_params` dumps them. The line is `refs: the
       centres of <A1, A12, H1> where the profile's [sample] puts the plate; this
       stand's plate is synthetic, so they are known, not measured`;
     - a `[sample]` of another plate: `blocked` is `the profile's sample is a <p>
       plate: choose <p> under plate-registration`;
     - otherwise `blocked` is `CAPTURE_WELLS`.
   - **`scan-plan`**: `page_params("scan-plan")`, and the line is `wells: the <n>
     wells selected on the map`. With no selection, `blocked` is
     `SELECT_WELLS`. Any other `ValueError` of `page_params` (a map of another
     plate than the registered one) is `blocked`, with its message.
   - **A dry-run page** (§8.8.1): `blocked` is `DRY_RUN_NO_FRAME` for `scan`, and
     for a tool that reads a `scan` that was a dry run.
   - **The profile's suggestion** for the tool, when it is valid (below): its
     pairs follow, and its line is `<pairs joined by a space>: suggested by the
     profile <stand>: <why>`.
5. **Blocked by the run document or the stand**: `blocked[tool]` from the view
   (a read slot unusable, `this stand has no <Capability>`, the checkpoint
   refusal of #252, `the stand is not connected` of #264) wins over step 4's
   `blocked`: it is the reason that holds first.
6. **Another way**: for every other registered tool that provides the same
   slot, a last line `or <name>: <its summary>`. On the plate this is
   `plate-finding`.
7. **`command`** is `command_line(tool, pairs, profile=<profile>, run_doc=self.run_doc,
   dry_run=<dry run>)`. `<profile>` is `str(profile.source)` when the profile
   was loaded from a file, else `profile.microscope.name`.

**`command_line`** gives `smc run <plugin> -p <profile> --run-doc <run_doc>`,
then `--dry-run` when it is one, then `--param <pair>` for each pair. It is
quoted for a POSIX shell by `shlex.join`, so a JSON value is one
single-quoted word. How it pastes into PowerShell or `cmd.exe` is not verified
(§9.1), and a page served on Windows says so under the line (§8.8.6). The run
document's record of the run is the reference, whatever shell reads the line.

**`run_doc`** is `self._run_doc.absolute()`, which is the path as the command
was given it, made absolute against the server's working directory. Symbolic
links are not resolved.

**The suggestions are checked when the page is built**, against the registry,
and kept as `(key, value)` pairs and `why`. For each `[suggest.<name>]`:

| What is wrong | The `problems` line of `view()`, and the entry is ignored |
|---|---|
| No registered plugin is called `<name>` | `profile [suggest.<name>]: no plugin named <name>` |
| A key is not a field of its `Params` | `profile [suggest.<name>]: <key> is not a parameter of <name> (keys: <the fields>)` |
| A key that the page fills itself (`PAGE_PARAMS`) | `profile [suggest.<name>]: <key> is filled by the page` |

A value is not checked there: a value that its plugin refuses is `run()`'s
outcome, `<plugin>: ParamsError: ...`, which names the profile's mistake when
the step is run.

**The tool's form agrees with the step.** In `tools()`, a suggested key is a
`key` row, after `KEY_PARAMS` and before the required fields, even when
`KEY_PARAMS` does not name it: what the stand suggests is never folded under
*Advanced*. `ToolView.suggested` carries the pairs and `suggested_why` carries
`why`. On the page, the field of a suggested key **starts with the suggested
value**, so a click on the tool's own Run sends it, as the step does. The step
never reads the form. It runs its own pairs, so what the operator reads is what
runs (#287's lesson). The two differ only after the operator edits the form.

The profile, the stand's name and the dry run are the page's `_profile` and
`_dry_run` (§7.2): the step works with the stand closed, and says so through
`blocked`.

### 8.4 Where the files are (the result card, the run document)

```python
class ResultCard:
    ...
    stale: bool = False
    files: tuple[str, ...] = ()  # the last field
```

```python
#: smc/ui/controller.py
_FILE_KEYS = ("file", "folder")


def _card(
    record: StageRecord, *, stale: bool = False, run_dir: Path | None = None
) -> ResultCard: ...
```

- **Which keys are files.** A top-level key of a `done` result that `_FILE_KEYS`
  names, with a string value, is a file. The built-in plugins call them so:
  - `scan`'s `folder`;
  - the `file` of `cells`, `report` and `nis-export`.

  A third party that follows the convention is shown the same way. Such a key
  is not one of the card's `lines`; it is a `files` entry, `<key>: <path>`.
- **The path.**
  - An absolute value is kept as it is: `nis-export --param out=...` writes one.
  - A relative value is relative to the run folder (`ctx.run_dir`, the run
    document's folder), so it is shown as `str(run_dir / value)`, with
    `run_dir = self.run_doc.parent`.
  - With `run_dir=None` it is shown as it is written. That is for the tests that
    build `_card` alone.

  The path is shown, not opened or checked: a card reads the document only.
  Each entry is cut at `_CARD_TEXT_CHARS`, like an assumption. Only a `done`
  record has `files` (FM-154, as its `lines`).
- **`card_text`** shows the `files`, each indented by two spaces, after the
  `lines` and before `assumed`. A `done` card with none says `  in the run
  document` there, so **every result card says where its result is**.
- **The run document's path** heads the checklist: the bold `Run document` pane
  says `Run document  <page.run_doc>`.
- **A card wraps.** Every card pane, and the step's text, are `pn.pane.Str(...,
  styles={"white-space": "pre-wrap", "overflow-wrap": "anywhere"})`. A path or
  an assumption longer than the column now wraps, where #293's ui-check found
  `assumed: no camera calibration for Nikon 10X S Fluor: ro...` cut at the
  column's edge. It is still text, never markup (FM-139).

### 8.5 The page (`smc/ui/plate_page.py`)

```python
class PlatePageView:
    next_text: pn.pane.Str
    next_note: pn.pane.Str
    next_button: pn.widgets.Button

    def on_next(self) -> None: ...


def next_step_text(step: NextStep | None) -> str: ...
```

- **The region** is the head of the right-hand column, **above** *Drive*: a
  bold `Next step`, then `next_text`, `next_button` and `next_note` (in the
  warning colour). It is the first thing the column shows, and it is above the
  fold at any height where the column starts on screen.
- **`next_step_text`** is pure:
  - `""` for `None`;
  - with a tool: its name, then each `why` line indented by two spaces, then
    `  command: <command>` (and on Windows one more line, §8.8.6);
  - without a tool: the `why` lines, unindented.
- **`next_button`**:
  - its label is `Run <plugin>` (`label=`, `color="primary"`, FM-147);
  - `visible=False` when the step has no tool;
  - disabled when `blocked` is not `None`, or while an activity goes, as every
    Run button is (`_draw_buttons`);
  - `next_note` says `blocked`.
- **`on_next`** runs **the step drawn**: the page keeps the `NextStep` that its
  last redraw drew (`_next_drawn`), and the click calls
  `page.run(drawn.plugin, list(drawn.params))` through `_try`, then `tick()`. It
  never computes the step again at the click: the operator runs the pairs they
  read (#287). With nothing drawn, or a drawn step that has no tool, it says
  `nothing to run: press Refresh` and runs nothing. A click on the disabled
  button still reaches the server (FM-169), and `run()`'s own refusals are the
  guard.
- **The tool's stage opens.** When the drawn step's tool changes, the stage's
  `Accordion` card that holds the tool is added to the open cards (`active =
  sorted({*active, index})`). It never closes a card the operator opened.
- **Registered through `_handler`** (§3, FM-171), as every handler.

**The *Wells* box follows the selection** (the owner's fourth point).

- The view keeps `_wells_shown`, the selection that the box stands for.
- After a typed range is applied, `_wells_shown` is `page.selection`, and the box
  **keeps what was typed** (`A1:D6` stays `A1:D6`).
- At every redraw, when `page.selection` differs from `_wells_shown` (a click,
  another tab, a plate chosen, a run document read), the box is set to
  `",".join(page.selection)` (`""` for none), and `_wells_shown` follows.
- The box is set under a guard, `_echoing = True` around the assignment, and
  `on_select_text` returns at once while it is set. A box the operator empties
  clears the selection (§8.8.5).
- Measured on Panel 1.10.0 / Bokeh 3.10.1 / Python 3.13.0 and Panel 1.9.4 /
  Bokeh 3.9.2 / Python 3.10.10 (2026-11-02, a throwaway script under
  `warnings.simplefilter("error")`):
  - a `param.watch` on `value` is called **synchronously** by the assignment, so
    the guard skips it;
  - the same value typed twice calls the watcher once;
  - `pn.pane.Str(..., styles={...})`, `Button(label=, color="primary")` with
    `label` and `disabled` changed later, and `Accordion.active` set to a
    larger list all build with `get_root()` and warn nothing.

  What a browser does with a box changed while the operator types in it is not
  measured. The box is changed only by a redraw, which a click or the end of an
  activity starts, never by the tick.

### 8.6 What this does not do

- The next step does not run a tool by itself, chain tools, or choose the wells.
- It does not suggest a value from an earlier result. The route's `labels` keep
  their default (`single`): the cells' classes are the route's to choose, and
  `single` is what the plugin defaults to.
- #293 does not move the map (#297), the *Density of* menu or the wheel over the
  map, and it does not tell the page that a run from a terminal has ended (#251's
  refresh).
- The page does not open a file. It names the file, and the operator opens it
  from the PC.

### 8.7 Tests and failure modes

The tests are named in #293's plan. **Failure modes**: FM-98, FM-125 (a
profile's suggestion and a result's `file` taken as they come), FM-139, FM-145
and FM-151 (the step is built by `view()`, never by the tick), FM-147, FM-153
(the region looked at with `/ui-check`), FM-154, FM-161 (the Wells echo through
a session's watcher, not only a direct call), FM-169, FM-171, FM-43, FM-44,
FM-45, FM-36, FM-37.

**Measured 2026-11-02** (Python 3.13.0, `profiles/demo-sample.toml`, Micro-Manager's
demo devices, `smc run` from a terminal; nis-export through `run_plugin`
because the main checkout's venv had stale entry points), the whole chain with
the sample's three references:

| Step | Time |
|---|---|
| `plate-registration` | 0.5 s |
| `scan-plan` (`wells=B2,B11`, `coverage=0.4`) | 0.4 s |
| `scan` | 18.1 s |
| `cells` (`detector=blobs`) | 1.0 s |
| `density` | 0.3 s |
| `route` | 0.3 s |
| `report` | 0.3 s |
| `nis-export` (10 positions) | under 0.1 s |

Every step was `done`. The report is `report/<token>.html` and the export is
`nis-export/<token>.xml`, both under the run document's folder.

### 8.8 The first run's follow-ups (#310)

Fixed by #310's plan (2026-11-03). The owner's live walk of the first run
(`demo-sample`, Claude in Chrome, `main` at 9b9cc75, a newcomer's Connect and then the
next step each time) found what the first run still costs; its findings are
on #310. What it decides, in the order of their cost:

**8.8.1 A dry-run page does not lead to a dead end.** The documented newcomer
command has no `--dry-run` (`docs/ui/plate-page.md`); the page of the walk had it
(the guard of `/ui-check` and of a live walk), and there the step path ended in a
loop. Reproduced headlessly on `PlatePage.closed(demo-sample, run_doc,
dry_run=True)` on 2026-11-03 (Python 3.13.0): `plate-registration`, `scan-plan` and
`scan` are `done`; the scan's card says `(dry run)` and `assumed: dry run: 36
fields walked, no frame taken`; the step is then `cells`, not blocked, and `cells`
ends `PluginRunError: cells: the scan has no frame (it was a dry run, or every
field failed); run scan again without --dry-run`, after which the step is `cells`
again. That advice cannot be followed from the page, and in a terminal a run that
is not a dry run refuses what a dry run made (`scan reads scan-plan, but
scan-plan was a dry run`), so the whole chain starts again at `plate-registration`.

```python
#: smc/ui/guide.py
DRY_RUN_NO_FRAME = (
    "this page is a dry run: scan walks the fields and takes no frame, so cells and "
    "the tools after it have nothing to read. Serve the page without --dry-run on a "
    "stand whose stage is simulated (demo-sample), or run the chain with smc run "
    "from plate-registration, without --dry-run: a run that is not a dry run "
    "refuses what a dry run made"
)
```

- In `PlatePage._next_step`, step 4 gains one row, after the `scan-plan` one and
  before the profile's suggestion: when the page is a dry run, `stuck` is
  `DRY_RUN_NO_FRAME` for the tool `scan`, and for any tool whose plugin `reads`
  `scan` while `cards["scan"]` exists and is a dry run (`ResultCard.dry_run`). A
  page that reads a `scan` of a terminal run without `--dry-run` is not stuck: a
  dry run may read a real result. Step 5 is unchanged: `blocked[tool]` wins.
- The step stops advising; nothing is taken away. The tool's own *Run* button, and
  so a dry-run `scan` to see the walk, stay as they are. `command:` keeps
  `--dry-run`, as it is the page's call.
- **By the plugin's name, not by a capability.** Measured 2026-11-03: `scan` and
  `scan-plan` both `require` `Camera` (the second for the pixel size) and
  `scan-plan` computes the same in a dry run, so "requires a camera" does not mean
  "takes no frame". That `scan` takes none in a dry run is the plugin's own fact
  (`scan/plugin.py`: `frames=() if ctx.dry_run`), and `_next_step` already names
  `plate-registration` and `scan-plan` where the page's values differ.
- The next step of a dry-run page that read a real chain (the walk's Refresh after
  the terminal chain) is the ordinary one: `route`, then `report`, `nis-export`.

**8.8.2 The strip says what a run assumed.** A dry-run `scan` ended `scan done in
0.0 s`, and the run document held `assumed: dry run: 36 fields walked, no frame
taken`, which only its card showed. In `PlatePage._run_plugin`, a record that ended
`done` gives the outcome `<plugin> done in <s> s` followed by
`; assumed: <first>` and, with more than one, a space and `(+<n> more, on the card)`.
The lines are `_card(record).assumed` (flat, bounded, never markup: FM-125, FM-139), the
first one cut at `_STRIP_ASSUMED_CHARS = 160`. A record with none has the outcome
it has today. A stopped or failed one never has them (FM-154). Measured on the same
chain: `scan-plan` has `assumed: no camera calibration for Nikon 10X S Fluor:
rotation 0 and no mirror assumed`, so the strip says that too.

**8.8.3 The strip no longer keeps an old error after a Refresh.** The outcome of
the last activity stays on the strip until the next activity replaces it, so that
`cells: PluginRunError: ... run scan again without --dry-run` was still there after
the chain had been run from a terminal and the page refreshed. `PlatePage.clear_outcome()`
sets `_outcome` to `None` under `_lock` when no activity goes (`_current_run is
None`), and does nothing otherwise. **Only the *Refresh* button** calls it, then
`refresh()` and `tick()`: `refresh()` itself is called by every other handler and by
the tick when a run ends, and must not drop the outcome it is about to show. The
stop note and the drive's kept outcome are not touched.

**8.8.4 The plate card says where its references came from.** The step says `known,
not measured` before the click; afterwards the card read `method: references`,
`residual_um: 0` and no `assumed:` line, because the plugin was given three
references and cannot tell a synthetic plate's from a measured one. In `view()`,
after the cards are built, the card of `plate-registration` gets one more
`assumed` line, last, when **all** of these hold: the profile has a `[sample]`; the
record is `done`, was made on this stand (`record.profile` is the stand's name) and
its `params["refs"]` are the sample's own: the same wells as `sample_refs(sample)`,
each centre within 0.01 µm (the record is read as it comes, FM-125: anything that is
not a list of objects with a text `well` and numbers `x_um` and `y_um` is not the
sample's). The line is the step's own, so that one text says it twice:

```text
refs: the centres of A1, A12, H1 where the profile's [sample] puts the plate; this stand's plate is synthetic, so they are known, not measured
```

The text is built by one function that `_registration_step` and the card share.
`smc.testing.sample_stand` is imported inside the function and only when
`profile.sample` is not `None`, as in §8.2: a real stand never loads it. Captures
the operator made are measured and read differently from the sample's, so they
never match.

**8.8.5 The Wells box emptied clears the selection.** A box the operator empties
changed nothing (`on_select_text` returned at once on an empty text, because
`select_text("")` raises `no wells selected`, measured 2026-11-03), and the box then
stood for a selection it no longer showed until the selection changed. Now
`PlatePage.select_text` takes a blank text (empty or only white space) as "no
wells": `selection` becomes `()` and nothing is raised. `on_select_text` no longer
returns on a blank text; the `_echoing` guard alone stops the echo of the box's own
assignment. After a blank text `_wells_shown` is `()`, so the next click echoes
again. A text that does not parse (`Z99`) keeps its refusal and the old selection.

**8.8.6 The command on Windows.** FM-175: `shlex.join` is POSIX quoting; no
Windows shell was at hand (this session was on macOS), and `cmd.exe` and
PowerShell 5.1 read quotes differently from each other, so no one quoting is right for
both. The page does not guess one; it says so. `next_step_text` gains a keyword:

```python
def next_step_text(step: NextStep | None, *, windows: bool | None = None) -> str: ...
```

`windows=None` is `sys.platform == "win32"`; a test passes it. With a tool and
`windows` true, one more line follows the `command:` line, indented like it:

```text
  that line is quoted for a POSIX shell (bash, zsh); PowerShell and cmd.exe read quotes differently: press the button, which makes the same call
```

`command_line` and `NextStep` do not change here (§10.5 gives `command_line` a
`preset` keyword, a separate change). A paste that fails fails loudly; the
line says why before it does. What the PC's shells do with it stays in §9.1.

**8.8.7 What this does not do.** The map does not draw the scan plan's fields
(finding 4, #320); a file's path is still text, not a link (§8.6 decided it); the
`dry_run: true` that `route`, `report` and `nis-export` record while writing complete
files is harmless today and is not changed; Bokeh's `AssertionError` in
`on_close` seen once at a closing tab is not reproduced and not handled here.

**8.8.8 Tests and failure modes.** The tests are named in #310's plan.
**Failure modes**: FM-98, FM-125, FM-139, FM-145 and FM-151 (the step and the cards
are built by `view()`, never by the tick), FM-154, FM-161 (the blank box and the
Refresh button reach their handler through a session), FM-169, FM-171, FM-175,
FM-176, FM-43.

## 9. Sequencing

1. **#28** first and alone: it fixes the one-activity rule, `Frame` and the
   regions the others fill.
2. Then **#262** and **#263**, one after the other (both edit `controller.py`
   and `plate_page.py`), and **#264** after them. **#262 carries the code of
   §2.4's stale mark and bounds** (#272), being the first of the two to edit
   `controller.py`.
3. **#251** and **#252** edit the same modules: never at the same time as one of
   the above.
4. **#264** after #277, which edits `plate_page.py`. It touches the
   controller's every read of the facade, so no other issue that edits
   `controller.py`, `plate_page.py` or `smc ui` in `cli.py` runs at the same time:
   #251, #287, #293, #295 and #297 come before it or after it.
5. **#293** after #264 and #252 (PR #301): it reads #264's `_profile` and
   `_dry_run`, and its step carries #252's checkpoint refusal. It edits
   `controller.py`, `plate_page.py`, `profile.py` and `sample_stand.py`, so
   #251, #287, #295 and #297 run before it or after it, never alongside.
6. **#310** after #293 (PR #309) and #317: it edits `controller.py`, `plate_page.py`
   and `guide.py` (§8.8), so no other issue that edits them runs at the same time;
   #320 (the scan plan on the map) comes after it.
7. **#319** (presets, §10) edits `controller.py`, `plate_page.py`, `guide.py`,
   `cli.py`, `profile.py`, `document.py` and `runner.py`: no other issue that
   edits one of them runs at the same time, #310 included (one before the other,
   never alongside).

### 9.1 Not verified (for the owner)

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
- **Drive's follow-ups (#295)**: *Light off* clicked behind a frame, then STOP, on
  the Ti2-E's `DiaLamp`: the light goes off after the stop and the page says so; a
  turret whose positions share a label, or two empty ones, if Micro-Manager allows
  them (the select then shows one option for both, §6.4); `smc objective set '""'`
  in Windows PowerShell 5.1 against an empty position (§6.2 states PowerShell's
  documented behaviour; it was not run here, nor on any PC).
- A newcomer registers a plate on the `demo-sample` stand from the page, without
  a manual (#28's last criterion), with a lab member.
- **The service on a microscope PC** (§7.4):
  - under NSSM in session 0: whether the vendor SDK (Ti2 SDK, PVCAM, Zeiss MTB)
    and the camera driver open there, under the account chosen;
  - Connect, then Disconnect, then NIS-Elements taking the stand, then Connect
    again;
  - `nssm stop` while a run goes: the Ctrl-C reaches Python, and the stand is
    released within `AppStopMethodConsole`;
  - a reboot and a logoff;
  - how long Windows lets the service stop at shutdown;
  - whether the lab's NSSM version names its rotation parameter
    `AppRotateFiles`.
- **A stand opened on the worker thread** that Connect starts and that then ends,
  where the command used to open it on the main thread.
- **The first run (§8)**:
  - a newcomer follows the next step on `demo-sample`, from Connect to the
    export, in a browser, with no terminal and no documentation open;
  - the step's command pasted into PowerShell and `cmd.exe` (it is quoted for a
    POSIX shell; the page says so on Windows, §8.8.6, and what the Windows shells
    do with it is still not run);
  - the *Wells* box set by a redraw while the operator types in it, and emptied
    by the operator in a real browser (§8.8.5);
  - the wrapped cards and the step's region on the PC's screen and over RDP;
  - the owner's walk with Live, Drive, the keyboard alone and a narrow window
    (the walk of #310 pressed none of them);
  - the strip's `assumed:` tail and the Refresh that clears the outcome (§8.8.2,
    §8.8.3) looked at in a browser: `/ui-check` never presses a Run button.
- **Presets (§10)**:
  - the owner's own assay saved, then loaded on another day, at a stand;
  - a preset saved from the service (§7.4) into the profile's folder: whether
    the service's account may write there;
  - a preset file edited in Notepad (a BOM, CRLF line ends) and read back on
    Windows.

## 10. Presets (#319)

Fixed by #319's plan (2026-11-03), from the owner's feedback: the page asks for
the same choices at every run, and an assay repeated every week is typed again
each time.

A **preset** is the set of choices an assay repeats, named and kept in a TOML
file: the plate type, the wells, and the plugins' `--param` pairs. It is a
convenience. **The run document stays the truth**: each record keeps the
values the run used (`params`), and names the preset that was loaded (§10.4).
A preset never holds a position (`refs`, `walls`, a position list) and never
moves anything.

### 10.1 The file (`smc/plugins/presets.py`, new)

```toml
# A preset of smc: the choices an assay repeats. See docs/ui/plate-page.md, Presets.
stand = "demo-sample"
plate = "96-well"
wells = "B2,B11"
saved_utc = 2026-11-03 10:00:00+00:00

[params]
scan-plan = ["coverage=0.4"]
cells = ["detector=blobs"]
```

```python
class Preset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stand: str | None = None
    plate: str | None = None
    wells: str | None = None
    saved_utc: AwareDatetime | None = None
    params: dict[str, tuple[str, ...]] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PresetFile:
    name: str
    path: Path
    preset: Preset | None
    error: str | None = None


class PresetError(Exception): ...


class PresetExistsError(PresetError):
    path: Path


def check_name(name: str) -> str: ...
def preset_dirs(profile: Profile) -> tuple[Path, ...]: ...
def list_presets(profile: Profile) -> tuple[PresetFile, ...]: ...
def load_preset(source: str, profile: Profile) -> PresetFile: ...
def dumps_preset(preset: Preset) -> str: ...
def save_preset(
    name: str, preset: Preset, profile: Profile, *, overwrite: bool = False
) -> Path: ...
```

The module imports no Panel and reaches no device: the command and the page
share it.

**The fields.**

- `stand` is the profile's `microscope.name` where it was saved.
- `plate` is a `useq` plate key.
- `wells` is the text `scan-plan`'s `wells` takes (`A1:D6`, `C*`, `B2,B11`).
- `params` holds, per plugin name, `smc run --param` pairs, as `[suggest]`
  does (§8.1): the preset *is* the command's call.
- Every field may be left out.
- `stand`, `plate` and `wells`, when given, are one non-blank line, else `<field>
  is one non-empty line; leave it out for none`.
- Each list of `params` has the shape `[suggest]`'s `params` has, with the same
  texts. That check moves out of `SuggestSection` into a public
  `check_pairs(items: Sequence[str]) -> None` of `smc/hardware/profile.py`,
  which both call. The one difference is the list's length: an empty list is
  allowed here and is not in `[suggest]`.
- An unknown key is refused (`extra="forbid"`), as in a profile.

**The name** is the file's stem.

- `check_name` accepts 1 to 64 characters, lower-case letters, digits, `-` and
  `_`, starting with a letter or a digit (`[a-z0-9][a-z0-9_-]{0,63}`). Any other
  name raises `PresetError("a preset's name is 1 to 64 lower-case letters,
  digits, - and _, starting with a letter or a digit; got '<name>'")`.
- A name whose upper case is in `RESERVED_NAMES` (`smc.plugins.scan.algorithm`)
  raises `PresetError("'<name>' is a name Windows keeps for a device; choose
  another")`.
- Lower case only, so two names never differ by case alone, and NTFS and APFS
  see two presets where the code sees two (FM-24).

**Where the presets are.** `preset_dirs(profile)` is, in this order, without
duplicates:

1. `<profile.source's folder>/presets`, when the profile was loaded from a file;
2. `<d>/presets` for each `d` of `search_paths()` (`$SMC_PROFILES`, `./profiles`,
   `.`).

Duplicates are compared as `os.path.normcase(os.path.abspath(d))`. A preset of
the stand sits beside its profile, and `profiles/presets/` serves every stand
of the checkout. A missing folder is skipped when presets are read.

- **`list_presets`** reads every `*.toml` of each folder, sorted by name. The
  first file of a name wins, as for profiles. A file that cannot be read, is not
  a valid preset, or whose stem `check_name` refuses is listed with `preset=None`
  and its `error`, never dropped: a failure is a finding.
- **`load_preset(source, profile)`** reads a path as given, by the profile's rule:
  a string that ends in `.toml` or contains `/` or `os.sep`. Its name is the
  stem. Any other source is a name: `check_name`, then the first
  `<dir>/<name>.toml` of `preset_dirs`. It always returns `preset` set; every
  failure is a `PresetError` whose message is the whole diagnosis, on one line:
  - none found: `no preset named <name>; looked in <dirs, joined by ", ">; smc
    presets list shows the presets`;
  - unreadable: `<path>: cannot be read: <strerror>`;
  - not TOML: `<path>: not TOML: <message>`;
  - invalid: `<path>: <loc>: <msg>`, the pydantic errors joined by `; `.

  The text is read as `encoding="utf-8-sig"`: a BOM that Notepad writes is not
  an error.
- **`dumps_preset`** is a header comment, `# A preset of smc: the choices an
  assay repeats. See docs/ui/plate-page.md, Presets.`, then
  `tomli_w.dumps(preset.model_dump(mode="python", exclude_none=True))`.
  `tomli-w>=1.0` becomes a dependency: nothing in the project writes TOML today.
- **`save_preset`** checks the name, then looks for the name as `load_preset`
  does. When one is found and `overwrite` is false, it raises
  `PresetExistsError` with that `path`: `preset <name> exists: <path>`. It then
  writes `preset_dirs(profile)[0] / f"{name}.toml"` through `write_atomically`
  (`smc.run.files`, FM-38, its folder made if missing), as UTF-8 without a BOM,
  and returns the path.
  - An `AtomicWriteError` is a `PresetError`, `<path>: not saved: <strerror>`,
    with `; the new preset is in <kept>` when only the rename failed.
  - An overwrite of a preset found in a later folder writes the first folder's
    file, which then shadows the other. The message of `PresetExistsError` names
    the file that exists, and the page says where it saved (§10.6).

Measured 2026-11-03 (a throwaway prototype of `Preset` and `dumps_preset`,
tomli-w 1.0.0 on Python 3.13.0 with pydantic 2.13.5, and on Python 3.10.10 with
tomli 2.0.1 and pydantic 2.7.0):

- the file above, a key `my.plugin`, a pair with `"`, `é`, `U+007F` and an emoji,
  and an empty list all came back equal through `tomllib.loads`;
- the same text behind a BOM read back equal through `utf-8-sig`;
- an unknown top-level key was refused;
- tomli-w writes a quoted key only where TOML needs one (`"my.plugin"`) and the
  datetime as `2026-11-03 10:00:00+00:00`. The page saves `saved_utc` to the
  second.

### 10.2 A preset checked against a stand

```python
NOT_IN_PARAMS: dict[str, dict[str, str]] = {
    "plate-registration": {
        "plate": "the preset's plate holds it",
        "refs": "a preset holds no positions",
        "walls": "a preset holds no positions",
    },
    "scan-plan": {"wells": "the preset's wells hold them"},
}


@dataclass(frozen=True, slots=True)
class PresetCheck:
    name: str
    plate: str | None
    wells: str | None
    params: dict[str, tuple[str, ...]]
    mismatches: tuple[str, ...]


def registered_plate(doc: RunDocument) -> str | None: ...
def check_preset(
    file: PresetFile,
    models: Mapping[str, type[BaseModel]],
    *,
    stand: str,
    registered: str | None,
    only: str | None = None,
) -> PresetCheck: ...
def command_pairs(
    check: PresetCheck, plugin: str, overrides: Sequence[str]
) -> list[str]: ...
```

**A mismatch is named, never refused silently.** `check_preset` gives what
applies and one `mismatches` line for everything else. `models` maps a plugin's
name to its `Params`. With `only`, only that plugin's pairs are checked and
returned, and the other plugins of the file are not mismatches (the command
runs one plugin, §10.3).

| What is wrong | The line | What is not applied |
|---|---|---|
| `stand` differs from `stand=` | `preset <n> was saved on <stand>; this stand is <stand>` | nothing: it is a note |
| `plate` is not a `useq` plate | `preset <n>: <plate> is not a plate useq knows` | `plate` and `wells` |
| `registered` is set and differs from `plate` | `preset <n> is for a <plate> plate; this run document's plate is registered as a <registered>: its plate and wells are not applied` | `plate` and `wells` |
| a plugin that `models` does not hold | `preset <n> [params.<plugin>]: no plugin named <plugin>` | its pairs |
| a key that is not a field of its `Params` | `preset <n> [params.<plugin>]: <key> is not a parameter of <plugin> (keys: <fields, joined by ", ">)` | that pair |
| a key that `NOT_IN_PARAMS` names | `preset <n> [params.<plugin>]: <key> is not kept under params: <reason>` | that pair |

`PresetCheck.plate` and `wells` are the preset's when applied, else `None`.
`params` keeps the applied pairs in the file's order, and a plugin left with
none is not a key. A value is not checked: a value its plugin refuses is that
run's `ParamsError`, as for `[suggest]`.

The difference from `[suggest]` (§8.3) is deliberate. A suggestion with a bad
key is ignored whole. A preset drops only the pair: a plugin that renamed one
parameter keeps the rest of the assay.

**`registered_plate(doc)`** is the plate this run document has registered:

- it reads the `done` records that provide the slot `plate`
  (`plate-registration`, `plate-finding`);
- among them, the one with the latest `finished_utc`;
- the value is its `result["plate"]` when that is a string, else `None`;
- `None` when there is none.

It reads the document only.

**`command_pairs(check, plugin, overrides)`** gives the pairs the command line
runs, in this order:

1. The preset's held field:
   - `plate=<check.plate>` for `plate-registration`;
   - `wells=<check.wells>` for `scan-plan`;
   - each when it is applied.
2. Then `check.params[plugin]`.
3. Then `overrides`.

A key that an override gives is dropped from 1 and 2, so `--param` wins. The
key of an item is its text before the first `=`. A malformed override is
passed on as it is, and `parse_params` names it.

### 10.3 The command line (`smc/cli.py`)

```
smc run NAME -p PROFILE --run-doc PATH [--preset NAME|PATH] [--param k=v ...] [--dry-run] [--resume]
smc presets list [-p PROFILE]
smc presets show NAME|PATH [-p PROFILE] [--run-doc PATH]
```

**`smc run --preset`**. Without `--preset`, `run_command` is unchanged and in
its order. With it, the order is:

1. `load_plugin`.
2. `Profile.load`.
3. `load_preset`. A `PresetError` is `typer.BadParameter(<message>,
   param_hint="--preset")`, which exits 2.
4. `_check_run_doc`.
5. `RunDocument.load_or_new`.
6. `check_preset(file, {name: entry.plugin.Params}, stand=<microscope.name>,
   registered=registered_plate(doc), only=name)`.
7. Each mismatch is printed with `_say`, before the run starts.
8. `parse_params(Params, command_pairs(check, name, param or []))`.
9. The rest as today.
10. `run_plugin(..., preset=file.name)`.

Only the plugin run is loaded: the command does not import every plugin to
check a preset.

**`smc presets`** is a Typer group, `presets_app`, attached as `presets`, like
`objective_app`.

- `list` prints one row per preset, aligned by `_aligned`: `<name>  <stand or
  ->  <plate or ->  <plugins, joined by ",">  <path>`. A broken file's row is
  `<name>  cannot be read: <error>  <path>`.
  - Then `searched: <dirs, joined by ", ">`.
  - With none, the output is `no presets; searched: <dirs>`.
  - Exit 0.
- `show` prints the preset:
  - its first line is `preset <name>  <path>`;
  - then `stand`, `plate`, `wells` and `saved_utc`, one per line;
  - then `<plugin>: <pairs, joined by a space>` per plugin.

  It then checks the preset against the whole registry (`discover()`), the
  profile's stand, and `registered_plate` of `--run-doc` when it is given (a
  missing file registers nothing). After that come the mismatches, or `no
  mismatch on <stand>`. Exit 0. A `PresetError` is `_fail(<message>)` and exit
  2 (`_EXIT_REFUSED`).

### 10.4 The run document (`smc/run/document.py`, `smc/plugins/runner.py`)

```python
class StageRecord(BaseModel):
    ...
    error: str | None = None
    preset: str | None = None  # the last field
```

- `preset` is the name of the preset the run was given: `--preset`, or the one
  loaded on the page (§10.5).
- The record's `params` are the values used. When the operator edits a field
  after loading, the record keeps the preset's name and the edited value.
- A `model_serializer(mode="wrap")` drops the key when it is `None`, so a
  record without a preset is written exactly as before.
- `SCHEMA_VERSION` stays 1:
  - an older smc reads every document that names no preset;
  - an older smc refuses a record that names one, by its unknown key `preset`.
- `run_plugin(..., preset: str | None = None)` writes it into **every** record
  the run writes:
  - the first;
  - its checkpoints;
  - the last.

Measured 2026-11-03 with the prototype, pydantic 2.13.5 and 2.7.0:

- the wrap serializer drops `preset: None`, inside a document's `stages` too;
- it keeps `preset: "weekly"`;
- a model without the field (an older smc) reads the record that has none;
- `model_json_schema()` still lists the field;
- `mypy --strict` passes on the serializer's signature,
  `(self, handler: SerializerFunctionWrapHandler, info: SerializationInfo) ->
  dict[str, Any]`.

### 10.5 The controller (`smc/ui/controller.py`)

```python
@dataclass(frozen=True, slots=True)
class LoadedPreset:
    file: PresetFile
    check: PresetCheck
    fields: dict[str, dict[str, str]]
    selection: tuple[str, ...] | None
    mismatches: tuple[str, ...]


class PlatePage:
    @property
    def preset(self) -> LoadedPreset | None: ...
    def presets(self) -> tuple[PresetFile, ...]: ...
    def load_preset(self, source: str) -> LoadedPreset: ...
    def forget_preset(self) -> None: ...
    def save_preset(
        self,
        name: str,
        fields: Mapping[str, Sequence[str]],
        wells: str,
        *,
        overwrite: bool = False,
    ) -> LoadedPreset: ...
```

The loaded preset is the page's shared state, as the plate and the selection
are. Every tab sees it, and it is kept under `_lock`.

**`presets()`** is `list_presets(self._profile)`.

**`load_preset(source)`**:

1. `load_preset(source, self._profile)`. A `PresetError` propagates, and the
   view shows its message.
2. `check_preset` against every registered plugin's `Params`, with
   `stand=self._profile.microscope.name` and `registered=registered_plate(<the
   run document>)`.
3. The page's own mismatch: when the check applies a plate that is not
   `self.plate` while the page holds captures (`refs` or `walls`), the line is
   `preset <n> is for a <p> plate; this page has captures on a <q> plate: its
   plate and wells are not applied`, and neither is applied. `choose_plate`
   would drop the captures.
4. An applied plate is `choose_plate(plate)`.
5. Applied wells are `select_text(wells)`. Its `ValueError` is the mismatch
   `preset <n>: wells <wells>: <message>`, and the selection stays as it was.
   `selection` is the selection the load set, else `None`.
6. `fields` is `check.params` split at each pair's first `=`.
7. `mismatches` are the check's, then the page's.

It needs no stand and is allowed while an activity goes: it reaches no device,
and a run in flight has already read its pairs.

**`forget_preset()`** sets it back to `None`.

**`run()`** passes `preset=self.preset.file.name` to `run_plugin` when a preset
is loaded. The next step's command says it too, below.

**`save_preset(name, fields, wells, *, overwrite=False)`**.

`fields` are the tools' non-empty fields, as the view's `on_run` collects them,
per plugin. Each plugin's pairs go through `parse_params` against its `Params`.
A `ParamsError` is `PresetError("<plugin>: <problems, joined by "; ">; nothing
saved")`, and a plugin that is not registered is `PresetError("no plugin named
<plugin>; nothing saved")`. A preset that its next run would refuse is never
written.

`wells` is the *Wells* box's text:

- with no selection, the preset has no `wells`;
- when the text, stripped, names exactly the selection on this plate
  (`parse_wells(self.plate, text) == self.selection`), it is kept as typed
  (`A1:D6` stays `A1:D6`);
- otherwise the preset's `wells` is `",".join(self.selection)`.

The preset is `Preset(stand=self._profile.microscope.name, plate=self.plate,
wells=..., saved_utc=<now, UTC, to the second>, params=<the pairs of each plugin
that has any>)`.

- `save_preset(name, preset, self._profile, overwrite=overwrite)` writes it.
  `PresetExistsError` propagates: the view asks before overwriting.
- What it saved becomes the loaded preset: the method returns
  `self.load_preset(str(path))`. Loading it again is a no-op on the page: same
  plate, same selection.

**The next step** (§8.3, step 4) takes the loaded preset as its suggestion when
the preset names the tool:

- the pairs are the page's pairs;
- then the profile's suggested pairs whose key the preset does not give;
- then the preset's pairs for the tool.

The why lines are:

- the profile's line, with only the pairs that remain, and none when none
  remain;
- then `<pairs, joined by a space>: from the preset <name>`.

`command_line` gains `preset: str | None = None`, and gives `--preset <preset>`
after `--dry-run` and before the pairs. The step passes `str(file.path)`, as it
passes the profile's path. Run from a terminal, the command then applies the
preset, whose pairs the `--param` pairs repeat, and its record names the preset
as the page's would.

### 10.6 The page (`smc/ui/plate_page.py`, `smc/ui/guide.py`)

```python
class PlatePageView:
    preset_select: pn.widgets.Select
    preset_load_button: pn.widgets.Button
    preset_name: pn.widgets.TextInput
    preset_save_button: pn.widgets.Button
    preset_confirm: pn.Row
    preset_note: pn.pane.Str

    def on_load_preset(self) -> None: ...
    def on_save_preset(self) -> None: ...
    def on_preset_overwrite(self) -> None: ...
    def on_preset_cancel(self) -> None: ...
```

```python
#: smc/ui/guide.py: no Panel, no device
def preset_differences(
    expected: Mapping[str, Mapping[str, str]],
    actual: Mapping[str, Mapping[str, str]],
) -> tuple[str, ...]: ...
def preset_note_text(
    loaded: LoadedPreset | None, differences: Sequence[str]
) -> str: ...
```

**The region** is a bold `Preset`, at the head of the right-hand column, above
*Next step*. A preset is what the operator chooses before the first step.

- One row: `preset_select` (`label="Preset"`, options `{"(none)": "", <name>:
  <name>, ...}` from `presets()`, the broken files left out), then
  `preset_load_button` (`label="Load"`).
- A second row: `preset_name` (`label="Save as"`, `placeholder="weekly"`), then
  `preset_save_button` (`label="Save"`).
- Then `preset_confirm`: a text, then `Overwrite` and `Cancel`, hidden unless
  armed, as `objective_confirm` is.
- Then `preset_note`, wrapped text as the cards are (§8.4), never Markdown
  (FM-139).
- The options are read at `refresh()` and after a save, never by the tick.

**What Load sets** (`on_load_preset`):

- With `(none)`, the button calls `forget_preset()` and the note says `no preset
  loaded`.
- Otherwise it calls `page.load_preset(name)` through `_try`.
- Then it sets **every** field of every tool on the page to the preset's value,
  else the profile's suggested value (`ToolView.suggested`), else `""`. The form
  shows what a fresh page with this preset shows. A field typed before the load
  is replaced: loading is choosing the preset's values.
- These expected values are kept in `_preset_expected`, with the loaded preset.
- `preset_name` takes the preset's name, so that a Save after edits asks to
  overwrite it.
- Then `refresh()`, which draws the plate, the *Wells* box and the next step.
- The fields are set under a guard, `_filling`, so the field watchers below draw
  the note once, after the last.

**The note** is `preset_note_text(page.preset, differences)`:

- `no preset loaded`;
- or `loaded: <name>  (<path>)`, then each mismatch indented by two spaces, then
  either `changed since loading:` with each difference indented, or `the page
  holds the preset's values`;
- the mismatches and the differences are each cut at 8 lines, then `… <n> more`,
  as a card's lines (§2.4).

`preset_differences` compares each expected field with the field's value,
stripped. It gives `<plugin> <key>: <value> (the preset: <expected>)` per field
that differs, in tool order, with `(default)` for an empty text. The view adds
the page's own two lines:

- `plate: <plate> (the preset: <p>)` when the loaded check applied a plate that
  is not `page.plate`;
- `wells: <n> selected (the preset: <wells>)` when the load set a `selection`
  and `page.selection` differs from it.

**When the note is drawn**:

- at every redraw;
- from a watcher on the `value` of every tool field, of the *Wells* box and of
  `plate_select`;
- each watcher registered through `_handler` (FM-171) and proved through a
  session, not only by a direct call (FM-161).

Another tab keeps its own fields: its note names the loaded preset and lists
how its fields differ, which is true of what its Run sends.

**Save** (`on_save_preset`):

- An empty name says `type a name to save the preset under` and saves nothing.
- Otherwise it calls `page.save_preset(name, <the non-empty fields per tool, as
  on_run collects them>, self.select_input.value)`.
- `PresetExistsError` arms the confirmation: `_preset_armed = (name,)`, the text
  `<its message>: overwrite it?`, and `preset_confirm` shown.
- A save that worked:
  - refreshes the options and selects the saved name;
  - sets `_preset_expected` to the fields as they are;
  - says `saved <name> to <path>`, then the note.
- Any other `PresetError` is the page's message, through `_try`.

**The confirmation**:

- `on_preset_overwrite` saves the armed name with `overwrite=True`. It collects
  the fields at that click, which are what the operator sees then.
- When `preset_name` no longer holds the armed name, the click saves nothing and
  says `the name changed: press Save again`.
- `on_preset_cancel` and any change of `preset_name` disarm.

**Every button** is registered through `_handler` (§3, FM-171). Its label is
given as `label=` (FM-147). None is disabled while an activity goes: no button
here reaches the stand.

### 10.7 What this does not do

- **No positions**, and nothing that moves.
- **No exposure or channel.** The page offers neither as a plugin parameter:
  `scan`'s `Params` has none. The objective a scan is planned for is
  `scan-plan`'s `objective`, which a preset holds like any field. *Drive*'s
  objective select, the light and the autofocus are actions, not parameters.
- **No `smc presets save` in this issue.** The CLI's equivalent of *Save* is,
  for now, the file written by hand, in the documented format. A command that
  saves a run document's parameters as a preset is a follow-up issue.
- **No *Delete* button**: a preset is deleted as a file.
- **No `smc ui --preset`**: the page starts with no preset loaded.
- **The profile's `[suggest]` is unchanged.** A loaded preset wins over it, key
  by key.

### 10.8 Tests and failure modes

The tests are named in #319's plan. **Failure modes**:

- FM-24: a name, a Windows device name, case, the folders;
- FM-38: the write, through `write_atomically`;
- FM-98, FM-125 (a preset's values taken as they come), FM-139;
- FM-147, FM-151 (the options and the note are never drawn by the tick);
- FM-153 (the region looked at with `/ui-check`);
- FM-161, FM-169, FM-171;
- FM-36, FM-37 and FM-72 (the commands' output and help);
- FM-43, FM-44, FM-45.
