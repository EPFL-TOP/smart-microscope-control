# M5 design — the web UI's shell

- **Status**: design for issue #28 (§1 to §4, 2026-10-29), as built by #270 and
  settled by #272 (§2.1, §2.4, §3: the stale card, the card's bounds,
  `card_text`'s signature); live fixed by #262's plan (§5, 2026-10-29); drive
  fixed by #263's plan (§6, 2026-10-30); the stand's lifetime and the service
  fixed by #264's plan with the owner (§7, 2026-10-30); the first run fixed by
  #293's plan with the owner (§8, 2026-11-02); the drive's follow-ups fixed by
  #295's plan (§6.1 to §6.4, 2026-11-02: a switch-off at the end of live, the
  outcome live keeps, the captures' refusal, the readbacks from the state, the
  objective select's sync); presets fixed by #319's plan (§10, 2026-11-03);
  the plate page at the stand fixed by #332's plan (§11, 2026-11-05: the captures'
  target, why a Run is disabled, the plate the page starts with, what a Run
  replaces, a Ctrl-C that prints one line); the image's sides and what a move does
  to the image fixed by #327's plan (§13, 2026-11-09: the captures and the drive
  take image sides, the notes say what each is on the stage); the map's place of
  the stage fixed by #410's plan (§15, 2026-11-10: a cross that follows the frames
  in Live, a plate estimated from the stand's holder, the pose the title names).
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
  known here: a stand where `↑` moves the image down is the owner's finding (§13
  says it, from the camera calibration).
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

**Decided in #305's plan (2026-11-03)**, after the review of #264's PR:

- **A Disconnect whose `close()` raised `MicroscopeBusyError` keeps the facade**,
  so that the page tells the truth (the stand is still held) and Disconnect can
  retry. Any other error from `close()` lets the facade go, as before.
- **The facade reports a failed unload** (`Microscope.unload_error`, m1 §7), and
  `close()` still does not raise for it: a command that did its work must not exit
  1 because a driver would not let go, and its ERROR line is already printed.
  Disconnect says `may still be held`, never `released`.
- **The page counts its own lock-free reads** and Disconnect waits for them,
  bounded (§7.2, *Reads in flight*). The facade is not changed for it.
- **`release()` waits for a connect that is still opening**, bounded, inside the
  service's stop budget, and the closing lines say what became of it.
- **The log**: `smc ui` puts `smc.ui` at INFO (§7.1), so the page's outcomes and
  STOP's note are in the service's log without `-v`.

### 7.1 The command (`smc/cli.py`)

The options are §1's, unchanged. What changes is the order:

1. The checks are as they are today, and so are their texts and exit codes:
   - the page word;
   - `--run-doc`;
   - the `ui` extra;
   - the port;
   - `Profile.load` (a profile that does not load exits as today);
   - `RunDocument.load_or_new`.
2. Unless `--dry-run`, `_say_stop_refused(loaded.safety)` prints the line, when
   the profile declares one. Since #305 the helper takes the profile's safety
   section (`SafetySection`) and not the facade, so that the decision lives in one
   place: the three commands that hold a stand pass `microscope.profile.safety`,
   and this one passes the profile it has loaded, before any stand is open.
3. `controller = PlatePage.closed(loaded, path, dry_run=dry_run)`, then
   `plate_page.serve(controller, port=port)`. From the command's start to its end,
   the `smc.ui` logger is at INFO if it was above (`-v` and `--debug` keep their
   lower levels), and put back after, so the log holds what the page did (§7.4).
4. Whatever ends `serve`, a `finally` calls `controller.close()`. An inner
   `finally` then calls `controller.release(sys.exc_info()[1])`, which is the
   exception the command is unwinding with, or `None`. The stand is released
   whatever ends this, a second Ctrl-C during the wait included (FM-71), as the
   `with Microscope.open(...)` block did, and the release ends the facade as that
   block's exit did: a Ctrl-C leaving it is noted with the motions still tracked
   (m2 §11.11 E6). An error that `release()` raises is reported by `_Reported` as
   the `with` block's was.
5. `close_lines` and `close_ended` are read **after** the release, because
   `release()` may add to them (a Connect that was still opening, §7.2). The exit
   code is as today (§1, M3 §15.3).

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
    def release(self, unwinding: BaseException | None = None) -> None: ...
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
     - **outside** the lock, waits for the reads that started on that facade
       (*Reads in flight*, below), for at most `_READS_WAIT_S` (5 s);
     - **outside** the lock, calls `Microscope.close()`. That stops what still
       moves, then unloads every device, and it switches nothing off: a light
       or an autofocus stays as the stand keeps it (§6);
     - has one of these outcomes, in this order of precedence:
       - `close()` raised `MicroscopeBusyError`: `disconnect: MicroscopeBusyError:
         <message on one line>; the stand is still connected`. Nothing was
         unloaded and `close()` may be called again (m1 §7), so **the page keeps
         the facade**: it puts it back under `_lock`, unless the page is closed,
         and is then connected again, with Disconnect a click away. A page that
         shows `not connected` for a stand this process still holds would be a
         lie, and a stand held until the process ends, with no way to retry, was
         the first version's gap (#305);
       - `close()` raised anything else: `disconnect: <Type>: <message on one
         line>`, and the page has let go of the facade for good (it may be half
         unloaded);
       - `close()` returned and `microscope.unload_error` is not `None`:
         `disconnect: <stand> may still be held: its devices did not unload
         (<unload_error on one line>); restart smc ui before another program
         takes it`. The page is not connected, and a later Connect says what
         happened;
       - otherwise `disconnect: <stand> released`, then `, with <n> read(s) still
         in flight after 5 s` when the wait for the reads gave up.
  - The facade is dropped **before** it is closed, so no read that starts after
    the drop reaches a closed facade.
  - Measured (§7.5): a closed facade's `state()` reads `HardwareError ...
    another program holding it` for XY and Z, which would mislead, and its
    `has(XYStage)` still answers `True`.
- **`release(unwinding=None)`** releases the facade the page holds, if any:
  - It is for the command, after `close()`. Before it, it raises
    `RuntimeError("release() comes after close()")`, because the facade could
    otherwise be pulled from under an activity.
  - **A `connect` that `close()` gave up on is waited for**, for at most
    `_RELEASE_WAIT_S` (8 s), with `join()`'s slices (FM-83): `close()` waited its
    own 10 s, and the worker's late release (above) can only run while the process
    lives. `close()` keeps the name of the activity it gave up on, so that the wait
    is the same whether the connect ended before `release()` began or after.
    10 + 10 (the server's stop) + 8 is 28 s, inside `AppStopMethodConsole 30000`
    (§7.4). Only a `connect` is waited for: a run stuck in a vendor call is not
    made to wait longer, and `Microscope.close()` below meets it as before. What
    became of the connect is a closing line, added to `close_lines`:
    - it ended: its own outcome (`connect: <stand> was opened after the page
      closed, and released`, or what the opener raised), and `close_ended` is
      `True`: the stand is not held;
    - it did not: `connect: <stand> is still opening after <n> s: the process
      ends now, and whether the stand is left connected is not known`, with
      `<n>` the two waits' sum, and `close_ended` stays `False` (exit 1).
  - It drops the facade under `_lock`, waits for the reads in flight (below),
    then ends the facade as the `with Microscope.open(...)` block did, outside
    the lock: `microscope.__exit__(...)` with `unwinding`, so that a Ctrl-C the
    command is unwinding with is noted with the motions still tracked, as
    `Microscope.__exit__` does (m2 §11.11 E6). That is what a second Ctrl-C during
    `close()` lost when `release()` called `close()` directly. It raises what
    `close()` raises. A failed unload is not raised: `close()` logs it as an ERROR,
    which the command prints at any level, as it does for every other command.
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

**Reads in flight (#305).** The facade's reads take no lock, by design (FM-70): a
status tick, the light's and the turret's reads, a capture's position and STOP
all reach the devices while an activity holds the microscope lock. So
`unloadAllDevices` could run while one of them was inside the core, which was
reproduced on `FakeCore` (§7.5). `Microscope.close()` cannot see them: the page
holds the reads, and Disconnect is what makes the race a normal path. So the
page counts its own, with no lock.

- A private class `_Reading(readers, page)` is the registration. Its `__enter__`
  adds a token to the page's set `_readers` **first**, then returns the facade
  the page holds (or `None`); its `__exit__` discards the token. A class, not a
  `@contextmanager` (FM-98, FM-92), and `set.add` and `set.discard` take no lock
  and are atomic, so STOP can use it (FM-70).
- The order is the whole point. A reader that gets the facade registered before
  it took it; Disconnect drops the facade, then looks at the set. A reader that
  took the facade before the drop is therefore in the set when Disconnect looks,
  and one that comes after the drop gets `None`.
- **Which methods are reads**: `status`, `light_lacks`, `objective_labels`,
  `capture_centre`, `capture_wall` and `stop`. `lacks`, `_blocked` and `run` only
  look at which capabilities the roles give, which is no device read.
- **`_wait_for_reads()`** polls the set every 10 ms for at most `_READS_WAIT_S`
  (5 s, as long as a slow device read, and a hung one is not waited on for ever,
  FM-15) and says how many were left. `disconnect` and `release` call it after the
  drop and before `close()`. When it gives up, `close()` unloads under the read,
  which fails with a line on the page and never a crash (`_handler`), and the
  outcome says so.
- **STOP is a read that never waits.** It registers, sends the stop to the facade
  it took and unregisters: a Disconnect that began meanwhile waits for it, so the
  stop reaches loaded devices. A STOP that finds the facade dropped says `not
  connected: nothing to stop`, and `Microscope.close()` stops what still moves
  before it unloads.
- A token leaked by a Ctrl-C between `add` and the `with` body (FM-65) costs one
  `_READS_WAIT_S`, never a hang. Only the main thread gets a Ctrl-C, and it
  reads only in `stop()`, so a second Ctrl-C inside it is the one case.

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
  - while the `connect` activity goes (`status.running == "connect"`):
    `connecting <stand>…`, and while `disconnect` goes: `disconnecting
    <stand>…`. On a slow stand the row would otherwise say `Connect opens it`
    beside a greyed-out button, and the page would look stuck (#305);
  - in every case, then ` (dry run)` when it is a dry run.

  This one line is the reason for every control that needs the stand.
- **Disabled**:
  - `Connect` while connected, or while an activity goes;
  - `Disconnect` while closed, or while an activity goes (live included);
  - the capture buttons while closed, and while an activity other than this page's
    live goes (#295; the controller refuses a capture then, `_refuse_if_moving`,
    `connect` and `disconnect` included).

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
  --run-doc <absolute path> --port 5006`. It is **two processes** on Windows
  (corrected by #305; #264's plan said one): a venv's `Scripts\python.exe` is
  itself a launcher (a copy of `venvlauncher.exe`) that runs the base interpreter
  as a child, as the `smc.exe` console script does, and the interpreter that holds
  the stand is that child. It shares NSSM's console, so the Ctrl-C reaches it;
  that is to be checked at a stand (§9.1).
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
  `AppStopMethodConsole 30000` covers `close()`'s 10 s, the server's 10 s and
  `release()`'s 8 s wait for a connect still opening (§7.2). A `Microscope.close()`
  that waits for a hung driver is bounded by `device_timeout_ms` (60 s by
  default) and cannot be fitted in: NSSM then ends the process, as it always did.
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
  - **What the page says** goes to the log file, not to a console. Since #305
    the command puts the `smc.ui` logger at INFO (§7.1), so the file holds each
    activity's outcome (`connect: ... connected`, a failed Connect, `disconnect:
    ... released`) and STOP's note, as well as every warning and error. The
    format has no time (`%(levelname)s %(message)s`): a timestamp is the
    owner's to ask for, as it changes every `smc` log line.
- **Stopping the PC**: Disconnect first. How long Windows lets a service stop at
  shutdown is not known here.

### 7.5 Tests and failure modes

The tests are named in #264's plan.

**Failure modes**: FM-98, FM-70, FM-71, FM-92, FM-143, FM-146, FM-149, FM-151,
FM-169, FM-161, FM-117, FM-125, FM-44, FM-96, FM-43, FM-45, FM-36, FM-37, FM-20,
FM-24, FM-69. #305 adds FM-15, FM-49, FM-62, FM-65, FM-83 and three new
entries, named by what they say, whose ids the develop session takes when it
writes them: an unload during an unlocked read; a documented log content that the
default level drops; a venv's `python.exe` that is a launcher with a child
process (Windows).

**Measured 2026-11-03** (#305; Python 3.13.0, `FakeCore.demo_like()` and
`Profile.demo()`; throwaway scripts):

- **An unload during a lock-free read.** A `state()` replaced by a wrapper that
  holds inside the read (an `Event`): `close()` from another thread returned, and
  the core listed only `Core`, while the read was still inside it.
- **`close()` that raises.** With the microscope lock held by another thread and
  `_lock_timeout_s = 0.2`, `close()` raised `MicroscopeBusyError` after 0.20 s, with
  `_closed` still `False` and 14 devices loaded; once the lock was free, a second
  `close()` unloaded them. After an `unloadAllDevices` that raises,
  `close()` returned normally with `_closed` `True` (so a retry is a no-op),
  which `Microscope.unload_error` reports (m1 §7).
- **`sys.exc_info()` in a `finally`**: in the inner `finally` of a `try/finally`
  that an exception unwinds, `sys.exc_info()[1]` is that exception, and `None` on
  the normal path. It is what the command passes to `release()` (§7.1).
- **The three mechanisms of §7.2**, on a throwaway subclass of `PlatePage`
  (`mypy --strict` clean): a `status()` held inside its read kept Disconnect from
  unloading (14 devices while it held, `Core` alone after it ended, and its
  status read `connected=False`); with the wait shortened to 0.2 s, Disconnect
  unloaded under the read and said so; a Disconnect whose `close()` raised
  `MicroscopeBusyError` left the page connected, and a second one released; a
  connect still opening when `close()` gave up (wait 0.2 s), with `release()`
  waiting, returned after 0.60 s, with the outcome `connect: demo was opened
  after the page closed, and released` and the core listing only `Core`.

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

**Not measured (#305)**: an unload during a read on a real core (whether MMCore
survives it is not known here: the page treats it as a failure to avoid);
`device_timeout_ms` at its default in a Disconnect
that waits for a hung driver; the log's growth at INFO.

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
8. **#324** (`smc presets save`, §10.9) edits `presets.py`, `params.py` and `cli.py`,
   and the tests of the presets: not at the same time as #338 (the review's
   leftovers of #337, which edit `presets.py`, `cli.py` and the page).
9. **#346** (the page for `plate-walls`, §12) comes after #332 (PR #362) and edits
   `controller.py`, `plate_page.py` and `guide.py`: no other issue that edits them
   runs at the same time. #367 (the leftovers of #332: `serve()`'s filter, the stale
   label's plate check) and #327 (the *Wall* select) edit the same captures' region.
10. **#320** (the scan plan on the map and its estimate, §14) edits `controller.py`,
    `plate_page.py` and `guide.py`: no other issue that edits them runs at the same
    time. It comes after #327 (§13, the captures' region of `plate_page.py`).

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
  - `nssm stop` while a run goes: the Ctrl-C reaches Python, the child of the
    venv's launcher, and the stand is released within `AppStopMethodConsole`;
  - `nssm restart` while a Connect is still opening a slow stand: `release()`'s
    wait (8 s) and the late release (§7.2);
  - a Disconnect pressed while a status tick reads a slow device (§7.2);
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
    "objective-offsets": {
        "confirm_turns": "the stage is clear only for the run that confirmed it: "
        "every run confirms it again"
    },
    "plate-finding": {
        "centre_x_um": "a preset holds no positions",
        "centre_y_um": "a preset holds no positions",
    },
    "plate-registration": {
        "plate": "the preset's plate holds it",
        "refs": "a preset holds no positions",
        "walls": "a preset holds no positions",
        "objective": "the registration takes the objective in place",
    },
    "plate-walls": {
        "plate": "the preset's plate holds it",
        "refs": "a preset holds no positions",
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
   - `plate=<check.plate>` for `plate-registration` and for `plate-walls` (#371:
     a plugin whose `plate` `NOT_IN_PARAMS` names takes it here, or a run from the
     preset would use the plugin's default plate);
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
smc presets save NAME --run-doc PATH [-p PROFILE] [--overwrite]   # §10.9, #324
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
- **No `smc presets save` in #319.** The CLI's equivalent of *Save*, which writes
  a run document's parameters as a preset, is §10.9 (#324).
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

### 10.9 `smc presets save` (#324)

Fixed by #324's plan (2026-11-05). It is the command line's equivalent of the
page's *Save*: it writes a preset from the values a run document's `done` records
used, so that the next run takes them from the last one and not from a retyped
file. It reads a document and writes a file; it opens no stand and moves nothing.

```
smc presets save NAME --run-doc PATH [-p PROFILE] [--overwrite]
```

**Value to text** (`smc/plugins/params.py`).

```python
def format_param(field: FieldInfo, value: Any) -> str: ...
```

It is public, beside `parse_params`, and is the inverse of `_decode`: for a
recorded JSON value, the text that `--param key=<text>` takes.

- `None` is `null`.
- A `bool` is `true` or `false`.
- The value of a scalar field (`str`, `int`, `float`, `Path`, an `Enum` member, a
  `Literal`; the fields `_is_scalar` names) is its text, a float by `repr`.
- The value of any other field is `json.dumps(value)`, which is ASCII.

It does not promise that the text reads back as the value: a `str | None` field
that holds the string `null` reads back as `None`. The caller reads it back.

**The preset from a document** (`smc/plugins/presets.py`).

```python
@dataclass(frozen=True, slots=True)
class PresetFromRun:
    preset: Preset
    notes: tuple[str, ...]


def preset_from_document(
    doc: RunDocument,
    models: Mapping[str, type[PluginParams]],
    *,
    stand: str,
    saved_utc: datetime,
) -> PresetFromRun: ...
```

`models` maps a plugin's name to its `Params`. The fields of the preset:

- `stand` and `saved_utc` are the arguments.
- `plate` is `registered_plate(doc)`.
- `wells` is the `scan-plan` record's `wells`, as typed, when that record is
  `done` and the value is text.
- `params` has a key for each plugin that has at least one pair, in the order of
  the plugins' names.

**Which records.** The `done` ones. A `stopped`, `failed` or `running` record's
values are not those of a finished run; a dry run's record counts, its values
being the operator's. A document with no `done` record is
`PresetError("the run document has no finished stage; there is nothing to
save")`. When the `done` records give no plate, no wells and no pair, it is
`PresetError("nothing to save: <the plugins' names, joined by ", ">
used only default values")`.

**The pairs of a plugin**, from its record's `params`, which are data taken as
they come (FM-125):

1. A plugin that `models` does not hold gives the note `<plugin>: no plugin of
   that name is loaded; its parameters are not saved`.
2. The record's `params` go through the plugin's `Params.model_validate`; the
   plugin may have changed since the run. A `ValidationError` gives the note
   `<plugin>: its recorded parameters are not valid for this smc (<loc>: <msg>,
   joined by "; "); not saved`.
3. A pair, `<key>=<format_param(field, value)>`, for each field in the model's
   declaration order (the order `smc run --show-params` prints) that is required,
   or whose JSON value differs from the JSON form of its default
   (`to_jsonable_python(field.get_default(call_default_factory=True))`), except
   the keys that `NOT_IN_PARAMS` names for the plugin. A value equal to its
   default is left out: the preset follows the plugin's default, as `[suggest]`
   does.
4. **The pairs are read back.** `parse_params` reads them, with one pair for each
   key `NOT_IN_PARAMS` names (what the run takes from elsewhere: the preset's own
   plate and wells, the operator's positions), and the JSON dump of the result
   must equal the JSON dump of the validated record, key by key. When `parse_params` raises, or a key
   differs, **none of the plugin's pairs is saved**, because a subset may be
   refused (`rows` without `columns`): the note is `<plugin>: <keys, joined by
   ", "> would read back as another value; not saved`, or, when it raised,
   `<plugin>: its pairs do not read back (<the problems, joined by "; ">); not
   saved`.
5. A pair that does not encode as UTF-8 (a lone surrogate, which an objective
   label from a `.cfg` saved in cp1252 leaves in the run document, FM-05) saves
   none of the plugin's pairs: `<plugin>: <key> holds a character that is not
   text; not saved`. A `wells` text that does not encode is left out: `scan-plan:
   wells holds a character that is not text; not saved`.
6. A record whose `profile` is not `stand` gives the note `<plugin> was run on
   <profile>, not on <stand>`, and its pairs are saved.

The `Preset` is built last. A `ValidationError` (a `wells` of several lines) is
`PresetError("nothing saved: <the first message>")`.

**`NOT_IN_PARAMS` gains `plate-finding`'s `centre_x_um` and `centre_y_um`** (§10.2):
the search's centre is a stage position, and a preset holds none. The check then
says so for a hand-written preset that gives one.

**The command** (`smc/cli.py`), in this order:

1. `check_name(NAME)`. A `PresetError` is `typer.BadParameter(<message>,
   param_hint="NAME")`, which exits 2, before any file is read.
2. `Profile.load`.
3. `RunDocument.load(Path(--run-doc))`. A missing or unreadable document is a
   `RunDocumentError`, which exits 1 as `_report` maps it, and nothing is written.
4. `preset_from_document(doc, <the registry's Params>, stand=<microscope.name>,
   saved_utc=<now, UTC, to the second>)`, the registry from `discover_plugins()`
   as `show` reads it. A `PresetError` is `_fail(<message>)`, exit 2.
5. `save_preset(name, preset, profile, overwrite=--overwrite)`. A
   `PresetExistsError` is `_fail("<message>; pass --overwrite to replace it")`,
   exit 2: the message is the exception's own, so that it says what the page
   says. Any other `PresetError` (the write, a missing `tomli-w`) is
   `_fail(<message>)`, exit 1.
6. On success, through `_say`: `saved <name> to <path>`; then the preset's lines
   as `show` prints them (`stand`, `plate`, `wells`, `saved_utc`, `<plugin>: <pairs,
   joined by a space>`), from one helper that both commands call; then each note.

`--run-doc` is required (a usage error, exit 2). A preset that `smc run
scan-plan --preset NAME` then reads, on a document whose plate is registered,
gives `scan-plan` the params the saved document's record holds.

**What it does not do.** It takes no `--plugin` to choose among the records and
no `--dry-run`: it writes a file that moves nothing. It does not judge a value:
a `nis-export` `out` or a `camera-calibration` `from_run_doc` is saved like any
other pair, and `smc presets save` prints every pair it wrote.

**Failure modes**: FM-05 (the surrogate), FM-24 (the name; a Windows path in a
value), FM-36, FM-37 and FM-72 (the output and the help), FM-38 (the write, through
`save_preset`), FM-43 and FM-45 (the tests), FM-107 (what the generator leaves
out is said, not dropped), FM-125 (the record validated again). Its new kind,
for the checklist: *a text generated from a value that its reader reads back as
another value*.

## 11. The plate page at the stand (#332)

Fixed by #332's plan (2026-11-05), from Ti2-E session 2 (#198, report #334): the
first plate registration on the page, on a real stand, where the owner found it
« encore très difficile à prendre en main ». Five findings, each decided here; the
tests and the order of the work are #332's plan. Measured 2026-11-05 on the `demo`
stand (Python 3.13.0, Panel 1.10.0, Bokeh 3.10.1, Tornado 6.6).

This section owns the **captures' target**, **why a tool is disabled**, **the plate
the page starts with** and **what a Run replaces**. It does not own the capture
widgets' arrangement for `plate-walls` (#346) or the wall sides' frame (#327): each
builds on the names below and changes none of them.

### 11.1 Which well a capture records (finding 1)

Measured on the demo with a script that clicks B3, B4, B4 again and B3 again, then
types `C1:C2`:

```text
click B3: selection=('B3',)       current=B3
click B4: selection=('B3', 'B4')  current=B4
click B4: selection=('B3',)       current=B4   # a click that deselects B4
click B3: selection=()            current=B3   # and one that deselects B3
typed C1:C2: selection=('C1','C2') current=B3  # B3 is not even selected
```

`current` is what *Capture centre*, *Capture wall* and *Forget* act on, and nothing
on the page showed it. The session's wall was recorded under B4 because the
operator's last click on the map was B4's.

**Decided: `current` is a well the operator can see.**

- **Rule (controller).** `current` is the well the operator last *selected* by a
  click, or *drove to* (`goto`, which sets it without selecting the well).
  - a click that selects a well sets it; a click that deselects the well that is
    `current` drops it (`None`), and a deselecting click on another well leaves it;
  - a typed range or a preset's wells replace the selection: `current` stays only
    when it is in the new selection, so a driven-to well that the new selection does
    not hold is dropped too;
  - `choose_plate` drops it, as now.

  It is one helper, `_drop_unselected_current()`, called wherever the selection is
  assigned (`toggle`, `select_text`, a preset's wells), so that no path forgets it.
- **The buttons name what they record** (`smc/ui/guide.py`, pure functions, no
  Panel):

  ```python
  def capture_centre_label(well: str | None) -> str: ...
  def capture_wall_label(well: str | None, side: str) -> str: ...
  def forget_label(well: str | None) -> str: ...
  ```

  `Capture centre of B3`, `Capture B3 left wall`, `Forget B3`; with no well, the
  plain `Capture centre`, `Capture wall`, `Forget`, and the three buttons are
  disabled. The side is the *Wall* select's value as it stands: #327 changes what
  the select offers, and the label follows its value.
- **The well on the label is the well recorded** (FM-177). `PlatePageView` keeps
  `_capture_well`, the well the buttons were last drawn for. A capture or a forget
  whose `page.current` is not `_capture_well` records **nothing** and says `the
  well to record changed from B3 to B4: check the button, then press it again`
  (`none` for a `current` that was dropped); the view then redraws. When the
  buttons were drawn with no well, the text is `click a well on the map first`.
  Acting on the label would record B3's walls at a position the operator took for
  B4's; acting on the controller's state would do what the label did not say.
  Neither is safe, so the click asks again.
- **The *Wall* select redraws the label at once**, not at the next tick.
- A disabled button is a courtesy, not a guard (FM-169): `capture_wall` refuses a
  `None` well in the controller's own terms, as it refuses a well off the plate.

### 11.2 Why a tool's Run is disabled (finding 2)

`_draw_buttons` disables every Run while an activity goes, and the note under a tool
read `_blocked` only (a missing input, a closed stand). While Live runs the strip
says `live is running`, three regions away.

**Decided: it stays refused, and says why beside the button.** Not decided for
`plate-registration`: it requires and reads nothing (`requires = ()`,
`reads = ()`), and touches no device, so it could run during Live. Against it:

- one activity at a time is the page's rule (§2.1), and a pure tool goes through the
  same worker, the same `_current_run` and the same record write as `scan`; a
  second slot is a change of §2.1 and of STOP, `close()` and `join()` with them
  (FM-70), for three clicks saved;
- §5.2 decided that Live is not ended for another action, for the reasons it gives.

```python
def busy_note(running: str, *, live: bool) -> str: ...
```

in `guide.py`: `live` gives `Live is running: press End live, then run the tool.`
(the button's own label, `_END_LIVE_LABEL`); any other activity gives `<running> is
running: wait for it to end, or press STOP.` `_draw_buttons` puts it in each tool's
note and in the next step's note when the tool has no `blocked` reason of its own
(the stand closed, a missing input): that reason holds longer than the activity.

### 11.3 The plate the page starts with (finding 3)

`_initial_plate` took the plate of the `plate-registration` record, else of the usable
`plate` slot, else `96-well`. The last answer was silent: on a 24-well plate, wells
captured on a 96-well map and registered with `plate=96-well` put A1 two pitches
wrong, a seed has no residual, and nothing says so. And it ignored `plate-walls`
(#328), a second producer of `plate`: a run document holding an older
`plate-registration` and a newer `plate-walls` opened on the older plate (#346's
note on PR #351).

**Decided: carried over when anything knows, asked when nothing does.** The sources,
in order:

1. the plate of `doc.provider("plate")`: the record of the latest start, whatever its
   status, as the slot is defined (`RunDocument.provider`); its `params["plate"]`;
2. the profile's `[sample].plate`, when the profile has a `[sample]` (the stand knows
   what its camera images);
3. the placeholder `96-well`, which the page **names**.

`PlatePage.plate_chosen: bool` is `True` for 1 and 2, and after `choose_plate(name)`
for any known name, **including the one already shown** (the early return for the
same name only skips the reset). Where `_restore` takes the captures from, it follows
the same record: the latest provider of `plate` when it is `plate-registration`
(`refs` and `walls`) or `plate-walls` (`refs`), else none.

While `plate_chosen` is `False`:

- a note under the *Plate* select says `Which plate is on the stage? 96-well is only a
  placeholder: choose the plate, or press the button if it is a 96-well.`, and a
  button `The plate is a 96-well` (the placeholder's name) confirms it;
- **a tool whose `PAGE_PARAMS` include `plate` does not run**: `PLATE_NOT_CHOSEN =
  "choose the plate on the stage first: the page shows a 96-well placeholder"`
  (`guide.py`) is its `blocked` reason, ahead of every reason but the stand's being
  closed, it is the next step's `stuck`, and `run()` refuses it in the controller as
  the outcome `<plugin>: choose the plate on the stage first ...` before anything is
  read (FM-169). The pairs the next step carries are then empty, as when the step is
  stuck on `CAPTURE_WELLS`: the plugin's own default (`plate=96-well`) must never be
  what a click runs;
- captures stay allowed: a capture is a stage read, and `choose_plate` dropping them
  is the operator's own act, shown on the page.

`CAPTURE_WELLS` is unchanged: the step shows it once the plate is chosen, and
`PLATE_NOT_CHOSEN` before.

*Decided against:* a plate in the stand's profile. A profile is an instrument's, and
the plate changes between experiments; what a real stand suggests is the owner's call
after a session there (§8.1).

### 11.4 What a Run replaces (finding 4)

`RunDocument.provider` lets the latest attempt decide even when it failed: a re-run of
plate registration that fails leaves no plate, rather than the older one silently. The
page did not say it before the click, and after the click the seed that had just been
usable was gone from the document.

**Decided: say it before, and do not refuse.** FM-172 refuses a click that discards
work *no command can bring back*; a registration is a pure function of its parameters,
which are the captures, and the page keeps its captures across a failed run (and
`_restore` reads them back from the failed record after a restart). So the plate is
had again by the same command, minus the capture that was wrong, and the page says so:

```python
def replace_note(result: PlateResult, started_utc: datetime) -> str: ...
```

in `guide.py`: `Run replaces the plate registered 2026-11-03 15:20 (seed, 1 well). A
failed run leaves no plate; the captures stay, so Forget the wrong one and Run again
to get a plate back.` The time is the record's `started_utc` in the PC's local time,
to the minute; the method and `n_refs` are the result's own (`seed`, `references`,
`lattice-fit`, `wall-images`); `1 well` and `2 wells`.

`PlateView.replaces: Mapping[str, str]` (new, default empty): a tool's name to the
note, for each tool whose plugin `provides` the `plate` slot, when that slot is
usable now (`_slot` returns a result). The page appends it below the tool's own
note. It is **not** drawn for the other slots: a card already says the last result,
and a note under every tool teaches the operator to read none of them.

*Amended by #346 (§12.7)*: the second sentence of the note, `the captures stay, so
Forget the wrong one and Run again`, is true of the tools that run from the page's
captures (`plate-registration`, `plate-walls`) and false under `plate-finding`, which
has none. `replace_note` takes `keeps_captures: bool`, keyword-only, and the note of a
tool that keeps none ends after `A failed run leaves no plate.`

### 11.5 A Ctrl-C that prints one line (finding 5)

Reproduced off Windows, with the page's own server call (`pn.serve(..., threaded=True)`)
and one open session (`bokeh.client.pull_session`), stopped as `serve()` stops it:

```text
ERROR:tornado.application:Uncaught exception GET /ws (::1)
...
  File ".../bokeh/server/views/ws.py", line 322, in on_close
    self.connection.session.notify_connection_lost()
  File ".../bokeh/server/connection.py", line 64, in session
    assert self._session is not None
AssertionError
```

The line numbers are the owner's report's. With no session open the stop is silent
(0.00 s). Cause: a stop discards the sessions and detaches each connection from its
session (`_session = None`) before the socket's own close reaches `on_close`, which
then reads `connection.session`. It is Bokeh's order of work, not a race with a
connection being opened, and every Ctrl-C with the page open in a browser, which is
every Ctrl-C of a session, hits it.

**Decided: `serve()` drops that one record while it stops the server.** A
`logging.Filter` on `tornado.application`, attached around `_stop_server(thread)` and
removed in a `finally`: it drops a record whose message starts with `Uncaught
exception GET /ws` and whose `exc_info` holds an `AssertionError`, counts it, and
every other record passes. The count is one `log.info` line of the page's own logger
(`smc.ui`, not the console): `the server's stop closed N open sessions`. Measured: with
the filter the stop took 0.00 s and dropped exactly one `AssertionError`; without it
the traceback printed. *Decided against:* patching `ServerConnection.session` (a
vendor's class, and a version's) and closing the sockets first (private API).

### 11.6 What this does not do

- It does not add the *plate-walls* button or its card (#346); `PAGE_PARAMS["plate-walls"]`
  there gets §11.3's gate for free by naming `plate` in it.
- It does not change what a wall side means (#327). The *Capture wall* control stays
  under `plate-registration`, with the manual form of the command line.
- It does not let a tool run while Live goes, and does not ask for a plate on a stand
  whose run document or `[sample]` names one.

### 11.7 Tests and failure modes

FM-169 (every refusal is the controller's), FM-172 (a replaced result: decided, with
the reason), FM-177 (the label a capture acts on), FM-143, FM-149 and FM-83 (the stop
path), FM-151 (a view that draws what the shared controller holds: `current` is
shared by every tab), FM-161 (the Ctrl-C test goes through a real server and a real
session, with `interrupt_main`), FM-98 and FM-139 (text, never the error; plain text,
never markup). The new kind, for the checklist: *a record that bounds what a vendor
logs at shutdown by its message alone*, which hides a different failure with the same
words: the filter checks the exception's class too, and only while the server stops.

**Not verified (for the owner)**: the Ctrl-C of `smc ui` at the Ti2-E's PC, with the
page open (Python 3.14, Windows 11); the page walked by a newcomer on a real stand.

## 12. The plate page registers from images (#346)

Fixed by #346's plan (2026-11-09), on the page of #332 (PR #362) and the plugin of #328
(PR #351). m3 §17.5 fixes the interface (`STAGES`, `PAGE_PARAMS`, at least two centres,
a card from `walls`). This section decides the page: where a well is recorded, what the
button says before and during the run, what the card shows, and when the next step is
this tool. Measured 2026-11-09 on the `demo-sample` stand (Python 3.13.0, Panel 1.10.0,
Bokeh 3.10.1), with a script on `PlatePage` that passes the entry point by hand.

**What was measured.**

- *The tool on the page today.* `plate-walls` is there under `More`, with a form that
  offers `plate` and `refs`, and no captures: `page_params("plate-walls")` is `[]`, so
  its Run sends the plugin's own defaults. With `camera-calibration` run (0.2 s), the
  stage moved to A1 and H1 (250 µm and 150 µm off the sample's centres) and
  `capture_centre` called for each, a run with `plate=96-well` and those two centres as
  `refs` took **16.5 s** for 20 frames (10 per well) and gave `method: wall-images`,
  `residual_um` 0.003 and `rotation_deg` 1e-5. **Its card showed the result's first eight
  keys (`a1_center_xy_um: 2 items`, ...) and `… 2 more`: the walls, the part an operator
  wants to read, were the part cut away.**
- *The reason before the calibration* is the run document's own: `the run document has
  no camera-calibration; smc run --list shows which plugin provides it`. It is a sentence
  for a terminal, shown on a page that has the tool one region up.
- `view().replaces` already holds a note for `plate-walls`: it provides `plate`.
- *A Panel widget placed twice in one layout is drawn once.* `pn.Column(pn.Row(b),
  pn.Row(b)).get_root()` holds one Bokeh `Button`, and the widget's `_models` one entry
  per root; two widgets give two models, each with its own label.
- *The entry point.* The main checkout's venv was installed before #351:
  `entry_points(group="smc.plugins")` there has no `plate-walls` (FM-112). A worktree
  set up with `worktree.py setup` has it; the page's tests name the entry points they
  need, as they do for every other tool.

### 12.1 What the operator does

The owner's form, « positionner à la main avec le joystick sur 2 ou 3 wells (on clique
sur les wells sélectionnés), puis la calibration se fait toute seule », as the page
makes it:

1. *Connect*, and choose the plate on the stage (§11.3).
2. Run *camera-calibration*, the first tool of *Set up*, with the objective the
   registration is made at in place, on a field with texture (a well's wall will do):
   `plate-walls` reads it.
3. Drive to the middle of a well with the joystick (or the Drive pad); Live shows it.
4. Click that well on the map: the buttons name it (§11.1). Press **Capture centre of
   B3** in `plate-walls`' section.
5. Again for a second well, far from the first, and a third.
6. Press **Run plate-walls**: the next step's button (§12.4) or the tool's own. The
   strip counts `plate-walls: 3/8 B3 -X` while the stage images each well's four walls;
   then the plate is registered and the card says how well (§12.6).

### 12.2 The tool on the page (m3 §17.5)

- `STAGES`' *Set up* is `("camera-calibration", "plate-registration", "plate-walls",
  "plate-finding", "timing")`.
- `PAGE_PARAMS["plate-walls"] = ("plate", "refs")`: the form offers neither, and §11.3's
  gate (`_needs_plate`) covers the tool with no further code. `KEY_PARAMS` has no entry
  for it: `search_um`, `max_residual_um`, `max_rotation_deg`, `settle_ms` and
  `save_frames` stay under *Advanced*, closed. A newcomer sees buttons.
- `page_params("plate-walls")` is `[f"plate={plate}", f"refs={json}"]`, the `refs` dumped
  by the one function that dumps them for `plate-registration` too: a step and a click
  agree. `_page_pairs` returns them for any number of centres; the number is the gate's
  (§12.3), as the plate's is (§11.3), so that a preset's check of the tool's other
  fields is not refused for it.
- A preset holds the plate as its own field for this tool too (§10.2): a `plate` pair
  under `plate-walls` is the mismatch `plate is not kept under params: the preset's
  plate holds it`, and the step's pairs name the plate once (#371).
- Wall captures stay `plate-registration`'s. `plate-walls` images the walls itself and
  ignores them; the page says so when some exist (§12.5).

### 12.3 Why its Run is disabled, in order

`_blocked["plate-walls"]` is the first of: the stand closed (`NOT_CONNECTED`); the
placeholder plate (`PLATE_NOT_CHOSEN`, §11.3); and, new, in this order:

1. **A dry run** (`WALLS_DRY_RUN`): `plate-walls images the walls, which a dry run does
   not do: serve the page without --dry-run (demo-sample's stage is simulated), or
   register with plate-registration`. The plugin refuses a dry run before it moves
   anything, and its refusal is a `failed` record that replaces a usable plate (§11.4):
   so the controller refuses first, in `run()`, as the outcome `plate-walls: <text>`,
   nothing written (FM-169).
2. **Square wells** (`walls_square_wells(plate)`, *added by #375*): `plate-walls fits
   round wells, and the 384-well plate's are square: register it with
   plate-registration`, the plugin's own words (m3 §17.4, item 2; a test pins the pair).
   The plugin fits circles and refuses a 384- or 1536-well plate before it moves, and
   its refusal is a `failed` record that replaces a usable plate (§11.4): the same
   situation as the dry run, so the same answer. It is asked of the page's plate
   (`useq.WellPlate.from_str(plate).circular_wells`), after the dry run (a mode is the
   reason that holds longer) and before the calibration, which the operator would run
   for nothing. `run()` refuses it as the outcome `plate-walls: <text>`, after the plate's
   refusal and the dry run's, before the parameters are read, nothing written, so a
   registered plate stays the slot's provider (FM-169). One helper,
   `PlatePage._walls_refusal()` (the dry run, then the square wells), is the source of the
   button's reason, of `page_params` and `run()`, and of the next step (§12.4).
3. **The calibration.** What the tool reads, as for every tool; but with **no**
   `camera-calibration` record at all the reason is `WALLS_NEED_CALIBRATION`:
   `plate-walls reads the camera's calibration: run camera-calibration first (Set up),
   with the objective you register with in place, on a field with texture (a well's
   wall will do)`. A calibration that exists and is unusable (another stand, a dry run,
   stale) keeps the run document's own words.
4. **The stand**: what the tool requires (`XYStage`, `Camera`), as for every tool.
5. **The centres**: `walls_centres_problem(len(page.refs))`.

```python
WALLS_MIN_CENTRES = 2
WALLS_MAX_CENTRES = 6


def walls_centres_problem(captured: int) -> str | None: ...
```

in `guide.py`; the two bounds are the plugin's `_MIN_WELLS` and `_MAX_WELLS`, and a test
pins the pair. The function gives `None` from 2 to 6 centres, else:

- 0: `plate-walls needs the centres of 2 or 3 wells: centre a well under the objective
  (Live and the Drive pad), click it on the map and press Capture centre; then a second
  well, far from the first`;
- 1: `1 centre captured, plate-walls needs 2 or 3 wells: centre a second well, far from
  the first, click it on the map and press Capture centre`;
- more than 6: `plate-walls takes at most 6 wells and <n> centres are captured: Forget
  some`.

`page_params("plate-walls")` raises, after the plate's gate and in the order of the list
above, the dry run's text, then the square wells', then `ValueError(<that text>)` for the
count (*amended by #375*: it raised the count's first, so a click on a dry-run page showed
the count while the button said the dry run). The controller's `run()` has no refusal for
the count: the pairs it is given may not be the
page's captures (a step's drawn pairs, a hand-spelled call), and the plugin refuses a
count it does not accept before anything moves. A disabled button is a courtesy, not a
guard (FM-169).

### 12.4 The next step is this tool when the operator took this path

The `plate` slot has several providers (`plate-registration`, `plate-walls`,
`plate-finding`) and §8.3 took the first in `tools()`' order. **Decided: the step is
`plate-walls` when all of these hold, and `plate-registration` as before otherwise:**

- the plate is chosen (§11.3): else `plate-registration`'s `PLATE_NOT_CHOSEN` stays the
  step's reason;
- the page is not a dry run (FM-182: a path the page's own plugin refuses in one mode
  is not offered there, as `scan` is not, §8.8.1);
- the plate's wells are round (*stated by #375*; the plugin fits circles and refuses a
  384- or 1536-well plate, §12.3 item 2; the step stays `plate-registration`, which takes
  square wells). The two conditions are one call, `_walls_refusal() is None`;
- the stand can image: it has an `XYStage` and a `Camera`, or it is not connected (the
  step is then drawn for the stand the operator is about to open, and `NOT_CONNECTED`
  is its reason either way);
- no wall is captured (`page.walls` is empty): touching walls is `plate-registration`'s
  own path, and an operator who did it chose it;
- `walls_centres_problem(len(page.refs))` is `None`.

The step carries `page_params("plate-walls")`'s pairs and the lines of `why`: the slot's
own (`the run document has no plate yet`), then `plate, refs: the 2 centres of B3, D5
captured on this page; plate-walls images each well's four walls around its centre and
moves the stage in X and Y only` (the wells in capture order), then the other providers
as §8.3 lists them (`or plate-registration: ...`). Its `blocked` is the tool's: until the
calibration is run, the step says `WALLS_NEED_CALIBRATION` and the button is disabled.

*Amended by #375.* When `plate-walls` is the step only because it is the slot's first
provider (`plate-registration` did not load), on a dry-run page or a square-welled plate,
`_walls_step` fills **no page pair** (`plate`, `refs`), so the step's command carries no
`--param plate=...` a click or a terminal would run, and its reason is the refusal (§12.3
items 1 and 2), as it is the placeholder plate's when the plate is not chosen (§11.3). A
pair that the profile's `[suggest]` table or a loaded preset gives the tool is still
added after it, here as under the placeholder plate, and the step is blocked by the same
reason either way.

*Decided against:* a step that runs `camera-calibration` first. It moves the stage and
measures at a place nobody chose, and which objective the registration is made at is the
operator's. *The owner may flip the rule*: with two centres captured, `plate-registration`
fits the plate from the centres as they are (by eye), and `plate-walls` measures; the page
offers the measured one first because it is the path the owner asked for. Both stay on
the page.

### 12.5 Where a well is recorded: in the tool's section, with its own widgets

**Decided: `plate-walls`' section holds the centre captures, and only those.** The owner
clicks wells on the map and then presses the tool's button: the capture sits in the
tool's section, under its summary, above its Run. Top to bottom: the summary;
`[Capture centre of B3] [Forget B3]`; the note; the centres captured; the wall note; the
*Advanced* fold, closed; Run; the tool's note; the card.

- **Widgets of their own** (measured above): `walls_capture_button`,
  `walls_forget_button`, `walls_capture_note`, `walls_captures_text` and
  `walls_touches_note`, built beside `plate-registration`'s and wired to the same
  handlers, `on_capture("centre")` and `on_forget()`. `plate-registration`'s section
  keeps every widget it has, by the same names.
- **One well, one check.** `_draw_captures()` sets both sets from the one `current` it
  read, so `_capture_well` stays one value and §11.1's check (FM-177) stays one check: a
  press records the well its label named, whichever button it is. The labels are
  `capture_centre_label(well)` and `forget_label(well)`. `Forget B3` drops B3's centre and
  its wall touches, as `forget` does. Disabling is the registration's: *Capture centre*
  with the stand closed or an activity other than live going, or with no well; *Forget*
  with no well. The note is §11.1's `click a well on the map first: ...` with no well.
- **The centres** are `centres_text(page.refs)`, plain text, never markup (FM-139):

  ```python
  def centres_text(refs: Sequence[WellRef]) -> str: ...
  def wall_touches_note(touches: int) -> str: ...
  ```

  `no centre captured yet`, or `centres captured (2):` and one line for each, in
  capture order, indented by two spaces: `B3 (10182.0, -10087.9) µm`. `wall_touches_note` is `""` for 0 and
  `<n> wall captures are for plate-registration; plate-walls images the walls itself and
  ignores them` otherwise (`1 wall capture is`, `2 wall captures are`): what happens to
  the wall captures is said once, where the operator would wonder.

### 12.6 What the card shows

`_card` builds `lines` from a `done` record's result. For a `plate` result that holds
`walls` it builds these instead:

```python
def walls_card_lines(result: PlateResult) -> tuple[str, ...]: ...
```

(`guide.py`, pure.) One line per well of `walls.wells`, in the order given, then one for
the plate:

```text
B3: centre (10182.0, -10087.9) µm, 15298 µm across, 14 frames
D5: centre (1.0, 2.0) µm, 15298 µm across, 10 frames
plate 24-well: residual 3.2 µm, rotation 0.12°
```

`<well>: centre (<x:.1f>, <y:.1f>) µm, <diameter:.0f> µm across, <n> frames`, where `n`
is the sum of the `frames` of the well's four sides; `plate <plate>: residual <r:.1f>
µm, rotation <deg:.2f>°` (`residual unknown` when `residual_um` is `None`). At most six
wells and the plate line: inside `_CARD_LINES`; the cut `… <n> more` stays the bound for
a hand-edited document. The result's other keys are in the run document and are not
lines of this card; `assumed` is unchanged and carries `two wells: the residual checks
their distance only; a third well checks the fit`. **`files`** gains `folder: <run
folder>/plate-walls/<token>` from `walls.folder` when it is a non-blank text, through
`_file_path` (the frames the operator may open, §8.4). The result is read as data (FM-125):
`PlateResult.model_validate(record.result)`; a result that does not validate, or has no
`walls`, keeps the generic card, and a `plate-registration` card is unchanged.

### 12.7 The replace note under a tool with no captures (#367, item 3)

`replace_note(result, started_utc, *, keeps_captures: bool)` (§11.4's amendment): with
`keeps_captures` the text is §11.4's; without, `Run replaces the plate registered
2026-11-03 15:20 (references, 3 wells). A failed run leaves no plate.`
`_replace_notes` passes `"refs" in PAGE_PARAMS.get(tool.name, ())`: a tool whose pairs
the page fills from captures keeps them.

### 12.8 What this does not do

- It does not run `camera-calibration` for the operator (§12.4), and does not drive the
  stage to a well: the operator drives (joystick, Drive pad). A tool that drives from the
  nominal pitch is the owner's later step (m3 §17).
- It does not capture on a click: a click on the map selects, and the press records,
  because a click does not say the stage is in the well.
- It adds no wall capture to `plate-walls`, and changes neither the *Wall* select nor
  what a side means (#327), the plugin, `plate-registration`, `RunDocument` or the
  profile's `[suggest]` tables.

### 12.9 Tests and failure modes

FM-169 (the dry run and, by #375, the square wells are refused by the controller; the
count by the plugin, decided in §12.3), FM-177 (one `_capture_well` for both button sets),
FM-181 (a capture is refused while the tool moves the stage: unchanged), FM-182 (the step
is walked in the dry-run page, on a square-welled plate and on a stand without a camera),
FM-151 (a view draws what the shared controller
holds: the walls widgets redraw with `refs`), FM-125 (a result and the document are
data), FM-139 and FM-98 (text, never markup or the error), FM-171, FM-161 and FM-162
(the browser walk), FM-147 (no widget spelling that warns), FM-112 (the venv's entry
points). The new kind, for the checklist: *the same Panel widget placed twice in a
layout*, which is drawn once, so that the second place is empty and nothing fails: a
second place gets a widget of its own.

**Not verified (for the owner)**: the walk at the Ti2-E with the joystick, Live and the
page open (steps 3 to 6 of §12.1); the time of a run on a real stand at 10× (16.5 s is
the demo's stage); whether two captured centres should make `plate-walls` the next step
(§12.4); `plate-walls` on a 4× frame.

## 13. The page names the image's sides and what a move does to it (#327)

Fixed by #327's plan (2026-11-09), with the owner's note of 2026-11-03: use the
standard of the microscope software and keep the arrows (it is the stage that
moves, so the image is normally inverted: "do not change what is done in
microscopy"), and say on the controls what a move does to the image. The frames,
the words of each and the arithmetic that maps one to the other are M3 §19
(`smc/plugins/frames.py`); this section is the page's part. FM-189 is the
failure: the operator named a wall as seen on Live, and the layer read it in the
stage's frame.

### 13.1 The captures and the drive take image sides

- **The widgets.** The *Wall* select is labelled `Wall, as on the image` and the
  *Drive to* select `Drive to (wall as on the image)`; their options stay `left`,
  `right`, `top`, `bottom` (and `centre` for the drive). They are image sides
  (`IMAGE_SIDES`, M3 §19.2). The arrows of the pad stay in the stage's axes (§6.4).
- **`PlatePage.capture_wall(well, side)`**: `side` is an image side (a `ValueError`
  for any other word, as now). After the checks it already makes (the well is on the
  plate, the stage is at rest) and before it reads the stage, it takes the
  `matrix_um_per_px` of the **latest usable** `camera-calibration` of the run document
  (`_document()`, then `_slot(doc, "camera-calibration", CameraCalibration, [])`: usable
  means what `RunDocument.unusable` says: done, not stale, of this profile, and not
  a dry run's unless this page is one), whatever its objective (M3 §19.3), and records `WallTouch(side=stage_side_of_image_side(side,
  matrix), …)`. It replaces the earlier touch of the same well and the same
  **stage** side. Two refusals, raised before anything is recorded (FM-169: the
  controller's refusal is the guard, the button's state is a courtesy):
  - no usable calibration: `walls are named on the image, and no camera calibration
    says which stage side that is: run camera-calibration (Set up), then try again`
    (`guide.NO_CALIBRATION_FOR_SIDES`);
  - a calibration whose matrix is off a quarter turn, or cannot be inverted:
    `frames`' own text.
- **`PlatePage.goto(well, side="centre")`**: `centre` is as before; an image side is
  mapped in the same way, and `goto-well` runs with the **stage** side
  (`side=+X`). The mapping comes after `_refuse_another_registered_plate` and before
  `current` changes and before a thread starts, and its refusals are the two above.
  `click(..., drive=True, side=…)` passes the select's value on to `goto`.
- **`PlatePage.walls`** holds stage-side touches, so `_registration_pairs` dumps
  what the plugin takes and a click and a step agree (§11, `_registration_pairs`).
- **`PlateView.camera_matrix: Matrix2 | None = None`**, a field added at the end of
  the dataclass: the matrix of the latest usable calibration, `None` when there is
  none. `view()` reads it with the same `_slot` call that reads the other slots, so a
  calibration whose result does not validate is a line of `problems` and is drawn as
  absent (FM-125); the file is never written.

- **Beside §12.** `plate-walls`' section (§12.5) holds centre captures only and is
  not touched. A missing calibration has a precedent there, §12.3's
  `WALLS_NEED_CALIBRATION`: the same fix (run `camera-calibration`, *Set up*) and a
  text of its own, because here it is a side that cannot be mapped and not a tool
  that cannot run. `plate-registration`'s section keeps its widgets by their names.

### 13.2 What the page says

Two pure functions in `smc/ui/guide.py` build the texts (no Panel; plain text,
FM-139). Each takes the view's `camera_matrix` and has three cases; a matrix that
`frames` refuses gives `frames`' own text, taken as `str(exc)` where it is caught
(FM-98).

```python
def image_sides_note(matrix: Matrix2 | None) -> str: ...
def drive_image_note(matrix: Matrix2 | None) -> str: ...
```

| matrix | `image_sides_note` (under the capture buttons) | `drive_image_note` (under the pad) |
|---|---|---|
| `None` | `Walls are named on the image. No camera calibration says which stage side each one is: run camera-calibration (Set up) to capture a wall or drive to one.` | `Which way the image moves for an arrow is not known: run camera-calibration (Set up).` |
| a quarter turn (the Ti2-E at 10×: `((-1.1, 0.0), (0.0, 1.1))`) | `On the image: left is the stage's +X, right -X, top -Y, bottom +Y (from the camera calibration).` | `On the image, ↑ moves it up, → moves it right; ↓ and ← the opposite.` |
| off a quarter turn | `frames`' text | `frames`' text |

In the second row the words are the matrix's: `left is the stage's {l}, right
{r}, top {t}, bottom {b}` with `l`, `r`, `t`, `b` from `stage_side_of_image_side`,
and `↑ moves it {a}, → moves it {b}` with `a` from
`image_direction_of_stage_step("+Y", …)` and `b` from `("+X", …)`.

- **Where.** `image_note`, a wrapped `pn.pane.Str` beside `capture_note`, under the
  capture row; and `drive_image_note`, a wrapped `pn.pane.Str` between the pad and
  `drive_note`. Both are set in `refresh()` from `view.camera_matrix`, the places
  the rest of the view is drawn; the tick does not rebuild them. A calibration run
  from the page therefore shows after the run's `refresh()`.
- **The button's label.** `capture_wall_label(well, side)` keeps its signature
  (§11.1) and says `Capture B3 wall at image left` (`Capture wall` with no well).
- **The capture list** shows the touch as recorded, in the stage's words: `B3 +X
  wall (17967.9, -10237.1)`; the note above it says what each image side is.
- **A pad that says nothing is not a pad that works on every stand**: the notes are
  informative and disable nothing; the pad's own refusals are §6.

### 13.3 Tests and failure modes

- `tests/unit/test_ui_guide.py`: the six cells of the table above, written out;
  `capture_wall_label`'s new text.
- `tests/unit/test_plate_page_controller.py`: with a calibration seeded in the run
  document (as `tests/test_scan_plan_demo.py`'s `_seed` writes one), `capture_wall(
  "B3", "left")` records `+X`; the same call with no calibration is refused with the
  text above and records nothing; a calibration 6.0° off a quarter turn is refused;
  a later capture of the same stage side replaces the first; `goto("B3", "left")`
  runs `goto-well` with `side=+X` and with no calibration is refused before
  `current` changes and before a run starts.
- **The end-to-end check of FM-189**, `tests/test_plate_page_walls_demo.py` (new,
  unique basename): the demo stand, a calibration whose matrix is the Ti2-E's, the
  stage moved to the four touches of M3 §17.1 and each captured under its image
  name; `plate-registration` through the page gives B3's centre at (10267.7,
  −10038.05) within 0.05 µm. Another test captures the incident's two touches
  (`left`, `bottom`) and finds (10167.9, −10134.1).
- `tests/unit/test_plate_page_drive_view.py` (or a view test of its own): the two
  notes carry the texts of the table for a view with and without a matrix, and the
  labels of the selects are the ones above.

FM-189 (the check itself), FM-169 (every refusal is the controller's), FM-125 (a
result the page cannot read is a problem, not a crash), FM-98 and FM-139 (text, never
the error; plain text), FM-177 (the label a capture acts on: the touch's stage side is
read at the press, from the document then, and a calibration cannot change during a
press because a capture is refused while a run goes).

### 13.4 Not here

- It does not change the pad's axes, the Z direction or the plate map: the map is the
  stage's, with A1 at the top left of what `useq` draws.
- It does not let the page capture a wall without a calibration (M3 §19.6).
- It does not draw anything on Live (M3 §19.6).

**Not verified (for the owner)**: at the Ti2-E, after `camera-calibration` at 10×,
the notes read `left is the stage's +X, …` and `↑ moves it up, → moves it right`
(M3 §19.7), and agree with NIS-Elements' own XY control; B3's four walls captured
under their image names give a centre within a few tens of µm of the one the walls'
midpoints give; the page walked by a newcomer, who names a wall as they see it.


## 14. The map draws the scan plan's fields, and the card says how long the scan takes (#320)

Fixed by #320's plan (2026-11-09). Finding 4 of the owner's walk of the first run
(#310): after `scan-plan done` the map was unchanged, so a newcomer could not see what
`scan` was about to do, and the card did not say how long it would take.

**Measured 2026-11-09 on `demo-sample`** (Python 3.13.0, Panel 1.10.0, Bokeh 3.10.1; a
throwaway script, not committed): `plate-registration`, then `scan-plan wells=B2,B11
coverage=0.4` through the page's controller:

- the result holds 72 `PlannedField`s (6 × 6 in each of the two wells), `fov_um` is
  `(512.0, 512.0)`, `overlap` 0.1, `estimate_s` is `None`; `wells` is `['B11', 'B2']` and
  the fields come B11's first: **the visiting order is not the alphabetical one**;
- the card (§2.4) is `plate_plan: 5 entries`, `wells: 2 items`, `objective: ...`,
  `pixel_size_um: 1`, `matrix_um_per_px: 2 items`, `matrix_source: profile`,
  `image_shape_px: 2 items`, `fov_um: 2 items`, then `… 5 more`: `fov_um` is the 8th
  line, and **`fields` and `estimate_s` are in the cut part**, so the estimate never
  reached the page, `null` or not;
- after `timing` (5.7 s), the same `scan-plan` has `estimate_s` 42.16 (72 fields,
  `move_ms_per_mm` 100.5, `refocus_ms` 400, `snap_ms` 18.1);
- running `timing` again makes `scan-plan` stale: the checklist says `scan-plan read
  its inputs from an earlier run; run it again` and `view()` holds the card with
  `stale` set, as §2.4 says;
- `ScanPlanResult.model_validate` on 5000 and 13,824 fields takes 4 and 9 ms; a Bokeh
  figure with one `rect` glyph of N fields (four columns of N numbers) serialises to
  6.7 kB (72 fields, 8 ms), 455 kB (13,824, 24 ms), 672 kB (20,000, 32 ms) and 1.7 MB
  (50,000, 68 ms): about 33 bytes a field.

### 14.1 The layer

A layer of the plate map, **the fields of the usable scan plan**: one rectangle per
`PlannedField`, `fov_um[0]` wide and `fov_um[1]` high, centred on `(x_um, y_um)`, in
stage µm like the wells. The rules:

- **The same rule as the density layer.** `view()` reads the plan with
  `self._slot(doc, "scan-plan", ScanPlanResult, problems)`: the layer exists only
  while the slot is usable, so a plan that is stale (it read a plate or a timing that
  was produced again since, measured above), made on another profile, or a dry run on
  a page that is not one, draws nothing. The card of such a plan stays, marked as §2.4
  says. A plan that does not validate is `scan-plan: result does not validate: ...` in
  `problems`, drawn as absent (FM-125).
- **The plan's wells, not the selection.** The layer draws every field of the plan
  whatever the *Wells* box says now: the plan is what `scan` will do, and a selection
  changed since has not changed the plan.
- **The footprint's bounding box.** `fov_um` is the footprint's extent *in stage
  axes* (`ScanPlanResult`'s docstring): exact for a camera at 0° or 90°, larger than
  the frame for another angle. The layer draws that box and does not claim more.
- **No field is hoverable** and none is labelled: the wells keep the hover
  (`hover.renderers` stays the wells' renderer), the title and the card carry the numbers.
  The visiting order is not drawn (no line between fields: the route draws its own
  once it exists, and a second line would hide it).

```python
#: smc/ui/controller.py
MAP_FIELDS_UP_TO = 5000


@dataclass(frozen=True, slots=True)
class PlanLayer:
    """The scan plan as the map draws it, in stage µm; ``fields`` is empty when there are too many."""

    count: int  # every field of the plan, drawn or not
    wells: int  # len(result.wells)
    fov_um: tuple[float, float]
    fields: tuple[tuple[float, float], ...]  # (x_um, y_um) in visiting order, or ()
```

`PlateView` gains `plan: PlanLayer | None = None` as its **last** field. `view()` builds
it from the slot: `None` when the slot is not usable; otherwise `PlanLayer(count=len(
result.fields), wells=len(result.wells), fov_um=result.fov_um, fields=<the centres>
if count <= MAP_FIELDS_UP_TO else ())`. A `fov_um` with a side that is not greater
than 0 is `None` and the line `scan-plan: fov_um <fov_um> is not positive; its fields
are not drawn` in `problems` (the model does not bound it, and a rectangle of zero or
negative size is invisible or flipped).

**The bound** (FM-140: bounded per unit and not in all). A plan holds up to 100 × 100
fields per axis in each well (the plugin refuses a larger derived grid), so 1536 wells
at 3 × 3 is 13,824 fields and a plan of 1536 wells at 100 × 100 is 15 million. Past
`MAP_FIELDS_UP_TO` = 5000 fields **nothing is drawn, not a first part**: a first
part would show an arbitrary corner of the plan as if it were all of it. The title
says so (§14.3). The figure is chosen, not measured: 5000 is about 165 kB on the
websocket at each redraw (33 bytes a field, measured above) and more rectangles than
the eye separates on a plate. The 15 million fields are the plugin's concern (the
result would be a document of gigabytes); the bound here keeps the page's payload
bounded whatever a document holds. `view()` is called when the page redraws (a run
ended, the document changed, a click), not by the tick (FM-145); validating a plan is
the 9 ms above.

### 14.2 The page (`smc/ui/plate_page.py`)

`PlatePageView` gains `fields_source = ColumnDataSource({"x": [], "y": [], "w": [],
"h": []})` and one `figure.rect("x", "y", "w", "h", source=self.fields_source,
fill_color="#ff7f0e", fill_alpha=0.2, line_color="#ff7f0e", line_width=1)`, **added
in `__init__` before the route's line**, so that the route is drawn over it, and above
the wells (which are `underlay`). `refresh()` sets `fields_source.data` from
`view.plan` (the `fov_um` repeated in `w` and `h` for each row; empty columns when
`plan` is `None` or has no fields) and `_draw_wells` ends the title with `plan_title(
view.plan)` (§14.3). The orange is a first choice, chosen to differ from Viridis
(the wells' fill) and from the route's red; `/ui-check`'s PNG decides whether it
reads, and a change of colour is not a deviation.

### 14.3 What the title and the card say (`smc/ui/guide.py`, pure)

```python
def plan_title(
    plan: PlanLayer | None, *, drawn_up_to: int = MAP_FIELDS_UP_TO
) -> str: ...
def duration_text(seconds: float) -> str: ...
def scan_plan_card_lines(result: ScanPlanResult) -> tuple[str, ...]: ...
```

(`MAP_FIELDS_UP_TO` lives in `guide.py`, where `controller.py` imports it from; the
`drawn_up_to` keyword is for the test of the bound.) The figure's title, as `_draw_wells`
sets it today (`<plate>: the registered plate`, or the nominal text), gets one suffix:

| `plan` | Suffix of the title |
|---|---|
| `None` | none (as today) |
| 72 fields in 2 wells | `; scan plan: 72 fields in 2 wells` |
| 1 field in 1 well | `; scan plan: 1 field in 1 well` |
| 20000 fields in 1536 wells | `; scan plan: 20000 fields in 1536 wells, too many to draw (over 5000)` |

`duration_text` rounds to the nearest second (Python's `round`), then: under a minute
`<s> s`; under an hour `<m> min <ss> s` (two digits); from an hour `<h> h <mm> min`
(seconds dropped). `0.4` is `0 s`, `42.16` is `42 s`, `192` is `3 min 12 s`, `59.6` is
`1 min 00 s`, `3900` is `1 h 05 min`, `3599.6` is `1 h 00 min`.

**The scan plan's card** is built from the result, like `walls_card_lines` (§12.6),
instead of its first keys, which cut `estimate_s` away (measured above). `_card`
validates `record.result` as `ScanPlanResult` for a `done` record (a result that does
not validate keeps the generic card, never an exception: FM-125) and takes these lines:

```text
wells: 2 (B11, B2)
fields: 72, a grid of 6 × 6 in each well
field: 512 × 512 µm at Nikon 10X S Fluor, overlap 0.1
estimate: about 42 s (the moves, refocus and snaps of the timings)
```

- `wells: <n> (<names>)`: the names in the plan's order (`result.wells`), at most six,
  then `, … <n - 6> more` inside the parentheses (`wells: 96 (B11, B2, B3, B4, B5, B6, … 90 more)`).
- `fields: <len(fields)>, a grid of <rows> × <columns> in each well` when `len(fields) ==
  rows * columns * len(wells)`; else `fields: <len(fields)> of <rows * columns *
  len(wells)>, a grid of <rows> × <columns> in each well, the fields that miss the
  well left out`. The two numbers say that clipping happened; the count is the
  result's own.
- `field: <w:g> × <h:g> µm` then ` at <objective>` when it is not `None`, then
  `, overlap <overlap:g>`.
- `estimate: about <duration_text(estimate_s)> (the moves, refocus and snaps of the
  timings)` when `estimate_s` is not `None`; otherwise **`estimate: unknown, no usable
  timing: run timing, then scan-plan again`**. What the figure sums is what
  `estimate_s()` sums (`scan_plan/algorithm.py`): each move, a refocus when the stand
  has an autofocus and the timings know its cost, each snap. It does not include a
  channel change or the cells, the report or the export, and the text says "the moves,
  refocus and snaps" so as not to claim more. An unknown is not a zero (CLAUDE.md,
  *unknown beats guessed*): the line says what to run, and says it once.

The `×` is U+00D7; the source carries `# noqa: RUF001` as `scan_plan/plugin.py` does.
Four lines, never more than `_CARD_LINES`; each goes through `_cut(_flat(...,
_CARD_TEXT_CHARS))`, the limit of a diagnosis (600) and **not a value's 80**: the
clipped form of the `fields:` line is 90 characters and its end says why the two counts
differ, so a cut at 80 would drop exactly the words that explain them (found by the
adversarial review of #320's develop session, `## Plan deviation` on the issue). It is
still bounded (FM-139: plain text, an objective's label is the stand's and may hold
anything: a line break is a space and 10 kB is cut at 600).
`assumed` is unchanged (`no camera calibration for ...`). A scan plan that did not end
`done` has no lines (FM-154), and a **stale** one keeps them (§2.4): the estimate on a
stale card is of the timings it read, and the card is marked stale.

### 14.4 What this does not do

- It does not draw the visiting order, a field's name, a hover on a field, or the
  fields of a plan the operator has not run (a selection is not a plan).
- It does not make `timing` a next step: `timings` is an optional slot, never a step
  (§8.3). The card says what to run, which is where the operator reads the plan.
- It does not draw the fields of `scan`'s result (what was taken) or of `route`'s
  stops: the route has its own layer.
- It does not rotate a field: the box is in stage axes (§14.1).
- It does not change `scan-plan`, `ScanPlanResult` or the run document.

### 14.5 Tests and failure modes

Named in #320's plan. **Failure modes**: FM-125 (a result and a `fov_um` taken as they
come), FM-139 (plain text), FM-140 (the bound in all), FM-145 (`view()`, not the
tick), FM-151 (a second tab draws the shared plan), FM-154 (no lines unless `done`),
FM-170 (an unknown is said with what makes it unknown), FM-204 (a fixture whose wells
are not in alphabetical order: B11 before B2 is the demo's), FM-162 (the browser walk
names what it did not reach).

**Not verified (for the owner)**: the map at a real plate's zoom on the PC's screen
(5000 rectangles, and 72 of 512 µm at the full plate's extent, are a few pixels each
in headless Chromium: `/ui-check` shows what a newcomer sees at the default zoom, not
the stand's screen); the estimate against the time of a real scan on the Ti2-E
(`timing` measured it on the demo's simulated stage here).

## 15. The map says where the stage is on the plate (#410)

Fixed by #410's plan (2026-11-10). The owner's feedback from Ti2-E session 3
(#405), the three requests of the issue: show where the stage is on the plate
before any calibration, as a cross that follows it in real time; and register from
one position in each of three wells. This section decides what the map draws and
says before a registration exists, what the cross follows, and what the
three-well registration is. It adds one profile table and one field of `PlateView`;
it changes no tool, slot or run document.

**Measured 2026-11-10** on `origin/main` at `41e6993` (useq-schema 0.9.2,
Python 3.13.0, Panel 1.10.0, Bokeh 3.10.1), with throwaway scripts on `Profile.demo()`
and on `smc.plugins.plate_registration.algorithm`, not committed:

- **The cross exists.** `PlatePageView.stage_source` is a `cross` scatter that `tick`
  sets from `PageStatus.xy_um` once a second (`status()` reads `Microscope.state()`;
  0.7 ms on the demo). On a `nominal` map it is drawn at the stage's XY on a plate
  whose A1 is at (0, 0): on the Ti2-E, 48.6 mm and 29.2 mm from where the Greiner
  24-well's A1 lies (below), so it shows the stage and no well.
- **Live already reads the position ten times a second.** With Live on, the frame
  numbers over one second were 14 to 23, and every `Frame.xy_um` is the stage's
  position read right after that frame's snap (`_snap_once`: `state()` after
  `snap`). After `jog_xy(300, 0)` from (1000, 2000) during Live, `page.frame().xy_um`
  and `page.status().xy_um` both read (1300.0, 2000.0). A cross that follows the
  frames needs no device read of its own.
- **Where the plate lies on the Ti2-E.** The centre of the plate's grid of wells
  is a stage point, and a plate of equal pitches and round wells turned 180° lands
  on itself, so the centre does not depend on which way round the plate lies. From
  session 3's B3 (10 003, -10 051) µm, at its measured pitches of 19.49 and
  19.43 mm and the plate turned (columns towards -X, rows towards +Y, #414): the
  centre is (258, -336) µm. From session 2's B3 (10 267.7, -10 038.1) µm at
  19.45 mm: (543, -313) µm. **Both lie within 0.55 mm of the stage's origin,
  285 µm apart in X** (the plate was taken off and put back between them). Their
  mean is (400, -325) µm to 5 µm.
- **The Greiner plate centred there** (`greiner-24-well`, #400): A1 (-48 225,
  28 850), B3 (-9 325, 9 400), B5 (29 575, 9 400), C3 (-9 325, -10 050) µm. Row D
  (y = -29 500) is outside the Ti2-E's soft limits of ±29 000 µm: the map shows
  wells the stage cannot reach, as `docs/plugins/plate-registration.md` says of a
  96-well plate.
- **The tolerances of the two tools that take positions by hand.** `plate-walls`
  takes a rough position up to half the well's radius from the centre
  (`search_um`: 3 900 µm on a 24-well, `docs/plugins/plate-walls.md`): ten times the
  estimate's error. `plate-registration` takes centres and refuses a fit whose
  worst residual is beyond `GOOD_RESIDUAL_UM` = 300 µm. Three synthetic centres
  of the Greiner plate, each off by at most 130 µm: residual 121.6 µm, rotation
  0.10°, A1 off by 49.6 µm. One centre 900 µm off in X and the other two exact:
  residuals B3 572.8, B5 351.5, C3 247.2 µm, so the plugin refuses `B3 is 573 µm
  from where the others put it: centre it again`. Three centres each about 290 µm
  off (hand centring at the plugin's own limit) gave 300.5 µm, which the plugin
  also refuses.
- **A drive click on a map with no registered plate moves nothing**: the click on
  B2 of the nominal 96-well map returned `B2`, `goto-well` ended `UpstreamError:
  goto-well reads plate, but the run document has no plate`, and the stage stayed
  at (0, 0).
- **The lookup of a well under a point.** `WellGeometry(...).contains` built from
  a `WellView`'s fields and tried on all 1536 wells of a 1536-well plate: 0.19 ms.
- **`demo-sample`** centres its plate on the stage origin by construction
  (`[sample] a1_center_xy_um` is `None`, `SampleSection`).

### 15.1 Decisions

**D1. The cross keeps its tick, and follows the frames while Live runs.**

- `tick` keeps drawing the cross from `status().xy_um` once a second, as today.
- While Live runs, `frame_tick` draws it from the new frame's `xy_um`, ten times a
  second, and `tick` leaves it alone. A session holds `self._cross_from_frames:
  bool`, false at the start. `frame_tick` sets it when it draws a frame whose
  `xy_um` is not `None` with `page.is_live` true, and `tick` sets it false when
  `page.is_live` is false, and draws the cross from the status in that case and
  when the flag is false. A frame that carries no position (`xy_um` is `None`)
  draws no cross and leaves the flag as it was.
- It reads nothing the controller does not hold: `frame_tick` reads `page.frame()`
  and `page.is_live` and never the status or the view (FM-145). A snap that is not
  Live never moves the cross: its frame is old by the time it is drawn.
- **A faster cross outside Live is not decided here.** It would need a read of the
  stage alone, whose cost on the Ti2-E is not measured (#333 saw `state()` wait up to
  0.5 s during Live there). The Report names the lag: one second.

**D2. The estimate: the profile says where its holder puts the plate.**

A new optional table of the profile:

```toml
[holder]
plate_centre_xy_um = [400.0, -325.0]  # stage XY of the centre of a plate's grid of wells, seated in its holder
measured = "2026-11-10, sessions 2 and 3 (#334, #405): the Greiner 24-well's centre was (543, -313) and (258, -336) um; their mean to 5 um"
```

- **It is one point for every plate the holder takes.** An SBS plate's grid is
  centred on its footprint (assumed from the standard; measured for the Greiner
  24-well only), so the point is the same for a 96-well and a 24-well, and for a
  plate turned 180°. A plate of another footprint (a slide, a dish) is not an SBS
  plate: the title says "estimated" and nothing more is claimed.
- **A value needs its provenance**, as an objective's offset does (m2 §13.3):
  `plate_centre_xy_um` without a non-blank `measured` is refused, naming both. The
  value is two finite numbers within ±150 000 µm (a stage is not longer: a larger
  value is a unit mistake), and a TOML boolean is refused (FM-104, #106).
- **`profiles/nikon-ti2.toml`** gets the table above. **`profiles/demo-sample.toml`**
  gets `plate_centre_xy_um = [0.0, 0.0]` with `measured = "by construction: the
  sample's plate is centred on the stage origin (m1 §8)"`: the sample's wells are
  where the estimate puts them, so a test of the estimate can be checked against
  the camera's image. A `[sample]` that sets `a1_center_xy_um` makes that value
  wrong, and the map still says "estimated". `demo.toml` and `nikon-tie.toml` get
  no table: their plate is *unplaced* (D3).
- **Why the profile and not the last registration.** The profile is read, reviewed
  and versioned, and says where the number comes from. A remembered registration is
  state outside the repository that goes stale without a word, one per plate and
  stand. Writing the table from a registration (a command, as `smc objective offset
  --write` does for `[objectives]`) is a follow-up, not here.
- **An estimate is never a registration.** It fills no slot of the run document,
  moves nothing and feeds no tool: it is drawn.

**D3. Four poses.** The map is drawn in one of four states, and says which.

| `MapPose.kind` | When (first that holds) | Where A1 is drawn | `PlateView.nominal` |
|---|---|---|---|
| `registered` | a usable `plate` slot names the page's plate, its `method` is `references`, `lattice-fit` or `wall-images` | the slot's A1 and rotation | false |
| `seed` | the same, and `method` is `seed` | the slot's A1, rotation 0° | false |
| `estimated` | no such slot, and the profile's `[holder] plate_centre_xy_um` is set | the holder's centre minus the grid's centre offset from A1 (the mean of `useq`'s well positions for the plate at the origin), rotation 0° | true |
| `unplaced` | otherwise | (0, 0), rotation 0°, as today | true |

`nominal` keeps its meaning for every user of it: nothing but the title read it,
and `goto-well` still reads only the run document's slot.

**D4. The title says which, in these words.** `guide.map_title(plate, pose)`:

| Pose | Title (before `plan_title`'s suffix) |
|---|---|
| registered | `96-well: the registered plate` (as today) |
| seed | `96-well: placed from one well (B3), not registered: the rotation is assumed 0°, not measured` (a slot with no reference well: `96-well: placed from one well, not registered: ...`, the rest the same) |
| estimated | `24-well: estimated from the stand's holder, not registered: centred at stage (400, -325) µm, not turned, A1 at the top left of this map` |
| unplaced | `96-well: not registered, and this stand does not say where its plate lies: A1 is drawn at stage (0, 0), so the cross shows the stage and not a well` |

The numbers are `round`ed to whole µm (`:.0f`). The `unplaced` text replaces
`nominal map (A1 at 0, 0, not turned), the plate is not registered`, which the
operator could not act on. A title is plain text, never markup (FM-139).

**D5. The cross is labelled with the well it is over.** A `LabelSet` on
`stage_source` (a `text` column), drawn beside the cross:

```python
def stage_label(
    wells: Sequence[WellView],
    pose: MapPose,
    xy_um: tuple[float, float] | None,
) -> str: ...
```

- `""` when `xy_um` is `None`, when the pose is `unplaced` (the cross is on no real
  well), or when the point is on no well (`WellGeometry.contains`, built from the
  view's `WellView`s: the one place that knows a well's shape).
- Otherwise `over B3`, then `" (estimated)"` for an `estimated` pose and `" (rotation
  assumed)"` for a `seed`; nothing after it for `registered`. The first well in the
  view's order that contains the point wins; wells do not overlap.
- Computed from the **last drawn view** (`PlatePageView._view`), by `tick` and by
  `frame_tick`; it never asks for a new view (FM-145). `stage_source.data` is set
  with `x`, `y` and `text` at once.

**D6. A name read off an estimate is an assumption, so nothing acts on it.**
The estimate puts A1 at the top left of the map and does not know how the plate
lies in the holder. A plate turned 180° swaps every name with the same wells at the
same places (row r becomes R-1-r, column c becomes C-1-c: B3 is C4, #422), and no
residual shows it. So:

- **No capture takes its well from the cross.** *Capture centre of B3* still records
  the well the operator clicked (§11.1); the cross says where the stage is, the
  operator says which well it is.
- **Drive stays a registered plate's.** A drive click on an estimated map moves
  nothing (measured above), as on every map with no usable `plate`.
- The title says "A1 at the top left of this map" and the label "(estimated)": the
  operator can compare with the A1 printed on the plate. The choice of orientation
  is #422's, which will place the estimate by it.

**D7. The registration from one position in each of three wells needs no new tool.**
`plate-registration` already fits any number of centres from two up: three of them
fit the plate and check each other (a rotation and a translation from six
numbers), and refuse a centre that is far from where the others put it. The page
already records one centre per well (*Capture centre*, §11.1, §12.5) and runs the
tool. What the plan adds is proof on the lab's plate (the tests named in #410's
plan) and the page's words. What is *not* new, and the owner should know:

- **By eye is only as good as the eye.** `plate-registration` needs centres good to
  a few hundred µm, which an operator cannot judge in the middle of a 15.6 mm well at
  the 10× (the field is about 1.4 mm). The tool that takes a rough position and
  measures is `plate-walls`, whose rough position may be 3.9 mm off: that is the
  path the page offers first when two or more centres are captured (§12.4). Whether
  it registers the Greiner plate from one rough position in each of three wells is
  **verified at Ti2-E session 4** (#404's criterion, with the time it takes).
- **`CAPTURE_WELLS` says three.** It becomes `centre a well under the objective
  (Live and the Drive pad), click it on the map and press Capture centre; three
  wells far apart fit the plate and check each other` (`guide.py`; its test's
  literal changes with it).

**D8. Not here.**

- Registering from the estimate itself (the plate "where the holder puts it", with
  its error said) so that a few wells can be scanned at once. It needs a decision
  about `PlateResult.method` (a slot model m3 §3.2 fixes), `scan`'s refusal of an
  unmeasured plate, and what `assumed` says: a design of its own, the owner's call.
- The plate's orientation (#422): the choice, the names it changes, the map's
  back and front. The estimate assumes "not turned" until then.
- Writing `[holder]` from a registration; a holder per plate type; the cross
  faster than a second outside Live (D1); a click on the estimated map that
  records a position.

### 15.2 The interface

```python
#: smc/hardware/profile.py
MAX_HOLDER_CENTRE_UM = 150_000.0


class HolderSection(_Section):
    plate_centre_xy_um: tuple[float, float] | None = None
    measured: str = ""
```

`Profile` gains `holder: HolderSection = Field(default_factory=HolderSection)`, after
`objectives`. The validators are the ones of `ObjectiveOffsetEntry.xy_offset_um`
(two numbers, no boolean, finite, bounded by `MAX_HOLDER_CENTRE_UM`) and the
provenance check of its `_a_value_needs_its_provenance`.

```python
#: smc/ui/controller.py
PoseKind = Literal["registered", "seed", "estimated", "unplaced"]


@dataclass(frozen=True, slots=True)
class MapPose:
    kind: PoseKind
    holder_centre_um: tuple[float, float] | None = None  # "estimated" only
    seed_well: str = ""  # "seed" only: the one reference well
```

`PlateView` gains `pose: MapPose = field(default_factory=lambda: MapPose("unplaced"))`
as its **last** field. `PlatePage._registered_geometries(registered)` returns
`(geometries, pose)` where it returned `(geometries, nominal)`; `view()` sets
`nominal=pose.kind in ("estimated", "unplaced")` and `pose=pose`; `click` ignores it
as it ignores `nominal` today. The geometry's cache key stays `(plate, a1,
rotation)`, with the estimate's A1.

```python
#: smc/ui/guide.py (pure; MapPose and WellView are imported for typing only)
def map_title(plate: str, pose: MapPose) -> str: ...
def stage_label(
    wells: Sequence[WellView], pose: MapPose, xy_um: tuple[float, float] | None
) -> str: ...
```

`PlatePageView` gains `self._cross_from_frames`, the `LabelSet` on `stage_source`, a
private `_draw_cross(xy_um)` that sets `stage_source.data` (`x`, `y`, `text`),
`tick` and `frame_tick` as D1 says, and `_draw_wells` sets the title from
`map_title(view.plate, view.pose)` followed by `plan_title(view.plan)`.

### 15.3 Tests and failure modes

Named in #410's plan. **Failure modes**: FM-73 (the estimate's tests use a plate
whose grid is not square, the 96-well, beside the symmetric Greiner plate, so that
swapped axes or a sign flip show), FM-104 (a boolean where a number is expected),
FM-125 (a profile value taken as it comes), FM-139 (plain text), FM-145 (the label
is drawn from the last view; the tick never rebuilds it), FM-151 (a second tab
draws the same pose and the same cross), FM-170 (an unknown says what is unknown:
the `unplaced` title), FM-162 (the browser walk names what it did not reach). A new entry of the checklist, named by its text when the develop
session writes it: *an estimate drawn like a measurement* (a map that places a plate
from an assumption and labels wells by it must say on its face which of measured,
seeded, estimated or unplaced it is, because a name read off it is trusted; a test
for each state).

**Not verified (for the owner)**:

- The estimate against the plate on the Ti2-E: the Greiner plate put in its holder
  (either way round) and the page's cross over a well it is physically over. The
  value of `[holder]` is the mean of two sessions 285 µm apart; a plate put back
  moved by that much.
- The label's names against the printed plate: with the plate turned 180° the page
  says `over B3` where the plate's own label is C4 (D6, #422).
- The cross at the PC's screen and zoom, and its lag at the stand: a second outside
  Live, a tenth of a second in Live (the frame's own read; `state()` on the Ti2-E
  was seen to take up to 0.5 s during Live, #333).
- `plate-walls` from one rough position in each of three wells on the Greiner plate,
  and the time it takes (D7).
