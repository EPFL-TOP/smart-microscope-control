# The plate page

A web page on the microscope PC: the plate drawn in stage µm, wells you select with
a click or a typed range, a **Snap** and a **Live** view of the camera's frame, every registered
tool with a Run button and the result of its last run, and, once the workflow has
run, the density of each well and the route. It is **a view only**: every button is a
[plugin](../plugins/README.md) run with the parameters `smc run` takes, or a read of
a capability, so each has a command-line equivalent
([ADR-0006](../adr/0006-user-interface.md)). The design is
[`m3-plugins.md`](../design/m3-plugins.md) §15 and
[`m5-web-ui.md`](../design/m5-web-ui.md); this page is how to use it.

It needs the `ui` extra (Panel):

```text
pip install -e ".[ui]"
```

Without it `smc ui` exits 1 and says so; the rest of `smc` does not need it.

## Starting it

```text
smc ui -p <profile> --run-doc run.json [--port 5006] [--dry-run]
```

`smc ui plate ...` still works and serves the same page (it is the name this page
had before it grew). Any other word after `ui` exits 2 with `unknown page <word>:
smc ui serves one page`, before anything else is looked at.

`--run-doc` is the same run document `smc run` takes, and is required for the same
reason: a file is never created where you did not say. The command checks it and
serves the page **with the stand closed**: nothing is opened until you press
**Connect** on the page, so a page left running, or run as a service, never takes the
stand from NIS-Elements or ZEN. It prints

```text
serving the plate page on http://localhost:5006/ — Ctrl-C stops it
```

Open that address **as printed** (`localhost`, not `127.0.0.1`: the page's websocket
accepts the origin it was served from). The page listens on `localhost` only and
has no login: whoever reaches the port can move the stage, so it is not offered on
the network. To use it from another desk, use the PC's remote desktop (RDP). An SSH
tunnel is **untried**: it could work only with the same port number on both ends.

Unless it is a dry run, a stand whose profile declares that it refuses `stop()` gets
its `!` line first (`! the profile declares that this stand refuses stop() on ...`),
read from the profile, before anything is opened.

### Connect and Disconnect

The row above the strip holds **Connect**, **Disconnect** and one line that says
where the stand is:

| The stand | The row says | Enabled |
|---|---|---|
| closed (the page as it opens) | `<stand> is not connected: Connect opens it, and the controls that need the stand are disabled until then` | Connect |
| Connect is opening it | `connecting <stand>…` | neither |
| held | `Disconnect releases <stand> for another program (NIS-Elements, ZEN)` | Disconnect |
| Disconnect is releasing it | `disconnecting <stand>…` | neither |

`(dry run)` is added when it is one. Both buttons are disabled while anything goes
(a run, a snap, live, a drive action, a connect, a disconnect), and the capture
buttons with them. A slow stand takes seconds to open: `connecting…` is what says the
page is not stuck.

- **Connect** opens the profile the command names (one PC serves one stand; the demo
  is another `smc ui -p demo` on another port), on the page's worker. It says
  `connect: <stand> connected, <n> devices`, or why it could not, on one line: a
  stand that NIS-Elements holds is `connect: CoreError: ...`, and the page stays
  closed. STOP pressed while it opens halts the stand once it is open, and the line
  ends `, and halted: STOP was pressed while it connected`. A new stand is never
  halted: Connect clears the last STOP's note.
- **Disconnect** releases the stand for another program: `disconnect: <stand>
  released`. It waits, for up to 5 s, for a status read or a STOP that is inside the
  stand, then stops what still moves and unloads the devices, and **switches
  nothing off**: a light or an autofocus stays as the stand keeps it. It is refused
  while a run, a snap, live or a drive action goes (`<name> is running`), even if a
  browser sends the click. It says `released` only when it is true:
  - `disconnect: <stand> released, with <n> read(s) still in flight after 5 s`: a
    read did not end in time, and the devices were unloaded under it. The page showed
    a line, never a crash;
  - `disconnect: MicroscopeBusyError: <why>; the stand is still connected`: another
    call held the stand too long (a driver that does not answer), nothing was
    unloaded, and **the page is connected again**: press Disconnect to retry;
  - `disconnect: <stand> may still be held: its devices did not unload (<why>);
    restart smc ui before another program takes it`: the driver would not let go. The
    page is not connected, but the stand may be held until the process ends: do not
    start NIS-Elements or ZEN before restarting `smc ui`;
  - `disconnect: <Type>: <why>` for any other error: the page has let go of the
    stand, which may be half unloaded.
- **With the stand closed** every control that needs it is disabled, and the row's
  line is the reason: the strip says `<stand> | not connected` (and `DRY RUN`), every
  tool's note and the Drive region say `the stand is not connected` once, Snap,
  Live and the captures are greyed. Every tool needs the stand, as `smc run` opens it
  for every plugin. The map, the selection, the checklist and the result cards work
  all the same: they read the profile and the run document, not the stand. The last
  frame stays on the image after a Disconnect, with its time.
- The profile is read **once**, when the command starts: an edited profile takes
  effect when the command (or the [service](service.md)) restarts.

To run the page for weeks on a microscope PC, as a Windows service that starts at
boot, see [The plate page as a service](service.md).

| Refused | What it says |
|---|---|
| A word after `ui` other than `plate` | exit 2, `unknown page <word>: smc ui serves one page`; the stand is not opened |
| No `--run-doc` | exit 2, `a run needs --run-doc` |
| The `ui` extra is missing | exit 1, `install the ui extra: pip install -e ".[ui]"` and the import error |
| The port is taken (on `127.0.0.1` or on `::1`) | exit 2, `port N is in use; choose another with --port`; the stand is not opened |
| The run document cannot be read | exit 1, its reason; the file is left as it is |
| The server stops by itself | exit 1, `the plate page's server stopped on its own (is port N taken?)` |

**Ctrl-C** ends the server. If a run is going it is **told to stop first** (the same
stop as the STOP button, sent before the server is touched, because stopping the
server can wait for a slow read of the stand), the page waits up to 10 s for the run,
then stops the server (up to 10 s, and says so if it does not), and the stand, if
Connect opened it, is released after. What the command prints last depends on what happened:

| What happened | The last lines |
|---|---|
| The stand took the stop and the run ended stopped | `<plugin> was stopped`, exit 0 |
| The stand refused the stop (the Ti2-E's does) | `<plugin> told to stop; halted, but <why>`, then how the run ended: a move under way runs to its target, so it is never "stopped" by default |
| The run failed while it was stopped | the stop's line, then `<plugin>: <Error>: <message>`, exit 0 |
| The run did not end within the wait | the stop's line, then `✗ <plugin> did not end within 10 s`, **exit 1**; the stand is released under a run that is still alive |
| A Connect was still opening a slow stand | the command waits up to 8 s more, then `connect: <stand> was opened after the page closed, and released` (exit 0), or `✗ connect: <stand> is still opening after 18 s: the process ends now, and whether the stand is left connected is not known` (**exit 1**) |

A Ctrl-C with the page open in a browser prints **no traceback**: stopping Bokeh's server
logs an `AssertionError` (`Uncaught exception GET /ws`) for each open session, which is
how it closes them, and the command drops that one record while it stops the server and
counts it (`the server's stop closed 1 open sessions`, in the page's log, not on the
console). Any other error of the page is still logged. Not every Bokeh version raises
it: the command is silent on Bokeh 3.9 without the filter.

A second Ctrl-C during the wait still releases the stand, and the command exits 130;
the stop was already sent. The line it prints still names the motions the layer was
tracking when the stand was released. A closed or reloaded browser tab does **nothing** to a run:
the page belongs to the server, not to the tab. Only STOP, Ctrl-C of the command and
the plugin's own end stop a scan. Two tabs on one server share the selection, the plate
and the captures: a tab draws what another tab changed within a second.

## What is on the page

**The strip, at the top.** The stand's name (the profile's `microscope.name`), then
the stage `XY (x, y) µm`, `Z`, the objective, the autofocus
(`autofocus locked`, `autofocus on, not locked`, `autofocus off`, `autofocus ?` when a
read failed), the light, `DRY RUN` when it is one, and `HALTED` with **Resume** when the
stand is halted. `?` is an unknown, never a zero. With the stand closed it is
`<stand> | not connected` and nothing else. It is read once a second, and no
faster: a scan must not be slowed by a status poll. **STOP is always offered**, a run
or not. Under it: the run's progress as `done/total message`, how the last run ended
(`<plugin> done in 12.3 s`, `<plugin> stopped`, or `<plugin>: <Error>: <message>`) and
what the last STOP did. A run that ended `done` and **assumed** something says its first
assumption: `scan done in 0.0 s; assumed: dry run: 36 fields walked, no frame taken`, cut
at 160 characters, then `(+2 more, on the card)` when there are others; the tool's card
holds them all. **Refresh** also forgets how the last run ended, so that a line like
`cells: ... run scan again` does not outlive the chain that was run elsewhere; the
redraw that a run's end makes keeps it.

**STOP says what it did.** `<plugin> told to stop; stopped: the stand is halted
(Resume clears the halt)`, or `halted, but <why the stand refused>`, or `the stop failed:
<why>`. On a stand whose stages refuse a stop (the Ti2-E's, [#204](https://github.com/EPFL-TOP/smart-microscope-control/issues/204)) the strip also says so: a move already under way
runs to its target. After STOP every action is refused until **Resume**, which is
refused while a run is going.

**The regions, from the top** ([m5 §17.3](../design/m5-web-ui.md), at 1280 px wide and
more):

```text
[Connect][Disconnect]  demo-sample | XY … | Z … | 10x | …                 [Resume] [STOP]   the strip
outcome lines (connection, run, message, problems): an empty one takes no height
[1. Stand ✓][2. Plate ✓][3. Wells][4. Scan][5. Results]                       [Expert view]   the step bar
 demo-sample   96-well, registered   next                                                    its summaries
┌ step panel (400 px) ┬ map ──────────────────────┬ image ─────────────────────────────┐
│ title: a question   │ [Density of ▾] [Refresh]  │ [Snap] [Live]                      │
│ a sentence          │ the map's title, wrapped  │ the frame                          │
│ the next step       │ the plate map             │ the drive bar: the pad, the steps, │
│ the step's regions  │                           │ Z, autofocus, light, objective,    │
│ [Continue to …]     │                           │ Drive to, Drive (next click …)     │
└─────────────────────┴───────────────────────────┴────────────────────────────────────┘
```

- **There are no tabs.** The map and the image are side by side and **both always on
  screen**, so that the operator who centres a well (Live, then the arrows) sees the map,
  and the one who clicks a well sees the image. A snap, a Live, a run's end or a tick
  changes **no screen**. Neither figure is ever removed and added again, so a map redrawn
  while you look elsewhere is framed on the plate as it was drawn, and **Reset** in the
  map's toolbar returns to that frame ([FM-174](../design/failure-modes.md)).
- **The strip stays at the top of the window** when the page is scrolled, so that STOP is
  always in reach. At 1280 x 800 the **drive bar is partly below the fold** (its pad and
  its first rows are in view, the objective, *Drive to* and the toggle are a scroll
  down); the figures are 400 px high, the least the layout allows, to show as much of it
  as it can. A wider or taller window shows all of it.
- **The step panel (400 px) scrolls on its own** when a screen is taller than the window;
  the strip, the step bar and the figures do not scroll with it.

**The five steps.** A step opens at any time: none is ever disabled (an action that cannot
run is greyed with its reason beside it, as before). The button is `n. Title`, with a `✓`
when the step is done, and a line under it: the stand's name (and the preset), `not
connected`, `<plate>, registered`, `<w> wells, <n> fields`, `scanned`, `<k> of 5 done`,
and `next` under the one the run document needs next. **The page opens on that step**
(the stand while it is not connected, Results when everything is done) and afterwards only
you change the screen: a click on a step, or the *Continue to …* button at the foot of a
screen. The ticks and lines follow the run document after a run, a Refresh and a change of
the connection, and nothing else moves with them. Each screen is a title (the question it
answers), a sentence, **the next step** (the same one on every screen), the regions of its
step, and *Continue*:

| Step | What its screen holds today |
|---|---|
| *Stand* | the preset (load, save), `camera-calibration` and `timing` |
| *Plate* | *Plate*, *Plate lies* and the plate's confirmation, `plate-registration` |
| *Wells* | the *Wells* box, `scan-plan` |
| *Scan* | `scan` |
| *Results* | `cells`, `density`, `route`, `report`, `nis-export`, then the run document's path and its checklist |

These are today's regions listed by step; the screens' own designs (the known values, the
list of centres, the progress, the results' rows) come with the next builds of
[#431](https://github.com/EPFL-TOP/smart-microscope-control/issues/431).

**The expert view.** The *Expert view* switch replaces the step panel's contents with
everything the page had before the steps: the preset, the next step, *Plate*, *Plate lies*,
*Wells*, and an accordion of **every** tool, the ones no step holds included (`goto-well`,
`plate-walls`, `plate-finding`, `plate-orientation`), then the run document and its
checklist. The strip, the step bar, the map, the image and the drive bar stay. A click on
a step turns the switch off and shows that step; turning it off shows the step you left. A
value widget (the *Wells* box, a tool's field) is **one** widget listed on every screen that
shows it, so what you typed on one is there on the other.

**Per browser tab.** The step shown and the switch belong to the tab: two tabs may show two
screens, and what each draws (the ticks, the lines, the map) is the server's.

**The map**. The plate in stage µm, X right and Y up, wells as circles (96-well and
below) or squares (384 and 1536), turned by the plate's rotation. **Its title says how the
plate was placed**, in one of four ways, because a well name read off the map is trusted:

- `96-well: the registered plate, A1 at the back left`: A1 and the rotation come from a
  registration that measured them (`plate-registration`, `plate-walls`, a lattice fit), and
  the title names the corner A1 is at in the stand's words (in stage terms, `A1 at stage -X
  +Y`, on a stand whose profile does not say its sides).
- `96-well: placed from one well (B3), not registered: the rotation is taken from the
  declared orientation, A1 at the back left, not measured`: a registration from one well only
  (a *seed*): A1 is where that well says, the plate lies the way *Plate lies* said, and
  nothing checked it.
- `24-well: estimated from the stand's holder, not registered: centred at stage (400, -325)
  µm, declared not turned, A1 at the top left of this map`: no plate is registered (or the
  registered one is not the plate you chose), and the stand's profile says where its holder
  puts a plate ([`[holder]`](../../profiles/README.md#holder-where-the-holder-puts-a-plate)).
  The plate is drawn there, **an estimate good to a few hundred µm**: a plate put back moved
  by 285 µm on the Ti2-E. Which way round it lies is what *Plate lies* says (`declared turned
  180°, A1 at the bottom right of this map` when turned): compare the A1 of the map with the
  A1 printed on the plate (see *Which way the plate lies*, below).
- `96-well: not registered, and this stand does not say where its plate lies: A1 is drawn at
  stage (0, 0), so the cross shows the stage and not a well`: the stand's profile has no
  `[holder]`, so the wells are drawn from the origin and name nothing about the stage.

**The estimate is drawn, and only a drive acts on it.** It fills no slot of the run document,
no tool reads it, and a capture never takes its well from the cross (*Capture centre of B3*
records the well you clicked). A **drive click on an estimated map** runs `goto-well` to the
well's **centre** where the holder puts the page's plate, turned as *Plate lies* says; beside
the *Drive* toggle the page says `going to B3: the holder's estimate, not registered; centre
it by eye`, then the run's end, whose first `assumed` line is `target estimated from the
stand's holder, not registered: ...`. Centre the well by eye before you capture it. What the
page refuses before anything moves, beside the toggle too:

- on an *unplaced* map (no `[holder]`): `the plate is not registered, and this stand's
  profile does not say where its holder puts a plate ([holder] plate_centre_xy_um): register
  the plate, or drive with the arrows`;
- on the 96-well placeholder nothing named: `choose the plate on the stage first: ...`;
- a wall (*Drive to* other than `centre`): `a wall is reached through a registered plate: go
  to the centre, and register the plate`.

The line beside the toggle is the same in every tab, and *Refresh* clears it.

**The stand's sides and A1.** When the profile says which stage direction is the back of
the stand and which the operator's right (`[holder] back`, `right`:
[profiles](../../profiles/README.md#back-right-the-sides-of-the-stand)), the map labels
its four sides around the wells: *back of the stand*, *front of the stand*, *operator's
left*, *operator's right*, half a well's pitch outside them. On the Ti2-E the back is at
the top and the operator's right on the right; on a stand whose back is -X it is on the
left of the map. Without them there are no side labels. **A1 is always labelled, bold**,
on every plate: it is the one name you check against the plate.

Names are drawn up to 96 wells. The selection has a thick blue
outline; the cells' **density** colours the wells (grey where unknown, a well with no
frame is unknown, not zero), for the label chosen under *Density of*. The **route** is
a red line with its stops.

**The cross.** The stage is a black cross, with the well it is over written beside it:
`over B3` on a registered map, `over B3 (estimated)` on an estimated one, `over B3
(rotation assumed)` on a seed's, and nothing between wells, on a map the stand does not place,
or with the stand closed. **While Live runs the cross follows the frames**, ten times a
second: each frame carries the stage's position, read right after its snap, so the cross
costs no read of the stage. **Otherwise it follows the page's tick, once a second** (after a
drive, at rest): a snap never moves it, since its frame is old by the time it is drawn. When
Live ends the cross is drawn from the stage again within a second.

**Which way the plate lies.** A plate goes into its holder one of two ways round, and the
page cannot see which. **Say it in *Plate lies***, next to *Plate*: `A1 at the back left` or `Turned 180°: A1 at the front
right` on the Ti2-E, worded from the stand's sides (in stage terms, `Not turned: A1 at stage
-X +Y` and `Turned 180°: A1 at stage +X -Y`, with a note under the menu, on a stand whose
profile does not say them). Every plate tool is given the choice (`orientation=` of
`plate-registration`, `plate-walls`, `plate-finding` and `plate-orientation`); no tool's form
offers it a second time. The next plate chosen keeps it: it is likely put down the same way.

**Check it with the cross.** Look at where the objective really is on the plate (the well
under it, or A1 printed at a corner) and compare it with the cross on the map and its label.
If they disagree, the plate lies the other way: switch *Plate lies*.

- **On a registered plate the switch renames the wells, and measures nothing.** It runs
  `plate-orientation` (its outcome is on the strip, as any tool's; `smc run
  plate-orientation --param orientation=turned-180` is the same from a terminal): within a
  second the cross stays where it is and its label changes, `over B3` to `over C4` on a
  24-well, because a plate turned 180° lands on itself and only its names move. No stage
  moves and no fit is run again. The tools after it (`scan-plan` and on) read the plate
  again: their results are stale, since their wells changed names.
- **The captures follow; the selection keeps its names.** The centres and walls you captured
  are positions, so they are renamed (B3's centre becomes C4's), and the well the capture
  buttons named is cleared. A selection is the printed wells you chose (B3 is the B3 written
  on the plate), so it keeps its names, which now lie elsewhere; the strip says so once.
- **On an estimate** the switch turns the drawn plate about the holder's centre, at once;
  nothing is run or written.
- **It never** moves the stage, turns the plate by 90°, or warns that the cross and the
  plate disagree: only you see the plate. A switch while a tool runs, or with the stand
  closed, is refused and says why.

**The scan plan on the map.** Once `scan-plan` has run and its result is usable, the map
draws **one orange rectangle for each field of the plan**: the field of view wide and
high (the footprint's bounding box in stage axes, so a camera turned by another angle
than 0° or 90° shows a box larger than its frame), centred on the field. They are the
plan's own wells, whatever the *Wells* box says now: the plan is what `scan` will do, and
a selection changed since has not changed it. The title ends with the count, `; scan plan:
72 fields in 2 wells`. The route is drawn over the fields; no field is hoverable, labelled
or joined to the next (the plan's order is on the card, in the order of its wells).
**A plan that is not usable draws nothing**, as the density does not: stale (the plate or
the timings were produced again since: the checklist says so and the card is marked),
made on another stand, or a dry run's on a page that is not one. **A plan of more than
5000 fields draws nothing**, not a first part of it (a first part would pass for all of
it), and the title says so: `; scan plan: 20000 fields in 1536 wells, too many to draw
(over 5000)`. The bound keeps what the page sends the browser at about 165 kB at each
redraw; the card still says the count.

**Which plate.** The page starts on the plate the run document names: the plate of the
latest `plate-registration` or `plate-walls` record, whatever its status (the one you
were working on). Without one, the profile's `[sample]` names it (the stand's camera
images that plate). With neither, the *Plate* select shows `96-well` as a
**placeholder**: nothing said it is on the stage, and a registration run on a 96-well
map of a 24-well plate puts A1 two pitches wrong, with no residual to say so. So the
page says it, under the select:

```text
Which plate is on the stage? 96-well is only a placeholder: choose the plate, or press the button if it is a 96-well.
```

and offers a button, `The plate is a 96-well`, that confirms it (it clears nothing: the
captures you made are kept). Choosing a plate in the select confirms it too. Until then
**a tool that takes the plate does not run**: `plate-registration` is disabled with
`choose the plate on the stage first: the page shows a 96-well placeholder` beside its
button, the next step says the same, and the controller refuses the run with that
sentence if a click gets through. Captures stay allowed: a capture is a stage read.

**Choosing wells.** Click a well to select it, and again to unselect it. Or type a
range in *Wells*: `A1:D6`, `B2-B5`, `C*` (a row), `*3` (a column), `all`. The selection
becomes `scan-plan`'s `wells`, in the order you chose them. A click moves nothing.

**The well the captures act on** is the one you last **selected** by a click, or drove
to. Unselecting it (a click on it again) drops it, so do a typed range or a preset's
wells that do not hold it; unselecting some other well leaves it alone. The capture
buttons name it (see *Calibrating the plate*), so the map and the buttons never disagree
about which well a capture is for.

The *Wells* box **follows the selection**: a click on a well writes the selected wells in
it (`B2,B3`), and so does a plate chosen, a selection another tab made, or a run
document read at the start. A range you typed **stays as you typed it** (`A1:D6` stays
`A1:D6`) until something else changes the selection. **Emptying the box clears the
selection** (no wells), and the next click is written in it again; a text that names
no well (`Z99`) is refused with the reason, and the old selection stays.

**Driving to a well.** Switch **Drive** on (the strip then says `DRIVE: the next
click on a well moves the stage there`) and click a well: `goto-well` runs for it, at
its centre or at the wall chosen under **Drive to (wall as on the image)**: the wall is
named as the image shows it (`left`, `right`, `top`, `bottom`), the page maps it to the
stage's direction with the latest usable `camera-calibration`, and `goto-well` runs with
that (`side=+X` for the image's left on the Ti2-E at 10x). With no calibration it
refuses a wall and says to run it (`walls are named on the image, and no camera
calibration says which stage side that is: run camera-calibration (Set up), then try
again`); the centre needs none. Drive is **one click**: the toggle
goes back to select after it, even if the click found no well or was refused. The page
**refuses** to drive on a map of another plate than the registered one (it would send
the stage to the same-named well of the other plate): choose the registered plate, or
calibrate the one you chose.

**Snap and the image.** **Snap** takes one frame from the camera and draws it, beside the
map, within a second. The title is the frame's
own summary, as `smc snap` prints it (`512x512 uint16, exposure 10 ms, min 706, max
5846`); before the first frame it says `no frame yet`. The frame is kept as the camera
gave it, **never rescaled**, and a snap writes nothing: not a file, not the run
document (`smc snap --out` writes a TIFF). What is drawn is a copy for the screen: 8-bit
grey, stretched between the 0.5 and 99.5 percentiles of the frame, so a hot pixel does
not decide the contrast; a **flat** frame is shown against the camera's range instead, so
a saturated frame is white and a dark one black and not a stretched grey. A frame larger
than 1024 px on a side is shown at every n-th pixel (n = 2 for a 1412 px camera), and the
axes are still the **camera's pixels**. The frame is the same for every tab of the server: a
frame snapped in one tab is drawn by the others the next time they look for a new
frame, every 0.1 s. Snap is disabled with
`this stand has no Camera` when the stand has none, and, as every Run button, while
something else is going: a snap is one activity like a run, so only one runs at a time
(`snap is running`, `<plugin> is running`), and STOP during a snap says `snap told to
stop; ...` and halts the stand. A frame the page cannot show (not 2-D `uint8` or
`uint16`) is an outcome under the strip and is not kept.

**Live** (beside Snap). **Live** takes frames from the camera one after the
other, **at most 10 a second**, and the page draws each new frame: the frame number is
what it compares, every 0.1 s, so it draws nothing when there is nothing new. The
button then says **End live**. While live goes the line under the strip reads
`live: <n> frames`. A camera slower than 10 frames a second runs at its own rate. Live
is **one activity**, like a run: **Snap and every Run button are disabled**, a click on
the map in drive mode is refused, and each says `live is running`. Live is never ended
for them: end it with **End live**, which ends only the live view and **does not halt
the stand** (it says `live ended: <n> frames in <s> s`). A click does what the button
says, so a button that is a second out of date does not do the opposite: a stale
*Live* in a second tab says `live is running` and leaves the first tab's live going, and
a stale *End live* after a camera fault, with no live running, ends nothing and starts no
new live (if another tab has started a live since, it ends that one: live belongs to the
page, not to a tab). **STOP**
ends it **and halts the stand**, as it halts a run (`live told to stop; stopped: the
stand is halted (Resume clears the halt)`). A frame that fails, or that is not 2-D `uint8` or `uint16`, ends
live with `live ended after <n> frames: <why>`, and the last good frame stays. Live is
disabled with `this stand has no Camera` when the stand has none, and while another
activity goes. **A closed or reloaded tab does not end live**: the page belongs to the
server, so a live view with no tab open keeps the camera taking frames until STOP,
*End live* from a tab, or Ctrl-C of the command. A live view writes nothing to disk and
nothing to the run document. Every frame is sent to every open tab, up to 10 a second.
The **Drive** controls stay usable while live goes (below).

**Next step** (the right-hand column, under the [Preset](#presets) region and above
Drive). The tool the run document needs next, and the values to run it with, so that a
newcomer needs no manual.
It is the first slot of the checklist that is not optional and not usable
(`camera-calibration` and `timings` are never a next step), and its tool is the first
registered one that fills the slot. The region says:

- the tool's name, and why it is next: `the run document has no scan-plan yet`, or the
  checklist's own reason when the slot is there and not usable (`... ended failed; run it
  again`, `... read its inputs from an earlier run; ...`);
- where each value comes from, a line each: the wells **selected on the map**; the
  centres and walls **captured on this page**; the centres of the profile's `[sample]`
  (see below); or what the profile **suggests** for the tool (`coverage=0.4: suggested by
  the profile demo-sample: ...`, from its [`[suggest]`](../../profiles/README.md#suggest-what-the-stand-suggests-for-a-run) table);
- `or <name>: ...` for each other registered tool that fills the same slot (on the plate,
  `plate-registration` and `plate-finding`, or `plate-walls` and `plate-finding`);
- on the plate, **`plate-walls` is the step when you took its path**, and
  `plate-registration` otherwise. All of these hold: the plate is chosen, the page is
  not a dry run, the plate's wells are round (`plate-walls` fits circles and refuses the
  square wells of a 384- or 1536-well plate), the stand can image (an `XYStage` and a
  `Camera`; a closed stand is drawn for the stand you are about to open), **no wall is
  captured** (touching walls is `plate-registration`'s own path) and **2 to 6 centres**
  are captured. Its line says
  `plate, refs: the 2 centres of B3, D5 captured on this page; plate-walls images each
  well's four walls around its centre and moves the stage in X and Y only`. Until
  `camera-calibration` has been run, its button is disabled and says why (see
  *Registering the plate from images*). Both tools keep their sections and their Run
  buttons;
- `command: smc run ...`, the command that makes the same call, with the profile, the
  run document's absolute path and `--dry-run` when it is one. It is quoted for a POSIX
  shell, and a page served on Windows says so under it (`that line is quoted for a POSIX
  shell (bash, zsh); PowerShell and cmd.exe read quotes differently: press the button,
  which makes the same call`); what the two Windows shells do with it is **not
  verified**. The run document's record of the run is the reference, whichever shell
  reads the line;
- a button, **Run \<tool\>**, which runs **exactly the values the step shows** (never the
  tool's form, and never a step computed at the click), and a line in the warning colour
  when it cannot: `the stand is not connected`, a checkpoint to continue, the wells to
  choose (`select the wells to scan: click them on the map, or type a range in Wells`), or
  how to capture the plate (`centre a well under the objective ...`) on a stand with no
  sample. Nothing moves without that click, and the page never runs a tool by itself or
  chains two;
- on a page served with **`--dry-run`**, the step **stops at `scan`** (and at a tool that
  reads a `scan` that was a dry run) and says why: a dry run walks the fields and takes
  no frame, so `cells` and what follows have nothing to read, and a run that is not a
  dry run refuses what a dry run made. The tools' own Run buttons stay, and so does a
  dry-run `scan` to see the walk; a `scan` made in a terminal without `--dry-run` is
  read as it is.

**The plate from the profile.** On a stand whose profile has a `[sample]` (`demo-sample`),
with nothing captured on the page, the plate step registers the plate in one click from
the sample's own geometry: `plate-registration` with the centres of three wells (A1, A12
and H1 on 96 wells). The step says `... this stand's plate is synthetic, so they are known,
not measured`: the same fit as a real one, from positions nobody measured, and the
registration's card ends with the same line once it has run (it reads `method:
references`, `residual_um: 0`, and the plugin cannot tell). A capture made on the page
wins over it, and its card says nothing of the kind: it is measured. When the workflow is done the step lists what it wrote: every
file and folder a result names, by absolute path.

**Drive** (the right-hand column, below the next step). Each control is the call of a
command, so each has its command-line twin:

| Control | Command |
| --- | --- |
| the pad `↑` `←` `→` `↓` | `smc stage jog DX DY` |
| `Z +`, `Z −` | `smc z jog DZ` |
| **Autofocus on**, **Autofocus off** | `smc autofocus engage`, `smc autofocus disengage` |
| **Change objective**, then **Turn the turret** | `smc objective set LABEL --confirm` |
| **Light on**, **Light off** | `smc light on`, `smc light off` |

The pad is in the **stage's axes**, as the map (X right, Y up): `→` jogs `+step` in
X, `↑` `+step` in Y. The arrows stay so (it is the stage that moves, so the image is
normally inverted, and microscopy's standards are kept). **Which way the *image* moves**
depends on how the camera is turned on the stand, and the note under the pad says it from
the latest usable `camera-calibration`: `On the image, ↑ moves it up, → moves it right; ↓
and ← the opposite.` (the Ti2-E at 10x; an upright camera reads `↑ moves it down, →
moves it left`). With no calibration it says `Which way the image moves for an arrow is
not known: run camera-calibration (Set up).`, and for a camera more than 5° from a quarter
turn it gives the reason. The note disables nothing. The step is chosen under **XY step** (1, 10,
100 or 1000 µm; 100 at the start) and **Z step** (0.5, 1, 5, 10 or 50 µm; 1 at the
start); only the steps up to the profile's jog limits (`[safety] max_jog_um` and
`max_z_jog_um`) are offered. **A jog is one click**: there is no repeat while a button is
held, and no typed distance. With the autofocus locked, `Z +` and `Z −` suspend it and
lock it again after the move (the layer's interlock); on a stand whose autofocus drives Z
to its offset when it engages (the Ti2-E's PFS), that relock may bring Z back to the
offset's plane, so **focus by hand with the autofocus off** there.

**The objective needs a confirmation.** Choose it, press **Change objective**: the page
asks `Turn the turret to '<label>'? A turn can drive an objective into a loaded plate:
check that the stage is clear. The autofocus is switched off and not engaged again.`
Only **Turn the turret** turns it; **Cancel**, another choice, or a run starting in
any tab takes the question back. An empty turret position is offered as `(no label)`.
The choice **follows the turret until the page has read the objective in place once**
(a first read that fails is tried again on the next tick), and is yours after that: the
page never changes it again, and the strip says which objective is in place. A turret
whose positions could not be read at first (`objective: its positions could not be
read`) is read again on every tick until they are offered, and the failure is logged
once, not once a tick.

On a command line the empty position is `smc objective set "" --confirm`. **In Windows
PowerShell 5.1** an empty argument is dropped on its way to a program, so there it is
`smc objective set '""' --confirm` (`cmd.exe` takes `""`). That is PowerShell's
documented behaviour; it has not been run on a stand. Two turret positions that share a
label, or two empty ones, are **one choice** in the select, and the layer addresses a
position by its label too, so that limit is the layer's: whether Micro-Manager allows
such labels on a real stand is not known yet.

Each action is **one activity**, like a snap: it is disabled, and refused, while a run
or a snap goes, and its result is the line under the strip (`stage jog: XY (100.00,
0.00) µm`, `objective set: objective 'Nikon 20X Plan Fluor ELWD'`, `autofocus engage:
autofocus did not lock; it is off`, or the layer's refusal, such as the jog limit).
**While live goes, the drive controls work**: the live view runs the action between two
frames, and the next frame shows where the stage went. One action waits at most: a
second click while one waits is refused (`stage jog is waiting`), never queued. STOP,
*End live*, Ctrl-C or a frame that fails **drop** a move, a turn or an engage that still
waits: it is never sent, and live's last line says `; <action> was not run`. **A waiting
*Light off* or *Autofocus off* is run at the end instead**, after STOP too: switching off
is the call the layer never refuses, halted or not, and it sends no move, so the light is
not left on behind a STOP. Live's last line then says what it did:
`live ended: 3 frames in 0.4 s; light off: light off`. **Live's last line keeps the last
drive action's outcome** (`live ended: 2 frames in 0.1 s; stage jog: XY (100.00, -50.00)
µm`): the earlier ones were under the strip while live went, and the log has every one.
After STOP, a move, a
turn, an engage and *Light on* are refused (`MicroscopeHaltedError: ...`); *Autofocus
off* and *Light off* still run. A control is **disabled with the reason** when the stand
lacks it, one line each under the region: `pad: this stand has no XYStage`, `objective:
its positions could not be read`, `light: the light source LED has no on/off that smc
knows` (the demo stands' light, a `State` device: switch the light with its shutter).

**The tools.** One section per stage of the workflow (*Set up*, *Choose*, *Acquire*,
*Find*, *Report*), and a last one, **More**, for every other registered plugin by name
(`goto-well`, a third party's): every registered plugin is a tool. One block per tool:
its key parameters as fields, the rest under *Advanced*, a **Run** button and the
progress. A **required** parameter with no default is never folded under *Advanced*:
it is a field of its own. An empty field is not sent, so the plugin's default holds; the
placeholder is that default. What the page fills itself is not offered: the wells of
`scan-plan`, and the plate and the captures of `plate-registration` and `plate-walls`
(the second takes the centres only). A parameter the
profile suggests for a tool (`[suggest]`) is always a field of its own, never under
*Advanced*, and **starts with the suggested value** (its tooltip says why), so the tool's
own Run sends what the next step shows. A tool that cannot
run now is disabled with the reason beside it (`the run document has no plate; ...`,
`this stand has no XYStage`), and so is every tool while an activity goes, **with the
reason beside the button** and not only in the strip: during Live, `Live is running:
press End live, then run the tool.`; during any other activity, `<activity> is running:
wait for it to end, or press STOP.` A tool's own reason (the stand closed, a missing
input, the plate not chosen) wins over it, because it holds longer than the activity.
A tool does not run during Live: one activity at a time is the page's rule, and
`plate-registration` is no exception even though it reads nothing from the stand.
A tool that provides the plate (`plate-registration`, `plate-walls`) says, under its
note, what a Run would replace once a plate is usable:

```text
Run replaces the plate registered 2026-11-03 15:20 (seed, 1 well). A failed run leaves no plate; the captures stay, so Forget the wrong one and Run again to get a plate back.
```

The time is the record's start in the PC's local time. The page does not refuse: a
registration is a pure function of its captures, which the page keeps. Under
`plate-finding`, which runs from no capture, the note ends after `A failed run leaves no
plate.` No other tool has this note. A plugin that failed to load, or whose parameters cannot be described, is not
a tool: a line above the map says `plugin <name>: <why>`.

**A tool's result**, under its Run button, as plain text (a result's words are never
read as markup): `<status> <time> UTC`, then ` on <profile>` when the record was made
on another stand, ` (dry run)` when it was one and ` (stale)` when the tool read an
input that has been produced again since (the checklist says `<plugin> read its
inputs from an earlier run; run it again`; the values are kept, and marked), then the
result's values, two spaces in, a line each (a text cut at 80 characters, a list or an
object only counted, at most eight, then `… n more`), then **where the result is**, then
what the result **assumed**
(at most eight, then `… n more`, each cut at 600 characters) and the error (cut at 600
characters too). Where the result is: each `file` and `folder` the result names, as an
indented `file: <absolute path>` line (a relative one is under the run document's folder;
an absolute one is shown as written, in whichever OS's style the run document holds it;
each is cut at 600 characters), or an indented `in the run document` when it names none:
**every result card says where its result is**. The card of a plate registered from
images (`plate-walls`) is the exception to the result's values: it shows **one line per
well measured**, then one for the plate (`B3: centre (10182.0, -10087.9) µm, 15298 µm
across, 14 frames`, `plate 24-well: residual 3.2 µm, rotation 0.12°`; `residual unknown`
when nothing measured it), and the folder of the frames it saved as its file. **The card
of a scan plan** is the same kind of exception: its first eight keys cut `estimate_s`
away, the figure the operator wants before pressing `scan`, so it shows four lines,
`wells: 2 (B11, B2)` (the plan's visiting order, at most six names, then `, … n more`),
`fields: 72, a grid of 6 × 6 in each well` (or `fields: 344 of 392, a grid of 14 × 14 in
each well, the fields that miss the well left out` when clipping to the well removed
some), `field: 512 × 512 µm at <objective>, overlap 0.1` and `estimate: about 42 s (the
moves, refocus and snaps of the timings)`. **The estimate sums only that**: each move, a
refocus when the stand has an autofocus and the timings know its cost, and each snap; it
leaves out a channel change, the cells, the report and the export. These four lines are
cut at 600 characters, as an assumption is, and not at a value's 80: the clipped `fields:`
line is 90 long. With no usable
`timing` it says `estimate: unknown, no usable timing: run timing, then scan-plan again`
(an unknown is not a zero, and `timing` is not a step of the page: it is optional). A
second `timing` makes the plan stale: run `scan-plan` again to see the new estimate. The page shows the path and never opens it: open it
from the PC. A card is a summary, bounded whatever the record holds: the run
document and the log hold the rest. Cards, the next step and the run document's path
**wrap** at the column, so a long path is read whole. A tool whose last run was stopped or failed shows
its status and its error and **no values**: they are not a result. The card follows the run document, so it is drawn, in every
tab, when a run, a snap or a live view ends (a tool that takes less than a second included) and when
you press **Refresh**, not every second.

**Run never drops a checkpoint that `--resume` can continue, and never continues one.**
After STOP (or a failure) a stage's record is `stopped` or `failed`, and the checklist
says `run it again`. When that record holds a checkpoint that `smc run NAME --resume`
could still continue (made on this stand, not a dry run's checkpoint for a real run, and
nothing the stage read produced again since), its Run button is **disabled** with
``stopped with a checkpoint: continue it with `smc run NAME --resume` (close this page
first); `smc run NAME` without --resume starts afresh and drops it`` beside it. A click
that gets through anyway (a tab that was not redrawn) is refused with the same words, and
nothing is written. **To continue it, stop `smc ui` (Ctrl-C in its console) and run `smc
run NAME --run-doc ... --profile <this stand> --param KEY=VALUE ... --resume`.** The
page's "close this page first" means that: the stand is held by the command, and a closed
tab releases nothing. `smc run` takes the demo profile unless `--profile` (or
`SMC_PROFILE`) says otherwise, and `--resume` refuses parameters that differ from the
stopped run's, so give again every field you typed on the page as `--param KEY=VALUE`.
To give the checkpoint up, `smc run NAME --run-doc ...` without `--resume` starts afresh
(and logs that it does); the page does not offer that, and it is also the way out when
`--resume` itself is refused by the plugin (a scan whose frames were moved out of its run
folder: the page cannot know it and stays disabled). The fields are not compared: typing
another value does not lift the refusal. A stage with no checkpoint, or whose checkpoint
`--resume` would refuse for what the document tells (another stand, a dry run's, a stale
one), is started from the beginning by the button, as it always was. What the first
attempt wrote to disk (a scan's frames) is kept either way.

**Calibrating the plate.** Under `plate-registration`: centre a well under the objective
by hand, click it on the map, press **Capture centre**; for a wall, put the objective on
it, choose its side **as you see it on the image** under **Wall, as on the image**, press
**Capture wall**. A note under the buttons says what each side is on the stage, from the
latest usable `camera-calibration` (`On the image: left is the stage's +X, right -X, top
-Y, bottom +Y (from the camera calibration).` for the Ti2-E at 10x), and the capture is
**recorded in the stage's words**: the list shows `B3 +X wall (17967.9, -10237.1)`. With no
calibration the note says to run it, and **Capture wall** refuses (`walls are named on
the image, and no camera calibration says which stage side that is: run
camera-calibration (Set up), then try again`) and records nothing; a camera more than 5°
from a quarter turn is refused with the reason. Centres need no calibration. Prefer an
opposite pair per axis: only a pair measures the well. **The buttons name what they
record**: `Capture centre of B3`, `Capture B3 wall at image left` (the side is the *Wall*
select's value, and the label follows it at once), `Forget B3`. With no well selected they read `Capture
centre`, `Capture wall`, `Forget`, are disabled, and a note says `click a well on the map
first: a capture records the well you clicked last`. **A press records the well on the
button, or nothing**: if the well changed since the button was drawn (another tab, or a
click between two redraws), it records nothing and says `the well to record changed
from B3 to B4: check the button, then press it again` (`none` when the well was
unselected); the button then shows the new well. A capture records where the stage is, so it
is **refused while the stage is about to move or moves**: while a drive action waits
behind a frame or runs during live (`stage jog is waiting`, `stage jog is running`), and
while a run, a snap or a connect goes (`<name> is running`; `goto-well` is a run). The
buttons are greyed while an activity other than live goes; during live they stay usable
and the refusal is the message. A live with no drive action handed to it does not refuse:
a frame moves nothing. A second capture of the same well and **stage** side
replaces the first; **Forget** drops the captures of the well on its label. **Run
plate-registration** then fits the plate from what was captured, as `smc run
plate-registration --param refs=... --param walls=...` would. Choosing another plate
clears the selection and the captures.

**Three wells far apart are enough.** A plate seated in its holder hardly moves, so one
centre in each of three wells (B3, B5 and C3 of a 24-well plate: far apart, not in a line)
fits the plate and checks itself, two ways:

- **By eye**, with *Capture centre* and **Run plate-registration**. Three centres each off by
  at most 130 µm fit with a residual of 121.6 µm, a rotation of 0.10° the plate does not have
  and A1 within 55 µm; **one centre 900 µm off is refused and named** (`B3 is 573 µm from where
  the others put it: centre it again`). The limit is a worst residual of 300 µm, and **by
  eye is only as good as the eye**: in the middle of a 15.6 mm well, at the 10x whose field is
  about 1.4 mm, one cannot judge a few hundred µm. Numbers and limits:
  [How good the centres must be](../plugins/plate-registration.md#how-good-the-centres-must-be).
- **From a rough position**, with `plate-walls`: it takes a position up to 3 900 µm from the
  centre of a 24-well and measures the centre from the well's walls, which is why the page
  offers it first once two or more centres are captured. It takes longer. Whether it registers
  the Greiner plate from one rough position in each of three wells is **to be verified at
  the stand** (Ti2-E session 4).

The map's *estimated* plate does not shorten either: it is drawn to find the way, and
nothing registers from it.

**Registering the plate from images.** Under `plate-walls` (*Set up*, right after
`plate-registration`): the tool images each well's four walls itself and fits the plate
from them, so you centre two or three wells by eye and press a button.

1. **Connect**, and choose the plate on the stage.
2. **Run `camera-calibration`** (the first tool of *Set up*) with the objective you register
   with in place, on a field with texture (a well's wall will do): `plate-walls` reads it,
   and the page does not run it for you, because it moves the stage and measures at a
   place you chose.
3. Centre a well under the objective (Live and the **Drive** pad), **click it on the map**,
   and press **Capture centre of B3** in `plate-walls`' section: the buttons name the
   well they record and follow the same rules as the registration's (a press records
   the well on its label, or nothing). Then a second well, far from the first, and a third.
   The centres are listed under the buttons (`centres captured (2):`, a line each, in
   capture order). **Forget B3** drops that well's centre and its wall touches.
4. Press the next step's button, or **Run plate-walls**. The strip counts `plate-walls:
   3/8 B3 -X` while the stage moves in X and Y only; the plate is then registered, and the
   card says how well.

Those are the buttons of `plate-walls` only: **the wall captures stay
`plate-registration`'s**, and `plate-walls` ignores them (a line under the centres says so
once: `1 wall capture is for plate-registration; plate-walls images the walls itself and
ignores them`). **Why its Run is disabled**, in this order, beside the button: the stand is
closed; the plate is the placeholder; the page is a **dry run** (`plate-walls images the
walls, which a dry run does not do: serve the page without --dry-run (demo-sample's stage
is simulated), or register with plate-registration`: the page refuses the run itself,
since the plugin's own refusal would be a failed record that replaces a usable plate);
the plate's wells are **square** (384- and 1536-well plates: `plate-walls fits round wells,
and the 384-well plate's are square: register it with plate-registration`; the page refuses
that run too, and a registered plate stays the plate of the run document);
there is **no calibration** in the run document (`plate-walls reads the camera's
calibration: run camera-calibration first ...`; one that exists and is unusable keeps the
run document's own words); the stand has no `XYStage` or `Camera`; and the **centres**
(fewer than two, or more than six: `... Forget some`). The count is the page's check and
the plugin's own: a call the page did not make is refused by the plugin before anything moves.

**The run document**, as a checklist under a bold line with its path, `Run document
<absolute path>` (the file as the command was given it, made absolute where the server
runs; the same file the next step's command names): `✓ plate` (usable), `· cells` (not run),
`✗ scan: the last run of scan ended failed; run it again` (there, and not usable, with
why). The slots of the plate-scan workflow come first, in its order; any other slot the
document holds (a third-party plugin's stage) follows by name. A line above the
map, in the warning colour, says what the page could not read and did not hide: an
unreadable run document (the page then shows every stage as not run and the map as the
profile places it, estimated or unplaced, and never writes the file), or a result that no
longer matches its model.

## Presets

An assay repeated every week is the same choices every time: the plate, the wells, a
coverage, a detector. A **preset** keeps them under a name, in a file, so that the
page asks for them once. It is a convenience and never the truth: the run document
keeps the values each run used. A preset holds **no position** (the centres and walls
you captured), and no exposure or channel, which no tool takes as a parameter.

**The region** is at the head of the right-hand column, above the next step:
*Preset* (a select of the presets found, and **Load**), *Save as* (a name, and
**Save**), and a note.

**Load.** Choose a preset and press **Load**. The page sets:

- the **plate**, and the **wells** (they appear in *Wells* and on the map);
- **every field of every tool**: the preset's value, else the profile's suggestion,
  else empty. A field you typed before is replaced: loading is choosing the preset's
  values. Fields the preset does not name go back to the suggestion or to empty.

Choosing `(none)` and pressing **Load** forgets the preset: the fields stay as they
are, and the page stops naming it. Loading needs no stand and is allowed while a tool
runs: a run already started has read its values.

**The note** under the region says what was loaded and how the page stands against it:

```text
loaded: weekly  (/home/lab/smc/profiles/presets/weekly.toml)
  preset weekly was saved on nikon-ti2; this stand is demo-sample
changed since loading:
  scan-plan coverage: 0.5 (the preset: 0.4)
  wells: 1 selected (the preset: B2,B11)
```

Each indented line under `loaded:` is something of the preset that did not apply, and
every one is a **note, never a refusal**: the rest still applies. They are:

- another stand than the one it was saved on (everything still applies);
- a **plugin** or a **parameter** that no longer exists, or one that the preset may not
  hold (`wells`, `plate`, `refs` and `walls` are kept as the preset's own fields): that
  pair is left out and the others are kept;
- a **plate** that this run document registered otherwise, or that this page has
  captures on (a load never drops the captures: they were measured), and then its wells
  are not applied either;
- **wells** that do not read on the plate (the selection stays as it was).

Under it, `the page holds the preset's values`, or `changed since loading:` and one
line for each field, the plate or the wells that now differ (`(default)` stands for an
empty field, which sends the plugin's own default). The note follows every edit. Each
list is cut at eight lines.

**Save.** Type a name in *Save as* and press **Save**. The preset holds the plate, the
*Wells* box (as you typed it when it names exactly the selection, `A1:D6` staying
`A1:D6`; otherwise the selected wells, joined) and each tool's **non-empty** fields. A
value its tool refuses (`coverage=lots`) is the message under the strip, and nothing is
written: a preset that its next run would refuse is never saved. The saved preset
becomes the loaded one. **While the page shows the 96-well placeholder** (nothing named
the plate, see *Which plate*) the preset holds **no plate**: the placeholder is not a
plate anyone chose, and a preset that named it would choose it, with nobody having said
so, whenever it was loaded. A preset that does name a plate (saved after you chose
one, or written by hand) is your choice: loading it chooses that plate.

A name is 1 to 64 lower-case letters, digits, `-` and `_`, starting with a letter or a
digit, and not a name Windows keeps for a device (`nul`, `com1`...). The page never
changes the case of a name: `Weekly` is refused.

**Where it is written.** In the first of the folders presets are searched in: the
profile's own `presets/` (beside the profile file), or `./profiles/presets` for the
built-in `demo` profile. See [profiles/README.md](../../profiles/README.md#presets-the-assays-choices).

**Overwrite.** A name that exists asks first: `preset mine exists: <path>: overwrite it?`
with **Overwrite** and **Cancel**. **Overwrite** saves what the fields show **at that
click**. Typing another name, or **Cancel**, drops the question and writes nothing. A
name found in a later folder than the one a save writes is shadowed by the new file; the
question names the file that exists. **A preset checked in with the repository
(`profiles/presets/weekly.toml`) is overwritten by that name**, and git then shows it.

**The run document names the preset.** A tool run while a preset is loaded writes
`"preset": "weekly"` into its record, with the values it used in `params`, whatever was
edited after the load. A run with no preset loaded writes no such key.

**The next step.** With a preset loaded the step's values are the page's own, then the
profile's suggestion for the keys the preset does not give, then the preset's: the
preset wins over the profile key by key, and says so (`coverage=0.7: from the preset
weekly`). The `command:` line carries `--preset <path>`, so the same run from a
terminal applies the preset and its record names it.

**Tabs.** The loaded preset is the page's, like the plate and the selection: every tab
sees it. The fields are each tab's own, so another tab's note lists how its own fields
differ.

**From a terminal**, `smc run NAME --preset NAME|PATH` applies a preset, `smc presets
list` and `smc presets show` look at them, and `smc presets save NAME --run-doc PATH`
writes one from a run document's finished stages, as the page's *Save* does from its
fields (see
[plugins/README.md](../plugins/README.md#presets-smc-run---preset-and-smc-presets)).
There is no *Delete* button and no `smc ui --preset`: a preset is deleted as a file, and
the page starts with none loaded.

A preset file is plain TOML, and can be written by hand. The checked-in example for
`demo-sample` is [profiles/presets/weekly.toml](../../profiles/presets/weekly.toml):

```toml
# A preset of smc: the choices an assay repeats. See docs/ui/plate-page.md, Presets.
stand = "demo-sample"
plate = "96-well"
wells = "B2,B11"

[params]
scan-plan = ["coverage=0.4"]
cells = ["detector=blobs"]
```

Walk it on `demo-sample`: **Connect**, choose `weekly`, **Load**, then press **Run** on
each next step. No field is typed, and the page never asks which wells to scan
(`tests/test_plate_page_presets_demo.py`).

## Trying the workflow without a microscope

```text
smc ui plate -p demo-sample --run-doc run.json
```

`demo-sample` is the demo stand whose camera images a synthetic 96-well plate
(`profiles/demo-sample.toml`), so every tool of the page can be pressed on a PC with
no microscope, and the frames each one takes follow the stage the page moves. The
stage and the other devices are the demo's; only the pixels are drawn, at the stage's
position. Serve it **without** `--dry-run`, as above: the stage is simulated, so there is
nothing to protect, and in a dry run `scan` takes no frame, so the next step stops there
and says so. **Press Connect, then follow the Next step** at the head of the right-hand
column, one click for each tool, until it says `every stage of the workflow is done`:

1. **Connect.** The next step is `plate-registration`, with the centres of A1, A12 and H1
   from the profile's `[sample]` (known, not measured: the plate is synthetic). **Run**
   it: no terminal, and no 7 minutes of `plate-finding`. The page opens on the sample's
   plate, so there is no placeholder to answer; on the plain `demo` stand, which has no
   `[sample]`, the page asks which plate is on the stage first (see *Which plate*).
2. **Choose the wells** on the map when the step asks (`select the wells to scan`): click
   one or two wells, and they appear in *Wells*. This is the only choice the page leaves to
   you. The next step is `scan-plan` with those wells and the profile's `coverage=0.4`
   (a scan then takes seconds on the demo stage).
3. **Run** each step as it comes: `scan`, then `cells` with `detector=blobs` (the profile
   suggests it: the trained model does not see the synthetic cells, and `blobs` counts
   bright round spots, which on a real sample are not cells), `density`, `route`, `report`
   and `nis-export`. Each step says why it is next and where each value comes from.
4. **Read where the files are**: the checklist's heading names the run document, each
   result card names the files it wrote (the scan's folder, the cells' file, the report and
   the NIS-Elements position list), and the last step lists them all.

To practise the plate step as at a stand, **capture centres yourself**: start **Live**,
centre a well with the **Drive** pad (the frames follow the stage), click it on the map,
**Capture centre**, and again for two more wells far apart (three fit the plate and check
each other); the next step then uses your captures, not the sample's. The plate is centred on the stage origin, so A1 is at
(−49 500, 31 500) µm and H12 at (49 500, −31 500) µm: drive there in 1000 µm steps, then
finish with smaller ones. Or run **plate-finding**, which surveys the plate by itself and
takes about 7 minutes on the demo stage. Each step's `command:` line is the same call from a
terminal.

Or the owner's path, **registering from images**: **Run** `camera-calibration` (the first
tool of *Set up*), drive to D6 and D7 with the pad's X and Y steps, **Capture centre of
D6** and **Capture centre of D7** in `plate-walls`' section, then press the next step's
button, which is now `plate-walls`. A run on the demo stage took 16.5 s for two wells; it
needs a page served without `--dry-run`.

**What is synthetic, and the limits**, as `smc doctor -p demo-sample` prints them
after the roles:

```text
Sample: a synthetic 96-well plate, A1 at (-49500.0, 31500.0) µm, turned 0°; 40 cells per well on average, 30 % of them doublets (seed 0)
  frames are rendered where the stage is, not imaged: one flat plane at z = 0 (Z only blurs), no optics
  pixel sizes up to 4 µm (the texture aliases above); the cells are bright round spots: run cells with detector=blobs
```

The same commands, from a terminal, are in
[Trying the workflow without a microscope](../plugins/README.md#trying-the-workflow-without-a-microscope).
The page on this stand has been walked by a test, from Connect to the exported position
list (`tests/test_plate_page_first_run_demo.py`), and **not yet by a newcomer in a
browser** (see below).

## What it does not do

No light intensity or channel, no focusing through the autofocus offset, no
press-and-hold jog, no drive click on the map while live goes (it is a `goto-well`
run, which writes a record), no camera settings (exposure, ROI, binning) and no time
limit on live,
no choice of the stand or of the run document from the page (one of each per
command), nothing switched off at Disconnect,
no `--resume` (continue a stopped stage with `smc run --resume`: the Run button of such a
stage is disabled and says so), no `--host`, no login. The report and the NIS-Elements
export appear as tools when their plugins are registered; they have no page of their
own.

## Not verified yet

A real browser's tap and click, **the frame's
orientation** (Bokeh draws an image's first row at the bottom, so the page flips the
rows to put the camera's top row on top: that is its documented convention, checked
here on the data and not by eye), **a snap on a stand's camera**, the page's look, the periodic callback of a real
session, Windows (Ctrl-C of the command, the 0.1 s wait), RDP, a PC with no internet
(the served page of a few widgets has no external address; the figure's was not
checked offline), Panel's resolution on Windows with Python 3.14, the **cost of
reading the stand once a second during a scan** (it runs on the loop that handles a
click on STOP: a device read that hangs in a vendor driver would delay STOP; it is to
be timed on the Ti2-E) and what STOP's note says there. These are for the first session
at the stand ([#198](https://github.com/EPFL-TOP/smart-microscope-control/issues/198)).

**The cross and the estimate** add: the estimate against the plate on the Ti2-E (the Greiner
plate in its holder, either way round, and the cross over a well it is physically over: the
`[holder]` value is the mean of two sessions 285 µm apart); the label's names against the
printed plate (turned 180° and not declared so, the page says `over B3` where the plate says
C4); the cross's lag at
the stand (a second outside Live, a tenth of a second in Live, and `state()` was seen to take up
to 0.5 s there during Live); the cross and its label at the PC's screen and zoom; and
`plate-walls` from one rough position in each of three wells, with the time it takes.

**Which way the plate lies** adds, for the owner at the Ti2-E: the profile's `back = "+Y"`
and `right = "+X"` (inferred from session 3, not seen); the *Plate lies* switch read by an
operator who has not read this page, with the cross over a well they can see; and whether a
selection that keeps its names after a turn is what the operator expects.

**Live** adds: the rate a real browser shows, locally and over RDP (the demo camera
gives 10 frames a second on the server; ADR-0006's revisit criterion for Panel is what a
browser shows); the Ti2-E's camera at full frame (Prime 95B, 1412×1412, a stride of 2);
the cost of `Microscope.state()`, which is read for every frame, on a real stand;
a mechanical shutter under auto-shutter, which opens and closes for every frame (up to
10 times a second); and the frame's orientation in a browser.

**Drive** adds: the drive at a stand; which way the image moves for `↑` and `→` (the
note under the pad, against NIS-Elements' own XY control and the stand's joystick); a
jog, a focus step and an engage on the Ti2-E while live goes, and how long a click
waits for the frame in flight on a real exposure; a turret turn under a plate; *Light
on* and *Light off* on the Ti2-E's `DiaLamp`, and **a *Light off* clicked behind a frame,
then STOP, on that lamp** (the layer runs the switch-off after the halt; the PFS's
`disengage()` after a stop is not seen either); `smc objective set '""' --confirm` in
Windows PowerShell 5.1; a turret with shared or empty labels in Micro-Manager; what the
stand keeps on after the page
or a command ends (the light, the autofocus: closing the stand switches nothing off);
`Z +` and `Z −` with the autofocus locked on the Ti2-E (the relock may bring Z back to
the offset's plane); and **a newcomer registering a plate on the `demo-sample` stand
from the page without a manual**, with a lab member.

**The first run** adds: a newcomer following the **Next step** on `demo-sample` in a real
browser, with the owner (the walk of the [issue](https://github.com/EPFL-TOP/smart-microscope-control/issues/293)'s
live session); the `command:` line pasted into PowerShell and into `cmd.exe` (it is
quoted for a POSIX shell, and a Windows page says so; what the two shells do with it is
not known); the page on a Windows PC at all (the POSIX line); a real browser's emptied
*Wells* box; the strip's `assumed:` tail and **Refresh** with a Run pressed on the page;
the *Wells* box changed by a click while someone types in it (it is changed only when the
page redraws, never by the periodic callback); and the page over RDP.

**Presets** add: the region in a real browser, with the owner, on the assay they repeat;
a preset saved by the service ([service.md](service.md)) into the profile's folder, where
the service's account may not be allowed to write; a preset edited in Notepad (a
byte-order mark, CRLF line ends) and read back on Windows; and the width of the region's
select and buttons in the 400 px step panel.

**The steps and the regions** add, for the owner: the page on the stand PC's screen and
over RDP (`/ui-check` read it in headless Chromium at 1280 x 800 on macOS, on the two demo
stands, and at no other size); a 1280 px window with the browser's own zoom; whether a
newcomer reads the step bar as the path (the numbers, the ticks, `next`) and finds the
expert switch; how much of the drive bar a stand's window shows (at 1280 x 800 it is
partly under the fold, so the scroll that reaches it must leave STOP in view: the strip is
sticky, seen working in headless Chromium, not on the stand); and the *Continue to …*
button at the foot of a long screen, which is a scroll down in the step panel.

**Registering from images** adds, for the owner at the Ti2-E: the walk of the section above
(steps 3 to 6) with the joystick, Live and the page open; the time of a `plate-walls` run on
a real stand at 10× (16.5 s is the demo stage's, two wells); whether two captured centres
should make `plate-walls` the next step, instead of `plate-registration` (the page offers
the measured path first because it is the one the owner asked for; both stay on the page);
and `plate-walls` on a 4× frame.

**The scan plan on the map** adds, for the owner: the map at the stand's screen and zoom
(72 fields of 512 µm at the full plate's extent are a few pixels each, and 5000 are more
than the eye separates); the estimate against the time of a real scan on the Ti2-E (it
was measured on the demo's simulated stage); and the plan of a real 384 or 1536-well
plate in a browser.

**Connect and Disconnect** add: Connect, Disconnect, NIS-Elements taking the stand,
then Connect again, at a real stand; **a stand opened on the page's worker thread**,
which then ends, where the command used to open it on its main thread (a vendor
adapter that cares which thread loaded it is the finding); and the service of
[service.md](service.md), on a microscope PC.
