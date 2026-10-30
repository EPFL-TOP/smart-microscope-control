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
reason: a file is never created where you did not say. The command checks it, opens
the stand **once** and keeps it until you press **Ctrl-C**, then prints

```text
serving the plate page on http://localhost:5006/ — Ctrl-C stops it
```

Open that address **as printed** (`localhost`, not `127.0.0.1`: the page's websocket
accepts the origin it was served from). The page listens on `localhost` only and
has no login: whoever reaches the port can move the stage, so it is not offered on
the network. To use it from another desk, use the PC's remote desktop (RDP). An SSH
tunnel is **untried**: it could work only with the same port number on both ends.

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
then stops the server (up to 10 s, and says so if it does not), and the stand is
released after. What the command prints last depends on what happened:

| What happened | The last lines |
|---|---|
| The stand took the stop and the run ended stopped | `<plugin> was stopped`, exit 0 |
| The stand refused the stop (the Ti2-E's does) | `<plugin> told to stop; halted, but <why>`, then how the run ended: a move under way runs to its target, so it is never "stopped" by default |
| The run failed while it was stopped | the stop's line, then `<plugin>: <Error>: <message>`, exit 0 |
| The run did not end within the wait | the stop's line, then `✗ <plugin> did not end within 10 s`, **exit 1**; the stand is released under a run that is still alive |

A second Ctrl-C during the wait still releases the stand, and the command exits 130;
the stop was already sent. A closed or reloaded browser tab does **nothing** to a run:
the page belongs to the server, not to the tab. Only STOP, Ctrl-C of the command and
the plugin's own end stop a scan. Two tabs on one server share the selection, the plate
and the captures: a tab draws what another tab changed within a second.

## What is on the page

**The strip, at the top.** The stand's name (the profile's `microscope.name`), then
the stage `XY (x, y) µm`, `Z`, the objective, the autofocus
(`autofocus locked`, `autofocus on, not locked`, `autofocus off`, `autofocus ?` when a
read failed), the light, `DRY RUN` when it is one, and `HALTED` with **Resume** when the
stand is halted. `?` is an unknown, never a zero. It is read once a second, and no
faster: a scan must not be slowed by a status poll. **STOP is always offered**, a run
or not. Under it: the run's progress as `done/total message`, how the last run ended
(`<plugin> done in 12.3 s`, `<plugin> stopped`, or `<plugin>: <Error>: <message>`) and
what the last STOP did.

**STOP says what it did.** `<plugin> told to stop; stopped: the stand is halted
(Resume clears the halt)`, or `halted, but <why the stand refused>`, or `the stop failed:
<why>`. On a stand whose stages refuse a stop (the Ti2-E's, [#204](https://github.com/EPFL-TOP/smart-microscope-control/issues/204)) the strip also says so: a move already under way
runs to its target. After STOP every action is refused until **Resume**, which is
refused while a run is going.

**The centre has two tabs, *Plate* and *Image*.** Only the tab you see is drawn. The
run document's checklist stays under both.

**The map** (*Plate*). The plate in stage µm, X right and Y up, wells as circles (96-well and
below) or squares (384 and 1536), turned by the plate's rotation. Its title says which
map it is: **the registered plate** (A1 and the rotation from plate registration), or
a **nominal map** (A1 at 0, 0, not turned) when no plate is registered or the plate
you chose is not the registered one. Nothing in the nominal map says where the
plate really is. Names are drawn up to 96 wells. The selection has a thick blue
outline; the cells' **density** colours the wells (grey where unknown, a well with no
frame is unknown, not zero), for the label chosen under *Density of*. The **route** is
a red line with its stops, and the stage is a black cross.

**Choosing wells.** Click a well to select it, and again to unselect it. Or type a
range in *Wells*: `A1:D6`, `B2-B5`, `C*` (a row), `*3` (a column), `all`. The selection
becomes `scan-plan`'s `wells`, in the order you chose them. A click moves nothing.

**Driving to a well.** Switch **Drive** on (the strip then says `DRIVE: the next
click on a well moves the stage there`) and click a well: `goto-well` runs for it, at
its centre or at the wall chosen under *Drive to*. Drive is **one click**: the toggle
goes back to select after it, even if the click found no well or was refused. The page
**refuses** to drive on a map of another plate than the registered one (it would send
the stage to the same-named well of the other plate): choose the registered plate, or
calibrate the one you chose.

**Snap and the image** (*Image*). **Snap** takes one frame from the camera, makes the
*Image* tab the one you see and draws the frame within a second. The title is the frame's
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

**Live** (*Image*, beside Snap). **Live** takes frames from the camera one after the
other, **at most 10 a second**, and the page draws each new frame: the frame number is
what it compares, every 0.1 s, so it draws nothing when there is nothing new. The
button then says **End live**. While live goes the line under the strip reads
`live: <n> frames`. A camera slower than 10 frames a second runs at its own rate. Live
is **one activity**, like a run: **Snap and every Run button are disabled**, a click on
the map in drive mode is refused, and each says `live is running`. Live is never ended
for them: end it with **End live**, which ends only the live view and **does not halt
the stand** (it says `live ended: <n> frames in <s> s`). **STOP** ends it **and halts the
stand**, as it halts a run (`live told to stop; stopped: the stand is halted (Resume
clears the halt)`). A frame that fails, or that is not 2-D `uint8` or `uint16`, ends
live with `live ended after <n> frames: <why>`, and the last good frame stays. Live is
disabled with `this stand has no Camera` when the stand has none, and while another
activity goes. **A closed or reloaded tab does not end live**: the page belongs to the
server, so a live view with no tab open keeps the camera taking frames until STOP,
*End live* from a tab, or Ctrl-C of the command. A live view writes nothing to disk and
nothing to the run document. Every frame is sent to every open tab, up to 10 a second.
The **Drive** controls stay usable while live goes (below).

**Drive** (the right-hand column, above the tools). Each control is the call of a
command, so each has its command-line twin:

| Control | Command |
| --- | --- |
| the pad `↑` `←` `→` `↓` | `smc stage jog DX DY` |
| `Z +`, `Z −` | `smc z jog DZ` |
| **Autofocus on**, **Autofocus off** | `smc autofocus engage`, `smc autofocus disengage` |
| **Change objective**, then **Turn the turret** | `smc objective set LABEL --confirm` |
| **Light on**, **Light off** | `smc light on`, `smc light off` |

The pad is in the **stage's axes**, as the map (X right, Y up): `→` jogs `+step` in
X, `↑` `+step` in Y. Which way the *image* moves for `↑` depends on how the camera is
turned on the stand, and is not known yet. The step is chosen under **XY step** (1, 10,
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
The choice starts at the objective in place and is never changed by the page after
that: the strip says which objective is in place.

Each action is **one activity**, like a snap: it is disabled, and refused, while a run
or a snap goes, and its result is the line under the strip (`stage jog: XY (100.00,
0.00) µm`, `objective set: objective 'Nikon 20X Plan Fluor ELWD'`, `autofocus engage:
autofocus did not lock; it is off`, or the layer's refusal, such as the jog limit).
**While live goes, the drive controls work**: the live view runs the action between two
frames, and the next frame shows where the stage went. One action waits at most: a
second click while one waits is refused (`stage jog is waiting`), never queued. STOP,
*End live*, Ctrl-C or a frame that fails **drop** an action that still waits: it is
never sent, and live's last line says `; <action> was not run`. After STOP, a move, a
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
`scan-plan`, and the plate and the captures of `plate-registration`. A tool that cannot
run now is disabled with the reason beside it (`the run document has no plate; ...`,
`this stand has no XYStage`), and so is every tool while a run is going. A plugin that
failed to load, or whose parameters cannot be described, is not a tool: a line above the
map says `plugin <name>: <why>`.

**A tool's result**, under its Run button, as plain text (a result's words are never
read as markup): `<status> <time> UTC`, then ` on <profile>` when the record was made
on another stand, ` (dry run)` when it was one and ` (stale)` when the tool read an
input that has been produced again since (the checklist says `<plugin> read its
inputs from an earlier run; run it again`; the values are kept, and marked), then the
result's values, two spaces in, a line each (a text cut at 80 characters, a list or an
object only counted, at most eight, then `… n more`), then what the result **assumed**
(at most eight, then `… n more`, each cut at 600 characters) and the error (cut at 600
characters too). A card is a summary, bounded whatever the record holds: the run
document and the log hold the rest. A tool whose last run was stopped or failed shows
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
nothing is written. **To continue it, close the page (it holds the stand's connection) and
run `smc run NAME --run-doc ... --resume`.** To give the checkpoint up, `smc run NAME
--run-doc ...` without `--resume` starts afresh (and logs that it does); the page does not
offer that. The fields are not compared: typing another value does not lift the refusal.
A stage with no checkpoint, or whose checkpoint `--resume` would refuse (another stand, a
dry run's, a stale one), is started from the beginning by the button, as it always was.
What the first attempt wrote to disk (a scan's frames) is kept either way.

**Calibrating the plate.** Under `plate-registration`: centre a well under the objective
by hand, click it on the map, press **Capture centre**; for a wall, put the objective on
it, choose the side, press **Capture wall**. A second capture of the same well (and side)
replaces the first; **Forget** drops the current well's captures. **Run
plate-registration** then fits the plate from what was captured, as `smc run
plate-registration --param refs=... --param walls=...` would. Choosing another plate
clears the selection and the captures.

**The run document**, as a checklist: `✓ plate` (usable), `· cells` (not run),
`✗ scan: the last run of scan ended failed; run it again` (there, and not usable, with
why). The slots of the plate-scan workflow come first, in its order; any other slot the
document holds (a third-party plugin's stage) follows by name. A line above the
map, in the warning colour, says what the page could not read and did not hide: an
unreadable run document (the page then shows every stage as not run and the nominal
map, and never writes the file), or a result that no longer matches its model.

## Trying the workflow without a microscope

```text
smc ui plate -p demo-sample --run-doc run.json
```

`demo-sample` is the demo stand whose camera images a synthetic 96-well plate
(`profiles/demo-sample.toml`), so every tool of the page can be pressed on a PC with
no microscope, and the frames each one takes follow the stage the page moves. The
stage and the other devices are the demo's; only the pixels are drawn, at the stage's
position. Press the tools in the order of the checklist, as at a stand:

- **The plate**: run **plate-finding**, which surveys the plate by itself and takes
  about 7 minutes on the demo stage. Or register it **by captures**, as at a stand:
  start **Live**, centre a well with the **Drive** pad (the frames follow the stage),
  click it on the map, **Capture centre**, and again for a second well, then **Run
  plate-registration**. The plate is centred on the stage origin, so A1 is at
  (−49 500, 31 500) µm and H12 at (49 500, −31 500) µm: drive there in 1000 µm steps,
  then finish with smaller ones. The `plate-registration` command of
  [the plugins page](../plugins/README.md#trying-the-workflow-without-a-microscope)
  registers the same two wells from a terminal.
- **scan-plan** and **scan**: choose wells with the map, and give `coverage` a small
  value (0.4) so that a scan takes seconds.
- **cells**: choose **detector** `blobs` in its key parameters. The trained model does
  not see the synthetic cells, and `blobs` counts bright round spots; the result says
  that on a real sample those counts are not cells.
- **density**, **route** and **report** are used as on any stand.

**What is synthetic, and the limits**, as `smc doctor -p demo-sample` prints them
after the roles:

```text
Sample: a synthetic 96-well plate, A1 at (-49500.0, 31500.0) µm, turned 0°; 40 cells per well on average, 30 % of them doublets (seed 0)
  frames are rendered where the stage is, not imaged: one flat plane at z = 0 (Z only blurs), no optics
  pixel sizes up to 4 µm (the texture aliases above); the cells are bright round spots: run cells with detector=blobs
```

The same commands, from a terminal, are in
[Trying the workflow without a microscope](../plugins/README.md#trying-the-workflow-without-a-microscope).
The page on this stand has not been tried in a browser (see below).

## What it does not do

No light intensity or channel, no focusing through the autofocus offset, no
press-and-hold jog, no drive click on the map while live goes (it is a `goto-well`
run, which writes a record), no camera settings (exposure, ROI, binning) and no time
limit on live,
no Connect / Disconnect and no service ([#264](https://github.com/EPFL-TOP/smart-microscope-control/issues/264)),
no `--resume` (continue a stopped stage with `smc run --resume`: the Run button of such a
stage is disabled and says so), no `--host`, no login. The report and the NIS-Elements
export appear as tools when their plugins are registered; they have no page of their
own.

## Not verified yet

A real browser's tap and click, **the switch to the *Image* tab and the frame's
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

**Live** adds: the rate a real browser shows, locally and over RDP (the demo camera
gives 10 frames a second on the server; ADR-0006's revisit criterion for Panel is what a
browser shows); the Ti2-E's camera at full frame (Prime 95B, 1412×1412, a stride of 2);
the cost of `Microscope.state()`, which is read for every frame, on a real stand;
a mechanical shutter under auto-shutter, which opens and closes for every frame (up to
10 times a second); and the frame's orientation in a browser.

**Drive** adds: the drive at a stand; which way the image moves for `↑` and `→`; a
jog, a focus step and an engage on the Ti2-E while live goes, and how long a click
waits for the frame in flight on a real exposure; a turret turn under a plate; *Light
on* and *Light off* on the Ti2-E's `DiaLamp`; what the stand keeps on after the page
or a command ends (the light, the autofocus: closing the stand switches nothing off);
`Z +` and `Z −` with the autofocus locked on the Ti2-E (the relock may bring Z back to
the offset's plane); and **a newcomer registering a plate on the `demo-sample` stand
from the page without a manual**, with a lab member.
