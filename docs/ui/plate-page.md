# The plate page

A web page on the microscope PC: the plate drawn in stage µm, wells you select with
a click or a typed range, the tools of the plate-scan workflow with a Run button
each, and, once the workflow has run, the density of each well and the route. It is
**a view only**: every button is a [plugin](../plugins/README.md) run with the
parameters `smc run` takes, or a read of a capability, so each has a command-line
equivalent ([ADR-0006](../adr/0006-user-interface.md)). The design is
[`m3-plugins.md`](../design/m3-plugins.md) §15; this page is how to use it.

It needs the `ui` extra (Panel):

```text
pip install -e ".[ui]"
```

Without it `smc ui plate` exits 1 and says so; the rest of `smc` does not need it.

## Starting it

```text
smc ui plate -p <profile> --run-doc run.json [--port 5006] [--dry-run]
```

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
| No `--run-doc` | exit 2, `a run needs --run-doc` |
| The `ui` extra is missing | exit 1, `install the ui extra: pip install -e ".[ui]"` and the import error |
| The port is taken | exit 2, `port N is in use; choose another with --port`; the stand is not opened |
| The run document cannot be read | exit 1, its reason; the file is left as it is |
| The server stops by itself | exit 1, `the plate page's server stopped on its own (is port N taken?)` |

**Ctrl-C** ends the server. If a run is going it is **told to stop first** (the same
stop as the STOP button), the page waits up to 10 s for it, and the stand is released
after. The last line then says `<plugin> was stopped`. A second Ctrl-C during that
wait still releases the stand, and the command exits 130.

A closed or reloaded browser tab does **nothing** to a run: the page belongs to the
server, not to the tab. Only STOP, Ctrl-C of the command and the plugin's own end
stop a scan.

## What is on the page

**The strip, at the top.** The stage `XY (x, y) µm`, `Z`, the objective, the autofocus
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

**The map.** The plate in stage µm, X right and Y up, wells as circles (96-well and
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

**The tools.** One section per stage of the workflow (*Set up*, *Choose*, *Acquire*,
*Find*, *Report*), one block per registered tool: its key parameters as fields, the
rest under *Advanced*, a **Run** button and the progress. An empty field is not sent, so
the plugin's default holds; the placeholder is that default. What the page fills itself
is not offered: the wells of `scan-plan`, and the plate and the captures of
`plate-registration`. A tool that cannot run now is disabled with the reason beside it
(`the run document has no plate; ...`, `this stand has no XYStage`), and so is every
tool while a run is going. A tool that is not registered (a plugin that has not been
written, or an install that predates it) is not on the page.

**Calibrating the plate.** Under `plate-registration`: centre a well under the objective
by hand, click it on the map, press **Capture centre**; for a wall, put the objective on
it, choose the side, press **Capture wall**. A second capture of the same well (and side)
replaces the first; **Forget** drops the current well's captures. **Run
plate-registration** then fits the plate from what was captured, as `smc run
plate-registration --param refs=... --param walls=...` would. Choosing another plate
clears the selection and the captures.

**The run document**, as a checklist: `✓ plate` (usable), `· cells` (not run),
`✗ scan: the last run of scan ended failed; run it again` (there, and not usable, with
why). A line above the map, in the warning colour, says what the page could not read
and did not hide: an unreadable run document (the page then shows every stage as not
run and the nominal map, and never writes the file), or a result that no longer
matches its model.

## What it does not do

No live view, no jog pad, no Z or turret control, no `--resume` (continue a stopped
stage with `smc run --resume`), no `--host`, no login. The report and the NIS-Elements
export appear as tools when their plugins are registered; they have no page of their
own.

## Not verified yet

A real browser's tap and click, the page's look, the periodic callback of a real
session, Windows (Ctrl-C of the command, the 0.1 s wait), RDP, a PC with no internet
(the served page of a few widgets has no external address; the figure's was not
checked offline), Panel's resolution on Windows with Python 3.14, the **cost of
reading the stand once a second during a scan** (it runs on the loop that handles a
click on STOP: a device read that hangs in a vendor driver would delay STOP; it is to
be timed on the Ti2-E) and what STOP's note says there. These are for the first session
at the stand ([#198](https://github.com/EPFL-TOP/smart-microscope-control/issues/198)).
