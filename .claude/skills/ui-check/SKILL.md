---
name: ui-check
description: Look at the plate page like a user does — serve it on a demo stand in dry run, drive it with headless Chromium at 1280 x 800 through eight states (empty, a typed well range, a snap, the expert view, after density, with the route, Run all that are left, the plate turned), at 1280 x 620 through two more (the Stand screen with a calibration seeded, and the Plate screen), at 1280 x 800 through three guided ones (the fields laid out, the Time line read under the autofocus's three modes, the scan done) and, last, at 1280 x 620 the Stand screen after a calibration that failed, fourteen in all, and read the PNGs and summary.txt it writes. Use it on every area:ui issue, in /develop before the adversarial review and in /review of an area:ui PR; it also says how a session drives the served page through a Chrome MCP (tried once: screenshots and clicks, no page reading). Never an assertion or a baseline.
---

# /ui-check

The tests build the plate page and click on it without ever opening it in a
browser, so a defect that only a real Bokeh session shows passes all of them (#277: a
range typed in the Wells box raised on the server, and every test was green;
FM-161). This skill is how a session *looks* at the page: one script, a handful
of PNGs, a text summary and the server's log. A PNG is for looking and for citing;
it is never a baseline, because PNGs differ across systems, fonts and Chromium
builds.

## When

- **Every `area: ui` issue**, in `/develop`, after the tests are green and before
  `/adversarial-review`: on the states the change touches.
- **`/review` of an `area: ui` PR**: the same run, from the PR's own worktree.
- Always from the worktree of the change and **its own `.venv`**, never by switching
  the main checkout (other sessions share it). The script serves the code it sits
  in: `sys.executable -m smc.cli` of the venv that runs it.

## Once per PC

```text
pip install -e ".[dev,ui]"                  # a worktree's `worktree.py setup` does it
python -m playwright install chromium       # about 290 MB, in the user's cache
```

It is **once per Playwright version**, not once per PC: each version wants its
own chromium build, a fresh worktree's `.venv` may hold a newer Playwright than
the cache was filled for, and the install then **removes the older build** from
the shared cache, which breaks `/ui-check` in every worktree that still has the
older Playwright (#410, #382). When the script says the browser is missing, run
the install with that worktree's own `.venv/bin/python`, and expect another
worktree to need it again.

On a bare Linux add `--with-deps` to the second command. The script says what to run
when the package or the browser is missing, and does not start a server before it
has both.

## Run

```text
.venv/bin/python scripts/dev/ui_snap.py --out <scratchpad>/ui-check
```

(Windows: `.venv\Scripts\python`.) About 60 s on macOS (measured once, 2026-11-23, with
`wells-timed`; 33 s before it), 19 s of them the `scan` of
the chain; `--no-results` skips the chain when the change has nothing to do with
results, and then `after-density`, `with-route`, `results-all` and `turned` are named as not reached. `--stand demo` serves
the demo whose camera images no sample. Nothing else is a parameter: there is no
`--profile`, because the page can move a stage, and the script serves only
`demo-sample` and `demo`, the repository's own files, and always with `--dry-run`.

| State | What the script does | Reached on |
|---|---|---|
| `empty` | nothing: the page as it opens, with the stand closed (#264) | both stands |
| `wells-selected` | clicks Connect (when enabled) and waits for the strip, then the *3. Wells* step (its button is found by the name that starts with its number, `^3\. Wells`, since the label gains a `✓` when the step is done), then types `A1:B2` in the Wells box, Enter. The box's label is one match now that the step panel shows one screen, which ended #395 | both |
| `snap` | the Snap button, always in view beside the map: no tab, and the screen does not change | both |
| `expert` | the *Expert view* switch, a wait for its `bk-active` class (a pressed Toggle's button is `bk-btn bk-btn-light bk-active`), the photograph, then the switch off again | both (it is the fourth, so that `demo` reaches it) |
| `after-density` | runs the workflow while the page is up (`plate-registration`, `scan-plan`, `scan`, `cells`, `density`, on the demo's simulated stage), then a Snap (the page reads the run document again when it ends: #297, **no Refresh**), then the *5. Results* step and a wait for the density row's state line (the second `Done <date> UTC`; the checklist's `✓ density` is the expert view's now). Read the map for the **whole plate** and **plain tick labels** (−60000 to 60000, not `5.000e+4`: FM-174) | `demo-sample` |
| `with-route` | `route`, then Refresh (in the map's header row) | `demo-sample` |
| `results-all` | the *5. Results* step, a wait for the density row's `Done`, then **the screen's own *Run all that are left*** (found by its whole name, so no row's *Run* answers to it) and a wait for the strip's `run all done`. After `with-route` the run document holds a real `cells`, `density` and `route`, so on the dry-run page the chain runs `report` and `nis-export`, which move nothing (measured, m5 §17.11.1). Read the **five rows** (state lines, *Run* or *Again*, a wait under a button that has one), *Run all that are left* with its line, the files block (`Where the files are`) and the colour bar's title on the map (`<class> cells per mm²`). `summary.txt` says `row <name>:` for each of the five rows' state lines, `chain:` and `files:`, and for the frame's parts and the screen's **title** (`Results`) and **first action** (*Run all that are left*) whether each is wholly inside the 800 px page. The five rows are taller than the page: say in the Report what is below the fold (`results-all-full.png`; the screen's own panel scrolls) | `demo-sample` |
| `turned` | the *2. Plate* step, then the *Plate lies* select set to its `Turned 180°` option (after a wait for the select: the server draws the screen), then a wait for the strip's `plate-orientation done` line (m5 §16.4). Compare it with `with-route` by eye: the **same cross, other names** (the density and the route are stale after the turn, so they are gone) | `demo-sample` |
| `stand-620` | **a real `smc run camera-calibration` beside the page first** (the page is in dry run and "cannot measure the camera"; 0.2 s on the simulated stage: pixel size 1.0, rotation 0.0, not mirrored, the 10X; #658), then Connect (when enabled), *Refresh* (it forgets the outcome the earlier states left under the strip: D12 measures the header with its outcome lines empty, and the page reads the run document again, so it has the calibration), the viewport **resized to 1280 x 620**, then the *1. Stand* step and a wait for the screen's title and for the numbers pane's first line (`Camera calibration at <objective>, measured <when>.`, m5 §23.2); the viewport goes back to the run's after the photograph. `summary.txt` says for the strip's STOP, the step bar, the screen's **title** and its **first action** (the preset's *Load*) whether each is wholly inside the 620 px page (m5 §17.2 D12), and for the numbers pane **`calibration numbers, first line`**: whether its **first line** is inside (the pane is three lines and the box of the whole pane would say `NO` for a pane whose first line is in view: it is read by the first character's own box), then `calibration seed: seeded by a real camera-calibration on the simulated stage`. The calibration stays in the run document, so the states after this one (`plate-620`, the guided ones) are a calibrated stand's: the arrows' note says which way the image moves, and the plan's `assumed` lines have no `no camera calibration ... rotation 0 and no mirror assumed` line | both stands, last (no chain); the seed on `demo-sample` only: on `demo` the same run fails (its camera images no sample), nothing is seeded, the pane is `not on the page` and the seed line says `not seeded (the demo camera images no sample: it cannot be measured)`. A seed that fails on `demo-sample` is a state **not reached**, named |
| `plate-620` | The same, with the *2. Plate* step. Its first action is the first row's *Go to*. On `--stand demo` the screen is in its worst case (the placeholder plate, the confirmation, the orientation note of a stand that does not say its sides): read it there too, since `demo-sample` is registered by the chain's end | both stands, last (no chain) |
| `wells-laid-out` | the guided path to a plan, by the screens' own buttons: Connect (when enabled), the *2. Plate* step, *Register the plate* (a wait for the strip's `plate-registration done`), the *3. Wells* step, `B2,B11` in the Wells box, *Lay out the fields* (a wait for `scan-plan done` and for the known block's `Fields: <n> in 2 wells`). At 1280 x 800. Read the **orange fields** on the map (a field is 512 µm on a plate of 127 mm: a pixel or two) and the Wells screen's known block: `Time: not known …` on a page with no timing. `summary.txt` says for the STOP, the step bar, the map, the image, the drive bar and the screen's **title** (`Which wells?`) and **first action** (*Lay out the fields*) whether each is wholly inside the 800 px page | `demo-sample`, last |
| `wells-timed` | goes on from `wells-laid-out`, on the Wells screen (m5 §17.12.12 D42, #592): a real `smc run timing --param repeats=1` beside the dry-run page (`timing` refuses a dry run, so it cannot be a click of the page; the simulated stage, as the chain's steps), *Refresh* in the map's header row, *Lay out the fields* again (the plan before read no timings, so it is out of date), a wait for `Time: about`, which only the new plan says. It then reads the known block's `Time` line three times, with the scan's `autofocus` field left empty (the tool's default, `if-engaged`), typed `off` and typed `required` (Enter each time, a wait for the line to change, never a delay: FM-162), and puts the field back before the photograph. `summary.txt` says `time (autofocus default): …`, `time (autofocus off): …` and `time (autofocus required): …`, and for the STOP, the step bar, the map, the image, the drive bar and the screen's **title** (`Which wells?`) and **first action** (*Lay out the fields*) whether each is wholly inside the 800 px page. The three lines on the demo (the figures vary by a second with the run's own timing): `Time: about 15 s, with the autofocus locked again in each well; about 14 s if it is off at the start`, `Time: about 14 s, with the autofocus not used`, `Time: about 15 s, with the autofocus locked again in each well`. A `timing` that fails is named as every chain failure is | `demo-sample`, with or without `--no-results` (it needs no result of the chain), after `wells-laid-out` |
| `scanned` | then the *4. Scan* step and *Start the scan* (a wait for `scan done`). The page is a dry run, which walks the fields and takes **no frame**: the bar is full, the lines say `Field <n> of <n>` and `Elapsed`, the result line says `Scanned in a dry run: no frame was taken`, and the map has **no green field**. `summary.txt` says the progress lines and the result line (`progress:`, `result:`), and whether the title (`Scan B2 and B11`) and the first action (*Start the scan*) are inside the page. A scan in motion is the one-off session below | `demo-sample`, last |
| `stand-failed-620` | **a `camera-calibration` that must fail** beside the page first (`step_um=2000`, the tool's own largest probe: `the frames barely correlate ... a step over 40 % of the field may wrap` on `demo-sample`, `the two moves gave parallel shifts` on `demo`, both exit 1, measured; its record ends `failed` and **replaces** the good calibration, which is why this state is the last of all: #658), then the same as `stand-620` (Connect, *Refresh*, 1280 x 620, the *1. Stand* step) and a wait for the line under *Calibrate the camera* (`The last calibration failed: <the tool's words>`, m5 §22.2). `summary.txt` says the Stand screen's four parts as `stand-620` does, then **`calibration failed, first line`** (its first line only: the sentence wraps to several lines below it) and `failing calibration seed: camera-calibration exited 1: <the tool's own line>`. A seed that passes, hangs or exits with another code than 1 is a state **not reached**, named | both stands, last, with or without `--no-results` (it reads no result of the chain) |

The calibration that `stand-620` seeds is the only one of the dry-run page's states that
comes from a run beside it other than the chain's: the page cannot measure, and the Stand screen
says a calibration's numbers (m5 §23.2) only from a record. The failed one needs no sample.

The three guided states register the plate from the sample themselves and read none of the
chain's results (`wells-timed` runs one `timing` of its own), so they are reached with
`--no-results` and after a chain that failed. On `--stand demo` they are named: `not
reached: the demo has no sample to register from`.

A state it cannot reach is **named, never skipped**: `<state>: not reached: <why>`,
on stdout and in `summary.txt`. It never clicks a tool's own Run button, Drive, STOP or
Resume. The buttons of the **guided screens** (*Register the plate*, *Lay out the fields*,
*Start the scan*, and the Results screen's *Run all that are left*) are clicked, on the
dry-run page only, by the three guided states (`wells-timed` clicks *Refresh* and *Lay out
the fields* again, and types in the scan's `autofocus` field) and by `results-all`: each is a
run that moves nothing in a dry run. `turned` starts a run from the page too, its `plate-orientation`,
which moves nothing either.

**A scan in motion** (the progress bar, `Field n of 72`, *Elapsed* and *Left*, the green
fields growing) cannot be photographed on the dry-run page: its scan ends at once. It is a
**one-off browser session**, in the develop session's scratch directory, cited in its Report
and never committed: `demo-sample` served **without** `--dry-run` (a simulated stand, see
*Live mode* below for the check of the profile) on a scratch run folder, driven by Playwright
through the Plate, Wells and Scan screens, with PNGs taken while the scan runs (#445).

**The Wells screen's note at the 40X** (*Lay out the fields* will use the 40X's own calibration, or
says that none has been made, m5 §18.6 D24 to D26, #617) is not a state either, and cannot be one
on this page (#658): it needs a plate registered **from its walls** under another objective than
the one in place. The dry-run page cannot turn the turret (a confirmed move, refused in a dry run),
a registration from walls is a tool run that needs three wall captures, and each `smc run` opens
its own simulated stage at the default objective, so no run beside the page registers at another
one. What is left is a record written by hand with another objective's label (a fabricated state)
or serving without `--dry-run`, which this skill's guard forbids and the owner decides. It stays a
**one-off browser session** like the offsets block's below: the 10X registered from walls, the page
served without `--dry-run` on a simulated stand, the turret turned to the 40X on the Stand screen;
the note with no document, then after *Calibrate the camera*, then *Lay out the fields* (#617's
Report has the numbers).

**The offsets block** (*Measure the offsets between objectives*, m5 §18, #382) is not one of the
states above and must not appear in them: the script serves `demo-sample` always with `--dry-run`,
the tool refuses a dry run, and the dry-run page's plate is registered from the sample's centres,
not from walls, so the block is **hidden** there (the regression to read in `summary.txt`). It is
checked in a **one-off browser session** of the same kind: a scratch copy of `demo-sample.toml`
with `[sample]` turned as m3 §20.1 (`rotation_deg`, `camera_rotation_deg`, `mirrored`) and
`objective_shift_um` set for the 10X, 20X and 40X (m3 §20.8); a scratch run folder holding
**only** `run.json`, with the 10X's calibration and `plate-walls` on A1, H12 and A12, built through
`run_plugin` (since #501 the page makes the other calibrations itself, m5 §18.6 D18: no document
of the 40X or the 20X is made beforehand); the page served **without** `--dry-run` on that profile
and folder, on a free port; and Playwright at 1280 x 800: the Stand screen with the turret turned
to the 40X (*Change objective*, confirmed), the note under *Calibrate the camera* that says where
its calibration goes (the 40X's own document `calibrations/position-1.json`, the plate's
calibration staying as it is, m5 §18.6 D19) and its *Command* disclosure (the routed call and the
note that the page makes the folder); the click, then `calibrations/position-1.json` made and
`run.json` byte-identical; the Plate screen's block offered with its lead and its list of
documents (empty, every objective `no calibration yet`) and the button naming the 40X; the list
after the click; the same for the 20X, and a second click at the 20X (`Calibrate the camera again
at ...` and its note); the field left empty, *Measure the offsets* and its confirmation, the block
while it measures (twice, so that `Objective in place` shows two turns; at the demo's default
10 mm/s a run takes about a minute), the card and the preview, the session typed, the write, the
written line and the profile file read. One more with a missing path typed and one with the
plate's own document, for the waits. The light of the demo has no switch (`is_on()` reads
`None`), so the refusal for a light that reads off (D20) is not walked: the tests hold it. The
dry-run states above still show the block **hidden** or off the screen: read the `offsets block:`
line of `summary.txt` (below). The PNGs are cited in the develop session's Report and never
committed.

**A chain in motion** (*Run all that are left* with a step running and the others waiting:
`Running now`, `after <step>`, the chain's line `Running <step> (k of 5).`) is the same kind
of session, through the Plate, Wells, Scan and Results screens, with PNGs of Results before the
click, right after it and after it (#446). On the demo the five tools take about 2 s in all
after the scan (`cells` is cold for about 2 s, and a row may read `Running now` then): a chain
that is too fast to photograph is said so in the Report, and `results-all` above is what
the script reads.

## Read

1. **The exit code and the last line** of the output. `0` nothing to read; `3` the
   PNGs are written and something needs reading (a state not reached, a server or a
   browser error, a server that did not stop on Ctrl-C); `1` the job could not be
   done (no browser, the server never served: its log's tail is printed); `2`
   refused. Exit 3 is "read", not "broken".
2. **`<state>.png`**, the viewport (1280 x 800, the floor the layout is designed for: `--width` and `--height` change it), with the Read tool, for each
   state the change touches. Start here. `<state>-full.png` is the whole page and is
   shown too small to read: use it for the part below the fold only.
3. **`summary.txt`** for what a PNG cannot give: the strip, the outcome, the
   checklist and the result cards as text, the step shown (`step:`, the button of the class `smc-step-shown`; with the expert view on, none is), what is on screen (`on screen:` says for the strip's STOP, the step bar, the map, the image and the last control of the drive bar whether each is wholly in the viewport and where; for `stand-620`, `plate-620` and `stand-failed-620` it says it for STOP, the step bar, the screen's title and its first action (and the two Stand states for the **first line** of their calibration pane, `calibration numbers, first line` or `calibration failed, first line`, then the seed they ran: `calibration seed:` or `failing calibration seed:`), and for `wells-laid-out`, `wells-timed`, `scanned` and `results-all` for the frame's parts and the screen's title and its first action; `wells-timed` also says `time (autofocus default|off|required):`, the Wells screen's `Time` line read under each mode, `scanned` says `progress:` and `result:`, the progress lines and the result line, and `results-all` says `row <name>:` for each of the five rows, then `chain:` and `files:`; **every** state ends its `on screen:` lines with `offsets block: shown`, `hidden`, `not on this screen` or `?` (the browser refused the read), from the block's *Measure the offsets* button counted with the hidden ones: `hidden` is the block on a screen that draws it while the page does not offer it, `not on this screen` a screen that does not draw it at all (Panel draws only the screen shown), and the dry-run states of the Plate and expert screens are `hidden`: a `shown` there would be the regression, m5 §18.6 D23; never an assertion, read it), the Wells box, how the
   server stopped, the server's error lines and the browser's. The map and the image
   are **canvas**: they are in the PNGs only.
4. **`server.log`** (the server's stdout and stderr as it wrote them) when the
   summary counts `server errors`, and `chain.log` when a step failed.

## Cite a finding

`ui-check <state>.png: what the picture shows`, and the summary line when there is
one: `ui-check wells-selected.png: the box shows A1:B2 and no well is outlined;
summary.txt: server errors: 4 (Callback failed for object named 'Wells')`. Say which
stand and which commit. A finding that only a human can judge (a layout the owner
would not choose) is reported as what it shows, not as a defect.

## Known today (#268, macOS)

- `wells-selected` ends exit 3 with **4 server error lines** until **#277** is fixed:
  the typed range raises in `on_select_text` (`_pending_writes should be non-None…`)
  and draws nothing. That is the script doing its job; do not work around it.
- At 1280 x 620 (#658, macOS, `demo-sample`): the numbers pane of `stand-620` has its first line
  at y 559 to 573 and its second and third (the pixel size, the rotation, the file) **under the
  fold**; the failed line of `stand-failed-620` starts at y 600 to 614 of 620 and its sentence wraps
  below the page. Both first lines are inside; the rest is the owner's to judge.
- At 1280 x 800 the **drive bar is under the fold** (m5 §17.3 wants it on screen; #443
  measured its last control at y 927 to 958 with figures of 460 px and shortened them to
  400 px, the floor): `summary.txt` says `on screen: drive bar: NO`. That is a finding
  for the owner, not for the change you are making.
- `snap` shows noise and the strip says `DRY RUN`: in dry run the stage does not
  move, so every frame of `demo-sample` is plastic. The chain runs *without*
  `--dry-run` on the simulated stage for that reason, and the page, which is in dry
  run, only **reads** its results: a `Refresh` redraws it from the file.
- Not measured: Windows (the stop is `terminate()`, and the summary says that the
  Ctrl-C path was not exercised), Linux, Python 3.10, Panel 1.9, Playwright before
  1.64.

## Never

- Run a tool from its own section, drive the stage or press STOP through the page (the
  guided screens' three buttons, in the three guided states, and the Results screen's *Run all
  that are left*, in `results-all`, are the one thing it presses, on the dry-run page); start
  the page on any
  other stand, or without `--dry-run` (the one exception is below: a simulated stand
  in a live walk where the owner wants the Run buttons pressed); run two sessions on
  the same page.
- Compare a PNG with another, or put one in a test.
- Put a browser in `pytest` or in CI.

## Live mode through a Chrome MCP — tried once

For a design session with the owner, who watches the page change while the
session drives it. **What one session learned** (Claude in Chrome, `demo-sample`,
dry run, 2026-10-30, Report on #293): the MCP's page reading (`read_page`,
`find`, `get_page_text`) returns **nothing** on a Panel page, because its
widgets live in a shadow DOM; everything was done from **screenshots and clicks
at coordinates**, and the Wells box was filled by clicking it and typing. The
chain (`plate-registration`, `scan-plan`, `scan`, `cells`, `density`,
`route`) ran from a terminal beside the page, as the procedure says, and the
page was redrawn with Refresh; no Run button was pressed. It found #297 and the
seven first-run findings on #293.

**The walk of 2026-11-03** (Claude in Chrome, macOS, `demo-sample`, `main` at
9b9cc75, findings on #310): a newcomer's Connect and then the Next step each time, to
the export, with **the Run buttons pressed** at the owner's request. Screenshots and
clicks again; the Wells box was filled by **clicking the wells on the map** (B2, B11),
which the page echoes into the box. The page had been served with `--dry-run`, as the
guard says, and the path **ended at `cells`**: a dry-run `scan` takes no frame, `cells`
fails on it, and the step said `cells` again, advice the page cannot follow (#310). The
page now stops the step at `scan` on a dry run and says why; the guard below is the
other half of that decision.

**The walk of 2026-11-11** (Claude in Chrome from the VS Code extension, macOS,
`demo-sample`, B1's shell at 8549b4f, findings on #431):

- **Connecting.** The extension stayed `Not detected` until Claude Code had written
  its native-messaging host. That host is the file
  `com.anthropic.claude_code_browser_extension.json` in Chrome's
  `NativeMessagingHosts` folder. It was written by running the extension's bundled
  binary with `--chrome` in a terminal (`claude` was not on the PATH). Chrome then
  had to be quit and reopened. The tools appeared once the owner's message used
  `@browser`.
- **Clicking by name.** A script in the page (the MCP's `javascript_tool`) walked the
  shadow roots for `button` and `pre` elements. It clicked buttons by their text and
  read the strip and the outcome lines. That replaced clicks at coordinates, which
  missed as soon as an outcome line pushed the layout down by about 28 px.
- **The window.** A 1280 x 800 window is a 1280 x 623 page. The mouse wheel over a
  figure zooms the figure instead of scrolling the page. The window once changed size
  under the session (a 1568 x 565 page), and the map drew squeezed wells until the
  page was reloaded.
- **Live and Drive.** A drive click is refused while Live runs, with the refusal at
  the top of the page, out of view. A refused or missed click disarms *Drive*, so the
  next click only selects a well.

**Every later session adds what it learns here** (another MCP, keyboard use, a narrow
window, Live, Drive).

1. **The owner installs** Claude in Chrome or a Chrome DevTools MCP server on the
   machine. No such MCP is configured today; without one this section does not apply
   and the snapshots above are what there is.
2. **The session starts the page itself**, in a background shell, on the same two
   stands and always with `--dry-run` (the same guard, by hand):

   ```text
   .venv/bin/python -m smc.cli ui plate -p profiles/demo-sample.toml --run-doc <tmp>/run.json --dry-run --port <free port>
   ```

   with `SMC_PROFILE` and `SMC_PROFILES` unset in that shell, and the profile given
   by path: a name is looked up in the working directory. At the end it stops the
   server with SIGINT (never `kill -9`) and reads the exit code.

   **When the owner wants the Run buttons pressed** (a newcomer's walk to the export),
   serve a profile whose `[micromanager] config` is `""` **without** `--dry-run`: such a
   profile (`demo.toml`, `demo-sample.toml`) loads only Micro-Manager's own demo
   devices, so its stage is simulated and there is nothing real to move. In a dry run
   `scan` takes no frame and the path ends at `cells` (#310). Check it first, with one
   line of Python in that shell, and read the answer before serving:

   ```text
   .venv/bin/python -c "from smc.hardware.profile import Profile; print(repr(Profile.load('profiles/demo-sample.toml').micromanager.config))"
   ```

   (Windows: `.venv\Scripts\python`.) It must print `''`. **Every other profile keeps
   `--dry-run`**, and so does every walk that only reads: `scripts/dev/ui_snap.py`
   never presses a tool's own Run (only the guided screens' three buttons and the Results
   screen's *Run all that are left*, in dry run) and always serves in dry run.
3. **It opens `http://localhost:<port>/`**, exactly as the command prints it: the
   websocket accepts `localhost` only, not `127.0.0.1`.
4. **It drives the page with the MCP's own tools.** What can be inspected: buttons,
   inputs, selects, tabs and the `pre` text panes, in Panel's shadow DOM (whether
   Claude in Chrome's page snapshot does not pierce it; see above). What cannot: the plate map and
   the image are two canvases, pixels only. There are no tabs: Snap is always in view,
   and the step bar's buttons (`1. Stand` to `5. Results`) switch the screens.
5. **One session on the page at a time.** The page owns the stand; a second would
   open a second one. Use a port of its own.

The MCP is for designing. What a change must pass is still the script's run.
