---
name: ui-check
description: Look at the plate page like a user does — serve it on a demo stand in dry run, drive it with headless Chromium through five states (empty, a typed well range, a snap, after density, with the route), and read the PNGs and summary.txt it writes. Use it on every area:ui issue, in /develop before the adversarial review and in /review of an area:ui PR; it also says how a session drives the served page through a Chrome MCP (untried). Never an assertion or a baseline.
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

On a bare Linux add `--with-deps` to the second command. The script says what to run
when the package or the browser is missing, and does not start a server before it
has both.

## Run

```text
.venv/bin/python scripts/dev/ui_snap.py --out <scratchpad>/ui-check
```

(Windows: `.venv\Scripts\python`.) 33 s on macOS (measured), 19 s of them the `scan` of
the chain; `--no-results` skips the chain when the change has nothing to do with
results, and then the last two states are named as not reached. `--stand demo` serves
the demo whose camera images no sample. Nothing else is a parameter: there is no
`--profile`, because the page can move a stage, and the script serves only
`demo-sample` and `demo`, the repository's own files, and always with `--dry-run`.

| State | What the script does | Reached on |
|---|---|---|
| `empty` | nothing: the page as it opens | both stands |
| `wells-selected` | types `A1:B2` in the Wells box, Enter | both |
| `snap` | the Image tab, the Snap button | both |
| `after-density` | runs the workflow while the page is up (`plate-registration`, `scan-plan`, `scan`, `cells`, `density`, on the demo's simulated stage), then Refresh | `demo-sample` |
| `with-route` | `route`, then Refresh | `demo-sample` |

A state it cannot reach is **named, never skipped**: `<state>: not reached: <why>`,
on stdout and in `summary.txt`. It never clicks a tool's Run button, Drive, STOP or
Resume.

## Read

1. **The exit code and the last line** of the output. `0` nothing to read; `3` the
   PNGs are written and something needs reading (a state not reached, a server or a
   browser error, a server that did not stop on Ctrl-C); `1` the job could not be
   done (no browser, the server never served: its log's tail is printed); `2`
   refused. Exit 3 is "read", not "broken".
2. **`<state>.png`**, the viewport (1600 x 1000), with the Read tool, for each
   state the change touches. Start here. `<state>-full.png` is the whole page and is
   shown too small to read: use it for the part below the fold only.
3. **`summary.txt`** for what a PNG cannot give: the strip, the outcome, the
   checklist and the result cards as text, the active tab, the Wells box, how the
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
- The tools column (`width=420`) runs past the right edge of the viewport at 1500
  to 1700 px: the result cards are fully visible in `-full.png` only. A finding for
  the owner, not for the change you are making.
- `snap` shows noise and the strip says `DRY RUN`: in dry run the stage does not
  move, so every frame of `demo-sample` is plastic. The chain runs *without*
  `--dry-run` on the simulated stage for that reason, and the page, which is in dry
  run, only **reads** its results: a `Refresh` redraws it from the file.
- Not measured: Windows (the stop is `terminate()`, and the summary says that the
  Ctrl-C path was not exercised), Linux, Python 3.10, Panel 1.9, Playwright before
  1.64.

## Never

- Run a tool, drive the stage or press STOP through the page; start the page on any
  other stand, or without `--dry-run`; run two sessions on the same page.
- Compare a PNG with another, or put one in a test.
- Put a browser in `pytest` or in CI.

## Live mode through a Chrome MCP — **untried**

For a design session with the owner, who watches the page change while the
session drives it. Nothing here has been run: **the first session that uses it
writes what it learned into this section** (which tools exist, what they return,
what the page's shadow DOM allows).

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
3. **It opens `http://localhost:<port>/`**, exactly as the command prints it: the
   websocket accepts `localhost` only, not `127.0.0.1`.
4. **It drives the page with the MCP's own tools.** What can be inspected: buttons,
   inputs, selects, tabs and the `pre` text panes, in Panel's shadow DOM (whether
   the MCP's page snapshot pierces it is untried). What cannot: the plate map and
   the image are two canvases, pixels only. The tabs are dynamic: the Snap button
   does not exist until the Image tab is active.
5. **One session on the page at a time.** The page owns the stand; a second would
   open a second one. Use a port of its own.

The MCP is for designing. What a change must pass is still the script's run.
