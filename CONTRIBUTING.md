# Contributing

Thanks for helping. This page is the short version of
[ADR-0008](docs/adr/0008-repository-workflow.md); `CLAUDE.md` holds the
full set of engineering rules, and they apply to people too.

## Setup

```sh
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
mmcore install --test-adapters   # simulated microscope
pre-commit install
smc doctor && pytest
```

## The loop

1. **Issue first.** Use a template. Write acceptance criteria; add a short
   plan before you start coding.
2. **Branch**: `feat/<issue>-<slug>` (or `fix/`, `docs/`, `chore/`, `adr/`).
3. **Code + tests.** New behaviour is proven on the demo devices; pure
   algorithms get unit tests; hardware-only behaviour gets a `FakeCore`
   test. Follow the layer rules (hardware only via `smc.hardware`; plugins
   depend on capabilities; units in names).
4. **Verify**: `ruff check . && ruff format --check . && mypy && pytest`.
5. **Pull request** with the template. Title in Conventional Commits form
   (`feat(core): …`). One reviewer. Squash merge by the maintainer.

## Testing

`pytest` runs the unit tests, the tests on Micro-Manager's demo devices,
and the **contract suite** (`tests/contracts/`). Without the demo adapters
the demo tests skip; with `SMC_REQUIRE_MM=1` (CI) they fail instead.

Every contract runs once per backend:

- `[fake]` runs on `FakeCore` (`smc.testing`), an in-memory core that can
  model a stuck device, a failing one, or a vendor quirk;
- `[demo]` runs on the demo devices;
- `[hardware]` runs on a real stand. It is deselected unless you pass
  `-m hardware`, and skipped unless you also pass `--profile`.

The fixtures (`fake_core`, `fake_microscope`, `demo_microscope`,
`hardware_microscope`, …) come from the pytest plugin
`smc.testing.fixtures`. A plugin's own repository can load it with
`pytest_plugins = ["smc.testing.fixtures"]` in its top-level `conftest.py`.

### Running the contracts at a stand

Only at the microscope, with the vendor software closed and
`smc doctor --config <cfg>` passing (see *Hardware sessions*):

```sh
pytest -m hardware --profile nikon-ti2 tests/contracts
```

The same command works in PowerShell, without quotes. `--profile` takes a
profile name or a path. `--profile demo` rehearses the run on the
simulator.

**What it moves.** Before each test, the suite records where the stand
is. It then replaces the profile's `[safety]` with limits around that
start: XY ± 100 µm, Z ± 3 µm, and a 50 µm jog limit.

- The XY stage moves at most 60 µm from the start.
- Z moves at most 2 µm, and only below the start (away from the sample on
  an inverted stand).
- The shutter opens and closes, and the camera snaps, so the light comes
  on.
- The exposure and auto-shutter change.

Every target outside those limits is one the layer must refuse, so none
is sent. After each test the stand is put back where it started, with the
light off first if it started off. If a motion is left, or the test
halted the stand, the suite stops the stages instead of moving them back.

The suite opens the stand once per test, about 30 times, so allow a few
minutes. A stand that lacks a capability skips its contracts.

## Supervising and execution sessions

Issues move through three hands. The **supervising session** runs `/point`
to take stock, writes the plans (`/plan`, fixed format with the risk, the
failure modes to handle and the suggested model) and sets `status: ready`.
An **execution session** — a fresh session, human or agent, Sonnet by
default — takes one ready issue, implements its plan, has cheap reviewer
agents attack the change (`/adversarial-review`), opens the PR and posts a
`## Report` (`/develop`). The supervising session reviews lightly
(`/review`); the **owner** merges. Each execution runs in its own
`git worktree` with its own `.venv` (`python scripts/dev/worktree.py setup`;
`clean` removes merged ones). Labels tell you where an issue is: `ready` →
`in-progress` → `in-review`.

## Commit messages

`type(scope): summary` — types `feat fix docs test chore refactor perf ci`;
scopes `core plugins nikon zeiss viventis ui cli docs`. Body explains why.

## Decisions

Anything that constrains future work is an ADR: open an *Architecture
decision* issue, add `docs/adr/NNNN-title.md` with status `proposed` in a
PR, discuss in the review, merge as `accepted`.

## Hardware sessions

Only at the microscope, only with the vendor software closed, only after
`smc doctor --config <cfg>`. File a *Hardware session report* issue the
same day. Never run `pytest -m hardware` from a machine that is not
connected to the stand. How to run the contract suite there, and what it
moves: *Testing*, above.

## Porting from the sibling repositories

Much of the first year is porting `nikon-control/src/nikon_control/scope/`
and `lightsheet-live-tracking-tool/tracking_tools/microscope_interface/`.
Port a module **with its tests**, split the pure algorithm from the hardware
loop, replace device names with capabilities, and keep the docstrings that
explain hardware behaviour — they are the most valuable lines in those
repositories.
