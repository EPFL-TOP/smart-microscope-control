# 0009 — A light CI on pull requests, the full matrix every night

- **Status**: **accepted** 2026-11-05 by the owner (in chat)
- **Date**: 2026-11-05
- **Issue**: #354 (implemented by #353)
- **Amends**: [0005](0005-testing-strategy.md) (the CI paragraph and the
  "Linux, macOS and Windows" line), [0008](0008-repository-workflow.md)
  (the *CI gates* bullet and the protection of `main`)

## Context

Measured on the last 150 CI runs (2026-10-29 to 2026-11-05) and on the
settings of `main`:

- **The wait is `pytest`.** 11 to 14 minutes per test job (about 5 700
  tests, run serially) against about 1 minute of install. The five test jobs
  run in parallel, so the wall clock of a run is its slowest job: median 16
  minutes, p90 20. The repository is public, so runner minutes are free; the
  cost is latency, and dropping a job saves latency only when it was the
  slowest (windows py3.11 and ubuntu py3.10 often are, by 2 to 3 minutes).
- **`main` requires branches to be up to date** (`required_status_checks.strict`)
  and receives about 7 merges a day. Each merge invalidates the CI of every
  open PR and forces a merge of `main` and a full re-run: #350 waited 15
  more minutes after it was green. The rule also caught a real cross-PR
  conflict on 2026-11-05 (the `plate-walls` plugin of #328 against the cover
  test of #324).
- **What fails, and where.** 14 red runs of 150 in the week, 12 of them on
  the macOS job (8 on pushes to `main`), the last on 2026-11-02 (timing
  tests, fixed by #311).
- **The required checks** are `lint & types` and four of the five test jobs;
  `tests (windows-latest, py3.14)` runs but is not required, although it is
  the Python the microscope PCs run. ADR-0005 says "a 4-job matrix" (it is
  five) and `CLAUDE.md` says "CI must be green on Linux, macOS and Windows".
- Every develop session already runs `ruff`, `mypy` and the whole suite
  locally before the push (`CLAUDE.md`, step 5), and the owner develops on
  macOS.

The owner's decision of 2026-11-05: the turnover of PRs that follow one
another matters more than a full matrix before each merge; the full suite
catches the rest overnight. On the owner's go, `strict` was already set to
`false` that day (undo: `gh api -X PATCH
repos/EPFL-TOP/smart-microscope-control/branches/main/protection/required_status_checks
-F strict=true`); this record is the decision behind it.

## Decision

Two sets of CI jobs, with the job names of today so that the required checks
keep matching:

| When | What runs |
|---|---|
| a pull request; a push to `main` | the **light set**: `lint & types`, `tests (ubuntu-latest, py3.13)`, `tests (windows-latest, py3.14)` |
| every night on `main` (`schedule`); `workflow_dispatch` | the **full matrix**: the five test jobs of today, plus `lint & types` |
| a pull request carrying the label `ci: full` | the full matrix, on that PR |

- **The light set** is the baseline (ubuntu py3.13) and what the microscope
  PCs run (windows py3.14). It is what `main` requires before a merge: the
  required checks become exactly these three.
- **`ci: full`** is for a PR whose failure class is platform-specific:
  paths, encodings, subprocesses, file locking, a hardware bridge, a change
  of the minimum Python. `/plan` names the label in the plan when the risk
  calls for it, `/review` may add it, the owner may add it at the merge.
- **No "up to date" rule.** `main` no longer requires a branch to be up to
  date. The class it caught (two green PRs that break together) is caught
  minutes after the merge by the light set on the push to `main`.
- **A red `main` stops the merges.** When the light set fails on a push to
  `main`, no other PR is merged until the owner has chosen to revert or to
  fix forward, and `/point` lists it first.
- **The nightly reports by itself.** When the full matrix fails, the run
  opens or updates **one** issue (`CI nightly red`, with the run's URL and the
  failing tests); the next green run closes it. `scripts/dev/point.py` lists
  an open one, so nobody has to look at the Actions tab. A red nightly is
  fixed before the next merge of the code it names, like a red `main`.
- **A release** is tagged only from a commit whose full matrix is green (the
  nightly of that commit, or a manual `workflow_dispatch`).
- **Amendments.** In ADR-0005, layer 2 reads "run on every PR on Linux and
  Windows, and every night on Linux, macOS and Windows", and the tooling
  line reads "CI runs lint → tests: the light set per PR, the full matrix
  nightly". In ADR-0008, the *CI gates* bullet reads the same. Everything
  else in both records stands, including `SMC_REQUIRE_MM=1` (a missing
  install fails in every job).

Not decided here, measured by #353 and adopted only on the evidence:
`pytest-xdist` on the light jobs (with the wall-clock tests behind a marker
that runs serially), and a `mypy --python-version 3.10` step in `lint` to
catch part of the py3.10 class cheaply.

## Alternatives considered

- **The status quo** (full matrix on every PR, "up to date" required): the
  guarantee is complete and the wait is 16 minutes, repeated for every open
  PR at every merge. Lost on latency, which is the cost that matters on a
  public repository with free runners.
- **A merge queue** instead of "up to date": it keeps the guarantee without
  manual merges of `main`, but every queued merge still runs a CI, so it does
  not shorten the wait; the owner merges by hand and the queue would
  serialise them.
- **The full matrix on release only:** no daily signal, and a regression
  would be bisected over weeks of merges instead of one day's seven.
- **A path filter** (the full matrix when the diff touches certain files):
  it would drift from the failure classes. An encoding bug sits in any
  module, and a list of risky paths nobody owns is a rule nobody applies. The
  label is a judgement made where the risk is known (`/plan`, `/review`).
- **A different light set** (add macOS, or the oldest Python): each added
  job is a possible slowest job. macOS failed most often in the week, but
  8 of the 12 were on pushes to `main`, where the light set does not run it
  either; the nightly reports it. The owner develops on macOS and the whole
  suite runs there before every push.
- **Keep the five required checks and add the nightly:** the nightly would be
  an extra, and nothing would get shorter.

## Consequences

- A PR waits for three jobs, not six, and no longer re-runs when another PR
  merges.
- **What is lost:** a failure that only py3.10, windows py3.11 or macOS shows
  is found the next morning, after the merge, instead of before it. The
  window is one day of merges, and the nightly's issue names the failing
  tests; a fix is a normal PR.
- A PR can be green against an older `main` and break after the merge. The
  push to `main` catches it, so a red `main` becomes a state the process has
  a rule for (above) rather than an exception.
- `tests (windows-latest, py3.14)` becomes a required check.
- Implementation, in #353 and not in this record: `ci.yml` selects the matrix
  by event and label; the nightly has its own `concurrency` group (the group
  of today is the ref, so a push to `main` would cancel the nightly);
  scheduled workflows are disabled by GitHub after 60 days without activity
  on the repository, a risk the nightly's issue cannot report; `CLAUDE.md`
  changes from "CI must be green on Linux, macOS and Windows" to "before a
  release and every night"; `/plan`, `/review` and `/point` learn the label
  and the nightly's issue.
- **The owner, at the merge of the workflow PR,** changes the required checks
  of `main` (Settings, Branches) to the three of the light set, at the same
  time; before that, a PR waits for checks that no longer run.
- **Once accepted,** the status lines of 0005 and 0008 are edited to
  `accepted (amended in part by 0009)`.
