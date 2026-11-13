---
name: develop
description: Execution-session skill — implement one planned GitHub issue in a clean session and its own git worktree, within the issue's plan, test-first against the demo devices; attack the change with the adversarial review before the PR; verify, open the PR and report back on the issue. Never changes an interface the design fixed, never touches ADRs, never runs hardware. Argument: the issue number.
---

# /develop <issue-number>

You are an **execution session**: fresh context, one issue, one worktree,
one PR. The design lives in the issue's plan and the design doc. Implement
it faithfully, attack it before anyone else does, and prove it. The plan's
**Suggested model** is what the owner should pick for this session; tell
them if it is not the one you run on.

## Before touching code

0. If the issue already has an open PR whose latest `## Review` says
   **Verdict: changes requested**, go to *Addressing a review* at the end.
1. Read `CLAUDE.md`, then `gh issue view $ARGUMENTS --comments` (if it
   prints nothing, `gh issue view $ARGUMENTS --json body,comments`). Find the
   latest comment starting with `## Plan`. No plan, or no `status: ready`
   label → stop and comment "No plan / not ready — needs `/plan` first."
   One exception: `status: blocked` or `status: needs-decision` whose
   every **Depends on** is merged (a plan that waited for its own design
   PR; merging it was the decision). Check each with `gh pr view`/`gh
   issue view`, set `status: ready` and say so in the `Starting` comment
   (#144 and #12 met a stale `blocked`; #204 and #205 a stale
   `needs-decision`).
2. Read the design-doc sections and ADRs the plan names, the failure modes
   it lists (`docs/design/failure-modes.md`), and the code you will touch.
3. `Depends on`: if a dependency is not merged into `main`, stop and say so
   on the issue. A design PR the plan opened is named on its **Ready for
   /develop** line (`blocked by #<pr>`): it counts as a dependency too.
4. Comment `Starting — <scope in three lines>`; set `status: in-progress`
   (remove `status: ready`).

## Isolate — every execution session works in its own worktree

5. Call the **`EnterWorktree`** tool with `name: "issue-<N>-<slug>"`, then,
   inside the worktree:

   ```sh
   git fetch origin main                    # SSH may fail in agent shells: see below
   git merge --ff-only origin/main          # start from the latest main
   git branch -m <type>/<N>-<slug>          # <type>: feat, fix, docs, chore (CLAUDE.md)
   python scripts/dev/worktree.py setup     # this worktree's own .venv + editable install
   ```

   - If `EnterWorktree` fails, create the worktree yourself and enter it:
     `git worktree add .claude/worktrees/issue-<N>-<slug> -b <type>/<N>-<slug> origin/main`,
     then `EnterWorktree` with `path` set to that directory (skip `git branch -m`).
   - If the fast-forward fails, the worktree was not created from `main`:
     leave it (`ExitWorktree`, `remove`) and use the fallback above.
   - If `EnterWorktree` says you are already in a worktree, check
     `git worktree list` and continue only if it is this issue's.
   - **SSH**: an agent shell may have no SSH key loaded, so `fetch` and
     `push` against the `git@github.com:` remote fail while `gh` works.
     Write the https URL out: `git fetch
     https://github.com/<owner>/<repo>.git main` then `git merge --ff-only
     FETCH_HEAD`, and `git push https://github.com/<owner>/<repo>.git
     HEAD:<branch>`. A worktree session refuses the older `git -c
     url…insteadOf` form (eight Reports, #408 to #462). Never change the
     remote.

   Never use the main checkout's `.venv` from a worktree: it is an editable
   install of the main checkout's code. From now on call the worktree's
   tools (`.venv/bin/ruff`, `.venv/bin/mypy`, `.venv/bin/pytest`; Windows
   `.venv\Scripts\…`) and commit with a plain `git commit …`: the hooks find
   their own ruff, and a worktree session refuses an inline `PATH=…` before
   `git commit` (the same eight Reports).

   **Shell in the worktree.** Its sandbox refuses compound commands:
   heredocs, `$(...)`, pipes into an interpreter, long `--jq` filters. Keep
   each Bash call to one plain command. Write throwaway scripts and files
   with the Write tool into your scratchpad, or into a `mktemp -d` directory
   created by a command on its own, and run them with `.venv/bin/python
   <file>`. Read `gh` output with `--json <fields>` and no filter.
   Three recipes, each met by several sessions (#224, #248, #260, #263):
   - **Save a GitHub comment to a file**: run `gh issue view N --comments`
     (no redirection, no `--jq`), then Write what you read into the file.
   - **Append to a test file**: Read it, then Edit with the last lines as the
     anchor, or Write the whole file; never `cat >>` or a heredoc.
   - **A second Edit of the same lines**: the PostToolUse formatter rewraps
     a file after every Edit or Write, so Read the region again before you
     edit it a second time (#213, #224).
   - **`Literal["a", "b"]` in an annotation**: write the `from typing import
     Literal` line **before** the annotation, or the formatter (ruff UP037)
     reads the strings as forward references and strips their quotes
     (#331, #341); a module-level alias also works.

## Implement

6. Follow the plan's steps in order; one commit per step, conventional
   messages. Steps that edit the same file may share a commit (the
   worktree's shell has no `git add -p`, #158); say so in the Report.
7. Test-first where behaviour is checkable. Simulator tests use the demo
   fixtures; pure logic gets unit tests (+ `hypothesis` for invariants);
   vendor quirks get `FakeCore` tests. The default suite never loads a
   vendor adapter (FM-40).
8. Interfaces named in the design doc are fixed: do not rename them, do not
   change a signature.
   - **Beyond the plan, allowed**: a defensive fix that changes no
     interface — handling a failure mode the plan missed, a better error
     message, isolating a test. Make it, test it, and list it in the Report
     under "Beyond the plan".
   - **Not allowed**: an interface change, a new behaviour visible to users,
     anything an ADR decides. Comment `## Plan deviation` on the issue with
     the problem and the smallest options, then stop or apply the smallest
     local fix that comment describes — say which.
9. Stay in scope otherwise: note adjacent problems as follow-ups.

## Verify and attack

10. `.venv/bin/ruff check . && .venv/bin/ruff format --check . &&
    .venv/bin/mypy && .venv/bin/pytest` — all green in the worktree. If the
    change touches the CLI or its tests, run them once more as CI does,
    `GITHUB_ACTIONS=1 .venv/bin/pytest tests/test_cli.py` (FM-72: typer then
    forces colour; #84's first push failed every CI job on this).
    **Can each new test fail?** For a test that guards a claim, run one mutant
    of the line it guards: `.venv/bin/python scripts/dev/mutant.py <file>
    <old> <new> <test-id>`, from the worktree root. It copies the working tree
    (uncommitted edits included) to a temporary directory and never writes to
    the worktree. Exit 0 `KILLED`: the test can fail. Exit 1 `SURVIVED`: it
    proves nothing about that line, so strengthen it. Exit 2: no verdict (the
    pattern was not found exactly once, or the test does not pass unmutated).
    A mutant that needs several edits (a hoisting, a rename) is still one run:
    `--also <file> <old> <new>`, repeatable, adds a replacement to the same
    copy, each pattern matched once in the file as it is on disk. A pattern of
    several lines goes through `--old-file <path> --new-file <path>` (and
    `--also-file <file> <old-path> <new-path>`) instead of the shell's quoting,
    which a formatter's reflow breaks; a dash-leading pattern of `--also` goes
    through `--also-file`.
    Never restore a mutant with `git checkout`: it discards uncommitted work
    (#93).
11. **An `area: ui` issue only**: look at the page you changed, with
    **`/ui-check`** (`scripts/dev/ui_snap.py`, from the worktree's `.venv`,
    `--out` your scratchpad). Read the viewport PNGs of the states the change
    touches and `summary.txt`; a server error line or a state not reached is
    a finding to fix or to put in the Report. The tests build the page and
    never open a session, so a handler that fails only in one passes them all
    (FM-161). It never replaces a test: it is for looking.
12. Run **`/adversarial-review $ARGUMENTS`**: cheap reviewer agents attack
    the change for its risk level, a verifier reproduces or refutes each
    finding, and you fix what is confirmed and in scope. Re-run step 10.

## PR and report

13. Push (`git push -u origin <type>/<N>-<slug>`; the SSH workaround of step 5
    if needed), open the PR with the template (`gh pr create --fill
    --body-file …`, `Closes #N`, conventional title), and watch CI:
    first `gh pr view --json mergeable`, then wait for the run **in the
    background**, never with `gh pr checks --watch` or `gh run watch` (each
    refresh reprints the whole job tree, about 60 KB for one wait, #353):
    `gh pr checks N` once to get the run id, then one Bash call with
    `run_in_background`: `python scripts/dev/wait_for_run.py <id>` (it polls
    every 60 s and prints one line with the conclusion; the sandbox refuses
    an `until` loop with `$(...)`, #342, #380), and read `gh pr checks N`
    when it returns. **The
    label**: when the plan's *Definition of done* has the line `the PR carries
    the label ci: full`, open the PR with `--label "ci: full"` (the CI reads
    the label when a run starts, so the first run already has the full
    matrix). Without the label a PR runs the light set only (`lint & types`,
    `tests (ubuntu-latest, py3.13)`, `tests (windows-latest, py3.14)`,
    ADR-0009): three checks are green, not five. A PR that conflicts with
    `main` gets **no CI** (`--watch` exits at once with `no checks
    reported`): merge `main` into the branch, push, and watch again (#264,
    #291, #297 met this when a PR on the same files merged first). A PR that
    is only *behind* `main` needs no merge: `main` no longer requires an up
    to date branch (ADR-0009), so do not merge `main` for that alone. Fix
    failures; never disable a check.
14. Comment on the issue, exactly in this format, and set `status: in-review`:

    ```markdown
    ## Report (develop session, YYYY-MM-DD)

    **PR**: #N · branch `feat/…` · head `<sha>` · worktree `.claude/worktrees/issue-…`
    **Done**: 3–6 lines — what exists now that did not before
    **Verified**: ruff · format · mypy · pytest (N passed, M skipped) · CI <green|link> · `area: ui`: ui-check (exit code, states reached, where the PNGs are)
    **Adversarial review**: <the summary /adversarial-review produced>
    **Beyond the plan**: none | defensive fixes that changed no interface
    **Not verified**: what a human or hardware must still check
    **Deviations from plan**: none | what and why
    **Friction**: none | what in the skill, the plan or the tooling slowed you down
    **Follow-ups**: issues to open (not opened by you unless trivial)
    ```

15. Call **`ExitWorktree`** with `action: "keep"`. This leaves the worktree
    and branch on disk for the fix rounds a review may send you back to,
    but releases this session's own lock on it, so a dead session does not
    leave `worktree.py clean` unable to remove it later. Then stop: do not
    merge, do not start another issue.

## Addressing a review

When the latest `## Review` on the PR says **Verdict: changes requested**:

1. Enter the issue's worktree: `EnterWorktree` with `path` set to
   `.claude/worktrees/issue-<N>-<slug>` (see `git worktree list`). If it is
   gone, recreate it from the PR's branch —
   `git worktree add .claude/worktrees/issue-<N>-<slug> <type>/<N>-<slug>` —
   then `python scripts/dev/worktree.py setup`.
2. Fix exactly the **Blocking** items, in order; optional items only if the
   review lists them as optional and they are small.
3. If `main` moved: `git fetch origin main`, then `git merge origin/main`
   (never rebase a pushed branch).
4. Verify (step 10); if the fixes change behaviour, run
   `/adversarial-review` on them (step 12), and `/ui-check` (step 11) when the
   fixes touch the page. Push, watch CI.
5. Comment on the PR `## Review addressed (develop session, YYYY-MM-DD)`,
   one line per blocking item: what changed and where, then a
   `**Follow-ups**:` line for what the fix round found and deferred, as in
   the Report (`point.py` lists it when the PR merges). Mark the PR ready
   again (`gh pr ready N`: `/review` turned it into a draft). Set the issue to
   `status: in-review`. Call `ExitWorktree` with `action: "keep"`. Stop.

## In a lane

When `/lane` runs this skill (its brief says so), you are a **subagent** with
`isolation: "worktree"`: you start in `.claude/worktrees/agent-<id>`, on a
branch `worktree-agent-<id>`, at the main checkout's *local* `HEAD` (it can
lag `origin/main`), with no `.venv`, and `EnterWorktree` and `ExitWorktree`
are refused. You cannot ask the owner anything: where this skill would ask,
stop and say what you need.

- **Isolate** (step 5): skip `EnterWorktree`. Run `git branch -m
  <type>/<N>-<slug>`, then step 5's fetch and `git merge --ff-only
  origin/main`, then `python scripts/dev/worktree.py setup`. If the
  fast-forward fails, `git status --porcelain` and `git log origin/main..HEAD`
  must both print nothing (a fresh worktree): then `git reset --hard
  origin/main`; otherwise stop and say so. A worktree session may refuse the
  `git -c url…insteadOf` form of the SSH workaround: write the https URL out
  (`git fetch https://github.com/<owner>/<repo>.git main`, then `git merge
  --ff-only FETCH_HEAD`).
- **CI** (step 13): do not rely on a background wait, a subagent is not woken
  by one. Open the PR, read `gh pr checks N` once, and say in the Report that
  the checks were running; the lane waits for them itself and stops for the
  owner when they are red.
- **Step 15**: no `ExitWorktree`; the harness keeps a worktree that holds
  commits.
- **Addressing a review**: the first develop's worktree holds the PR's
  branch, and git refuses to check a branch out twice. Find that worktree
  (`git worktree list --porcelain`), remove it with `git worktree remove
  <path>` (never `--force`: git refuses a dirty or locked one, and then you
  stop and say so), then `git switch <branch>`, `git pull --ff-only` (the
  SSH workaround above), `python scripts/dev/worktree.py setup`, and go on
  with *Addressing a review* from its step 2. Its last step has no
  `ExitWorktree` either.

## Never

- Commit to `main`; force-push; merge; edit ADRs; run `pytest -m hardware`;
  move a real stage; change a fixed interface; silence a failing check; use
  the main checkout's `.venv` from a worktree.
