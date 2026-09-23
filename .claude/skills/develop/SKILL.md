---
name: develop
description: Execution-session skill — implement one planned GitHub issue in a clean session and its own git worktree, strictly within the issue's plan, test-first against the demo devices, verify, open the PR and report back on the issue. Never redesigns, never touches ADRs, never runs hardware. Argument: the issue number.
---

# /develop <issue-number>

You are an **execution session**: fresh context, one issue, one worktree,
one PR. The design was done elsewhere and lives in the issue's plan and the
design doc. Your job is to implement it faithfully and prove it.

## Before touching code

0. If the issue already has an open PR whose latest `## Review` comment
   says **Verdict: changes requested**, skip to *Addressing a review* at
   the end of this skill.
1. Read `CLAUDE.md`, then `gh issue view $ARGUMENTS --comments`. Find the
   latest comment starting with `## Plan`. No plan, or the label
   `status: ready` missing → stop and comment: "No plan / not ready — needs
   `/plan` first." Do not invent one.
2. Read every design-doc section the plan references, and the ADRs it
   names. Read the existing code you will touch.
3. Check `Depends on`: if a dependency is not merged into `main`, stop and
   say so on the issue.
4. Comment `Starting — <short restatement of scope in 3 lines>` and set
   `status: in-progress` (remove `status: ready`).

## Isolate — every execution session works in its own worktree

5. Call the **`EnterWorktree`** tool with `name: "issue-<N>-<slug>"`. It
   creates `.claude/worktrees/issue-<N>-<slug>` on a fresh branch from
   `origin/main` and moves this session into it. Then, inside the worktree:

   ```sh
   git fetch origin main                    # SSH may fail in agent shells: see below
   git merge --ff-only origin/main          # start from the latest main
   git branch -m feat/<N>-<slug>            # the project's branch naming
   python scripts/dev/worktree.py setup     # this worktree's own .venv + editable install
   ```

   - If `EnterWorktree` fails, create the worktree yourself and enter it:
     `git worktree add .claude/worktrees/issue-<N>-<slug> -b feat/<N>-<slug> origin/main`,
     then `EnterWorktree` with `path` set to that directory (skip the
     `git branch -m` above).
   - If the fast-forward fails, the worktree was not created from `main`:
     leave it (`ExitWorktree`, `remove`) and use the fallback above.
   - If `EnterWorktree` says you are already in a worktree, check
     `git worktree list` and continue only if it is this issue's.
   - **SSH**: an agent shell may have no SSH key loaded, so `fetch` and
     `push` against the `git@github.com:` remote fail while `gh` works. Run
     the git command as
     `git -c url."https://github.com/".insteadOf="git@github.com:" <command>`
     instead — `gh` is the HTTPS credential helper. Never change the remote.

   The main checkout's `.venv` must not be used from a worktree: it is an
   editable install of the *main checkout's* `src/`, so its tests would run
   the wrong code. `setup` builds this worktree's venv from the main
   checkout's interpreter and proves `smc` imports from the worktree. From
   now on call the worktree's tools explicitly (`.venv/bin/ruff`,
   `.venv/bin/mypy`, `.venv/bin/pytest`; Windows `.venv\Scripts\…`) and
   commit with `PATH=".venv/bin:$PATH" git commit …` so the shared
   pre-commit hook finds them.

## Implement

6. Follow the plan's steps in order; one commit per step, conventional
   messages.
7. Test-first where behaviour is checkable: write the named tests, watch
   them fail, make them pass. Simulator tests use the fixtures from
   `smc.testing.fixtures` (or `tests/conftest.py` until #9 lands); pure
   logic gets unit tests (+ `hypothesis` for invariants); vendor quirks get
   `FakeCore` tests.
8. Respect the architecture rules in `CLAUDE.md`. Interfaces named in the
   design doc are fixed: do not rename, do not "improve" a signature. If the
   plan cannot be implemented as written (a wrong assumption about MMCore,
   a missing hook), **do not redesign**: comment `## Plan deviation` on the
   issue with the problem and the smallest options, then either stop or
   implement the smallest local fix the comment describes — say which.
9. Stay in scope. Something adjacent looks wrong? Note it in the report as
   a follow-up; do not fix it here.

## Verify

10. `.venv/bin/ruff check . && .venv/bin/ruff format --check . &&
    .venv/bin/mypy && .venv/bin/pytest` — all green, in the worktree. Run
    `.venv/bin/smc doctor` if `smc.hardware` changed. Check the diff for
    `print`, hard-coded device labels, units without suffixes,
    `read_text()` without `encoding`.
11. Push (`git push -u origin feat/<N>-<slug>`; the SSH workaround of step
    5 if needed), open the PR with the
    template (`gh pr create --fill --body-file …`, `Closes #N`, conventional
    title), then watch CI: `gh pr checks --watch`. Fix CI failures; never
    disable a check to pass.

## Report

12. Comment on the issue, exactly in this format, and set
    `status: in-review`:

    ```markdown
    ## Report (develop session, YYYY-MM-DD)

    **PR**: #N · branch `feat/…` · head `<sha>` · worktree `.claude/worktrees/issue-…`
    **Done**: 3–6 lines — what exists now that did not before
    **Verified**: ruff · format · mypy · pytest (N passed, M skipped) · CI <green|link>
    **Not verified**: what a human or hardware must still check
    **Deviations from plan**: none | what and why
    **Follow-ups**: issues to open (not opened by you unless trivial)
    ```

13. Leave the worktree in place until the PR is merged — a review may send
    you back to it. Merged worktrees are removed from the main checkout with
    `python scripts/dev/worktree.py clean`.
14. Stop. Do not merge. Do not start the next issue in this session.

## Addressing a review

When the design session's `## Review` on the PR says **Verdict: changes
requested**:

1. Enter the issue's existing worktree: `EnterWorktree` with `path` set to
   `.claude/worktrees/issue-<N>-<slug>` (listed by `git worktree list`). If
   it is gone, recreate it from the PR's branch:
   `git worktree add .claude/worktrees/issue-<N>-<slug> feat/<N>-<slug>`,
   then `python scripts/dev/worktree.py setup`.
2. Fix exactly the **Blocking** items, in order; nits only if trivial. No
   other change.
3. If `main` moved, bring it in: `git fetch origin main` then
   `git merge origin/main` (never rebase a pushed branch).
4. Verify as in step 10, push, watch CI.
5. Comment on the PR `## Review addressed (develop session, YYYY-MM-DD)`
   with one line per blocking item: what changed, where. Set the issue to
   `status: in-review`. Stop.

## Never

- Commit to `main`; force-push; merge; edit ADRs; run `pytest -m hardware`;
  move a real stage; widen scope; silence a failing check; use the main
  checkout's `.venv` from a worktree.
