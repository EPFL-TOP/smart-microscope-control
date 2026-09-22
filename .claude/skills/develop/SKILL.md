---
name: develop
description: Execution-session skill — implement one planned GitHub issue in a clean session, strictly within its plan, test-first against the demo devices, verify, open the PR and report back on the issue. Never redesigns, never touches ADRs, never runs hardware. Argument: the issue number.
---

# /develop <issue-number>

You are an **execution session**: fresh context, one issue, one PR. The
design was done elsewhere and lives in the issue's plan and the design doc.
Your job is to implement it faithfully and prove it.

## Before touching code

1. Read `CLAUDE.md`, then `gh issue view $ARGUMENTS --comments`. Find the
   latest comment starting with `## Plan`. No plan, or the label
   `status: ready` missing → stop and comment: "No plan / not ready — needs
   `/plan` first." Do not invent one.
2. Read every design-doc section the plan references, and the ADRs it
   names. Read the existing code you will touch.
3. Check `Depends on`: if a dependency is not merged into `main`, stop and
   say so on the issue.
4. Comment `Starting — <short restatement of scope in 3 lines>` and set
   `status: in-progress` (remove `status: ready`). If another develop
   session is active in this checkout, work in a worktree:
   `git worktree add ../smc-<issue> -b feat/<issue>-<slug> main`.

## Implement

5. `git switch -c feat/<issue>-<slug> main` (or the worktree). Follow the
   plan's steps in order; one commit per step, conventional messages.
6. Test-first where behaviour is checkable: write the named tests, watch
   them fail, make them pass. Simulator tests use the fixtures from
   `smc.testing.fixtures`; pure logic gets unit tests (+ `hypothesis` for
   invariants); vendor quirks get `FakeCore` tests.
7. Respect the architecture rules in `CLAUDE.md`. Interfaces named in the
   design doc are fixed: do not rename, do not "improve" a signature. If the
   plan cannot be implemented as written (a wrong assumption about MMCore,
   a missing hook), **do not redesign**: comment `## Plan deviation` on the
   issue with the problem and the smallest options, then either stop or
   implement the smallest local fix the comment describes — say which.
8. Stay in scope. Something adjacent looks wrong? Note it in the report as
   a follow-up; do not fix it here.

## Verify

9. `ruff check . && ruff format --check . && mypy && pytest` — all green,
   locally. Run `smc doctor` if `smc.hardware` changed. Check the diff for
   `print`, hard-coded device labels, units without suffixes, `read_text()`
   without `encoding`.
10. Push (`git push -u origin <branch>`), open the PR with the template
    (`gh pr create --fill --body-file …`, `Closes #N`, conventional title),
    then watch CI: `gh pr checks --watch`. Fix CI failures; never disable a
    check to pass.

## Report

11. Comment on the issue, exactly in this format, and set
    `status: in-review`:

    ```markdown
    ## Report (develop session, YYYY-MM-DD)

    **PR**: #N · branch `feat/…` · head `<sha>`
    **Done**: 3–6 lines — what exists now that did not before
    **Verified**: ruff · format · mypy · pytest (N passed, M skipped) · CI <green|link>
    **Not verified**: what a human or hardware must still check
    **Deviations from plan**: none | what and why
    **Follow-ups**: issues to open (not opened by you unless trivial)
    ```

12. Stop. Do not merge. Do not start the next issue in this session.

## Never

- Commit to `main`; force-push; merge; edit ADRs; run `pytest -m hardware`;
  move a real stage; widen scope; silence a failing check.
