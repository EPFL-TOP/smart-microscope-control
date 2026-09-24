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
1. Read `CLAUDE.md`, then `gh issue view $ARGUMENTS --comments`. Find the
   latest comment starting with `## Plan`. No plan, or no `status: ready`
   label → stop and comment "No plan / not ready — needs `/plan` first."
2. Read the design-doc sections and ADRs the plan names, the failure modes
   it lists (`docs/design/failure-modes.md`), and the code you will touch.
3. `Depends on`: if a dependency is not merged into `main`, stop and say so
   on the issue.
4. Comment `Starting — <scope in three lines>`; set `status: in-progress`
   (remove `status: ready`).

## Isolate — every execution session works in its own worktree

5. Call the **`EnterWorktree`** tool with `name: "issue-<N>-<slug>"`, then,
   inside the worktree:

   ```sh
   git fetch origin main                    # SSH may fail in agent shells: see below
   git merge --ff-only origin/main          # start from the latest main
   git branch -m feat/<N>-<slug>            # the project's branch naming
   python scripts/dev/worktree.py setup     # this worktree's own .venv + editable install
   ```

   - If `EnterWorktree` fails, create the worktree yourself and enter it:
     `git worktree add .claude/worktrees/issue-<N>-<slug> -b feat/<N>-<slug> origin/main`,
     then `EnterWorktree` with `path` set to that directory (skip `git branch -m`).
   - If the fast-forward fails, the worktree was not created from `main`:
     leave it (`ExitWorktree`, `remove`) and use the fallback above.
   - If `EnterWorktree` says you are already in a worktree, check
     `git worktree list` and continue only if it is this issue's.
   - **SSH**: an agent shell may have no SSH key loaded, so `fetch` and
     `push` against the `git@github.com:` remote fail while `gh` works. Run
     the git command as
     `git -c url."https://github.com/".insteadOf="git@github.com:" <command>`.
     Never change the remote.

   Never use the main checkout's `.venv` from a worktree: it is an editable
   install of the main checkout's code. From now on call the worktree's
   tools (`.venv/bin/ruff`, `.venv/bin/mypy`, `.venv/bin/pytest`; Windows
   `.venv\Scripts\…`) and commit with `PATH=".venv/bin:$PATH" git commit …`.

   **Shell in the worktree.** Its sandbox refuses compound commands:
   heredocs, `$(...)`, pipes into an interpreter, long `--jq` filters. Keep
   each Bash call to one plain command. Write throwaway scripts and files
   with the Write tool into your scratchpad, or into a `mktemp -d` directory
   created by a command on its own, and run them with `.venv/bin/python
   <file>`. Read `gh` output with `--json <fields>` and no filter.

## Implement

6. Follow the plan's steps in order; one commit per step, conventional
   messages.
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
    .venv/bin/mypy && .venv/bin/pytest` — all green in the worktree.
11. Run **`/adversarial-review $ARGUMENTS`**: cheap reviewer agents attack
    the change for its risk level, a verifier reproduces or refutes each
    finding, and you fix what is confirmed and in scope. Re-run step 10.

## PR and report

12. Push (`git push -u origin feat/<N>-<slug>`; the SSH workaround of step 5
    if needed), open the PR with the template (`gh pr create --fill
    --body-file …`, `Closes #N`, conventional title), and watch CI:
    `gh pr checks --watch`. Fix failures; never disable a check.
13. Comment on the issue, exactly in this format, and set `status: in-review`:

    ```markdown
    ## Report (develop session, YYYY-MM-DD)

    **PR**: #N · branch `feat/…` · head `<sha>` · worktree `.claude/worktrees/issue-…`
    **Done**: 3–6 lines — what exists now that did not before
    **Verified**: ruff · format · mypy · pytest (N passed, M skipped) · CI <green|link>
    **Adversarial review**: <the summary /adversarial-review produced>
    **Beyond the plan**: none | defensive fixes that changed no interface
    **Not verified**: what a human or hardware must still check
    **Deviations from plan**: none | what and why
    **Friction**: none | what in the skill, the plan or the tooling slowed you down
    **Follow-ups**: issues to open (not opened by you unless trivial)
    ```

14. Call **`ExitWorktree`** with `action: "keep"`. This leaves the worktree
    and branch on disk for the fix rounds a review may send you back to,
    but releases this session's own lock on it, so a dead session does not
    leave `worktree.py clean` unable to remove it later. Then stop: do not
    merge, do not start another issue.

## Addressing a review

When the latest `## Review` on the PR says **Verdict: changes requested**:

1. Enter the issue's worktree: `EnterWorktree` with `path` set to
   `.claude/worktrees/issue-<N>-<slug>` (see `git worktree list`). If it is
   gone, recreate it from the PR's branch —
   `git worktree add .claude/worktrees/issue-<N>-<slug> feat/<N>-<slug>` —
   then `python scripts/dev/worktree.py setup`.
2. Fix exactly the **Blocking** items, in order; optional items only if the
   review lists them as optional and they are small.
3. If `main` moved: `git fetch origin main`, then `git merge origin/main`
   (never rebase a pushed branch).
4. Verify (step 10); if the fixes change behaviour, run
   `/adversarial-review` on them (step 11). Push, watch CI.
5. Comment on the PR `## Review addressed (develop session, YYYY-MM-DD)`,
   one line per blocking item: what changed and where. Set the issue to
   `status: in-review`. Call `ExitWorktree` with `action: "keep"`. Stop.

## Never

- Commit to `main`; force-push; merge; edit ADRs; run `pytest -m hardware`;
  move a real stage; change a fixed interface; silence a failing check; use
  the main checkout's `.venv` from a worktree.
