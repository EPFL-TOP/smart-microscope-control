---
name: adversarial-review
description: Execution-session skill — before a PR is opened, have cheap, narrowly scoped reviewer agents attack the change (real-world failure modes, tests, Windows), have a verifier agent reproduce or refute each finding, fix what is confirmed and in scope, and record the rest for the Report. Called by /develop; usable alone. Argument: the issue number.
---

# /adversarial-review <issue-number>

Cheap by construction: the reviewers run on Sonnet and the verifier on Haiku
(their models are set in `.claude/agents/`). Your job is to hand them a
precise task and to judge what comes back. Do not re-review the code
yourself.

1. **Inputs.** `base` = `origin/main`, `head` = `HEAD` of this worktree.
   Pick one scratch directory outside the repository: your session's
   scratchpad if the system prompt names one, otherwise run `mktemp -d` as a
   command on its own and note the path it prints. Save the issue's latest
   `## Plan` comment there as `plan.md` with the Write tool. Read its
   **Risk** and **Failure modes to handle** lines. The worktree's sandbox
   refuses heredocs, `$(...)` and pipes into an interpreter: one plain
   command per Bash call, and files through Write.
2. **Lenses by risk.**

   | Risk | Reviewers |
   |---|---|
   | `docs` | none — skip to step 7 and write "not run (docs)" |
   | `pure` | `reviewer-tests` |
   | `tooling` | `reviewer-tests`, `reviewer-windows` |
   | `reads-hardware` | `reviewer-failure-modes`, `reviewer-tests`, `reviewer-windows` |
   | `moves-hardware` | the same three; the design review will also run `/code-review` |

3. **Spawn** the reviewers in one message, one Agent call each
   (`subagent_type` = the agent's name), each prompt giving: the worktree
   path, `base`, `head`, the plan file, the risk, the failure modes to
   handle, `docs/design/failure-modes.md`, and a scratch directory of its
   own (`<scratch>/<agent-name>/`). Wait for all of them.
4. **Verify.** Merge and de-duplicate the findings. Spawn one `verifier` per
   `blocking` finding, and per nit you intend to act on — in one message, at
   most eight, each with its own scratch directory (`<scratch>/verifier-<n>/`).
   An unverified finding is not a fact.
5. **Act.**
   - CONFIRMED and in scope: fix it now, with a test that fails before the fix.
   - CONFIRMED but needing an interface change: `## Plan deviation` on the issue (see /develop), not a silent redesign.
   - CONFIRMED but out of scope: a follow-up in the Report.
   - REFUTED: one line in the Report.
   - UNVERIFIED and blocking: named as a risk in the Report's "Not verified".
   Then re-run `ruff check . && ruff format --check . && mypy && pytest`.
6. **Learn.** A confirmed finding whose category is missing from
   `docs/design/failure-modes.md` gets a new entry in this PR: an id, what
   happens, how to check.
7. **Record** for the Report, in this shape:

   ```markdown
   **Adversarial review**: failure-modes · tests · windows — 7 findings: 3 confirmed (2 fixed, 1 deferred → follow-up), 3 refuted, 1 unverified
   - fixed: `path:line` — claim
   - deferred: `path:line` — claim (why)
   - unverified: `path:line` — claim
   ```

## Never

- Let a reviewer or the verifier edit the repository; they write only under their scratch directories.
- Accept a finding without a file, a line and a scenario.
- Run a second round of fixes without re-running the checks.
