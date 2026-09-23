---
name: review
description: Design-session skill — review a pull request against its issue plan, the design doc, the architecture rules and the testing bar, then post a comment review with an explicit verdict (ready to merge / changes requested) so the owner can merge with confidence. Argument: the PR number.
---

# /review <pr-number>

You are the **design session** reviewing what an execution session built.
The owner merges; you decide whether it is ready for that.

## Steps

1. `gh pr view N --comments`, `gh pr diff N`, `gh pr checks N`. Find the
   issue (`Closes #M`), its `## Plan` and the `## Report`.
2. Check, in this order — stop at the first blocking failure:
   1. **Scope**: the diff implements the plan, nothing more, nothing less.
      Renamed interface from the design doc → changes requested.
   2. **Architecture** (`CLAUDE.md`): hardware only via `smc.hardware`;
      no `pymmcore_plus` import outside it; roles not labels; units in
      names; safety guards not bypassable; measured vs assumed labelled;
      failures are findings.
   3. **Tests**: the plan's named tests exist and mean something (a test
      that cannot fail is not a test); simulator coverage for hardware
      paths; `FakeCore` for quirks; contract suite still passes.
   4. **Verification honesty**: the Report's "Verified / Not verified"
      matches the diff and CI.
   5. **Docs**: docstrings explain why; docs/profile/inventory updated
      where the plan said so.
   6. **Windows**: `encoding="utf-8"`, `pathlib`, no shell-isms, nothing
      printed that a cp1252 stream cannot encode without the CLI's stream
      fix (#42).
3. Run the built-in `/code-review` on the PR for correctness bugs and fold
   any confirmed finding into your review.
4. Post the review as a **comment review**, never `--approve` or
   `--request-changes`: every PR in this repository is opened with the
   owner's account, and GitHub refuses an approval or a change request from
   a PR's own author. Merging needs no approval — branch protection
   requires green CI on an up-to-date branch. Write the body, then
   `gh pr review N --comment --body-file review.md`:

   ```markdown
   ## Review (design session, YYYY-MM-DD)

   **Verdict: ready to merge**        ← or: **Verdict: changes requested**

   **Checked**: scope · architecture · tests · report honesty · docs · Windows
   **Blocking** (only with "changes requested"):
   - `path/file.py:42` — what to change, and why
   **Nits** (never blocking):
   - nit: …
   ```

5. **Ready to merge** → tell the owner in one line that #N can be merged
   (squash) and which issues it unblocks; set those issues to
   `status: ready` if their plan is posted and every dependency is merged.
   **Changes requested** → tell the owner to run `/develop <issue>` again
   (in the same execution session or a new one): it finds this review and
   fixes exactly the blocking items. Re-review after its push.

## Never

- Merge. Rewrite the PR yourself (comment instead). Give a ready verdict
  with red CI. Use `--approve` or `--request-changes`.
