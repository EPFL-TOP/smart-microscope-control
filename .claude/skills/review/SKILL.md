---
name: review
description: Design-session skill — review a pull request against its issue plan, the design doc, the architecture rules and the testing bar, then post a GitHub review (approve or request changes) so the owner can merge with confidence. Argument: the PR number.
---

# /review <pr-number>

You are the **design session** reviewing what an execution session built.
The owner merges; you decide whether it is ready for that.

## Steps

1. `gh pr view N --comments`, `gh pr diff N`, `gh pr checks N`. Find the
   issue (`Closes #M`), its `## Plan` and the `## Report`.
2. Check, in this order — stop at the first blocking failure:
   1. **Scope**: the diff implements the plan, nothing more, nothing less.
      Renamed interface from the design doc → request changes.
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
   6. **Windows**: `encoding="utf-8"`, `pathlib`, no shell-isms.
3. Run the built-in `/code-review` on the PR for correctness bugs and fold
   any confirmed finding into your review.
4. Post the review: `gh pr review N --approve` with a short summary of what
   was checked, or `gh pr review N --request-changes` with one bullet per
   blocking item, each naming file and line and what to change. Nits are
   marked "nit:" and never block.
5. On approval, tell the owner in one line that #N is ready to merge and
   which issues it unblocks (update their labels to `status: ready` if
   their plan is posted).

## Never

- Merge. Rewrite the PR yourself (comment instead; the develop session
  or the owner fixes). Approve with red CI.
