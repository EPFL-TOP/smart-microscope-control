---
name: feature
description: Take a GitHub issue from plan to pull request in this repository — plan in the issue, branch, implement test-first against the demo devices, verify, open the PR. Use for any feature, fix or port. Argument: the issue number.
---

# /feature <issue-number>

Follow `CLAUDE.md`. Steps, in order; do not skip the verification.

1. **Read the issue**: `gh issue view $ARGUMENTS --comments`. Read the ADRs
   it links. If acceptance criteria are missing or ambiguous, propose them
   in a comment and ask the owner before coding.
2. **Plan in the issue** (`gh issue comment $ARGUMENTS --body-file …`):
   3–10 bullet steps, files touched, tests that prove it, whether hardware
   is needed. Mark the issue `status: in-progress`.
3. **Branch**: `git switch -c feat/$ARGUMENTS-<slug>` from an up-to-date
   `main`.
4. **Implement test-first**. Simulator tests use the `demo_core` /
   `demo_microscope` fixtures; pure algorithms get unit tests (use
   `hypothesis` for invariants); vendor quirks get a `FakeCore` test. When
   porting from `../nikon-control` or `../lightsheet-live-tracking-tool`,
   carry the tests and the *why* docstrings over.
5. **Verify**: `ruff check . && ruff format --check . && mypy && pytest`.
   Fix root causes. Run `smc doctor` if `smc.hardware` changed.
6. **Docs**: docstrings, `docs/` page or ADR if a decision changed,
   `docs/hardware/inventory.md` if new stand knowledge appeared.
7. **Commit** with conventional messages; **push** the branch; **open the
   PR** with `gh pr create --fill --body-file` using
   `.github/PULL_REQUEST_TEMPLATE.md`, `Closes #$ARGUMENTS`, and an honest
   "How it was verified". Never merge.
8. Report to the owner: what landed, what was verified, what was not.
