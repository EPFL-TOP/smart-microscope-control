---
name: review
description: Design-session skill — a light, proportionate review of a pull request against its plan, the architecture rules and the develop session's own adversarial review, delegating any code reading to a cheaper subagent; posts a comment review with an explicit verdict (ready to merge / changes requested, at most three blocking items). Argument: the PR number.
---

# /review <pr-number>

You are the **design session**, and this review must stay cheap. The
develop session has already attacked its own change (`/adversarial-review`).
You judge the result against the plan and the rules; you do not re-review
every line.

## Steps

1. Read: `gh pr view N` (body), `gh pr checks N`, the issue's `## Plan` and
   `## Report` (with its **Adversarial review** line), `gh pr diff N
   --name-only`, and only the parts of the diff that implement the
   interfaces the design doc names.
2. Decide on these points only, in order:
   1. **Scope** — the plan, plus "Beyond the plan" fixes that change no
      interface. A renamed or changed fixed interface is blocking.
   2. **Rules** — ruff enforces most of `CLAUDE.md` (imports, `print`,
      encodings). Check what it cannot: roles rather than labels, safety
      guards that cannot be bypassed, measured versus assumed.
   3. **Evidence** — the plan's named tests exist; the adversarial review ran
      with the lenses its risk requires; every confirmed blocking finding is
      fixed or deferred with a reason.
   4. **The default path** — the question a cheap reviewer may miss: on the
      path the owner will actually use, can this change fail silently?
   5. **Report honesty** — "Verified / Not verified" matches the diff and CI.
3. When a point needs code reading, delegate it: one Agent call with
   `model: "sonnet"`, the exact question, the files, and the answer format
   you want. Keep only its conclusion.
4. Risk `moves-hardware` only: run `/code-review` on the PR and fold the
   confirmed findings in.
5. Post a **comment review** — never `--approve` or `--request-changes`:
   every PR here is opened with the owner's account, and GitHub refuses
   those from a PR's own author. Branch protection needs green CI, not an
   approval. `gh pr review N --comment --body-file review.md`, with:

   ```markdown
   ## Review (design session, YYYY-MM-DD)

   **Verdict: ready to merge**        ← or: **Verdict: changes requested**

   **Checked**: scope · rules · evidence · default path · report
   **Blocking** (at most three, only with "changes requested"):
   - `path/file.py:42` — what to change, and why
   **Optional** (never blocking; what is not done goes to #follow-up):
   - …
   ```

   Anything that does not threaten the default path or break a rule is
   optional, or a follow-up issue you open.
6. **Ready to merge** → tell the owner in one line that #N can be merged
   (squash) and what it unblocks; set those issues to `status: ready` if
   their plan is posted and every dependency is merged. **Changes
   requested** → tell the owner to run `/develop <issue>` again: it finds
   this review and fixes the blocking items.

## Never

- Merge. Rewrite the PR yourself. Give a ready verdict with red CI.
- Read the whole diff of a `docs`, `pure`, `tooling` or `reads-hardware` change.
- Use `--approve` or `--request-changes`.
