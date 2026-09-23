---
name: plan
description: Design-session skill — turn an issue into an executable plan that a separate, clean /develop session can implement without further design decisions. Names the risk level, the failure modes the code must handle and the suggested model; writes the plan as an issue comment in the fixed hand-off format, links the design doc, sets dependencies and the "status: ready" label. Argument: the issue number.
---

# /plan <issue-number>

You are the **design session**. Your output is a plan good enough that
another session, starting from zero context and possibly on a cheaper model,
implements it correctly without asking a design question.

## Steps

1. Read the issue (`gh issue view N --comments`), `CLAUDE.md`, the ADRs it
   touches, and the design doc for its milestone (`docs/design/*.md`). If the
   interface the issue needs is not fixed in a design doc, fix it there first
   (a PR of its own): parallel executors depend on it.
2. **Think through the failure modes before writing the steps.** For code
   that reads or moves hardware, go through `docs/design/failure-modes.md`
   and name the entries this code must handle. A gap found later in review
   is a defect of the plan, not of the implementation.
3. Check dependencies: which issues must be merged first, which run in
   parallel and must not touch the same modules.
4. Post the plan as a comment, exactly in this format:

   ```markdown
   ## Plan (design session, YYYY-MM-DD)

   **Design reference**: docs/design/<file>.md §N, §M
   **Depends on**: #a, #b — or "none"
   **Risk**: docs | pure | tooling | reads-hardware | moves-hardware
   **Failure modes to handle**: FM-xx, FM-yy — or "the defaults for this risk"
   **Suggested model**: sonnet — or opus, only for design-heavy or safety-critical work
   **In scope**: …  **Out of scope**: …

   ### Steps
   1. …  (ordered; each step small enough to be one commit)

   ### Files
   - `src/smc/…` (new|edit) — what it holds
   - `tests/…` (new) — what it proves

   ### Tests that prove it
   - `tests/unit/test_x.py::test_behaviour_or_failure_prevented`

   ### Definition of done
   - [ ] the issue's acceptance criteria, made concrete
   - [ ] `ruff check . && ruff format --check . && mypy && pytest` green; CI green
   - [ ] adversarial review run for this risk level; confirmed findings fixed or deferred with a reason

   ### Decided here (do not re-decide)
   - defaults, names, edge-case behaviour the executor might otherwise guess

   ### Risks / watch for
   - …

   **Ready for /develop**: yes | blocked by #a
   ```

5. Labels: `status: ready` when unblocked (remove `status: blocked`),
   otherwise `status: blocked` and name the blocker. Set the milestone.

## Rules

- Plans name modules, classes, signatures and test names. "Implement the
  facade" is not a plan; "add `Microscope.require()` per design §7, test
  `test_require_missing_role_names_the_role`" is.
- **Risk** sets how hard the change is attacked before its PR and what may
  block its review: `docs` nothing; `pure` tests; `tooling` tests and
  Windows; `reads-hardware` failure modes, tests and Windows;
  `moves-hardware` all of that plus `/code-review` in the design review.
- Do not write the implementation here. Keep the plan consistent with the
  design doc; if reality contradicts the doc, update the doc in a PR.
