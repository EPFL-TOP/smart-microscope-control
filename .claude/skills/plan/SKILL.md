---
name: plan
description: Design-session skill — turn an issue into an executable plan that a separate, clean /develop session can implement without further design decisions. Writes the plan as an issue comment in the fixed handoff format, links the design doc, sets dependencies and the "status: ready" label. Argument: the issue number.
---

# /plan <issue-number>

You are the **design session**. Your output is a plan good enough that
another session, starting from zero context, implements it correctly
without asking a design question. If the plan needs a decision the issue
does not settle, make it here (or write the ADR) — never leave it to the
executor.

## Steps

1. Read the issue (`gh issue view N --comments`), `CLAUDE.md`, the ADRs it
   touches, and the design doc for its milestone (`docs/design/*.md`). If
   the interface the issue needs is not fixed in a design doc, fix it there
   first (that is a PR of its own) — parallel executors depend on it.
2. Check dependencies: which issues must be merged first? Which run in
   parallel and must not touch the same modules?
3. Post the plan as a comment, exactly in this format:

   ```markdown
   ## Plan (design session, YYYY-MM-DD)

   **Design reference**: docs/design/<file>.md §N, §M
   **Depends on**: #a, #b — or "none"
   **In scope**: …  **Out of scope**: …

   ### Steps
   1. …  (ordered; each step small enough to be one commit)

   ### Files
   - `src/smc/…` (new|edit) — what it holds
   - `tests/…` (new) — what it proves

   ### Tests that prove it
   - `tests/unit/test_x.py::test_behaviour_or_failure_prevented`
   - contract / simulator cases by name

   ### Definition of done
   - [ ] the issue's acceptance criteria, made concrete
   - [ ] `ruff check . && ruff format --check . && mypy && pytest` green; CI green
   - [ ] docstrings say why; docs updated where named above

   ### Decided here (do not re-decide)
   - defaults, names, edge-case behaviour the executor might otherwise guess

   ### Risks / watch for
   - …

   **Ready for /develop**: yes | blocked by #a
   ```

4. Labels: add `status: ready` when unblocked (remove `status: blocked`);
   otherwise `status: blocked` and name the blocker. Set the milestone.
5. Tell the owner which issues are ready and which can run in parallel
   (separate `git worktree`s), so they can start `/develop` sessions.

## Rules

- Plans name modules, classes, signatures and test names. "Implement the
  facade" is not a plan; "add `Microscope.require()` per design §7, test
  `test_require_missing_role_names_the_role`" is.
- Do not write the implementation here. If you find yourself coding, stop:
  either the plan is unclear (fix it) or you are in the wrong skill.
- Keep the plan consistent with the design doc; if reality contradicts the
  doc, update the doc in a PR and reference the change.
