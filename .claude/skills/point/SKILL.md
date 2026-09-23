---
name: point
description: Supervising-session skill — take stock of the project (open PRs and their verdicts, issues by status, develop reports, review outcomes, field feedback from hardware sessions and survey folders), do the housekeeping, and give the owner one short list — what to launch in develop sessions, what to merge, what needs the owner, what is waiting — plus any process improvement. Use at the start of every supervising session and whenever the owner asks where things stand.
---

# /point

You are the **supervising (design) session**. This skill keeps you cheap: a
script collects the facts, you add judgement. Read its output, not GitHub.

1. **Facts.** Run `python scripts/dev/point.py` and read only its Markdown:
   open PRs with checks and the latest design verdict, issues by status,
   blocked issues whose dependencies are now merged, what develop sessions
   reported since the last point (follow-ups, not verified, deviations),
   the owner's own notes, survey folders not yet curated, ADR statuses,
   milestone progress.
2. **Field feedback** turns into work, never into chat:
   - a new folder in `local/surveys/` → the task "curate survey `<folder>`" for this session (the brief is `docs/hardware/microscope-pc-setup.md` §3). Delegate the reading to one Sonnet subagent with that brief; review its output; never publish raw survey content.
   - a hardware-session report → issues, profile entries, tests or entries in `docs/design/failure-modes.md`.
   - an owner note that asks for something → an issue with acceptance criteria.
3. **Housekeeping — do it, do not list it.** For a blocked issue whose
   dependencies are merged: re-read its plan against the merged code, fix the
   plan if the code moved, then set `status: ready`. Remove status labels
   from closed issues. Open the follow-up issues the reports and reviews
   name, if they do not exist yet.
4. **Plans.** Any issue that belongs in the next wave and has no plan gets
   one (`/plan`). Plans carry a **Risk**, the **Failure modes to handle** and
   a **Suggested model**.
5. **Next wave.** Up to three `status: ready` issues that touch disjoint
   files, in milestone and priority order. For each: the command and the
   suggested model. PRs awaiting a review → run `/review` on them now.
6. **Process.** Look at the reports' deviations and friction and at the
   review outcomes since the last point. When the same problem shows up
   twice, or a confirmed finding reveals a category missing from the
   failure-modes list, fix the skill, the agent or the list — a small PR of
   its own — and mention it.
7. **Output.** Post the list as a comment on the open issue titled
   `Point: next steps` (create and pin it if missing; one comment per point,
   in English), then give the owner the same list in their language, at most
   about fifteen lines:

   ```markdown
   **To launch** — `/develop 6` (Sonnet), `/develop 5` (Sonnet)
   **To merge** — #47, ready to merge
   **For you** — decide ADR-0007; survey the Ti2 PC
   **Waiting** — #8, on #5 and #6
   **Process** — the develop skill now asks for …
   ```

   No history and no explanations unless the owner asks.

## Never

- Read full diffs or crawl every issue yourself; that is what the script and subagents are for.
- Launch develop sessions yourself, or merge.
