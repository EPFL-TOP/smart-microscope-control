---
name: point
description: Supervising-session skill — take stock of the project (open PRs and their verdicts, issues by status, develop reports, review outcomes, field feedback from hardware sessions and survey folders), do the housekeeping, and give the owner one short list — what to launch in develop sessions, what to merge, what needs the owner, what is waiting — plus any process improvement. Use at the start of every supervising session and whenever the owner asks where things stand.
---

# /point

You are the **supervising (design) session** that opens a cycle. This skill
keeps you cheap and short: a script collects the facts, you add judgement,
and you **hand the design work to fresh sessions** by listing it, just as
you hand implementation to `/develop`. Read the script's output, not GitHub.
The session ends when the list is posted.

1. **Facts.** Run `python scripts/dev/point.py` and read only its Markdown:
   first the **CI that stops the merges** (a red `main`, an open `CI nightly
   red` issue, a failed nightly run with no issue: ADR-0009), then open PRs with checks and the latest design
   verdict, issues by status,
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
   name, if they do not exist yet. Run `python scripts/dev/worktree.py
   clean` from the main checkout; one line in the output on what it
   removed.
4. **Plans.** Any issue that belongs in the next wave and has no plan goes on
   the list as `/plan <issue>`, for a fresh session: Opus for risk
   `moves-hardware` or design-heavy work, Sonnet otherwise. Write a plan here
   only when it is short and needs no design decision.
5. **Next wave.** Up to three `status: ready` issues that touch disjoint
   files, in milestone and priority order, each with its command and
   suggested model. Each PR awaiting a review goes on the list as
   `/review <pr>`, for a fresh session: Opus when its risk is
   `moves-hardware` (it runs `/code-review`), Sonnet otherwise. Review a PR
   here only when its risk is `docs`.
6. **Process.** Look at the reports' deviations and friction and at the
   review outcomes since the last point. When the same problem shows up
   twice, or a confirmed finding reveals a category missing from the
   failure-modes list, fix the skill, the agent or the list — a small PR of
   its own — and mention it.
7. **Output.** Post the list as a comment on the open issue titled
   `Point: next steps` (create and pin it if missing; one comment per point,
   in English), then give the owner the same list in their language, at most
   about twenty lines. **A red `main`, or an open `CI nightly red` issue, is
   step 0**, above everything: no other PR is merged until the owner has
   chosen to revert or to fix forward (ADR-0009). Name the run or the issue,
   the tests it names, and the command that fixes it (a `/develop` or a
   revert), and put every merge of the day after it.

   The list is written as **lanes**: up to three lanes run in parallel
   (`S1`, `S2`, `S3`), one session at a time inside a lane, so the owner
   opens at most three sessions and never has to judge what may run with
   what. A lane holds the issues that share files (the plans' **Do not run
   at the same time** lines decide), in priority order; issues on disjoint
   files go in different lanes. Each lane is a numbered sequence of steps
   (`S1.1`, `S1.2`, ...), and **each step is one session or one merge**, with
   its command, its model and what it waits for. Every issue is followed
   from its plan to its merge inside its lane: `/plan n` → merge the design
   PR, if there is one → `/develop n` (if its **Depends on** is merged) →
   `/review <pr>` → merge; a `/feature n` replaces the plan and the develop
   for a small issue. A plan with no develop step after it is an incomplete
   lane. A design PR that a `/plan` opened gets **no `/review` session**:
   the owner reads its decision and merges it on green; a `/review` is for
   a develop session's PR. A step that waits for another lane names it
   (`after S2.3`). A
   session the owner runs at the stand (`/hardware-session`, `during`) is a
   lane of its own. The last step of the day is a fresh `/point`.

   ```markdown
   **Today**
   0. Only when the CI stops the merges: main is red (run <url>), #N `CI nightly red` is open, or the nightly run <url> failed with no issue: revert or `/develop <n>` for the fix, before any merge.
   Run each lane with `/lane S1`, `/lane S2`, `/lane S3` (Sonnet), each in a fresh session; a lane stops where you are needed and says so on this issue.

   **S1 — the plate definitions (#400), then the scan plan (#21)**
   S1.1 `/plan 400` (Opus) → merge its design PR
   S1.2 `/develop 400` (the plan's model) → `/review <pr>` (Sonnet) → merge
   S1.3 `/plan 21` (Sonnet) → merge its design PR → `/develop 21` → `/review` (Opus: moves-hardware) → merge

   **S2 — the profile (#401)**
   S2.1 `/feature 401` (Sonnet) → `/review <pr>` (Sonnet) → merge

   **S3 — the decision (#402), then the stand**
   S3.1 `/plan 402` (Opus), **with you**: its design PR ends with the decision; read it and merge it
   S3.2 after S1.2 and S2.1: `/hardware-session nikon-ti2 before` (Opus)
   S3.3 tomorrow at the stand: `/hardware-session nikon-ti2 during` (Opus); back: `... after`, then `/point`

   **End of the day**: `/point`.
   **For you** — decisions and hardware that today's steps do not wait for
   **Waiting** — #26, on #25
   **Process** — the develop skill now asks for …
   ```

   **The grammar of a lane** is read by `scripts/dev/lane.py` (`/lane` drives
   the lane from it), so it is written exactly: a lane is a line beginning
   `**S<k> ` (bold, then a dash and its title); **one step per line**,
   beginning `S<k>.<n> `; the segments of a step are joined by ` → `; each
   session segment holds **one backticked command** (`/plan <n>`,
   `/develop <n>`, `/feature <n>`, `/review <pr>`) with its model in the
   parenthesis right after it (`(Opus)`, `(the plan's model)`); a design PR's
   merge says `merge its design PR`, and a last `merge` is the PR's; a session
   that needs the owner (a walk, a decision, the stand) says `with you` or is a
   `/hardware-session`, `/adr` or `/point`; a step that waits for others starts
   `after S<k>.<n>:`. A `/plan` whose design PR is the whole result (a
   decision) has no develop after it: write it `with you`. An issue built as
   several PRs is written as owner steps: a lane drives one PR per issue. **Before posting, run `python
   scripts/dev/lane.py --check <draft file>` and fix what it prints** (a plan
   with no develop after it, a review or a merge with no issue, an `after`
   naming no step, a later step of its own lane, or a cycle of lanes that
   wait for each other).

   No history and no explanations unless the owner asks.

## Never

- Read full diffs or crawl every issue yourself; that is what the script and subagents are for.
- Launch develop sessions yourself, or merge.
- Run several reviews or plans in this session: list them for fresh sessions instead.
