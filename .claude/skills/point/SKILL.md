---
name: point
description: Supervising-session skill — take stock of one track of the project (open PRs and their verdicts, issues by status, develop reports, review outcomes, field feedback from hardware sessions and survey folders), do the housekeeping, and give the lead of the track one short list — what to launch in develop sessions, what to merge, what needs the lead, what is waiting, what the other track is asked — plus any process improvement. Use at the start of every supervising session and whenever the lead asks where things stand. Argument - `--track analysis` for the analysis track (the stand's with none).
---

# /point

You are the **supervising (design) session** that opens a cycle. This skill
keeps you cheap and short: a script collects the facts, you add judgement,
and you **hand the design work to fresh sessions** by listing it, just as
you hand implementation to `/develop`. Read the script's output, not GitHub.
The session ends when the list is posted.

**Two tracks (ADR-0010).** The repository has two tracks, each with a lead: the
**stand** (the owner; every path no line of `.github/CODEOWNERS` frees, and the
issues with no `track:` label) and the **analysis** track (the collaborator; the
paths of the lines with no owner, and the issues labelled `track: analysis`). A
Point is one track's: `/point` is the stand's, `/point --track analysis` the
analysis track's, each posted on its own pinned issue (*Point: next steps*,
*Point (analysis): next steps*). "The lead" below is the lead of the Point's
track, and "the owner" in the stand's Point. Everything you list is of this
track; the other track's issues are named in one place only, *Asks to the other
track*.

1. **Facts.** Run `python scripts/dev/point.py` (`--track analysis` for the
   analysis track) and read only its Markdown:
   first the **CI that stops the merges** (a red `main`, an open `CI nightly
   red` issue, a failed nightly run with no issue: ADR-0009), then open PRs with checks and the latest design
   verdict (the PRs whose paths are this track's; one marked `crossing` belongs
   to an issue of the other track), issues by status (this track's),
   blocked issues whose dependencies are now merged, what develop sessions
   reported since the last point (follow-ups, not verified, deviations),
   the leads' own notes, survey folders not yet curated (the stand's only), ADR
   statuses, milestone progress, and last **Asks to the other track**.
2. **Field feedback** (the stand's only: the analysis track reads data, not a
   microscope) turns into work, never into chat:
   - a new folder in `local/surveys/` → the task "curate survey `<folder>`" for this session (the brief is `docs/hardware/microscope-pc-setup.md` §3). Delegate the reading to one Sonnet subagent with that brief; review its output; never publish raw survey content.
   - a hardware-session report → issues, profile entries, tests or entries in `docs/design/failure-modes.md`.
   - an owner note that asks for something → an issue with acceptance criteria.
3. **Housekeeping — do it, do not list it**, on this track's issues only. For a blocked issue whose
   dependencies are merged: re-read its plan against the merged code, fix the
   plan if the code moved, then set `status: ready`. Remove status labels
   from closed issues. Open the follow-up issues the reports and reviews
   name, if they do not exist yet. Run `python scripts/dev/worktree.py
   clean` from the main checkout; one line in the output on what it
   removed.
4. **Plans.** Any issue of this track that belongs in the next wave and has no plan goes on
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
   `Point: next steps` (`Point (analysis): next steps` for `--track analysis`;
   create it, label it `track: analysis` where it is the analysis track's, and
   pin it, if missing; one comment per point,
   in English), then give the lead the same list in their language, at most
   about twenty lines. **A red `main`, or an open `CI nightly red` issue, is
   step 0**, above everything: no other PR is merged until the lead whose
   merge turned it red has chosen to revert or to fix forward (ADR-0009,
   ADR-0010); the failing tests name the code and so the track. Name the run
   or the issue, the tests it names, and the command that fixes it (a
   `/develop` or a revert), and put every merge of the day after it.

   The list is written as **lanes**: up to three lanes run in parallel
   (`S1`, `S2`, `S3` on the stand track; `A1`, `A2` on the analysis track), one
   session at a time inside a lane, so the lead
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

   **Lanes the owner can follow** (the lanes of 2026-11-18 were hard to
   follow: S1 stopped in its middle for an owner session, then waited for S2;
   S3 began with an owner session and stopped at once):
   - **Lanes do not wait for each other.** Two issues where one waits for the
     other go in the same lane. An `after S<k>.<n>:` is the exception, at
     most one in a Point, and never in both directions between two lanes.
   - **A session with the owner is never in the middle of a lane.** `with
     you`, `/adr` and `/hardware-session` steps go in a lane of the owner's
     own (the last lane, which `/lane` does not run: the owner opens those
     sessions), in the order the owner does them; a lane that `/lane` runs
     holds only steps that need the owner for a merge or a decision it
     meets on the way.
   - **Every open thing has one place.** What a lane's **For /point** lines,
     the Reports and the reviews raised becomes an issue, a housekeeping
     step done here, or one line of **For you**: nothing is left for the
     owner to remember from a lane note.

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
   **Asks to the other track** — #26 (awaited by #25)
   **Process** — the develop skill now asks for …
   ```

   **Every Point ends with *Asks to the other track*** (rule 5 of ADR-0010), as
   issue numbers: what this track waits for from the other, copied from the
   last section of `point.py` (`- none` when there is nothing). The analysis
   track's Point has the same shape, with `A1`, `A2` for its lanes and its own
   issues. A Point names **no issue of the other track anywhere else**, in a
   lane or under **For you**. A pull request of the stand's list that is marked
   `crossing` (an analysis issue that touches a stand path) is a line of **For
   you** on the stand's Point, "approve #N (a crossing; its lead merges)": the
   owner approves it on GitHub, the lead of its issue's track merges it. The
   two leads read both Points together once a week.

   **The grammar of a lane** is read by `scripts/dev/lane.py` (`/lane` drives
   the lane from it), so it is written exactly: a lane is a line beginning
   `**S<k> ` (`**A<k> ` on the analysis track's Point; bold, then a dash and its
   title; **one Point holds the lanes of one letter**); **one step per line**,
   beginning `S<k>.<n> ` (`A<k>.<n> `); the segments of a step are joined by ` → `; each
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

   **A Point posted while lanes run ends them** (2026-11-25, morning: `/lane S1` and
   `/lane S2` stopped on the new Point's comment id, and `lane.py` dates a plan against
   the newest Point, so a lane copied as it was would have asked for finished plans
   again). When `point.py` shows a lane whose newest note is not `lane done`, tell the
   owner **before posting** that those lanes will stop and must be relaunched, and
   write each of them from its **first unfinished segment**: `/develop n` for an issue
   whose plan is posted, never a finished `/plan n`; the step is `lane.py <S>`'s
   first line that is not `done`.

   No history and no explanations unless the owner asks.

## Never

- Read full diffs or crawl every issue yourself; that is what the script and subagents are for.
- Launch develop sessions yourself, or merge.
- Run several reviews or plans in this session: list them for fresh sessions instead.
- Name an issue of the other track anywhere but *Asks to the other track*.
