---
name: lane
description: Supervising-session skill — run one lane of the latest Point (S1, S2, S3) from where it stands to the next place the owner is needed. Each step is a fresh subagent in its own worktree with the step's model; the session waits for CI and merges, and stops with a note on the tracking issue where a decision, a walk, a stand session, a red PR or a second "changes requested" needs the owner. Argument - the lane, optionally "from S<k>.<n>".
---

# /lane <S> [from S<k>.<n>]

You are a **supervising session** (run it on **Sonnet**, in the main checkout, in
a fresh session) that drives one lane of the Point: the owner opens up to
three of these, one per lane, and acts only on merges and decisions. You never
edit a file, never read a diff, never merge, never decide and never widen a
step. `scripts/dev/lane.py` reads GitHub and says what to do next; you do
exactly that, and write one short note after each step.

The hand-off is GitHub: the Point's lane (what to do), the plans, reports and
reviews the subagents post (what was done), and your `## Lane` notes on the
tracking issue (where the lane is). Memory keeps nothing of it.

## The loop

1. Run `python scripts/dev/lane.py <S>` (add `--from S<k>.<n>` when the
   argument has `from`). Read **only its output**. It prints a header
   `lane S1 | point <createdAt> (comment <id>) | tracking #51`, one line per
   segment with its state, and exactly one `NEXT:` line. Remember the Point's
   comment id from the header: when it differs between two runs, a new Point
   replaced this one; finish nothing new, post a note "a new Point was posted;
   relaunch `/lane` from it", and end. Exit 1 (an unknown lane, no Point), 2 (a
   `gh` failure), 4 (a usage error, or a crash of the script: FM-82) or 130
   (interrupted): say what it printed and end. Exit 3 is the timeout of
   `--wait`, step 3.
2. `NEXT: session <skill> <arg> <model>` followed by `BRIEF: <text>`: **one**
   `Agent` call with `subagent_type: "general-purpose"`, `model: <model>`,
   `isolation: "worktree"`, `description: "<skill> <arg> (lane <S>)"` and, as
   `prompt`, the text after `BRIEF: ` **verbatim**: nothing a previous subagent
   returned, nothing you know about the issue. A fresh agent every time: never
   `SendMessage` to an earlier one, never `fork`. When it returns, post the
   note, then go to 1. When its last lines say it stopped (a `## Plan
   deviation`, "not ready", an error, a refused command), post them as the
   note's **Next** and end: never work around a refusal, never retry the step
   by hand.

   **A step that changed nothing is not run twice.** `lane.py` derives every
   state from GitHub, so a subagent that returned without changing anything
   there (a refused command, a crash before the `Starting` comment, a review
   never posted, a claim of success with no effect, no output at all) leaves
   the same step to do. Its last lines are not evidence; the next run of
   `lane.py` is. When the `NEXT: session ...` line **and** the `BRIEF:` line
   of that run are identical to those of the step you just ran, post the note
   with **Next** `you: <skill> <arg> ended without a result; read its worktree
   or run it by hand`, add that `/lane <S>` resumes it (the state is on
   GitHub, nothing was marked done), and end: never start it again, never read
   the worktree yourself.
   - Compare with the step **just run**, never an earlier one, and compare
     both lines: a fix round's `NEXT:` line equals the first run's
     (`NEXT: session develop 5 opus` twice), only its `BRIEF:` says "this is
     the fix round". So `review 12`, `develop 5` (the fix round), `review 12`
     is not a repeat, while `develop 5` twice or `review 12` twice in a row is.
   - A bound on top: the same `BRIEF:` text is started at most twice in one
     lane session, the second time only after a fix round. Only a `session`
     line is compared: a `wait` has its own timeout (step 3), an `owner` line
     ends the lane.
3. `NEXT: wait <what>` (a CI run, a merge, a step of another lane): post the
   note, unless your previous note already says the same wait; then **one**
   Bash call with `run_in_background: true`: `python scripts/dev/lane.py <S>
   --wait` (the same `--from`). It re-reads every 120 s, for at most 8 h, and
   prints when the next action is no longer a wait; go to 1 when it returns.
   Exit 3 is the timeout: post "still waiting for <what> after 8 h; resume with
   `/lane <S>`" and end. A call that ended with no output at all was killed by
   the harness: go to 1, `lane.py` is cheap to run again.
4. `NEXT: owner <what>`: post the note whose **Next** reads `you: <what>`, then
   how to resume: the `RESUME:` line that follows it (`/lane <S> from <step>`;
   without one, the lane has no later step). End.
5. `NEXT: done`: post the note "lane done" and end.

## The note

On the tracking issue (its number is in the header), one comment per step you
finished or wait you start, written with the Write tool into your scratchpad
and posted with `gh issue comment <n> --body-file <file>`. Header exactly
`## Lane (<S>, <date>)`: `point.py` selects on it and lists the newest note of
each lane in the next Point.

```markdown
## Lane (S1, 2026-11-12)

**Done**: S1.2 `/review 445` (sonnet): ready to merge, https://github.com/…
**Next**: you merge #445; the lane waits for it
```

**Done** is the step and what its subagent's last lines say (a URL, a verdict);
**Next** is what `lane.py` printed next, in a sentence. Two short lines: this
is a log for the owner and for `/point`, not a report.

## Cost

Your context holds `lane.py`'s output, the subagents' five-line endings and your
notes, nothing else. Do not read the plan, the report, the PR or the diff to
check a subagent: `lane.py` derives the state from GitHub on the next run.

## Never

- Edit a file, read a diff, merge (`gh pr merge` is denied), decide a verdict or
  a plan deviation, or run a step the Point does not list.
- Reuse a subagent, fork, or put anything but the `BRIEF:` text in a prompt.
- Switch branches, pull or run `git` in the main checkout: other sessions share it.
- Go on past a `NEXT: owner`, whatever you think the owner would answer.
