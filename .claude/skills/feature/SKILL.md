---
name: feature
description: Single-session shortcut that runs /plan then /develop on one issue when no separate execution session is available. Prefer the split (design session plans and reviews; a clean session develops) for anything non-trivial. Argument: the issue number.
---

# /feature <issue-number>

The project's normal loop is **/plan** (design session) → **/develop**
(clean execution session) → **/review** (design session) → owner merges.
Use this shortcut only for small, self-contained issues where a hand-off
costs more than it saves.

1. Run the steps of `/plan $ARGUMENTS`; post the plan comment.
2. Run the steps of `/develop $ARGUMENTS` in the same session — including
   the worktree isolation, the Report comment and the PR.
3. Tell the owner the PR is open and that it was planned and built in one
   session — so they review it more carefully than a split-session PR.
