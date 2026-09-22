---
name: adr
description: Propose an architecture decision — create the numbered ADR file from the template with status proposed, add it to the ADR index, open the matching decision issue, and open a PR for review. Argument: a short title.
---

# /adr <title>

1. Read `docs/adr/README.md` and any record the new one touches; a change
   of an accepted decision is a **new** record that supersedes it.
2. Next number = highest existing + 1. Create
   `docs/adr/NNNN-<slug>.md` from the template: Status `proposed`, Date,
   Context (facts), Decision (present tense), Alternatives (each with why
   it lost), Consequences.
3. Add the row to the index table in `docs/adr/README.md`.
4. `gh issue create` with the *Architecture decision* template
   (`type: adr`, `status: needs-decision`), summarising options and the
   recommendation; put the issue number in the record.
5. Branch `adr/NNNN-<slug>`, commit `docs(adr): propose NNNN <title>`,
   open the PR. The review is the discussion; the owner merges it as
   `accepted` (editing the status line) or closes it as `rejected`.
