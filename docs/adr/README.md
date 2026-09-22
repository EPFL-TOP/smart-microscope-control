# Architecture decision records

Decisions that shape this project, one file each, numbered in the order they
were made. A record is never edited once **accepted**: a change of mind is a
new record that supersedes it. That is what makes the folder useful two
years from now — it says not just what we do, but what we chose *against*
and why.

| # | Title | Status |
|---|-------|--------|
| [0001](0001-record-architecture-decisions.md) | Record architecture decisions | accepted |
| [0002](0002-pymmcore-plus-as-the-hardware-core.md) | Python, Micro-Manager and pymmcore-plus as the hardware core | accepted |
| [0003](0003-capabilities-roles-and-profiles.md) | The abstraction layer: capabilities, roles and profiles | accepted |
| [0004](0004-plugin-architecture.md) | Tools are plugins over capabilities | accepted |
| [0005](0005-testing-strategy.md) | Testing strategy: simulator first, hardware by session | accepted |
| [0006](0006-user-interface.md) | User interface: CLI first, then a web UI | **proposed** |
| [0007](0007-non-micromanager-hardware.md) | Hardware without a Micro-Manager adapter (Zeiss MTB, Viventis) | **proposed** |
| [0008](0008-repository-workflow.md) | Repository workflow: issues, branches, PRs, CI | accepted |

## Statuses

`proposed` → `accepted` | `rejected` → later `superseded by NNNN` or `deprecated`.

## Template

```markdown
# NNNN — Title

- **Status**: proposed | accepted | rejected | superseded by NNNN
- **Date**: YYYY-MM-DD
- **Issue**: #N

## Context
What forces are at play. Facts, not opinions.

## Decision
What we do. Present tense.

## Alternatives considered
Each with why it lost.

## Consequences
What becomes easier, what becomes harder, what we must remember.
```

Propose a decision by opening an *Architecture decision* issue, then a PR
adding the record with status `proposed`. The PR review is the discussion;
merging with status `accepted` is the decision.
