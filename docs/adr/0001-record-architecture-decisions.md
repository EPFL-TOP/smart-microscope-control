# 0001 — Record architecture decisions

- **Status**: accepted
- **Date**: 2026-09-22

## Context

This project will run for years, across five microscopes, several people and
an AI coding agent. The two repositories it grows from (`nikon-control`,
`lightsheet-live-tracking-tool`) each carry hard-won knowledge in docstrings
and `docs/*.md` — but their *decisions* (why MTB instead of the Micro-Manager
Zeiss adapter, why a role-based device layer, why Bokeh) are scattered and
easy to re-litigate.

## Decision

Every decision that constrains future work is written down as an
architecture decision record in `docs/adr/`, following the template in the
folder's README. Records are immutable once accepted; changes are new records.
An issue labelled `type: adr` tracks each decision while it is open.

## Alternatives considered

- **A single ARCHITECTURE.md**: drifts, and loses the history of why.
- **Decisions in issue threads only**: not discoverable from the code.

## Consequences

- Anyone (human or agent) proposing to change direction must first read the
  relevant record and write a superseding one.
- Small cost per decision; large saving per re-litigation avoided.
