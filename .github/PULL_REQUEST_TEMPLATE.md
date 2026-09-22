<!-- Title: conventional commit style, e.g. "feat(core): XYStage capability over CMMCorePlus" -->

Closes #<issue>

## What

<!-- One paragraph: what changes for a user of the library / CLI / UI. -->

## Why

<!-- The decision behind it. Link the ADR if one applies (docs/adr/). -->

## How it was verified

- [ ] `ruff check . && ruff format --check . && mypy` pass
- [ ] `pytest` passes locally against the demo devices
- [ ] New behaviour has a test (simulator or unit); hardware-only behaviour has a `FakeCore` test
- [ ] Hardware session run (if applicable) — link the session report issue: #

## Safety

<!-- Does this move anything on a real microscope? Which guards apply (jog limit, confirm, dry-run)? Write "n/a" if purely software. -->

## Docs

- [ ] Docstrings explain *why*, units are in names (`_um`, `_ms`)
- [ ] README / docs / ADR updated if behaviour or a decision changed
