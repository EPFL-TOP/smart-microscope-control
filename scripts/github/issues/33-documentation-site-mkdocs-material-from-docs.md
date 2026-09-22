---
title: docs: documentation site (mkdocs-material) from docs/
labels: [type: docs, area: docs, priority: p2, good first issue]
---
## Goal
Publish `docs/` (architecture, ADRs, hardware inventory, plugin pages, CLI reference generated from typer) on GitHub Pages so lab members read it without cloning.

## Acceptance criteria
- [ ] `mkdocs.yml` + `[docs]` extra; `mkdocs build --strict` in CI
- [ ] Deployed on merge to `main`
