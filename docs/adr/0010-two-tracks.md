# 0010 — Two tracks in one repository: the stand track and the analysis track

- **Status**: **accepted** 2026-11-25 by the owner (in chat)
- **Date**: 2026-11-25
- **Issue**: #606
- **Amends**: [0008](0008-repository-workflow.md) (the protection of `main`
  and who merges), [0009](0009-light-ci-full-nightly.md) (who chooses
  between a revert and a fix forward when `main` turns red)

## Context

- A second developer, the collaborator, has started on the analysis side:
  the selection kit of #554 (`smc.data`, `smc.selection`, the guide
  `docs/algorithms/README.md`, `/new-selector`) exists for them, and three
  datasets of real scans are on the share (#648). He has ideas for the
  interface too.
- The working arrangement of today (`docs/design/algorithms.md` §7.4 and
  §7.5, `CLAUDE.md`) makes them an executor: they run `/develop` on
  `area: algorithms` issues; the owner plans, reviews and merges
  everything. The owner already runs up to three lanes a day and about
  seven merges; the collaborator's work would wait in the same queue.
- The owner decided three things on 2026-11-20 (#606): the collaborator
  runs **their own lane and their own piloting session**; their territory in the
  interface is **the analysis side, in new modules**, the acquisition
  screens staying the owner's; they **merge their own pull requests on
  their own paths**. These contradict §7.4 ("the owner remains the only one who
  merges"), §7.5 ("`/point`, `/lane`, `/plan`, `/review` stay the
  owner's") and ADR-0008's last bullet, hence this record.
- What the stand side protects is real: code under `src/smc/hardware/`,
  `profiles/` and the plugins' hardware loops moves stages and turrets on
  five microscopes. What the analysis side writes runs on acquired data
  and reaches a microscope only through a wiring issue of the owner's
  (§7.2).
- **The settings of `main` today** (read 2026-11-25): pull request
  required, the light set's three checks required, `strict: false` (no
  "up to date" rule, ADR-0009), linear history, `enforce_admins: true`,
  `required_approving_review_count: 0`, `require_code_owner_reviews:
  false`. `.github/CODEOWNERS` has the owner's seven lines of #554, which
  therefore enforce nothing yet. No ruleset exists. The repository has five
  administrators besides the collaborator's yet unknown login.
- **How GitHub behaves**, which the mechanism has to fit: an author can
  never approve their own pull request; when a path has several code
  owners, one approval of any of them suffices; a later line of
  `CODEOWNERS` wins over an earlier one, and a line with no owner leaves
  its paths unowned; any user with write access can merge any pull
  request that meets the rules; with `enforce_admins: true`, the classic
  protection binds administrators too, while a **ruleset** can name roles
  that may bypass it, for pull requests only.

## Decision

### Two tracks

**A track is a territory with a lead.** Every path and every issue
belongs to exactly one track.

| | Stand track | Analysis track |
|---|---|---|
| Lead | the owner | the collaborator |
| What | everything that drives or needs a microscope, the architecture, the process | everything that works on acquired data |
| Paths | **every path not listed for the analysis track** (the default) | `src/smc/selection/` except `api.py` and `registry.py`; `src/smc/analysis/` (new, below); `docs/algorithms/`; their tests (`tests/**/test_selector_*.py`, `tests/analysis/`) |
| Issue label | none (the default) | `track: analysis` |
| Piloting | `/point` on #51 | `/point --track analysis` on their own pinned issue |
| Lanes | `S1`, `S2`, `S3` | `A1`, `A2` |

- **The default is the stand.** A new path belongs to the stand track until
  a pull request that adds it to the analysis list is approved by the
  owner. A path missing from the list therefore errs on the side that
  protects the microscopes.
- **`src/smc/analysis/`** is the analysis track's package for its new
  modules, its interface included (the review of fields, the labels of
  models, the results of a selection). Its first design section is its
  own issue, planned by both leads (#606, *What to build*).
- **The interfaces between the tracks belong to both**: the run folder and
  the dataset (`algorithms.md` §2), the loader `smc.data` (§3), the
  selection contract `smc.selection.api` and how `route` finds a name
  (`registry.py`, §5), the plugin API, the design documents that fix
  them. They stay stand paths, so that a change from the analysis side
  needs the owner's approval; a change from either side starts as an
  issue that both leads agree on before any code, and comes back from a
  build as `## Plan deviation`, as today.

### Who does what on each track

- **The lead merges on their track's paths**, on a green light set and a
  `ready to merge` review (`/review`, on that track). Nobody else merges
  there; agents never merge (ADR-0008 stands).
- **The lead runs the supervising skills on their track**: `/point`,
  `/plan`, `/review`, `/lane`, with the track as an argument. `/adr` may be
  run by either; **ADRs are decided by the owner**, and so is everything
  under `docs/adr/` and `docs/design/`. Hardware sessions stay the
  owner's.
- **Comments keep their headers**, and the lane notes carry the lane's
  name (`## Lane (A1, <date>)`); `point.py` already names an unheaded note
  by its author.

### How it is enforced

1. **`.github/CODEOWNERS` is the map**, written from the default:

   ```text
   *                          @clementhelsens
   /src/smc/selection/
   /src/smc/selection/api.py       @clementhelsens
   /src/smc/selection/registry.py  @clementhelsens
   /src/smc/analysis/
   /docs/algorithms/
   /tests/**/test_selector_*.py
   /tests/analysis/
   ```

   The analysis lines have no owner on purpose: the collaborator, as the
   author of their pull requests, could not approve them, and naming the
   owner there would bring the queue back. The collaborator's login
   therefore never needs to be in the file.
2. **A ruleset on `main`, *Code owners***: pull request required, review
   from code owners required, **no general approval count**, bypass
   allowed for the repository-administrator role, *for pull requests
   only*. A pull request that touches a stand path needs the owner's
   approval unless an administrator merges it; one that touches only
   analysis paths needs CI only. The classic protection of today stays as
   it is (the three checks, linear history, `enforce_admins: true`), so
   nobody, administrators included, merges past a red light set.
3. **The owner's own pull requests** (the sessions run under the owner's
   account) touch stand paths and cannot be approved by their author: the
   owner merges them through the ruleset's bypass, which skips only the
   code-owner rule.
4. **The collaborator has write access, not admin**, so the bypass is not
   theirs.

The settings are the owner's to apply (not a pull request); the
acceptance of #606 tries them once on a throwaway pull request from each
side.

### Five rules between the tracks

1. **A path has one track**, and `CODEOWNERS` says which.
2. **A need across the border is an issue for the other track, never an
   edit.** It is §7.2's pair, in both directions: the algorithm, then its
   wiring; a field the page should expose, then the analysis screen that
   uses it. The owner's bypass does not change this rule: `/review` checks
   the paths of every pull request against its track, as it checks them
   against §7.1 today.
3. **One issue, one assignee, one branch.** Nobody pushes to another's
   branch.
4. **Shared files are edited by small appends**: `pyproject.toml`'s entry
   points, `docs/design/failure-modes.md`, `scripts/github/labels.json`.
   They are stand paths, so such an append from the analysis side asks
   for one approval of the owner: a new entry point is a new name on the
   Results screen and a new dependency installs on every microscope PC.
   `main` has no "up to date" rule (ADR-0009): a textual conflict blocks
   the merge, and rebasing is the second author's; two appends that only
   break together are caught by the light set on the push to `main`.
5. **Each Point ends with *Asks to the other track***, as issue numbers:
   what this track waits for from the other. A Point lists no issue of the
   other track anywhere else. The two leads read both Points together once
   a week.

### A red `main` (amends ADR-0009)

The **lead whose merge turned `main` red** chooses between a revert and a
fix forward, the same day; until then neither track merges code that the
failing tests name. The same holds for an open `CI nightly red` issue,
whose failing tests name the code and so the track.

### An alternative interface

An alternative interface is allowed as an **experiment of the analysis
track**, on three conditions:

- it lives in its own module under `src/smc/analysis/`, with its own
  optional dependencies and its own command, and changes nothing of the
  existing page;
- it is a **view over the plugins and the `Microscope` facade**: every
  action it offers is a plugin run that `smc run` can do, it imports
  nothing of `pymmcore_plus`, and it goes through the runner, so that the
  layer's guards apply to it as to any tool. To drive a stand it uses the
  session object the page uses; taking that object out of
  `smc.ui.controller` is a design issue of the stand track, asked by the
  analysis track (rule 2);
- it becomes an interface of the project only through an ADR that amends
  ADR-0006 ("one interface for every microscope").

### Amendments

- **ADR-0008**: the *main is protected* bullet adds "and the *Code owners*
  ruleset (ADR-0010)"; the last bullet reads "The agent follows
  `CLAUDE.md`: it plans in the issue, works on a branch, verifies locally,
  opens the PR, and never merges; **the lead of the track merges** on that
  track's paths (ADR-0010)". The rest stands.
- **ADR-0009**: the *A red `main` stops the merges* bullet and the
  nightly's last sentence read "the lead whose merge turned it red"
  instead of "the owner". The rest stands.

## Alternatives considered

- **The arrangement of today** (§7.4 and §7.5: the collaborator develops,
  the owner plans, reviews and merges). Simple and safe; every analysis
  change waits in the owner's queue, behind three lanes. Rejected by the
  owner on 2026-11-20.
- **A separate repository for the analysis**, its algorithms registered in
  `smc` by entry point. The clearest border; but the interfaces the
  analysis builds on (`smc.data`, `smc.selection.api`, the dataset) are
  still moving, and each of their changes would become two pull requests
  in two repositories, a version pin and a CI of the pair. Worth
  reconsidering once the interfaces have been stable for a milestone.
- **A long-lived `analysis` branch**, merged into `main` from time to time.
  Each merge would be large and late, and the light set would see the
  analysis code against the stand's only at that merge.
- **Both leads as code owners of the analysis paths.** An author cannot
  approve their own pull request, so every analysis pull request would need
  the owner's approval: the queue again.
- **The collaborator as the only code owner of the analysis paths.** The
  same block: their own pull requests could never be approved, unless they had
  the administrator's bypass, which also skips the stand's paths.
- **Rules by convention only** (no code-owner review; `/review` checks the
  paths). Free to set up, and fine between two careful people; but a
  mistaken edit of a stand path from the analysis side would reach a
  microscope PC unreviewed, and the cost of that mistake is a crash of a
  stage, not a red test.
- **`enforce_admins: false`** instead of a ruleset, so that the owner may
  bypass the code-owner rule. It also lets every administrator merge past
  a red light set; the ruleset's bypass skips the code-owner rule alone.

## Consequences

- The collaborator plans, reviews and merges their work without waiting for
  the owner, and the owner's queue carries only the stand's pull requests
  and the analysis side's few crossings (an interface, a shared file).
- **A stand path cannot change without the owner**, whoever writes it; the
  analysis paths need CI and their lead's review only.
- **What is lost**: the owner no longer sees every change before it lands.
  An analysis change that is wrong but green reaches `main`; it reaches a
  microscope only through a wiring issue of the owner's, which is where it
  is checked at a stand.
- **The bypass belongs to the role, not to a person**: the other
  administrators of the repository could also merge past the code-owner
  rule (never past CI). Lowering them to write access, when they do not
  administer the repository, closes that.
- **The owner's rule 2 is kept by discipline and by `/review`**, not by
  GitHub, since the owner's pull requests are merged through the bypass.
- **To build, each its own issue of the stand track** (#606, *What to
  build*): `.github/CODEOWNERS` as above and the ruleset, set by the owner
  in the settings; `point.py --track analysis`, `lane.py` and the skills
  `/point`, `/lane`, `/plan`, `/review` taking the track; `CLAUDE.md`'s
  *Two kinds of session*, step 6's "never merge" and the *collaborator*
  paragraph; `docs/design/algorithms.md` §7.4 and §7.5 rewritten to point
  here; the first design section of `src/smc/analysis/`, planned by both;
  an issue template for an interface idea.
- **Once accepted**, the status lines of 0008 and 0009 read `accepted
  (amended in part by 0009, 0010)` and `accepted (amended in part by
  0010)`.
