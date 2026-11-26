# 0011 — Where the models run: one detector interface, local or served by imaging-server-kit

- **Status**: proposed
- **Date**: 2026-11-25
- **Issue**: #599

## Context

- **One model, in the acquisition PC's process.** `cells` (m3 §16) runs a
  Faster R-CNN (`cell_classes_model-cat-v2.pth`) through `TorchDetector`,
  which imports torch inside its constructor; the `detect` extra
  (`torch`, `torchvision`) is installed on the microscope PC for it. It
  takes 0.5 to 0.8 s a frame on the Ti2-E's PC (sessions 2 and 6), so 15 to
  25 minutes for a whole well at the 40×, after the scan.
- **The boundary already exists in the code.** `Detector`
  (`smc.plugins.cells.detector`) is a `Protocol`: `classes`, `device`, and
  `detect(frame) -> list[Detection]`, a frame in and boxes with a label and
  a score out. `TorchDetector` and `BlobDetector` implement it; the tests
  stub it. No other module of `src/` calls a model: `density`, `route`,
  `report`, `smc.data` and the selectors read the cells file.
- **What a result says of its model today**: the weights' file name (never
  its path) and their SHA-256, the classes, the device, the score
  threshold, the time per frame. A checkpoint refuses to resume with other
  weights (FM-111). The pre-processing (percentiles, input size) is not
  part of that identity.
- **The owner's direction** (#599, 2026-11-19): on-line evaluation uses
  `imaging-server-kit` on a remote machine dedicated to it, so as not to
  load the acquisition PC. Detection on the fly (#635), several models by
  name (#600) and closed-loop tracking (#32) all wait on this record.
- **The lab already runs it.** `lightsheet-live-tracking-tool` serves its
  tracker (CoTracker3) and a detector from a GPU machine with
  `serverkit_main.py` (`imaging-server-kit==0.0.12`, two
  `@algorithm_server`s in a `MultiAlgorithmServer`, uvicorn on port 8000);
  its client calls `Client(address).run_algorithm(algorithm=..., image=frame)`.
  The server's script carries its own pre-processing (a min-max
  normalisation), different from the local one. The client sets no
  timeout, catches every exception, and answers an empty box with a score
  of 0 when the server fails. The version is at 0.0.x, and its wire format
  and timeouts were not inspected for this record (it is not installed
  here).
- **The repository is public.** No profile holds an address today; the
  only machine-level settings are environment variables (`SMC_PROFILE`,
  `SMC_PROFILES`, `SMC_MODELS`). Run documents carry no host or account
  name (`algorithms.md` §2.2).
- **A frame is about 4 MB** (1412² 16-bit, Prime 95B); a scan writes one
  every few seconds. What sending it over the lab's network costs is not
  measured.
- **Two tracks** (ADR-0010): the interfaces between them, the plugin API
  among them, are stand paths; models are trained and evaluated on the
  analysis side and used at the stand.

## Decision

### One interface, two places to run

- **A detector is reached through the `Detector` protocol only**, which
  moves from `smc.plugins.cells.detector` to a new package
  **`smc.inference`** (`smc.models` would read as the pydantic models every
  module has). It holds the protocol and `Detection`, the local
  implementation (`TorchDetector`), the **served implementation** (a client
  of `imaging-server-kit`), the catalogue of named models and the
  resolution of a name to a detector. A plugin asks for a model by name and
  receives a `Detector`; **it never knows where the model runs**.
- `smc.inference` is a stand path (an interface between the tracks,
  ADR-0010). It imports neither torch nor `imaging-server-kit` at module
  level; each is imported inside its implementation, behind its extra.
- **Trackers (#32) follow the same rules** (a name, an identity, two
  places, the same failure rules, the same transport), with a protocol of
  their own, written when #32 is planned. Selectors call no model: they
  read detections.

### A model has a name and an identity

- **A model is a card in the repository**: its name (`cell-classes-v2`),
  its weights' file name and SHA-256, its classes, the pixel size it was
  trained at (or `null`, said as unknown), its pre-processing. The weights
  stay out of the repository (`$SMC_MODELS`, the lab's share). The card's
  format is fixed by the plan of its issue.
- **Its identity is the weights' SHA-256 and its pre-processing**, so that a
  change of either is another model. A result records the name, that
  identity, and **where it ran** (`local:<device>` or `served`). It never
  records the server's address.
- **The server runs smc's own code.** It is started from this repository,
  under an extra `serve` (`detect` + `imaging-server-kit`): it loads the
  same cards and runs the same `TorchDetector`, so the pre-processing and
  the post-processing are the same in both places by construction. It
  answers each call with the model's identity, and the client **refuses an
  answer whose identity differs from the card's**. The client asks the
  server which models it serves, and checks them, before a run.
- **Who deploys**: the owner keeps the server (the machine, how it is
  started, its deployment note in `docs/hardware/`); a model is added by a
  pull request that adds its card, the collaborator for theirs. The cards'
  folder is listed for the analysis track in `CODEOWNERS`: a wrong card
  gives wrong detections, it moves no stage.

### Failures are findings; nothing switches silently

- **No fallback from one place or one model to another.** A run uses the
  model and the place that were asked for, or says why it could not; going
  local is an explicit choice of the user's, and the result records it.
- **Before a run**: the check (the server answers, serves the model, with
  the card's identity) is reported at the start of the stage. It does not
  refuse the scan: the stand's time is worth more than a detection that
  can be done later.
- **During an acquisition** (#635): a call never blocks the loop. A frame
  whose call fails, times out or comes back with another identity is
  named as missed; after a few failures in a row the detection stops for
  that run and says so, so that the loop does not pay a timeout on every
  frame. The scan ends `done`, and `cells` finishes the missed frames
  later.
- **After an acquisition** (`cells`): a server that does not answer ends the
  stage `failed` with its diagnosis; the checkpoint keeps the frames
  already done, and a resume goes on from them.
- **Every call has a timeout**, taken from the card or from the
  measurement of #635, and the time per frame, cold and warm apart, is in
  `assumed`.

### The network and the data

- **The address is a machine setting, never in the repository**: an
  environment variable, `SMC_MODEL_SERVER`, as `SMC_MODELS` is for the
  weights. The server is the lab's, not a stand's, so it is not in a
  profile either. When it is set, a model resolves to the served one; when
  it is not, to the local one, which needs the `detect` extra.
- **What leaves the PC is the frame's pixels as acquired, and its pixel
  size.** Never a path, a run document, a host or an account name. The
  server is reached on the lab's network only; if it ever needs a token,
  the token is a machine setting too.

### The local model stays, as an explicit choice

- **The `detect` extra stays**: for the collaborator's laptop (`smc run cells
  --offline`), for work away from the lab's network, and for a stand with
  no server. An acquisition PC with a server set needs no torch.

### Tests

- **The default suite never needs a server.** `smc.testing` gets a fake
  model server, built as `FakeMTB` is (knobs for absent, slow, failing, and
  answering with another identity; the calls recorded), and a stub
  detector replaces the tests' in-file stubs.
- **A contract suite**, `tests/contracts/test_detector.py`, runs on every
  implementation: the stub, `BlobDetector`, the served client against the
  fake server, and `TorchDetector` where torch is installed. A test against
  a real `imaging-server-kit` server runs where it is installed, outside
  the light set.

### What this record does not cover

Training and evaluating models (#342, #600's evaluation); how several
models' detections sit side by side in a run document and are reviewed
(#600's plan); the tracker's protocol (#32); the acquisition engine (#556);
the plate page served from a lab server (ADR-0006).

## Alternatives considered

- **Local only, as today.** No network, no new dependency; but torch and
  0.5 to 0.8 s of CPU a frame on the PC that drives the camera, and no GPU.
  Against the owner's direction.
- **Served only, the local model being a server on `localhost`.** One code
  path; but the laptop and every test that runs a model need a server
  process started beside them, and an offline run depends on it. The owner
  kept the local model (2026-11-25).
- **Fall back to the same weights locally when the server fails.** Loads
  torch on the acquisition PC exactly when the network is bad, and makes
  where a model ran depend on the network's weather. Rejected by the owner
  (2026-11-25).
- **Refuse the scan when the server does not answer.** The scan and the
  stand's time are worth more than a detection that `cells` can finish
  later.
- **Copy the sibling repository's server script.** Its pre-processing
  differs from `TorchDetector`'s, so the served model and the local one
  would not give the same detections on the same frame.
- **Our own HTTP service instead of `imaging-server-kit`.** No dependency
  at version 0.0.x and full control of the wire; but the lab already runs
  `imaging-server-kit` on its GPU machine for the tracker, and it is the
  owner's direction. The served implementation is one module: changing the
  transport changes that module only.
- **The address in the profile.** The repository is public, and the server
  is the lab's, shared by every stand.

## Consequences

- The acquisition PC no longer needs torch to detect; a scan can detect as
  it goes (#635), and several models can be named and compared (#600).
- **A result says which model and where**, so two results are comparable and
  a run is reproducible; a model changed under the same file name is
  noticed.
- **What becomes harder**: a second process to keep running, on a machine
  the owner maintains; a network call inside a tool; a dependency at
  version 0.0.x, pinned exactly and isolated in one module.
- **What we must remember**: the cost of sending a frame is not measured
  yet; the first issue measures it on the lab's network, cold and warm.
- **To build, each its own issue**, once this record is accepted:
  `smc.inference` with the protocol moved, the catalogue and the cards
  (`cells` takes a model by name, #600 item 1), the fake server and the
  contract suite; the served implementation, the `serve` extra and the
  server's entry point; the deployment note in `docs/hardware/` (which
  machine, which GPU, how it is started), with the first measurement of a
  call; the cards' folder in `CODEOWNERS`. #635 and #600 are then planned
  on it.
