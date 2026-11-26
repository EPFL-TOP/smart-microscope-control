# Inference design — models by name, one detector interface

- **Status**: design for issue #684 (2026-11-26), the first step of
  [ADR-0011](../adr/0011-where-the-models-run.md). #685 (the served detector)
  builds on §2 to §5; #635 (cells on the fly) and #600 (several models by
  name) are planned on top of it.
- **Owner**: the design session. **Executors**: `/develop` sessions.
- **Rule**: as in M1 to M5, names, module paths and signatures below are
  fixed; an executor who needs to change one stops and comments on its issue
  (`## Plan deviation`). §2 and §6 **amend m3 §9.1 and §9.3**: the detector
  module moves, and `cells` takes a model by name.

Read first: ADR-0011, then m3 §9 (the detector, the plugin, the measurement
of #342) and ADR-0010 (the two tracks: `smc.inference` is a stand path, its
cards' folder the analysis track's).

## 0. Scope

`smc.inference` receives the `Detector` protocol, `Detection`, the local
implementation (`TorchDetector`), the blob detector, the pre-processing, and
names models by **cards** in the repository. `cells` asks for a model by name,
and its result records the model's name, its **identity** and **where it
ran**. There is no server here: #685 adds the served implementation behind
the same `load_model` (§8).

Out of scope: the served client, the `serve` extra, `SMC_MODEL_SERVER`
(#685); detection during a scan (#635); several models' detections side by
side, the page's choice of a model, the evaluation (#600); the tracker's
protocol (#32); a second architecture than Faster R-CNN (§9).

## 1. Measured (2026-11-26, planning #684)

On the main checkout's `.venv` (Python 3.13.0, pydantic 2.13.5) and, for
torch, `../nikon-control/.venv` (torch 2.13.0, torchvision 0.28.0, Apple M5
Pro, CPU) with `PYTHONPATH=src`:

- **TOML has no null**: `tomllib.loads("x = null")` raises
  `TOMLDecodeError` (*Invalid value*). A card says *unknown* and *native* in
  words (§3).
- `cell_classes_model-cat-v2.pth` in the main checkout's `local/models` has
  the SHA-256 `b1160edae7dd2e8e1897f959fd55b77481bbfa92d236aaa1202cd94101b2cb7e`,
  and every one of the 14 `model_sha256` that run documents under `local/`
  record (stand runs included) is that value.
- pydantic's lax `float` takes `true` as `1.0`. pydantic's `StrictFloat`
  (`Annotated[float, Strict()]`) takes an `int` (as `1.0`), refuses a `bool`
  and a `str`, and a `tuple[...]` of it takes a TOML array (a `list`); a
  model-wide `strict=True` refuses the list. A field validator in
  `mode="before"` runs first, so the words of §3 are mapped before the strict
  check.
- The canonical JSON of §4 for `cell-classes-v2`'s pre-processing is
  `{"model_um_per_px":null,"percentiles":[0.5,99.5],"version":1}`; an `int` in
  the card is a float there (`[1, 99]` and `1` give `[1.0,99.0]` and `1.0`).
- A random-init `fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None,
  num_classes=4)` saved with `torch.save` is a 166 MB file; built, saved,
  loaded by `TorchDetector` and run twice on a 120 × 160 frame in 1.6 s. It
  returns 100 boxes on noise and on a frame of zeros (so "nothing on a blank
  frame" is not a contract), every box inside `[0, W] × [0, H]` with a label
  of its classes and a score in `[0, 1]`, and the same boxes on the same frame
  twice.
- `BlobDetector` on that frame (noise `N(1000, 30)`, seed 0, three discs of
  radius 4 px at 3000 counts centred at rows/columns (30, 40), (60, 100),
  (95, 140)) finds exactly the three as `single`, in `uint16` and in
  `float32`.
- The torch venv runs `tests/unit/test_cells_detector.py` from this checkout
  (87 passed; the subprocess test of the entry point fails there because smc
  is not installed in that venv, which is the venv's, not the code's).

## 2. The package (amends m3 §9.1)

```text
src/smc/inference/
  __init__.py       the public names below; imports neither torch nor skimage
  errors.py         ModelError
  detector.py       Detection, Detector                (moved, unchanged)
  preprocessing.py  Preprocessing, normalize_plane, model_input_size_px,
                    MIN_INPUT_PX, MAX_INPUT_PX          (moved, unchanged)
                    ModelPreprocessing, PREPROCESSING_VERSION,
                    preprocessing_sha256, preprocessing_for      (new)
  local.py          TorchDetector, detections_from_output       (moved)
                    load_local                (was load_detector's body)
  blobs.py          BlobDetector, BLOB_DETECTOR,
                    BLOB_DETECTOR_IDENTITY                       (moved)
                    blob_model                                   (new)
  cards.py          ModelCard, DEFAULT_MODEL_NAME, model_card,
                    model_names, model_cards, parse_card         (new)
  models.py         ResolvedModel, model_identity, weights_folder,
                    file_sha256, load_model                      (new)
  cards/            one TOML file per model: cell-classes-v2.toml
```

- **`smc/plugins/cells/detector.py` is deleted**; nothing stays importable
  from it. Its importers are this repository's own (`cells`' `algorithm.py`
  and `plugin.py`, `scripts/dev/cells_check.py`, five test modules); a
  re-export would keep two names for one thing.
- **No torch at module level**, anywhere in the package: torch and
  torchvision are imported inside `TorchDetector.__init__`, as today.
  `__init__.py` re-exports `Detection`, `Detector`, `ModelError`, `ModelCard`,
  `ModelPreprocessing`, `Preprocessing`, `ResolvedModel`,
  `DEFAULT_MODEL_NAME`, `model_card`, `model_names`, `model_cards`,
  `load_model`, `model_identity` and `weights_folder`; `TorchDetector`,
  `BlobDetector` and `blob_model` are imported from their modules, so that
  `import smc.inference` loads neither torch nor scikit-image.
- **`smc.inference` imports nothing of `smc.plugins`, `smc.hardware`,
  `smc.testing` or `smc.ui`**: it is part of the collaborator's kit
  (algorithms §8), and `cells` depends on it, not the reverse. Its errors are
  `ModelError`; `cells` turns one into `PluginRunError(cells, str(exc))`, so
  the messages a user reads are the ones `load_detector` gave.

```python
class ModelError(Exception):
    """A model that cannot be used here: the message says why and what to do."""
```

`Detection`, `Detector`, `Preprocessing`, `normalize_plane`,
`model_input_size_px`, `TorchDetector`, `detections_from_output` and
`BlobDetector` keep m3 §9.1 and §9.4 word for word.
`load_local(weights, *, device, score_threshold, preprocessing) ->
TorchDetector` is `load_detector` without the path's defaulting and look-up
(§5 does them): it builds `TorchDetector` and maps its failures as
`load_detector` did (`ImportError`: "install the detect extra: pip install
-e .[detect]"; `OSError` or `RuntimeError`: "torch is installed but did not
load …"; `ValueError`: its message), raising `ModelError`.

## 3. The card

A model is a TOML file in `src/smc/inference/cards/`, named `<name>.toml`,
shipped in the package (hatch includes it, as `plates.toml`). The folder is
the **analysis track's** in `.github/CODEOWNERS` (ADR-0011): a model is added
by a pull request that adds its card.

```toml
# src/smc/inference/cards/cell-classes-v2.toml
# The single / doublet / debris Faster R-CNN that nikon-control trained (m3 §9.4).
name = "cell-classes-v2"
weights = "cell_classes_model-cat-v2.pth"
weights_sha256 = "b1160edae7dd2e8e1897f959fd55b77481bbfa92d236aaa1202cd94101b2cb7e"
classes = ["single", "doublet", "debris"]
# The checkpoint records no pixel size, and nikon-control's one ND2 (a Prime 95B
# frame at 0.27555 um) does not say the training frames came from that camera.
trained_um_per_px = "unknown"

[preprocessing]
percentiles = [0.5, 99.5]
model_um_per_px = "native"
```

```python
class ModelPreprocessing(BaseModel):
    """What is done to a frame before the model sees it, as a rule (the card's table)."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, allow_inf_nan=False, protected_namespaces=()
    )

    percentiles: tuple[StrictFloat, StrictFloat]
    model_um_per_px: StrictFloat | None
```

```python
class ModelCard(BaseModel):
    """A model by name: its weights, its classes, its training pixel size, its pre-processing."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    name: str
    weights: str
    weights_sha256: str
    classes: tuple[str, ...]
    trained_um_per_px: StrictFloat | None
    preprocessing: ModelPreprocessing
```

`StrictFloat` is pydantic's (§1). Every key is required: a card that leaves
one out is refused, so nothing is defaulted silently.

- `name`: `[a-z0-9]+([.-][a-z0-9]+)*`, at most 64 characters, not `blobs`
  (the blob detector's name in a result, §6), and **equal to the file's
  stem**.
- `weights`: a file name, never a path: refused when empty, `.`, `..`, or
  containing `/`, `\` or `:`. It is looked for in `weights_folder()` (§5).
- `weights_sha256`: 64 lowercase hexadecimal characters.
- `classes`: at least one, distinct, non-empty, none `__background__`; the
  order is the checkpoint's after the background, which §5 checks.
- `trained_um_per_px`: a pixel size in µm, `0 < x <= 10`, or the word
  `unknown`, read as `None`. It changes nothing in a detection today (§9).
- `preprocessing.percentiles`: `0 <= low < 50 < high <= 100`, the bounds of
  `cells`' `low_percentile` and `high_percentile`, so that a parameter that
  replaces one of them always leaves a valid pair (§6).
- `preprocessing.model_um_per_px`: the pixel size the frame is shown to the
  model at, `0 < x <= 10`, or the word `native`, read as `None`: the model's
  own resize (torchvision's 800 and 1333 for Faster R-CNN).
- Any other word for these two, a `bool`, or a string where a number goes, is
  refused with the key's name.

```python
DEFAULT_MODEL_NAME = "cell-classes-v2"


def model_names() -> tuple[str, ...]: ...
def model_card(name: str) -> ModelCard: ...
def model_cards() -> dict[str, ModelCard]: ...
def parse_card(data: bytes, *, source: str) -> ModelCard: ...
```

- `model_names()`: the stems of the folder's `*.toml`, sorted; the folder is
  `importlib.resources.files("smc.inference") / "cards"`. Other files are not
  cards.
- `model_card(name)`: a name that does not match the pattern, or has no
  file, is a `ModelError` **before any file is opened**: `no model named
  {name!r}; the models are: cell-classes-v2 (cards in
  src/smc/inference/cards/)`. Otherwise it reads that one file, so a bad card
  does not stop the others.
- `parse_card(data, source=...)`: decodes `utf-8-sig` (a byte-order mark is
  accepted, FM-80), parses TOML, validates, and checks the name against
  `source`'s stem. Every failure is a `ModelError` that starts with `source`
  (`cards/<name>.toml`, never an absolute path, FM-50) and gives the first
  problem as `<key>: <message>`; a file that is not UTF-8 says to save it as
  UTF-8.
- `model_cards()`: every card by name, raising on the first bad one; the
  repository test reads every card through it.

## 4. The identity

A model's identity is the weights' SHA-256 and a digest of its
pre-processing **as the run applies it** (the card's, with any parameter of
§6 in place), so that a change of either is another model (ADR-0011).

```python
PREPROCESSING_VERSION = 1


def preprocessing_sha256(rule: ModelPreprocessing | None) -> str: ...
def model_identity(weights_sha256: str, rule: ModelPreprocessing | None) -> str: ...
```

- The canonical text is `json.dumps(payload, sort_keys=True,
  separators=(",", ":"), allow_nan=False)` of `{"version":
  PREPROCESSING_VERSION, "percentiles": [low, high], "model_um_per_px": x}`
  (`null` for native), and of `None` for a detector with no pre-processing
  (`null`); the digest is the SHA-256 of its ASCII bytes. The values are the
  model's floats, so `1` and `1.0` give one text (§1).
- **`PREPROCESSING_VERSION` stands for what the code does that the rule does
  not say**: the float32 scaling of `normalize_plane`, the plane repeated to
  three channels. A change of either increments it.
- `model_identity` is `f"{weights_sha256}+{preprocessing_sha256(rule)}"`: the
  two halves stay readable, and a resume can say which one changed (§6).
- Pinned (the tests copy these, computed 2026-11-26):

| Rule | `preprocessing_sha256` |
|---|---|
| `cell-classes-v2`'s, `(0.5, 99.5)`, native | `2df8feb4070d6d7b527502090c9f1be7003a1b50b7ee09390b81b8413a6cfff1` |
| `(1, 99)` at `1` µm (ints in the card) | `6b31dfc76ff951636112c94fe6e220013f8e9f547d98929e93304b00e0e16417` |
| `None` (the blob detector) | `74234e98afe7498fb5daf1f36ac2d78acc339464f950703b8c019892f982b90b` |

## 5. From a name to a detector

```python
@dataclass(frozen=True, slots=True)
class ResolvedModel:
    """A detector with what a result says of it: its name, its identity, where it runs."""

    name: str
    identity: str
    weights_sha256: str
    runs_on: str
    detector: Detector
```

```python
def weights_folder() -> Path: ...
def file_sha256(path: Path) -> str: ...
def load_model(
    card: ModelCard,
    *,
    preprocessing: ModelPreprocessing,
    frame_shape_px: tuple[int, int],
    frame_um_per_px: float,
    device: str,
    score_threshold: float,
    weights: Path | None = None,
) -> ResolvedModel: ...
def preprocessing_for(
    rule: ModelPreprocessing,
    *,
    frame_shape_px: tuple[int, int],
    frame_um_per_px: float,
) -> Preprocessing: ...
def blob_model() -> ResolvedModel: ...
```

- `weights_folder()`: `$SMC_MODELS` when it is set and not blank (FM-78),
  else `<checkout>/local/models` with the checkout
  `Path(smc.__file__).resolve().parents[2]`: `default_model_path`'s rule,
  without the file name.
- `file_sha256(path)`: read in 1 MiB pieces; an `OSError` propagates.
- `preprocessing_for` (in `preprocessing.py`):
  `Preprocessing(percentiles=rule.percentiles, input_size_px=None)` for a
  native rule, else with
  `model_input_size_px(frame_shape_px, frame_um_per_px, rule.model_um_per_px)`
  (its `ValueError` propagates).
- `load_model`, in this order, each failure a `ModelError`:
  1. `preprocessing_for(...)`; a `ValueError` is `model_um_per_px={x:g}
     cannot be used: {exc}`, before any file is looked for (a refused size
     does not wait for 166 MB).
  2. The path: `weights` if given, else `weights_folder() / card.weights`.
     `is_file()` raising `OSError` (a share that dropped, FM-116) is `cannot
     look for the model at {path} ({strerror}); set SMC_MODELS to the folder
     that holds {card.weights}, or give model_path`; no file is `no model at
     {path}; put {card.weights} there, set SMC_MODELS to its folder, or give
     model_path` (m3 §9.1's two messages, the card's file name in them).
  3. `file_sha256(path)`; an `OSError` is `{path.name} cannot be read to hash
     it ({strerror}); is another program using it?`.
  4. **A SHA-256 other than the card's** is `{path.name} has the SHA-256
     {sha}, and the card {card.name} names {card.weights_sha256}: it is not
     the model the card describes; put the card's file there, or add a card
     for this one`. torch is never imported for it.
  5. `load_local(path, device=..., score_threshold=..., preprocessing=...)`.
  6. **Classes other than the card's** (in order) is `{path.name} has the
     classes [...], and the card {card.name} names [...]`.
  7. `ResolvedModel(name=card.name, identity=model_identity(sha,
     preprocessing), weights_sha256=sha, runs_on=f"local:{detector.device}",
     detector=detector)`.
- `blob_model()`: `BlobDetector()`, name `BLOB_DETECTOR` (`blobs`),
  `weights_sha256` the SHA-256 of `BLOB_DETECTOR_IDENTITY` (UTF-8), identity
  `model_identity(that, None)`, `runs_on` `local:cpu`. It reads
  `BLOB_DETECTOR_IDENTITY` at call time.

## 6. `cells` takes a model by name (amends m3 §9.3)

**Parameters.** `CellsParams` gains `model`, and three parameters become
"unset is the card's":

```python
class CellsParams(PluginParams):  # changes only
    model: str | None = None  # a card's name; None is DEFAULT_MODEL_NAME
    model_path: Path | None = None  # the weights, instead of the card's file
    low_percentile: float | None = Field(None, ge=0.0, lt=50.0)
    high_percentile: float | None = Field(None, gt=50.0, le=100.0)
    model_um_per_px: float | None = Field(None, gt=0.0, le=10.0)
```

- `model`: unset is `cell-classes-v2`. Its description names the cards'
  folder. An unknown name is refused when the stage runs, by `model_card`'s
  message (it lists the names), not when the parameters are parsed.
- `model_path`: the weights file, in place of `card.weights` in
  `weights_folder()`; its SHA-256 must be the card's (§5 step 4).
- `low_percentile`, `high_percentile`, `model_um_per_px`: unset is the card's
  value; set, each replaces **its own part** of the card's pre-processing.
  The rule the run applies is

```python
low = params.low_percentile
high = params.high_percentile
size = params.model_um_per_px
rule = ModelPreprocessing(
    percentiles=(
        card.preprocessing.percentiles[0] if low is None else low,
        card.preprocessing.percentiles[1] if high is None else high,
    ),
    model_um_per_px=card.preprocessing.model_um_per_px if size is None else size,
)
```

A native card with `model_um_per_px` unset stays native; asking for native
over a card that names a size is not possible here, and no card names one.

- With `detector=blobs` the validator refuses `model` that is set (`model is
  for detector=model: the blob detector is not a named model; remove model,
  or use detector=model`), as it refuses `model_path`, `model_um_per_px` and
  `checked_um_per_px`; it refuses a percentile that is set **and** differs
  from 0.5 (99.5): a run document recorded before #684 holds `0.5` and
  `99.5` for a blobs run, and a page that re-validates it must not fail.

**The run.** Step 2 of m3 §9.3 becomes: `detector=blobs` is `blob_model()`;
otherwise `card = model_card(params.model or DEFAULT_MODEL_NAME)`, the rule
above, and `load_model(card, preprocessing=rule,
frame_shape_px=scan.image_shape_px, frame_um_per_px=scan.pixel_size_um,
device=params.device, score_threshold=params.score_threshold,
weights=params.model_path)`, both called through the plugin module's names
(tests replace `load_model` with `monkeypatch.setattr` on
`smc.plugins.cells.plugin`). A `ModelError` is `PluginRunError("cells",
str(exc))`. The plugin no longer hashes anything.

**The checkpoint.** `CellsCheckpoint.model_sha256` is replaced by
`identity: str`. A checkpoint whose identity differs is refused, naming the
half that changed: the weights half, `the weights changed since the run
being resumed; run cells without --resume` (m3 §9.3's message); else `the
pre-processing changed since the run being resumed; run cells without
--resume`. A checkpoint saved before #684 has no `identity`, and
`ctx.checkpoint` refuses it (`ResumeError`: "run it again without
--resume"); a resume across the upgrade is refused by the runner anyway,
since the parameters' dump changed.

**The result.** `CellsResult` gains two fields **with a default**, so that
every run document and dataset recorded before #684 still loads:

```python
class CellsResult(PluginResult):  # gains
    model_identity: str | None = None
    model_runs_on: str | None = None
```

- `model` is the card's **name** (`cell-classes-v2`), or `blobs`; it was the
  weights' file name. `model_sha256` keeps its meaning (the weights', or the
  blob identity string's). `model_identity` is `ResolvedModel.identity`,
  `model_runs_on` its `runs_on` (`local:cpu`). `None` in both means a result
  recorded before #684. The result never records a path or an address.
- `device`, `classes`, `assumed` (m3 §9.4's line about `checked_um_per_px`
  included) do not change. The report's *Detector* line is unchanged and now
  reads the name.

## 7. Tests

- **`smc.testing.StubDetector`** (`src/smc/testing/detectors.py`, exported by
  `smc.testing`) replaces `_Stub` of `tests/test_cells_demo.py` and
  `_Scripted` of `tests/test_dev_cells_check.py`:

```python
class StubDetector:
    """Returns what a test scripted for the frame it is given, keyed by its first pixel."""

    def __init__(
        self,
        scripted: Mapping[int, Sequence[Detection]] | None = None,
        *,
        default: Sequence[Detection] = (),
        classes: tuple[str, ...] = ("single", "doublet", "debris"),
        device: str = "cpu",
        on_frame: Callable[[int], None] | None = None,
    ) -> None: ...
```

`detect` refuses a frame that is not 2-D with `ValueError` (`detect takes
one 2-D frame, got an array of shape …`), takes `int(frame[0, 0])` as the
key, appends it to `seen`, calls `on_frame(key)`, and returns a new list of
`scripted.get(key, default)`. It does not check its labels against its
classes: `cells`' refusal of an undeclared label is tested with it.
`stub_model(detector, *, name="stub", weights_sha256="0" * 64,
preprocessing=None) -> ResolvedModel` wraps one with
`model_identity(weights_sha256, preprocessing)` and `runs_on`
`f"local:{detector.device}"`, so that a stand-in for `load_model` computes
the identity from the rule it is given (FM-260).

- **The contract**, `tests/contracts/test_detector.py`: every case runs on
  the stub, on `BlobDetector`, and on `TorchDetector` around a random-init
  Faster R-CNN saved once per module (skipped without torch and torchvision,
  "the detect extra is not installed"; §1: 166 MB, 1.6 s). On §1's 120 × 160
  frame (not square, FM-73): it is a `Detector`, its `classes` are distinct
  non-empty strings without `__background__`, its `device` a non-empty string
  other than `auto`; it finds at least one box there, each inside `[0, W] ×
  [0, H]` with `row0 <= row1`, `col0 <= col1`, finite, a label of its classes
  and a score in `[0, 1]`; it refuses a 3-D and a 1-D frame with a
  `ValueError` naming 2-D; the same frame twice gives the same detections; it
  leaves the frame as it was; it takes the frame as `uint16` and as
  `float32`. #685 adds the served client to the cases.
- **The kit**: `tests/test_kit_imports.py` covers `smc.inference` (no
  `smc.hardware`, `smc.testing`, `smc.ui`, `pymmcore`), and a subprocess
  imports every module of the package with `torch` and `torchvision` blocked.
- The default suite needs neither torch nor the weights (FM-40). The tests
  of the trained model and of the card against the real file are skipped
  without them, and are run by hand with the torch venv and `SMC_MODELS`
  (§1).

## 8. What #685 adds, for the record

`load_model` resolves to the served implementation when `SMC_MODEL_SERVER`
is set (ADR-0011), with `runs_on` `served`; the server process reads the same
cards (`model_cards()`) and runs `load_model` locally, so the pre-processing
is the same by construction, and answers with `model_identity`, which the
client compares with `model_identity(card.weights_sha256, rule)`. The fake
model server joins `smc.testing` and the contract suite. Nothing of §2 to §6
changes for it but `load_model`'s first branch.

## 9. Not decided here

- **A second architecture.** Every card is a Faster R-CNN read by
  `TorchDetector`; the first card of another kind adds a key `architecture`
  with the default `fasterrcnn_resnet50_fpn`, in its own design change.
- **What the trained pixel size changes.** It is recorded and shown as
  unknown; using it as the default of `model_um_per_px`, or saying in
  `assumed` that the scale the model sees is not known to match, waits for a
  card that knows it (#342, #600).
- **Cards outside the repository** (a lab share, `$SMC_MODELS`): ADR-0011 puts
  them in the repository; a model is added by a pull request.
- The page's choice of a model (#600).
