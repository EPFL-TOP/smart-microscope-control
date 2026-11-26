# Inference design — models by name, one detector interface

- **Status**: design for issue #684 (2026-11-26), the first step of
  [ADR-0011](../adr/0011-where-the-models-run.md), and for #685 (the served
  detector, §8), and for #635 (detection while the scan goes, §10). #600
  (several models by name) is planned on top of them.
- **Owner**: the design session. **Executors**: `/develop` sessions.
- **Rule**: as in M1 to M5, names, module paths and signatures below are
  fixed; an executor who needs to change one stops and comments on its issue
  (`## Plan deviation`). §2 and §6 **amend m3 §9.1 and §9.3**: the detector
  module moves, and `cells` takes a model by name. §10.4 amends m3 §9.3
  again: `cells` uses the detections a worker made while the scan went.

Read first: ADR-0011, then m3 §9 (the detector, the plugin, the measurement
of #342) and ADR-0010 (the two tracks: `smc.inference` is a stand path, its
cards' folder the analysis track's).

## 0. Scope

`smc.inference` receives the `Detector` protocol, `Detection`, the local
implementation (`TorchDetector`), the blob detector, the pre-processing, and
names models by **cards** in the repository. `cells` asks for a model by name,
and its result records the model's name, its **identity** and **where it
ran**. There is no server in §2 to §7: #685 adds the served implementation
behind the same `load_model` (§8).

Out of scope: the served client, the `serve` extra, `SMC_MODEL_SERVER`
(#685, §8); detection during a scan (#635, §10); several models' detections side by
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
- `load_model`, in this order, each failure a `ModelError` (#685 renames this
  body `load_local_model` and makes `load_model` the dispatcher of §8.5):
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

## 8. The served detector (#685)

`load_model` resolves to the served implementation when `SMC_MODEL_SERVER` is
set (ADR-0011), with `runs_on` `served`. The server runs this repository's own
code: it reads the same cards and runs the same `TorchDetector`, so the
pre-processing is the same by construction, and it answers with the model's
identity, which the client compares with `model_identity(card.weights_sha256,
rule)`. Nothing of §2 to §6 changes but `load_model`'s first branch (§8.5), the
checkpoint's one new field and what `cells` says of a served run (§8.7).

### 8.1 Measured (2026-11-26, planning #685)

On the main checkout's `.venv` (Python 3.13.0, macOS arm64) and a scratch venv
with `imaging-server-kit==0.0.12` (fastapi 0.165.0, starlette 1.10.0, uvicorn
0.54.0, tifffile 2026.11.16, numpy 2.5.5, pydantic 2.14.4); the client part also
on Python 3.10.10:

- **Versions.** PyPI has 0.0.1 to 0.0.16, 0.1.0 to 0.1.3, 0.2.0 and 0.3.x.
  0.0.12 (`Requires-Python >=3.9`) is what the lab runs
  (`lightsheet-live-tracking-tool/requirements.txt`). 0.0.13 to 0.0.16 are `>=3.9`,
  and the `client/client.py` of each has no `timeout`. 0.1.3 (`>=3.9`) is another
  API (`AlgorithmRunner`, msgpack on the wire) whose client has one timeout, fixed at
  3600 s (`TIMEOUT_SEC`); 0.3.2 needs Python 3.11 and is yet another API
  (`core/algorithm.py`, `remote/client.py`), so not installable on smc's minimum, 3.10.
  None is a drop-in for the lab's server.
- **The kit's `Client` has no timeout.** Its constructor calls
  `requests.get(f"{url}/services")` and `run_algorithm` calls `requests.post(...)`,
  neither with `timeout=`, and the class takes no such argument (source of
  0.0.12): a host that drops packets holds the caller until the OS gives up.
- **`/services` lists algorithm names only**, no identity:
  `{"services": ["models", "detect"]}`.
- **The wire** (real server): `POST /<algorithm>/process` with a JSON body of the
  parameters, an image as the base64 of a TIFF (`tifffile`); the answer is 201
  and a JSON list of `{"type", "data", "data_params"}`; a result of type `text`
  carries a string. A 1412 × 1412 `uint16` frame is a 3 987 744-byte TIFF and a
  5 316 992-byte JSON field, encoded in 4 to 8 ms, and the function receives it as
  the same `uint16` array (first pixel included). A 4096 × 4096 frame (a 44.7 MB
  body) went through the default server in 0.09 s on loopback: no body limit.
  An algorithm declared with `parameters={}` takes `{}`.
- **The errors**: a function that raises is 500
  `{"error_type": "internal_server_error", "message": str(exc)}`; a parameter
  that fails validation (`ge`, `extra="forbid"`, missing) is 422 `{"detail": [...]}`;
  an unknown algorithm is 404 `{"detail": ...}`; an image that is not a TIFF is a
  500 (the kit's 422 handler fails to serialise the exception). Every body is JSON.
- **`MultiAlgorithmServer.lifespan` talks to an "algorithm hub"**:
  `requests.get("http://algorithm_hub:8000/")` at start, with no timeout, and
  `requests.post(.../deregister)` at shutdown, unguarded: with no hub, shutdown
  logs `Application shutdown failed`. Setting `server.app.router.lifespan_context`
  to a no-op async context manager after construction removes both.
- **Importing the kit is heavy**: `import imaging_server_kit` takes 6.4 s cold and
  0.30 s warm and loads `cv2`, `scipy`, `skimage`, `fastapi`, `starlette`,
  `imantics` and `requests` (1041 modules); its metadata installs
  `opencv-python` and `opencv-contrib-python-headless` together. That is not
  for the acquisition PC, which holds the stand's DLLs (FM-95).
- **`http.client`** (Python 3.13 and 3.10.10), a connect timeout and then
  `sock.settimeout` for the rest: a server that does not answer raises
  `TimeoutError` (3.10: the alias of `socket.timeout`), a closed port
  `ConnectionRefusedError`, a server that closes the connection
  `RemoteDisconnected`, an unroutable address `TimeoutError` from `connect`. It
  never uses a proxy (FM-152).
- **Not measured**: the server on Python 3.10 and 3.14 (the only 3.10 of this Mac
  fails to load scipy's own binary, a `dlopen` error unrelated to the kit), on
  Windows, and on a GPU.

### 8.2 The version, the extra and the modules

- **`imaging-server-kit==0.0.12`**, exactly: the lab's, `>=3.9`, its wire read
  here. Moving the pin is a design change and a run of the real-server test (§8.9).
- **`serve = ["torch>=2.2", "torchvision>=0.17", "imaging-server-kit==0.0.12"]`**:
  the `detect` extra and the kit, written out, with a test that its torch lines
  are `detect`'s. It is the **server's** extra; the acquisition PC installs
  nothing for a served model (the client is the standard library, numpy and
  tifffile).
- **The client does not import the kit** (§8.1: no timeout, and the import cost).
  It speaks the kit's `/process` wire itself, in `served.py`; the kit is imported
  **only in `server.py`, inside functions**, and a test walks the source for it.
  This departs from ADR-0011's "one module" and #685's "a client of
  `imaging-server-kit`": the transport is the kit's, in two modules and the fake
  that share the wire's constants and codecs; changing the transport changes those.

```text
src/smc/inference/
  served.py     the wire and the client: constants, ServedError, ServedModel, ServedAnswer,
                encode_frame, encode_listing/decode_listing, encode_answer/decode_answer,
                ModelWire, HttpWire, open_wire, ServedCheck, check_served,
                ServedDetector, load_served        (stdlib, numpy, tifffile; no kit, no torch)
  server.py     ModelServer, prepare, build_app, serve                     (the kit, inside functions)
  models.py     load_local_model (what load_model was) and load_model (the dispatcher, §8.5)
src/smc/testing/fake_model_server.py   FakeModelServer, ServedCall      (exported by smc.testing)
```

`smc.inference.__init__` does not change (`served` and `server` are imported from
their modules, as `local` is). Neither module imports torch, scikit-image, the kit,
uvicorn or fastapi at module level.

### 8.3 The wire

Two algorithms, as the kit names them, on one `MultiAlgorithmServer`
(`server_name="smc-models"`):

```text
POST /models/process   {}                                                    -> 201 [text: LISTING]
POST /detect/process   {"model": str, "image": <base64 TIFF>, "pixel_size_um": float}  -> 201 [text: ANSWER]
```

```json
{"schema": 1, "models": [{"name": "cell-classes-v2", "identity": "<sha>+<sha>",
                          "classes": ["single", "doublet", "debris"], "device": "cuda"}]}
{"schema": 1, "model": "cell-classes-v2", "identity": "<sha>+<sha>", "device": "cuda",
 "detections": [[row0, col0, row1, col1, "single", 0.93], ...]}
```

- A result is `{"type": "text", "data": <the JSON above as a string>, "data_params":
  {"name": "models" | "detections"}}` (`data_params` must be a dict). The client
  reads the first result of type `text` whose `data` is a `str`.
- The JSON is `json.dumps(..., allow_nan=False)`; a float is Python's `repr`, so a
  detection crosses the wire exactly. A box is `(row0, col0, row1, col1)` in the
  frame's pixels, as `Detection`. `pixel_size_um` has `ge=0`.
- **What is sent: these three keys and nothing else** (the kit's parameter model is
  `extra="forbid"`). The frame is `tifffile.imwrite` of the array as it is (dtype
  and values kept), base64. Never a path, a run document, a host, an account, or
  the score threshold.
- **The score threshold stays on the client.** The server runs its detectors with
  `SERVER_SCORE_FLOOR = 0.0` (torchvision keeps the scores above 0.05, which is no
  higher than any `cells` threshold, `ge=0.05`), and the client keeps `score >= threshold`, the
  predicate of `detections_from_output`: a served run and a local one on the same
  weights give the same detections.
- **A pixel size of 0 is sent as 0.0 and means unknown**, never defaulted; a
  negative or non-finite one is refused by the client before anything is sent. A
  native rule (the only kind the server serves, §8.6) does not read it; it is
  carried so that a sized rule can be served without changing the wire.

### 8.4 The error

```python
class ServedError(ModelError):
    """A call to the model server that did not give a usable answer; `kind` says which."""

    kind: Literal["unreachable", "timeout", "error", "identity", "malformed"]
```

- `unreachable`: no connection (refused, unroutable, no name, no answer within
  `CONNECT_TIMEOUT_S`), or the connection broke mid-call (`RemoteDisconnected`).
  `timeout`: connected, and no answer within the call's read timeout. `error`: the
  server answered a status other than 201. `identity`: the answer is for another
  model than the one asked for, or carries another identity. `malformed`: not
  JSON, another `schema`, a field of the wrong type, a number that is not
  finite, more than `MAX_ANSWER_BYTES`.
- **No message holds the address, the host, the port or the URL**: a failed
  stage's diagnosis is kept in the run folder (FM-50). A message is built from the
  kind, the status, `strerror` and the server's own text, never from `str(exc)` of a
  library that puts the URL in it, and **never from the address**. The server's text
  (`message` or `detail`) is cut to its first line and 200 characters, and says
  `the server said: ...`.
- `ServedError` is a `ModelError`, so `cells` and `cells_check` already turn it into
  a diagnosis. #635 reads `kind` to count a failure in a row.

### 8.5 The client

```python
#: http://host:port; blank is unset (FM-78).
SERVER_ENV = "SMC_MODEL_SERVER"
#: The read timeout of a detect call, in seconds; blank is unset.
TIMEOUT_ENV = "SMC_MODEL_TIMEOUT_S"
#: Assumed: a LAN host that is up answers at once.
CONNECT_TIMEOUT_S = 3.0
#: Assumed: the listing is a few hundred bytes.
CHECK_TIMEOUT_S = 10.0
#: Assumed: 0.5 to 0.8 s a frame on a CPU, and a cold first call loads kernels.
DETECT_TIMEOUT_S = 60.0
MAX_ANSWER_BYTES = 4 * 1024 * 1024
```

```python
class ModelWire(Protocol):
    def post(
        self, path: str, body: Mapping[str, object], *, read_timeout_s: float
    ) -> object:
        """The parsed JSON of a 201 answer; a ServedError for anything else."""


def open_wire(address: str) -> ModelWire:
    """HttpWire(address): the seam a test replaces."""
```

- **`HttpWire(address)`** is `http.client` and nothing else: `address` is `http://host:port` or
  `https://host:port`, no user, no path, an explicit port (otherwise `ModelError`:
  `SMC_MODEL_SERVER is not an address like http://host:8000`, **without the value**). One
  connection per call: `HTTPConnection(host, port, timeout=CONNECT_TIMEOUT_S)`, `connect()`,
  then `sock.settimeout(read_timeout_s)`, `request("POST", ...)` with `Content-Type` and
  `Accept` of `application/json`, `getresponse()`, `read(MAX_ANSWER_BYTES + 1)`, `close()`
  in a `finally`. The read timeout is per socket operation: it bounds the call because the
  server writes its answer, a few kB, in one piece after it has computed it. **No proxy,
  ever** (FM-152): the server is reached on the lab's network only (ADR-0011). Its
  `__repr__` does not show the address.
- `ServedModel(name, identity, classes, device)` and `ServedAnswer(model, identity, device,
  detections)` are frozen dataclasses; `decode_listing(text)` and `decode_answer(text)` raise
  `ServedError("malformed", ...)` and validate: `schema == 1`, the types, every number finite
  (a `bool` is not a number), a detection a list of six with a `str` label.
- **`check_served(wire, card, *, timeout_s=CHECK_TIMEOUT_S) -> ServedCheck`** never raises:

```python
@dataclass(frozen=True, slots=True)
class ServedCheck:
    """What the check found; `ok` says whether a call can be expected to work."""

    ok: bool
    kind: str | None
    message: str
    device: str | None
```

  `kind` is `None` when `ok`, else a `ServedError` kind or `"not_served"` (the server answered
  and does not serve the card's name); `device` is what the server said, when it answered.
  It posts `/models/process`, finds `card.name` and compares the listed identity with
  `model_identity(card.weights_sha256, card.preprocessing)`. The messages (the tests assert
  the words in italics): ok: `model server: cell-classes-v2 is served on cuda, identity
  b1160edae7dd+2df8feb4070d` (the first 12 characters of each half); *does not answer* /
  *did not answer in N s* for `unreachable` / `timeout`; `does not serve cell-classes-v2 (it
  serves: a, b)` for `not_served`, `(it serves no model)` for an empty list; *another identity*
  naming which half differs (`the weights differ`, `the pre-processing differs`, or both)
  for `identity`.

- **`ServedDetector`** (`classes` is the card's; `device` is the last answer's, else the
  check's, else `"unknown"`, never `auto`; `check` is the `ServedCheck` of its construction):

  `detect(frame)` refuses a frame that is not 2-D with `ValueError` (`detect takes one 2-D
  frame, got an array of shape ...`) before anything is sent; posts `/detect/process` with
  the three keys, `read_timeout_s=` its own; decodes; refuses an answer whose `model` is not
  the one asked for or whose `identity` is not the expected one (`ServedError("identity")`:
  `the server answered with another model identity: ...`, the halves told apart); keeps
  `score >= score_threshold`; sets `device`; returns a new list. It retries nothing and
  never answers an empty list for a failure ("failures are findings").
- **`load_model` dispatches** (the first branch): `load_local_model(...)` is the body that
  `load_model` had; `load_model` first does what both places need (§5 step 1,
  `preprocessing_for`, in a helper both call), then `load_served(...)` when `SERVER_ENV` is
  set and not blank, else `load_local_model(...)`. There is **no fallback in either
  direction**. `load_served(card, *, preprocessing, frame_um_per_px, device, score_threshold,
  weights, address)` refuses, each a `ModelError` and before any call, what a served model
  cannot honour:
  - `weights` set: `model_path names a file on this PC, and the model runs on the server:
    remove model_path, or unset SMC_MODEL_SERVER to run it here`;
  - `device` other than `auto`: `device=cuda is for a model that runs on this PC; the server
    chooses its own: remove device, or unset SMC_MODEL_SERVER`;
  - `preprocessing` other than `card.preprocessing`: `the server runs the card's
    pre-processing, and low_percentile, high_percentile or model_um_per_px change it: remove
    them, or unset SMC_MODEL_SERVER to run the model here`;
  - a pixel size that is negative or not finite; an address that is not one; a
    `SMC_MODEL_TIMEOUT_S` that is not a number above 0 (`SMC_MODEL_TIMEOUT_S must be a number
    of seconds above 0`).

  Then `open_wire(address)`, `check_served(...)`, `log.info` (ok) or `log.warning` (not) with
  its message, and `ResolvedModel(name=card.name, identity=model_identity(card.weights_sha256,
  card.preprocessing), weights_sha256=card.weights_sha256, runs_on="served",
  detector=ServedDetector(...))`. **A failed check does not refuse**: it is reported
  (§8.7) and every call checks again; `cells` ends `failed` at its first frame, and a scan
  (#635) goes on.

### 8.6 The server

- **`ModelServer(models: Mapping[str, ResolvedModel])`** is the server without HTTP:
  `listing() -> str` (LISTING) and `detect(model, frame, pixel_size_um) -> str` (ANSWER). One
  `threading.Lock` makes the forward passes one at a time (the kit runs a function on a
  thread of the loop's pool, and two calls would share the GPU). Its errors are `ValueError`s
  that name no path: `model is required`, `the server does not serve {model!r}; it serves: ...`,
  `detect takes one 2-D frame, got an array of shape ...`, `pixel_size_um must be a number
  of at least 0`. The fake model server **is** this class behind `ModelWire` (§8.8), so the
  two cannot drift.
- **`prepare(names, *, device, load=load_local_model) -> tuple[ModelServer, list[str]]`**:
  `names` is the cards to serve, or `None` for every card. Each is loaded with `load` (never
  `load_model`: a `SMC_MODEL_SERVER` in the server's own environment must not make it a client
  of itself) with `preprocessing=card.preprocessing`, `score_threshold=SERVER_SCORE_FLOOR`
  and a native rule's placeholders for the frame (`frame_shape_px=(1, 1)`,
  `frame_um_per_px=1.0`, which `preprocessing_for` never reads for a native rule). A card
  whose rule names a size is **not served**: the server serves native rules only (a sized
  rule needs a model per input size, and an unbounded cache; a design change when a card
  needs it, §9). With `names=None` a card that cannot be served (no weights here, a wrong
  SHA-256, a sized rule) is skipped and its reason goes to the returned notes; with names
  given, any failure is a `ModelError`; no model at all is a `ModelError`. The notes may
  name paths: they go to the server's own console.
- **`build_app(server) -> object`** imports the kit (inside), declares the two algorithms
  with `@algorithm_server`: `models` with `parameters={}`, and `detect` with
  `{"model": StringUI(), "image": ImageUI(dimensionality=[2]), "pixel_size_um":
  FloatSpinBoxUI(min=0.0, max=1000.0, default=0.0)}`, each returning
  `[(text, {"name": ...}, "text")]`; builds `MultiAlgorithmServer("smc-models", [...])`;
  sets `app.router.lifespan_context` to a no-op (§8.1); returns `server.app`.
- **`serve(server, *, host, port) -> None`**: `uvicorn.run(app, host=host, port=port,
  log_level="info")`; logs each call at INFO (the model, the frame's shape, the seconds;
  never pixels).
- **`smc serve-models [--host 127.0.0.1] [--port 8000] [--device auto] [--model NAME ...]`**
  (`smc.cli`, imports `smc.inference.server` inside the command): a port that is held is
  `port N is in use; choose another with --port`, exit 2, **before** any model is loaded
  (FM-144, FM-150: every address of `host`, an address this host cannot bind is not a taken
  port); a missing kit or uvicorn is `install the serve extra: pip install -e ".[serve]"`,
  exit 1; the notes of `prepare`, then **the models it serves with their identities**, one per
  line, then (for a `host` that is not a loopback address) `listening on <host>: anyone who
  reaches this port can run the models; it has no login`. Ctrl-C ends it by uvicorn's own
  graceful shutdown, exit 0. The default host is loopback: the lab's interface is the
  deployment note's (#686).

### 8.7 `cells` (amends §6)

- **A call that fails ends the stage `failed`.** `cells` wraps `detector.detect(pixels)`:
  a `ModelError` is `PluginRunError("cells", str(exc))`. The checkpoint (per well, as before)
  keeps the wells done, and `--resume` goes on from them.
- **Reported at the start**: after `_resolve`, for a `ServedDetector`: `ctx.progress(0,
  len(scan.frames), detector.check.message)` and the same text on the plugin's logger
  (INFO, WARNING when the check is not ok). The check does not refuse the stage.
- **`assumed`** gains one line for a served run with at least one frame timed in this
  attempt: `the model ran on the server: the first frame took 2.31 s and the next 11
  frames 0.12 s on average, network included (the first can carry the server's start-up
  work); the server's device was cuda` (only the first part with one frame).
- **`CellsCheckpoint.runs_on: str | None = None`** (a default, so that a checkpoint saved
  before this change loads) is saved with every checkpoint. A resume whose `runs_on` differs
  from the earlier attempt's is **not refused** (a server that died, and the operator
  going local, is the case ADR-0011 keeps a resume for) and `assumed` says so: `n of m
  frames were detected by an earlier attempt that ran on served; the others ran on
  local:cpu`. A checkpoint with no `runs_on` says nothing.
- `CellsParams.device`'s description says: with `SMC_MODEL_SERVER` set the server chooses and
  only `auto` is accepted. `cells_check.py` is unchanged: it calls `load_model`, so with a
  server set it measures the served model and records no place (§9).

### 8.8 The fake model server

`FakeModelServer(models: Mapping[str, ResolvedModel], *, address="http://models.test:8000")`
(`smc.testing`, `fake_model_server.py`) is a `ModelWire` with no socket, as `FakeMTB` is the
MTB's components: it wraps a `ModelServer` and speaks the wire as the kit's server does
(404 for another path, 422 for a body with another key than the three or a pixel size below 0,
500 `{"message"}` for a `ValueError` or an image that is not a TIFF, 201 with the result list),
so the real `ServedDetector` runs against it. Knobs, as attributes: `absent` (every call is
`ServedError("unreachable")`, and the server never saw it), `latency_s` (a call whose
`read_timeout_s` is below it is `ServedError("timeout")`; nothing sleeps), `fail_next: int`
(the next n calls are `ServedError("error")` with `HTTP 500`), `identity_override: str | None`
(the listing and the answers carry it), `hidden: set[str]` (names the server does not serve),
`on_call: Callable[[ServedCall], None] | None` (called as a call arrives, before it is answered:
where a test sets `absent` after n frames). `calls: list[ServedCall]` records, in order,
`path`, `body` (the exact dict the client built), `frame` (decoded from it) and
`read_timeout_s`. `install(monkeypatch)` sets `SERVER_ENV` to `address` and replaces
`served.open_wire` by a function that returns the fake for its address and raises `ServedError(
"unreachable")` for any other: a test that forgot the environment variable sees `load_model`
go local, and one that set another address sees a refusal, never a network call. The identity
it answers with is the one its models carry (`stub_model` computes it from the rule, FM-260).

### 8.9 Tests

- **The default suite never needs the kit, a socket to anywhere but loopback, torch or a
  weights file.** An autouse fixture of `tests/conftest.py` deletes `SMC_MODEL_SERVER` and
  `SMC_MODEL_TIMEOUT_S` for every test: a developer's PC that has them set must not send a
  local test over the network (FM-40).
- **The contract** (§7) gains the case `served`: a `ServedDetector` on a `FakeModelServer`
  around the stub's detector. It passes the same cases (the frame goes through TIFF and JSON
  and comes back identical).
- **`HttpWire` is tested on a real socket in the light set**, against a standard-library
  `ThreadingHTTPServer` in a thread that answers as the kit does (201 and the list, 500 and
  `{"message"}`, 404, 422, a body that is not JSON, one over `MAX_ANSWER_BYTES`, a connection it
  closes), with a handler that waits on a `threading.Event` for the timeout (FM-44) and a port
  that is closed for the refusal; the request body's keys are asserted to be the three; a
  `HTTP_PROXY` pointing nowhere does not divert it.
- **A real server test** (`tests/test_model_server_kit.py`) runs where `imaging_server_kit` is
  installed (`pytest.importorskip`, never in the light set: no CI job installs the kit): a
  `ModelServer` of stub models behind `build_app`, `uvicorn` in a thread on a free loopback
  port, `HttpWire` and `ServedDetector` against it, errors included.
- **The address is nowhere in a run folder**: a `cells` run through `load_model` with
  `SMC_MODEL_SERVER` set, done and failed (each kind), then every byte of the folder is searched
  for the host and the port. The failed one is the case that matters: its diagnosis is kept.

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
- **A sized rule on the server** (a card with `model_um_per_px`): the input size depends
  on the frame, so it needs a detector per size and a bound on how many (§8.6). The first
  such card brings its design change.
- **A token or TLS** for the server: a machine setting too, when the lab needs one; the
  address already accepts `https://`.
- **What a frame costs on the lab's network**: the frame goes as an uncompressed TIFF
  (5.3 MB for 1412 x 1412 `uint16`), and the time of a call, cold and warm, is measured on
  the lab's machine by #686, which also sets `DETECT_TIMEOUT_S` for good.
- **A CI leg that installs the kit** for the real-server test: no leg does today, and the
  kit's wheels on Windows with Python 3.14 are not measured.
- **`cells_check.py` and the place**: it records name, SHA-256 and identity, and not where
  the model ran; with a server set it measures the served model.

## 10. Detection while the scan goes (#635)

Today `cells` runs after the scan, over every frame (0.5 to 0.8 s a frame on the Ti2-E's PC,
15 to 25 minutes for a whole well at the 40×). Here a worker beside the scan detects each
frame as it is written, the scan never waits for it, and the `cells` stage that follows uses
what the worker found and detects only the rest. The model is the one §5 and §8 resolve,
local or served; nothing of §2 to §8 changes. **The scan does not change** either: the worker
listens on the frame channel that m3 §25 gave the page (`run_plugin(..., on_frame=)`,
`ShownFrame`), which hands every frame the scan writes, after its write, in order (§25.4).
§10.4 amends m3 §9.3 (`cells`).

### 10.1 Measured (2026-11-26, planning #635)

On the worktree's code with the main checkout's `.venv` (Python 3.13.0, macOS arm64), the
`demo-sample` stand of `tests/plate_check_stand.py` (`build_documents`, `opened(profile())`),
`scan-plan wells=B2,B3 rows=5 columns=5` (50 fields) and `rows=10 columns=10` (176 fields),
`scan check_plate=false` through `run_plugin`:

- **What the channel hands is what the TIFF holds.** For each of 50 frames, the `ShownFrame`'s
  `image` and `load_frame` of the written TIFF have the same SHA-256 over shape, `dtype.str`
  and bytes (both `<u2`, 512 × 512, C-contiguous), and `label` is the frame's `name`. The
  names are the well's and the field's (`B3_0000`), unique in a plan.
- **A worker beside the scan does not slow it**: the blob detector on a thread fed from
  `on_frame`, three runs each, alternating: 50 fields in 1.67, 1.69, 1.70 s without it and
  1.71, 1.71, 1.67 s with it; 176 fields in 5.93, 5.96, 5.95 s and 6.02, 5.96, 5.96 s. The
  worker kept up (5.5 ms a frame, the first 24 to 41 ms) and had nothing left when the scan
  ended.
- **The digest of a frame** (SHA-256 of `f"{h}x{w}|{dtype.str}|"` and the bytes): 0.17 ms at
  512 × 512 and 1.25 ms at 1412 × 1412 `uint16`. `load_frame` of a 512 × 512 TIFF: 0.20 ms.
- `cells detector=blobs` on those 50 frames: 0.24 s for the stage, 4.5 ms a frame.
- **A read-only view** (`view.flags.writeable = False`) goes through `BlobDetector.detect`
  (the same detections as the writable array) and through `served.encode_frame`. The local
  model converts it with `normalize_plane`, which makes a new `float32` array before
  `torch.from_numpy` (read in `local.py`, not run).
- **The store's line** (§10.3, prototyped): `LiveRecord` dumped as below and read back with
  `model_validate_json` is equal to itself, `score_threshold` 0.5 included; a line cut short is
  `json_invalid`, another `schema_version` `literal_error`. Under the repository's mypy, a
  field written `Field(None, ge=0.0)` is required at the constructor; `Field(default=None,
  ge=0.0)` is not.
- **Not measured**: the stand PC (a Prime 95B frame every 2 to 3 s, the model at 0.5 to 0.8 s
  a frame on its CPU, or served over the lab's network, #686); a store appended on Windows
  while `cells` reads it.

### 10.2 Decisions

- **D1 A listener, not a scan parameter.** The worker is `LiveCells`, a callable for
  `on_frame`, built by whoever runs the scan (the page, a test, a script) from the scan plan
  and the `CellsParams` the `cells` stage will run with. The scan, its parameters, its result
  and its checkpoint do not change, and a scan run without it is the scan of today. A scan
  with no listener (`smc run scan`) detects nothing while it goes.
- **D2 The scan never waits.** `put` takes a lock for a few appends and returns: no digest, no
  copy, no file, no model on the scan's thread. Every frame waits in a backlog held in memory,
  bounded by `LIVE_MAX_BACKLOG_BYTES`; a frame that would pass the bound is not held, and is
  named missed (`full`). `put` never raises (m3 §25 D3 would drop the listener for the rest
  of the run).
- **D3 What the worker finds is a memo, keyed by the pixels.** Each frame's detections go to a
  **store** in the run folder, one line per frame, with the frame's **digest**, the model's
  identity and the score threshold. `cells` reads every frame from its TIFF as it does today,
  digests it, and uses a line whose digest, identity and threshold are its own; any other
  frame is detected by the stage. The same pixels, the same model and the same threshold give
  the same detections, so the stage's result is the result it would have computed: the same
  suppression, the same merge, the same file, by construction. A name is not the key: a scan
  resumed in the middle of a well scans the well again whole and writes **new pixels under the
  same file names** (m3 §8), and a memo keyed by the name would give the old frame's cells to
  the new one.
- **D4 Failures are findings, and none ends the scan.** A frame whose call fails is named
  missed with its kind (`ServedError.kind` for the served model, §8.4) and the next frame is
  tried. After `LIVE_MAX_FAILED_IN_ROW` failures in a row the worker stops calling (ADR-0011:
  the loop must not pay a timeout on every frame), says so, and names every later frame
  missed. A model that cannot be resolved names every frame missed. `cells` detects every
  missed frame after the scan: it finds no line for it.
- **D5 The backlog is finished after the scan.** `finish()` stops taking frames and waits for
  the backlog, up to a timeout or a stop that its caller gives; the frames it did not reach
  are named missed (`unfinished`) and nothing is written for them afterwards, even when a call
  in flight comes back. Who runs the scan runs `finish()` when the scan's stage has ended, then
  `cells`.
- **D6 A resumed scan resumes its detections** through D3: the lines of the first attempt's
  store stay in the run folder, the resumed attempt's listener writes its own store, and
  `cells` reads them all; a frame scanned again has a new digest and is detected again.
- **D7 The detector gets a read-only view** of the scan's array (m3 §25 D2: the listener does
  not write into it, and the page may be drawing it): a detector that writes into its input
  fails that frame, and never changes a frame another reader holds.
- **D8 Nothing of the page here.** The counts and the cells as they come are given by
  `snapshot()` and `cells_since()`; the page (the Scan screen, the plate view, STOP during
  `finish()`) is #716's. No command of `smc` runs a scan with the listener here.

### 10.3 The store — `src/smc/plugins/cells/store.py` (new)

```python
#: The folder of the stores, relative to the run folder.
LIVE_FOLDER = "cells/live"
#: The one version of a store line this smc writes and reads.
LIVE_SCHEMA = 1

LiveMissKind = Literal[
    "unreachable",  # §8.4's kinds, for a served model
    "timeout",
    "error",  # §8.4's error, and a ModelError of a local model
    "identity",
    "malformed",
    "failed",  # another Exception from detect (a frame it refused, a bug)
    "label",  # a label the detector does not declare
    "no-model",  # the model could not be resolved
    "stopped",  # after LIVE_MAX_FAILED_IN_ROW failures in a row
    "full",  # the backlog was full when the frame came
    "store",  # the store could not be written
    "unfinished",  # not detected when finish() returned
]
```

```python
def frame_digest(pixels: npt.NDArray[Any]) -> str:
    """SHA-256 hex of f"{h}x{w}|{dtype.str}|" and the C-contiguous bytes of a 2-D frame."""
```

```python
class LiveRecord(BaseModel):
    """One line of a store: a frame's detections, or why the frame was missed."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal[1] = 1
    frame: str
    digest: str | None = None
    model: str | None = None
    identity: str | None = None
    runs_on: str | None = None
    device: str | None = None
    score_threshold: float | None = None
    seconds: float | None = Field(default=None, ge=0.0)
    detections: tuple[tuple[float, float, float, float, str, float], ...] | None = None
    missed: LiveMissKind | None = None
    message: str = ""
```

- A line is a **hit** (`detections` set, `missed` None, and `digest`, `identity`,
  `score_threshold`, `seconds`, `runs_on`, `device` set) or a **miss** (`missed` set,
  `detections` None); a validator refuses anything else. A detection is `(row0, col0, row1,
  col1, label, score)` as `Detection`'s fields, the detector's output as it came (the score
  threshold applied by the detector, nothing suppressed). A miss carries the digest when it
  was computed (a frame that failed its call), never for `full`, which the scan's thread does
  not hash.
- **A store** is `<run folder>/cells/live/<new_token()>.jsonl`, one per `LiveCells`, written
  only by it (§10.5): one JSON object a line, **`encode_record(record: LiveRecord) -> str`**,
  which is `json.dumps(record.model_dump(mode="json"), ensure_ascii=True, allow_nan=False,
  separators=(",", ":"))` and `"\n"` (the worker and the tests share it), appended and flushed
  line by line (opened `"a"`, `encoding="ascii"`, `newline="\n"`). No `fsync`: a line lost in
  a crash is a frame `cells` detects again. smc never removes a store; a new scan in the same
  run folder leaves the old ones, whose digests no new frame has.
- **`read_live(run_dir, *, identity, score_threshold, classes) -> LiveIndex`**: every
  `cells/live/*.jsonl` in name order (none, or no folder: an empty index), each read whole with
  `retry_on_permission_error`, `utf-8` decoded, split on `"\n"`; each line validated alone. A
  line that does not parse, does not validate, or is of another `schema_version` is skipped,
  and the lines before and after it still count (FM-194: a store cut by a crash ends with half
  a line). A hit is kept when its `identity` and `score_threshold` are the ones asked for
  (`==`: both come from the same parameter, through JSON's `repr`) and every label is in
  `classes`; a later hit of the same digest replaces an earlier one. A file that cannot be
  read (`OSError`, `UnicodeDecodeError`) is skipped. Every file skipped or with a line skipped
  is named, relative to the run folder, in `LiveIndex.unreadable`.

```python
@dataclass(frozen=True, slots=True)
class LiveHit:
    """A frame's detections from a store, with where and how long they took."""

    frame: str
    detections: tuple[Detection, ...]
    seconds: float
    runs_on: str
    device: str
```

```python
@dataclass(frozen=True, slots=True)
class LiveIndex:
    """The hits of every store of a run folder for one model and threshold, by digest."""

    hits: Mapping[str, LiveHit]
    unreadable: tuple[str, ...]
```

`store.py` imports nothing of `plugin.py` or `live.py` (both import it).

### 10.4 `cells` uses the store (amends m3 §9.3 and §8.7)

- **Resolution is shared.** `_resolve(params, scan)` becomes **`resolve_model(params, *,
  frame_shape_px, frame_um_per_px) -> ResolvedModel`**, public in `plugin.py`, with the same
  body and the same `PluginRunError`; `cells` calls it with `scan.image_shape_px` and
  `scan.pixel_size_um`, `LiveCells` with the plan's. It still calls `load_model`,
  `model_card` and `blob_model` through `plugin.py`'s names, so a test that replaces
  `plugin_mod.load_model` replaces it for both.
- **The stage always resolves the model**, as today, even when every frame has a hit: the
  identity it compares is the resolved one, and a stage whose model cannot be resolved fails
  as today.
- After `_resume`: `live = read_live(ctx.run_dir, identity=resolved.identity,
  score_threshold=params.score_threshold, classes=detector.classes)`; each name of
  `live.unreadable` is one WARNING on the plugin's logger, `<file> could not be read whole:
  its frames are detected again`.
- **In the loop**, after `_read_frame`: when `live.hits` is not empty, `hit =
  live.hits.get(frame_digest(pixels))`. A hit's detections are used as `detector.detect`'s
  would be (the label check, `suppress_in_frame`, `to_cells`, the progress); no call is made
  and its `seconds` go to a list of their own; `places = _places(places, hit.runs_on)`. A
  frame with no hit is detected as today. With no hit at all the stage digests nothing.
- **The result**:
  - `seconds_per_frame`: the mean over the stage's calls **and** the hits used (`None` when
    neither).
  - `model_runs_on`: `resolved.runs_on` when no hit was used; the hits' places in order of
    first use, joined by `+` with `_places`, then `resolved.runs_on` when the stage called
    the detector at least once.
  - `device`: `detector.device` when the stage called the detector at least once, else the
    last hit's `device`.
  - **`assumed`** gains, after §8.7's served line, when at least one hit was used: `{k} of {n}
    frames were detected during the scan, as they were written ({places}: {mean:.2f} s a
    frame on average, the first {first:.2f} s); this stage detected the other {n - k}`, with
    `{places}` the hits' places joined by ` and `, the first being the first hit used in the
    scan's order, and `the other 0` written `none`. With one hit: `({places}: {first:.2f}
    s)`. The served line keeps counting the stage's own calls only.
- The checkpoint, the resume, the identity's refusal, the partial file: unchanged. A resumed
  `cells` reads the stores again; their hits are a memo, not a state the checkpoint carries.

### 10.5 The worker — `src/smc/plugins/cells/live.py` (new)

```python
#: Assumed: 256 MiB of frames waiting is 64 Prime 95B frames, minutes of a 40x scan.
LIVE_MAX_BACKLOG_BYTES = 256 * 1024 * 1024
#: ADR-0011's "a few failures in a row": chosen, not measured.
LIVE_MAX_FAILED_IN_ROW = 3
```

```python
@dataclass(frozen=True, slots=True)
class LiveMiss:
    """A frame the worker did not detect: its name, why (a kind), and the diagnosis."""

    frame: str
    kind: LiveMissKind
    message: str
```

```python
@dataclass(frozen=True, slots=True)
class LiveSnapshot:
    """What the worker has done so far, for a page's counts and a test."""

    state: Literal["starting", "running", "stopped", "failed", "finished"]
    message: str
    handed: int
    detected: int
    cells: int
    waiting: int
    missed: tuple[LiveMiss, ...]
    seconds_per_frame: float | None
    model: str | None
    runs_on: str | None
    device: str | None
```

```python
class LiveCells:
    """Detects a scan's frames as the scan hands them, beside it, and stores what it finds."""

    def __init__(
        self, run_dir: Path, plan: ScanPlanResult, params: CellsParams
    ) -> None: ...
    def start(self) -> LiveCells: ...
    def put(self, shown: ShownFrame) -> None: ...
    def snapshot(self) -> LiveSnapshot: ...
    def cells_since(self, index: int) -> tuple[tuple[Cell, ...], int]: ...
    def finish(
        self,
        *,
        timeout_s: float | None = None,
        stop: threading.Event | None = None,
    ) -> LiveSnapshot: ...
    def wait_ended(self, timeout_s: float) -> bool: ...
```

- **The constructor** keeps the plan's fields by name (`name -> PlannedField`), its
  `matrix_um_per_px`, `image_shape_px` and `pixel_size_um`, the run folder, and the params; it
  starts nothing and writes nothing.
- **`start()`** starts one daemon thread, `smc-live-cells`, and returns `self`; a second call
  is a `RuntimeError`. The thread first resolves the model, `resolve_model(params,
  frame_shape_px=plan.image_shape_px, frame_um_per_px=plan.pixel_size_um)` (a served model's
  check, or torch and the weights: seconds, off the scan's thread). A `PluginRunError` there
  is state `failed` with its reason as `message`, one WARNING, and every frame handed, before
  or after, missed `no-model`. A served model's `check.message` is the `message` while
  `running`, logged as `cells` does it (INFO, or WARNING when the check is not ok).
- **`put(shown)`**, on the scan's thread, never raises (its body is in a `try` that catches
  `Exception`; the first one is logged once at WARNING, by its text, FM-98), takes the lock
  for appends only (FM-70), and:
  - ignores a frame whose `plugin` is not `scan`, whose `label` is not a field of the plan, or
    whose `xy_um` is `None` (counted nowhere: not the scan's, or not placeable), and every
    frame after `finish()` was called (one WARNING, the first time);
  - otherwise counts it `handed` and, when the backlog's bytes plus `image.nbytes` exceed
    `LIVE_MAX_BACKLOG_BYTES`, names it missed `full` (`the backlog held {n} frames ({m} MiB)
    waiting for the detector`); else appends it to the backlog and notifies the thread.
- **The thread**, oldest frame first: takes a frame under the lock, then outside it makes the
  read-only view (D7), its digest, the call timed with `perf_counter`, and the outcome:
  - detections whose labels are all in `detector.classes`: a hit line, the frame's cells
    (`suppress_in_frame` with `params.max_overlap`, then `to_cells` with a `ScanFrame` built
    from the planned field's well, index and name, `shown.xy_um` as `x_um, y_um`, `file=""`,
    and `frame` the order in which this worker detected it), the in-a-row count reset;
  - a `ServedError`: missed with its `kind`; another `ModelError`: `error`; another label:
    `label`, naming it; another `Exception`: `failed`, `<Type>: <first line>`. Each adds one
    to the in-a-row count; at `LIVE_MAX_FAILED_IN_ROW` the state is `stopped`, `message` is
    `the detection stopped after {n} failures in a row, the last on {frame}: {message}; cells
    detects the frames it missed after the scan`, one WARNING, and every frame waiting or to
    come is missed `stopped` without a call. A `BaseException` that is not an `Exception`
    ends the thread (the process is going away).
  - **The store** is opened at the thread's first line (`mkdir(parents=True,
    exist_ok=True)`). An `OSError` opening or writing it is state `failed`, `cannot write
    {relative path} ({strerror}): cells detects the frames after the scan`, one WARNING; that
    frame and every later one are missed `store`, and nothing more is written.
  - Writing a line and the `finished` check happen under a second lock, `_write_lock`, that
    `finish()` also takes, so no line is written after `finish()` returned.
- **`finish(*, timeout_s=None, stop=None)`**, on the caller's thread: no more frames are taken;
  it waits, in slices of at most 0.1 s, until nothing is waiting or in flight, or `timeout_s`
  has passed, or `stop` is set (a `ValueError` for a `timeout_s` that is negative or not
  finite). Then, under `_write_lock`: the state becomes `finished` (the `message` of a
  `stopped` or `failed` worker is kept), every frame still waiting or in flight is missed
  `unfinished` (`not detected when the detection was finished`, and `, {t:g} s after the
  scan` when the timeout ended the wait, `: stopped` when `stop` did) and written to the store
  when it is open, and the store is closed. It logs one line, INFO when nothing was missed,
  WARNING otherwise: `cells during the scan: {detected} of {handed} frames detected, {cells}
  cells; missed {m}: {kind} {count}, ... (the first {frame}: {message})`. It returns the
  snapshot, and a second call returns the same snapshot at once. A thread still inside a call
  is left to end on its own; what it returns is dropped.
- **`snapshot()`**: the counts under the lock; `waiting` is the backlog plus the frame in
  flight; `cells` the number of cells of the hits; `seconds_per_frame` the mean of the hits'
  seconds.
- **`cells_since(index)`**: the cells of the frames detected from the `index`-th on, in the
  order they were detected, and the index to ask from next time. They are in stage µm,
  suppressed within their frame and **not merged across fields**: a cell two fields saw is
  there twice until `cells` merges it. `Cell.frame` is the worker's own order, not a position
  in `ScanResult.frames`.
- **`wait_ended(timeout_s)`**: joins the thread for at most `timeout_s`, and says whether it
  has ended (a test's check, FM-44).
- **Never**: a digest, a copy, a file or a model call on the scan's thread; a lock held across
  a call, a write or a log line; a frame written into; an address in a line or a message
  (§8.4's messages hold none, FM-50).

### 10.6 Tests

- **`tests/unit/test_cells_store.py`**: the digest (shape, dtype and one pixel change it; a
  TIFF written by `encode_frame` of the scan and read back by `load_frame` keeps it); a hit
  and a miss round-trip; the validator's refusals; `read_live` keeps the identity and the
  threshold asked for, drops a hit with an undeclared label, lets a later hit of a digest
  replace an earlier one, skips a torn last line and a line of another schema while the
  others count, and names an unreadable file.
- **`tests/unit/test_cells_live.py`**: `LiveCells` fed `ShownFrame`s by hand, with no stand,
  the model a `StubDetector` through `plugin_mod.load_model` replaced (§7), and threads that
  meet on `threading.Event` (FM-44): `put` returns at once while the detector is blocked; the
  order; frames that are ignored; each kind of miss; the stop after the failures in a row
  (the detector called exactly that many times); a model that cannot be resolved; a full
  backlog (`LIVE_MAX_BACKLOG_BYTES` set small); `finish` after the backlog, at its timeout and
  at a stop, and no line written after it; the read-only view; a store that cannot be
  written; `cells_since`'s stage µm; a served model's check as the `message`
  (`FakeModelServer`).
- **`tests/test_cells_demo.py`** (amended): `cells` uses a hit and calls the detector for the
  rest; a hit of another identity, another threshold or another digest (a frame rewritten
  since) is not used; the `assumed` line, `seconds_per_frame`, `model_runs_on` and `device`
  of §10.4.
- **`tests/test_cells_live_demo.py`** (new, `demo-sample` through `run_plugin`, the helpers
  of `tests/plate_check_stand.py`): a scan with `LiveCells(detector=blobs)` as `on_frame`,
  then `finish()`, then `cells`, gives the cells file, the counts and `merged` of the same
  stage run again with the stores removed, and the first `cells` made no detector call; a
  detector that raises on one frame, and one that never answers, leave the scan `done` and
  the missed frames named; a scan stopped in its second well and resumed with a second
  `LiveCells` leaves `cells` no frame to detect. **The bound** (`@pytest.mark.wallclock`):
  three scans with the listener and three without, alternating, of §10.1's 50 fields; the
  median with it is at most 1.15 times the median without (measured: 2 %).

### 10.7 Not decided here

- **The page** (#716): the toggle, `finish()` and `cells` after a scan, STOP during
  `finish()`, the counts and the cells on the plate view.
- **A command** that scans with the listener (`smc run` runs one stage).
- **A store's frames that a resumed scan holds but no listener saw** (a first attempt run
  without one): `cells` detects them; nothing feeds them to the second listener.
- **Removing old stores** of a run folder.
- **A deadline for the whole served call** (#711): until then a slow trickle can hold the
  worker longer than `DETECT_TIMEOUT_S`; the scan does not wait for it, and `finish()`'s
  timeout bounds the wait after it.
