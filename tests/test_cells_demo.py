"""``cells`` through ``run_plugin``: frames in, cells in stage µm out, and what a stop leaves (m3 §9.3, #23).

The plugin touches nothing of the stand (``requires = ()``): ``demo_microscope``
is here because ``run_plugin`` needs a ``Microscope``. The scan is a ``scan``
record the test writes, with frames it saves (``np.full`` of the frame's position, so
that a stub detector knows which frame it was given), and the detector is a
stub that returns what the test scripted: no torch and no weights, so the
default suite stays away from a vendor adapter (FM-40).

The geometry is 60 x 80 pixels at 0.5 µm, upright. A cell at column 70 of the
first frame of a well is at column 10 of the second, which is 30 µm further
along X: the one object both fields saw.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import threading
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import tifffile
from pydantic import BaseModel

from smc.hardware import Microscope
from smc.inference import blobs as blobs_mod
from smc.inference import local as local_mod
from smc.inference.blobs import BLOB_DETECTOR, BLOB_DETECTOR_IDENTITY
from smc.inference.cards import DEFAULT_MODEL_NAME, ModelCard, model_card
from smc.inference.detector import Detection
from smc.inference.errors import ModelError
from smc.inference.models import ResolvedModel, file_sha256, model_identity
from smc.inference.preprocessing import ModelPreprocessing
from smc.plugins import (
    ParamsError,
    PluginRunError,
    Progress,
    ResumeError,
    RunContext,
    load,
    parse_params,
    run_plugin,
)
from smc.plugins.cells import plugin as plugin_mod
from smc.plugins.cells.plugin import CellsParams, CellsPlugin
from smc.plugins.slots import (
    Cell,
    CellsFile,
    CellsResult,
    ScanFrame,
    ScanResult,
    load_cells,
)
from smc.run import RunDocument, StageRecord, StageStatus, write_stage
from smc.run.files import AtomicWriteError
from smc.testing import FakeModelServer, ServedCall, StubDetector, stub_model

pytestmark = pytest.mark.demo

SHAPE = (60, 80)  # (height, width): not square, on purpose (FM-73)
LOGGER = "smc.plugins.cells"
WEIGHTS = b"weights, version 1"

#: Seen by the first frame of a well at column 70, 9.5 px from its edge...
SHARED = Detection(25.0, 65.0, 35.0, 75.0, "single", 0.9)
#: ...and by the second frame of the well at column 10: the same cell, 10.5 px from the edge.
SECOND_VIEW = Detection(25.0, 5.0, 35.0, 15.0, "single", 0.7)
#: A cell only the first frame sees, in the middle of it.
OWN = Detection(25.0, 15.0, 35.0, 25.0, "doublet", 0.8)

#: Two wells of two overlapping frames each; the frame's position is its pixel value.
TWO_WELLS = [("A1", 0.0), ("A1", 30.0), ("A2", 0.0), ("A2", 30.0)]
SCRIPT = {0: [SHARED, OWN], 1: [SECOND_VIEW], 2: [SHARED, OWN], 3: [SECOND_VIEW]}


def _install(
    monkeypatch: pytest.MonkeyPatch, stub: StubDetector
) -> list[dict[str, Any]]:
    """Make ``load_model`` give ``stub``; the calls it received are returned.

    The stand-in works its identity out from the rule it receives and from the
    weights it is given (hashed, as the real one does), so a run with another
    rule or other weights has another identity, and the plugin cannot ignore
    either (FM-260). Without ``weights`` the file is the model's own and its
    SHA-256 is a constant.
    """
    calls: list[dict[str, Any]] = []

    def fake(
        card: ModelCard,
        *,
        preprocessing: ModelPreprocessing,
        frame_shape_px: tuple[int, int],
        frame_um_per_px: float,
        device: str,
        score_threshold: float,
        weights: Path | None = None,
    ) -> ResolvedModel:
        calls.append(
            {
                "card": card,
                "rule": preprocessing,
                "frame_shape_px": frame_shape_px,
                "frame_um_per_px": frame_um_per_px,
                "device": device,
                "threshold": score_threshold,
                "weights": weights,
            }
        )
        sha = file_sha256(weights) if weights is not None else "0" * 64
        return stub_model(
            stub, name=card.name, weights_sha256=sha, preprocessing=preprocessing
        )

    monkeypatch.setattr(plugin_mod, "load_model", fake)
    return calls


def _frame_name(position: int) -> str:
    return f"f{position}"


def _seed_scan(
    tmp_path: Path, frames: Sequence[tuple[str, float]] = TWO_WELLS
) -> ScanResult:
    """A finished ``scan`` in ``tmp_path/run.json``, with a TIFF for every frame."""
    folder = tmp_path / "scan" / "t"
    folder.mkdir(parents=True, exist_ok=True)
    scan_frames = []
    fields_of: dict[str, int] = {}
    for position, (well, x_um) in enumerate(frames):
        field = fields_of.get(well, 0)
        fields_of[well] = field + 1
        name = _frame_name(position)
        tifffile.imwrite(
            folder / f"{name}.tif", np.full(SHAPE, position, dtype=np.uint16)
        )
        scan_frames.append(
            ScanFrame(
                well=well,
                index=field,
                name=name,
                file=f"scan/t/{name}.tif",
                x_um=x_um,
                y_um=0.0,
                z_um=None,
                focus="none",
                pfs_offset=None,
                t_s=float(position),
            )
        )
    scan = ScanResult(
        folder="scan/t",
        objective="20x",
        pixel_size_um=0.5,
        matrix_um_per_px=((0.5, 0.0), (0.0, -0.5)),
        image_shape_px=SHAPE,
        exposure_ms=10.0,
        frames=tuple(scan_frames),
        failed=(),
    )
    then = datetime(2026, 10, 27, 9, 30, tzinfo=timezone.utc)
    write_stage(
        tmp_path / "run.json",
        StageRecord(
            plugin="scan",
            provides="scan",
            attempt="s1",
            status=StageStatus.done,
            started_utc=then,
            finished_utc=then,
            profile="demo",
            dry_run=False,
            smc_version="0",
            params={},
            result=scan.model_dump(mode="json"),
        ),
    )
    return scan


def _weights(tmp_path: Path, content: bytes = WEIGHTS) -> Path:
    path = tmp_path / "w.pth"
    path.write_bytes(content)
    return path


def _run(
    microscope: Microscope,
    tmp_path: Path,
    *,
    resume: bool = False,
    stop: threading.Event | None = None,
    on_progress: Callable[[Progress], None] | None = None,
    **params: Any,
) -> StageRecord:
    params.setdefault("model_path", tmp_path / "w.pth")
    params.setdefault("device", "cpu")
    return run_plugin(
        CellsPlugin(),
        microscope,
        CellsParams(**params),
        tmp_path / "run.json",
        resume=resume,
        stop=stop,
        on_progress=on_progress,
    )


def _result(record: StageRecord) -> CellsResult:
    assert record.status is StageStatus.done
    assert record.result is not None
    return CellsResult.model_validate(record.result)


def _stored(tmp_path: Path) -> StageRecord:
    return RunDocument.load(tmp_path / "run.json").stages["cells"]


def _cells(tmp_path: Path, record: StageRecord) -> tuple[Cell, ...]:
    return load_cells(tmp_path, _result(record))


def _stopping_at(position: int, stop: threading.Event) -> Callable[[int], None]:
    """The ``on_frame`` of a stub that asks for a stop when it is given frame ``position``."""

    def on_frame(seen: int) -> None:
        if seen == position:
            stop.set()

    return on_frame


def _stop_in_the_second_well(
    microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> StageRecord:
    """The first attempt of ``TWO_WELLS``: A1 done, stopped on the first frame of A2."""
    stop = threading.Event()
    _install(monkeypatch, StubDetector(SCRIPT, on_frame=_stopping_at(2, stop)))
    return _run(microscope, tmp_path, stop=stop)


# --------------------------------------------------------------- registration


def test_the_entry_point_loads_cells() -> None:
    entry = load("cells")

    assert entry.plugin.requires == ()
    assert entry.plugin.reads == ("scan",)
    assert entry.plugin.provides == "cells"
    assert entry.target == "smc.plugins.cells.plugin:CellsPlugin"


# ------------------------------------------------------------------ the result


def test_cells_of_a_scan_are_written_counted_and_merged(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = [("A1", 0.0), ("A1", 30.0), ("B2", 0.0)]
    scan = _seed_scan(tmp_path, frames)
    _weights(tmp_path)
    stub = StubDetector(
        {0: [SHARED, OWN], 1: [SECOND_VIEW]}
    )  # B2's frame shows nothing
    _install(monkeypatch, stub)

    record = _run(demo_microscope, tmp_path)

    result = _result(record)
    cells = _cells(tmp_path, record)
    assert {(c.well, c.label, c.frame) for c in cells} == {
        ("A1", "doublet", 0),
        (
            "A1",
            "single",
            1,
        ),  # the second field saw the shared cell farther from its edge
    }
    shared = next(c for c in cells if c.label == "single")
    assert (shared.x_um, shared.y_um) == pytest.approx((15.25, -0.25))
    assert result.counts == {
        "A1": {"single": 1, "doublet": 1, "debris": 0},
        "B2": {"single": 0, "doublet": 0, "debris": 0},
    }
    assert result.merged == 1
    assert result.frames == len(scan.frames) == 3
    assert result.classes == ("single", "doublet", "debris")
    assert result.device == "cpu"
    assert result.file.startswith("cells/")
    assert result.seconds_per_frame is not None
    assert result.seconds_per_frame >= 0.0
    assert stub.seen == [0, 1, 2]


def test_the_parameters_reach_the_detector(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(tmp_path)
    weights = _weights(tmp_path)
    calls = _install(monkeypatch, StubDetector({}))

    record = _run(demo_microscope, tmp_path, score_threshold=0.8)

    assert calls == [
        {
            "card": model_card(DEFAULT_MODEL_NAME),
            "weights": weights,
            "device": "cpu",
            "threshold": 0.8,
            # No new parameter: the model as it was trained, its own resize and percentiles.
            "rule": ModelPreprocessing(percentiles=(0.5, 99.5), model_um_per_px=None),
            "frame_shape_px": SHAPE,
            "frame_um_per_px": 0.5,
        }
    ]
    assert _result(record).score_threshold == 0.8


def test_max_overlap_reaches_the_suppression_of_a_frame(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two boxes sharing 0.5 of the smaller one: one object at 0.4, two at the default 0.7."""
    _seed_scan(tmp_path, frames=[("A1", 0.0)])
    _weights(tmp_path)
    single = Detection(25.0, 15.0, 35.0, 25.0, "single", 0.6)
    doublet = Detection(25.0, 20.0, 35.0, 30.0, "doublet", 0.9)
    _install(monkeypatch, StubDetector({0: [single, doublet]}))

    strict = _result(_run(demo_microscope, tmp_path, max_overlap=0.4))
    default = _result(_run(demo_microscope, tmp_path))

    assert strict.counts["A1"] == {"single": 0, "doublet": 1, "debris": 0}
    assert default.counts["A1"] == {"single": 1, "doublet": 1, "debris": 0}


def test_the_result_names_the_model_by_its_card_never_by_a_path(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(tmp_path)
    _weights(tmp_path)
    _install(monkeypatch, StubDetector(SCRIPT))

    record = _run(demo_microscope, tmp_path)

    assert _result(record).model == "cell-classes-v2"
    text = json.dumps(record.result)
    # json.dumps doubles every backslash: a Windows path leaks as its escaped
    # spelling, which `str(tmp_path)` is not (FM-43). The folder name is the
    # test id, which nothing of the result holds.
    escaped = json.dumps(str(tmp_path))[1:-1]
    assert str(tmp_path) not in text
    assert escaped not in text
    assert tmp_path.as_posix() not in text
    assert tmp_path.name not in text


def test_the_result_records_the_models_name_identity_and_where_it_ran(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hashing itself is ``smc.inference``'s: here the result says what ``load_model`` returned."""
    _seed_scan(tmp_path)
    _weights(tmp_path)
    _install(monkeypatch, StubDetector({}, device="cuda"))

    result = _result(_run(demo_microscope, tmp_path, device="cuda"))

    sha = hashlib.sha256(WEIGHTS).hexdigest()
    trained = ModelPreprocessing(percentiles=(0.5, 99.5), model_um_per_px=None)
    assert result.model == "cell-classes-v2"
    assert result.model_sha256 == sha
    assert result.model_identity == model_identity(sha, trained)
    assert result.model_runs_on == "local:cuda"
    assert result.device == "cuda"


def test_cells_asks_for_the_model_by_name_with_the_cards_preprocessing_and_the_scans_geometry(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The seam between ``cells`` and ``load_model`` (FM-124): every argument the plugin owes it."""
    _seed_scan(tmp_path)
    weights = _weights(tmp_path)
    calls = _install(monkeypatch, StubDetector({}))

    # A device and a threshold that are not the helper's defaults (cpu, 0.5): a
    # parameter replaced by a constant would still pass with those.
    _run(
        demo_microscope,
        tmp_path,
        model="cell-classes-v2",
        device="cuda",
        score_threshold=0.8,
    )

    (call,) = calls
    assert call["card"] == model_card("cell-classes-v2")
    assert call["rule"] == ModelPreprocessing(
        percentiles=(0.5, 99.5), model_um_per_px=None
    )
    assert call["frame_shape_px"] == (60, 80)
    assert call["frame_um_per_px"] == 0.5
    assert call["weights"] == weights
    assert call["device"] == "cuda"
    assert call["threshold"] == 0.8


def test_a_parameter_replaces_only_its_part_of_the_cards_preprocessing(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(tmp_path)
    _weights(tmp_path)
    calls = _install(monkeypatch, StubDetector({}))

    _run(demo_microscope, tmp_path, low_percentile=1)

    assert _rule_of(calls) == ModelPreprocessing(
        percentiles=(1.0, 99.5), model_um_per_px=None
    )


def test_the_parameters_edges_are_valid_rules_not_a_raw_validation_error(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``low_percentile=0``, ``high_percentile=100`` and ``model_um_per_px=10`` are inside ``CellsParams``' bounds, so they must be inside the rule's."""
    _seed_scan(tmp_path)
    _weights(tmp_path)
    calls = _install(monkeypatch, StubDetector({}))

    _run(
        demo_microscope,
        tmp_path,
        low_percentile=0,
        high_percentile=100,
        model_um_per_px=10,
    )

    assert _rule_of(calls) == ModelPreprocessing(
        percentiles=(0.0, 100.0), model_um_per_px=10.0
    )


def test_an_unknown_model_is_refused_with_the_known_names(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real ``model_card``: the message lists the cards there are."""
    _seed_scan(tmp_path)
    _weights(tmp_path)
    calls = _install(monkeypatch, StubDetector({}))

    with pytest.raises(PluginRunError) as caught:
        _run(demo_microscope, tmp_path, model="cell-classes-v9")

    assert "no model named 'cell-classes-v9'" in caught.value.reason
    assert "cell-classes-v2" in caught.value.reason
    assert calls == []
    assert _stored(tmp_path).status is StageStatus.failed


def test_a_model_error_is_a_plugin_run_error_with_its_text(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(tmp_path)

    def refused(*args: object, **kwargs: object) -> ResolvedModel:
        raise ModelError("w.pth has the SHA-256 aa, and the card x names bb")

    monkeypatch.setattr(plugin_mod, "load_model", refused)

    with pytest.raises(PluginRunError) as caught:
        _run(demo_microscope, tmp_path)

    assert caught.value.reason == "w.pth has the SHA-256 aa, and the card x names bb"
    assert isinstance(caught.value.__cause__, ModelError)


def test_the_merge_radius_is_named_as_chosen_and_not_measured(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(tmp_path)
    _weights(tmp_path)
    _install(monkeypatch, StubDetector({}))

    # The scan is at 0.5 um per pixel: checked there, the model adds no line of its own.
    chosen = _result(
        _run(demo_microscope, tmp_path, merge_radius_um=3.5, checked_um_per_px=0.5)
    )
    off = _result(
        _run(demo_microscope, tmp_path, merge_radius_um=0.0, checked_um_per_px=0.5)
    )

    assert chosen.assumed == (
        "cells within 3.5 µm seen by two fields are counted once: a chosen radius, "
        "not measured",
    )
    assert off.assumed == ()
    assert (chosen.merge_radius_um, off.merge_radius_um) == (3.5, 0.0)


# ------------------------------------------- the model's pre-processing (m3 §9.4, #342)
#
# The scan is 60 x 80 pixels at 0.5 um, not square (FM-73). The expected input
# sizes are worked out by hand: shown at 0.1 um the frame has 5 times the pixels,
# 300 x 400; shown at 0.5 it is the frame itself, 60 x 80, far under the 256 pixels
# the model is given.

MERGE_LINE = "cells within 5 µm seen by two fields are counted once: a chosen radius, not measured"
UNCHECKED_LINE = (
    "the model's detections were not measured on frames of this stand: "
    "checked_um_per_px is not set (docs/plugins/cells.md)"
)


def _rule_of(calls: list[dict[str, Any]]) -> ModelPreprocessing:
    (call,) = calls
    rule = call["rule"]
    assert isinstance(rule, ModelPreprocessing)
    return rule


def _real_load_model(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, Any]]:
    """The real ``load_model`` on a card that names the test's weights, around a recording ``TorchDetector``.

    This is the seam between ``cells`` and ``load_model`` (FM-124): what the
    detector receives is worked out by the real function from what the plugin
    passes it, so a parameter that did not reach it, or reached it as another
    part, is seen. The card is the shipped one with the SHA-256 of ``WEIGHTS``.
    """
    card = model_card(DEFAULT_MODEL_NAME).model_copy(
        update={"weights_sha256": hashlib.sha256(WEIGHTS).hexdigest()}
    )
    monkeypatch.setattr(plugin_mod, "model_card", lambda name: card)
    built: list[dict[str, Any]] = []

    def torch_detector(weights: Path, **kwargs: Any) -> StubDetector:
        built.append({"weights": weights, **kwargs})
        return StubDetector({})

    monkeypatch.setattr(local_mod, "TorchDetector", torch_detector)
    return built


def test_model_um_per_px_sets_the_model_input_from_the_scan_pixel_size(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(tmp_path)
    _weights(tmp_path)
    built = _real_load_model(monkeypatch)

    _run(demo_microscope, tmp_path, model_um_per_px=0.1)

    # (min_size, max_size) = (short side, long side) = (60, 80) x 0.5 / 0.1.
    (call,) = built
    assert call["preprocessing"].input_size_px == (300, 400)


def test_without_model_um_per_px_the_model_keeps_its_own_resize(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(tmp_path)
    _weights(tmp_path)
    built = _real_load_model(monkeypatch)

    _run(demo_microscope, tmp_path)

    (call,) = built
    assert call["preprocessing"].input_size_px is None


def test_the_percentiles_reach_the_detector(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(tmp_path)
    _weights(tmp_path)
    built = _real_load_model(monkeypatch)

    _run(demo_microscope, tmp_path, low_percentile=1.0, high_percentile=99.0)

    # Low first, high second: a pair that was exchanged would be refused by
    # `ModelPreprocessing`, and one built from the wrong parameters would not read (1, 99).
    (call,) = built
    assert call["preprocessing"].percentiles == (1.0, 99.0)


def test_a_model_input_out_of_bounds_is_refused_before_the_model_loads(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """60 x 80 at 0.5 shown at 0.5 is 60 x 80 pixels: under the 256 the model is given."""
    _seed_scan(
        tmp_path
    )  # and no weights: the size is refused before they are looked for
    monkeypatch.setenv("SMC_MODELS", str(tmp_path))
    built = _real_load_model(monkeypatch)

    with pytest.raises(PluginRunError) as caught:
        _run(demo_microscope, tmp_path, model_um_per_px=0.5, model_path=None)

    assert "model_um_per_px" in caught.value.reason
    assert "60x80" in caught.value.reason
    assert "no model at" not in caught.value.reason
    assert built == []  # 166 MB of weights were not read for it
    record = _stored(tmp_path)
    assert record.status is StageStatus.failed


def test_an_unchecked_model_says_so_in_assumed(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(tmp_path)
    _weights(tmp_path)
    _install(monkeypatch, StubDetector({}))

    result = _result(_run(demo_microscope, tmp_path))

    assert result.assumed == (UNCHECKED_LINE, MERGE_LINE)


def test_a_scan_far_from_the_checked_pixel_size_says_so(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(tmp_path)
    _weights(tmp_path)
    _install(monkeypatch, StubDetector({}))

    # The scan is at 0.5 um per pixel, checked at 0.55: 9.1 % under it.
    result = _result(_run(demo_microscope, tmp_path, checked_um_per_px=0.55))

    assert result.assumed == (
        "the frames are at 0.5 µm per pixel and the model was checked at 0.55 µm per "
        "pixel: its counts at this pixel size are not measured",
        MERGE_LINE,
    )


def test_a_scan_within_five_percent_of_the_checked_pixel_size_adds_nothing(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(tmp_path)
    _weights(tmp_path)
    _install(monkeypatch, StubDetector({}))

    # 0.5 against 0.51 is 2.0 % under it.
    result = _result(_run(demo_microscope, tmp_path, checked_um_per_px=0.51))

    assert result.assumed == (MERGE_LINE,)


@pytest.mark.parametrize(
    ("checked", "says"),
    [
        (0.4762, False),  # 0.5 / 0.4762 - 1 = +4.998 %
        (0.4761, True),  # +5.020 %
        (0.5262, False),  # 0.5 / 0.5262 - 1 = -4.979 %
        (0.5264, True),  # -5.015 %
    ],
)
def test_the_tolerance_on_the_checked_pixel_size_is_five_percent_of_the_checked_one(
    demo_microscope: Microscope,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checked: float,
    says: bool,
) -> None:
    """Relative to the checked size, on both sides: ``abs(p / c - 1) > 0.05``."""
    _seed_scan(tmp_path)
    _weights(tmp_path)
    _install(monkeypatch, StubDetector({}))

    result = _result(_run(demo_microscope, tmp_path, checked_um_per_px=checked))

    assert (len(result.assumed) == 2) is says


def test_the_checked_pixel_size_changes_nothing_in_the_detection(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(tmp_path)
    _weights(tmp_path)
    calls = _install(monkeypatch, StubDetector(SCRIPT))

    result = _result(_run(demo_microscope, tmp_path, checked_um_per_px=0.55))

    assert _rule_of(calls) == ModelPreprocessing(
        percentiles=(0.5, 99.5), model_um_per_px=None
    )
    assert result.counts["A1"] == {"single": 1, "doublet": 1, "debris": 0}


def test_the_blob_detector_says_nothing_of_a_checked_model(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    _seed_scan(tmp_path)

    result = _result(_blobs(demo_microscope, tmp_path, merge_radius_um=0.0))

    assert not any("checked_um_per_px" in line for line in result.assumed)


@pytest.mark.parametrize(
    "extra",
    [
        "model_um_per_px=0.1",
        "checked_um_per_px=0.5",
        "low_percentile=1",
        "high_percentile=99",
    ],
)
def test_blobs_refuse_the_model_preprocessing(extra: str) -> None:
    name = extra.split("=")[0]

    with pytest.raises(ParamsError) as caught:
        parse_params(CellsParams, ["detector=blobs", extra])

    assert f"{name} is for detector=model" in str(caught.value)
    assert "blob detector" in str(caught.value)


def test_blobs_refuse_a_model_name() -> None:
    with pytest.raises(ParamsError) as caught:
        parse_params(CellsParams, ["detector=blobs", "model=cell-classes-v2"])

    assert "model is for detector=model" in str(caught.value)
    assert "not a named model" in str(caught.value)


def test_blobs_take_the_trained_percentiles_when_they_are_given() -> None:
    """A run document recorded before #684 holds 0.5 and 99.5 for a blobs run: re-validating it must not fail."""
    params = parse_params(
        CellsParams, ["detector=blobs", "low_percentile=0.5", "high_percentile=99.5"]
    )

    assert params.detector == "blobs"
    assert (params.low_percentile, params.high_percentile) == (0.5, 99.5)


def test_blobs_refuse_a_percentile_that_is_not_the_trained_one() -> None:
    for extra in ("low_percentile=1", "high_percentile=99"):
        with pytest.raises(ParamsError):
            parse_params(CellsParams, ["detector=blobs", extra])


def test_the_percentiles_take_their_extreme_values() -> None:
    """``low_percentile`` may be 0 (the minimum) and ``high_percentile`` 100 (the maximum), and no more."""
    params = parse_params(CellsParams, ["low_percentile=0", "high_percentile=100"])

    assert (params.low_percentile, params.high_percentile) == (0.0, 100.0)
    just_under = parse_params(
        CellsParams, ["low_percentile=49.9", "high_percentile=50.1"]
    )
    assert (just_under.low_percentile, just_under.high_percentile) == (49.9, 50.1)


@pytest.mark.parametrize(
    "bad",
    [
        "model_um_per_px=0",
        "model_um_per_px=-0.3",
        "model_um_per_px=nan",
        "model_um_per_px=inf",
        "model_um_per_px=11",
        "checked_um_per_px=0",
        "checked_um_per_px=nan",
        "checked_um_per_px=11",
        "low_percentile=-1",
        "low_percentile=50",
        "low_percentile=nan",
        "high_percentile=50",
        "high_percentile=101",
        "high_percentile=nan",
    ],
)
def test_the_preprocessing_parameters_are_bounded(bad: str) -> None:
    with pytest.raises(ParamsError):
        parse_params(CellsParams, [bad])


def test_a_resume_with_another_model_input_size_is_refused(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cells of two pre-processings are not one result: the runner refuses it on the parameters."""
    _seed_scan(tmp_path)
    _weights(tmp_path)
    stop = threading.Event()
    _install(monkeypatch, StubDetector(SCRIPT, on_frame=_stopping_at(2, stop)))
    _run(demo_microscope, tmp_path, stop=stop, model_um_per_px=0.1)
    counting = StubDetector(SCRIPT)
    _install(monkeypatch, counting)

    with pytest.raises(ResumeError) as caught:
        _run(demo_microscope, tmp_path, resume=True, model_um_per_px=0.11)

    assert "model_um_per_px" in str(caught.value)
    assert counting.seen == []


# --------------------------------------------------------------- bad frames


def _failure(
    microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> PluginRunError:
    _weights(tmp_path)
    _install(monkeypatch, StubDetector(SCRIPT))
    with pytest.raises(PluginRunError) as caught:
        _run(microscope, tmp_path)
    record = _stored(tmp_path)
    assert record.status is StageStatus.failed
    assert record.error is not None
    return caught.value


def test_a_missing_frame_file_names_it(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(tmp_path)
    (tmp_path / "scan" / "t" / "f1.tif").unlink()

    error = _failure(demo_microscope, tmp_path, monkeypatch)

    assert "scan/t/f1.tif" in error.reason
    assert error.reason.endswith("run scan again")
    assert "scan/t/f1.tif" in (_stored(tmp_path).error or "")


def test_a_corrupt_frame_file_names_it(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(tmp_path)
    (tmp_path / "scan" / "t" / "f1.tif").write_bytes(b"garbage")

    error = _failure(demo_microscope, tmp_path, monkeypatch)

    assert "scan/t/f1.tif" in error.reason
    assert error.reason.endswith("run scan again")


def test_a_tiff_error_that_is_not_a_value_error_still_names_the_file(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``TiffFileError`` is a ``ValueError`` only in the newer tifffile.

    The 2025.5.10 of Python 3.10 derives it from ``Exception``: the plan measured
    the newest release, and CI's py3.10 job got the raw error out of the stage.
    """

    class OlderTiffFileError(Exception):
        pass

    def corrupt(run_dir: Path, frame: ScanFrame) -> np.ndarray:
        raise OlderTiffFileError("not a TIFF file b'garb'")

    monkeypatch.setattr(tifffile, "TiffFileError", OlderTiffFileError)
    monkeypatch.setattr(plugin_mod, "load_frame", corrupt)
    _seed_scan(tmp_path)

    error = _failure(demo_microscope, tmp_path, monkeypatch)

    assert "scan/t/f0.tif" in error.reason
    assert "not a TIFF file" in error.reason
    assert error.reason.endswith("run scan again")


def test_a_frame_of_another_shape_than_the_scan_is_refused(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The matrix is only right on the scan's own shape (FM-73)."""
    _seed_scan(tmp_path)
    tifffile.imwrite(
        tmp_path / "scan" / "t" / "f1.tif", np.zeros((SHAPE[1], SHAPE[0]), np.uint16)
    )

    error = _failure(demo_microscope, tmp_path, monkeypatch)

    assert "scan/t/f1.tif" in error.reason
    assert str(SHAPE) in error.reason


def test_a_frame_that_is_not_two_dimensional_is_refused(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(tmp_path)
    tifffile.imwrite(
        tmp_path / "scan" / "t" / "f1.tif", np.zeros((*SHAPE, 3), np.uint8)
    )

    error = _failure(demo_microscope, tmp_path, monkeypatch)

    assert "scan/t/f1.tif" in error.reason


def test_a_float_frame_with_a_value_that_is_not_finite_is_refused(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(tmp_path)
    tifffile.imwrite(
        tmp_path / "scan" / "t" / "f1.tif", np.full(SHAPE, np.nan, np.float32)
    )

    error = _failure(demo_microscope, tmp_path, monkeypatch)

    assert "scan/t/f1.tif" in error.reason
    assert "finite" in error.reason


def test_a_scan_with_no_frame_is_refused(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(tmp_path, frames=[])

    error = _failure(demo_microscope, tmp_path, monkeypatch)

    assert "no frame" in error.reason


def test_a_label_the_detector_does_not_declare_is_refused(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(tmp_path)
    _weights(tmp_path)
    alien = Detection(25.0, 15.0, 35.0, 25.0, "alien", 0.9)
    _install(monkeypatch, StubDetector({0: [alien]}))

    with pytest.raises(PluginRunError) as caught:
        _run(demo_microscope, tmp_path)

    assert "alien" in caught.value.reason


# ---------------------------------------------------------------- stop, resume


def test_a_resume_skips_the_wells_done(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(tmp_path)
    _weights(tmp_path)
    stopped = _stop_in_the_second_well(demo_microscope, tmp_path, monkeypatch)
    assert stopped.status is StageStatus.stopped
    assert stopped.checkpoint is not None
    assert stopped.checkpoint["wells_done"] == ["A1"]
    token = stopped.checkpoint["token"]
    partial = tmp_path / "cells" / f"{token}.partial.json"
    assert partial.is_file()

    counting = StubDetector(SCRIPT)
    _install(monkeypatch, counting)
    record = _run(demo_microscope, tmp_path, resume=True)

    assert counting.seen == [2, 3]  # only A2's frames
    result = _result(record)
    assert result.file == f"cells/{token}.json"
    assert result.merged == 2  # one per well: A1's came through the checkpoint
    assert result.counts == {
        "A1": {"single": 1, "doublet": 1, "debris": 0},
        "A2": {"single": 1, "doublet": 1, "debris": 0},
    }
    assert {(c.well, c.label, c.frame) for c in _cells(tmp_path, record)} == {
        ("A1", "doublet", 0),
        ("A1", "single", 1),
        ("A2", "doublet", 2),
        ("A2", "single", 3),
    }
    assert not partial.exists()


def test_a_resume_with_other_weights_is_refused(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(tmp_path)
    _weights(tmp_path)
    _stop_in_the_second_well(demo_microscope, tmp_path, monkeypatch)
    _weights(tmp_path, b"weights, version 2")
    counting = StubDetector(SCRIPT)
    _install(monkeypatch, counting)

    with pytest.raises(PluginRunError) as caught:
        _run(demo_microscope, tmp_path, resume=True)

    assert "weights changed" in caught.value.reason
    assert "without --resume" in caught.value.reason
    assert counting.seen == []


def test_a_resume_with_another_preprocessing_is_refused(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The weights and the parameters are the same: only the identity's second half says it."""
    _seed_scan(tmp_path)
    _weights(tmp_path)
    _stop_in_the_second_well(demo_microscope, tmp_path, monkeypatch)
    # The card's percentiles change between the two attempts, as a pull request on
    # the card would change them: no parameter of the run does.
    changed = model_card(DEFAULT_MODEL_NAME).model_copy(
        update={
            "preprocessing": ModelPreprocessing(
                percentiles=(1.0, 99.5), model_um_per_px=None
            )
        }
    )
    monkeypatch.setattr(plugin_mod, "model_card", lambda name: changed)
    counting = StubDetector(SCRIPT)
    _install(monkeypatch, counting)

    with pytest.raises(PluginRunError) as caught:
        _run(demo_microscope, tmp_path, resume=True)

    assert "the pre-processing changed" in caught.value.reason
    assert "weights changed" not in caught.value.reason
    assert "without --resume" in caught.value.reason
    assert counting.seen == []


def test_a_checkpoint_saved_before_identities_asks_to_run_again(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resume across #684 is refused: its checkpoint holds ``model_sha256`` and no ``identity``."""
    _seed_scan(tmp_path)
    _weights(tmp_path)
    stopped = _stop_in_the_second_well(demo_microscope, tmp_path, monkeypatch)
    assert stopped.checkpoint is not None
    before = stopped.model_copy(
        update={
            "checkpoint": {
                "token": stopped.checkpoint["token"],
                "wells_done": ["A1"],
                "model_sha256": hashlib.sha256(WEIGHTS).hexdigest(),
                "merged": 0,
            }
        }
    )
    write_stage(tmp_path / "run.json", before)
    counting = StubDetector(SCRIPT)
    _install(monkeypatch, counting)

    with pytest.raises(ResumeError) as caught:
        _run(demo_microscope, tmp_path, resume=True)

    assert "without --resume" in str(caught.value)
    assert "identity" in str(caught.value)
    assert counting.seen == []


def test_a_cells_result_recorded_before_identities_still_loads(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run documents on shares and datasets (#595) hold results without the two new fields."""
    _seed_scan(tmp_path)
    _weights(tmp_path)
    _install(monkeypatch, StubDetector({}))
    stored = dict(_run(demo_microscope, tmp_path).result or {})
    stored.pop("model_identity")
    stored.pop("model_runs_on")
    stored["model"] = "w.pth"  # what the result said then: the weights' file name

    old = CellsResult.model_validate(stored)

    assert old.model == "w.pth"
    assert old.model_identity is None
    assert old.model_runs_on is None


def test_a_lost_partial_file_redoes_every_well(
    demo_microscope: Microscope,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _seed_scan(tmp_path)
    _weights(tmp_path)
    stopped = _stop_in_the_second_well(demo_microscope, tmp_path, monkeypatch)
    assert stopped.checkpoint is not None
    token = stopped.checkpoint["token"]
    (tmp_path / "cells" / f"{token}.partial.json").unlink()
    counting = StubDetector(SCRIPT)
    _install(monkeypatch, counting)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        record = _run(demo_microscope, tmp_path, resume=True)

    assert counting.seen == [0, 1, 2, 3]
    assert any(
        f"{token}.partial.json" in r.getMessage()
        for r in caplog.records
        if r.name == LOGGER and r.levelno == logging.WARNING
    )
    result = _result(record)
    assert result.file == f"cells/{token}.json"
    assert (
        result.merged == 2
    )  # not 3: the checkpoint's count was dropped with the cells
    assert result.counts["A1"] == {"single": 1, "doublet": 1, "debris": 0}
    assert result.counts["A2"] == {"single": 1, "doublet": 1, "debris": 0}


def test_a_partial_file_that_does_not_validate_redoes_every_well(
    demo_microscope: Microscope,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _seed_scan(tmp_path)
    _weights(tmp_path)
    stopped = _stop_in_the_second_well(demo_microscope, tmp_path, monkeypatch)
    assert stopped.checkpoint is not None
    token = stopped.checkpoint["token"]
    (tmp_path / "cells" / f"{token}.partial.json").write_bytes(b'{"cells": [1]}')
    counting = StubDetector(SCRIPT)
    _install(monkeypatch, counting)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        record = _run(demo_microscope, tmp_path, resume=True)

    assert counting.seen == [0, 1, 2, 3]
    assert [r for r in caplog.records if r.name == LOGGER]
    assert _result(record).merged == 2


def test_a_crash_between_the_partial_file_and_the_checkpoint_counts_no_well_twice(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The partial file is written first, so it can hold a well the checkpoint does not name."""
    _seed_scan(tmp_path)
    _weights(tmp_path)
    _install(monkeypatch, StubDetector(SCRIPT))
    saved = RunContext.save_checkpoint
    calls: list[int] = []

    def crash_on_the_second(self: RunContext, state: BaseModel) -> None:
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("the run document is gone")
        saved(self, state)

    monkeypatch.setattr(RunContext, "save_checkpoint", crash_on_the_second)
    with pytest.raises(RuntimeError, match="gone"):
        _run(demo_microscope, tmp_path)
    monkeypatch.undo()
    failed = _stored(tmp_path)
    assert failed.status is StageStatus.failed
    assert failed.checkpoint is not None
    assert failed.checkpoint["wells_done"] == ["A1"]
    token = failed.checkpoint["token"]
    partial = CellsFile.model_validate_json(
        (tmp_path / "cells" / f"{token}.partial.json").read_bytes()
    )
    assert {c.well for c in partial.cells} == {"A1", "A2"}  # more than the checkpoint

    counting = StubDetector(SCRIPT)
    _install(monkeypatch, counting)
    record = _run(demo_microscope, tmp_path, resume=True)

    assert counting.seen == [2, 3]
    result = _result(record)
    assert result.counts["A2"] == {"single": 1, "doublet": 1, "debris": 0}
    assert len(_cells(tmp_path, record)) == 4
    assert result.merged == 2


def test_the_partial_file_left_behind_is_a_warning(
    demo_microscope: Microscope,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The result is whole, so a file that cannot be removed does not fail the stage."""
    _seed_scan(tmp_path)
    _weights(tmp_path)
    _install(monkeypatch, StubDetector(SCRIPT))

    def locked(action: Callable[[], object]) -> object:
        raise PermissionError(13, "in use")

    monkeypatch.setattr(plugin_mod, "retry_on_permission_error", locked)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        record = _run(demo_microscope, tmp_path)

    result = _result(record)
    token = result.file.removeprefix("cells/").removesuffix(".json")
    assert (tmp_path / "cells" / f"{token}.partial.json").is_file()
    assert any(
        f"cells/{token}.partial.json" in r.getMessage()
        for r in caplog.records
        if r.name == LOGGER and r.levelno == logging.WARNING
    )
    assert str(tmp_path) not in " ".join(r.getMessage() for r in caplog.records)


def test_progress_counts_every_frame_once(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(tmp_path)
    _weights(tmp_path)
    first: list[Progress] = []
    stop = threading.Event()
    _install(monkeypatch, StubDetector(SCRIPT, on_frame=_stopping_at(2, stop)))
    _run(demo_microscope, tmp_path, stop=stop, on_progress=first.append)
    second: list[Progress] = []
    _install(monkeypatch, StubDetector(SCRIPT))

    _run(demo_microscope, tmp_path, resume=True, on_progress=second.append)

    assert [p.done for p in first] == [1, 2, 3]
    assert [p.done for p in second] == [3, 4]  # A1's two frames were done already
    assert {p.total for p in first + second} == {4}
    assert second[-1].done == len(TWO_WELLS)
    assert second[0].message == "A2 f2"


# ------------------------------------------------- review of #23: what was missed


# A3 has two frames: a stop asked for during its first one is looked at before the second.
THREE_WELLS = [
    ("A1", 0.0),
    ("A1", 30.0),
    ("A2", 0.0),
    ("A2", 30.0),
    ("A3", 0.0),
    ("A3", 30.0),
]
SCRIPT_OF_THREE = {**SCRIPT, 4: [OWN]}


def test_the_parameters_default_to_the_designs_values() -> None:
    params = CellsParams()

    # Unset is the card's: no default of its own, so that a blobs run records none.
    assert params.model is None
    assert params.model_path is None
    assert params.device == "auto"
    assert params.score_threshold == 0.5
    assert params.max_overlap == 0.7
    assert params.merge_radius_um == 5.0
    assert params.model_um_per_px is None
    assert (params.low_percentile, params.high_percentile) == (None, None)
    assert params.checked_um_per_px is None


def test_seconds_per_frame_is_the_mean_of_the_detect_times(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(tmp_path, frames=[("A1", 0.0), ("A1", 30.0)])
    _weights(tmp_path)
    _install(monkeypatch, StubDetector({}))
    # Two detects, which last 1 s and 3 s: the mean is 2, not the sum.
    ticks = iter([0.0, 1.0, 10.0, 13.0])
    monkeypatch.setattr(
        plugin_mod, "time", SimpleNamespace(perf_counter=lambda: next(ticks))
    )

    record = _run(demo_microscope, tmp_path)

    assert _result(record).seconds_per_frame == 2.0


def test_the_merge_uses_the_scans_frame_shape_and_not_its_transpose(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The unit tests pin the merge on a shape; this pins the shape the plugin gives it (FM-73).

    One cell, at column 55 of the first frame (24.5 px from its right edge) and
    at column 12 of the second (12.5 px from its left edge). With the shape the
    wrong way round the first view would be 4.5 px from an edge and lose.
    """
    _seed_scan(tmp_path, frames=[("A1", 0.0), ("A1", 21.0)])
    _weights(tmp_path)
    _install(
        monkeypatch,
        StubDetector(
            {
                0: [Detection(25.0, 50.0, 35.0, 60.0, "single", 0.9)],
                1: [Detection(25.0, 7.0, 35.0, 17.0, "single", 0.9)],
            }
        ),
    )

    record = _run(demo_microscope, tmp_path)

    assert _result(record).merged == 1
    assert [c.frame for c in _cells(tmp_path, record)] == [0]


def test_a_fallback_resume_does_not_leave_a_checkpoint_that_claims_more_than_the_partial_holds(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The partial file is lost, and the fallback's first checkpoint write fails.

    The fallback rewrites the partial with the first well only; a checkpoint
    that still named two wells would make the next resume skip a well whose
    cells are not in the file, and record a ``done`` result with zeros (FM-111).
    """
    _seed_scan(tmp_path, frames=THREE_WELLS)
    _weights(tmp_path)
    stop = threading.Event()
    _install(monkeypatch, StubDetector(SCRIPT_OF_THREE, on_frame=_stopping_at(4, stop)))
    stopped = _run(demo_microscope, tmp_path, stop=stop)
    assert stopped.checkpoint is not None
    assert stopped.checkpoint["wells_done"] == ["A1", "A2"]
    token = stopped.checkpoint["token"]
    (tmp_path / "cells" / f"{token}.partial.json").unlink()

    _install(monkeypatch, StubDetector(SCRIPT_OF_THREE))
    saved = RunContext.save_checkpoint
    calls: list[BaseModel] = []

    def crash_on_the_second(self: RunContext, state: BaseModel) -> None:
        calls.append(state)
        if len(calls) == 2:  # the first is the fallback's own; the second names A1
            raise RuntimeError("the run document is gone")
        saved(self, state)

    monkeypatch.setattr(RunContext, "save_checkpoint", crash_on_the_second)
    with pytest.raises(RuntimeError, match="gone"):
        _run(demo_microscope, tmp_path, resume=True)
    monkeypatch.undo()
    failed = _stored(tmp_path)
    assert failed.checkpoint is not None
    assert failed.checkpoint["wells_done"] == []

    counting = StubDetector(SCRIPT_OF_THREE)
    _install(monkeypatch, counting)
    result = _result(_run(demo_microscope, tmp_path, resume=True))

    assert counting.seen == [0, 1, 2, 3, 4, 5]
    assert result.counts["A2"] == {"single": 1, "doublet": 1, "debris": 0}
    assert result.counts["A3"] == {"single": 0, "doublet": 1, "debris": 0}


def test_a_partial_file_with_a_label_the_detector_does_not_declare_redoes_every_well(
    demo_microscope: Microscope,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _seed_scan(tmp_path)
    _weights(tmp_path)
    stopped = _stop_in_the_second_well(demo_microscope, tmp_path, monkeypatch)
    assert stopped.checkpoint is not None
    token = stopped.checkpoint["token"]
    alien = Cell(
        well="A1",
        x_um=0.0,
        y_um=0.0,
        label="alien",
        score=0.5,
        frame=0,
        box_px=(0.0, 0.0, 1.0, 1.0),
    )
    partial = tmp_path / "cells" / f"{token}.partial.json"
    partial.write_bytes(CellsFile(cells=(alien,)).model_dump_json().encode("ascii"))
    counting = StubDetector(SCRIPT)
    _install(monkeypatch, counting)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        record = _run(demo_microscope, tmp_path, resume=True)

    assert counting.seen == [0, 1, 2, 3]
    assert any("alien" in r.getMessage() for r in caplog.records if r.name == LOGGER)
    assert {c.label for c in _cells(tmp_path, record)} == {"single", "doublet"}


def test_a_partial_file_that_is_locked_for_a_moment_is_still_used(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An antivirus holds the file the stage wrote a moment ago: the read is retried (FM-24)."""
    _seed_scan(tmp_path)
    _weights(tmp_path)
    _stop_in_the_second_well(demo_microscope, tmp_path, monkeypatch)
    counting = StubDetector(SCRIPT)
    _install(monkeypatch, counting)
    real = Path.read_bytes
    refused: list[str] = []

    def locked_once(self: Path) -> bytes:
        if self.name.endswith(".partial.json") and not refused:
            refused.append(self.name)
            raise PermissionError(13, "in use")
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", locked_once)

    record = _run(demo_microscope, tmp_path, resume=True)

    assert refused
    assert counting.seen == [2, 3]  # the partial was read in the end
    assert _result(record).merged == 2


def test_both_cells_files_are_written_atomically(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never a half file, after a Ctrl-C or a full disk: every write goes through ``write_atomically`` (FM-38)."""
    _seed_scan(tmp_path)
    _weights(tmp_path)
    _install(monkeypatch, StubDetector(SCRIPT))
    real = plugin_mod.write_atomically
    written: list[str] = []

    def spy(path: Path, data: bytes, **kwargs: Any) -> None:
        written.append(path.name)
        real(path, data, **kwargs)

    monkeypatch.setattr(plugin_mod, "write_atomically", spy)

    record = _run(demo_microscope, tmp_path)

    token = _result(record).file.removeprefix("cells/").removesuffix(".json")
    # One partial file per well, then the final one.
    assert written == [f"{token}.partial.json"] * 2 + [f"{token}.json"]


def test_a_final_file_that_cannot_be_written_fails_the_stage_and_a_resume_finishes_it(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The partial file is the only saved work: it stays until the final file exists."""
    _seed_scan(tmp_path)
    _weights(tmp_path)
    _install(monkeypatch, StubDetector(SCRIPT))
    real = plugin_mod.write_atomically

    def disk_full_for_the_final_file(path: Path, data: bytes, **kwargs: Any) -> None:
        if not path.name.endswith(".partial.json"):
            raise AtomicWriteError(OSError(28, "No space left on device"))
        real(path, data, **kwargs)

    monkeypatch.setattr(plugin_mod, "write_atomically", disk_full_for_the_final_file)
    with pytest.raises(PluginRunError) as caught:
        _run(demo_microscope, tmp_path)
    failed = _stored(tmp_path)
    assert failed.status is StageStatus.failed
    assert failed.checkpoint is not None
    assert failed.checkpoint["wells_done"] == ["A1", "A2"]
    token = failed.checkpoint["token"]
    assert f"cells/{token}.json" in caught.value.reason
    assert "No space left on device" in caught.value.reason
    assert "--resume" in caught.value.reason
    assert (tmp_path / "cells" / f"{token}.partial.json").is_file()

    monkeypatch.setattr(plugin_mod, "write_atomically", real)
    counting = StubDetector(SCRIPT)
    _install(monkeypatch, counting)
    record = _run(demo_microscope, tmp_path, resume=True)

    assert counting.seen == []  # every well was done
    result = _result(record)
    assert result.seconds_per_frame is None  # unknown, not 0: nothing was detected
    assert result.merged == 2
    assert len(_cells(tmp_path, record)) == 4


def test_a_refused_rename_names_the_file_that_was_kept(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the rename failed: the whole new content waits in a hidden file, which the error names (FM-38)."""
    _seed_scan(tmp_path)
    _weights(tmp_path)
    _install(monkeypatch, StubDetector(SCRIPT))
    kept = tmp_path / "cells" / ".abc.partial.json.1234abcd.smc-tmp"

    def rename_refused(path: Path, data: bytes, **kwargs: Any) -> None:
        raise AtomicWriteError(PermissionError(13, "Access is denied"), kept=kept)

    monkeypatch.setattr(plugin_mod, "write_atomically", rename_refused)

    with pytest.raises(PluginRunError) as caught:
        _run(demo_microscope, tmp_path)

    reason = caught.value.reason
    assert "Access is denied" in reason
    assert "cells/.abc.partial.json.1234abcd.smc-tmp" in reason
    assert (
        "free some space" not in reason
    )  # the advice for a lock is not for a full disk
    assert str(tmp_path) not in reason


def test_a_model_path_that_is_a_folder_is_a_plugin_run_error_with_the_models_words(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    """The real ``load_model``: a folder where the weights should be is no file, on every OS."""
    _seed_scan(tmp_path)

    with pytest.raises(PluginRunError) as caught:
        _run(demo_microscope, tmp_path, model_path=tmp_path)

    assert "no model at" in caught.value.reason
    assert "cell_classes_model-cat-v2.pth" in caught.value.reason
    assert "SMC_MODELS" in caught.value.reason
    assert _stored(tmp_path).status is StageStatus.failed


# ------------------------------------------ detector=blobs, for the synthetic plate (#260)

BLOB_SHA256 = hashlib.sha256(BLOB_DETECTOR_IDENTITY.encode("utf-8")).hexdigest()


def _blobs(microscope: Microscope, tmp_path: Path, **params: Any) -> StageRecord:
    """``_run`` with the blob detector: no weights file, no model path."""
    return _run(microscope, tmp_path, detector="blobs", model_path=None, **params)


def _refuse_load_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def refused(*args: object, **kwargs: object) -> object:
        raise AssertionError("detector=blobs must not load a model")

    monkeypatch.setattr(plugin_mod, "load_model", refused)


def test_cells_with_detector_blobs_needs_neither_torch_nor_weights(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(tmp_path)  # and no weights file: there is nothing to look for
    monkeypatch.setitem(sys.modules, "torch", None)  # `import torch` raises
    _refuse_load_model(monkeypatch)

    result = _result(_blobs(demo_microscope, tmp_path))

    assert result.model == BLOB_DETECTOR == "blobs"
    assert result.model_sha256 == BLOB_SHA256
    assert result.model_identity == model_identity(BLOB_SHA256, None)
    assert result.model_runs_on == "local:cpu"
    assert result.device == "cpu"
    assert result.classes == ("single", "doublet")
    assert result.counts == {
        "A1": {"single": 0, "doublet": 0},
        "A2": {"single": 0, "doublet": 0},
    }
    assert any("blob detector" in line for line in result.assumed)
    assert any("detector=blobs" in line for line in result.assumed)


def test_the_blob_detector_says_in_the_result_that_its_counts_are_not_cells(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    _seed_scan(tmp_path)

    result = _result(_blobs(demo_microscope, tmp_path, merge_radius_um=0.0))

    assert result.assumed == (
        "cells were found by the blob detector (detector=blobs), which counts bright "
        "round spots for the synthetic plate of the demo-sample profile; on a real "
        "sample its counts are not cells",
    )


def test_the_trained_model_leaves_no_blob_statement_in_its_result(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(tmp_path)
    _weights(tmp_path)
    _install(monkeypatch, StubDetector({}))

    result = _result(_run(demo_microscope, tmp_path))

    assert not any("blob" in line for line in result.assumed)
    assert result.model == "cell-classes-v2"


def test_cells_with_detector_blobs_finds_a_spot_where_its_frame_says(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    # A frame of 60 x 80, not square (FM-73): a 10 x 10 spot at rows 25-34 and
    # columns 35-44, which is `(col, row) = (40, 30)` by its box, a quarter of a
    # micron from the stage position of the centre pixel (39.5, 29.5) at 0.5 um.
    _seed_scan(tmp_path, [("A1", 0.0)])
    frame = np.random.default_rng(0).normal(1000, 20, SHAPE)
    frame[25:35, 35:45] += 3000
    tifffile.imwrite(
        tmp_path / "scan" / "t" / "f0.tif", np.rint(frame).astype(np.uint16)
    )

    record = _blobs(demo_microscope, tmp_path)

    (cell,) = _cells(tmp_path, record)
    assert (cell.well, cell.label, cell.frame) == ("A1", "single", 0)
    assert (cell.x_um, cell.y_um) == pytest.approx((0.25, -0.25), abs=0.01)
    assert _result(record).counts == {"A1": {"single": 1, "doublet": 0}}


def test_a_resume_with_another_detector_is_refused(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The trained model first, the blob detector on the resume: cells of two
    # detectors are not one result (FM-111). The runner refuses it first, on the
    # parameters, which say `detector`.
    _seed_scan(tmp_path)
    _weights(tmp_path)
    _stop_in_the_second_well(demo_microscope, tmp_path, monkeypatch)

    with pytest.raises(ResumeError) as caught:
        _blobs(demo_microscope, tmp_path, resume=True)

    assert "parameters differ" in str(caught.value)
    assert "detector" in str(caught.value)


def test_a_resume_from_the_blob_detector_to_the_model_is_refused(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(tmp_path)
    _weights(tmp_path)
    stopped = _blobs_stopped_after_the_first_well(demo_microscope, tmp_path)
    assert stopped.status is StageStatus.stopped
    counting = StubDetector(SCRIPT)
    _install(monkeypatch, counting)

    with pytest.raises(ResumeError) as caught:
        _run(demo_microscope, tmp_path, resume=True)

    assert "detector" in str(caught.value)
    assert counting.seen == []


def test_a_resume_after_the_blob_detectors_rules_changed_is_refused(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The parameters are the same on both sides, so only the identity string
    # tells that another version of the rules made the cells already kept: the
    # SHA-256 of that string is checked like the weights' (FM-111).
    _seed_scan(tmp_path)
    _blobs_stopped_after_the_first_well(demo_microscope, tmp_path)
    monkeypatch.setattr(
        blobs_mod, "BLOB_DETECTOR_IDENTITY", BLOB_DETECTOR_IDENTITY + ", version 2"
    )

    with pytest.raises(PluginRunError) as caught:
        _blobs(demo_microscope, tmp_path, resume=True)

    assert "weights changed" in caught.value.reason
    assert "without --resume" in caught.value.reason


def _blobs_stopped_after_the_first_well(
    microscope: Microscope, tmp_path: Path
) -> StageRecord:
    stop = threading.Event()

    def stop_after_the_first_well(progress: Progress) -> None:
        if progress.done == 2:  # A1's two frames
            stop.set()

    return _blobs(
        microscope, tmp_path, stop=stop, on_progress=stop_after_the_first_well
    )


def test_a_resume_with_the_same_blob_detector_continues(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    _seed_scan(tmp_path)
    _blobs_stopped_after_the_first_well(demo_microscope, tmp_path)

    record = _blobs(demo_microscope, tmp_path, resume=True)

    assert set(_result(record).counts) == {"A1", "A2"}


@pytest.mark.parametrize(
    ("params", "phrase"),
    [
        (
            ["detector=blobs", "model_path=/some/w.pth"],
            "model_path is for detector=model",
        ),
        (["detector=blobs", "device=cuda"], "the blob detector runs on the cpu"),
        (["detector=blobs", "device=mps"], "the blob detector runs on the cpu"),
    ],
)
def test_detector_blobs_refuses_a_model_path_and_a_gpu(
    params: list[str], phrase: str
) -> None:
    with pytest.raises(ParamsError) as caught:
        parse_params(CellsParams, params)

    assert phrase in str(caught.value)


@pytest.mark.parametrize("extra", [[], ["device=auto"], ["device=cpu"]])
def test_detector_blobs_takes_the_default_and_the_cpu(extra: list[str]) -> None:
    params = parse_params(CellsParams, ["detector=blobs", *extra])

    assert params.detector == "blobs"
    assert params.model_path is None


def test_the_model_detector_still_takes_a_model_path_and_a_gpu() -> None:
    params = parse_params(CellsParams, ["model_path=/some/w.pth", "device=cuda"])

    assert params.detector == "model"
    assert params.device == "cuda"


def test_a_detector_that_is_neither_the_model_nor_blobs_is_refused() -> None:
    with pytest.raises(ParamsError) as caught:
        parse_params(CellsParams, ["detector=yolo"])

    assert "detector" in str(caught.value)


# --------------------------------------------- the model on a server (#685, inference.md §8)

#: A host and a port nothing else in a run folder says: the search for them is the test.
SERVER_HOST = "gpu-box.lab.test"
SERVER_PORT = "48151"
SERVER_ADDRESS = f"http://{SERVER_HOST}:{SERVER_PORT}"
CHECK_OK = "model server: cell-classes-v2 is served on cuda, identity b1160edae7dd+2df8feb4070d"


def _server(stub: StubDetector | None = None) -> FakeModelServer:
    """A model server for ``cell-classes-v2``: the real card's identity, a stub that says cuda."""
    card = model_card(DEFAULT_MODEL_NAME)
    model = stub_model(
        stub or StubDetector(SCRIPT, device="cuda"),
        name=card.name,
        weights_sha256=card.weights_sha256,
        preprocessing=card.preprocessing,
    )
    return FakeModelServer({card.name: model}, address=SERVER_ADDRESS)


def _run_served(
    microscope: Microscope,
    tmp_path: Path,
    fake: FakeModelServer,
    monkeypatch: pytest.MonkeyPatch,
    **kwargs: Any,
) -> StageRecord:
    """``cells`` through the real ``load_model`` with the variable set: no model path, no device."""
    fake.install(monkeypatch)
    return _run(microscope, tmp_path, model_path=None, device="auto", **kwargs)


def _detect_calls(fake: FakeModelServer) -> list[ServedCall]:
    return [c for c in fake.calls if c.path == "/detect/process"]


def _local_standin(monkeypatch: pytest.MonkeyPatch, stub: StubDetector) -> None:
    """A stand-in for the local ``load_model`` whose identity is the card's, as a real local load's is."""

    def load(card: ModelCard, **kwargs: Any) -> ResolvedModel:
        return stub_model(
            stub,
            name=card.name,
            weights_sha256=card.weights_sha256,
            preprocessing=kwargs["preprocessing"],
        )

    monkeypatch.setattr(plugin_mod, "load_model", load)


def test_cells_with_a_server_set_records_the_served_model_and_finds_what_the_local_stub_finds(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    card = model_card(DEFAULT_MODEL_NAME)
    served_dir, local_dir = tmp_path / "served", tmp_path / "local"
    _seed_scan(served_dir)
    _seed_scan(local_dir)
    fake = _server()

    record = _run_served(demo_microscope, served_dir, fake, monkeypatch)

    result = _result(record)
    assert result.model == "cell-classes-v2"
    assert result.model_runs_on == "served"
    assert result.model_identity == model_identity(
        card.weights_sha256, card.preprocessing
    )
    assert result.model_sha256 == card.weights_sha256
    # The server's device, as it said it, and never `auto`.
    assert result.device == "cuda"
    assert len(_detect_calls(fake)) == 4
    assert all(
        set(c.body) == {"model", "image", "pixel_size_um"} for c in fake.calls[1:]
    )
    assert [int(c.frame[0, 0]) for c in _detect_calls(fake)] == [0, 1, 2, 3]  # type: ignore[index]

    # The same counts and the same cells as the local stub finds on the same scan.
    _local_standin(monkeypatch, StubDetector(SCRIPT))
    local = _run(demo_microscope, local_dir, model_path=None, device="auto")
    assert _result(local).counts == result.counts
    assert {
        (c.well, c.label, c.frame, c.x_um, c.y_um) for c in _cells(local_dir, local)
    } == {
        (c.well, c.label, c.frame, c.x_um, c.y_um) for c in _cells(served_dir, record)
    }


def _everything_in(folder: Path) -> list[tuple[Path, bytes]]:
    files = [(p, p.read_bytes()) for p in sorted(folder.rglob("*")) if p.is_file()]
    assert len(files) > 3, "the run folder holds nothing to search"
    return files


@pytest.mark.parametrize(
    "scenario", ["done", "absent", "timeout", "fail_next", "identity"]
)
def test_the_address_is_nowhere_in_the_run_folder(
    demo_microscope: Microscope,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
) -> None:
    # A failed stage keeps its diagnosis in the run folder, and run documents are
    # attached to issues: the host and the port must not be in any byte of it (FM-50).
    _seed_scan(tmp_path)
    fake = _server()
    if scenario == "absent":
        fake.absent = True
    elif scenario == "timeout":
        fake.latency_s = 5.0
        monkeypatch.setenv("SMC_MODEL_TIMEOUT_S", "1")
    elif scenario == "fail_next":
        # The check's own call would use it up (and the stage would go on, the
        # check not refusing): it is the first frame's call that fails.
        armed = [True]

        def fail_the_first_frame(call: ServedCall) -> None:
            if call.path == "/detect/process" and armed:
                armed.clear()
                fake.fail_next = 1

        fake.on_call = fail_the_first_frame
    elif scenario == "identity":
        fake.identity_override = f"{'f' * 64}+{'e' * 64}"

    if scenario == "done":
        record = _run_served(demo_microscope, tmp_path, fake, monkeypatch)
        assert record.status is StageStatus.done
    else:
        with pytest.raises(PluginRunError):
            _run_served(demo_microscope, tmp_path, fake, monkeypatch)
        assert _stored(tmp_path).status is StageStatus.failed

    for path, data in _everything_in(tmp_path):
        assert SERVER_HOST.encode() not in data, path
        assert SERVER_PORT.encode() not in data, path
    # The failed ones did say what failed.
    if scenario != "done":
        assert _stored(tmp_path).error


def test_cells_with_the_server_absent_ends_failed_with_its_diagnosis_at_the_first_frame(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(tmp_path)
    fake = _server()
    fake.absent = True
    shown: list[Progress] = []

    with pytest.raises(PluginRunError) as caught:
        _run_served(
            demo_microscope, tmp_path, fake, monkeypatch, on_progress=shown.append
        )

    assert "does not answer" in caught.value.reason
    assert isinstance(caught.value.__cause__, ModelError)
    assert _stored(tmp_path).status is StageStatus.failed
    # The reason was on the screen before the first frame was read.
    assert shown[0].done == 0
    assert "does not answer" in shown[0].message
    assert [p.done for p in shown] == [0]
    # The server never saw a frame, and no cell was invented for the failure.
    assert fake.calls == []
    assert not (tmp_path / "cells").exists() or not list(
        (tmp_path / "cells").glob("*.json")
    )


def test_a_server_that_dies_in_the_second_well_fails_the_stage_and_a_resume_goes_on_from_the_first(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(tmp_path)
    fake = _server()

    def die_on_the_third_frame(call: ServedCall) -> None:
        if call.path == "/detect/process" and len(_detect_calls(fake)) == 3:
            fake.absent = True

    fake.on_call = die_on_the_third_frame
    with pytest.raises(PluginRunError) as caught:
        _run_served(demo_microscope, tmp_path, fake, monkeypatch)

    assert "does not answer" in caught.value.reason
    failed = _stored(tmp_path)
    assert failed.status is StageStatus.failed
    assert failed.checkpoint is not None
    assert failed.checkpoint["wells_done"] == ["A1"]
    assert failed.checkpoint["runs_on"] == "served"
    before = len(fake.calls)

    fake.absent = False
    fake.on_call = None
    record = _run(
        demo_microscope, tmp_path, model_path=None, device="auto", resume=True
    )

    # Only A2's frames were sent again, and the cells are those of an uninterrupted run.
    resent = [int(c.frame[0, 0]) for c in _detect_calls_after(fake, before)]  # type: ignore[index]
    assert resent == [2, 3]
    assert {(c.well, c.label, c.frame) for c in _cells(tmp_path, record)} == {
        ("A1", "doublet", 0),
        ("A1", "single", 1),
        ("A2", "doublet", 2),
        ("A2", "single", 3),
    }
    result = _result(record)
    assert result.model_runs_on == "served"
    assert result.merged == 2
    # One place for the whole run: no line about two.
    assert not [line for line in result.assumed if "earlier attempt" in line]


def _detect_calls_after(fake: FakeModelServer, count: int) -> list[ServedCall]:
    return [c for c in fake.calls[count:] if c.path == "/detect/process"]


def test_the_start_of_the_stage_reports_the_check(
    demo_microscope: Microscope,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _seed_scan(tmp_path)
    shown: list[Progress] = []
    caplog.set_level(logging.INFO, logger=LOGGER)

    _run_served(
        demo_microscope, tmp_path, _server(), monkeypatch, on_progress=shown.append
    )

    assert (shown[0].done, shown[0].total, shown[0].message) == (0, 4, CHECK_OK)
    assert [p.done for p in shown] == [0, 1, 2, 3, 4]
    ours = [
        r for r in caplog.records if r.name == LOGGER and r.getMessage() == CHECK_OK
    ]
    assert [r.levelno for r in ours] == [logging.INFO]


def test_the_start_of_the_stage_warns_when_the_check_is_not_ok(
    demo_microscope: Microscope,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _seed_scan(tmp_path)
    fake = _server()
    fake.hidden = {DEFAULT_MODEL_NAME}
    caplog.set_level(logging.INFO, logger=LOGGER)

    with pytest.raises(PluginRunError):
        _run_served(demo_microscope, tmp_path, fake, monkeypatch)

    warned = [r for r in caplog.records if r.name == LOGGER]
    assert [r.levelno for r in warned] == [logging.WARNING]
    assert "does not serve cell-classes-v2" in warned[0].getMessage()


def test_a_served_run_says_its_timings_in_assumed(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(tmp_path)
    # Four detects: the first lasts 2.31 s, the next three 0.12 s each.
    ticks = iter([0.0, 2.31, 10.0, 10.12, 20.0, 20.12, 30.0, 30.12])
    monkeypatch.setattr(
        plugin_mod, "time", SimpleNamespace(perf_counter=lambda: next(ticks))
    )

    record = _run_served(demo_microscope, tmp_path, _server(), monkeypatch)

    assert (
        "the model ran on the server: the first frame took 2.31 s and the next 3 "
        "frames 0.12 s on average, network included (the first can carry the "
        "server's start-up work); the server's device was cuda"
    ) in _result(record).assumed


def test_a_served_run_of_one_frame_says_only_its_first(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(tmp_path, frames=[("A1", 0.0)])
    ticks = iter([0.0, 2.31])
    monkeypatch.setattr(
        plugin_mod, "time", SimpleNamespace(perf_counter=lambda: next(ticks))
    )

    record = _run_served(demo_microscope, tmp_path, _server(), monkeypatch)

    assert (
        "the model ran on the server: the first frame took 2.31 s, network included "
        "(the first can carry the server's start-up work); the server's device was cuda"
    ) in _result(record).assumed


def test_a_local_run_says_nothing_of_a_server(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(tmp_path)
    _weights(tmp_path)
    _install(monkeypatch, StubDetector(SCRIPT))

    record = _run(demo_microscope, tmp_path)

    assert not [
        line
        for line in _result(record).assumed
        if "server" in line or "earlier attempt" in line
    ]


def _stopped_served_in_the_second_well(
    microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> StageRecord:
    """The first attempt of ``TWO_WELLS`` on a server: A1 done, stopped on the first frame of A2."""
    stop = threading.Event()
    fake = _server(StubDetector(SCRIPT, device="cuda", on_frame=_stopping_at(2, stop)))
    return _run_served(microscope, tmp_path, fake, monkeypatch, stop=stop)


def test_a_resume_on_another_place_says_so_in_assumed(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(tmp_path)
    stopped = _stopped_served_in_the_second_well(demo_microscope, tmp_path, monkeypatch)
    assert stopped.checkpoint is not None
    assert stopped.checkpoint["runs_on"] == "served"
    # The operator goes local (the server is gone): a resume is not refused for it.
    monkeypatch.delenv("SMC_MODEL_SERVER")
    local = StubDetector(SCRIPT)
    _local_standin(monkeypatch, local)

    record = _run(
        demo_microscope, tmp_path, model_path=None, device="auto", resume=True
    )

    result = _result(record)
    assert local.seen == [2, 3]
    assert result.model_runs_on == "local:cpu"
    assert (
        "2 of 4 frames were detected by an earlier attempt that ran on served; "
        "the others ran on local:cpu"
    ) in result.assumed
    assert {(c.well, c.label) for c in _cells(tmp_path, record)} == {
        ("A1", "doublet"),
        ("A1", "single"),
        ("A2", "doublet"),
        ("A2", "single"),
    }


def test_a_third_attempt_still_knows_the_first_wells_were_not_detected_where_it_runs(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Served for A1, local for A2, then local again: a checkpoint that kept only the
    # last place would say the first well was detected here.
    frames = [("A1", 0.0), ("A2", 0.0), ("A3", 0.0)]
    script = {0: [OWN], 1: [OWN], 2: [OWN]}
    _seed_scan(tmp_path, frames)
    stop = threading.Event()
    fake = _server(StubDetector(script, device="cuda", on_frame=_stopping_at(0, stop)))
    _run_served(demo_microscope, tmp_path, fake, monkeypatch, stop=stop)
    assert _stored(tmp_path).checkpoint["wells_done"] == ["A1"]  # type: ignore[index]

    monkeypatch.delenv("SMC_MODEL_SERVER")
    stop = threading.Event()
    _local_standin(monkeypatch, StubDetector(script, on_frame=_stopping_at(1, stop)))
    second = _run(
        demo_microscope,
        tmp_path,
        model_path=None,
        device="auto",
        resume=True,
        stop=stop,
    )
    assert second.checkpoint is not None
    assert second.checkpoint["wells_done"] == ["A1", "A2"]
    assert second.checkpoint["runs_on"] == "served+local:cpu"

    _local_standin(monkeypatch, StubDetector(script))
    third = _run(demo_microscope, tmp_path, model_path=None, device="auto", resume=True)

    assert (
        "2 of 3 frames were detected by an earlier attempt that ran on "
        "served+local:cpu; the others ran on local:cpu"
    ) in _result(third).assumed


def test_a_checkpoint_without_runs_on_still_resumes(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint saved before #685 has no ``runs_on``: it loads, and a resume says nothing of places."""
    _seed_scan(tmp_path)
    _weights(tmp_path)
    stopped = _stop_in_the_second_well(demo_microscope, tmp_path, monkeypatch)
    assert stopped.checkpoint is not None
    older = {k: v for k, v in stopped.checkpoint.items() if k != "runs_on"}
    assert "runs_on" in stopped.checkpoint
    write_stage(tmp_path / "run.json", stopped.model_copy(update={"checkpoint": older}))
    counting = StubDetector(SCRIPT)
    _install(monkeypatch, counting)

    record = _run(demo_microscope, tmp_path, resume=True)

    assert counting.seen == [2, 3]
    assert not [line for line in _result(record).assumed if "earlier attempt" in line]


def test_model_path_and_a_device_with_a_server_set_are_plugin_run_errors_with_the_servers_words(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(tmp_path)
    fake = _server()
    fake.install(monkeypatch)

    with pytest.raises(PluginRunError) as with_path:
        _run(demo_microscope, tmp_path, model_path=tmp_path / "w.pth", device="auto")
    with pytest.raises(PluginRunError) as with_device:
        _run(demo_microscope, tmp_path, model_path=None, device="cuda")
    with pytest.raises(PluginRunError) as with_rule:
        _run(
            demo_microscope,
            tmp_path,
            model_path=None,
            device="auto",
            low_percentile=1.0,
        )

    assert "model_path names a file on this PC" in with_path.value.reason
    assert "device=cuda is for a model that runs on this PC" in with_device.value.reason
    assert "the server runs the card's pre-processing" in with_rule.value.reason
    assert isinstance(with_path.value.__cause__, ModelError)
    assert fake.calls == []


def test_a_model_error_during_detect_is_a_plugin_run_error_with_its_text(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(tmp_path)
    _weights(tmp_path)

    def fall_over(position: int) -> None:
        if position == 2:
            raise ModelError("the model fell over on this frame")

    _install(monkeypatch, StubDetector(SCRIPT, on_frame=fall_over))

    with pytest.raises(PluginRunError) as caught:
        _run(demo_microscope, tmp_path)

    assert caught.value.reason == "the model fell over on this frame"
    assert isinstance(caught.value.__cause__, ModelError)
    failed = _stored(tmp_path)
    assert failed.status is StageStatus.failed
    # The wells done before it are kept.
    assert failed.checkpoint is not None
    assert failed.checkpoint["wells_done"] == ["A1"]
