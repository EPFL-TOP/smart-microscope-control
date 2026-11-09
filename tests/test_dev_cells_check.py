"""scripts/dev/cells_check.py: the reference set for the cell model and its measurement (m3 §9.4, #342).

The script is loaded from its file, as ``scripts/dev/loc.py`` is. The frames are
``PlateSample`` frames of 200 x 320 pixels at 0.5 um (not square, FM-73) with
three single spots and a pair, which the blob detector finds exactly (m3 §9.1), so
no test needs torch: where the model matters, ``load_detector`` of the script is
replaced by a stub that returns what the test scripts.

Spots are 25 um apart or more and 30 um from the frame's edges, so that no
expected number depends on a spot that the edge cuts.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import tifffile

from smc.plugins.cells.detector import Detection, Preprocessing
from smc.plugins.errors import PluginRunError
from smc.plugins.slots import ScanFrame, ScanResult
from smc.run import StageRecord, StageStatus, write_stage
from smc.testing import Blob, PlateSample

ROOT = Path(__file__).resolve().parents[1]
SHAPE = (200, 320)  # (rows, columns)
PIXEL_UM = 0.5

#: Where the spots are, in um from the frame centre (x to the right, y up).
SINGLES = [(-50.0, 25.0), (50.0, 25.0), (-50.0, -25.0)]
PAIR = [(26.0, -25.0), (34.0, -25.0)]


@pytest.fixture(scope="module")
def check() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "cells_check_script", ROOT / "scripts" / "dev" / "cells_check.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _frame(seed: int, *, spots: bool = True) -> npt.NDArray[Any]:
    """A frame with three single spots and a pair (or none): what blobs finds exactly."""
    blobs = [Blob(x, y, 5.0, 3000.0) for x, y in [*SINGLES, *PAIR]] if spots else []
    return np.asarray(
        PlateSample(blobs=blobs, seed=seed).render(
            0.0, 0.0, 0.0, shape=SHAPE, pixel_size_um=PIXEL_UM
        )
    )


def _seed_run(
    tmp_path: Path,
    frames: list[tuple[str, npt.NDArray[Any]]],
    *,
    names: list[str] | None = None,
    status: StageStatus = StageStatus.done,
) -> Path:
    """A run folder whose ``scan`` holds ``frames`` (a well and pixels each); returns its run.json."""
    run_dir = tmp_path / "run"
    folder = run_dir / "scan" / "t"
    folder.mkdir(parents=True)
    fields: dict[str, int] = {}
    scan_frames = []
    for position, (well, pixels) in enumerate(frames):
        index = fields.get(well, 0)
        fields[well] = index + 1
        name = names[position] if names else f"{well}_{index:04d}"
        tifffile.imwrite(folder / f"{name}.tif", pixels)
        scan_frames.append(
            ScanFrame(
                well=well,
                index=index,
                name=name,
                file=f"scan/t/{name}.tif",
                x_um=100.0 * position,
                y_um=-50.0 * position,
                z_um=None,
                focus="none",
                pfs_offset=None,
                t_s=float(position),
            )
        )
    scan = ScanResult(
        folder="scan/t",
        objective="40x",
        pixel_size_um=PIXEL_UM,
        matrix_um_per_px=((PIXEL_UM, 0.0), (0.0, -PIXEL_UM)),
        image_shape_px=SHAPE,
        exposure_ms=10.0,
        frames=tuple(scan_frames),
        failed=(),
    )
    then = datetime(2026, 11, 9, 9, 30, tzinfo=timezone.utc)
    write_stage(
        run_dir / "run.json",
        StageRecord(
            plugin="scan",
            provides="scan",
            attempt="s1",
            status=status,
            started_utc=then,
            finished_utc=then,
            profile="demo",
            dry_run=False,
            smc_version="0",
            params={},
            result=scan.model_dump(mode="json"),
        ),
    )
    return run_dir / "run.json"


def _two_wells(tmp_path: Path) -> Path:
    """A1 and A2 with five frames each: ten frames, ``A1_0000``..``A2_0004``."""
    frames = [(well, _frame(seed)) for seed, well in enumerate(["A1"] * 5 + ["A2"] * 5)]
    return _seed_run(tmp_path, frames)


def _val(dataset: Path) -> dict[str, Any]:
    loaded = json.loads((dataset / "annotations" / "val.json").read_text("utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _save_val(dataset: Path, payload: dict[str, Any]) -> None:
    (dataset / "annotations" / "val.json").write_text(
        json.dumps(payload, indent=1), encoding="utf-8"
    )


class _Scripted:
    """A detector that returns what the test scripted, whatever the frame."""

    classes = ("single", "doublet", "debris")
    device = "cpu"

    def __init__(self, found: list[Detection]) -> None:
        self._found = found

    def detect(self, frame: npt.NDArray[Any]) -> list[Detection]:
        return list(self._found)


def _stub_loader(
    monkeypatch: pytest.MonkeyPatch, check: ModuleType, found: list[Detection]
) -> list[dict[str, Any]]:
    """``cells_check.load_detector`` gives a scripted detector; its calls are returned."""
    calls: list[dict[str, Any]] = []

    def fake(
        model_path: Path | None,
        *,
        device: str,
        score_threshold: float,
        preprocessing: Preprocessing,
    ) -> _Scripted:
        calls.append(
            {
                "path": model_path,
                "device": device,
                "score": score_threshold,
                "preprocessing": preprocessing,
            }
        )
        return _Scripted(found)

    monkeypatch.setattr(check, "load_detector", fake)
    return calls


def _refusing_loader(monkeypatch: pytest.MonkeyPatch, check: ModuleType) -> None:
    def refused(*args: object, **kwargs: object) -> object:
        raise AssertionError("the model must not be loaded here")

    monkeypatch.setattr(check, "load_detector", refused)


def _weights(tmp_path: Path) -> Path:
    path = tmp_path / "w.pth"
    path.write_bytes(b"weights, version 1")
    return path


def _export(
    check: ModuleType, run_json: Path, out: Path, *args: str, prefill: str = "blobs"
) -> int:
    return int(
        check.main(["export", str(run_json), str(out), "--prefill", prefill, *args])
    )


# --------------------------------------------------------------------- export


def test_export_keeps_count_frames_evenly_spaced_in_the_wells_asked(
    check: ModuleType, tmp_path: Path
) -> None:
    run_json = _two_wells(tmp_path)
    out = tmp_path / "ref"

    code = _export(check, run_json, out, "--wells", "A2", "--count", "3")

    assert code == 0
    # A2 holds five frames, 0 to 4; three evenly spaced are 0, 2 and 4.
    assert sorted(p.name for p in (out / "images").iterdir()) == [
        "A2_0000.tif",
        "A2_0002.tif",
        "A2_0004.tif",
    ]
    manifest = json.loads((out / "manifest.json").read_text("utf-8"))
    assert [(i["well"], i["file_name"]) for i in manifest["images"]] == [
        ("A2", "A2_0000.tif"),
        ("A2", "A2_0002.tif"),
        ("A2", "A2_0004.tif"),
    ]
    assert [im["file_name"] for im in _val(out)["images"]] == [
        "A2_0000.tif",
        "A2_0002.tif",
        "A2_0004.tif",
    ]


def test_export_without_wells_spreads_the_count_over_every_frame_of_the_scan(
    check: ModuleType, tmp_path: Path
) -> None:
    run_json = _two_wells(tmp_path)
    out = tmp_path / "ref"

    assert _export(check, run_json, out, "--count", "4") == 0

    # linspace(0, 9, 4) rounds to 0, 3, 6, 9.
    names = sorted(p.name for p in (out / "images").iterdir())
    assert names == ["A1_0000.tif", "A1_0003.tif", "A2_0001.tif", "A2_0004.tif"]


def test_export_keeps_every_frame_when_there_are_fewer_than_the_count(
    check: ModuleType, tmp_path: Path
) -> None:
    run_json = _two_wells(tmp_path)
    out = tmp_path / "ref"

    assert _export(check, run_json, out, "--wells", "A1", "--count", "20") == 0

    assert len(list((out / "images").iterdir())) == 5


def test_export_copies_the_frames_unchanged(check: ModuleType, tmp_path: Path) -> None:
    run_json = _two_wells(tmp_path)
    out = tmp_path / "ref"

    assert _export(check, run_json, out, "--wells", "A1", "--count", "1") == 0

    source = run_json.parent / "scan" / "t" / "A1_0000.tif"
    assert (out / "images" / "A1_0000.tif").read_bytes() == source.read_bytes()


def test_export_refuses_a_folder_that_is_not_empty(
    check: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The owner's annotations are never overwritten; and the folder is looked at before the model loads (FM-30)."""
    run_json = _two_wells(tmp_path)
    out = tmp_path / "ref"
    (out / "annotations").mkdir(parents=True)
    precious = out / "annotations" / "val.json"
    precious.write_text('{"mine": true}', encoding="utf-8")
    _refusing_loader(monkeypatch, check)
    _weights(tmp_path)

    code = _export(check, run_json, out, prefill="model")

    assert code == 1
    assert precious.read_text("utf-8") == '{"mine": true}'
    assert not (out / "images").exists()
    assert "not empty" in capsys.readouterr().err


def test_export_takes_an_empty_folder(check: ModuleType, tmp_path: Path) -> None:
    run_json = _two_wells(tmp_path)
    out = tmp_path / "ref"
    out.mkdir()

    assert _export(check, run_json, out, "--count", "2") == 0

    assert (out / "annotations" / "val.json").is_file()


def test_export_writes_coco_boxes_columns_first(
    check: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A box of (row0, col0, row1, col1) is [x, y, w, h] = [col0, row0, col1 - col0, row1 - row0] in COCO (FM-73)."""
    run_json = _seed_run(tmp_path, [("A1", _frame(0))])
    weights = _weights(tmp_path)
    found = [Detection(20.0, 100.0, 40.0, 130.0, "doublet", 0.9)]
    calls = _stub_loader(monkeypatch, check, found)
    out = tmp_path / "ref"

    code = _export(check, run_json, out, "--model-path", str(weights), prefill="model")

    assert code == 0
    coco = _val(out)
    # The image is 200 rows high and 320 columns wide.
    (image,) = coco["images"]
    assert (image["height"], image["width"]) == (200, 320)
    assert image["reviewed"] is False
    assert coco["categories"] == [
        {"id": 1, "name": "single"},
        {"id": 2, "name": "doublet"},
        {"id": 3, "name": "debris"},
    ]
    (annotation,) = coco["annotations"]
    assert annotation["bbox"] == [100.0, 20.0, 30.0, 20.0]
    assert annotation["area"] == 600.0
    assert annotation["category_id"] == 2
    assert annotation["image_id"] == image["id"]
    assert annotation["auto"] is True
    assert annotation["score"] == pytest.approx(0.9)
    # The prefill asked for what the plan says: the score 0.3 and the model as trained.
    assert calls[0]["score"] == 0.3
    assert calls[0]["preprocessing"] == Preprocessing()
    assert calls[0]["path"] == weights


def test_the_prefill_keeps_the_better_label_of_one_object_and_drops_the_weak_boxes(
    check: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """As ``cells`` does: the model's second label for one object is dropped, and a box under the score is not offered."""
    run_json = _seed_run(tmp_path, [("A1", _frame(0))])
    weights = _weights(tmp_path)
    found = [
        # One object seen as a single (0.6) and as a doublet (0.9): 80 of the
        # smaller box's 100 px are shared, over the 0.7 the prefill suppresses at.
        Detection(20.0, 100.0, 30.0, 110.0, "single", 0.6),
        Detection(20.0, 102.0, 30.0, 112.0, "doublet", 0.9),
        # Far from them, under the prefill's score of 0.3.
        Detection(150.0, 10.0, 170.0, 30.0, "debris", 0.2),
        # Far from them, above it.
        Detection(150.0, 200.0, 170.0, 230.0, "debris", 0.3),
    ]
    _stub_loader(monkeypatch, check, found)
    out = tmp_path / "ref"

    code = _export(check, run_json, out, "--model-path", str(weights), prefill="model")

    assert code == 0
    kept = sorted((a["category_id"], a["bbox"]) for a in _val(out)["annotations"])
    assert kept == [
        (2, [102.0, 20.0, 10.0, 10.0]),
        (3, [200.0, 150.0, 30.0, 20.0]),
    ]


def test_export_prefills_the_boxes_of_the_blob_detector_when_asked(
    check: ModuleType, tmp_path: Path
) -> None:
    run_json = _seed_run(tmp_path, [("A1", _frame(0))])
    out = tmp_path / "ref"

    assert _export(check, run_json, out) == 0

    labels = sorted(a["category_id"] for a in _val(out)["annotations"])
    # Three singles (1) and a pair (2), every one still to be confirmed.
    assert labels == [1, 1, 1, 2]
    assert all(a["auto"] for a in _val(out)["annotations"])


def test_export_without_a_prefill_writes_no_annotation(
    check: ModuleType, tmp_path: Path
) -> None:
    run_json = _seed_run(tmp_path, [("A1", _frame(0))])
    out = tmp_path / "ref"

    assert _export(check, run_json, out, prefill="none") == 0

    assert _val(out)["annotations"] == []
    manifest = json.loads((out / "manifest.json").read_text("utf-8"))
    assert manifest["prefill"]["detector"] == "none"


def test_export_passes_the_model_input_size_from_the_scan_pixel_size(
    check: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_json = _seed_run(tmp_path, [("A1", _frame(0))])
    weights = _weights(tmp_path)
    calls = _stub_loader(monkeypatch, check, [])

    code = _export(
        check,
        run_json,
        tmp_path / "ref",
        "--model-path",
        str(weights),
        "--model-um-per-px",
        "0.25",
        "--prefill-score",
        "0.6",
        prefill="model",
    )

    assert code == 0
    # (200, 320) at 0.5 um shown at 0.25: twice the pixels, (400, 640).
    assert calls[0]["preprocessing"] == Preprocessing(input_size_px=(400, 640))
    assert calls[0]["score"] == 0.6
    manifest = json.loads((tmp_path / "ref" / "manifest.json").read_text("utf-8"))
    assert manifest["prefill"]["model_um_per_px"] == 0.25
    assert manifest["prefill"]["score"] == 0.6


def test_the_model_input_size_out_of_bounds_stops_the_export_before_anything_is_written(
    check: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_json = _seed_run(tmp_path, [("A1", _frame(0))])
    weights = _weights(tmp_path)
    _refusing_loader(monkeypatch, check)
    out = tmp_path / "ref"

    code = _export(
        check,
        run_json,
        out,
        "--model-path",
        str(weights),
        "--model-um-per-px",
        "0.5",
        prefill="model",
    )

    assert code == 1
    assert not out.exists()
    assert "200x320" in capsys.readouterr().err


def test_a_model_that_will_not_load_leaves_no_folder(
    check: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_json = _seed_run(tmp_path, [("A1", _frame(0))])

    def refused(*args: object, **kwargs: object) -> object:
        raise PluginRunError("cells", "no model at x.pth; put it there")

    monkeypatch.setattr(check, "load_detector", refused)
    out = tmp_path / "ref"

    assert _export(check, run_json, out, prefill="model") == 1

    assert not out.exists()
    assert "no model at x.pth" in capsys.readouterr().err


def test_export_refuses_two_frames_with_one_file_name(
    check: ModuleType, tmp_path: Path
) -> None:
    """Two files of one name would overwrite each other in ``images/``; a case-folding volume sees ``a`` and ``A`` as one."""
    run_json = _seed_run(
        tmp_path,
        [("A1", _frame(0)), ("A2", _frame(1))],
        names=["same", "SAME"],
    )
    out = tmp_path / "ref"

    assert _export(check, run_json, out) == 1

    assert not out.exists()


def test_export_names_the_wells_the_scan_has_when_one_is_unknown(
    check: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_json = _two_wells(tmp_path)

    code = _export(check, run_json, tmp_path / "ref", "--wells", "A1,Z9")

    assert code == 1
    error = capsys.readouterr().err
    assert "Z9" in error
    assert "A1" in error
    assert "A2" in error


@pytest.mark.parametrize("status", [StageStatus.failed, StageStatus.stopped])
def test_export_refuses_a_scan_that_did_not_finish(
    check: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    status: StageStatus,
) -> None:
    run_json = _seed_run(tmp_path, [("A1", _frame(0))], status=status)

    assert _export(check, run_json, tmp_path / "ref") == 1

    assert status.value in capsys.readouterr().err


def test_export_without_a_run_document_says_so(
    check: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _export(check, tmp_path / "nope.json", tmp_path / "ref") == 1

    assert "nope.json" in capsys.readouterr().err


def test_the_manifest_names_the_run_document_without_its_folder(
    check: ModuleType, tmp_path: Path
) -> None:
    """Run folders are attached to issues: the file's name, never its path (FM-50)."""
    run_json = _seed_run(tmp_path, [("A1", _frame(0)), ("A1", _frame(1))])
    out = tmp_path / "ref"

    assert _export(check, run_json, out) == 0

    text = (out / "manifest.json").read_text("utf-8")
    manifest = json.loads(text)
    assert manifest["run_document"] == "run.json"
    escaped = json.dumps(str(tmp_path))[1:-1]
    assert str(tmp_path) not in text
    assert escaped not in text
    assert tmp_path.as_posix() not in text
    assert manifest["scan"] == {
        "objective": "40x",
        "pixel_size_um": PIXEL_UM,
        "image_shape_px": [200, 320],
        "exposure_ms": 10.0,
    }
    assert manifest["prefill"]["detector"] == "blobs"
    assert datetime.fromisoformat(manifest["exported_utc"]).tzinfo is not None
    first = manifest["images"][0]
    assert first["frame_file"] == "scan/t/A1_0000.tif"
    assert (first["well"], first["x_um"], first["y_um"]) == ("A1", 0.0, 0.0)
    assert manifest["images"][1]["x_um"] == 100.0


def test_the_manifest_names_the_models_file_and_hash(
    check: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    run_json = _seed_run(tmp_path, [("A1", _frame(0))])
    weights = _weights(tmp_path)
    _stub_loader(monkeypatch, check, [])
    out = tmp_path / "ref"

    assert (
        _export(check, run_json, out, "--model-path", str(weights), prefill="model")
        == 0
    )

    prefill = json.loads((out / "manifest.json").read_text("utf-8"))["prefill"]
    assert prefill["model"] == "w.pth"
    assert prefill["model_sha256"] == hashlib.sha256(weights.read_bytes()).hexdigest()
    assert str(tmp_path) not in json.dumps(prefill)


# -------------------------------------------------------------------- measure


def _numbers(output: str, label: str) -> list[str]:
    """The five cells after ``label`` in the first row of the table that starts with it."""
    for line in output.splitlines():
        if line.startswith(label + " "):
            return line.split()[-5:]
    raise AssertionError(f"no row {label!r} in:\n{output}")


def _reviewed_dataset(tmp_path: Path, check: ModuleType) -> Path:
    """Two frames exported with the blobs prefill; the first is reviewed and corrected, the second is not.

    The first frame's reference is blobs' own boxes, with one single relabelled
    doublet, one single removed, and a debris box added in an empty corner. The
    second frame carries a box over nothing and is left unreviewed.
    """
    run_json = _seed_run(tmp_path, [("A1", _frame(3)), ("A1", _frame(4, spots=False))])
    out = tmp_path / "ref"
    assert _export(check, run_json, out) == 0
    coco = _val(out)
    first, second = coco["images"]
    own = [a for a in coco["annotations"] if a["image_id"] == first["id"]]
    singles = sorted((a for a in own if a["category_id"] == 1), key=lambda a: a["bbox"])
    assert len(singles) == 3
    assert len([a for a in own if a["category_id"] == 2]) == 1
    singles[0]["category_id"] = 2  # a single the owner calls a doublet
    removed = singles[1]  # a single the owner deletes
    coco["annotations"] = [a for a in coco["annotations"] if a is not removed]
    next_id = max(a["id"] for a in coco["annotations"]) + 1
    coco["annotations"] += [
        {  # debris in the empty bottom-right corner
            "id": next_id,
            "image_id": first["id"],
            "category_id": 3,
            "bbox": [270.0, 170.0, 30.0, 20.0],
            "area": 600.0,
            "iscrowd": 0,
            "cell": None,
            "auto": False,
            "score": None,
        },
        {  # a box over nothing, on the frame that is not reviewed
            "id": next_id + 1,
            "image_id": second["id"],
            "category_id": 1,
            "bbox": [100.0, 100.0, 24.0, 24.0],
            "area": 576.0,
            "iscrowd": 0,
            "cell": None,
            "auto": False,
            "score": None,
        },
    ]
    first["reviewed"] = True
    _save_val(out, coco)
    return out


def test_measure_scores_only_the_reviewed_frames(
    check: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = _reviewed_dataset(tmp_path, check)
    capsys.readouterr()

    code = check.main(["measure", str(dataset), "--detector", "blobs"])

    out = capsys.readouterr().out
    assert code == 0
    # reference / detected / matched, precision, recall: worked out from the corrected boxes.
    assert _numbers(out, "single") == ["1", "3", "1", "0.333", "1.000"]
    assert _numbers(out, "doublet") == ["2", "1", "1", "1.000", "0.500"]
    assert _numbers(out, "debris") == ["1", "0", "0", "-", "0.000"]
    assert _numbers(out, "any box") == ["4", "4", "3", "0.750", "0.750"]
    assert "reviewed frames: 1 of 2" in out


def test_measure_prints_the_confusion_of_the_non_zero_pairs(
    check: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = _reviewed_dataset(tmp_path, check)
    capsys.readouterr()

    assert check.main(["measure", str(dataset), "--detector", "blobs"]) == 0

    out = capsys.readouterr().out
    for line in [
        "single -> single: 1",
        "doublet -> single: 1",
        "doublet -> doublet: 1",
        "none -> single: 1",
        "debris -> none: 1",
    ]:
        assert line in out
    assert "single -> doublet" not in out


def test_measure_without_a_reviewed_frame_says_what_to_do(
    check: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_json = _seed_run(tmp_path, [("A1", _frame(0))])
    out = tmp_path / "ref"
    assert _export(check, run_json, out) == 0
    capsys.readouterr()

    code = check.main(["measure", str(out), "--detector", "blobs"])

    assert code == 1
    error = capsys.readouterr().err
    assert "reviewed" in error
    assert "/review" in error


def test_measure_without_a_dataset_says_so(
    check: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert check.main(["measure", str(tmp_path / "none"), "--detector", "blobs"]) == 1

    assert "annotations" in capsys.readouterr().err


def test_measure_sweeps_the_model_preprocessing(
    check: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _reviewed_dataset(tmp_path, check)
    weights = _weights(tmp_path)
    calls = _stub_loader(monkeypatch, check, [])

    code = check.main(
        [
            "measure",
            str(dataset),
            "--detector",
            "model",
            "--model-path",
            str(weights),
            "--model-um-per-px",
            "0.25,0.3",
            "--percentiles",
            "0.5:99.5,1:99",
            "--score-threshold",
            "0.6",
        ]
    )

    assert code == 0
    # 2 sizes x 2 percentile pairs, the sizes of the manifest's pixel size (0.5 um):
    # (200, 320) x 2 = (400, 640) and x 5/3 = (333, 534), written out by hand.
    assert [
        (c["preprocessing"].input_size_px, c["preprocessing"].percentiles)
        for c in calls
    ] == [
        ((400, 640), (0.5, 99.5)),
        ((400, 640), (1.0, 99.0)),
        ((333, 534), (0.5, 99.5)),
        ((333, 534), (1.0, 99.0)),
    ]
    assert {c["score"] for c in calls} == {0.6}
    assert {c["path"] for c in calls} == {weights}


def test_measure_native_keeps_the_models_own_resize(
    check: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _reviewed_dataset(tmp_path, check)
    weights = _weights(tmp_path)
    calls = _stub_loader(monkeypatch, check, [])

    code = check.main(
        [
            "measure",
            str(dataset),
            "--detector",
            "model",
            "--model-path",
            str(weights),
            "--model-um-per-px",
            "native",
        ]
    )

    assert code == 0
    assert [c["preprocessing"] for c in calls] == [Preprocessing()]


def test_measure_puts_the_models_file_and_hash_in_the_header(
    check: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import hashlib

    dataset = _reviewed_dataset(tmp_path, check)
    weights = _weights(tmp_path)
    _stub_loader(monkeypatch, check, [])
    capsys.readouterr()

    code = check.main(
        [
            "measure",
            str(dataset),
            "--detector",
            "model",
            "--model-path",
            str(weights),
            "--model-um-per-px",
            "0.25",
        ]
    )

    assert code == 0
    out = capsys.readouterr().out
    digits = hashlib.sha256(weights.read_bytes()).hexdigest()[:12]
    header = next(line for line in out.splitlines() if line.startswith("=== model"))
    assert "w.pth" in header
    assert digits in header
    assert "0.25 um per pixel" in header
    assert "400x640" in header
    assert str(tmp_path) not in out


def _detected_by_the_stub(
    check: ModuleType,
    dataset: Path,
    weights: Path,
    capsys: pytest.CaptureFixture[str],
    *options: str,
) -> list[str]:
    """The ``any box`` row (reference, detected, matched, ...) of a measure of the stub on the reviewed frame."""
    capsys.readouterr()
    code = check.main(
        [
            "measure",
            str(dataset),
            "--detector",
            "model",
            "--model-path",
            str(weights),
            *options,
        ]
    )
    assert code == 0
    return _numbers(capsys.readouterr().out, "any box")


def test_max_overlap_reaches_the_suppression_of_a_frame(
    check: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Two boxes sharing 0.5 of the smaller are one object at 0.4 and two at the default 0.7."""
    dataset = _reviewed_dataset(tmp_path, check)
    weights = _weights(tmp_path)
    _stub_loader(
        monkeypatch,
        check,
        [
            Detection(20.0, 100.0, 30.0, 110.0, "single", 0.6),
            Detection(20.0, 105.0, 30.0, 115.0, "doublet", 0.9),
        ],
    )

    default = _detected_by_the_stub(check, dataset, weights, capsys)
    strict = _detected_by_the_stub(
        check, dataset, weights, capsys, "--max-overlap", "0.4"
    )

    # The second column is the number of boxes detected.
    assert (default[1], strict[1]) == ("2", "1")


def test_match_iou_reaches_the_matching(
    check: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A box on the reference debris with an IoU of 0.6 is a match at 0.5 and none at 0.7."""
    dataset = _reviewed_dataset(tmp_path, check)
    weights = _weights(tmp_path)
    # The debris reference is rows 170-190, columns 270-300 (600 px). This box is
    # the same rows and columns 270-288 (360 px, all inside it): IoU 360 / 600 = 0.6.
    _stub_loader(
        monkeypatch, check, [Detection(170.0, 270.0, 190.0, 288.0, "debris", 0.9)]
    )

    loose = _detected_by_the_stub(check, dataset, weights, capsys, "--match-iou", "0.5")
    tight = _detected_by_the_stub(check, dataset, weights, capsys, "--match-iou", "0.7")

    # Third column: matched.
    assert (loose[2], tight[2]) == ("1", "0")


def test_measure_scores_the_model_stub_on_the_reference(
    check: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A model that finds the reference's debris and one box more, and nothing else: scored per class."""
    dataset = _reviewed_dataset(tmp_path, check)
    weights = _weights(tmp_path)
    # The reference holds 1 single, 2 doublets and 1 debris. The stub finds the
    # debris box exactly, and a box over nothing: debris 1/1 and 1/2.
    found = [
        Detection(170.0, 270.0, 190.0, 300.0, "debris", 0.9),
        Detection(5.0, 5.0, 25.0, 25.0, "debris", 0.9),
    ]
    _stub_loader(monkeypatch, check, found)
    capsys.readouterr()

    code = check.main(
        ["measure", str(dataset), "--detector", "model", "--model-path", str(weights)]
    )

    out = capsys.readouterr().out
    assert code == 0
    assert _numbers(out, "debris") == ["1", "2", "1", "0.500", "1.000"]
    assert _numbers(out, "single") == ["1", "0", "0", "-", "0.000"]


def test_a_model_that_will_not_load_is_a_finding_and_blobs_still_runs(
    check: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The default measures both detectors: one that cannot be built does not end the other's result."""
    dataset = _reviewed_dataset(tmp_path, check)

    def refused(*args: object, **kwargs: object) -> object:
        raise PluginRunError("cells", "no model at x.pth; put it there")

    monkeypatch.setattr(check, "load_detector", refused)
    capsys.readouterr()

    code = check.main(["measure", str(dataset)])

    captured = capsys.readouterr()
    assert code == 1
    assert _numbers(captured.out, "single") == ["1", "3", "1", "0.333", "1.000"]
    assert "no model at x.pth" in captured.out


def test_measure_refuses_an_image_that_is_not_the_size_its_annotation_file_says(
    check: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Height and width exchanged in the file: boxes would be read against the wrong axes (FM-73)."""
    dataset = _reviewed_dataset(tmp_path, check)
    coco = _val(dataset)
    coco["images"][0]["height"], coco["images"][0]["width"] = 320, 200
    _save_val(dataset, coco)
    capsys.readouterr()

    code = check.main(["measure", str(dataset), "--detector", "blobs"])

    assert code == 1
    assert "A1_0000.tif" in capsys.readouterr().err


def test_measure_refuses_a_model_option_without_the_model(
    check: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = _reviewed_dataset(tmp_path, check)
    capsys.readouterr()

    code = check.main(
        ["measure", str(dataset), "--detector", "blobs", "--model-um-per-px", "0.25"]
    )

    assert code == 1
    assert "--detector model" in capsys.readouterr().err


def test_measure_refuses_an_input_size_without_the_manifest(
    check: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset = _reviewed_dataset(tmp_path, check)
    (dataset / "manifest.json").unlink()
    weights = _weights(tmp_path)
    _stub_loader(monkeypatch, check, [])
    capsys.readouterr()

    code = check.main(
        [
            "measure",
            str(dataset),
            "--detector",
            "model",
            "--model-path",
            str(weights),
            "--model-um-per-px",
            "0.25",
        ]
    )

    assert code == 1
    assert "manifest.json" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argument",
    [
        ["--model-um-per-px", "0"],
        ["--model-um-per-px", "nan"],
        ["--model-um-per-px", "11"],
        ["--percentiles", "99:1"],
        ["--percentiles", "0.5"],
        ["--score-threshold", "0"],
        ["--max-overlap", "1.5"],
        ["--match-iou", "0"],
        ["--detector", "yolo"],
    ],
)
def test_measure_refuses_a_value_that_is_not_one(
    check: ModuleType, tmp_path: Path, argument: list[str]
) -> None:
    with pytest.raises(SystemExit) as caught:
        check.main(["measure", str(tmp_path), *argument])

    assert caught.value.code == 2


def test_the_json_output_holds_the_printed_numbers(
    check: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = _reviewed_dataset(tmp_path, check)
    target = tmp_path / "numbers.json"
    capsys.readouterr()

    code = check.main(
        ["measure", str(dataset), "--detector", "blobs", "--json", str(target)]
    )

    out = capsys.readouterr().out
    assert code == 0
    document = json.loads(target.read_text("utf-8"))
    (configuration,) = document["configurations"]
    assert configuration["detector"] == "blobs"
    assert configuration["error"] is None
    assert configuration["frames"] == 1
    for name in ["single", "doublet", "debris"]:
        score = configuration["classes"][name]
        printed = _numbers(out, name)
        assert [score["reference"], score["detected"], score["matched"]] == [
            int(v) for v in printed[:3]
        ]
        for value, cell in zip(
            (score["precision"], score["recall"]), printed[3:], strict=True
        ):
            assert ("-" if value is None else f"{value:.3f}") == cell
    assert configuration["classes"]["debris"]["precision"] is None
    assert configuration["any_box"]["matched"] == 3
    assert sorted(configuration["confusion"]) == sorted(
        [
            ["single", "single", 1],
            ["doublet", "single", 1],
            ["doublet", "doublet", 1],
            ["none", "single", 1],
            ["debris", "none", 1],
        ]
    )
    assert document["reviewed_images"] == 1
    assert document["match_iou"] == 0.5


def test_the_json_of_a_sweep_names_each_configuration(
    check: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _reviewed_dataset(tmp_path, check)
    weights = _weights(tmp_path)
    _stub_loader(monkeypatch, check, [])
    target = tmp_path / "numbers.json"

    code = check.main(
        [
            "measure",
            str(dataset),
            "--model-path",
            str(weights),
            "--model-um-per-px",
            "native,0.25",
            "--json",
            str(target),
        ]
    )

    assert code == 0
    configurations = json.loads(target.read_text("utf-8"))["configurations"]
    assert [
        (c["detector"], c["model_um_per_px"], c["input_size_px"])
        for c in configurations
    ] == [("blobs", None, None), ("model", None, None), ("model", 0.25, [400, 640])]
    assert configurations[1]["model"] == "w.pth"
    assert len(configurations[1]["model_sha256_12"]) == 12
    assert configurations[0]["model"] is None


def test_the_output_is_ascii(
    check: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A folder with a micro sign in its name is printed with an escape, never with the sign (Windows consoles)."""
    odd = tmp_path / "\N{MICRO SIGN}m check"
    odd.mkdir()
    run_json = _seed_run(odd, [("A1", _frame(3))])
    out = odd / "ref"

    assert _export(check, run_json, out) == 0
    exported = capsys.readouterr()
    coco = _val(out)
    coco["images"][0]["reviewed"] = True
    _save_val(out, coco)
    assert check.main(["measure", str(out), "--detector", "blobs"]) == 0
    measured = capsys.readouterr()
    assert check.main(["export", str(run_json), str(out)]) == 1  # not empty
    refused = capsys.readouterr()

    for text in (
        exported.out,
        exported.err,
        measured.out,
        measured.err,
        refused.out,
        refused.err,
    ):
        assert text.isascii()
    assert "\\xb5m" in refused.err  # the escape, so the folder is still named


def test_the_files_are_written_as_ascii_json(check: ModuleType, tmp_path: Path) -> None:
    """``nikon-control`` reads them with the platform's default encoding: ASCII is the same in all of them."""
    odd = tmp_path / "\N{MICRO SIGN}m"
    odd.mkdir()
    run_json = _seed_run(odd, [("A1", _frame(3))])
    out = odd / "ref"

    assert _export(check, run_json, out) == 0

    for name in ("annotations/val.json", "manifest.json"):
        assert (out / name).read_bytes().isascii()


# ------------------------------------------- what the adversarial review found (#342)
#
# One frame in every test above hid what a loop over frames does: the sums, the
# numbering, the times. These use three frames with spots, none and spots, so that
# a frame's boxes cannot be taken for another's.


def _three_frames(tmp_path: Path) -> Path:
    """A1 with a frame of spots, an empty one and a frame of spots."""
    return _seed_run(
        tmp_path,
        [("A1", _frame(3)), ("A1", _frame(4, spots=False)), ("A1", _frame(5))],
    )


def _boxes_of(frame: npt.NDArray[Any]) -> list[list[float]]:
    """What the blob detector finds on ``frame``, as COCO boxes: an oracle that reads nothing of the script."""
    from smc.plugins.cells.detector import BlobDetector

    return sorted(
        [d.col0_px, d.row0_px, d.col1_px - d.col0_px, d.row1_px - d.row0_px]
        for d in BlobDetector().detect(frame)
    )


def test_export_numbers_the_boxes_and_attaches_each_to_its_own_frame(
    check: ModuleType, tmp_path: Path
) -> None:
    """``nikon-control`` keys a box by its id: a repeated id is one box, and a box on the wrong image is on the wrong frame."""
    run_json = _three_frames(tmp_path)
    out = tmp_path / "ref"

    assert _export(check, run_json, out) == 0

    coco = _val(out)
    ids = [a["id"] for a in coco["annotations"]]
    assert sorted(ids) == list(range(1, 9))
    by_image: dict[int, list[list[float]]] = {}
    for a in coco["annotations"]:
        by_image.setdefault(a["image_id"], []).append(a["bbox"])
    assert sorted(by_image) == [1, 3]  # the empty frame, image 2, has no box
    scan_frames = [_frame(3), _frame(4, spots=False), _frame(5)]
    for image_id in (1, 3):
        assert sorted(by_image[image_id]) == _boxes_of(scan_frames[image_id - 1])
        assert len(by_image[image_id]) == 4


def test_measure_sums_the_confusions_of_every_reviewed_frame_and_times_them_apart(
    check: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Three reviewed frames: the sums are of all of them, each reads its own boxes, and cold and warm differ."""
    from types import SimpleNamespace

    out = tmp_path / "ref"
    assert _export(check, _three_frames(tmp_path), out) == 0
    coco = _val(out)
    for image in coco["images"]:
        image["reviewed"] = True
    _save_val(out, coco)
    # The detector's clock: 1 s, 3 s and 5 s for the three frames.
    ticks = iter([0.0, 1.0, 10.0, 13.0, 20.0, 25.0])
    monkeypatch.setattr(
        check, "time", SimpleNamespace(perf_counter=lambda: next(ticks))
    )
    target = tmp_path / "numbers.json"
    capsys.readouterr()

    code = check.main(
        ["measure", str(out), "--detector", "blobs", "--json", str(target)]
    )

    printed = capsys.readouterr().out
    assert code == 0
    # Every reference box is blobs' own: 6 singles and 2 doublets, all found. The
    # empty frame (image 2) has none: it must not be given the first frame's boxes.
    assert _numbers(printed, "single") == ["6", "6", "6", "1.000", "1.000"]
    assert _numbers(printed, "doublet") == ["2", "2", "2", "1.000", "1.000"]
    assert _numbers(printed, "any box") == ["8", "8", "8", "1.000", "1.000"]
    assert "reviewed frames: 3 of 3" in printed
    assert "first (cold) 1.000 s, others (warm) 4.000 s" in printed
    configuration = json.loads(target.read_text("utf-8"))["configurations"][0]
    assert configuration["frames"] == 3
    assert configuration["first_frame_s"] == 1.0
    assert configuration["mean_other_frames_s"] == 4.0


def test_measure_with_one_reviewed_frame_has_no_warm_time(
    check: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = _reviewed_dataset(tmp_path, check)
    capsys.readouterr()

    assert check.main(["measure", str(dataset), "--detector", "blobs"]) == 0

    assert "others (warm) -" in capsys.readouterr().out


def _into_two_splits(dataset: Path) -> None:
    """Image 2 and its boxes go to ``train.json`` as its image 1, as ``nikon-control`` numbers a split: ids collide."""
    coco = _val(dataset)
    second = coco["images"][1]
    boxes = [a for a in coco["annotations"] if a["image_id"] == second["id"]]
    train = {
        "categories": coco["categories"],
        "images": [{**second, "id": 1, "reviewed": True}],
        "annotations": [
            {**a, "id": n, "image_id": 1} for n, a in enumerate(boxes, start=1)
        ],
    }
    coco["images"] = [{**coco["images"][0], "reviewed": True}]
    coco["annotations"] = [a for a in coco["annotations"] if a["image_id"] == 1]
    _save_val(dataset, coco)
    (dataset / "annotations" / "train.json").write_text(
        json.dumps(train, indent=1), encoding="utf-8"
    )


def test_measure_reads_the_reviewed_frames_of_both_splits(
    check: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_json = _seed_run(tmp_path, [("A1", _frame(3)), ("A1", _frame(5))])
    out = tmp_path / "ref"
    assert _export(check, run_json, out) == 0
    _into_two_splits(out)
    capsys.readouterr()

    code = check.main(["measure", str(out), "--detector", "blobs"])

    printed = capsys.readouterr().out
    assert code == 0
    # 4 boxes from val.json's frame and 4 from train.json's, each on its own image.
    assert _numbers(printed, "any box") == ["8", "8", "8", "1.000", "1.000"]
    assert "reviewed frames: 2 of 2" in printed


def test_measure_reads_a_dataset_that_has_only_a_train_split(
    check: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_json = _seed_run(tmp_path, [("A1", _frame(3)), ("A1", _frame(5))])
    out = tmp_path / "ref"
    assert _export(check, run_json, out) == 0
    _into_two_splits(out)
    (out / "annotations" / "val.json").unlink()
    capsys.readouterr()

    code = check.main(["measure", str(out), "--detector", "blobs"])

    assert code == 0
    assert _numbers(capsys.readouterr().out, "any box") == [
        "4",
        "4",
        "4",
        "1.000",
        "1.000",
    ]


def test_the_device_reaches_the_model_in_both_commands(
    check: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A timing taken on the wrong device is the wrong timing: ``--device cpu`` must reach ``load_detector``, and ``auto`` is the default."""
    dataset = _reviewed_dataset(tmp_path, check)
    weights = _weights(tmp_path)
    calls = _stub_loader(monkeypatch, check, [])
    run_json = (
        tmp_path / "run" / "run.json"
    )  # the scan `_reviewed_dataset` exported from

    export = ["--model-path", str(weights)]
    assert _export(check, run_json, tmp_path / "e1", *export, prefill="model") == 0
    assert (
        _export(
            check,
            run_json,
            tmp_path / "e2",
            *export,
            "--device",
            "cpu",
            prefill="model",
        )
        == 0
    )
    measure = [
        "measure",
        str(dataset),
        "--detector",
        "model",
        "--model-path",
        str(weights),
    ]
    assert check.main(measure) == 0
    assert check.main([*measure, "--device", "cpu"]) == 0

    assert [c["device"] for c in calls] == ["auto", "cpu", "auto", "cpu"]


# -- refusals that leave nothing behind ----------------------------------------


@pytest.mark.parametrize("prefill", ["none", "blobs"])
def test_export_refuses_a_missing_frame_file_before_it_writes_anything(
    check: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    prefill: str,
) -> None:
    run_json = _seed_run(tmp_path, [("A1", _frame(0)), ("A1", _frame(1))])
    (run_json.parent / "scan" / "t" / "A1_0001.tif").unlink()
    out = tmp_path / "ref"

    code = _export(check, run_json, out, prefill=prefill)

    assert code == 1
    assert not out.exists()
    assert "scan/t/A1_0001.tif" in capsys.readouterr().err


def test_export_finds_a_missing_frame_before_the_model_is_loaded(
    check: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The frames are looked at first, so that a scan with a hole does not wait for 166 MB of weights (FM-30)."""
    run_json = _seed_run(tmp_path, [("A1", _frame(0)), ("A1", _frame(1))])
    (run_json.parent / "scan" / "t" / "A1_0001.tif").unlink()
    weights = _weights(tmp_path)
    _refusing_loader(monkeypatch, check)

    code = _export(
        check, run_json, tmp_path / "ref", "--model-path", str(weights), prefill="model"
    )

    assert code == 1
    assert "scan/t/A1_0001.tif" in capsys.readouterr().err


def test_export_refuses_a_scan_with_no_frame(
    check: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_json = _seed_run(tmp_path, [])

    assert _export(check, run_json, tmp_path / "ref") == 1

    assert "no frame" in capsys.readouterr().err
    assert not (tmp_path / "ref").exists()


def test_export_refuses_a_frame_path_that_leaves_the_run_folder(
    check: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run document copied from elsewhere must not make the script read any file of the PC."""
    # `scan/t/../../x.tif` is `run/x.tif`: it exists, so only the path check can refuse it.
    run_json = _seed_run(tmp_path, [("A1", _frame(0))], names=["../../x"])
    assert (run_json.parent / "x.tif").is_file()

    assert _export(check, run_json, tmp_path / "ref") == 1

    assert "not usable" in capsys.readouterr().err
    assert not (tmp_path / "ref").exists()


@pytest.mark.parametrize("prefill", ["blobs", "none"])
@pytest.mark.parametrize(
    "option",
    [["--model-path", "w.pth"], ["--device", "cpu"], ["--model-um-per-px", "0.25"]],
)
def test_export_refuses_a_model_option_next_to_another_prefill(
    check: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    prefill: str,
    option: list[str],
) -> None:
    run_json = _seed_run(tmp_path, [("A1", _frame(0))])

    code = _export(check, run_json, tmp_path / "ref", *option, prefill=prefill)

    assert code == 1
    assert f"{option[0]} is for --prefill model" in capsys.readouterr().err
    assert not (tmp_path / "ref").exists()


def test_export_spaces_a_fractional_count_by_rounding(
    check: ModuleType, tmp_path: Path
) -> None:
    """4 of 6 is ``linspace(0, 5, 4)`` = 0, 1.67, 3.33, 5: rounded, 0, 2, 3 and 5 (a floor gives 1, a ceiling 4)."""
    run_json = _seed_run(tmp_path, [("A1", _frame(n)) for n in range(6)])
    out = tmp_path / "ref"

    assert _export(check, run_json, out, "--count", "4") == 0

    assert sorted(p.name for p in (out / "images").iterdir()) == [
        "A1_0000.tif",
        "A1_0002.tif",
        "A1_0003.tif",
        "A1_0005.tif",
    ]


def test_measure_refuses_an_image_that_is_not_a_2d_frame(
    check: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = _reviewed_dataset(tmp_path, check)
    tifffile.imwrite(
        dataset / "images" / "A1_0000.tif", np.zeros((200, 320, 3), np.uint8)
    )
    capsys.readouterr()

    code = check.main(["measure", str(dataset), "--detector", "blobs"])

    assert code == 1
    assert "not a 2-D frame" in capsys.readouterr().err


def test_measure_refuses_an_image_that_is_not_the_size_of_the_scans_frames(
    check: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The image and its annotation file agree, the manifest does not: the model input size would be worked out on another shape."""
    dataset = _reviewed_dataset(tmp_path, check)
    tifffile.imwrite(
        dataset / "images" / "A1_0000.tif", np.zeros((100, 160), np.uint16)
    )
    coco = _val(dataset)
    coco["images"][0]["height"], coco["images"][0]["width"] = 100, 160
    _save_val(dataset, coco)
    capsys.readouterr()

    code = check.main(["measure", str(dataset), "--detector", "blobs"])

    assert code == 1
    assert "manifest.json says" in capsys.readouterr().err


# -- what the Windows review found ----------------------------------------------


def _locked(real: Any, *, fail_on: Any, times: int) -> Any:
    """``real`` that raises a sharing violation ``times`` times on the calls ``fail_on`` accepts, then works."""
    state = {"left": times}

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if fail_on(*args, **kwargs) and state["left"] > 0:
            state["left"] -= 1
            raise PermissionError(
                13,
                "The process cannot access the file because it is being used by another process",
            )
        return real(*args, **kwargs)

    return wrapper


def test_a_frame_that_is_held_for_a_moment_is_still_copied(
    check: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Antivirus or a sync client holds a frame the scan wrote a moment ago (FM-24)."""
    import shutil

    run_json = _seed_run(
        tmp_path, [("A1", _frame(0)), ("A1", _frame(1)), ("A1", _frame(2))]
    )
    monkeypatch.setattr(
        shutil,
        "copyfile",
        _locked(
            shutil.copyfile,
            fail_on=lambda src, dst: str(src).endswith("A1_0001.tif"),
            times=2,
        ),
    )
    out = tmp_path / "ref"

    assert _export(check, run_json, out) == 0

    assert len(list((out / "images").iterdir())) == 3


def test_a_frame_that_stays_held_names_the_frame_and_leaves_no_half_made_folder(
    check: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The next export of the same command must not be refused for a folder this one half made."""
    import shutil

    run_json = _seed_run(
        tmp_path, [("A1", _frame(0)), ("A1", _frame(1)), ("A1", _frame(2))]
    )
    real = shutil.copyfile
    monkeypatch.setattr(
        shutil,
        "copyfile",
        _locked(
            real, fail_on=lambda src, dst: str(src).endswith("A1_0001.tif"), times=99
        ),
    )
    out = tmp_path / "ref"

    assert _export(check, run_json, out) == 1

    error = capsys.readouterr().err
    assert "A1_0001.tif" in error  # the file that is held, not the folder
    assert not out.exists()
    monkeypatch.setattr(shutil, "copyfile", real)
    assert _export(check, run_json, out) == 0


def test_an_empty_folder_given_for_the_export_is_kept_when_it_fails(
    check: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    run_json = _seed_run(tmp_path, [("A1", _frame(0)), ("A1", _frame(1))])
    monkeypatch.setattr(
        shutil,
        "copyfile",
        _locked(
            shutil.copyfile,
            fail_on=lambda src, dst: str(src).endswith("A1_0001.tif"),
            times=99,
        ),
    )
    out = tmp_path / "ref"
    out.mkdir()

    assert _export(check, run_json, out) == 1

    assert out.is_dir()
    assert list(out.iterdir()) == []


def test_the_refusal_of_a_folder_does_not_claim_it_holds_annotations(
    check: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_json = _seed_run(tmp_path, [("A1", _frame(0))])
    out = tmp_path / "ref"
    out.mkdir()
    (out / "notes.txt").write_text("mine", encoding="utf-8")

    assert _export(check, run_json, out) == 1

    error = capsys.readouterr().err
    assert "not empty" in error
    assert "never overwritten" in error
    assert "the annotations in it" not in error


def test_a_json_file_saved_with_a_byte_order_mark_is_read(
    check: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Notepad before Windows 10 1903 saves "UTF-8" with a mark; the owner may edit a file by hand (FM-80)."""
    dataset = _reviewed_dataset(tmp_path, check)
    for name in ("annotations/val.json", "manifest.json"):
        path = dataset / name
        path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    capsys.readouterr()

    code = check.main(["measure", str(dataset), "--detector", "blobs"])

    assert code == 0
    assert _numbers(capsys.readouterr().out, "any box") == [
        "4",
        "4",
        "3",
        "0.750",
        "0.750",
    ]


def test_a_file_that_is_held_for_a_moment_is_still_read_by_measure(
    check: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The review page saved ``val.json`` a moment ago and antivirus holds it, as for the image (FM-24)."""
    dataset = _reviewed_dataset(tmp_path, check)
    monkeypatch.setattr(
        Path,
        "read_bytes",
        _locked(Path.read_bytes, fail_on=lambda path: path.name == "val.json", times=2),
    )
    monkeypatch.setattr(
        tifffile,
        "imread",
        _locked(
            tifffile.imread,
            fail_on=lambda path, *a, **k: Path(path).name == "A1_0000.tif",
            times=2,
        ),
    )
    capsys.readouterr()

    code = check.main(["measure", str(dataset), "--detector", "blobs"])

    assert code == 0
    assert _numbers(capsys.readouterr().out, "any box") == [
        "4",
        "4",
        "3",
        "0.750",
        "0.750",
    ]


def test_the_next_commands_are_quoted_when_a_folder_has_a_space(
    check: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The owner copies them into a console: a folder with a space is two words without quotes."""
    run_json = _seed_run(tmp_path, [("A1", _frame(0))])
    out = tmp_path / "my check" / "ref"

    assert _export(check, run_json, out) == 0

    lines = capsys.readouterr().out.splitlines()
    dashboard = next(line for line in lines if "nikon-control-dashboard" in line)
    measure = next(line for line in lines if line.startswith("then:"))
    assert f'--data-dir "{out.parent}"' in dashboard
    assert f'measure "{out}"' in measure
