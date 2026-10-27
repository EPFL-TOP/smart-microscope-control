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
import threading
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import tifffile
from pydantic import BaseModel

from smc.hardware import Microscope
from smc.plugins import PluginRunError, Progress, RunContext, load, run_plugin
from smc.plugins.cells import plugin as plugin_mod
from smc.plugins.cells.detector import Detection
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


class _Stub:
    """A detector that returns what was scripted for the frame it is given."""

    classes = ("single", "doublet", "debris")
    device = "cpu"

    def __init__(
        self,
        scripted: dict[int, list[Detection]],
        on_frame: Callable[[int], None] | None = None,
    ) -> None:
        self._scripted = scripted
        self._on_frame = on_frame
        self.seen: list[int] = []

    def detect(self, frame: npt.NDArray[Any]) -> list[Detection]:
        position = int(frame[0, 0])
        self.seen.append(position)
        if self._on_frame is not None:
            self._on_frame(position)
        return self._scripted.get(position, [])


def _install(monkeypatch: pytest.MonkeyPatch, stub: _Stub) -> list[dict[str, Any]]:
    """Make ``load_detector`` give ``stub``; the calls it received are returned."""
    calls: list[dict[str, Any]] = []

    def fake(model_path: Path | None, *, device: str, score_threshold: float) -> _Stub:
        calls.append(
            {"path": model_path, "device": device, "threshold": score_threshold}
        )
        return stub

    monkeypatch.setattr(plugin_mod, "load_detector", fake)
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
    _install(monkeypatch, _Stub(SCRIPT, on_frame=_stopping_at(2, stop)))
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
    stub = _Stub({0: [SHARED, OWN], 1: [SECOND_VIEW]})  # B2's frame shows nothing
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
    calls = _install(monkeypatch, _Stub({}))

    record = _run(demo_microscope, tmp_path, score_threshold=0.8)

    assert calls == [{"path": weights, "device": "cpu", "threshold": 0.8}]
    assert _result(record).score_threshold == 0.8


def test_max_overlap_reaches_the_suppression_of_a_frame(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two boxes sharing 0.5 of the smaller one: one object at 0.4, two at the default 0.7."""
    _seed_scan(tmp_path, frames=[("A1", 0.0)])
    _weights(tmp_path)
    single = Detection(25.0, 15.0, 35.0, 25.0, "single", 0.6)
    doublet = Detection(25.0, 20.0, 35.0, 30.0, "doublet", 0.9)
    _install(monkeypatch, _Stub({0: [single, doublet]}))

    strict = _result(_run(demo_microscope, tmp_path, max_overlap=0.4))
    default = _result(_run(demo_microscope, tmp_path))

    assert strict.counts["A1"] == {"single": 0, "doublet": 1, "debris": 0}
    assert default.counts["A1"] == {"single": 1, "doublet": 1, "debris": 0}


def test_the_result_names_the_model_file_never_its_path(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(tmp_path)
    _weights(tmp_path)
    _install(monkeypatch, _Stub(SCRIPT))

    record = _run(demo_microscope, tmp_path)

    assert _result(record).model == "w.pth"
    text = json.dumps(record.result)
    # json.dumps doubles every backslash: a Windows path leaks as its escaped
    # spelling, which `str(tmp_path)` is not (FM-43). The folder name is the
    # test id, which nothing of the result holds.
    escaped = json.dumps(str(tmp_path))[1:-1]
    assert str(tmp_path) not in text
    assert escaped not in text
    assert tmp_path.as_posix() not in text
    assert tmp_path.name not in text


@pytest.mark.parametrize(
    "content", [WEIGHTS, b"a long checkpoint " * 150_000], ids=["short", "three-chunks"]
)
def test_model_sha256_is_the_hash_of_the_weights(
    demo_microscope: Microscope,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    content: bytes,
) -> None:
    """The file is read in 1 MiB pieces: all of them count, not the first."""
    _seed_scan(tmp_path)
    _weights(tmp_path, content)
    _install(monkeypatch, _Stub({}))

    record = _run(demo_microscope, tmp_path)

    assert _result(record).model_sha256 == hashlib.sha256(content).hexdigest()


def test_the_merge_radius_is_named_as_chosen_and_not_measured(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(tmp_path)
    _weights(tmp_path)
    _install(monkeypatch, _Stub({}))

    chosen = _result(_run(demo_microscope, tmp_path, merge_radius_um=3.5))
    off = _result(_run(demo_microscope, tmp_path, merge_radius_um=0.0))

    assert chosen.assumed == (
        "cells within 3.5 µm seen by two fields are counted once: a chosen radius, "
        "not measured",
    )
    assert off.assumed == ()
    assert (chosen.merge_radius_um, off.merge_radius_um) == (3.5, 0.0)


# --------------------------------------------------------------- bad frames


def _failure(
    microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> PluginRunError:
    _weights(tmp_path)
    _install(monkeypatch, _Stub(SCRIPT))
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
    _install(monkeypatch, _Stub({0: [alien]}))

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

    counting = _Stub(SCRIPT)
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
    counting = _Stub(SCRIPT)
    _install(monkeypatch, counting)

    with pytest.raises(PluginRunError) as caught:
        _run(demo_microscope, tmp_path, resume=True)

    assert "weights changed" in caught.value.reason
    assert "without --resume" in caught.value.reason
    assert counting.seen == []


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
    counting = _Stub(SCRIPT)
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
    counting = _Stub(SCRIPT)
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
    _install(monkeypatch, _Stub(SCRIPT))
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

    counting = _Stub(SCRIPT)
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
    _install(monkeypatch, _Stub(SCRIPT))

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
    _install(monkeypatch, _Stub(SCRIPT, on_frame=_stopping_at(2, stop)))
    _run(demo_microscope, tmp_path, stop=stop, on_progress=first.append)
    second: list[Progress] = []
    _install(monkeypatch, _Stub(SCRIPT))

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

    assert params.model_path is None
    assert params.device == "auto"
    assert params.score_threshold == 0.5
    assert params.max_overlap == 0.7
    assert params.merge_radius_um == 5.0


def test_seconds_per_frame_is_the_mean_of_the_detect_times(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(tmp_path, frames=[("A1", 0.0), ("A1", 30.0)])
    _weights(tmp_path)
    _install(monkeypatch, _Stub({}))
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
        _Stub(
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
    _install(monkeypatch, _Stub(SCRIPT_OF_THREE, on_frame=_stopping_at(4, stop)))
    stopped = _run(demo_microscope, tmp_path, stop=stop)
    assert stopped.checkpoint is not None
    assert stopped.checkpoint["wells_done"] == ["A1", "A2"]
    token = stopped.checkpoint["token"]
    (tmp_path / "cells" / f"{token}.partial.json").unlink()

    _install(monkeypatch, _Stub(SCRIPT_OF_THREE))
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

    counting = _Stub(SCRIPT_OF_THREE)
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
    counting = _Stub(SCRIPT)
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
    counting = _Stub(SCRIPT)
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
    _install(monkeypatch, _Stub(SCRIPT))
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
    _install(monkeypatch, _Stub(SCRIPT))
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
    counting = _Stub(SCRIPT)
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
    _install(monkeypatch, _Stub(SCRIPT))
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


def test_weights_that_cannot_be_read_are_a_plugin_run_error(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The detector read the file a moment ago; a file replaced or locked since says so."""
    _seed_scan(tmp_path)
    _install(monkeypatch, _Stub({}))

    # A folder where the weights should be: it opens as no file, on every OS.
    with pytest.raises(PluginRunError) as caught:
        _run(demo_microscope, tmp_path, model_path=tmp_path)

    assert "cannot be read to hash it" in caught.value.reason
