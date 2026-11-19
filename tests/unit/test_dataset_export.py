"""``smc dataset export``: a scan's run folder written as a dataset (algorithms design §2.1, #565).

The run folders are ``tests/select_fields_seed.py::seed_run``'s, written with no
Micro-Manager; the oracles are ``hashlib`` and the bytes on disk, never the
export's own helpers.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import textwrap
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import smc.data.dataset as dataset_module
from select_fields_seed import PIXEL_UM, SHAPE, Seed, cell_at, matrix_of, seed_run
from smc.data import DatasetError, load_scan
from smc.data.dataset import (
    DatasetFile,
    DatasetManifest,
    DatasetWriteError,
    _check_names,
    _copy_and_hash,
    _sha256_of,
    check_dataset,
    export_dataset,
    readme_text,
)
from smc.plugins.slots import Cell
from smc.run import RunDocument, StageRecord

# --- the copy and the manifest ---------------------------------------------------


def _bytes_of(size: int) -> bytes:
    """``size`` bytes that are not all alike, so that a chunk written twice or lost shows."""
    return bytes(range(251)) * (size // 251) + bytes(range(size % 251))


def test_the_copy_hashes_the_source_and_never_overwrites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.bin"
    payload = _bytes_of(3 * 1024 * 1024 + 5)
    source.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()

    destination = tmp_path / "below" / "copy.bin"
    assert _copy_and_hash(source, destination) == (len(payload), expected)
    assert destination.read_bytes() == payload

    # Cut into chunks of seven bytes: the same size, the same hash, the same bytes.
    monkeypatch.setattr(dataset_module, "_CHUNK", 7)
    small = tmp_path / "small.bin"
    small.write_bytes(payload[:100])
    again = tmp_path / "again.bin"
    assert _copy_and_hash(small, again) == (
        100,
        hashlib.sha256(payload[:100]).hexdigest(),
    )
    assert again.read_bytes() == payload[:100]

    # A destination that exists is never opened for writing, and not touched.
    before = destination.read_bytes()
    with pytest.raises(FileExistsError):
        _copy_and_hash(small, destination)
    assert destination.read_bytes() == before


def test_a_source_that_cannot_be_read_leaves_no_destination(tmp_path: Path) -> None:
    destination = tmp_path / "below" / "copy.bin"
    with pytest.raises(FileNotFoundError):
        _copy_and_hash(tmp_path / "missing.bin", destination)
    assert not destination.exists()


def test_the_hash_of_a_file_reads_it_in_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "f.bin"
    payload = _bytes_of(1000)
    path.write_bytes(payload)
    monkeypatch.setattr(dataset_module, "_CHUNK", 7)
    assert _sha256_of(path) == hashlib.sha256(payload).hexdigest()


def test_the_manifest_dumps_to_json_and_loads_back_equal() -> None:
    manifest = DatasetManifest(
        smc_version="1.2.3",
        exported_utc=datetime(2026, 11, 18, 12, 12, 47, 918585, tzinfo=timezone.utc),
        source="run.json",
        profile="demo",
        objective=None,
        pixel_size_um=5.0,
        image_shape_px=(60, 80),
        wells=("A1", "B2"),
        frames=8,
        cells=False,
        cells_absent="no cells record",
        files={"scan/t/A1_0000.tif": DatasetFile(bytes=12, sha256="ab" * 32)},
        scrubbed=("/stages/scan/params/out",),
    )
    text = manifest.model_dump_json()
    assert '"exported_utc":"2026-11-18T12:12:47.918585Z"' in text
    assert DatasetManifest.model_validate_json(text) == manifest
    assert manifest.schema_version == 1
    # Strict and frozen: an unknown field is refused, and the model is not edited.
    with pytest.raises(ValueError, match="extra"):
        DatasetManifest.model_validate({**manifest.model_dump(), "extra": 1})
    with pytest.raises(ValueError, match="frozen"):
        manifest.frames = 9  # type: ignore[misc]


def test_the_windows_limits_are_module_attributes_a_test_can_set() -> None:
    assert dataset_module.MAX_PATH == 260
    assert dataset_module.ON_WINDOWS is (os.name == "nt")


# --- the README ------------------------------------------------------------------


def _manifest(**changes: Any) -> DatasetManifest:
    """A manifest built by hand, for the README: what a scan of two wells with cells says."""
    fields: dict[str, Any] = {
        "smc_version": "1.2.3",
        "exported_utc": datetime(2026, 11, 18, 12, 12, 47, tzinfo=timezone.utc),
        "source": "run.json",
        "profile": "demo-sample",
        "objective": "Nikon 10X S Fluor",
        "pixel_size_um": 0.6,
        "image_shape_px": (60, 80),
        "wells": ("A1", "B2"),
        "frames": 8,
        "cells": True,
        "cells_absent": None,
        "files": {},
        "scrubbed": (),
    }
    fields.update(changes)
    return DatasetManifest(**fields)


def _readme(manifest: DatasetManifest, **changes: Any) -> str:
    arguments: dict[str, Any] = {
        "scanned_utc": datetime(2026, 11, 17, 14, 32, 5, tzinfo=timezone.utc),
        "cells_file": "cells/t1.json",
        "classes": ("single", "doublet"),
    }
    arguments.update(changes)
    return readme_text(manifest, **arguments)


def test_the_readme_says_what_the_design_lists() -> None:
    text = _readme(_manifest())
    lines = [line.split() for line in text.splitlines()]

    assert ["scanned", "2026-11-17", "14:32", "UTC"] in lines
    assert ["exported", "2026-11-18", "12:12", "UTC", "by", "smc", "1.2.3"] in lines
    assert ["profile", "demo-sample"] in lines
    assert ["objective", "Nikon", "10X", "S", "Fluor"] in lines
    assert ["pixel", "size", "0.6000", "um", "per", "pixel"] in lines
    assert "image shape  60 x 80 pixels (height x width)" in text
    assert "A1, B2 (the order of the scan plan)" in text
    assert ["frames", "8"] in lines
    assert "detected, listed in cells/t1.json (classes: single, doublet)" in text
    # The layout, the conventions, the two ways to use it, and what was left out.
    for name in ("run.json", "README.md", "manifest.json", "cells/<token>.json"):
        assert name in text
    assert "scan/<token>/<well>/<name>.tif" in text
    assert "well_dx_um" in text
    assert "(dx_um, dy_um) = M @ (d_col, d_row)" in text
    assert "a pixel size of 0 is unknown" in text.lower()
    assert "from smc.data import load_scan" in text
    assert "smc dataset check" in text
    assert "work on a copy" in text.lower()
    assert "only the scan's frames and the cells file were copied" in text.lower()
    # Neither a hash nor a path: the manifest has the first, and FM-50 the second.
    assert "sha256" not in text.lower()
    assert not re.search(r"[0-9a-f]{32}", text)
    assert not re.search(r"(/Users/|/home/|[A-Za-z]:\\\\)", text)
    assert text.isascii()
    assert text.endswith("\n")


def test_the_readme_says_unknown_for_what_the_scan_does_not_say() -> None:
    text = _readme(_manifest(objective=None), scanned_utc=None)
    lines = [line.split() for line in text.splitlines()]
    assert ["objective", "unknown"] in lines
    assert ["scanned", "unknown"] in lines


def test_the_readme_of_a_scan_with_no_cells_says_why_and_lists_no_cells_file() -> None:
    manifest = _manifest(cells=False, cells_absent="no cells record")
    text = _readme(manifest, cells_file=None, classes=())
    assert "none: no cells record" in text
    assert "cells/<token>.json" not in text
    assert "classes" not in text


def test_an_objective_with_a_greek_letter_passes_through_and_reads_back_as_utf8(
    tmp_path: Path,
) -> None:
    text = _readme(_manifest(objective="Plan Apo \N{GREEK SMALL LETTER LAMDA} 20x"))
    assert not text.isascii()
    path = tmp_path / "README.md"
    path.write_bytes(text.encode("utf-8"))
    assert "Plan Apo \N{GREEK SMALL LETTER LAMDA} 20x" in path.read_text(
        encoding="utf-8"
    )
    # Only the objective's own word is outside ASCII: the fixed text is not.
    other = [line for line in text.splitlines() if "Plan Apo" not in line]
    assert all(line.isascii() for line in other)


# --- the export ------------------------------------------------------------------


def _three_cells() -> list[Cell]:
    return [
        cell_at("A1", 1.0, 1.0, "single"),
        cell_at("A1", 2.0, 1.0, "doublet"),
        cell_at("B2", 1.0, 1.0, "single"),
    ]


def _source(tmp_path: Path, **changes: Any) -> Seed:
    """A seeded run folder of two wells and three cells, in ``tmp_path/src``."""
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    arguments: dict[str, Any] = {"cells": _three_cells(), "wells": ("A1", "B2")}
    arguments.update(changes)
    return seed_run(src, **arguments)


def _edit_stages(path: Path, edit: Callable[[dict[str, Any]], None]) -> None:
    """Change the raw JSON of a run document: the stages, as a person editing the file would."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    edit(raw["stages"])
    path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frame_files(seed: Seed) -> list[str]:
    return [frame.file for frame in seed.scan.frames]


def _cells_file(seed: Seed) -> str:
    return seed.result.file


def _no_copy(*_: object) -> tuple[int, str]:
    raise AssertionError("a file was copied: the export had to refuse first")


def test_export_writes_the_folder_of_the_design_and_it_loads_as_a_scan(
    tmp_path: Path,
) -> None:
    seed = _source(tmp_path)
    out = tmp_path / "out"
    before = datetime.now(timezone.utc)

    manifest = export_dataset(seed.path, out)

    after = datetime.now(timezone.utc)
    names = [*_frame_files(seed), _cells_file(seed), "run.json", "README.md"]
    assert len(_frame_files(seed)) == 8
    for name in [*names, "manifest.json"]:
        assert (out / name).is_file(), name
    for name in [*_frame_files(seed), _cells_file(seed)]:
        assert (out / name).read_bytes() == (seed.path.parent / name).read_bytes()

    written = DatasetManifest.model_validate_json(
        (out / "manifest.json").read_text(encoding="utf-8")
    )
    assert written == manifest
    assert manifest.profile == "demo"
    assert manifest.objective == "Nikon 10X S Fluor"
    assert manifest.wells == ("A1", "B2")
    assert manifest.frames == 8
    assert manifest.cells is True
    assert manifest.cells_absent is None
    assert manifest.source == "run.json"
    assert manifest.scrubbed == ()
    assert before <= manifest.exported_utc <= after

    original, exported = load_scan(seed.path), load_scan(out / "run.json")
    assert exported.wells == original.wells
    assert exported.profile == original.profile
    assert exported.frames() == original.frames()
    assert exported.cells == original.cells
    assert exported.pixel_size_um == original.pixel_size_um
    assert exported.image_shape_px == original.image_shape_px
    assert exported.matrix_um_per_px == original.matrix_um_per_px


def test_the_manifest_lists_every_file_but_itself_in_the_written_order_with_its_hash(
    tmp_path: Path,
) -> None:
    seed = _source(tmp_path)
    out = tmp_path / "out"
    manifest = export_dataset(seed.path, out)

    assert list(manifest.files) == [
        *_frame_files(seed),
        _cells_file(seed),
        "run.json",
        "README.md",
    ]
    for name, entry in manifest.files.items():
        assert entry.bytes == (out / name).stat().st_size, name
        assert entry.sha256 == _sha(out / name), name
    on_disk = (out / "manifest.json").read_bytes()
    assert on_disk.isascii()
    assert on_disk.endswith(b"\n")
    # The order is the file's: the JSON object keeps it.
    assert list(json.loads(on_disk)["files"]) == list(manifest.files)


def test_the_manifest_is_written_last_and_the_files_in_the_designs_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = _source(tmp_path)
    out = tmp_path / "out"
    root = out.resolve()
    order: list[str] = []
    copy, write = dataset_module._copy_and_hash, dataset_module.write_atomically

    def spy_copy(source: Path, destination: Path) -> tuple[int, str]:
        order.append(destination.relative_to(root).as_posix())
        return copy(source, destination)

    def spy_write(path: Path, data: bytes, **kwargs: Any) -> None:
        order.append(path.relative_to(root).as_posix())
        write(path, data, **kwargs)

    monkeypatch.setattr(dataset_module, "_copy_and_hash", spy_copy)
    monkeypatch.setattr(dataset_module, "write_atomically", spy_write)
    export_dataset(seed.path, out)

    assert order == [
        *_frame_files(seed),
        _cells_file(seed),
        "run.json",
        "README.md",
        "manifest.json",
    ]


def test_an_interrupted_export_has_no_manifest_and_says_how_far_it_got(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = _source(tmp_path)
    out = tmp_path / "out"
    copy = dataset_module._copy_and_hash
    calls: list[Path] = []

    def fails_on_the_third(source: Path, destination: Path) -> tuple[int, str]:
        calls.append(destination)
        if len(calls) == 3:
            raise OSError("the share went away")
        return copy(source, destination)

    monkeypatch.setattr(dataset_module, "_copy_and_hash", fails_on_the_third)
    with pytest.raises(DatasetWriteError) as caught:
        export_dataset(seed.path, out)

    third = _frame_files(seed)[2]
    n = len(_frame_files(seed)) + 1 + 3
    message = str(caught.value)
    assert third in message
    assert f"stopped after 2 of {n} files" in message
    assert "the share went away" in message
    assert "no manifest.json" in message
    assert "remove it" in message
    assert isinstance(caught.value.__cause__, OSError)
    # What was written stays (nothing is deleted), and there is no manifest.
    assert (out / _frame_files(seed)[0]).is_file()
    assert (out / _frame_files(seed)[1]).is_file()
    assert not (out / "run.json").exists()
    assert not (out / "manifest.json").exists()


def test_a_write_error_of_a_generated_file_counts_the_files_written_so_far(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = _source(tmp_path)
    out = tmp_path / "out"
    write = dataset_module.write_atomically

    def fails_on_the_manifest(path: Path, data: bytes, **kwargs: Any) -> None:
        if path.name == "manifest.json":
            raise OSError(28, "No space left on device")
        write(path, data, **kwargs)

    monkeypatch.setattr(dataset_module, "write_atomically", fails_on_the_manifest)
    with pytest.raises(DatasetWriteError) as caught:
        export_dataset(seed.path, out)

    n = len(_frame_files(seed)) + 1 + 3
    assert f"stopped after {n - 1} of {n} files at manifest.json" in str(caught.value)
    assert "No space left on device" in str(caught.value)
    assert (out / "README.md").is_file()
    assert not (out / "manifest.json").exists()


def _longest_destination(out: Path, seed: Seed) -> int:
    """The longest path the export makes: a frame, the cells file, or a generated file's temporary one."""
    root = out.resolve()
    named = [*_frame_files(seed), _cells_file(seed)]
    named += [
        f".{name}.00000000.smc-tmp"
        for name in ("run.json", "README.md", "manifest.json")
    ]
    return max(len(str(root / name)) for name in named)


def _need_of(seed: Seed) -> int:
    """The bytes the export asks the disk for: the files it copies, and a megabyte."""
    folder = seed.path.parent
    sizes = [(folder / name).stat().st_size for name in _frame_files(seed)]
    sizes.append((folder / _cells_file(seed)).stat().st_size)
    return sum(sizes) + 1_000_000


def _free(monkeypatch: pytest.MonkeyPatch, free: int) -> None:
    monkeypatch.setattr(
        dataset_module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=free * 2, used=free, free=free),
    )


def _refusal_cases() -> list[str]:
    return [
        "out is a file",
        "out is a folder with a file in it",
        "a dry run",
        "no scan stage",
        "a stopped scan",
        "a path over the Windows limit",
        "a disk too small",
    ]


@pytest.mark.parametrize("case", _refusal_cases())
def test_export_refuses_before_writing_anything(
    case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    source: Path
    reason: str
    wrong: Path
    if case == "a dry run":
        seed = _source(tmp_path, dry_run=True)
        source, reason, wrong = seed.path, "dry run", seed.path
    else:
        seed = _source(tmp_path)
        source, reason, wrong = seed.path, "", out
    existed = False
    if case == "out is a file":
        out.write_text("not a folder", encoding="utf-8")
        existed, reason = True, "exists and is a file"
    elif case == "out is a folder with a file in it":
        out.mkdir()
        (out / "sentinel.txt").write_bytes(b"keep me")
        existed, reason = True, "not empty"
    elif case == "no scan stage":
        _edit_stages(source, lambda stages: stages.pop("scan"))
        reason, wrong = "no scan", source
    elif case == "a stopped scan":
        _edit_stages(source, lambda stages: stages["scan"].update(status="stopped"))
        reason, wrong = "ended stopped", source
    elif case == "a path over the Windows limit":
        monkeypatch.setattr(dataset_module, "ON_WINDOWS", True)
        monkeypatch.setattr(dataset_module, "MAX_PATH", _longest_destination(out, seed))
        reason = "Windows takes"
    elif case == "a disk too small":
        _free(monkeypatch, _need_of(seed) - 1)
        reason = "MB free and the dataset needs"
    monkeypatch.setattr(dataset_module, "_copy_and_hash", _no_copy)

    with pytest.raises(DatasetError) as caught:
        export_dataset(source, out)

    assert reason in str(caught.value)
    assert str(wrong) in str(caught.value)
    assert out.exists() is existed
    if case == "out is a folder with a file in it":
        assert (out / "sentinel.txt").read_bytes() == b"keep me"
        assert [p.name for p in out.iterdir()] == ["sentinel.txt"]
    if case == "out is a file":
        assert out.read_text(encoding="utf-8") == "not a folder"


def test_export_takes_the_boundary_values_it_names_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = _source(tmp_path)
    monkeypatch.setattr(dataset_module, "ON_WINDOWS", True)
    longest = _longest_destination(tmp_path / "ok", seed)
    # One character under the limit is accepted, the limit itself is not.
    monkeypatch.setattr(dataset_module, "MAX_PATH", longest + 1)
    export_dataset(seed.path, tmp_path / "ok")
    monkeypatch.setattr(dataset_module, "MAX_PATH", longest)
    with pytest.raises(DatasetError, match="Windows takes"):
        export_dataset(seed.path, tmp_path / "no")
    monkeypatch.setattr(dataset_module, "ON_WINDOWS", False)

    # Free space equal to the need is accepted, one byte less is not.
    need = _need_of(seed)
    _free(monkeypatch, need)
    export_dataset(seed.path, tmp_path / "enough")
    _free(monkeypatch, need - 1)
    with pytest.raises(DatasetError, match="MB free"):
        export_dataset(seed.path, tmp_path / "short")

    # A disk that does not say how much is free is not a reason to refuse.
    def unknown(_path: Path) -> None:
        raise OSError("no answer")

    monkeypatch.setattr(dataset_module.shutil, "disk_usage", unknown)
    export_dataset(seed.path, tmp_path / "unknown")
    assert (tmp_path / "unknown" / "manifest.json").is_file()


def _record_free_space(monkeypatch: pytest.MonkeyPatch, free: int) -> list[Path]:
    """``disk_usage`` as the OS answers it: an error for a path that does not exist.

    The paths it is asked about are kept, so that a test sees *which* folder the
    export measured, which a stand-in that ignores its argument cannot tell.
    """
    asked: list[Path] = []

    def usage(path: Path) -> SimpleNamespace:
        asked.append(Path(path))
        if not Path(path).exists():
            raise FileNotFoundError(2, "No such file or directory", str(path))
        return SimpleNamespace(total=free * 2, used=free, free=free)

    monkeypatch.setattr(dataset_module.shutil, "disk_usage", usage)
    return asked


def test_the_free_space_is_asked_of_the_nearest_folder_that_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `out` does not exist yet, the normal case, and `disk_usage` of a path that
    # is not there is an error that the export reads as "unknown": a mutant that
    # asks about `out` itself is never refused (adversarial review of #565).
    seed = _source(tmp_path)
    need = _need_of(seed)
    out = tmp_path / "a" / "b"

    asked = _record_free_space(monkeypatch, need - 1)
    with pytest.raises(DatasetError, match="MB free"):
        export_dataset(seed.path, out)
    assert asked == [tmp_path]
    assert not (tmp_path / "a").exists()

    asked = _record_free_space(monkeypatch, need)
    export_dataset(seed.path, out)
    assert asked == [tmp_path]
    assert (out / "manifest.json").is_file()


def _fail_for(
    monkeypatch: pytest.MonkeyPatch, name: str, target: Path, error: OSError
) -> None:
    """Make ``Path.<name>`` raise ``error`` for ``target`` only, as a share that drops does."""
    original = getattr(Path, name)

    def refuse(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self == target:
            raise error
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, name, refuse)


@pytest.mark.parametrize("call", ["resolve", "is_dir", "iterdir", "exists"])
def test_an_out_that_cannot_be_examined_is_a_refusal_naming_it(
    call: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # FM-116: a share that drops answers each of these with an OSError, and an
    # export that lets it out ends on a traceback instead of the refusal.
    seed = _source(tmp_path)
    out = tmp_path / "out"
    if call == "iterdir":
        out.mkdir()  # only an existing folder is listed
    _fail_for(
        monkeypatch,
        call,
        out,
        OSError(64, "The specified network name is no longer available"),
    )
    monkeypatch.setattr(dataset_module, "_copy_and_hash", _no_copy)

    with pytest.raises(DatasetError) as caught:
        export_dataset(seed.path, out)

    message = str(caught.value)
    assert message.startswith(str(out))
    assert "cannot be examined" in message
    assert "network name is no longer available" in message


class _Refused:
    """A frame's source file whose ``stat`` the file system refuses: what ``Path`` is on a share that errs."""

    def is_file(self) -> bool:
        raise PermissionError(13, "Permission denied")


def test_a_frame_that_cannot_be_examined_is_not_a_crash_and_not_a_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = _source(tmp_path)
    frames = _frame_files(seed)
    real = dataset_module.resolve_in_run

    def refuses_the_first(base: Path, relative: str) -> Any:
        return _Refused() if relative == frames[0] else real(base, relative)

    monkeypatch.setattr(dataset_module, "resolve_in_run", refuses_the_first)
    monkeypatch.setattr(dataset_module, "_copy_and_hash", _no_copy)

    with pytest.raises(DatasetError) as caught:
        export_dataset(seed.path, tmp_path / "out")

    message = str(caught.value)
    assert "1 of 8 frame files cannot be read now (Permission denied)" in message
    assert frames[0] in message
    assert "are missing" not in message
    assert isinstance(caught.value.__cause__, PermissionError)
    assert not (tmp_path / "out").exists()


def test_a_run_folder_that_stops_answering_is_not_reported_as_missing_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # FM-116: `resolve_in_run` says ValueError for an OSError of `resolve` and
    # chains it; read as "not there", a dropped share told the operator that the
    # scan's folder was moved or partly deleted, and a night's scan looked lost.
    seed = _source(tmp_path)
    frames = _frame_files(seed)
    real = dataset_module.resolve_in_run
    dropped = OSError(64, "The specified network name is no longer available")

    def share_dropped(base: Path, relative: str) -> Path:
        if relative in frames:
            raise ValueError(
                f"{relative}: cannot be resolved ({dropped.strerror})"
            ) from dropped
        return real(base, relative)

    monkeypatch.setattr(dataset_module, "resolve_in_run", share_dropped)
    monkeypatch.setattr(dataset_module, "_copy_and_hash", _no_copy)

    with pytest.raises(DatasetError) as caught:
        export_dataset(seed.path, tmp_path / "out")

    message = str(caught.value)
    assert "8 of 8 frame files cannot be read now" in message
    assert "network name is no longer available" in message
    assert ", ".join(frames[:5]) in message
    assert "(and 3 more)" in message
    assert "try again" in message
    assert "are missing" not in message
    assert "moved" not in message
    assert str(seed.path) in message
    assert not (tmp_path / "out").exists()


def test_five_missing_frames_are_all_named_and_nothing_is_said_to_be_more(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The limit itself: `>` and not `>=` would write "(and 0 more)".
    seed = _source(tmp_path)
    frames = _frame_files(seed)
    for name in frames[:5]:
        (seed.path.parent / name).unlink()
    monkeypatch.setattr(dataset_module, "_copy_and_hash", _no_copy)

    with pytest.raises(DatasetError) as caught:
        export_dataset(seed.path, tmp_path / "out")

    message = str(caught.value)
    assert "5 of 8 frame files are missing" in message
    assert ", ".join(frames[:5]) in message
    assert "(and" not in message


def test_missing_frames_are_counted_and_the_first_five_are_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = _source(tmp_path)
    frames = _frame_files(seed)
    for name in frames[1:]:
        (seed.path.parent / name).unlink()
    monkeypatch.setattr(dataset_module, "_copy_and_hash", _no_copy)

    with pytest.raises(DatasetError) as caught:
        export_dataset(seed.path, tmp_path / "out")

    message = str(caught.value)
    assert "7 of 8 frame files are missing" in message
    assert ", ".join(frames[1:6]) in message
    assert frames[6] not in message
    assert "(and 2 more)" in message
    assert str(seed.path) in message
    assert not (tmp_path / "out").exists()


def test_a_frame_that_leads_out_of_the_run_folder_is_counted_as_missing_and_not_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = _source(tmp_path)
    (seed.path.parent.parent / "x.tif").write_bytes(b"not a frame of this run")

    def lead_out(stages: dict[str, Any]) -> None:
        stages["scan"]["result"]["frames"][0]["file"] = "../x.tif"

    _edit_stages(seed.path, lead_out)
    monkeypatch.setattr(dataset_module, "_copy_and_hash", _no_copy)

    with pytest.raises(
        DatasetError, match=r"1 of 8 frame files are missing.*\.\./x\.tif"
    ):
        export_dataset(seed.path, tmp_path / "out")


@pytest.mark.parametrize("second", ["scan/t/A1_0000.tif", "run.json", "README.md"])
def test_two_files_of_one_path_are_refused_before_the_first_is_copied(
    second: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The copy opens its destination exclusively, so a duplicate would stop the
    # export half-way; a frame named run.json would be replaced by the real one.
    seed = _source(tmp_path)
    assert _frame_files(seed)[0] == "scan/t/A1_0000.tif"
    (seed.path.parent / "README.md").write_bytes(
        b"a frame, as far as the document says"
    )

    def same_name(stages: dict[str, Any]) -> None:
        stages["scan"]["result"]["frames"][1]["file"] = second

    _edit_stages(seed.path, same_name)
    monkeypatch.setattr(dataset_module, "_copy_and_hash", _no_copy)

    with pytest.raises(DatasetError, match="two files of the dataset"):
        export_dataset(seed.path, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_names_that_differ_only_in_case_are_one_file_to_the_check() -> None:
    # On Windows and macOS `A1.tif` and `a1.TIF` are one file, and the second
    # copy would find the first one in its way.
    _check_names(Path("run.json"), ["scan/A1.tif", "scan/B1.tif"])
    with pytest.raises(DatasetError, match="two files of the dataset"):
        _check_names(Path("run.json"), ["scan/A1.tif", "scan/a1.TIF"])


def test_an_empty_existing_out_is_accepted_and_missing_parents_are_created(
    tmp_path: Path,
) -> None:
    seed = _source(tmp_path)
    deep = tmp_path / "a" / "b" / "c"
    export_dataset(seed.path, deep)
    assert (deep / "manifest.json").is_file()

    empty = tmp_path / "empty"
    empty.mkdir()
    export_dataset(seed.path, empty)
    assert (empty / "manifest.json").is_file()


def test_the_account_name_never_reaches_a_text_file_of_the_dataset(
    tmp_path: Path,
) -> None:
    seed = _source(tmp_path)
    out = tmp_path / "out"

    def with_paths(stages: dict[str, Any]) -> None:
        stages["scan"]["params"] = {
            "out": "C:\\Users\\alice\\models\\m.pt",
            "note": "/home/alice/models/m.pt",
            "unc": "\\\\server\\alice\\models\\m.pt",
        }
        stages["scan"]["result"]["assumed"] = ["read /home/alice/x.tif for the offsets"]
        stages["scan"]["error"] = "cannot open C:\\Users\\alice\\m.pt: not found"
        stages["cells"]["checkpoint"] = {"folder": "/Users/alice/run1"}

    _edit_stages(seed.path, with_paths)
    manifest = export_dataset(seed.path, out)

    for path in out.rglob("*"):
        if path.suffix in (".json", ".md"):
            text = path.read_text(encoding="utf-8")
            assert "alice" not in text, path.name
            assert str(seed.path.parent) not in text, path.name
            assert str(out) not in text, path.name
    assert manifest.scrubbed == (
        "/stages/scan/params/out",
        "/stages/scan/params/note",
        "/stages/scan/params/unc",
        "/stages/scan/result/assumed/0",
        "/stages/scan/error",
        "/stages/cells/checkpoint/folder",
    )
    on_disk = DatasetManifest.model_validate_json(
        (out / "manifest.json").read_text(encoding="utf-8")
    )
    assert on_disk.scrubbed == manifest.scrubbed
    assert load_scan(out / "run.json").wells == ("A1", "B2")
    # What the rule wrote in place of each path.
    document = RunDocument.load(out / "run.json")
    assert document.stages["scan"].params["out"] == "m.pt"
    assert document.stages["scan"].result is not None
    assert document.stages["scan"].result["assumed"] == ["read x.tif for the offsets"]
    assert document.stages["scan"].error == "cannot open m.pt: not found"
    assert document.stages["cells"].checkpoint == {"folder": "run1"}


def test_a_record_keeps_its_preset_and_every_other_field_through_the_export(
    tmp_path: Path,
) -> None:
    # FM-107: a field the scrub does not touch is carried as it is, whatever a
    # later version adds to the record: the loop is over the model's own list.
    seed = _source(tmp_path)

    def mark(stages: dict[str, Any]) -> None:
        stages["scan"]["preset"] = "mine"
        stages["cells"]["offline"] = True
        stages["scan"]["params"] = {"out": "/home/alice/x"}

    _edit_stages(seed.path, mark)
    out = tmp_path / "out"
    export_dataset(seed.path, out)

    before, after = RunDocument.load(seed.path), RunDocument.load(out / "run.json")
    assert list(after.stages) == list(before.stages)
    assert len(after.stages) == 4
    for name, record in before.stages.items():
        for field in StageRecord.model_fields:
            if field in ("params", "result", "checkpoint", "error"):
                continue
            assert getattr(after.stages[name], field) == getattr(record, field), (
                name,
                field,
            )
    assert after.stages["scan"].preset == "mine"
    assert after.stages["cells"].offline is True
    assert after.created_utc == before.created_utc


def test_an_export_of_an_export_rewrites_nothing_and_changes_no_byte(
    tmp_path: Path,
) -> None:
    seed = _source(tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"
    one = export_dataset(seed.path, first)
    two = export_dataset(first / "run.json", second)

    assert two.scrubbed == ()
    assert two.source == "run.json"
    assert (second / "run.json").read_bytes() == (first / "run.json").read_bytes()
    for name in [*_frame_files(seed), _cells_file(seed)]:
        assert two.files[name].sha256 == one.files[name].sha256, name


@pytest.mark.parametrize(
    "how", ["no cells stage", "failed cells stage", "count edited"]
)
def test_no_cells_is_a_dataset_with_the_reason(how: str, tmp_path: Path) -> None:
    seed = _source(tmp_path)
    if how == "no cells stage":
        _edit_stages(seed.path, lambda stages: stages.pop("cells"))
    elif how == "failed cells stage":
        _edit_stages(seed.path, lambda stages: stages["cells"].update(status="failed"))
    else:
        # A cell removed from the file: the file no longer holds what the result counted.
        cells_path = seed.path.parent / _cells_file(seed)
        raw = json.loads(cells_path.read_text(encoding="utf-8"))
        raw["cells"].pop()
        cells_path.write_text(json.dumps(raw), encoding="utf-8")
    reason = load_scan(seed.path).cells_absent
    assert reason

    out = tmp_path / "out"
    manifest = export_dataset(seed.path, out)

    assert manifest.cells is False
    assert manifest.cells_absent == reason
    assert not (out / "cells").exists()
    assert _cells_file(seed) not in manifest.files
    assert len(manifest.files) == len(_frame_files(seed)) + 2
    assert ("cells" in RunDocument.load(out / "run.json").stages) is (
        how != "no cells stage"
    )
    exported = load_scan(out / "run.json")
    assert exported.cells is None
    assert exported.cells_absent
    assert "none: " in (out / "README.md").read_text(encoding="utf-8")


def test_a_cells_file_with_no_cell_is_still_a_dataset_with_cells(
    tmp_path: Path,
) -> None:
    # Detected, found none, is not "never detected": kills `if scan.cells`.
    seed = _source(tmp_path, cells=[])
    out = tmp_path / "out"
    manifest = export_dataset(seed.path, out)

    assert manifest.cells is True
    assert manifest.cells_absent is None
    assert _cells_file(seed) in manifest.files
    assert (out / _cells_file(seed)).is_file()
    assert load_scan(out / "run.json").cells == ()


def test_the_cameras_values_are_the_scans_not_the_plans(tmp_path: Path) -> None:
    # FM-124: the plan was laid for another camera than the scan took.
    seed = _source(
        tmp_path, plan_matrix=matrix_of(2.0, 0.0, False), plan_shape=(100, 100)
    )
    manifest = export_dataset(seed.path, tmp_path / "out")
    assert manifest.image_shape_px == SHAPE == (60, 80)
    assert manifest.pixel_size_um == PIXEL_UM


def test_a_run_folder_with_spaces_and_brackets_exports(tmp_path: Path) -> None:
    # FM-24, FM-36: nothing in the export takes a path for a pattern or a markup.
    src = tmp_path / "src [1]"
    src.mkdir()
    seed = seed_run(src, cells=_three_cells(), wells=("A1", "B2"))
    document = src / "my run.json"
    seed.path.rename(document)
    out = tmp_path / "out [2]"

    manifest = export_dataset(document, out)

    assert manifest.source == "my run.json"
    for path in out.rglob("*"):
        if path.suffix in (".json", ".md"):
            text = path.read_text(encoding="utf-8")
            assert "src [1]" not in text, path.name
            assert "out [2]" not in text, path.name
    assert (out / "run.json").is_file()
    assert load_scan(out / "run.json").wells == ("A1", "B2")


def test_progress_is_called_once_per_file_copied(tmp_path: Path) -> None:
    seed = _source(tmp_path)
    calls: list[tuple[int, int]] = []

    export_dataset(
        seed.path, tmp_path / "out", progress=lambda d, t: calls.append((d, t))
    )

    n = len(_frame_files(seed)) + 1
    assert calls == [(k, n) for k in range(1, n + 1)]


def test_an_out_that_cannot_be_made_says_that_nothing_was_written(
    tmp_path: Path,
) -> None:
    # A drive that is not there (an unplugged stick) fails at the first `mkdir`,
    # before any file exists: "holds part of the dataset ... remove it" would
    # send the operator to a folder that is not there (adversarial review of #565).
    seed = _source(tmp_path)
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a folder", encoding="utf-8")
    out = blocker / "dataset"

    with pytest.raises(DatasetWriteError) as caught:
        export_dataset(seed.path, out)

    message = str(caught.value)
    assert message.startswith(str(out))
    assert "cannot be created" in message
    assert "nothing was written" in message
    assert "holds part of the dataset" not in message
    assert isinstance(caught.value.__cause__, OSError)
    assert not out.exists()


def _load_it_block(readme: str) -> str:
    """The code that the README's "Load it" section shows, dedented, ready to run."""
    lines = readme.splitlines()
    section = lines[lines.index("## Load it") + 1 : lines.index("## Check it")]
    code = [line for line in section if line.startswith("    ") or not line.strip()]
    return textwrap.dedent("\n".join(code))


def test_the_readme_of_an_export_says_what_this_scan_is_and_its_example_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Wells that are not A1, with cells: the README that `export_dataset` writes
    # must say this scan's date, cells file and classes (a mutant that passes
    # `None` for each still passes a test of `readme_text` on a hand-made
    # manifest), and the example it shows must run on this scan, which has no A1.
    seed = _source(
        tmp_path,
        wells=("B1", "B2"),
        cells=[cell_at("B1", 1.0, 1.0, "single"), cell_at("B2", 2.0, 1.0, "doublet")],
    )
    out = tmp_path / "out"
    export_dataset(seed.path, out)
    text = (out / "README.md").read_text(encoding="utf-8")

    started = RunDocument.load(seed.path).provider("scan")
    assert started is not None
    day = started.started_utc.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")
    lines = [line.split() for line in text.splitlines()]
    assert ["scanned", *day.split(), "UTC"] in lines
    assert (
        f"detected, listed in {_cells_file(seed)} "
        f"(classes: {', '.join(seed.result.classes)})"
    ) in text
    assert "none:" not in text
    assert "B1, B2 (the order of the scan plan)" in text

    monkeypatch.chdir(out)
    exec(compile(_load_it_block(text), "README.md", "exec"), {})


def test_a_non_ascii_objective_is_escaped_in_the_json_and_whole_in_the_readme(
    tmp_path: Path,
) -> None:
    # run.json and manifest.json are ASCII on purpose (the arguments of
    # `write_stage`), and nothing else in the export carries a non-ASCII
    # string: the flag that makes it so is read here, not in a table.
    objective = "Plan Apo \N{GREEK SMALL LETTER LAMDA} 20x"
    seed = _source(tmp_path, scan_objective=objective)
    out = tmp_path / "out"

    manifest = export_dataset(seed.path, out)

    assert manifest.objective == objective
    for name in ("run.json", "manifest.json"):
        assert (out / name).read_bytes().isascii(), name
    assert objective in (out / "README.md").read_text(encoding="utf-8")
    assert load_scan(out / "run.json").profile == "demo"
    assert check_dataset(out).problems == ()
