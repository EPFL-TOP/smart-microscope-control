"""``smc dataset export``: a scan's run folder written as a dataset (algorithms design §2.1, #565).

The run folders are ``tests/select_fields_seed.py::seed_run``'s, written with no
Micro-Manager; the oracles are ``hashlib`` and the bytes on disk, never the
export's own helpers.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

import smc.data.dataset as dataset_module
from smc.data.dataset import DatasetFile, DatasetManifest, _copy_and_hash, _sha256_of

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
