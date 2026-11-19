"""``smc dataset check``: a dataset folder against its manifest (algorithms design §2.4, #565).

The datasets are ``export_dataset``'s own, of a seeded run folder; every damage
is done to the bytes on disk, and every oracle is ``hashlib`` or the bytes, never
the check's own helpers.
"""

from __future__ import annotations

import codecs
import hashlib
import json
import shutil
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import smc.data.dataset as dataset_module
from select_fields_seed import Seed, cell_at, seed_run
from smc.data import DatasetError, Scan
from smc.data.dataset import DatasetManifest, check_dataset, export_dataset


def _exported(tmp_path: Path) -> tuple[Seed, Path, DatasetManifest]:
    """A seeded run folder of two wells and three cells, exported to ``tmp_path/out``."""
    src = tmp_path / "src"
    src.mkdir()
    seed = seed_run(
        src,
        cells=[cell_at("A1", 1.0, 1.0), cell_at("B2", 2.0, 1.0, "doublet")],
        wells=("A1", "B2"),
    )
    out = tmp_path / "out"
    return seed, out, export_dataset(seed.path, out)


def _frames(seed: Seed) -> list[str]:
    return [frame.file for frame in seed.scan.frames]


def _write_manifest(out: Path, raw: dict[str, Any]) -> None:
    (out / "manifest.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")


def _raw_manifest(out: Path) -> dict[str, Any]:
    loaded = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _reseal(out: Path) -> None:
    """Rewrite the manifest to match the files as they are, as a tool that edits the dataset and means it."""
    raw = _raw_manifest(out)
    for name, entry in raw["files"].items():
        data = (out / name).read_bytes()
        entry["bytes"] = len(data)
        entry["sha256"] = hashlib.sha256(data).hexdigest()
    _write_manifest(out, raw)


def test_check_accepts_a_fresh_export_and_the_same_folder_zipped_and_unzipped(
    tmp_path: Path,
) -> None:
    _, out, manifest = _exported(tmp_path)

    fresh = check_dataset(out)
    assert fresh.problems == ()
    assert isinstance(fresh.scan, Scan)
    assert fresh.manifest == manifest

    archive = shutil.make_archive(str(tmp_path / "travelled"), "zip", root_dir=out)
    elsewhere = tmp_path / "elsewhere"
    shutil.unpack_archive(archive, elsewhere)
    unzipped = check_dataset(elsewhere)
    assert unzipped.problems == ()
    assert isinstance(unzipped.scan, Scan)


def test_check_names_a_changed_a_truncated_and_a_missing_file(tmp_path: Path) -> None:
    seed, out, manifest = _exported(tmp_path)
    frames = _frames(seed)
    flipped, truncated, missing = frames[0], frames[2], frames[4]

    data = bytearray((out / flipped).read_bytes())
    data[-1] ^= 0xFF  # the last byte: the size is the same
    (out / flipped).write_bytes(bytes(data))
    kept = (out / truncated).read_bytes()
    (out / truncated).write_bytes(kept[:-10])
    (out / missing).unlink()

    result = check_dataset(out)

    assert result.problems == (
        f"{flipped}: changed (same size, other content)",
        f"{truncated}: changed (size {len(kept) - 10} instead of {len(kept)})",
        f"{missing}: missing",
    )
    assert result.scan is None
    assert result.manifest == manifest


def test_check_reads_each_file_once_and_not_a_file_of_another_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # FM-39: a hash is read once per file; a size that already differs is a
    # verdict, and reading the file as well would be a second pass over it.
    seed, out, manifest = _exported(tmp_path)
    shorter = _frames(seed)[1]
    (out / shorter).write_bytes((out / shorter).read_bytes()[:-1])
    reads: Counter[str] = Counter()
    original = dataset_module._sha256_of

    def counting(path: Path) -> str:
        reads[path.relative_to(out.resolve()).as_posix()] += 1
        return original(path)

    monkeypatch.setattr(dataset_module, "_sha256_of", counting)
    check_dataset(out)

    assert shorter not in reads
    assert set(reads) == set(manifest.files) - {shorter}
    assert set(reads.values()) == {1}


def test_check_says_an_interrupted_export_has_no_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    seed = seed_run(src, cells=[], wells=("A1",))
    out = tmp_path / "out"
    copy = dataset_module._copy_and_hash
    calls: list[int] = []

    def fails_on_the_third(source: Path, destination: Path) -> tuple[int, str]:
        calls.append(1)
        if len(calls) == 3:
            raise OSError("the stick was pulled out")
        return copy(source, destination)

    monkeypatch.setattr(dataset_module, "_copy_and_hash", fails_on_the_third)
    with pytest.raises(dataset_module.DatasetWriteError):
        export_dataset(seed.path, out)

    with pytest.raises(DatasetError) as caught:
        check_dataset(out)
    message = str(caught.value)
    assert str(out) in message
    assert "manifest.json" in message
    assert "did not finish" in message

    with pytest.raises(DatasetError, match="is not a folder") as not_a_folder:
        check_dataset(seed.path)
    assert str(seed.path) in str(not_a_folder.value)
    with pytest.raises(DatasetError, match="is not a folder"):
        check_dataset(tmp_path / "nowhere")


def _schema_two(raw: dict[str, Any]) -> None:
    raw["schema_version"] = 2


def _without_a_field(raw: dict[str, Any]) -> None:
    del raw["profile"]


def _with_an_unknown_field(raw: dict[str, Any]) -> None:
    raw["bogus"] = 1


def _without_the_run_document(raw: dict[str, Any]) -> None:
    del raw["files"]["run.json"]


def _no_schema_version(raw: dict[str, Any]) -> None:
    del raw["schema_version"]


@pytest.mark.parametrize(
    ("damage", "words"),
    [
        (_schema_two, "newer smc"),
        (_without_a_field, "profile"),
        (_with_an_unknown_field, "bogus"),
        (_without_the_run_document, "lists no run.json"),
        (_no_schema_version, "schema_version"),
    ],
)
def test_check_refuses_a_manifest_it_cannot_read(
    damage: Callable[[dict[str, Any]], None], words: str, tmp_path: Path
) -> None:
    _, out, _ = _exported(tmp_path)
    raw = _raw_manifest(out)
    damage(raw)
    _write_manifest(out, raw)

    with pytest.raises(DatasetError) as caught:
        check_dataset(out)
    assert str(caught.value).startswith(str(out / "manifest.json"))
    assert words in str(caught.value)


@pytest.mark.parametrize(
    ("content", "words"),
    [
        (b"{not json", "not valid JSON"),
        (b"[1, 2]", "JSON list"),
        (b"\xff\xfe\x00", "not UTF-8"),
        (b'{"schema_version": true}', "schema_version"),
    ],
)
def test_check_refuses_a_manifest_that_is_not_one(
    content: bytes, words: str, tmp_path: Path
) -> None:
    _, out, _ = _exported(tmp_path)
    (out / "manifest.json").write_bytes(content)

    with pytest.raises(DatasetError) as caught:
        check_dataset(out)
    assert str(caught.value).startswith(str(out / "manifest.json"))
    assert words in str(caught.value)


def test_a_manifest_with_a_byte_order_mark_is_read(tmp_path: Path) -> None:
    # FM-80: Notepad adds one when someone opens the manifest and saves it.
    _, out, _ = _exported(tmp_path)
    path = out / "manifest.json"
    path.write_bytes(codecs.BOM_UTF8 + path.read_bytes())

    assert check_dataset(out).problems == ()


@pytest.mark.parametrize("key", ["../outside.bin", "/abs/x", "C:\\x", "a/../../b"])
def test_a_manifest_that_lists_a_path_outside_the_folder_is_refused_before_any_file_is_read(
    key: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, out, _ = _exported(tmp_path)
    # A file outside the folder whose size and hash match what the manifest says:
    # a check that does not guard the path would pass.
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"a file of the PC")
    raw = _raw_manifest(out)
    raw["files"][key] = {
        "bytes": outside.stat().st_size,
        "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
    }
    _write_manifest(out, raw)

    def never(_path: Path) -> str:
        raise AssertionError("a file was read before the manifest's paths were judged")

    monkeypatch.setattr(dataset_module, "_sha256_of", never)
    with pytest.raises(DatasetError) as caught:
        check_dataset(out)
    assert repr(key) in str(caught.value)
    assert "not a path in the dataset folder" in str(caught.value)


def test_check_refuses_a_dataset_the_loader_refuses(tmp_path: Path) -> None:
    # Every file matches its manifest, and the scan in the folder is a dry run.
    _, out, _ = _exported(tmp_path)
    document = json.loads((out / "run.json").read_text(encoding="utf-8"))
    document["stages"]["scan"]["dry_run"] = True
    (out / "run.json").write_text(json.dumps(document, indent=2), encoding="utf-8")
    _reseal(out)

    with pytest.raises(DatasetError, match="dry run"):
        check_dataset(out)


def test_check_ignores_what_the_manifest_does_not_list(tmp_path: Path) -> None:
    # The collaborator's own outputs live beside the dataset.
    _, out, _ = _exported(tmp_path)
    (out / "draw").mkdir()
    (out / "draw" / "A1.png").write_bytes(b"a drawing")
    (out / "notes.txt").write_text("my notes", encoding="utf-8")

    assert check_dataset(out).problems == ()


def test_an_unreadable_file_is_a_problem_not_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed, out, manifest = _exported(tmp_path)
    locked = _frames(seed)[3]
    original = dataset_module._sha256_of
    checked: list[str] = []

    def refuses_one(path: Path) -> str:
        name = path.relative_to(out.resolve()).as_posix()
        checked.append(name)
        if name == locked:
            raise PermissionError(13, "Permission denied")
        return original(path)

    monkeypatch.setattr(dataset_module, "_sha256_of", refuses_one)
    result = check_dataset(out)

    assert result.problems == (f"{locked}: cannot be read (Permission denied)",)
    assert result.scan is None
    assert checked == list(manifest.files)  # the others were still checked


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


def test_a_folder_that_cannot_be_examined_is_not_a_folder_to_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # FM-116: `is_dir` of a share that dropped raises; the refusal is the one of
    # a path that is not a folder, not a traceback.
    _, out, _ = _exported(tmp_path)
    _fail_for(
        monkeypatch,
        "is_dir",
        out,
        OSError(64, "The specified network name is no longer available"),
    )

    with pytest.raises(DatasetError, match="is not a folder") as caught:
        check_dataset(out)
    assert str(out) in str(caught.value)


def test_a_manifest_that_cannot_be_read_is_a_refusal_that_starts_with_its_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, out, _ = _exported(tmp_path)
    _fail_for(
        monkeypatch,
        "read_bytes",
        out / "manifest.json",
        OSError(5, "Input/output error"),
    )

    with pytest.raises(DatasetError) as caught:
        check_dataset(out)
    assert str(caught.value).startswith(str(out / "manifest.json"))
    assert "cannot be read (Input/output error)" in str(caught.value)
    assert isinstance(caught.value.__cause__, OSError)


def test_a_file_whose_stat_fails_is_a_problem_and_the_others_are_still_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed, out, manifest = _exported(tmp_path)
    locked = _frames(seed)[3]
    original = Path.stat

    def refuses_one(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self.as_posix().endswith("/" + locked):
            raise PermissionError(13, "Permission denied")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", refuses_one)
    result = check_dataset(out)

    assert result.problems == (f"{locked}: cannot be read (Permission denied)",)
    assert result.scan is None
    assert result.manifest == manifest


def test_progress_is_called_once_per_listed_file(tmp_path: Path) -> None:
    _, out, manifest = _exported(tmp_path)
    calls: list[tuple[int, int]] = []

    check_dataset(out, progress=lambda done, total: calls.append((done, total)))

    n = len(manifest.files)
    assert calls == [(k, n) for k in range(1, n + 1)]
