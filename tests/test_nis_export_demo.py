"""``nis-export`` through ``run_plugin``: the route in, one NIS-Elements position list out (m3 §14, #196).

The plugin reads the ``route`` record and nothing else (``requires = ()``): the demo stand is
here because ``run_plugin`` needs a ``Microscope``. The record is written by the test, as
``tests/test_report_demo.py`` writes its own, and the file is read back with ``decode``, whose
own tests are ``tests/unit/test_nis_format.py``. Every expected value is computed here from the
seed and never read from the result's own fields (FM-124).
"""

from __future__ import annotations

import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from smc.hardware import Microscope
from smc.plugins import PluginRunError, load, run_plugin
from smc.plugins.errors import ParamsError
from smc.plugins.nis_export import plugin as plugin_module
from smc.plugins.nis_export.format import NisPosition, decode
from smc.plugins.nis_export.plugin import NisExportParams, NisExportPlugin
from smc.plugins.params import parse_params
from smc.plugins.slots import NisExportResult, RoutePosition, RouteResult
from smc.run import RunDocument, StageRecord, StageStatus, write_stage
from smc.run.files import AtomicWriteError

pytestmark = pytest.mark.demo

#: Exactly the texts of m3 §14: an operator reads them, and the owner compares them at the stand.
FRAME = (
    "x_um, y_um and z_um are Micro-Manager's readings in µm and pfs_offset is "
    "PFSOffset's own unit, written unconverted: that NIS-Elements shares their "
    "origin and direction is not measured"
)
BOTH_FLAGS = (
    "bIncludeZ and bPFSEnabled are both set, which the NIS-Elements sample did not show"
)


def _stop(
    order: int,
    name: str,
    x_um: float,
    y_um: float,
    z_um: float | None,
    pfs_offset: float | None,
) -> RoutePosition:
    return RoutePosition(
        order=order,
        well=name.split("_")[0],
        name=name,
        x_um=x_um,
        y_um=y_um,
        z_um=z_um,
        pfs_offset=pfs_offset,
        counts={"single": 1},
    )


#: Three stops in two wells, off the integers so that a value read from the wrong field shows.
STOPS = (
    _stop(0, "A1_r000", -11613.300000000001, 4822.5, 4089.12, 9370.0),
    _stop(1, "A1_r001", -8271.25, 5612.6, 4089.1000000000004, 9368.0),
    _stop(2, "B2_r000", 12000.75, -9500.125, 4100.5, 9401.0),
)


def _route(stops: tuple[RoutePosition, ...] = STOPS) -> RouteResult:
    return RouteResult(
        objective="Nikon 10X S Fluor",
        matrix_um_per_px=((5.0, 0.0), (0.0, -5.0)),
        image_shape_px=(60, 80),
        fov_um=(400.0, 300.0),
        labels=("single",),
        positions=stops,
        uncovered={},
        distance_um=0.0,
        travel_s=None,
    )


def _record(path: Path, result: RouteResult) -> None:
    """The ``route`` record as an earlier run would have left it, ``done``."""
    then = datetime(2026, 10, 29, 9, 30, tzinfo=timezone.utc)
    write_stage(
        path,
        StageRecord(
            plugin="route",
            provides="route",
            attempt="a1",
            status=StageStatus.done,
            started_utc=then,
            finished_utc=then,
            profile="demo",
            dry_run=False,
            smc_version="0",
            params={},
            result=result.model_dump(mode="json"),
        ),
    )


def _seed(tmp_path: Path, stops: tuple[RoutePosition, ...] = STOPS) -> None:
    _record(tmp_path / "run.json", _route(stops))


def _run(
    microscope: Microscope,
    tmp_path: Path,
    *,
    stop: threading.Event | None = None,
    **params: Any,
) -> StageRecord:
    return run_plugin(
        NisExportPlugin(),
        microscope,
        NisExportParams(**params),
        tmp_path / "run.json",
        stop=stop,
    )


def _result(record: StageRecord) -> NisExportResult:
    assert record.status is StageStatus.done
    assert record.result is not None
    return NisExportResult.model_validate(record.result)


def _stored(tmp_path: Path) -> StageRecord:
    return RunDocument.load(tmp_path / "run.json").stages["nis-export"]


def _points_of(tmp_path: Path, record: StageRecord) -> tuple[NisPosition, ...]:
    return decode((tmp_path / _result(record).file).read_bytes())


def _refused(microscope: Microscope, tmp_path: Path, **params: Any) -> PluginRunError:
    """Run and expect a ``PluginRunError``; the record is ``failed``, has no result, no file.

    Raised outside the handler that caught its cause (FM-98), so that it holds neither the
    file's error nor the microscope's frames.
    """
    with pytest.raises(PluginRunError) as info:
        _run(microscope, tmp_path, **params)
    assert _stored(tmp_path).status is StageStatus.failed
    assert _stored(tmp_path).result is None
    assert not list((tmp_path / "nis-export").glob("*.xml"))
    assert info.value.__context__ is None
    assert info.value.__cause__ is None
    return info.value


# ---------------------------------------------------------------- registration


def test_the_entry_point_loads_nis_export() -> None:
    entry = load("nis-export")

    assert entry.plugin.provides == "nis-export"
    assert entry.plugin.requires == ()
    assert entry.plugin.reads == ("route",)
    assert entry.plugin.reads_optional == ()
    assert entry.target == "smc.plugins.nis_export.plugin:NisExportPlugin"


# ------------------------------------------------------------------ the result


def test_nis_export_through_the_runner_writes_the_route_in_order(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    _seed(tmp_path)

    first = _run(demo_microscope, tmp_path)

    name = _result(first).file
    assert re.fullmatch(r"nis-export/[0-9a-f]{12}\.xml", name)
    assert "\\" not in name  # POSIX on every OS
    assert _result(first).positions == 3
    assert _stored(tmp_path) == first
    data = (tmp_path / name).read_bytes()
    assert data.startswith(b"\xff\xfe")  # the file NIS reads: UTF-16 with a BOM
    # Names, X and Y in the route's order: the seed's own values.
    points = decode(data)
    assert [(p.name, p.x_um, p.y_um) for p in points] == [
        ("A1_r000", -11613.300000000001, 4822.5),
        ("A1_r001", -8271.25, 5612.6),
        ("B2_r000", 12000.75, -9500.125),
    ]
    before = data

    second = _run(demo_microscope, tmp_path)

    # A `done` result's file is never rewritten: the next run has its own.
    assert _result(second).file != name
    assert (tmp_path / name).read_bytes() == before
    assert len(list((tmp_path / "nis-export").glob("*.xml"))) == 2


def test_stops_that_all_have_z_and_an_offset_are_exported_in_pfs_offset_mode(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    _seed(tmp_path)

    record = _run(demo_microscope, tmp_path)

    result = _result(record)
    assert result.z_mode == "pfs-offset"
    assert result.assumed == (FRAME, BOTH_FLAGS)
    text = (tmp_path / result.file).read_bytes().decode("utf-16")
    assert 'bIncludeZ runtype="bool" value="true"/>' in text
    assert 'bPFSEnabled runtype="bool" value="true"/>' in text
    assert [(p.z_um, p.pfs_offset) for p in decode(text.encode("utf-16"))] == [
        (4089.12, 9370.0),
        (4089.1000000000004, 9368.0),
        (4100.5, 9401.0),
    ]


def test_a_stop_without_an_offset_exports_z_only_and_says_how_many(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    stops = (STOPS[0], _stop(1, "A1_r001", -8271.25, 5612.6, 4090.5, None), STOPS[2])
    _seed(tmp_path, stops)

    record = _run(demo_microscope, tmp_path)

    result = _result(record)
    assert result.z_mode == "z"
    assert result.assumed == (
        FRAME,
        "1 of 3 positions have no pfs_offset: the export carries Z without the "
        "autofocus (bPFSEnabled false, dPFSOffset written as 0)",
    )
    text = (tmp_path / result.file).read_bytes().decode("utf-16")
    assert 'bIncludeZ runtype="bool" value="true"/>' in text
    assert 'bPFSEnabled runtype="bool" value="false"/>' in text
    points = _points_of(tmp_path, record)
    assert [p.z_um for p in points] == [4089.12, 4090.5, 4100.5]
    assert all(p.pfs_offset is None for p in points)


def test_a_stop_without_z_exports_no_focus_and_says_how_many(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    # The first stop has a Z and an offset, the second has neither (its well was all unlocked,
    # FM-154): the whole list goes without a focus, and the first stop's Z is not carried.
    stops = (STOPS[0], _stop(1, "A1_r001", -8271.25, 5612.6, None, None), STOPS[2])
    _seed(tmp_path, stops)

    record = _run(demo_microscope, tmp_path)

    result = _result(record)
    assert result.z_mode == "none"
    assert result.assumed == (
        FRAME,
        "1 of 3 positions have no z_um: the export carries neither Z nor an autofocus "
        "offset (bIncludeZ and bPFSEnabled false, both written as 0); NIS-Elements "
        "keeps the focus it has",
    )
    text = (tmp_path / result.file).read_bytes().decode("utf-16")
    assert 'bIncludeZ runtype="bool" value="false"/>' in text
    assert 'bPFSEnabled runtype="bool" value="false"/>' in text
    assert all(
        p.z_um is None and p.pfs_offset is None for p in _points_of(tmp_path, record)
    )


def test_the_count_of_a_stop_without_z_is_not_the_count_without_an_offset(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    # `none` counts the stops without z_um (2), and not those without an offset (3).
    stops = (
        _stop(0, "A1_r000", 1.5, 2.5, None, None),
        _stop(1, "A1_r001", 3.5, 4.5, 10.0, None),
        _stop(2, "B2_r000", 5.5, 6.5, None, None),
    )
    _seed(tmp_path, stops)

    result = _result(_run(demo_microscope, tmp_path))

    assert result.z_mode == "none"
    assert result.assumed[1].startswith("2 of 3 positions have no z_um:")


def test_a_stop_with_an_offset_but_no_z_exports_no_focus(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    # A scan writes both values from one frame, so this is a record edited behind the
    # runner's back: Z decides first, and an offset alone does not carry the list.
    stops = (STOPS[0], _stop(1, "A1_r001", -8271.25, 5612.6, None, 9368.0), STOPS[2])
    _seed(tmp_path, stops)

    record = _run(demo_microscope, tmp_path)

    result = _result(record)
    assert result.z_mode == "none"
    assert result.assumed[1].startswith("1 of 3 positions have no z_um:")
    assert all(
        p.z_um is None and p.pfs_offset is None for p in _points_of(tmp_path, record)
    )


def test_a_successful_run_logs_the_file_the_positions_and_the_mode(
    demo_microscope: Microscope, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _seed(tmp_path)

    with caplog.at_level("INFO", logger="smc.plugins.nis-export"):
        record = _run(demo_microscope, tmp_path)

    ours = [
        r.getMessage() for r in caplog.records if r.name == "smc.plugins.nis-export"
    ]
    assert ours == [
        f"NIS-Elements export written: {_result(record).file} "
        f"(3 positions, z_mode pfs-offset)"
    ]


def test_a_dry_run_writes_the_export_like_any_other(
    demo_microscope_dry: Microscope, tmp_path: Path
) -> None:
    # Nothing here moves the stand.
    _seed(tmp_path)

    record = _run(demo_microscope_dry, tmp_path)

    assert (tmp_path / _result(record).file).is_file()
    assert record.dry_run is True


# ----------------------------------------------------------------- the refusals


def test_an_empty_route_is_refused(demo_microscope: Microscope, tmp_path: Path) -> None:
    _seed(tmp_path, ())

    refused = _refused(demo_microscope, tmp_path)

    assert refused.reason == (
        "the route has no positions, so there is nothing to export; run route again "
        "with wells that hold cells of the wanted labels"
    )


@pytest.mark.parametrize(
    ("orders", "position", "found"),
    [
        pytest.param((0, 2, 1), 1, 2, id="an order above its position"),
        pytest.param((0, 0, 1), 1, 0, id="a duplicate: an order below its position"),
        pytest.param((1, 2, 3), 0, 1, id="a route that starts at 1"),
    ],
)
def test_a_route_whose_orders_are_not_0_to_n_is_refused(
    demo_microscope: Microscope,
    tmp_path: Path,
    orders: tuple[int, ...],
    position: int,
    found: int,
) -> None:
    # `!=` and not `>`: the slot model refuses neither a duplicate nor a late start.
    stops = tuple(
        stop.model_copy(update={"order": order})
        for stop, order in zip(STOPS, orders, strict=True)
    )
    _seed(tmp_path, stops)

    refused = _refused(demo_microscope, tmp_path)

    assert refused.reason == (
        f"the route's position {position} has order {found}; run route again"
    )


def test_a_name_the_list_cannot_carry_is_refused_naming_the_route(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    name = "A1_r001" + chr(0x01)
    stops = (STOPS[0], STOPS[1].model_copy(update={"name": name}), STOPS[2])
    _seed(tmp_path, stops)

    refused = _refused(demo_microscope, tmp_path)

    assert refused.reason == (
        f"position 1: name {name!r} holds {chr(0x01)!r}, which the list cannot "
        f"carry; run route again"
    )


def test_a_stop_set_before_the_run_leaves_no_file(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    _seed(tmp_path)
    stop = threading.Event()
    stop.set()

    record = _run(demo_microscope, tmp_path, stop=stop)

    assert record.status is StageStatus.stopped
    assert _stored(tmp_path).status is StageStatus.stopped
    assert not (tmp_path / "nis-export").exists()


# ------------------------------------------------------------------- out, overwrite


def test_out_writes_there_and_records_its_absolute_posix_path(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    _seed(tmp_path)
    folder = tmp_path / "nis folder"
    folder.mkdir()
    out = folder / "plate 1.XML"  # a space and an upper-case suffix (FM-24)

    record = _run(demo_microscope, tmp_path, out=out)

    assert _result(record).file == out.resolve().as_posix()
    assert _result(record).positions == 3
    assert [p.name for p in decode(out.read_bytes())] == [
        "A1_r000",
        "A1_r001",
        "B2_r000",
    ]
    assert not (tmp_path / "nis-export").exists()


def test_a_relative_out_is_relative_to_the_current_folder(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(tmp_path)
    here = tmp_path / "elsewhere"
    here.mkdir()
    monkeypatch.chdir(here)

    record = _run(demo_microscope, tmp_path, out=Path("list.xml"))

    assert _result(record).file == (here / "list.xml").resolve().as_posix()
    assert (here / "list.xml").is_file()


def test_an_out_inside_the_run_folder_is_still_recorded_as_an_absolute_path(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    _seed(tmp_path)
    out = tmp_path / "list.xml"

    record = _run(demo_microscope, tmp_path, out=out)

    assert _result(record).file == out.resolve().as_posix()


def test_an_existing_out_is_refused_without_overwrite_and_left_untouched(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    _seed(tmp_path)
    out = tmp_path / "list.xml"
    out.write_bytes(b"NIS holds this one open")

    refused = _refused(demo_microscope, tmp_path, out=out)

    assert refused.reason == (
        f"{out.resolve().as_posix()} exists; pass overwrite=true to replace it, "
        f"or choose another out"
    )
    assert out.read_bytes() == b"NIS holds this one open"


def test_overwrite_replaces_an_existing_out(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    _seed(tmp_path)
    out = tmp_path / "list.xml"
    out.write_bytes(b"the old list")

    record = _run(demo_microscope, tmp_path, out=out, overwrite=True)

    assert _result(record).file == out.resolve().as_posix()
    assert len(decode(out.read_bytes())) == 3
    # No temporary file is left beside it.
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".")] == []


def test_an_out_that_is_a_folder_is_refused_even_with_overwrite(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    _seed(tmp_path)
    out = tmp_path / "d.xml"
    out.mkdir()
    reason = (
        f"{out.resolve().as_posix()} is a folder; out names the file to write, "
        f"ending in .xml"
    )

    assert _refused(demo_microscope, tmp_path, out=out).reason == reason
    assert _refused(demo_microscope, tmp_path, out=out, overwrite=True).reason == reason
    assert list(out.iterdir()) == []


def test_an_out_in_a_folder_that_does_not_exist_is_refused_not_created(
    demo_microscope: Microscope, tmp_path: Path
) -> None:
    run_folder = tmp_path / "run"
    run_folder.mkdir()
    _seed(run_folder)
    out = tmp_path / "no such folder" / "list.xml"

    refused = _refused(demo_microscope, run_folder, out=out)

    # The strerror is the OS's own words: the whole reason is not pinned here.
    assert refused.reason.startswith(f"cannot write {out.resolve().as_posix()} (")
    assert refused.reason.endswith(
        "); check that its folder exists and can be written, then run nis-export again"
    )
    assert not out.parent.exists()


def test_out_must_end_with_xml() -> None:
    with pytest.raises(ParamsError) as txt:
        parse_params(NisExportParams, ["out=list.txt"])
    with pytest.raises(ParamsError) as empty:
        parse_params(NisExportParams, ["out="])

    assert txt.value.problems == (
        "out: Value error, out must end with .xml, not 'list.txt'",
    )
    assert empty.value.problems == ("out: Value error, out must end with .xml, not ''",)
    assert parse_params(NisExportParams, ["out=a.XML"]).out == Path("a.XML")
    assert parse_params(NisExportParams, []).out is None


def test_the_defaults_write_into_the_run_folder_and_do_not_overwrite() -> None:
    params = NisExportParams()

    assert params.out is None
    assert params.overwrite is False


@pytest.mark.parametrize(
    ("error", "shown"),
    [
        pytest.param(
            (64, "The specified network name is no longer available"),
            "The specified network name is no longer available",
            id="with an errno and its text",
        ),
        # Some Windows network failures carry a message and no errno: `strerror` is None.
        pytest.param(("the share dropped",), "the share dropped", id="a message only"),
    ],
)
def test_a_resolve_that_fails_is_a_plugin_run_error(
    demo_microscope: Microscope,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: tuple[Any, ...],
    shown: str,
) -> None:
    # FM-116: Windows re-raises a few network errors out of `resolve` (a share that
    # drops). Only the operator's `out` fails: the runner resolves paths too.
    _seed(tmp_path)
    out = tmp_path / "share" / "list.xml"
    real = Path.resolve

    def resolve(self: Path, strict: bool = False) -> Path:
        if self == out:
            raise OSError(*error)
        return real(self, strict=strict)

    monkeypatch.setattr(Path, "resolve", resolve)

    refused = _refused(demo_microscope, tmp_path, out=out)

    assert refused.reason == (
        f"cannot resolve out {out} ({shown}); check the path, then run nis-export again"
    )


@pytest.mark.parametrize("check", ["is_dir", "exists"])
def test_a_filesystem_error_from_a_check_of_out_is_a_plugin_run_error(
    demo_microscope: Microscope,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    check: str,
) -> None:
    # FM-116: `is_dir` and `exists` raise what pathlib does not ignore (a folder the
    # user cannot search, a share that drops), and the stage must name the path, not
    # end with a bare exception. Both calls are tried: `exists` runs after `is_dir`.
    _seed(tmp_path)
    out = tmp_path / "share" / "list.xml"
    target = out.resolve()
    real = getattr(Path, check)

    def failing(self: Path, *args: Any, **kwargs: Any) -> bool:
        if self == target:
            raise OSError(64, "The specified network name is no longer available")
        return bool(real(self, *args, **kwargs))

    monkeypatch.setattr(Path, check, failing)

    refused = _refused(demo_microscope, tmp_path, out=out)

    assert refused.reason == (
        f"cannot resolve out {out} (The specified network name is no longer "
        f"available); check the path, then run nis-export again"
    )


# ------------------------------------------------------------------- the write


def test_a_write_that_fails_names_the_file_and_says_to_run_nis_export_again(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(tmp_path)
    monkeypatch.setattr(plugin_module, "new_token", lambda: "abc123abc123")

    def disk_full(path: Path, data: bytes, **kwargs: Any) -> None:
        raise AtomicWriteError(OSError(28, "No space left on device"))

    monkeypatch.setattr(plugin_module, "write_atomically", disk_full)

    refused = _refused(demo_microscope, tmp_path)

    assert refused.reason == (
        "cannot write nis-export/abc123abc123.xml (No space left on device); check "
        "that its folder exists and can be written, then run nis-export again"
    )
    assert str(tmp_path) not in refused.reason


def test_a_write_that_only_failed_to_rename_names_the_kept_file(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the rename failed: the whole export waits in a temporary file, which the error names (FM-38)."""
    _seed(tmp_path)
    monkeypatch.setattr(plugin_module, "new_token", lambda: "abc123abc123")
    kept = tmp_path / "nis-export" / ".abc123abc123.xml.1234abcd.smc-tmp"

    def rename_refused(path: Path, data: bytes, **kwargs: Any) -> None:
        raise AtomicWriteError(PermissionError(13, "Access is denied"), kept=kept)

    monkeypatch.setattr(plugin_module, "write_atomically", rename_refused)

    refused = _refused(demo_microscope, tmp_path)

    assert refused.reason == (
        "cannot write nis-export/abc123abc123.xml (Access is denied); the whole "
        "export is kept as nis-export/.abc123abc123.xml.1234abcd.smc-tmp; close any "
        "program that holds nis-export/abc123abc123.xml (NIS-Elements may), then run "
        "nis-export again"
    )


def test_a_write_to_an_out_outside_the_run_folder_names_it_by_its_absolute_path(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_folder = tmp_path / "run"
    run_folder.mkdir()
    _seed(run_folder)
    out = tmp_path / "nis" / "list.xml"
    out.parent.mkdir()
    kept = out.parent / ".list.xml.1234abcd.smc-tmp"
    seen: list[tuple[Path, bool]] = []

    def rename_refused(path: Path, data: bytes, **kwargs: Any) -> None:
        seen.append((path, kwargs.get("make_parents", True)))
        raise AtomicWriteError(PermissionError(13, "Access is denied"), kept=kept)

    monkeypatch.setattr(plugin_module, "write_atomically", rename_refused)

    refused = _refused(demo_microscope, run_folder, out=out)

    where = out.resolve().as_posix()
    assert refused.reason == (
        f"cannot write {where} (Access is denied); the whole export is kept as "
        f"{kept.as_posix()}; close any program that holds {where} (NIS-Elements "
        f"may), then run nis-export again"
    )
    # A mistyped folder is refused, not created: `make_parents=False` for an `out`.
    assert seen == [(out.resolve(), False)]


def test_an_out_inside_the_run_folder_is_named_in_a_write_error_relative_to_it(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(tmp_path)
    out = tmp_path / "list.xml"

    def disk_full(path: Path, data: bytes, **kwargs: Any) -> None:
        raise AtomicWriteError(OSError(28, "No space left on device"))

    monkeypatch.setattr(plugin_module, "write_atomically", disk_full)

    refused = _refused(demo_microscope, tmp_path, out=out)

    assert refused.reason == (
        "cannot write list.xml (No space left on device); check that its folder "
        "exists and can be written, then run nis-export again"
    )


def test_a_write_error_names_an_out_by_its_absolute_path_when_it_cannot_be_compared_to_the_run_folder(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `relative_to_run` resolves both paths, and `resolve` re-raises a few network errors
    # (FM-116): naming the file in the error must not be what ends the stage.
    _seed(tmp_path)
    out = tmp_path / "list.xml"

    def cannot_compare(run_dir: Path, path: Path) -> str:
        raise OSError(64, "The specified network name is no longer available")

    def disk_full(path: Path, data: bytes, **kwargs: Any) -> None:
        raise AtomicWriteError(OSError(28, "No space left on device"))

    monkeypatch.setattr(plugin_module, "relative_to_run", cannot_compare)
    monkeypatch.setattr(plugin_module, "write_atomically", disk_full)

    refused = _refused(demo_microscope, tmp_path, out=out)

    assert refused.reason == (
        f"cannot write {out.resolve().as_posix()} (No space left on device); check "
        f"that its folder exists and can be written, then run nis-export again"
    )


def test_a_write_error_without_a_text_does_not_print_none(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(tmp_path)
    monkeypatch.setattr(plugin_module, "new_token", lambda: "abc123abc123")

    def no_text(path: Path, data: bytes, **kwargs: Any) -> None:
        raise AtomicWriteError(OSError("the share dropped"))

    monkeypatch.setattr(plugin_module, "write_atomically", no_text)

    refused = _refused(demo_microscope, tmp_path)

    assert refused.reason.startswith("cannot write nis-export/abc123abc123.xml (")
    assert "(None)" not in refused.reason
    assert refused.reason.endswith(
        "); check that its folder exists and can be written, then run nis-export again"
    )


def test_the_default_file_is_made_in_a_folder_that_is_made_for_it(
    demo_microscope: Microscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `make_parents` is True for the run folder's own `nis-export/` (the default).
    _seed(tmp_path)
    seen: list[bool] = []
    real = plugin_module.write_atomically

    def recording(path: Path, data: bytes, **kwargs: Any) -> None:
        seen.append(kwargs.get("make_parents", True))
        real(path, data, **kwargs)

    monkeypatch.setattr(plugin_module, "write_atomically", recording)

    _run(demo_microscope, tmp_path)

    assert seen == [True]
