"""Presets: the file, its names and folders, what it is checked against, and the command's pairs (m5 §10, #319).

Every test writes under ``tmp_path`` with ``SMC_PROFILES`` and the working
directory pointed there, so that none writes into the repository's
``profiles/presets``.
"""

from __future__ import annotations

import codecs
import json
import os
import pickle
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import BaseModel

from smc.hardware.profile import Profile
from smc.plugins.presets import (
    NOT_IN_PARAMS,
    Preset,
    PresetCheck,
    PresetError,
    PresetExistsError,
    PresetFile,
    check_name,
    check_preset,
    command_pairs,
    dumps_preset,
    list_presets,
    load_preset,
    preset_dirs,
    registered_plate,
    save_preset,
)
from smc.run import RunDocument, StageRecord, StageStatus
from smc.run import files as files_mod

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on the 3.10 CI job
    import tomli as tomllib

T0 = datetime(2026, 11, 3, 9, 0, tzinfo=timezone.utc)


@pytest.fixture
def profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Profile:
    """A stand whose profile is in ``work/profiles``, with another folder in ``$SMC_PROFILES``."""
    work = tmp_path / "work"
    (work / "profiles").mkdir(parents=True)
    (tmp_path / "site").mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setenv("SMC_PROFILES", str(tmp_path / "site"))
    path = work / "profiles" / "bench.toml"
    path.write_text('[microscope]\nname = "bench"\n', encoding="utf-8")
    return Profile.load(path)


def _put(folder: Path, name: str, text: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.toml"
    path.write_text(text, encoding="utf-8")
    return path


def _file(preset: Preset, name: str = "weekly") -> PresetFile:
    return PresetFile(name, Path(f"{name}.toml"), preset)


class ScanPlanParams(BaseModel):
    wells: str = ""
    coverage: float = 1.0
    objective: str = ""


class CellsParams(BaseModel):
    detector: str = "blobs"


class RegistrationParams(BaseModel):
    plate: str = ""
    refs: str = ""
    walls: str = ""


MODELS: dict[str, type[BaseModel]] = {
    "scan-plan": ScanPlanParams,
    "cells": CellsParams,
    "plate-registration": RegistrationParams,
}


# -- the file --------------------------------------------------------------------


def test_preset_round_trips_through_its_toml_file() -> None:
    awkward = (
        'q="hi"',
        "é=" + chr(0x7F),  # DEL is not allowed bare in a TOML string
        "smile=" + chr(0x1F600),
    )
    preset = Preset(
        stand="bench",
        plate="96-well",
        wells="B2,B11",
        saved_utc=datetime(2026, 11, 3, 10, 0, 0, tzinfo=timezone.utc),
        params={"my.plugin": awkward, "cells": ("a=b=c", "d="), "scan-plan": ()},
    )
    text = dumps_preset(preset)

    assert Preset.model_validate(tomllib.loads(text)) == preset
    assert text.startswith(
        "# A preset of smc: the choices an assay repeats. "
        "See docs/ui/plate-page.md, Presets.\n"
    )
    assert '"my.plugin"' in text  # a quoted key only where TOML needs one


def test_a_preset_with_nothing_set_is_an_empty_preset() -> None:
    text = dumps_preset(Preset())

    assert Preset.model_validate(tomllib.loads(text)) == Preset()
    assert "stand" not in text
    assert "wells" not in text


def test_a_preset_behind_a_bom_is_read_as_the_same_preset(
    profile: Profile, tmp_path: Path
) -> None:
    preset = Preset(stand="bench", params={"cells": ("detector=blobs",)})
    path = tmp_path / "notepad.toml"
    path.write_bytes(codecs.BOM_UTF8 + dumps_preset(preset).encode("utf-8"))

    assert load_preset(str(path), profile).preset == preset


def test_a_saved_preset_is_utf8_without_a_bom(profile: Profile) -> None:
    path = save_preset("weekly", Preset(wells="é1"), profile)

    raw = path.read_bytes()
    assert not raw.startswith(codecs.BOM_UTF8)
    assert raw.decode("utf-8").startswith("# A preset of smc")
    assert "é1" in raw.decode("utf-8")


@pytest.mark.parametrize("field", ["stand", "plate", "wells"])
@pytest.mark.parametrize("value", ["", "   ", "A1\nB2"])
def test_stand_plate_and_wells_are_one_non_blank_line(
    profile: Profile, tmp_path: Path, field: str, value: str
) -> None:
    path = tmp_path / "x.toml"
    # json.dumps is a TOML basic string too: "A1\nB2" is two lines in the file's value.
    path.write_text(f"{field} = {json.dumps(value)}\n", encoding="utf-8")

    with pytest.raises(PresetError) as bad:
        load_preset(str(path), profile)
    assert str(bad.value) == (
        f"{path}: {field}: {field} is one non-empty line; leave it out for none"
    )


# -- names -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["Weekly", "my preset", "a" * 65, "", "-x", "_x", "a.b"]
)
def test_check_name_refuses_upper_case_a_space_and_65_characters(name: str) -> None:
    with pytest.raises(PresetError) as bad:
        check_name(name)
    assert str(bad.value) == (
        "a preset's name is 1 to 64 lower-case letters, digits, - and _, starting "
        f"with a letter or a digit; got '{name}'"
    )


@pytest.mark.parametrize("name", ["a", "0a", "a_b-c", "a" * 64, "com10", "console"])
def test_check_name_accepts_what_the_rule_allows(name: str) -> None:
    assert check_name(name) == name


@pytest.mark.parametrize("name", ["nul", "com1", "lpt9", "con", "aux", "prn"])
def test_check_name_refuses_a_windows_device_name(name: str) -> None:
    with pytest.raises(PresetError) as bad:
        check_name(name)
    assert str(bad.value) == (
        f"'{name}' is a name Windows keeps for a device; choose another"
    )


# -- the folders -----------------------------------------------------------------


def test_preset_dirs_put_the_profile_folder_first_without_duplicates(
    profile: Profile, tmp_path: Path
) -> None:
    work = tmp_path / "work"

    # profiles/ is the profile's folder and a search path: listed once, first.
    assert preset_dirs(profile) == (
        work / "profiles" / "presets",
        tmp_path / "site" / "presets",
        work / "presets",
    )


def test_a_profile_without_a_file_uses_the_search_paths_only(
    tmp_path: Path, profile: Profile
) -> None:
    assert preset_dirs(Profile.demo()) == (
        tmp_path / "site" / "presets",
        tmp_path / "work" / "profiles" / "presets",
        tmp_path / "work" / "presets",
    )


def test_preset_dirs_fold_case_where_the_file_system_does(
    profile: Profile, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # FM-24: NTFS and the default APFS name one folder by two spellings. Windows'
    # `normcase` is `lower`; the same is used here, on a platform whose own is not.
    monkeypatch.setenv("SMC_PROFILES", str(tmp_path / "work" / "PROFILES"))
    monkeypatch.setattr(os.path, "normcase", lambda text: text.lower())

    folders = preset_dirs(profile)

    assert folders == (
        tmp_path / "work" / "profiles" / "presets",
        tmp_path / "work" / "presets",
    )


# -- loading and listing ---------------------------------------------------------


def test_load_preset_by_name_takes_the_first_folder_that_has_it(
    profile: Profile, tmp_path: Path
) -> None:
    _put(tmp_path / "site" / "presets", "weekly", 'wells = "A1"\n')
    mine = _put(tmp_path / "work" / "profiles" / "presets", "weekly", 'wells = "B2"\n')

    loaded = load_preset("weekly", profile)

    assert loaded.name == "weekly"
    assert loaded.path == mine
    assert loaded.preset == Preset(wells="B2")
    assert loaded.error is None


def test_load_preset_reads_a_path_as_given(profile: Profile, tmp_path: Path) -> None:
    # Not in any folder, and with a stem no name could have: a path is read as is.
    path = _put(tmp_path / "elsewhere", "My Preset", 'plate = "96-well"\n')

    loaded = load_preset(str(path), profile)

    assert (loaded.name, loaded.path) == ("My Preset", path)
    assert loaded.preset == Preset(plate="96-well")


def test_an_unknown_preset_names_the_folders_searched(
    profile: Profile, tmp_path: Path
) -> None:
    looked = ", ".join(str(folder) for folder in preset_dirs(profile))

    with pytest.raises(PresetError) as unknown:
        load_preset("nope", profile)

    assert str(unknown.value) == (
        f"no preset named nope; looked in {looked}; smc presets list shows the presets"
    )
    assert str(tmp_path / "site" / "presets") in looked


def test_a_name_that_is_not_a_name_is_refused_before_any_folder_is_read(
    profile: Profile,
) -> None:
    with pytest.raises(PresetError, match=r"^a preset's name is 1 to 64"):
        load_preset("Weekly", profile)


def test_an_invalid_preset_names_its_file_and_field(
    profile: Profile, tmp_path: Path
) -> None:
    folder = tmp_path / "work" / "profiles" / "presets"
    unknown_key = _put(folder, "colour", "colour = 1\n")
    no_equals = _put(folder, "pairs", '[params]\nscan-plan = ["coverage"]\n')
    twice = _put(folder, "twice", '[params]\ncells = ["a=1", "a=2"]\n')
    local_time = _put(folder, "local", "saved_utc = 2026-11-03 10:00:00\n")

    keys = "stand, plate, wells, saved_utc, params"
    with pytest.raises(PresetError) as bad_key:
        load_preset("colour", profile)
    assert (
        str(bad_key.value) == f"{unknown_key}: colour: unknown key; the keys are {keys}"
    )

    with pytest.raises(PresetError) as bad_pair:
        load_preset("pairs", profile)
    assert str(bad_pair.value) == (
        f"{no_equals}: params: scan-plan: each item of params is key=value, as "
        "smc run --param takes it; got 'coverage'"
    )

    with pytest.raises(PresetError) as given_twice:
        load_preset("twice", profile)
    assert str(given_twice.value) == f"{twice}: params: cells: 'a' given twice"

    with pytest.raises(PresetError) as naive:
        load_preset("local", profile)
    assert str(naive.value).startswith(f"{local_time}: saved_utc: ")


def test_a_file_that_is_not_toml_or_not_text_is_a_one_line_diagnosis(
    profile: Profile, tmp_path: Path
) -> None:
    broken = _put(tmp_path / "x", "broken", "wells = \n")
    binary = tmp_path / "x" / "binary.toml"
    binary.write_bytes(b"\xff\xfe\x00")
    missing = tmp_path / "x" / "missing.toml"

    with pytest.raises(PresetError) as not_toml:
        load_preset(str(broken), profile)
    assert str(not_toml.value).startswith(f"{broken}: not TOML: ")
    assert "\n" not in str(not_toml.value)

    with pytest.raises(PresetError, match=r"cannot be read: not UTF-8 text") as raw:
        load_preset(str(binary), profile)
    assert str(raw.value).startswith(f"{binary}: ")

    with pytest.raises(PresetError) as gone:
        load_preset(str(missing), profile)
    assert str(gone.value).startswith(f"{missing}: cannot be read: ")


def test_list_presets_keeps_a_broken_file_with_its_error(
    profile: Profile, tmp_path: Path
) -> None:
    mine = tmp_path / "work" / "profiles" / "presets"
    site = tmp_path / "site" / "presets"
    _put(mine, "weekly", 'wells = "B2"\n')
    _put(site, "weekly", 'wells = "A1"\n')  # shadowed: the first file of a name wins
    _put(site, "alpha", 'plate = "96-well"\n')
    bad = _put(site, "bad", "wells = \n")
    upper = _put(
        site, "Upper", 'wells = "A1"\n'
    )  # valid content, a name no one may use

    listed = list_presets(profile)

    assert [entry.name for entry in listed] == ["Upper", "alpha", "bad", "weekly"]
    by_name = {entry.name: entry for entry in listed}
    assert by_name["weekly"].preset == Preset(wells="B2")
    assert by_name["weekly"].path == mine / "weekly.toml"
    assert by_name["alpha"].preset == Preset(plate="96-well")
    assert by_name["bad"].preset is None
    assert (by_name["bad"].error or "").startswith(f"{bad}: not TOML: ")
    assert by_name["Upper"].preset is None
    assert by_name["Upper"].path == upper
    assert (by_name["Upper"].error or "").startswith("a preset's name is 1 to 64")


def test_list_presets_with_no_folder_is_empty(profile: Profile) -> None:
    assert list_presets(profile) == ()


# -- saving ----------------------------------------------------------------------


def test_save_preset_writes_the_first_folder_and_makes_it(
    profile: Profile, tmp_path: Path
) -> None:
    path = save_preset("weekly", Preset(wells="A1"), profile)

    assert path == tmp_path / "work" / "profiles" / "presets" / "weekly.toml"
    assert load_preset("weekly", profile).preset == Preset(wells="A1")


def test_save_preset_refuses_an_existing_name_and_leaves_the_file(
    profile: Profile, tmp_path: Path
) -> None:
    later = _put(tmp_path / "site" / "presets", "weekly", 'wells = "A1"\n')
    before = later.read_bytes()

    with pytest.raises(PresetExistsError) as exists:
        save_preset("weekly", Preset(wells="B2"), profile)

    # The file that exists, though it is not the one a save would write.
    assert exists.value.path == later
    assert str(exists.value) == f"preset weekly exists: {later}"
    assert later.read_bytes() == before
    assert not (tmp_path / "work" / "profiles" / "presets").exists()


def test_save_preset_overwrites_when_asked(profile: Profile, tmp_path: Path) -> None:
    first = save_preset("weekly", Preset(wells="A1"), profile)

    again = save_preset("weekly", Preset(wells="B2"), profile, overwrite=True)

    assert again == first
    assert load_preset("weekly", profile).preset == Preset(wells="B2")


def test_an_overwrite_of_a_later_folder_writes_the_first_and_shadows_it(
    profile: Profile, tmp_path: Path
) -> None:
    later = _put(tmp_path / "site" / "presets", "weekly", 'wells = "A1"\n')

    written = save_preset("weekly", Preset(wells="B2"), profile, overwrite=True)

    assert written == tmp_path / "work" / "profiles" / "presets" / "weekly.toml"
    assert load_preset("weekly", profile).path == written
    assert later.read_text(encoding="utf-8") == 'wells = "A1"\n'


def test_save_preset_refuses_a_name_that_is_not_one(profile: Profile) -> None:
    with pytest.raises(PresetError, match=r"^a preset's name is 1 to 64"):
        save_preset("My Preset", Preset(), profile)
    with pytest.raises(PresetError, match=r"a name Windows keeps for a device"):
        save_preset("nul", Preset(), profile)


def test_a_failed_rename_keeps_the_old_preset_and_names_the_new_one(
    profile: Profile, monkeypatch: pytest.MonkeyPatch
) -> None:
    # FM-38: Windows refuses to replace a file another program holds open.
    path = save_preset("weekly", Preset(wells="A1"), profile)
    before = path.read_bytes()

    def refuse(src: object, dst: object) -> None:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(files_mod.os, "replace", refuse)
    monkeypatch.setattr(files_mod.time, "sleep", lambda seconds: None)
    with pytest.raises(PresetError) as refused:
        save_preset("weekly", Preset(wells="B2"), profile, overwrite=True)

    assert path.read_bytes() == before
    kept = [p for p in path.parent.iterdir() if p.name.endswith(".smc-tmp")]
    assert len(kept) == 1
    assert str(refused.value) == (
        f"{path}: not saved: Permission denied; the new preset is in {kept[0]}"
    )
    assert 'wells = "B2"' in kept[0].read_text(encoding="utf-8")


def test_a_write_that_fails_before_the_rename_leaves_nothing(
    profile: Profile, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(path: object, flags: int, mode: int = 0o777) -> int:
        raise PermissionError(13, "Permission denied")

    # `os.open` is the process's: put it back before anything else uses it.
    with monkeypatch.context() as patched:
        patched.setattr(files_mod.os, "open", refuse)
        with pytest.raises(PresetError) as refused:
            save_preset("weekly", Preset(), profile)

    folder = preset_dirs(profile)[0]
    assert (
        str(refused.value) == f"{folder / 'weekly.toml'}: not saved: Permission denied"
    )
    assert list(folder.iterdir()) == []


def test_a_preset_exists_error_survives_pickle(tmp_path: Path) -> None:
    error = PresetExistsError("weekly", tmp_path / "weekly.toml")

    again = pickle.loads(pickle.dumps(error))

    assert isinstance(again, PresetExistsError)
    assert again.path == error.path
    assert str(again) == str(error)


# -- the check -------------------------------------------------------------------


def test_check_preset_applies_everything_on_the_stand_it_was_saved_on() -> None:
    preset = Preset(
        stand="bench",
        plate="96-well",
        wells="B2,B11",
        params={"scan-plan": ("coverage=0.4",), "cells": ("detector=blobs",)},
    )

    check = check_preset(_file(preset), MODELS, stand="bench", registered="96-well")

    assert check == PresetCheck(
        "weekly",
        "96-well",
        "B2,B11",
        {"scan-plan": ("coverage=0.4",), "cells": ("detector=blobs",)},
        (),
    )


def test_check_preset_drops_a_parameter_the_plugin_no_longer_has_and_keeps_the_rest() -> (
    None
):
    preset = Preset(
        params={"scan-plan": ("coverage=0.4", "overlap=0.2", "objective=10x")}
    )

    check = check_preset(_file(preset), MODELS, stand="bench", registered=None)

    assert check.params == {"scan-plan": ("coverage=0.4", "objective=10x")}
    assert check.mismatches == (
        "preset weekly [params.scan-plan]: overlap is not a parameter of scan-plan "
        "(keys: wells, coverage, objective)",
    )


def test_a_plugin_left_with_no_pair_is_not_a_key() -> None:
    preset = Preset(params={"scan-plan": ("overlap=0.2",), "cells": ()})

    check = check_preset(_file(preset), MODELS, stand="bench", registered=None)

    assert check.params == {}
    assert len(check.mismatches) == 1


def test_check_preset_names_an_unknown_plugin() -> None:
    preset = Preset(params={"cellz": ("detector=blobs",), "cells": ("detector=blobs",)})

    check = check_preset(_file(preset), MODELS, stand="bench", registered=None)

    assert check.params == {"cells": ("detector=blobs",)}
    assert check.mismatches == ("preset weekly [params.cellz]: no plugin named cellz",)


def test_check_preset_names_refs_and_wells_under_params() -> None:
    preset = Preset(
        params={
            "plate-registration": ("plate=96-well", "refs=[]", "walls={}"),
            "scan-plan": ("wells=A1", "coverage=0.4"),
        }
    )

    check = check_preset(_file(preset), MODELS, stand="bench", registered=None)

    assert check.params == {"scan-plan": ("coverage=0.4",)}
    assert check.mismatches == (
        "preset weekly [params.plate-registration]: plate is not kept under params: "
        "the preset's plate holds it",
        "preset weekly [params.plate-registration]: refs is not kept under params: "
        "a preset holds no positions",
        "preset weekly [params.plate-registration]: walls is not kept under params: "
        "a preset holds no positions",
        "preset weekly [params.scan-plan]: wells is not kept under params: "
        "the preset's wells hold them",
    )


def test_every_row_of_not_in_params_is_a_mismatch() -> None:
    # A rule held as data: each row, in turn (FM-43).
    for plugin, keys in NOT_IN_PARAMS.items():
        for key, reason in keys.items():
            preset = Preset(params={plugin: (f"{key}=x",)})

            check = check_preset(_file(preset), MODELS, stand="bench", registered=None)

            assert check.params == {}
            assert check.mismatches == (
                f"preset weekly [params.{plugin}]: {key} is not kept under params: "
                f"{reason}",
            )


def test_check_preset_does_not_apply_a_plate_the_run_document_did_not_register() -> (
    None
):
    preset = Preset(
        plate="96-well", wells="B2,B11", params={"cells": ("detector=blobs",)}
    )

    check = check_preset(_file(preset), MODELS, stand="bench", registered="384-well")

    assert (check.plate, check.wells) == (None, None)
    assert check.params == {"cells": ("detector=blobs",)}
    assert check.mismatches == (
        "preset weekly is for a 96-well plate; this run document's plate is "
        "registered as a 384-well: its plate and wells are not applied",
    )


def test_a_plate_the_document_registered_or_none_at_all_is_applied() -> None:
    preset = Preset(plate="96-well", wells="B2")

    for registered in ("96-well", None):
        check = check_preset(
            _file(preset), MODELS, stand="bench", registered=registered
        )

        assert (check.plate, check.wells) == ("96-well", "B2")
        assert check.mismatches == ()


def test_a_plate_useq_does_not_know_is_not_applied_with_its_wells() -> None:
    preset = Preset(plate="96-wel", wells="B2")

    check = check_preset(_file(preset), MODELS, stand="bench", registered="96-well")

    assert (check.plate, check.wells) == (None, None)
    assert check.mismatches == ("preset weekly: 96-wel is not a plate useq knows",)


def test_wells_without_a_plate_are_applied() -> None:
    check = check_preset(
        _file(Preset(wells="B2")), MODELS, stand="bench", registered="384-well"
    )

    assert (check.plate, check.wells) == (None, "B2")


def test_check_preset_names_another_stand_and_still_applies_everything() -> None:
    preset = Preset(
        stand="nikon-ti2",
        plate="96-well",
        wells="B2",
        params={"cells": ("detector=blobs",)},
    )

    check = check_preset(_file(preset), MODELS, stand="demo", registered=None)

    assert check.mismatches == (
        "preset weekly was saved on nikon-ti2; this stand is demo",
    )
    assert (check.plate, check.wells) == ("96-well", "B2")
    assert check.params == {"cells": ("detector=blobs",)}


def test_check_preset_only_checks_the_plugin_it_is_given() -> None:
    preset = Preset(
        params={
            "scan-plan": ("coverage=0.4", "overlap=1"),
            "cells": ("detector=blobs",),
            "other": ("a=1",),
        }
    )
    only_scan_plan = {"scan-plan": ScanPlanParams}

    check = check_preset(
        _file(preset), only_scan_plan, stand="bench", registered=None, only="scan-plan"
    )

    assert check.params == {"scan-plan": ("coverage=0.4",)}
    # `cells` and `other` are not mismatches: this run imports no other plugin.
    assert check.mismatches == (
        "preset weekly [params.scan-plan]: overlap is not a parameter of scan-plan "
        "(keys: wells, coverage, objective)",
    )


def test_check_preset_refuses_a_file_that_holds_no_preset() -> None:
    broken = PresetFile("bad", Path("bad.toml"), None, "bad.toml: not TOML: x")

    with pytest.raises(PresetError) as refused:
        check_preset(broken, MODELS, stand="bench", registered=None)
    assert str(refused.value) == "bad.toml: bad.toml: not TOML: x"


def _plate_record(
    plugin: str,
    minute: int,
    result: dict[str, object] | None,
    *,
    status: StageStatus = StageStatus.done,
    provides: str = "plate",
) -> StageRecord:
    started = T0 + timedelta(minutes=minute)
    return StageRecord(
        plugin=plugin,
        provides=provides,
        attempt=f"{plugin}-{minute}",
        status=status,
        started_utc=started,
        finished_utc=started + timedelta(seconds=30),
        profile="bench",
        dry_run=False,
        smc_version="0.0.1",
        params={},
        result=result,
    )


def _document(*records: StageRecord) -> RunDocument:
    return RunDocument(
        created_utc=T0, stages={record.plugin: record for record in records}
    )


def test_registered_plate_is_the_latest_done_plate_record() -> None:
    document = _document(
        _plate_record("plate-registration", 0, {"plate": "96-well"}),
        _plate_record("plate-finding", 10, {"plate": "384-well"}),
        # Later, but it did not finish: not a plate.
        _plate_record("plate-other", 20, None, status=StageStatus.failed),
        # Done and later, but it fills another slot.
        _plate_record("scan-plan", 30, {"plate": "6-well"}, provides="scan-plan"),
    )

    assert registered_plate(document) == "384-well"


def test_registered_plate_is_none_without_a_done_record_or_a_plate_name() -> None:
    assert registered_plate(_document()) is None
    assert (
        registered_plate(
            _document(
                _plate_record("plate-registration", 0, None, status=StageStatus.stopped)
            )
        )
        is None
    )
    # The latest decides, even when it names no plate: not the older one.
    assert (
        registered_plate(
            _document(
                _plate_record("plate-registration", 0, {"plate": "96-well"}),
                _plate_record("plate-finding", 10, {"plate": 96}),
            )
        )
        is None
    )


# -- the command's pairs -------------------------------------------------------


def _applied() -> PresetCheck:
    return PresetCheck(
        "weekly",
        "96-well",
        "B2,B11",
        {
            "scan-plan": ("coverage=0.4", "objective=10x"),
            "cells": ("detector=blobs",),
        },
        (),
    )


def test_command_pairs_let_param_win_over_the_preset() -> None:
    check = _applied()

    assert command_pairs(check, "scan-plan", []) == [
        "wells=B2,B11",
        "coverage=0.4",
        "objective=10x",
    ]
    assert command_pairs(check, "scan-plan", ["coverage=0.5"]) == [
        "wells=B2,B11",
        "objective=10x",
        "coverage=0.5",
    ]
    # The preset's wells are one of the pairs an override can replace.
    assert command_pairs(check, "scan-plan", ["wells=A1"]) == [
        "coverage=0.4",
        "objective=10x",
        "wells=A1",
    ]


def test_command_pairs_hold_the_plate_for_plate_registration_and_nothing_else() -> None:
    check = _applied()

    assert command_pairs(check, "plate-registration", []) == ["plate=96-well"]
    assert command_pairs(check, "cells", []) == ["detector=blobs"]
    assert command_pairs(check, "density", []) == []


def test_command_pairs_leave_out_a_plate_that_was_not_applied() -> None:
    check = PresetCheck("weekly", None, None, {"cells": ("detector=blobs",)}, ())

    assert command_pairs(check, "plate-registration", []) == []
    assert command_pairs(check, "scan-plan", ["coverage=1"]) == ["coverage=1"]


def test_command_pairs_pass_a_malformed_override_on() -> None:
    # `parse_params` names it; it is not this function's to refuse, nor to drop
    # a pair of the preset because of it.
    check = _applied()

    assert command_pairs(check, "cells", ["oops"]) == ["detector=blobs", "oops"]
    assert command_pairs(check, "cells", ["=1", "other"]) == [
        "detector=blobs",
        "=1",
        "other",
    ]
    # The key of an item is its text before the first `=`, the whole text without one.
    assert command_pairs(check, "cells", ["detector"]) == ["detector"]
