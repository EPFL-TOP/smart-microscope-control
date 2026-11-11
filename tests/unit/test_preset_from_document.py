"""A preset from a run document's recorded parameters (m5 §10.9, #324).

The records are built the way the runner writes them (the whole ``Params`` dumped
as JSON), with the real plugins' ``Params``: the round trip of every built-in
plugin is the test that a new plugin or a new field cannot slip past
(``test_the_round_trip_cases_cover_every_plugin_and_every_field...``).
Every refusal and every note is asserted by its whole text (FM-45).
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic import model_validator
from pydantic_core import to_jsonable_python

from smc.plugins import discover
from smc.plugins.api import PluginParams
from smc.plugins.params import format_param, parse_params
from smc.plugins.presets import (
    NOT_IN_PARAMS,
    Preset,
    PresetError,
    PresetFile,
    PresetFromRun,
    check_preset,
    command_pairs,
    dumps_preset,
    preset_from_document,
)
from smc.run import RunDocument, StageRecord, StageStatus

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on the 3.10 CI job
    import tomli as tomllib

T0 = datetime(2026, 11, 5, 9, 0, tzinfo=timezone.utc)
SAVED = datetime(2026, 11, 5, 10, 0, tzinfo=timezone.utc)
STAND = "demo-sample"

# Only this distribution's plugins: a lab's own plugin, installed in the same venv
# through the entry-point group, is not a case this module can write.
_REGISTRY = {
    name: entry
    for name, entry in discover().plugins.items()
    if entry.distribution == "smart-microscope-control"
}
MODELS: dict[str, Any] = {
    name: entry.plugin.Params for name, entry in _REGISTRY.items()
}
PROVIDES = {name: entry.plugin.provides for name, entry in _REGISTRY.items()}

#: One lone surrogate, as an objective label read from a ``.cfg`` saved in cp1252
#: leaves in a run document (FM-05). Built, not written, so no editor decodes it.
SURROGATE = chr(0xD800)


def _recorded(plugin: str, **given: Any) -> dict[str, Any]:
    """The ``params`` a run of ``plugin`` records: the whole model, as JSON."""
    dumped: dict[str, Any] = (
        MODELS[plugin].model_validate(given).model_dump(mode="json")
    )
    return dumped


def _record(
    plugin: str,
    params: dict[str, Any],
    *,
    status: StageStatus = StageStatus.done,
    profile: str = STAND,
    dry_run: bool = False,
    result: dict[str, Any] | None = None,
    minute: int = 0,
) -> StageRecord:
    started = T0 + timedelta(minutes=minute)
    return StageRecord(
        plugin=plugin,
        provides=PROVIDES.get(plugin, plugin),
        attempt=f"{plugin}-{minute}",
        status=status,
        started_utc=started,
        finished_utc=None
        if status is StageStatus.running
        else started + timedelta(seconds=30),
        profile=profile,
        dry_run=dry_run,
        smc_version="0.0.1",
        params=params,
        result=result,
    )


def _document(*records: StageRecord) -> RunDocument:
    return RunDocument(
        created_utc=T0, stages={record.plugin: record for record in records}
    )


def _preset_of(
    *records: StageRecord, models: dict[str, Any] | None = None
) -> PresetFromRun:
    return preset_from_document(
        _document(*records),
        MODELS if models is None else models,
        stand=STAND,
        saved_utc=SAVED,
    )


def _registered(plate: str, **given: Any) -> StageRecord:
    return _record(
        "plate-registration",
        _recorded("plate-registration", plate=plate, **given),
        result={"plate": plate},
    )


def _plan(**given: Any) -> StageRecord:
    return _record("scan-plan", _recorded("scan-plan", **given))


def _file(preset: Preset, name: str = "weekly") -> PresetFile:
    return PresetFile(name, Path(f"{name}.toml"), preset)


# -- which records, which pairs ----------------------------------------------------


def test_a_preset_holds_the_plate_the_wells_and_the_non_default_pairs_of_the_done_records() -> (
    None
):
    refs = [
        {"well": "A1", "x_um": -49500.0, "y_um": 31500.0},
        {"well": "A12", "x_um": 49500.0, "y_um": 31500.0},
    ]

    made = _preset_of(
        _registered("96-well", refs=refs),
        _plan(wells="B2,B11", coverage=0.4),
        _record("scan", _recorded("scan", autofocus="off")),
        _record("cells", _recorded("cells", detector="blobs")),
    )

    assert made.preset == Preset(
        stand=STAND,
        plate="96-well",
        wells="B2,B11",
        saved_utc=SAVED,
        params={
            "cells": ("detector=blobs",),
            "scan": ("autofocus=off",),
            "scan-plan": ("coverage=0.4",),
        },
    )
    # The plugins are in the order of their names, and plate-registration is no key.
    assert list(made.preset.params) == ["cells", "scan", "scan-plan"]
    assert made.notes == ()


def test_a_value_equal_to_its_default_is_not_saved() -> None:
    # Recorded explicitly, as every run records the whole model: the preset follows the
    # plugin's default, as `[suggest]` does.
    made = _preset_of(_plan(wells="B2", coverage=1.0, overlap=0.1))

    assert made.preset.wells == "B2"
    assert made.preset.params == {}


def test_a_required_field_is_always_saved() -> None:
    made = _preset_of(_record("goto-well", _recorded("goto-well", well="A1")))

    assert made.preset.params == {"goto-well": ("well=A1",)}


def test_positions_are_never_saved() -> None:
    walls = [{"well": "A1", "side": "-X", "x_um": 1.0, "y_um": 2.0}]
    refs = [{"well": "A1", "x_um": 1.0, "y_um": 2.0}]

    made = _preset_of(
        _registered("24-well", refs=refs, walls=walls),
        _record(
            "plate-finding",
            _recorded(
                "plate-finding", plate="24-well", centre_x_um=1.0, centre_y_um=2.0
            ),
            result={"plate": "24-well"},
            minute=5,
        ),
        _plan(wells="B2"),
    )

    pairs = [pair for items in made.preset.params.values() for pair in items]
    assert not [p for p in pairs if p.startswith(("refs=", "walls=", "centre_"))]
    assert "plate-registration" not in made.preset.params
    # Its own `plate` is a pair like any other (it differs from the default); the
    # centre it was given is not.
    assert made.preset.params["plate-finding"] == ("plate=24-well",)
    assert made.preset.plate == "24-well"


def test_plate_finding_centre_is_not_kept_under_params_for_a_hand_written_preset() -> (
    None
):
    preset = Preset(params={"plate-finding": ("centre_x_um=1.0",)})

    check = check_preset(_file(preset), MODELS, stand=STAND, registered=None)

    assert check.params == {}
    assert check.mismatches == (
        "preset weekly [params.plate-finding]: centre_x_um is not kept under params: "
        "a preset holds no positions",
    )


def test_plate_walls_plate_and_refs_are_never_saved_and_not_kept_for_a_hand_written_preset() -> (
    None
):
    # plate-walls (#328) landed after this module's first cases: its `refs` are the
    # stage positions of wells centred by eye, as plate-registration's are, and its
    # `plate` is the preset's own, which the page fills too (#371).
    refs = [
        {"well": "A1", "x_um": 1.0, "y_um": 2.0},
        {"well": "B2", "x_um": 3.0, "y_um": 4.0},
    ]
    recorded = _recorded("plate-walls", plate="24-well", refs=refs, search_um=500.0)

    made = _preset_of(_record("plate-walls", recorded, result={"plate": "24-well"}))

    assert made.preset.params == {"plate-walls": ("search_um=500.0",)}
    assert made.preset.plate == "24-well"
    assert made.notes == ()

    preset = Preset(params={"plate-walls": ("plate=24-well", "refs=[]")})
    check = check_preset(_file(preset), MODELS, stand=STAND, registered=None)
    assert check.params == {}
    assert check.mismatches == (
        "preset weekly [params.plate-walls]: plate is not kept under params: "
        "the preset's plate holds it",
        "preset weekly [params.plate-walls]: refs is not kept under params: "
        "a preset holds no positions",
    )


def test_a_turn_confirmation_is_never_saved_and_not_kept_for_a_hand_written_preset() -> (
    None
):
    # objective-offsets (#380) turns the turret over a loaded plate: "the stage is clear"
    # is true of one run, and a preset that kept it would confirm the next session's
    # turns for the operator (FM-14).
    recorded = _recorded(
        "objective-offsets",
        targets=["Nikon 40X Plan Fluor ELWD"],
        calibrations=["cal-40x.json"],
        confirm_turns=True,
        search_um=500.0,
    )

    made = _preset_of(_record("objective-offsets", recorded, result={}))

    pairs = made.preset.params["objective-offsets"]
    assert pairs == (
        'targets=["Nikon 40X Plan Fluor ELWD"]',
        'calibrations=["cal-40x.json"]',
        "search_um=500.0",
    )
    assert made.notes == ()

    preset = Preset(params={"objective-offsets": ("confirm_turns=true",)})
    check = check_preset(_file(preset), MODELS, stand=STAND, registered=None)
    assert check.params == {}
    assert check.mismatches == (
        "preset weekly [params.objective-offsets]: confirm_turns is not kept under "
        "params: the stage is clear only for the run that confirmed it: every run "
        "confirms it again",
    )


@pytest.mark.parametrize(
    "status", [StageStatus.stopped, StageStatus.failed, StageStatus.running]
)
def test_a_stage_that_is_not_done_is_not_saved(status: StageStatus) -> None:
    made = _preset_of(
        _plan(wells="B2"),
        _record("cells", _recorded("cells", detector="blobs"), status=status),
    )

    assert "cells" not in made.preset.params


def test_a_dry_run_record_is_saved() -> None:
    made = _preset_of(
        _record("cells", _recorded("cells", detector="blobs"), dry_run=True)
    )

    assert made.preset.params == {"cells": ("detector=blobs",)}


def test_a_document_with_no_finished_stage_is_refused() -> None:
    failed = _record("cells", _recorded("cells"), status=StageStatus.failed)

    for records in ([], [failed]):
        with pytest.raises(PresetError) as refused:
            _preset_of(*records)
        assert str(refused.value) == (
            "the run document has no finished stage; there is nothing to save"
        )


def test_a_document_whose_stages_used_only_defaults_is_refused_and_names_them() -> None:
    with pytest.raises(PresetError) as refused:
        _preset_of(
            _record("report", _recorded("report")),
            _record("cells", _recorded("cells")),
        )

    assert str(refused.value) == (
        "nothing to save: cells, report used only default values"
    )


def test_a_document_that_saves_nothing_but_refused_a_plugin_says_so() -> None:
    # Not "used only default values": the objective was not default, it did not read back.
    with pytest.raises(PresetError) as refused:
        _preset_of(
            _record("route", _recorded("route", objective="null")),
            _record("report", _recorded("report")),
        )

    assert str(refused.value) == (
        "nothing to save: report used only default values; "
        "route: objective would read back as another value; not saved"
    )


# -- the round trip of every built-in plugin ---------------------------------------

#: Per plugin, recorded values that differ from every default (a field per case at
#: least), and that its ``Params`` accepts. ``cells`` has two cases because
#: ``detector="blobs"`` refuses a ``model_path``, a ``device`` and the model's
#: pre-processing (``model_um_per_px``, the percentiles, ``checked_um_per_px``).
CASES: dict[str, list[dict[str, Any]]] = {
    "camera-calibration": [
        {"step_um": 12.5, "settle_ms": 30.0, "from_run_doc": "C:\\runs\\a b\\run.json"}
    ],
    "cells": [
        {
            "detector": "blobs",
            "score_threshold": 0.25,
            "max_overlap": 0.55,
            "merge_radius_um": 7.5,
        },
        {
            "detector": "model",
            "model_path": "/x/w.pt",
            "device": "cpu",
            "model_um_per_px": 0.49,
            "low_percentile": 1.0,
            "high_percentile": 99.0,
            "checked_um_per_px": 0.2756,
        },
    ],
    "density": [{"area_grid_um": 12.5}],
    "focus-on-cells": [
        {
            "range_um": 2.5,
            "step_um": 0.5,
            "score": "gradient-energy",
            "relock": False,
        }
    ],
    "goto-well": [{"well": "B2", "side": "-X"}],
    "nis-export": [{"out": "/tmp/x y/out.xml", "overwrite": True}],
    "objective-offsets": [
        {
            "targets": ["Nikon 40X Plan Fluor ELWD", "Nikon 20X Plan Fluor ELWD"],
            "calibrations": ["cal-40x.json", "C:\\runs\\a b\\cal-20x.json"],
            "confirm_turns": True,
            "wells": ["A1", "H12"],
            "search_um": 500.0,
            "max_spread_um": 5.0,
            "settle_ms": 30.0,
            "save_frames": False,
        }
    ],
    "plate-finding": [
        {
            "plate": "24-well",
            "plates": ["24-well", "96-well"],
            "centre_x_um": 100.5,
            "centre_y_um": -20.0,
            "step_um": 50.0,
            "margin_pitches": 2.0,
            "jitter": 0.25,
            "seed": 7,
            "statistic": "std",
            "max_failed_in_row": 5,
            "orientation": "turned-180",
        }
    ],
    "plate-registration": [
        {
            "plate": "24-well",
            "refs": [{"well": "A1", "x_um": 1.0, "y_um": 2.0}],
            "walls": [{"well": "A1", "side": "-X", "x_um": 1.0, "y_um": 2.0}],
            "max_rotation_deg": 5.0,
            "orientation": "turned-180",
            "objective": "Nikon 10X S Fluor",
        }
    ],
    "plate-walls": [
        {
            "plate": "24-well",
            "refs": [
                {"well": "A1", "x_um": 1.0, "y_um": 2.0},
                {"well": "B2", "x_um": 3.0, "y_um": 4.0},
            ],
            "search_um": 500.0,
            "max_residual_um": 50.0,
            "max_rotation_deg": 5.0,
            "orientation": "turned-180",
            "settle_ms": 30.0,
            "save_frames": False,
        }
    ],
    "report": [{"max_cells_drawn_per_well": 500}],
    "route": [
        {
            "labels": ["a", "b"],
            "fields_per_well": 4,
            "min_cells": 2,
            "exclude_centre_um": 10.0,
            "wall_margin_um": 5.0,
            "objective": "Nikon 10X S Fluor",
            "candidate_step_fraction": 0.5,
        }
    ],
    "scan": [
        {
            "autofocus": "off",
            "max_failed_in_row": 2,
            "max_unlocked_in_row": 3,
            "landing_tolerance": 0.1,
            "cell_focus": "per-well",
            "focus_range_um": 2.5,
            "focus_step_um": 0.5,
            "focus_score": "normalized-variance",
        }
    ],
    "scan-plan": [
        {
            "wells": "B2,B11",
            "serpentine": False,
            "objective": "Nikon 10X S Fluor",
            "overlap": 0.2,
            "coverage": 0.4,
            "rows": 3,
            "columns": 2,
            "clip_to_well": False,
            "pixel_size_um": 0.65,
        },
        # `pixel_size_um` and `calibration` are refused together, so a case each; a
        # path with a backslash and a space is the one a reader may take for an escape (FM-190).
        {"wells": "B2", "calibration": "C:\\runs\\a b\\cal-40x.json"},
    ],
    "stage-report": [{"samples": 5, "interval_ms": 100.0, "include_z": False}],
    "timing": [
        {
            "repeats": 5,
            "distances_um": [100.0, 5000.0],
            "with_autofocus": False,
            "channels": ["DAPI", "GFP"],
        }
    ],
}

ROUND_TRIPS = [
    pytest.param(plugin, given, id=f"{plugin}-{index}")
    for plugin, cases in CASES.items()
    for index, given in enumerate(cases)
]


@pytest.mark.parametrize(("plugin", "given"), ROUND_TRIPS)
def test_every_builtin_plugins_recorded_params_come_back_through_a_preset(
    plugin: str, given: dict[str, Any]
) -> None:
    recorded = _recorded(plugin, **given)
    plate = recorded["plate"] if PROVIDES[plugin] == "plate" else "96-well"
    records = [_record(plugin, recorded, result={"plate": plate})]
    if PROVIDES[plugin] != "plate":
        records.append(_registered(plate))

    made = _preset_of(*records)

    assert made.notes == ()
    # Through the file: what `smc presets save` writes is what `smc run --preset` reads.
    reread = Preset.model_validate(tomllib.loads(dumps_preset(made.preset)))
    assert reread == made.preset
    check = check_preset(
        _file(reread), MODELS, stand=STAND, registered=None, only=plugin
    )
    assert check.mismatches == ()
    # The operator gives the positions again; the preset holds the plate and the wells.
    fields = MODELS[plugin].model_fields
    positions = [
        f"{key}={format_param(fields[key], recorded[key])}"
        for key in NOT_IN_PARAMS.get(plugin, {})
        if key in fields and key not in ("plate", "wells")
    ]
    again = parse_params(MODELS[plugin], command_pairs(check, plugin, positions))
    assert again.model_dump(mode="json") == recorded


def test_the_round_trip_cases_cover_every_plugin_and_every_field_with_a_non_default_value() -> (
    None
):
    # A plugin or a field added later fails here until a case is written for it.
    assert set(MODELS) == set(CASES)
    for plugin, cases in CASES.items():
        for key, field in MODELS[plugin].model_fields.items():
            default = (
                None
                if field.is_required()
                else to_jsonable_python(field.get_default(call_default_factory=True))
            )
            seen = [_recorded(plugin, **given)[key] for given in cases]
            assert field.is_required() or any(value != default for value in seen), (
                f"{plugin}.{key} is at its default in every case"
            )


# -- what is left out, and says so -------------------------------------------------


def test_a_string_null_in_an_optional_field_saves_none_of_the_plugins_pairs_and_names_the_key() -> (
    None
):
    made = _preset_of(_plan(wells="B2", objective="null", coverage=0.4))

    assert "scan-plan" not in made.preset.params
    assert made.notes == (
        "scan-plan: objective would read back as another value; not saved",
    )
    # The wells are the preset's own field and read back as they are.
    assert made.preset.wells == "B2"


def test_a_lone_surrogate_in_a_value_saves_none_of_the_plugins_pairs_and_names_the_key() -> (
    None
):
    made = _preset_of(_plan(wells="B2", objective=SURROGATE, coverage=0.4))

    assert "scan-plan" not in made.preset.params
    assert made.notes == (
        "scan-plan: objective holds a character that is not text; not saved",
    )
    assert dumps_preset(made.preset).encode("utf-8")


def test_a_wells_text_that_does_not_encode_is_left_out_and_named() -> None:
    made = _preset_of(_plan(wells="B2" + SURROGATE, coverage=0.4))

    assert made.preset.wells is None
    assert made.preset.params == {"scan-plan": ("coverage=0.4",)}
    assert made.notes == (
        "scan-plan: wells holds a character that is not text; not saved",
    )
    assert dumps_preset(made.preset).encode("utf-8")


def test_a_wells_text_of_two_lines_is_refused_as_nothing_saved() -> None:
    with pytest.raises(PresetError) as refused:
        _preset_of(_plan(wells="B2\nB3", coverage=0.4))

    assert str(refused.value) == (
        "nothing saved: wells is one non-empty line; leave it out for none"
    )


def test_recorded_params_this_smc_refuses_are_noted_and_the_other_plugins_are_saved() -> (
    None
):
    over = {**_recorded("scan-plan", wells="B2"), "coverage": 7.0}
    unknown = {**_recorded("scan-plan", wells="B2"), "gone": 1}
    cells = _record("cells", _recorded("cells", detector="blobs"))

    above = _preset_of(_record("scan-plan", over), cells)
    gone = _preset_of(_record("scan-plan", unknown), cells)

    assert above.preset.params == {"cells": ("detector=blobs",)}
    assert above.notes == (
        "scan-plan: its recorded parameters are not valid for this smc "
        "(coverage: Input should be less than or equal to 1.5); not saved",
    )
    assert gone.preset.params == {"cells": ("detector=blobs",)}
    assert gone.notes == (
        "scan-plan: its recorded parameters are not valid for this smc "
        "(gone: Extra inputs are not permitted); not saved",
    )


def test_a_plugin_the_registry_does_not_hold_is_noted() -> None:
    models = {name: model for name, model in MODELS.items() if name != "cells"}

    made = _preset_of(
        _plan(wells="B2", coverage=0.4),
        _record("cells", _recorded("cells", detector="blobs")),
        models=models,
    )

    assert made.preset.params == {"scan-plan": ("coverage=0.4",)}
    assert made.notes == (
        "cells: no plugin of that name is loaded; its parameters are not saved",
    )


def test_a_stage_run_on_another_stand_is_noted_and_saved() -> None:
    made = _preset_of(
        _record("cells", _recorded("cells", detector="blobs"), profile="nikon-ti2")
    )

    assert made.preset.params == {"cells": ("detector=blobs",)}
    assert made.notes == ("cells was run on nikon-ti2, not on demo-sample",)


def test_a_cross_field_rule_is_read_back_on_the_whole_set_not_pair_by_pair() -> None:
    # `rows` without `columns` is refused by scan-plan: a subset of the pairs may be
    # refused, so the read-back takes them together and the preset is saved.
    made = _preset_of(_plan(wells="B2", rows=3, columns=2))

    assert made.preset.params == {"scan-plan": ("rows=3", "columns=2")}
    assert made.notes == ()


class _NeedsALabel(PluginParams):
    """A plugin whose ``count`` is refused without a ``label``: a rule between two fields."""

    label: str | None = None
    count: int

    @model_validator(mode="after")
    def _count_needs_a_label(self) -> _NeedsALabel:
        if self.count and self.label is None:
            raise ValueError("count needs a label")
        return self


def test_pairs_that_the_plugin_refuses_as_a_whole_are_not_saved_and_the_note_says_why() -> (
    None
):
    # `label="null"` is valid in the record, but its text reads back as None, and then
    # the whole set is refused: not one of the plugin's pairs is saved.
    recorded = {"label": "null", "count": 2}

    made = _preset_of(
        _record("needs-a-label", recorded),
        _plan(wells="B2"),
        models={**MODELS, "needs-a-label": _NeedsALabel},
    )

    assert made.preset.params == {}
    assert made.preset.wells == "B2"
    assert made.notes == (
        "needs-a-label: its pairs do not read back "
        "(parameters: Value error, count needs a label); not saved",
    )


def test_a_plate_name_that_does_not_encode_is_left_out_and_named() -> None:
    made = _preset_of(_registered("96-well" + SURROGATE), _plan(wells="B2"))

    assert made.preset.plate is None
    assert made.preset.wells == "B2"
    assert made.notes == ("plate holds a character that is not text; not saved",)
    assert dumps_preset(made.preset).encode("utf-8")


def test_a_record_is_validated_again_before_its_values_are_used() -> None:
    # FM-125: a record written by an older smc lacks the keys added since, and one edited
    # by hand holds `"0.4"` for a float. Neither is a reason to fail or to drop the pair.
    older = _record("scan-plan", {"wells": "B2", "coverage": 0.4})
    typed = _record(
        "scan-plan", {**_recorded("scan-plan", wells="B2"), "coverage": "0.4"}
    )

    for record in (older, typed):
        made = _preset_of(record)

        assert made.preset.params == {"scan-plan": ("coverage=0.4",)}
        assert made.notes == ()
