"""``scan-plan`` on the demo devices: which pixel size and matrix a plan rests on (m3 §7.3, #21).

The plugin reads the camera's frame and pixel size and the turret, and moves
nothing. What these tests are about is where the numbers come from and what the
result says about them: the demo's camera reports 0.5 µm on a 300 x 512 frame
(``demo_microscope_with_sample``), while its turret's profile says 0.65 µm for
the 10x in place and 0.1625 µm for the 40x, so the camera's size and the turret's
cannot agree by accident, and a plan that took the wrong one is seen. The frame
is not square (FM-73). Every ``assumed`` line and every reason is asserted whole
(FM-45), because a word such as ``objective`` is also in the objective's name.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from smc.hardware import Microscope
from smc.hardware.capabilities import Autofocus, Camera, ObjectiveTurret
from smc.plugins import PluginRunError, load, run_plugin
from smc.plugins.geometry import Matrix2, pixel_matrix
from smc.plugins.scan_plan.algorithm import estimate_s
from smc.plugins.scan_plan.plugin import ScanPlanParams, ScanPlanPlugin
from smc.plugins.slots import (
    CameraCalibration,
    PlateResult,
    ScanPlanResult,
    TimingsResult,
)
from smc.run import RunDocument, StageRecord, StageStatus, write_stage
from smc.testing.fixtures import SampleMicroscopeFactory
from smc.testing.synthetic import PlateSample, SampleCamera

pytestmark = pytest.mark.demo

SHAPE = (300, 512)  # (height, width): not square, on purpose (FM-73)
PIXEL_UM = 0.5  # what the camera reports; the profile says 0.65 for the 10x in place
IN_PLACE = "Nikon 10X S Fluor"
FORTY = "Nikon 40X Plan Fluor ELWD"
#: The labels of the demo's turret, in position order.
LABELS = (
    "Nikon 40X Plan Fluor ELWD, Nikon 10X S Fluor, Objective-2, "
    "Nikon 20X Plan Fluor ELWD, Objective-4, Objective-5"
)
#: A small explicit grid, for the tests that are not about the grid.
SMALL = {"rows": 2, "columns": 2}
NO_PIXEL_SIZE = (
    "no pixel size for {objective}: run camera-calibration with it in place, "
    "or add it to the profile's [camera] pixel_size_um"
)
NO_CALIBRATION = (
    "no camera calibration for {objective}: rotation 0 and no mirror assumed"
)


def _matrix(pixel_size_um: float, rotation_deg: float, mirrored: bool) -> Matrix2:
    """``ps * R(theta) * diag(s, -1)`` of m3 §3.1, written here and not taken from the code."""
    angle = math.radians(rotation_deg)
    cos, sin = math.cos(angle), math.sin(angle)
    sign = -1.0 if mirrored else 1.0
    return (
        (pixel_size_um * cos * sign, pixel_size_um * sin),
        (pixel_size_um * sin * sign, -pixel_size_um * cos),
    )


def _seed(
    path: Path,
    plugin: str,
    provides: str,
    result: PlateResult | CameraCalibration | TimingsResult,
    *,
    profile: str = "demo",
    dry_run: bool = False,
) -> None:
    """A stage record as an earlier run of ``plugin`` would have left it, ``done``."""
    then = datetime(2026, 10, 27, 9, 30, tzinfo=timezone.utc)
    write_stage(
        path,
        StageRecord(
            plugin=plugin,
            provides=provides,
            attempt="a1",
            status=StageStatus.done,
            started_utc=then,
            finished_utc=then,
            profile=profile,
            dry_run=dry_run,
            smc_version="0",
            params={},
            result=result.model_dump(mode="json"),
        ),
    )


def _write_plate(path: Path, *, dry_run: bool = False) -> PlateResult:
    plate = PlateResult(
        plate="96-well",
        a1_center_xy_um=(1000.0, -500.0),
        rotation_deg=0.0,
        method="seed",
        residual_um=None,
        scale_error=None,
        n_refs=0,
    )
    _seed(path, "plate-registration", "plate", plate, dry_run=dry_run)
    return plate


def _write_calibration(
    path: Path,
    *,
    objective: str | None = IN_PLACE,
    shape: tuple[int, int] = SHAPE,
    pixel_size_um: float = 0.4,
    rotation_deg: float = 3.5,
    mirrored: bool = False,
) -> CameraCalibration:
    calibration = CameraCalibration(
        objective=objective,
        image_shape_px=shape,
        matrix_um_per_px=_matrix(pixel_size_um, rotation_deg, mirrored),
        pixel_size_um=pixel_size_um,
        rotation_deg=rotation_deg,
        mirrored=mirrored,
        residual_um=0.01,
        response=0.97,
        return_error_um=0.4,
        step_um=48.0,
    )
    _seed(path, "camera-calibration", "camera-calibration", calibration)
    return calibration


def _write_timings(path: Path) -> TimingsResult:
    timings = TimingsResult(
        move_overhead_ms=100.0,
        move_ms_per_mm=50.0,
        refocus_ms=400.0,
        snap_ms=20.0,
        samples=5,
    )
    _seed(path, "timing", "timings", timings)
    return timings


def _stand(
    factory: SampleMicroscopeFactory,
    tmp_path: Path,
    *,
    pixel_size_um: float = PIXEL_UM,
    shape: tuple[int, int] = SHAPE,
) -> Microscope:
    """The demo stand with a 300 x 512 camera, and a registered 96-well plate in the run document."""
    _write_plate(tmp_path / "run.json")
    return factory(shape=shape, pixel_size_um=pixel_size_um)


def _run(microscope: Microscope, tmp_path: Path, **params: object) -> StageRecord:
    return run_plugin(
        ScanPlanPlugin(),
        microscope,
        ScanPlanParams(**params),  # type: ignore[arg-type]
        tmp_path / "run.json",
    )


def _result(record: StageRecord) -> ScanPlanResult:
    assert record.status is StageStatus.done
    assert record.result is not None
    return ScanPlanResult.model_validate(record.result)


def _stored(tmp_path: Path) -> StageRecord:
    return RunDocument.load(tmp_path / "run.json").stages["scan-plan"]


def _without(
    microscope: Microscope, monkeypatch: pytest.MonkeyPatch, capability: type
) -> None:
    """The stand as if no device filled this capability's role."""
    real_get = microscope.get
    monkeypatch.setattr(
        microscope,
        "get",
        lambda wanted: None if wanted is capability else real_get(wanted),
    )


# --------------------------------------------------------------- registration


def test_the_entry_point_loads_scan_plan() -> None:
    entry = load("scan-plan")

    assert entry.plugin.provides == "scan-plan"
    assert entry.plugin.requires == (Camera,)
    assert entry.plugin.reads == ("plate",)
    assert entry.plugin.reads_optional == ("camera-calibration", "timings")
    assert entry.target == "smc.plugins.scan_plan.plugin:ScanPlanPlugin"


# ------------------------------------------------------------ where the size is from


def test_the_objective_in_place_takes_the_cameras_pixel_size(
    demo_microscope_with_sample: SampleMicroscopeFactory, tmp_path: Path
) -> None:
    microscope = _stand(demo_microscope_with_sample, tmp_path)

    record = _run(microscope, tmp_path, wells="A1", **SMALL)

    result = _result(record)
    assert result.objective == IN_PLACE
    # The camera's own size, 0.5, and not the profile's 0.65 for this objective.
    assert result.pixel_size_um == PIXEL_UM
    assert result.matrix_um_per_px == pixel_matrix(PIXEL_UM)
    assert result.matrix_source == "profile"
    assert result.image_shape_px == SHAPE
    # Width from the 512 columns, height from the 300 rows.
    assert result.fov_um == (256.0, 150.0)
    assert result.assumed == (NO_CALIBRATION.format(objective=IN_PLACE),)
    assert _stored(tmp_path) == record


def test_another_objective_takes_the_profiles_size_and_says_binning_is_not_applied(
    demo_microscope_with_sample: SampleMicroscopeFactory, tmp_path: Path
) -> None:
    microscope = _stand(demo_microscope_with_sample, tmp_path)

    result = _result(_run(microscope, tmp_path, wells="A1", objective=FORTY, **SMALL))

    assert result.objective == FORTY
    assert result.pixel_size_um == 0.1625  # the turret's, not the camera's 0.5
    assert result.fov_um == pytest.approx((512 * 0.1625, 300 * 0.1625))
    assert result.matrix_source == "profile"
    assert result.assumed == (
        f"pixel size of {FORTY} from the profile, for binning 1: "
        "the camera's binning is not applied",
        NO_CALIBRATION.format(objective=FORTY),
    )


def test_the_matrix_comes_from_a_calibration_of_the_same_objective_only(
    demo_microscope_with_sample: SampleMicroscopeFactory, tmp_path: Path
) -> None:
    microscope = _stand(demo_microscope_with_sample, tmp_path)
    calibration = _write_calibration(tmp_path / "run.json")

    result = _result(_run(microscope, tmp_path, wells="A1", **SMALL))

    assert result.matrix_source == "camera-calibration"
    assert result.matrix_um_per_px == calibration.matrix_um_per_px
    assert result.pixel_size_um == 0.4
    angle = math.radians(3.5)
    assert result.fov_um == pytest.approx(
        (
            0.4 * (512 * math.cos(angle) + 300 * math.sin(angle)),
            0.4 * (512 * math.sin(angle) + 300 * math.cos(angle)),
        )
    )
    # 3.5 degrees at 10 % overlap leaves gaps; 13 % closes them. No other line:
    # nothing here was assumed about the camera's rotation.
    assert result.assumed == (
        "the camera is turned 3.5° from the stage axes: at overlap 0.1 the grid "
        "leaves gaps between fields; use overlap=0.13",
    )


@pytest.mark.parametrize(
    ("seeded", "why"),
    [
        (
            {"objective": FORTY},
            f"it is of {FORTY}, the plan is for {IN_PLACE}",
        ),
        (
            {"objective": None},
            f"it is of an unknown objective, the plan is for {IN_PLACE}",
        ),
        (
            {"shape": (512, 300)},
            "it is of a frame of (512, 300), the camera gives (300, 512)",
        ),
    ],
    ids=["another-objective", "unknown-objective", "transposed-frame"],
)
def test_a_calibration_of_another_objective_or_frame_is_ignored_and_says_why(
    seeded: dict[str, object],
    why: str,
    demo_microscope_with_sample: SampleMicroscopeFactory,
    tmp_path: Path,
) -> None:
    microscope = _stand(demo_microscope_with_sample, tmp_path)
    _write_calibration(tmp_path / "run.json", **seeded)  # type: ignore[arg-type]

    result = _result(_run(microscope, tmp_path, wells="A1", **SMALL))

    assert result.matrix_source == "profile"
    assert result.pixel_size_um == PIXEL_UM
    assert result.matrix_um_per_px == pixel_matrix(PIXEL_UM)
    assert result.assumed == (
        f"camera calibration not used: {why}",
        NO_CALIBRATION.format(objective=IN_PLACE),
    )


def test_a_calibration_of_the_objective_asked_for_is_used_though_another_is_in_place(
    demo_microscope_with_sample: SampleMicroscopeFactory, tmp_path: Path
) -> None:
    microscope = _stand(demo_microscope_with_sample, tmp_path)
    _write_calibration(
        tmp_path / "run.json", objective=FORTY, pixel_size_um=0.2, rotation_deg=0.0
    )

    result = _result(_run(microscope, tmp_path, wells="A1", objective=FORTY, **SMALL))

    assert result.objective == FORTY
    assert result.matrix_source == "camera-calibration"
    assert result.pixel_size_um == 0.2
    assert result.assumed == ()  # measured, not turned: nothing to say


def test_a_calibration_that_the_overlap_covers_says_nothing_and_one_beyond_it_says_so(
    demo_microscope_with_sample: SampleMicroscopeFactory, tmp_path: Path
) -> None:
    microscope = _stand(demo_microscope_with_sample, tmp_path)
    _write_calibration(tmp_path / "run.json", rotation_deg=3.5, mirrored=True)

    covered = _result(_run(microscope, tmp_path, wells="A1", overlap=0.2, **SMALL))
    assert covered.matrix_source == "camera-calibration"
    assert covered.assumed == ()

    _write_calibration(tmp_path / "run.json", rotation_deg=30.0)
    turned = _result(_run(microscope, tmp_path, wells="A1", **SMALL))
    assert turned.assumed == (
        "the camera is turned 30.0° from the stage axes: no overlap below 0.9 "
        "closes the gaps between fields; straighten the camera",
    )


def test_a_parameter_pixel_size_wins_over_a_calibration(
    demo_microscope_with_sample: SampleMicroscopeFactory, tmp_path: Path
) -> None:
    microscope = _stand(demo_microscope_with_sample, tmp_path)
    _write_calibration(tmp_path / "run.json")

    result = _result(_run(microscope, tmp_path, wells="A1", pixel_size_um=0.3, **SMALL))

    assert result.matrix_source == "parameter"
    assert result.pixel_size_um == 0.3
    assert result.matrix_um_per_px == pixel_matrix(0.3)
    assert result.assumed == (
        "pixel size given as a parameter; rotation 0 and no mirror assumed",
    )


# ------------------------------------------------------- refusing to plan


def test_no_pixel_size_refuses_to_plan(
    demo_microscope_with_sample: SampleMicroscopeFactory, tmp_path: Path
) -> None:
    """Objective-2's profile size is 0.0 (measured): not measured, so no plan (FM-104)."""
    microscope = _stand(demo_microscope_with_sample, tmp_path)

    with pytest.raises(PluginRunError) as raised:
        _run(microscope, tmp_path, wells="A1", objective="Objective-2", **SMALL)

    assert raised.value.reason == NO_PIXEL_SIZE.format(objective="Objective-2")
    stored = _stored(tmp_path)
    assert stored.status is StageStatus.failed
    assert stored.result is None


@pytest.mark.parametrize("bad", [0.0, -1.5, float("nan"), float("inf")])
def test_a_pixel_size_that_is_not_finite_and_positive_refuses_to_plan(
    bad: float,
    demo_microscope_with_sample: SampleMicroscopeFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FM-104: infinity is above 0, and 0 means "not measured"; neither is a size."""
    microscope = _stand(demo_microscope_with_sample, tmp_path)

    monkeypatch.setattr(microscope.require(Camera), "pixel_size_um", lambda: bad)
    with pytest.raises(PluginRunError) as from_camera:
        _run(microscope, tmp_path, wells="A1", **SMALL)
    assert from_camera.value.reason == NO_PIXEL_SIZE.format(objective=IN_PLACE)
    monkeypatch.undo()

    turret = microscope.require(ObjectiveTurret)
    monkeypatch.setattr(turret, "pixel_size_um", lambda label=None: bad)
    with pytest.raises(PluginRunError) as from_turret:
        _run(microscope, tmp_path, wells="A1", objective=FORTY, **SMALL)
    assert from_turret.value.reason == NO_PIXEL_SIZE.format(objective=FORTY)

    assert _stored(tmp_path).result is None


def test_an_objective_not_on_the_turret_or_without_a_turret_is_refused(
    demo_microscope_with_sample: SampleMicroscopeFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    microscope = _stand(demo_microscope_with_sample, tmp_path)

    with pytest.raises(PluginRunError) as unknown:
        _run(microscope, tmp_path, wells="A1", objective="Nikon 60X", **SMALL)
    assert (
        unknown.value.reason == f"objective 'Nikon 60X' is not on the turret: {LABELS}"
    )

    # An empty position has the label "", which is no objective that can be asked for.
    with pytest.raises(PluginRunError, match="objective '' is not on the turret"):
        _run(microscope, tmp_path, wells="A1", objective="", **SMALL)

    _without(microscope, monkeypatch, ObjectiveTurret)
    with pytest.raises(PluginRunError) as no_turret:
        _run(microscope, tmp_path, wells="A1", objective=IN_PLACE, **SMALL)
    assert no_turret.value.reason == (
        "the stand has no turret, so objective cannot be checked; leave it unset"
    )
    assert _stored(tmp_path).result is None


def test_without_a_turret_the_objective_is_none_and_the_cameras_size_is_used(
    demo_microscope_with_sample: SampleMicroscopeFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    microscope = _stand(demo_microscope_with_sample, tmp_path)
    _without(microscope, monkeypatch, ObjectiveTurret)

    result = _result(_run(microscope, tmp_path, wells="A1", **SMALL))

    assert result.objective is None  # no turret: unknown, not guessed
    assert result.pixel_size_um == PIXEL_UM
    assert result.assumed == (NO_CALIBRATION.format(objective="this stand"),)


def test_a_calibration_of_an_unknown_objective_is_used_on_a_stand_without_a_turret(
    demo_microscope_with_sample: SampleMicroscopeFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    microscope = _stand(demo_microscope_with_sample, tmp_path)
    _write_calibration(tmp_path / "run.json", objective=None, rotation_deg=0.0)
    _without(microscope, monkeypatch, ObjectiveTurret)

    result = _result(_run(microscope, tmp_path, wells="A1", **SMALL))

    assert result.objective is None
    assert result.matrix_source == "camera-calibration"
    assert result.pixel_size_um == 0.4


def test_the_camera_must_report_a_frame(
    demo_microscope_with_sample: SampleMicroscopeFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    microscope = _stand(demo_microscope_with_sample, tmp_path)
    monkeypatch.setattr(microscope.require(Camera), "image_shape", lambda: (0, 512))

    with pytest.raises(PluginRunError) as raised:
        _run(microscope, tmp_path, wells="A1", **SMALL)

    assert raised.value.reason == (
        "the camera reports a frame of (0, 512) pixels: check the camera's region of interest"
    )


# ------------------------------------------------------------ the grid


def test_the_grid_is_derived_from_the_cameras_field_and_the_well(
    demo_microscope_with_sample: SampleMicroscopeFactory, tmp_path: Path
) -> None:
    """A 96-well well is 6400 µm: 256 x 150 µm fields take 28 columns (widths) and 48 rows (heights)."""
    microscope = _stand(demo_microscope_with_sample, tmp_path)

    result = _result(_run(microscope, tmp_path, wells="A1"))

    assert (result.rows, result.columns) == (48, 28)
    assert result.overlap == 0.1
    # Clipping to the round well drops the corners of the 48 x 28 grid.
    assert 0 < len(result.fields) < 48 * 28
    assert result.fields[0].index == 0

    half = _result(_run(microscope, tmp_path, wells="A1", coverage=0.5))
    assert (half.rows, half.columns) == (24, 14)


def test_given_rows_and_columns_win_over_the_coverage(
    demo_microscope_with_sample: SampleMicroscopeFactory, tmp_path: Path
) -> None:
    microscope = _stand(demo_microscope_with_sample, tmp_path)

    result = _result(
        _run(microscope, tmp_path, wells="A1", rows=3, columns=2, coverage=0.2)
    )

    assert (result.rows, result.columns) == (3, 2)
    assert (
        len(result.fields) == 6
    )  # a 3 x 2 grid of 256 x 150 µm lies inside a 6400 µm well


def test_clip_to_well_false_keeps_every_field(
    demo_microscope_with_sample: SampleMicroscopeFactory, tmp_path: Path
) -> None:
    microscope = _stand(demo_microscope_with_sample, tmp_path)

    clipped = _result(_run(microscope, tmp_path, wells="A1"))
    whole = _result(_run(microscope, tmp_path, wells="A1", clip_to_well=False))

    assert len(whole.fields) == 48 * 28
    assert len(clipped.fields) < len(whole.fields)
    assert {f.name for f in clipped.fields} < {f.name for f in whole.fields}


def test_a_derived_grid_over_100_per_axis_is_refused(
    demo_microscope_with_sample: SampleMicroscopeFactory, tmp_path: Path
) -> None:
    """FM-121: the default is derived from the very value that may be wrong (here a size 500 times too small)."""
    microscope = _stand(demo_microscope_with_sample, tmp_path, pixel_size_um=0.001)

    with pytest.raises(PluginRunError) as raised:
        _run(microscope, tmp_path, wells="A1")

    found = re.fullmatch(
        r"the grid would be (\d+) × (\d+) fields per well at 0\.001 µm per pixel: "  # noqa: RUF001
        r"check the pixel size, or give rows and columns",
        raised.value.reason,
    )
    assert found is not None, raised.value.reason
    # 0.3 µm high fields at 90 % step: 6400 / 0.27; 0.512 µm wide: 6400 / 0.4608.
    assert int(found.group(1)) == pytest.approx(23704, abs=2)
    assert int(found.group(2)) == pytest.approx(13890, abs=2)
    assert _stored(tmp_path).result is None

    # What the message offers: a grid that is given is not derived from the pixel size.
    assert _result(_run(microscope, tmp_path, wells="A1", rows=3, columns=3)).rows == 3


def test_a_derived_grid_over_100_on_one_axis_only_is_refused(
    demo_microscope_with_sample: SampleMicroscopeFactory, tmp_path: Path
) -> None:
    """A flat 30 x 512 frame: 474 rows of 15 µm fields, but only 28 columns of 256 µm."""
    microscope = _stand(demo_microscope_with_sample, tmp_path, shape=(30, 512))

    with pytest.raises(PluginRunError) as raised:
        _run(microscope, tmp_path, wells="A1")

    assert raised.value.reason == (
        "the grid would be 474 × 28 fields per well at 0.5 µm per pixel: "  # noqa: RUF001
        "check the pixel size, or give rows and columns"
    )


# ------------------------------------------------------------------ the wells


def test_wells_that_cannot_be_read_fail_the_stage_and_name_them(
    demo_microscope_with_sample: SampleMicroscopeFactory, tmp_path: Path
) -> None:
    microscope = _stand(demo_microscope_with_sample, tmp_path)

    with pytest.raises(PluginRunError) as raised:
        _run(microscope, tmp_path, wells="A1, Q99", **SMALL)

    assert raised.value.reason == (
        "wells not understood on a 96-well plate: 'Q99'; "
        "give wells as A1, B2-B5, A1:D6, C*, *3 or all"
    )
    assert _stored(tmp_path).status is StageStatus.failed

    with pytest.raises(PluginRunError) as nothing:
        _run(microscope, tmp_path, wells=" , ", **SMALL)
    assert nothing.value.reason == (
        "no wells selected; give wells as A1, B2-B5, A1:D6, C*, *3 or all"
    )


def test_the_typing_order_never_reaches_the_plan_but_the_serpentine_does(
    demo_microscope_with_sample: SampleMicroscopeFactory, tmp_path: Path
) -> None:
    microscope = _stand(demo_microscope_with_sample, tmp_path)

    serpentine = _result(_run(microscope, tmp_path, wells="B2, B1, A2, A1", **SMALL))
    raster = _result(
        _run(microscope, tmp_path, wells="B2 B1 A2 A1", serpentine=False, **SMALL)
    )

    assert serpentine.wells == ("A1", "A2", "B2", "B1")
    assert raster.wells == ("A1", "A2", "B1", "B2")
    assert [f.well for f in serpentine.fields] == (
        ["A1"] * 4 + ["A2"] * 4 + ["B2"] * 4 + ["B1"] * 4
    )
    assert [f.well for f in raster.fields] == (
        ["A1"] * 4 + ["A2"] * 4 + ["B1"] * 4 + ["B2"] * 4
    )


def test_rows_and_columns_are_given_together_or_not_at_all() -> None:
    with pytest.raises(ValidationError, match="rows and columns are given together"):
        ScanPlanParams(wells="A1", rows=3)  # type: ignore[call-arg]
    with pytest.raises(ValidationError, match="rows and columns are given together"):
        ScanPlanParams(wells="A1", columns=3)  # type: ignore[call-arg]

    assert ScanPlanParams(wells="A1", rows=3, columns=3).rows == 3
    assert ScanPlanParams(wells="A1").rows is None


@pytest.mark.parametrize(
    "bad",
    [
        {"overlap": 0.9},
        {"overlap": -0.1},
        {"coverage": 0.0},
        {"coverage": 1.6},
        {"rows": 0, "columns": 1},
        {"rows": 101, "columns": 1},
        {"pixel_size_um": 0.0},
        {"pixel_size_um": 100.5},
        {"pixel_size_um": float("nan")},
    ],
)
def test_the_parameters_are_bounded(bad: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ScanPlanParams(wells="A1", **bad)  # type: ignore[arg-type]


def test_a_plan_the_grid_cannot_make_fails_the_stage_with_its_reason(
    demo_microscope_with_sample: SampleMicroscopeFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The grid refuses a well that no field touches: the stage fails, it does not drop the well."""
    microscope = _stand(demo_microscope_with_sample, tmp_path)

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise ValueError("no field of the grid touches well A1: made up for the test")

    monkeypatch.setattr("smc.plugins.scan_plan.plugin.plan_fields", refuse)
    with pytest.raises(PluginRunError) as raised:
        _run(microscope, tmp_path, wells="A1", **SMALL)

    assert (
        raised.value.reason
        == "no field of the grid touches well A1: made up for the test"
    )
    assert _stored(tmp_path).status is StageStatus.failed


# --------------------------------------------------------------- the estimate


def test_the_estimate_is_none_without_timings(
    demo_microscope_with_sample: SampleMicroscopeFactory, tmp_path: Path
) -> None:
    microscope = _stand(demo_microscope_with_sample, tmp_path)

    result = _result(_run(microscope, tmp_path, wells="A1", **SMALL))

    assert result.estimate_s is None  # unknown, not guessed


def test_the_estimate_counts_refocus_on_a_stand_with_autofocus(
    demo_microscope_with_sample: SampleMicroscopeFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    microscope = _stand(demo_microscope_with_sample, tmp_path)
    timings = _write_timings(tmp_path / "run.json")
    assert microscope.get(Autofocus) is not None

    with_autofocus = _result(_run(microscope, tmp_path, wells="A1:A2", **SMALL))
    assert with_autofocus.estimate_s == pytest.approx(
        estimate_s(with_autofocus.fields, timings, refocus=True)
    )
    assert with_autofocus.estimate_s != pytest.approx(
        estimate_s(with_autofocus.fields, timings, refocus=False)
    )

    # One field, so no move: a refocus (400 ms) and a snap (20 ms), or the snap alone.
    one = _result(_run(microscope, tmp_path, wells="A1", rows=1, columns=1))
    assert one.estimate_s == pytest.approx(0.42)

    _without(microscope, monkeypatch, Autofocus)
    without = _result(_run(microscope, tmp_path, wells="A1", rows=1, columns=1))
    assert without.estimate_s == pytest.approx(0.02)


# ------------------------------------------------------------ dry run, reading back


def test_a_dry_run_plans_like_a_real_run(
    demo_microscope_dry: Microscope, tmp_path: Path
) -> None:
    """Nothing moves, so a dry run is not a rehearsal of the plan: it is the plan."""
    _write_plate(tmp_path / "run.json")
    demo_microscope_dry.override(
        Camera,
        SampleCamera(
            demo_microscope_dry, PlateSample(), pixel_size_um=PIXEL_UM, shape=SHAPE
        ),
    )
    assert demo_microscope_dry.dry_run

    result = _result(_run(demo_microscope_dry, tmp_path, wells="A1", **SMALL))

    assert result.pixel_size_um == PIXEL_UM
    assert result.assumed == (NO_CALIBRATION.format(objective=IN_PLACE),)
    assert [f.name for f in result.fields] == [f"A1_{i:04d}" for i in range(4)]
    assert _stored(tmp_path).dry_run is True


def test_the_result_reads_back_from_the_run_document(
    demo_microscope_with_sample: SampleMicroscopeFactory, tmp_path: Path
) -> None:
    microscope = _stand(demo_microscope_with_sample, tmp_path)

    record = _run(microscope, tmp_path, wells="A1:B2", **SMALL)

    result = _result(record)
    stored = RunDocument.load(tmp_path / "run.json").stages["scan-plan"]
    assert stored.result is not None
    assert ScanPlanResult.model_validate(stored.result) == result
    assert [str(name) for name in result.plate_plan.selected_well_names] == [
        "A1",
        "A2",
        "B2",
        "B1",
    ]
    assert result.wells == ("A1", "A2", "B2", "B1")
    # The fields are in visiting order, a well at a time.
    assert [f.well for f in result.fields] == ["A1"] * 4 + ["A2"] * 4 + ["B2"] * 4 + [
        "B1"
    ] * 4
