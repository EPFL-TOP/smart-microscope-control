"""Finding the plate from a survey: the pure search of ``plate_find.py``, ported with its tests (m3 §6, #20).

Run against synthetic plates whose type, position and rotation are known, so the
answer is checked against truth. The metric is the one that matters in the lab:
the **worst well-centre error anywhere on the plate** against the well radius (a
fit 1 mm out on a 35 mm well is fine and the same error on a 3.4 mm well is
not). The module runs with ``RuntimeWarning`` as an error (FM-131): numpy warns
where Python's floats do not, and on one numpy only.

The first tests are those of ``nikon-control``'s ``tests/test_plate_find.py``
(``1aad7c9``), edited as the plan lists: the survey is a list of ``SurveyPoint``,
a test that could not fail (an ``or`` of three disjuncts, a ``fit is not None``)
asserts the one thing it names, and the fit's conversion is checked against what
``plate-registration`` produces here.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from smc.plugins.plate_finding import algorithm
from smc.plugins.plate_finding.algorithm import (
    DECISIVE_MARGIN,
    DEFAULT_JITTER,
    MAX_SURVEY_POINTS,
    PlateFit,
    SurveyPoint,
    decide,
    fit,
    fit_one,
    frame_value,
    grid_points,
    inside_mask,
    lattice_score,
    plate_pitch_um,
    plate_shape,
    suggested_step_um,
    survey_bounds_um,
    survey_extent_note,
    survey_points,
    well_size_um,
)
from smc.plugins.plate_registration.algorithm import calibrate
from smc.plugins.slots import WellRef

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")


def survey_for(
    plate: str,
    a1: tuple[float, float],
    rotation: float,
    *,
    polarity: int = 1,
    step_fraction: float | None = None,
    margin_pitches: float = 1.0,
    noise: float = 30.0,
    jitter: float = DEFAULT_JITTER,
    seed: int = 0,
) -> list[SurveyPoint]:
    """A survey of a plate that really is at ``(a1, rotation)``.

    The signal is made with ``inside_mask(..., inside_fraction=0.9)``: that is
    the original's ground truth (a well is "inside" out to 90 % of its radius),
    and not the code under test, which scores at 75 %.
    """
    rows, cols = plate_shape(plate)
    pitch_x, pitch_y = plate_pitch_um(plate)
    centre = (a1[0] + (cols - 1) * pitch_x / 2.0, a1[1] - (rows - 1) * pitch_y / 2.0)
    step = (min(pitch_x, pitch_y) * step_fraction) if step_fraction else None
    points = survey_points(
        plate,
        centre,
        step_um=step,
        margin_pitches=margin_pitches,
        jitter=jitter,
        seed=seed,
    )
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    # The real signal: light gets through a well and not through the plastic.
    inside = inside_mask(xs, ys, plate, a1, rotation, inside_fraction=0.9)
    rng = np.random.default_rng(seed + 1)
    values = 1000 + polarity * 400 * inside + rng.normal(0, noise, len(xs))
    return [
        SurveyPoint(float(x), float(y), float(v))
        for x, y, v in zip(xs, ys, values, strict=True)
    ]


def worst_well_error(
    plate: str,
    a1_true: tuple[float, float],
    rot_true: float,
    a1_fit: tuple[float, float],
    rot_fit: float,
) -> float:
    """Largest distance between a true and a fitted well centre, over the plate."""
    rows, cols = plate_shape(plate)
    pitch_x, pitch_y = plate_pitch_um(plate)

    def node(
        a1: tuple[float, float], rot: float, i: int, j: int
    ) -> tuple[float, float]:
        t = math.radians(rot)
        c, s = math.cos(t), math.sin(t)
        lx, ly = j * pitch_x, -i * pitch_y
        return a1[0] + c * lx - s * ly, a1[1] + s * lx + c * ly

    return max(
        math.dist(node(a1_true, rot_true, i, j), node(a1_fit, rot_fit, i, j))
        for i in range(rows)
        for j in range(cols)
    )


# --------------------------------------------------------------- the fit


@pytest.mark.parametrize(
    ("plate", "a1", "rot"),
    [
        ("96-well", (1000.0, 2000.0), 0.0),
        ("96-well", (-3210.0, 15400.0), 1.7),
        ("96-well", (20000.0, -5000.0), -2.3),
        ("384-well", (0.0, 0.0), 0.4),
        ("24-well", (500.0, -500.0), -0.8),
        ("48-well", (0.0, 0.0), 0.5),
    ],
)
def test_the_plate_is_identified_and_located(
    plate: str, a1: tuple[float, float], rot: float
) -> None:
    best, _notes = decide(fit(survey_for(plate, a1, rot)))

    assert best is not None, "no usable fit"
    assert best.plate == plate, f"identified as {best.plate}"
    err = worst_well_error(plate, a1, rot, best.a1_center_xy_um, best.rotation_deg)
    radius = min(well_size_um(plate)) / 2
    assert err < radius * 0.30, f"{err:.0f} µm is {err / radius:.0%} of a well radius"


def test_wells_darker_than_the_plastic_work_just_as_well() -> None:
    """A black-walled plate inverts the contrast; nothing should be assumed."""
    best, _ = decide(fit(survey_for("96-well", (1000.0, 2000.0), 0.6, polarity=-1)))

    assert best is not None
    assert best.plate == "96-well"
    assert best.polarity == -1
    err = worst_well_error(
        "96-well", (1000.0, 2000.0), 0.6, best.a1_center_xy_um, best.rotation_deg
    )
    assert err < 1000


def test_a_96_well_plate_is_not_mistaken_for_a_384_well_one() -> None:
    """The finite extent is what separates them, not the pitch alone."""
    fits = fit(survey_for("96-well", (0.0, 0.0), 0.0))

    by_plate = {f.plate: f.score for f in fits}
    assert fits[0].plate == "96-well"
    assert by_plate["96-well"] > by_plate["384-well"] + DECISIVE_MARGIN


# ------------------------------------------------- why it is built this way


def test_a_survey_that_stops_at_the_last_well_cannot_place_a1() -> None:
    """The margin is evidence, not padding.

    With no samples past the outer wells, sliding the plate along its own
    lattice still lands every well on a well: the plastic outside is what says
    which node is A1. The notes of the decision say so, whatever the fit found
    (the original asserted a three-way ``or`` that a fit that happened to be
    right passed without a note, FM-43).
    """
    tight = survey_for("96-well", (1000.0, 2000.0), 0.0, margin_pitches=0.05)

    _best, notes = decide(fit(tight, plates=["96-well"]))

    assert any("A1 cannot be pinned down" in note for note in notes)


def test_the_survey_extent_is_checked_and_reported() -> None:
    """A survey with no margin is too small for its own plate, and its fit says so."""
    tight = fit_one(
        survey_for("96-well", (0.0, 0.0), 0.0, margin_pitches=0.0, jitter=0.0),
        "96-well",
    )

    assert tight.notes == (
        "the survey spans 99x63 mm; this plate needs 108x72 mm — one pitch of "
        "margin past the outer wells — or which well is A1 cannot be pinned down",
    )


def test_sampling_is_jittered_by_default() -> None:
    """A regular raster's comb beats against the plate lattice and the
    rotation locks onto the sampling pattern instead of the plate."""
    regular = grid_points((0, 0, 1000, 1000), 100, jitter=0.0)
    jittered = grid_points((0, 0, 1000, 1000), 100)

    assert DEFAULT_JITTER > 0
    assert regular != jittered
    assert len({round(p[0]) for p in regular}) < len({round(p[0]) for p in jittered})


def test_the_suggested_step_is_a_third_of_the_pitch() -> None:
    """Half was measured failing on three of eight plate types."""
    for plate in ("96-well", "384-well", "24-well"):
        assert suggested_step_um(plate) == pytest.approx(min(plate_pitch_um(plate)) / 3)


# ------------------------------------------------------- honest refusals


def test_a_featureless_survey_is_refused_not_fitted() -> None:
    """Uniform brightness carries no lattice; a confident answer would be a lie."""
    points = survey_points("96-well", (0.0, 0.0))
    flat = [SurveyPoint(x, y, 1000.0) for x, y in points]

    best, notes = decide(fit(flat, plates=["96-well", "384-well"]))

    assert best is None
    assert any("too low to trust" in note for note in notes)


def test_too_few_samples_is_refused() -> None:
    tiny = [SurveyPoint(0.0, 0.0, 1.0), SurveyPoint(1.0, 1.0, 2.0)]

    fitted = fit_one(tiny, "96-well")

    assert not fitted.usable
    assert any("too few" in note for note in fitted.notes)


def test_a_close_call_between_plate_types_is_flagged() -> None:
    close = [
        PlateFit("96-well", (0, 0), 0.0, 0.80, 1, 100, 20),
        PlateFit("384-well", (0, 0), 0.0, 0.79, 1, 100, 20),
    ]

    best, notes = decide(close)

    assert best is not None
    assert best.plate == "96-well"
    assert any("barely beats" in note for note in notes)


# ------------------------------------------------------------- the output


def test_a_fit_becomes_the_plate_result_a_registration_produces() -> None:
    """A fit fills the ``plate`` slot with the model a registration fills it with."""
    fits = fit(survey_for("96-well", (1234.0, -5678.0), 0.9))
    best, notes = decide(fits)
    assert best is not None
    margin = fits[0].score - fits[1].score

    result = best.to_result(runner_up=fits[1].plate, margin=margin, notes=notes)

    # A lattice fit measured no reference well: it does not claim any.
    assert result.method == "lattice-fit"
    assert result.residual_um is None
    assert result.scale_error is None
    assert result.n_refs == 0
    assert result.refs == ()
    assert result.plate == "96-well"
    assert result.a1_center_xy_um == best.a1_center_xy_um
    assert result.fit is not None
    assert result.fit.score == best.score
    assert result.fit.polarity == best.polarity
    assert result.fit.runner_up == fits[1].plate
    assert result.fit.margin == pytest.approx(margin)
    assert result.fit.n_samples == best.n_samples
    assert result.fit.n_inside == best.n_inside
    # ...and it drives the rest of the pipeline as a registration's does.
    seed = calibrate("96-well", [WellRef(well="A1", x_um=1234.0, y_um=-5678.0)])
    assert math.dist(result.well_center_um("C5"), seed.well_center_um("C5")) < 2000
    plan = result.to_plan(selected_wells=([0, 1], [0, 1]))
    assert [str(p.name) for p in plan.image_positions] == ["A1", "B2"]


def test_to_result_without_a_runner_up_has_no_margin() -> None:
    """One plate fitted: nothing was compared, so the margin is unknown and not 0."""
    only = PlateFit("96-well", (10.0, 20.0), 0.5, 0.7, -1, 100, 20)

    result = only.to_result(
        runner_up=None, margin=None, notes=("a note",), assumed=("an assumption",)
    )

    assert result.fit is not None
    assert result.fit.runner_up is None
    assert result.fit.margin is None
    assert result.fit.polarity == -1
    assert result.fit.notes == ("a note",)
    assert result.assumed == ("an assumption",)


def test_inside_mask_follows_the_plate_frame_not_the_stage() -> None:
    """A rotated plate's wells are not on stage-axis rows.

    The effect grows with distance from A1: one pitch out, 5° leaves the point
    0.8 mm off its row and still inside the well; four pitches out it is 3.1 mm
    off and on the plastic. That growth with distance is exactly why rotation
    is the parameter worth spending samples on.

    Further out still it comes back "inside": the point has wandered into the
    NEXT row's well. That is real lattice behaviour, not a bug, and it is the
    same ambiguity the finite-extent stage of the fit exists to resolve.
    """
    near, far = 9000.0, 36000.0

    flat = inside_mask([near, far], [0.0, 0.0], "96-well", (0.0, 0.0), 0.0)
    turned = inside_mask([near, far], [0.0, 0.0], "96-well", (0.0, 0.0), 5.0)

    assert flat.tolist() == [True, True]
    assert turned.tolist() == [True, False]


# ----------------------------------------------------------- the survey


def test_without_jitter_the_survey_is_the_raster_walked_in_snake_order() -> None:
    """Rows alternate direction, so the stage does not cross the survey between two rows."""
    p = 39120.0  # the 6-well plate's pitch, 39.12 mm

    points = survey_points(
        "6-well",
        (0.0, 0.0),
        step_um=plate_pitch_um("6-well")[0],
        margin_pitches=0.0,
        jitter=0.0,
    )

    expected = [
        (-p, -p / 2),
        (0, -p / 2),
        (p, -p / 2),
        (p, p / 2),
        (0, p / 2),
        (-p, p / 2),
    ]
    assert points == [(pytest.approx(x), pytest.approx(y)) for x, y in expected]


def _original_span(low: float, high: float, step: float) -> list[float]:
    n = max(1, math.floor(abs(high - low) / step))
    return [low + i * (high - low) / n for i in range(n + 1)]


def _original_raster(
    bounds: tuple[float, float, float, float], step: float, jitter: float, seed: int
) -> list[tuple[float, float]]:
    """``plate_find.py``'s ``grid_points`` loop, quoted: the row-by-row raster."""
    x0, y0, x1, y1 = bounds
    rng = np.random.default_rng(seed)
    out = []
    for y in _original_span(y0, y1, step):
        for x in _original_span(x0, x1, step):
            if jitter:
                x += float(rng.uniform(-jitter * step, jitter * step))
                y_j = y + float(rng.uniform(-jitter * step, jitter * step))
            else:
                y_j = y
            out.append((x, y_j))
    return out


@pytest.mark.parametrize(
    ("jitter", "seed"), [(DEFAULT_JITTER, 3), (DEFAULT_JITTER, 11), (0.0, 0)]
)
def test_snake_order_keeps_the_points_of_the_raster(jitter: float, seed: int) -> None:
    """The same points from the same seed; only the order of every other row changes."""
    plate = "96-well"
    step = suggested_step_um(plate)
    bounds = survey_bounds_um(plate, (0.0, 0.0))

    points = survey_points(plate, (0.0, 0.0), jitter=jitter, seed=seed)
    raster = _original_raster(bounds, step, jitter, seed)

    assert sorted(points) == sorted(raster)
    assert points != raster
    walk = max(math.dist(a, b) for a, b in itertools.pairwise(points))
    original_walk = max(math.dist(a, b) for a, b in itertools.pairwise(raster))
    assert walk <= 3 * step
    # The original returns across the whole survey after every row.
    assert original_walk > 20 * step


@pytest.mark.parametrize(
    ("plate", "expected"),
    [
        ("6-well", 130),
        ("12-well", 208),
        ("24-well", 352),
        ("48-well", 616),
        ("96-well", 1120),
        ("384-well", 3952),
        ("1536-well", 14_800),
    ],
)
def test_a_survey_at_the_default_step_has_the_measured_number_of_points(
    plate: str, expected: int
) -> None:
    """The counts measured while planning (m3 §6.2): the step and the margin did not move."""
    assert len(survey_points(plate, (0.0, 0.0), jitter=0.0)) == expected


def test_a_step_that_does_not_divide_the_span_is_floored() -> None:
    """A 96-well survey spans 117 x 81 mm: 11.7 steps of 10 mm in x and 8.1 in y."""
    points = survey_points("96-well", (0.0, 0.0), step_um=10_000.0, jitter=0.0)

    assert len(points) == (11 + 1) * (8 + 1)


def test_a_survey_beyond_the_point_bound_is_refused_before_it_is_made(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A step of 1 µm on a 96-well plate is 9.5 billion points: it builds none."""

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("a list of points was made")

    monkeypatch.setattr(algorithm, "grid_points", forbidden)

    with pytest.raises(ValueError, match="the survey would take") as info:
        survey_points("96-well", (0.0, 0.0), step_um=1.0)

    count = (117_000 + 1) * (81_000 + 1)
    assert str(info.value) == (
        f"the survey would take {count} points, more than {MAX_SURVEY_POINTS}: "
        "give a larger step_um or a smaller margin_pitches"
    )
    assert MAX_SURVEY_POINTS == 25_000


def test_the_point_bound_is_on_the_count_and_not_on_the_plate() -> None:
    """The 1536-well plate at its default step is 14 800 points, and a margin of 5 is 21 328: both are under the bound."""
    assert len(survey_points("1536-well", (0.0, 0.0), margin_pitches=5.0)) == 21_328
    with pytest.raises(ValueError, match=r"more than 25000"):
        survey_points(
            "1536-well", (0.0, 0.0), step_um=suggested_step_um("1536-well") / 2
        )


@pytest.mark.parametrize("step", [0.0, -1.0, math.nan])
def test_a_step_that_is_not_positive_is_refused(step: float) -> None:
    with pytest.raises(ValueError, match="step must be positive"):
        grid_points((0.0, 0.0, 100.0, 100.0), step)


def test_a_span_that_is_not_a_number_of_points_is_refused() -> None:
    """``400 / 5e-324`` is infinity, and ``math.floor(inf)`` is an ``OverflowError`` (FM-74)."""
    with pytest.raises(ValueError, match="is not a number of points"):
        grid_points((0.0, 0.0, 400.0, 400.0), 5e-324)
    with pytest.raises(ValueError, match="is not a number of points"):
        grid_points((0.0, 0.0, math.inf, 400.0), 100.0)


def test_the_extent_note_names_what_the_plate_needs() -> None:
    default = survey_points("96-well", (0.0, 0.0))
    no_margin = survey_points("96-well", (0.0, 0.0), margin_pitches=0.0, jitter=0.0)
    # 24-well: 247 points spanning 114 760 x 76 760 µm. The 24-well plate needs
    # 114 x 76 mm (5 x 3 pitches of 19 mm, plus one); the 6-well plate 117.36 x
    # 78.24 mm (2 x 1 pitches of 39.12 mm, plus one).
    larger = survey_points("24-well", (0.0, 0.0), margin_pitches=0.52, jitter=0.0)

    assert survey_extent_note(default, "96-well") is None
    assert survey_extent_note(no_margin, "96-well") == (
        "the survey spans 99x63 mm; this plate needs 108x72 mm — one pitch of "
        "margin past the outer wells — or which well is A1 cannot be pinned down"
    )
    assert len(larger) == 247
    assert survey_extent_note(larger, "24-well") is None
    assert survey_extent_note(larger, "6-well") == (
        "the survey spans 115x77 mm; this plate needs 117x78 mm — one pitch of "
        "margin past the outer wells — or which well is A1 cannot be pinned down"
    )
    assert survey_extent_note([], "96-well") is None
    assert survey_extent_note([(0.0, 0.0)], "96-well") is None


def test_a_fit_notes_the_extent_it_was_given() -> None:
    """``fit_one`` and the plugin share one check: the note is the function's, word for word."""
    survey = survey_for("96-well", (0.0, 0.0), 0.0, margin_pitches=0.0, jitter=0.0)
    points = [(p.x_um, p.y_um) for p in survey]

    note = survey_extent_note(points, "96-well")

    assert note is not None
    assert fit_one(survey, "96-well").notes == (note,)


# ------------------------------------------------------ a frame's number


def test_frame_value_is_the_float64_mean_or_std() -> None:
    extremes = np.array([[0, 65535]], dtype=np.uint16)
    small = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    # 2**24 + 1 + 1 + 1 in float32 loses every 1: the sum needs float64.
    wide = np.array([[16_777_216.0, 1.0, 1.0, 1.0]], dtype=np.float32)

    assert frame_value(extremes, "mean") == 32767.5
    assert frame_value(extremes, "std") == 32767.5
    assert frame_value(small, "mean") == 3.5
    assert frame_value(small, "std") == pytest.approx(math.sqrt(17.5 / 6))
    assert frame_value(wide, "mean") == 4_194_304.75
    assert isinstance(frame_value(extremes, "mean"), float)


# ------------------------------------------------------- the score's edges


def test_a_perfect_correlation_scores_one_and_no_more() -> None:
    """``LatticeFit.score`` is bounded by 1: a sum made by hand may land an ulp above it."""
    a1 = (1000.0, 2000.0)
    points = survey_points("96-well", (1000.0 + 49500.0, 2000.0 - 31500.0), seed=4)
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    inside = inside_mask(xs, ys, "96-well", a1, 0.0)
    values = np.where(inside, 2000.0, 700.0)

    score, polarity, n_inside = lattice_score(xs, ys, values, "96-well", a1, 0.0)

    assert score <= 1.0
    assert score == pytest.approx(1.0)
    assert polarity == 1
    assert n_inside == int(inside.sum())


def test_a_sample_that_is_not_finite_is_not_fitted() -> None:
    """A ``nan`` brightness (a hand-built point) is a note and a score of 0, not a warning."""
    survey = survey_for("96-well", (0.0, 0.0), 0.0)
    survey[10] = SurveyPoint(survey[10].x_um, survey[10].y_um, math.nan)

    fitted = fit_one(survey, "96-well")

    assert fitted.score == 0.0
    assert not fitted.usable
    assert fitted.notes == ("a sample is not finite",)


def test_the_unknown_plate_is_the_registrys_error() -> None:
    with pytest.raises(ValueError, match="Unknown plate name 'glass'"):
        fit_one([SurveyPoint(float(i), 0.0, 1.0) for i in range(10)], "glass")
