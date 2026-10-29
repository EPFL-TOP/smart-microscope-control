"""Find the plate automatically: which type it is, and where it sits (m3 §6).

A port of ``nikon-control``'s ``plate_find.py`` (``1aad7c9``), with its search
unchanged. The idea is what screening systems do: survey the plate at low power
and match what is seen against the catalogue. Two details make it much cheaper
than a tiled scan:

**Three numbers are fitted, not a picture.** A plate is a rigid lattice of
known pitch, so locating it means finding a phase and a rotation: A1's centre
and the plate's rotation, which is what everything downstream consumes. That
needs enough samples to pin the lattice down, not every pixel of the plate.

**The field of view is irrelevant, so the objective never changes.** Each
sample is one number (is there a well here or plastic?) and a 40x field answers
that as well as a 4x one, because light through a well and through the plastic
between wells differ by far more than focus does. Whether a frame's mean
separates the two at 40x, or through a black-walled plate, is not measured.

**No threshold.** The score is the correlation between the measured brightness
and the *predicted* in-a-well indicator. That works whether wells read brighter
or darker than the plastic (the sign is reported, not assumed), and it is
comparable between candidate plates, which is how the plate type is identified
rather than asked for.

The search has four stages, and each earned its place by failing without it:

1. **Rotation**, ranked by Fourier strength at the lattice frequency, with the
   phase solved analytically for each: phase and rotation are strongly coupled
   over a 100 mm plate (1 degree displaces the far edge by a fifth of a pitch),
   so they cannot be gridded independently at any affordable resolution.
2. **Phase sharpening** against an infinite lattice. The Fourier phase is
   biased by about 0.15 of a pitch because a lattice of discs is not a
   sinusoid, which at a 9 mm pitch is over a millimetre, past the tolerance the
   scoring uses.
3. **Which node is A1**, by integer offsets in *both* directions. This is the
   stage that uses the plate's finite extent, and so the one that tells a
   96-well plate from a 384-well one.
4. A local **refinement** on all three parameters against the real score.

Measured on synthetic plates of eight types at rotations from -2.3 to +1.7
degrees: the type identified in every case, and well centres located to
0.3-0.7 mm on a 96-well plate, about a fifth of a well radius. That places a
scan grid; it is coarser than the ~50 µm a wall touch gives.

What this port adds, each on a measurement (m3 §6.2):

* **Snake order.** The demo stage moves at its ``Velocity``, and the original
  raster ends every row with a return across the whole survey (27 returns of
  117 mm on a 96-well plate). Rows are walked alternately; the points and the
  fit are the same.
* **A bound on the points** (``MAX_SURVEY_POINTS``), counted in integers before
  any is made (FM-74, FM-32).
* **No matrix product on coordinates** (FM-131): the rotation is written out,
  and the correlation is summed by hand (``np.corrcoef`` is a ``dot``).
* **Units in names, a ``PlateResult`` out.** A fit claims no references that
  nobody took (``n_refs=0``), not ``nikon-control``'s ``max(2, n_inside)``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import useq

from smc.plugins.slots import LatticeFit, PlateResult

__all__ = [
    "DECISIVE_MARGIN",
    "DEFAULT_JITTER",
    "INSIDE_FRACTION",
    "MAX_ROTATION_DEG",
    "MAX_SURVEY_POINTS",
    "MIN_SCORE",
    "ROTATION_CANDIDATES",
    "STEP_FRACTION",
    "PlateFit",
    "SurveyPoint",
    "decide",
    "fit",
    "fit_one",
    "frame_value",
    "grid_points",
    "inside_mask",
    "lattice_score",
    "plate_pitch_um",
    "plate_shape",
    "suggested_step_um",
    "survey_bounds_um",
    "survey_extent_note",
    "survey_points",
    "well_size_um",
]

#: Wells are sampled as "inside" out to this fraction of the nominal radius.
#: Short of the wall, because the meniscus and the wall itself are neither
#: clean well nor clean plastic.
INSIDE_FRACTION = 0.75
#: How far the plate can plausibly be rotated in its holder.
MAX_ROTATION_DEG = 4.0
#: A winning plate type must beat the runner-up by this much to be called.
DECISIVE_MARGIN = 0.08
#: Below this correlation nothing was really found.
MIN_SCORE = 0.25
#: How many distinct rotation peaks to carry into the full scoring.
ROTATION_CANDIDATES = 3
#: Random nudge applied to each sample, as a fraction of the sampling step.
DEFAULT_JITTER = 0.5
#: Sampling step as a fraction of the well pitch. A third works; a half was
#: measured failing on 3 of 8 plate types, by up to 42 % of a well radius.
STEP_FRACTION = 1 / 3
#: The most points one survey may plan. At the default step and margin a plate
#: takes 130 (6-well) to 14 800 (1536-well); a margin of 5 on the 1536-well
#: plate takes 21 328. A step of 1 µm would take billions (FM-74, FM-32).
MAX_SURVEY_POINTS = 25_000

_MIN_SAMPLES = 8


@dataclass(frozen=True, slots=True)
class SurveyPoint:
    """One survey sample: where the stage was and how bright the frame was there."""

    x_um: float
    y_um: float
    value: float


@dataclass(frozen=True, slots=True)
class PlateFit:
    """One candidate answer: which plate, where, and how well it fits.

    ``score`` is the absolute correlation with the well indicator, 0 to 1;
    ``polarity`` is +1 when wells read brighter than the plastic and -1 when
    darker.
    """

    plate: str
    a1_center_xy_um: tuple[float, float]
    rotation_deg: float
    score: float
    polarity: Literal[1, -1]
    n_samples: int
    n_inside: int
    notes: tuple[str, ...] = ()

    @property
    def usable(self) -> bool:
        """Whether the fit found a lattice: a score of ``MIN_SCORE`` and more than 3 samples in wells."""
        return self.score >= MIN_SCORE and self.n_inside > 3

    def describe(self) -> str:
        """One line for the log: the plate, A1, the rotation and what the wells read like."""
        sign = "brighter" if self.polarity > 0 else "darker"
        return (
            f"{self.plate}: A1 at ({self.a1_center_xy_um[0]:.0f}, "
            f"{self.a1_center_xy_um[1]:.0f}) µm, rotation "
            f"{self.rotation_deg:+.3f}°, score {self.score:.3f} "
            f"({self.n_inside}/{self.n_samples} samples in wells, wells "
            f"read {sign})"
        )

    def to_result(
        self,
        *,
        runner_up: str | None,
        margin: float | None,
        notes: Sequence[str],
        assumed: Sequence[str] = (),
    ) -> PlateResult:
        """The fit as the ``plate`` slot's value, built once and validated.

        A lattice fit measures no reference well: ``residual_um`` and
        ``scale_error`` are unknown (``None``) and ``n_refs`` is 0. ``notes``
        are ``decide``'s, which include the winner's own and the "barely beats"
        one; ``runner_up`` and ``margin`` are ``None`` when one plate was
        fitted.
        """
        return PlateResult(
            plate=self.plate,
            a1_center_xy_um=self.a1_center_xy_um,
            rotation_deg=self.rotation_deg,
            method="lattice-fit",
            residual_um=None,
            scale_error=None,
            n_refs=0,
            refs=(),
            fit=LatticeFit(
                score=self.score,
                polarity=self.polarity,
                runner_up=runner_up,
                margin=margin,
                n_samples=self.n_samples,
                n_inside=self.n_inside,
                notes=tuple(notes),
            ),
            assumed=tuple(assumed),
        )


# ----------------------------------------------------------------- the plate


def plate_pitch_um(plate: str) -> tuple[float, float]:
    """Well-to-well spacing along x and y, in µm (``useq`` keeps it in mm).

    Raises:
        ValueError: ``useq`` has no plate of this name (its error, unchanged).
    """
    pitch_x, pitch_y = useq.WellPlate.from_str(plate).well_spacing
    return float(pitch_x) * 1000.0, float(pitch_y) * 1000.0


def plate_shape(plate: str) -> tuple[int, int]:
    """The number of rows and of columns of wells.

    Raises:
        ValueError: ``useq`` has no plate of this name.
    """
    found = useq.WellPlate.from_str(plate)
    return int(found.rows), int(found.columns)


def well_size_um(plate: str) -> tuple[float, float]:
    """The well's width and height, in µm (``useq`` keeps it in mm).

    Raises:
        ValueError: ``useq`` has no plate of this name.
    """
    width, height = useq.WellPlate.from_str(plate).well_size
    return float(width) * 1000.0, float(height) * 1000.0


# ---------------------------------------------------------------- sampling


def _count(low: float, high: float, step_um: float) -> int:
    """How many points ``_span`` puts between two limits, from the floats alone.

    The quotient is checked before it becomes an integer: ``400.0 / 5e-324`` is
    ``inf``, and ``math.floor(inf)`` is an ``OverflowError`` (FM-74). Written
    ``not step_um > 0`` so that a ``nan`` step is refused with the rest.
    """
    if not step_um > 0.0:
        raise ValueError("step must be positive")
    quotient = abs(high - low) / step_um
    if not math.isfinite(quotient):
        raise ValueError(
            f"a span of {abs(high - low):g} µm in steps of {step_um:g} µm is not a "
            "number of points: give finite bounds and a larger step_um"
        )
    return max(1, math.floor(quotient)) + 1


def _span(low: float, high: float, step_um: float) -> list[float]:
    """Evenly spaced values from ``low`` to ``high``, at most ``step_um`` apart."""
    n = _count(low, high, step_um) - 1
    return [low + i * (high - low) / n for i in range(n + 1)]


def grid_points(
    bounds_um: tuple[float, float, float, float],
    step_um: float,
    *,
    jitter: float = DEFAULT_JITTER,
    seed: int = 0,
) -> list[tuple[float, float]]:
    """A coarse raster over a box, each point randomly nudged, walked in snake order.

    **The jitter matters.** A perfectly regular raster is itself periodic, and
    its comb beats against the plate's lattice: the rotation estimate then locks
    onto the sampling pattern instead of the plate. Nudging each point by a
    fraction of a step destroys that resonance and costs nothing.

    Sampling along a few *lines* instead is tempting (180 points against 1120
    for a 96-well plate) and it does not work. Lines under-constrain a
    two-dimensional lattice: measured over several plates and rotations, line
    surveys produced fits scoring 0.99 that were millimetres wrong, which is
    worse than no answer.

    **Snake order.** The jitter is drawn exactly as ``plate_find.py`` draws it
    (row by row from the lowest y, an x draw and then a y draw per point, from
    one generator), and then every other row is reversed, so that the stage
    does not cross the whole survey between two rows. The points are the same.
    The caller bounds the count (``survey_points`` does): a list is made here.

    Raises:
        ValueError: ``step_um`` is not positive, or a span is not finite.
    """
    x0, y0, x1, y1 = bounds_um
    rng = np.random.default_rng(seed)
    out: list[tuple[float, float]] = []
    xs = _span(x0, x1, step_um)
    for index, y in enumerate(_span(y0, y1, step_um)):
        row: list[tuple[float, float]] = []
        for x in xs:
            if jitter:
                x_j = x + float(rng.uniform(-jitter * step_um, jitter * step_um))
                y_j = y + float(rng.uniform(-jitter * step_um, jitter * step_um))
            else:
                x_j, y_j = x, y
            row.append((x_j, y_j))
        if index % 2:
            row.reverse()
        out.extend(row)
    return out


def suggested_step_um(plate: str) -> float:
    """A sampling step fine enough to resolve this plate's lattice: a third of the pitch.

    Half was measured failing on three of eight plate types, by up to 42 % of a
    well radius: enough to put a scan grid off the well it was aiming at.
    """
    return min(plate_pitch_um(plate)) * STEP_FRACTION


def survey_bounds_um(
    plate: str,
    centre_um: tuple[float, float],
    margin_pitches: float = 1.0,
) -> tuple[float, float, float, float]:
    """A box ``(x_min, y_min, x_max, y_max)`` around the plate's wells, plus a margin of pitches.

    **The margin is not padding, it is evidence.** Without samples beyond the
    outermost wells, nothing contradicts a hypothesis that slides the plate
    along its own lattice: every well still lands on a well. What pins down
    which node is A1 is measuring the plastic past the last one. Measured across
    eight plate types, a 0.1-pitch margin got four wrong; a one-pitch margin got
    seven of eight right.
    """
    rows, cols = plate_shape(plate)
    pitch_x, pitch_y = plate_pitch_um(plate)
    centre_x, centre_y = centre_um
    half_x = (cols - 1) * pitch_x / 2.0 + pitch_x * margin_pitches
    half_y = (rows - 1) * pitch_y / 2.0 + pitch_y * margin_pitches
    return centre_x - half_x, centre_y - half_y, centre_x + half_x, centre_y + half_y


def survey_points(
    plate: str,
    centre_um: tuple[float, float],
    *,
    step_um: float | None = None,
    margin_pitches: float = 1.0,
    jitter: float = DEFAULT_JITTER,
    seed: int = 0,
) -> list[tuple[float, float]]:
    """Everywhere to sample, for a plate whose centre is roughly known.

    The count is worked out in Python integers from the bounds before any point
    is made: a step of 1 µm on a 96-well plate would otherwise be a list of
    billions (FM-74).

    Raises:
        ValueError: The step is not positive, or the survey would take more than
            ``MAX_SURVEY_POINTS`` points; the message names the count.
    """
    step = suggested_step_um(plate) if step_um is None else step_um
    bounds = survey_bounds_um(plate, centre_um, margin_pitches)
    x0, y0, x1, y1 = bounds
    n_points = _count(x0, x1, step) * _count(y0, y1, step)
    if n_points > MAX_SURVEY_POINTS:
        raise ValueError(
            f"the survey would take {n_points} points, more than "
            f"{MAX_SURVEY_POINTS}: give a larger step_um or a smaller margin_pitches"
        )
    return grid_points(bounds, step, jitter=jitter, seed=seed)


def survey_extent_note(
    points_um: Sequence[tuple[float, float]], plate: str
) -> str | None:
    """Why these points cannot pin down A1 of this plate, or ``None`` when they can.

    The survey has to reach *beyond* the wells, or nothing contradicts a
    hypothesis that slides the plate along its own lattice: the evidence that
    fixes which node is A1 is the plastic outside the last well. A survey
    without margin still placed a 96-well plate within 440 µm while this note
    reported it, so what it shows is that A1 could be a whole well off, not
    that it is. ``None`` for fewer than 2 points: there is no extent to judge.
    """
    if len(points_um) < 2:
        return None
    x0 = min(p[0] for p in points_um)
    x1 = max(p[0] for p in points_um)
    y0 = min(p[1] for p in points_um)
    y1 = max(p[1] for p in points_um)
    rows, cols = plate_shape(plate)
    pitch_x, pitch_y = plate_pitch_um(plate)
    need_x, need_y = (cols - 1) * pitch_x, (rows - 1) * pitch_y
    if (x1 - x0) < need_x + pitch_x or (y1 - y0) < need_y + pitch_y:
        return (
            f"the survey spans {(x1 - x0) / 1000:.0f}x{(y1 - y0) / 1000:.0f} mm; "
            f"this plate needs {(need_x + pitch_x) / 1000:.0f}x"
            f"{(need_y + pitch_y) / 1000:.0f} mm — one pitch of margin past the "
            f"outer wells — or which well is A1 cannot be pinned down"
        )
    return None


def frame_value(frame: npt.NDArray[Any], statistic: Literal["mean", "std"]) -> float:
    """The survey's one number for a frame: its mean or its standard deviation, in float64.

    ``dtype=np.float64`` because a ``uint16`` frame would otherwise be
    accumulated in a narrower type.
    """
    if statistic == "mean":
        return float(np.mean(frame, dtype=np.float64))
    return float(np.std(frame, dtype=np.float64))


# ----------------------------------------------------------------- scoring


def inside_mask(
    xs_um: npt.ArrayLike,
    ys_um: npt.ArrayLike,
    plate: str,
    a1_center_xy_um: tuple[float, float],
    rotation_deg: float,
    *,
    inside_fraction: float = INSIDE_FRACTION,
    infinite: bool = False,
) -> npt.NDArray[np.bool_]:
    """Which sampled points fall inside a well, under this hypothesis.

    Works in the plate's own frame by undoing the rotation and the translation,
    then locates each point in the lattice by arithmetic rather than by searching
    every well, which is what keeps a search over tens of thousands of hypotheses
    fast. With ``infinite`` the plate's extent is ignored (a lattice without
    edges). The rotation is written out (FM-131).
    """
    rows, cols = plate_shape(plate)
    pitch_x, pitch_y = plate_pitch_um(plate)
    well_x, well_y = well_size_um(plate)
    radius_x = well_x / 2.0 * inside_fraction
    radius_y = well_y / 2.0 * inside_fraction

    theta = math.radians(rotation_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    dx = np.asarray(xs_um, dtype=np.float64) - a1_center_xy_um[0]
    dy = np.asarray(ys_um, dtype=np.float64) - a1_center_xy_um[1]
    # Inverse rotation into plate coordinates.
    lx = cos_t * dx + sin_t * dy
    ly = -sin_t * dx + cos_t * dy

    # useq's convention: columns run +x from A1, rows run -y.
    col = np.rint(lx / pitch_x)
    row = np.rint(-ly / pitch_y)
    near_x = np.abs(lx - col * pitch_x)
    near_y = np.abs(ly + row * pitch_y)
    inside = (near_x <= radius_x) & (near_y <= radius_y)
    if not infinite:
        inside &= (col >= 0) & (col < cols) & (row >= 0) & (row < rows)
    return np.asarray(inside, dtype=np.bool_)


def lattice_score(
    xs_um: npt.ArrayLike,
    ys_um: npt.ArrayLike,
    values: npt.ArrayLike,
    plate: str,
    a1_center_xy_um: tuple[float, float],
    rotation_deg: float,
    *,
    inside_fraction: float = INSIDE_FRACTION,
    infinite: bool = False,
) -> tuple[float, Literal[1, -1], int]:
    """Correlation between measured brightness and "this point is in a well".

    Returns ``(score, polarity, n_inside)``. The score is the absolute
    correlation, so a plate whose wells read *darker* than the plastic scores
    just as well: the sign is reported rather than assumed, which is what lets
    this work on a clear plate and a black-walled one without a setting.

    The correlation is summed by hand: ``np.corrcoef`` is a ``dot``, which
    warns on one numpy only (FM-131), and it clips to [-1, 1], which a perfect
    correlation summed by hand needs (the score is bounded by ``LatticeFit``).
    Values that are all equal, or a mask that holds every point or none, have no
    correlation: 0.
    """
    inside = inside_mask(
        xs_um,
        ys_um,
        plate,
        a1_center_xy_um,
        rotation_deg,
        inside_fraction=inside_fraction,
        infinite=infinite,
    )
    return _correlation(inside, np.asarray(values, dtype=np.float64))


def _correlation(
    inside: npt.NDArray[np.bool_], values: npt.NDArray[np.float64]
) -> tuple[float, Literal[1, -1], int]:
    """``lattice_score``'s arithmetic on a mask that is already made."""
    n_in = int(inside.sum())
    n = len(values)
    if n_in == 0 or n_in == n or values.std() == 0:
        return 0.0, 1, n_in
    indicator = inside.astype(np.float64) - n_in / n
    centred = values - values.mean()
    denominator = math.sqrt(
        float(np.sum(indicator * indicator)) * float(np.sum(centred * centred))
    )
    if denominator == 0.0:
        return 0.0, 1, n_in
    r = float(np.sum(indicator * centred)) / denominator
    if not math.isfinite(r):
        return 0.0, 1, n_in
    r = max(-1.0, min(1.0, r))
    polarity: Literal[1, -1] = 1 if r >= 0.0 else -1
    return abs(r), polarity, n_in


# ------------------------------------------------------------- the search


def _phase_scan(
    xs: npt.NDArray[np.float64],
    ys: npt.NDArray[np.float64],
    vals: npt.NDArray[np.float64],
    pitch_x: float,
    pitch_y: float,
    rotations: npt.NDArray[np.float64],
    keep: int = 1,
) -> list[tuple[float, float, tuple[float, float]]]:
    """Candidate rotations and their lattice phases, strongest first.

    For a fixed rotation the lattice phase is the *Fourier phase* of the
    measurements at the lattice frequency: one complex sum per axis, exact,
    O(N). That removes two of the three search dimensions, which is what made a
    plain grid search hopeless: over a 100 mm plate a 1 degree rotation error
    displaces the far edge by a fifth of a pitch, so phase and rotation cannot
    be gridded independently at any affordable resolution.

    Fourier strength is a good *proposal* and a poor *judge*: at larger
    rotations the true angle is often not the strongest peak, so several
    well-separated candidates are returned and the real score decides between
    them. Returns ``[(strength, rotation_deg, node_xy_um), ...]``.
    """
    centred = vals - vals.mean()
    norm = float(np.abs(centred).sum()) or 1.0
    x_c, y_c = float(np.mean(xs)), float(np.mean(ys))
    dx, dy = xs - x_c, ys - y_c

    found: list[tuple[float, float, tuple[float, float]]] = []
    for theta in rotations:
        rad = math.radians(float(theta))
        cos_t, sin_t = math.cos(rad), math.sin(rad)
        u = cos_t * dx + sin_t * dy
        w = -sin_t * dx + cos_t * dy
        sx = np.sum(centred * np.exp(-2j * np.pi * u / pitch_x))
        sy = np.sum(centred * np.exp(-2j * np.pi * w / pitch_y))
        strength = float(abs(sx) * abs(sy)) / (norm * norm)
        u0 = float(np.angle(sx)) * pitch_x / (2 * math.pi)
        w0 = float(np.angle(sy)) * pitch_y / (2 * math.pi)
        node = (x_c + cos_t * u0 - sin_t * w0, y_c + sin_t * u0 + cos_t * w0)
        found.append((strength, float(theta), node))

    found.sort(key=lambda t: t[0], reverse=True)
    # Keep peaks that are actually distinct angles, not neighbours of one peak.
    picked: list[tuple[float, float, tuple[float, float]]] = []
    spread = (
        max(0.25, float(rotations.max() - rotations.min()) / 24.0)
        if len(rotations) > 1
        else 0.0
    )
    for candidate in found:
        if all(abs(candidate[1] - p[1]) > spread for p in picked):
            picked.append(candidate)
        if len(picked) >= keep:
            break
    return picked or found[:1]


def _node(
    a1_um: tuple[float, float],
    rotation_deg: float,
    pitch_x: float,
    pitch_y: float,
    row: int,
    col: int,
) -> tuple[float, float]:
    """The stage position of the lattice node ``row`` rows and ``col`` columns from ``a1_um``."""
    theta = math.radians(rotation_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    lx, ly = col * pitch_x, -row * pitch_y
    return a1_um[0] + cos_t * lx - sin_t * ly, a1_um[1] + sin_t * lx + cos_t * ly


def fit_one(
    points: Sequence[SurveyPoint],
    plate: str,
    *,
    max_rotation_deg: float = MAX_ROTATION_DEG,
    rotation_step_deg: float = 0.02,
    rotation_candidates: int = ROTATION_CANDIDATES,
) -> PlateFit:
    """The best placement of one plate type over the survey.

    A survey that is too short for the plate, too small to fit, or holds a
    value that is not finite is not an exception: the fit says so in its notes
    and scores 0 where it could not look, because the caller ranks several
    plates and a refusal of one must not end the others.

    Raises:
        ValueError: ``useq`` has no plate of this name.
    """
    n = len(points)
    xs = np.array([p.x_um for p in points], dtype=np.float64)
    ys = np.array([p.y_um for p in points], dtype=np.float64)
    vals = np.array([p.value for p in points], dtype=np.float64)
    if n < _MIN_SAMPLES:
        return PlateFit(
            plate, (0.0, 0.0), 0.0, 0.0, 1, n, 0, ("too few samples to fit a lattice",)
        )
    if not (
        np.isfinite(xs).all() and np.isfinite(ys).all() and np.isfinite(vals).all()
    ):
        return PlateFit(
            plate, (0.0, 0.0), 0.0, 0.0, 1, n, 0, ("a sample is not finite",)
        )

    pitch_x, pitch_y = plate_pitch_um(plate)
    rows, cols = plate_shape(plate)
    notes: list[str] = []
    extent_note = survey_extent_note([(p.x_um, p.y_um) for p in points], plate)
    if extent_note is not None:
        notes.append(extent_note)

    def score_of(
        a1: tuple[float, float], rotation_deg: float, *, infinite: bool = False
    ) -> tuple[float, Literal[1, -1], int]:
        return lattice_score(xs, ys, vals, plate, a1, rotation_deg, infinite=infinite)

    # Stage 1: candidate rotations, each with its phase solved exactly.
    rotations = (
        np.arange(-max_rotation_deg, max_rotation_deg + 1e-9, rotation_step_deg)
        if max_rotation_deg > 0
        else np.array([0.0])
    )
    candidates = _phase_scan(
        xs, ys, vals, pitch_x, pitch_y, rotations, keep=rotation_candidates
    )

    # Stage 1b: sharpen the phase against the INFINITE lattice, which needs no
    # idea which node is A1. The Fourier phase is biased by ~0.15 of a pitch
    # (a lattice of discs is not a sinusoid), over a millimetre at a 9 mm
    # pitch, so stage 2 would otherwise start from a wrecked position and pick
    # a node at random.
    sharpened: list[tuple[float, float, tuple[float, float]]] = []
    for _strength, rot, node in candidates:
        best_phase = (-1.0, node)
        for span, steps in ((0.5, 21), (0.06, 13), (0.01, 9)):
            c_x, c_y = best_phase[1]
            for dx in np.linspace(-pitch_x * span, pitch_x * span, steps):
                for dy in np.linspace(-pitch_y * span, pitch_y * span, steps):
                    candidate = (c_x + float(dx), c_y + float(dy))
                    score, _pol, _n = score_of(candidate, rot, infinite=True)
                    if score > best_phase[0]:
                        best_phase = (score, candidate)
        sharpened.append((best_phase[0], rot, best_phase[1]))
    sharpened.sort(key=lambda t: t[0], reverse=True)

    # Stage 2: which lattice node is A1, for each candidate rotation. This is
    # the step that uses the plate's finite extent, and so the step that tells
    # a 96-well plate from a 384-well one. A1 may be many pitches from the
    # sharpened node in either direction, so the offsets are searched both
    # ways: searching only forwards makes A1 unreachable and the fit settles
    # on a plate hanging off the corner.
    best_score, a1_x, a1_y, best_rot = -1.0, 0.0, 0.0, 0.0
    best_pol: Literal[1, -1] = 1
    n_in = 0
    for _score, rot, node in sharpened:
        for i in range(-rows, rows + 1):
            for j in range(-cols, cols + 1):
                candidate = _node(node, rot, pitch_x, pitch_y, i, j)
                score, pol, n_here = score_of(candidate, rot)
                if score > best_score:
                    best_score, a1_x, a1_y = score, candidate[0], candidate[1]
                    best_rot, best_pol, n_in = rot, pol, n_here

    # Stage 3: refine against the real score. The Fourier phase is a little
    # biased, so this is not optional.
    for span in (0.3, 0.08, 0.02):
        improved = True
        while improved:
            improved = False
            for d_rot in np.linspace(
                -max_rotation_deg * span, max_rotation_deg * span, 7
            ):
                for dx in np.linspace(-pitch_x * span, pitch_x * span, 7):
                    for dy in np.linspace(-pitch_y * span, pitch_y * span, 7):
                        if dx == 0 and dy == 0 and d_rot == 0:
                            continue
                        candidate = (a1_x + float(dx), a1_y + float(dy))
                        theta = best_rot + float(d_rot)
                        score, pol, n_here = score_of(candidate, theta)
                        if score > best_score:
                            best_score, a1_x, a1_y = score, candidate[0], candidate[1]
                            best_rot, best_pol, n_in = theta, pol, n_here
                            improved = True
    return PlateFit(
        plate,
        (float(a1_x), float(a1_y)),
        float(best_rot),
        float(best_score),
        best_pol,
        n,
        int(n_in),
        tuple(notes),
    )


def fit(
    points: Sequence[SurveyPoint], plates: Sequence[str] | None = None
) -> list[PlateFit]:
    """Rank every candidate plate type against the survey, best first.

    Identifying the plate is the same computation as locating it: whichever
    type explains the measurements best wins, and the gap to the runner-up says
    whether that was a real decision or a coin toss. ``plates=None`` is every
    registered ``-well`` plate.

    Raises:
        ValueError: ``useq`` has no plate of a name in ``plates``.
    """
    if plates is None:
        keys = sorted(str(k) for k in useq.registered_well_plate_keys())
        plates = [k for k in keys if k.endswith("-well")]
    out = [fit_one(points, plate) for plate in plates]
    out.sort(key=lambda f: f.score, reverse=True)
    return out


def decide(fits: Sequence[PlateFit]) -> tuple[PlateFit | None, tuple[str, ...]]:
    """The winner, plus what is worth saying about how clear the win was.

    ``None`` when the best fit is not usable; the notes then say why.
    """
    if not fits:
        return None, ("nothing to decide between",)
    best = fits[0]
    notes: list[str] = list(best.notes)
    if not best.usable:
        notes.append(
            f"best score {best.score:.2f} is too low to trust — the survey "
            f"may not have crossed enough wells, or brightness does not "
            f"separate wells from plastic on this plate"
        )
        return None, tuple(notes)
    if len(fits) > 1:
        runner = fits[1]
        if best.score - runner.score < DECISIVE_MARGIN:
            notes.append(
                f"{best.plate} ({best.score:.3f}) barely beats {runner.plate} "
                f"({runner.score:.3f}) — sample more finely, or say which "
                f"plate it is"
            )
    return best, tuple(notes)
