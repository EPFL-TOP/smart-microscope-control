"""The frames of a wall: which stage side an image side is, and what a step does (m3 §19).

The table is **written out** here and never computed with ``frames.py`` (FM-124):
each camera is a matrix ``M = ps * R(theta) * diag(+-1, -1)`` (the convention of
``smc.testing.synthetic``, ``mirrored`` negating the first factor), written as its
four numbers, and each row gives the stage side of the image's four sides and what
a +X and a +Y step do to the image.
"""

from __future__ import annotations

import ast
import math
import re
from pathlib import Path

import pytest

import smc
from smc.plugins.frames import (
    QUARTER_TURN_TOLERANCE_DEG,
    image_direction_of_stage_step,
    stage_side_of_image_side,
)
from smc.plugins.geometry import Matrix2
from smc.plugins.slots import IMAGE_SIDES, ImageSide

#: name, matrix, (left, right, top, bottom) as stage sides, (+X step, +Y step) as
#: what the image content does. Each matrix has its own pixel size, so that no
#: row can pass by reading a magnitude.
_PS = 0.275
_TABLE: list[tuple[str, Matrix2, tuple[str, str, str, str], tuple[str, str]]] = [
    ("0", ((_PS, 0.0), (0.0, -_PS)), ("-X", "+X", "+Y", "-Y"), ("left", "down")),
    (
        "0 mirrored",
        ((-_PS, 0.0), (0.0, -_PS)),
        ("+X", "-X", "+Y", "-Y"),
        ("right", "down"),
    ),
    ("90", ((0.0, 2 * _PS), (2 * _PS, 0.0)), ("-Y", "+Y", "-X", "+X"), ("up", "left")),
    (
        "90 mirrored",
        ((0.0, 2 * _PS), (-2 * _PS, 0.0)),
        ("+Y", "-Y", "-X", "+X"),
        ("up", "right"),
    ),
    (
        "180",
        ((-1.1, 0.0), (0.0, 1.1)),
        ("+X", "-X", "-Y", "+Y"),
        ("right", "up"),
    ),
    (
        "180 mirrored",
        ((3.0, 0.0), (0.0, 3.0)),
        ("-X", "+X", "-Y", "+Y"),
        ("left", "up"),
    ),
    (
        "-90",
        ((0.0, -0.5), (-0.5, 0.0)),
        ("+Y", "-Y", "+X", "-X"),
        ("down", "right"),
    ),
    (
        "-90 mirrored",
        ((0.0, -4.0), (4.0, 0.0)),
        ("-Y", "+Y", "+X", "-X"),
        ("down", "left"),
    ),
]
_IDS = [row[0] for row in _TABLE]
_SIDE_ORDER: tuple[ImageSide, ...] = ("left", "right", "top", "bottom")


def _camera_at(theta_deg: float, ps: float = 1.1) -> Matrix2:
    """``ps * R(theta) * diag(1, -1)``, an upright camera turned by ``theta_deg``."""
    c, s = math.cos(math.radians(theta_deg)), math.sin(math.radians(theta_deg))
    return ((ps * c, ps * s), (ps * s, -ps * c))


@pytest.mark.parametrize(("name", "matrix", "sides", "steps"), _TABLE, ids=_IDS)
def test_each_image_side_is_the_stage_side_of_the_table(
    name: str,
    matrix: Matrix2,
    sides: tuple[str, str, str, str],
    steps: tuple[str, str],
) -> None:
    got = tuple(stage_side_of_image_side(side, matrix) for side in _SIDE_ORDER)
    assert got == sides, name


@pytest.mark.parametrize(("name", "matrix", "sides", "steps"), _TABLE, ids=_IDS)
def test_each_stage_step_moves_the_image_as_the_table_says(
    name: str,
    matrix: Matrix2,
    sides: tuple[str, str, str, str],
    steps: tuple[str, str],
) -> None:
    got = (
        image_direction_of_stage_step("+X", matrix),
        image_direction_of_stage_step("+Y", matrix),
    )
    assert got == steps, name
    # The opposite step moves the image the opposite way.
    opposite = {"left": "right", "right": "left", "up": "down", "down": "up"}
    assert image_direction_of_stage_step("-X", matrix) == opposite[steps[0]], name
    assert image_direction_of_stage_step("-Y", matrix) == opposite[steps[1]], name


def test_the_ti2e_matrix_puts_the_image_s_left_on_plus_x() -> None:
    # Ti2-E session 2 at 10x: the image's right is stage -X and its bottom +Y, and
    # an arrow moved the image the way it points (the owner's note, #327).
    ti2e: Matrix2 = ((-1.1, 0.0), (0.0, 1.1))
    assert stage_side_of_image_side("left", ti2e) == "+X"
    assert stage_side_of_image_side("right", ti2e) == "-X"
    assert stage_side_of_image_side("top", ti2e) == "-Y"
    assert stage_side_of_image_side("bottom", ti2e) == "+Y"
    assert image_direction_of_stage_step("+Y", ti2e) == "up"
    assert image_direction_of_stage_step("+X", ti2e) == "right"


def test_an_anisotropic_matrix_is_read_by_its_columns() -> None:
    # No pixel size is used: a side is the axis of the larger component of a
    # column of the matrix, whatever its length (FM-73).
    swapped: Matrix2 = ((0.0, 0.5), (1.0, 0.0))
    assert [stage_side_of_image_side(s, swapped) for s in _SIDE_ORDER] == [
        "-Y",
        "+Y",
        "-X",
        "+X",
    ]
    # M^-1 = ((0, 1), (2, 0)): +X gives (0, 2) and the image moves (0, -2); +Y
    # gives (1, 0) and the image moves (-1, 0).
    assert image_direction_of_stage_step("+X", swapped) == "up"
    assert image_direction_of_stage_step("+Y", swapped) == "left"
    # A frame whose columns are 4 times as wide as its rows are tall.
    wide: Matrix2 = ((-2.0, 0.0), (0.0, 0.5))
    assert [stage_side_of_image_side(s, wide) for s in _SIDE_ORDER] == [
        "+X",
        "-X",
        "-Y",
        "+Y",
    ]
    assert image_direction_of_stage_step("+X", wide) == "right"
    assert image_direction_of_stage_step("+Y", wide) == "up"
    # A little shear changes nothing as long as an axis dominates.
    sheared: Matrix2 = ((-1.1, 0.05), (0.02, 1.1))
    assert stage_side_of_image_side("left", sheared) == "+X"
    assert stage_side_of_image_side("bottom", sheared) == "+Y"


@pytest.mark.parametrize("base", [0.0, 90.0, 180.0, -90.0])
def test_a_camera_4_degrees_off_a_quarter_turn_is_read_and_6_degrees_is_refused(
    base: float,
) -> None:
    near = _camera_at(base + 4.0)
    exact = _camera_at(base)
    for side in IMAGE_SIDES:
        assert stage_side_of_image_side(side, near) == stage_side_of_image_side(
            side, exact
        )
    for step in ("+X", "-X", "+Y", "-Y"):
        assert image_direction_of_stage_step(step, near) == (
            image_direction_of_stage_step(step, exact)
        )
    # The other sense of the same turn, 4 degrees the other way.
    assert stage_side_of_image_side("right", _camera_at(base - 4.0)) == (
        stage_side_of_image_side("right", exact)
    )

    far = _camera_at(base + 6.0)
    for side in IMAGE_SIDES:
        with pytest.raises(ValueError, match=r"turned 6\.0° away"):
            stage_side_of_image_side(side, far)
    for step in ("+X", "-X", "+Y", "-Y"):
        with pytest.raises(ValueError, match=r"turned 6\.0° away"):
            image_direction_of_stage_step(step, far)


def test_a_camera_off_a_quarter_turn_says_how_far() -> None:
    assert QUARTER_TURN_TOLERANCE_DEG == 5.0
    text = (
        "the camera is turned 30.0° away from the stage's axes, a quarter turn at "
        "most 5° off: an arrow moves the image diagonally and an image side is "
        "not a stage side"
    )
    with pytest.raises(ValueError, match=re.escape(text)):
        stage_side_of_image_side("left", _camera_at(30.0))
    with pytest.raises(ValueError, match=r"turned 45\.0° away"):
        image_direction_of_stage_step("+X", _camera_at(45.0))


_NAN = float("nan")
_INF = float("inf")


@pytest.mark.parametrize(
    "matrix",
    [
        ((_NAN, 0.0), (0.0, 1.0)),
        ((1.0, 0.0), (0.0, _NAN)),
        ((_INF, 0.0), (0.0, 1.0)),
        ((1.0, 2.0), (2.0, 4.0)),
        ((0.0, 0.0), (0.0, 0.0)),
        ((1e-7, 0.0), (0.0, 1e-7)),
        # Finite entries whose determinant overflows to inf: it passes `>= 1e-12`
        # and a zero vector then reads as 0 degrees off an axis (adversarial review).
        ((1e200, 0.0), (0.0, 1e200)),
    ],
    ids=["nan-00", "nan-11", "inf", "singular", "zero", "tiny", "overflow"],
)
def test_a_nan_or_singular_matrix_is_refused(matrix: Matrix2) -> None:
    # `not abs(det) >= 1e-12`: a nan must be refused as well as a zero (FM-104).
    with pytest.raises(ValueError, match="cannot be inverted"):
        stage_side_of_image_side("left", matrix)
    with pytest.raises(ValueError, match="run camera-calibration"):
        image_direction_of_stage_step("+X", matrix)


@pytest.mark.parametrize("pixel_um", [0.02, 0.002])
def test_a_small_but_real_pixel_size_is_read_not_refused(pixel_um: float) -> None:
    # The bound on the determinant is for a camera that images nothing, not for a
    # fine one: 0.02 µm per pixel is a determinant of 4e-4, 0.002 one of 4e-6. Only
    # the refusing side was tested, so a bound of 1e-3 passed unseen.
    ti2e_like: Matrix2 = ((-pixel_um, 0.0), (0.0, pixel_um))
    assert stage_side_of_image_side("left", ti2e_like) == "+X"
    assert stage_side_of_image_side("bottom", ti2e_like) == "+Y"
    assert image_direction_of_stage_step("+X", ti2e_like) == "right"
    assert image_direction_of_stage_step("+Y", ti2e_like) == "up"


def test_a_matrix_whose_two_axes_lie_on_one_stage_axis_is_refused() -> None:
    # Both columns within 5 degrees of the X axis: the determinant (-0.01) is above
    # the bound, so only the columns can tell that this is no camera. Without the
    # refusal `left` and `top` both name -X, and a second capture replaces the first.
    flat: Matrix2 = ((1.0, 1.0), (0.05, 0.04))
    for side in IMAGE_SIDES:
        with pytest.raises(ValueError, match=r"both lie along the stage's X axis"):
            stage_side_of_image_side(side, flat)
    for step in ("+X", "-X", "+Y", "-Y"):
        with pytest.raises(ValueError, match=r"both lie along the stage's X axis"):
            image_direction_of_stage_step(step, flat)
    tall: Matrix2 = ((0.05, 0.04), (1.0, 1.0))
    with pytest.raises(ValueError, match=r"both lie along the stage's Y axis"):
        stage_side_of_image_side("right", tall)


def test_frames_takes_nothing_from_the_hardware_layer() -> None:
    # It is pure arithmetic on the camera matrix (m3 §19.3): a plugin or the page
    # imports it without loading a backend.
    source = (Path(smc.__file__).parent / "plugins" / "frames.py").read_text(
        encoding="utf-8"
    )
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert not [
        name for name in imported if name.startswith(("smc.hardware", "pymmcore"))
    ]
    assert "smc.plugins.geometry" in imported
    assert "smc.plugins.slots.plate" in imported
