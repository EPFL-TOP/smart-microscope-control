"""The profile's ``[holder]`` table: where the stand's plate holder puts a plate (m5 §15 D2, #410).

The point is an *estimate* the map draws before any registration, so the table is held to the
standard of every value a profile gives the layer: two finite numbers, no TOML boolean
(FM-104), a bound that catches a unit mistake (FM-125), and a provenance that says where the
number comes from.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import smc.hardware.profile as profile_module
from smc.hardware.errors import ProfileError
from smc.hardware.profile import Profile

PROFILES = Path(__file__).resolve().parents[2] / "profiles"
MEASURED = '"2026-11-10, session 3 (#405)"'


def _write(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('[microscope]\nname = "t"\n' + body, encoding="utf-8")
    return path


def _holder(body: str, *, measured: str | None = MEASURED) -> str:
    """A ``[holder]`` table with ``body`` and, unless ``measured`` is ``None``, its provenance."""
    text = f"[holder]\n{body}\n"
    if measured is not None:
        text += f"measured = {measured}\n"
    return text


def test_a_profile_without_a_holder_has_no_plate_centre() -> None:
    holder = Profile.demo().holder

    assert holder.plate_centre_xy_um is None
    assert holder.measured == ""


def test_a_holder_loads_from_toml(tmp_path: Path) -> None:
    path = _write(tmp_path / "p.toml", _holder("plate_centre_xy_um = [400.0, -325.0]"))

    holder = Profile.load(path).holder

    assert holder.plate_centre_xy_um == (400.0, -325.0)
    assert holder.measured == "2026-11-10, session 3 (#405)"


def test_a_plate_centre_without_measured_is_refused(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "p.toml",
        _holder("plate_centre_xy_um = [400.0, -325.0]", measured=None),
    )

    with pytest.raises(ProfileError) as info:
        Profile.load(path)

    message = str(info.value)
    assert "plate_centre_xy_um" in message
    assert "measured" in message


@pytest.mark.parametrize("measured", ['""', '"   "'])
def test_a_blank_measured_is_refused(tmp_path: Path, measured: str) -> None:
    path = _write(
        tmp_path / "p.toml",
        _holder("plate_centre_xy_um = [400.0, -325.0]", measured=measured),
    )

    with pytest.raises(ProfileError) as info:
        Profile.load(path)

    assert "needs measured" in str(info.value)


def test_a_plate_centre_beyond_the_bound_is_refused_and_at_the_bound_is_accepted(
    tmp_path: Path,
) -> None:
    assert profile_module.MAX_HOLDER_CENTRE_UM == 150_000.0

    at_the_bound = _write(
        tmp_path / "at.toml", _holder("plate_centre_xy_um = [150000.0, -150000.0]")
    )
    assert Profile.load(at_the_bound).holder.plate_centre_xy_um == (
        150_000.0,
        -150_000.0,
    )

    for value in ("[150001.0, 0.0]", "[0.0, -150001.0]"):
        beyond = _write(
            tmp_path / "beyond.toml", _holder(f"plate_centre_xy_um = {value}")
        )
        with pytest.raises(ProfileError) as info:
            Profile.load(beyond)
        message = str(info.value)
        assert "holder.plate_centre_xy_um" in message
        assert "a unit mistake" in message


@pytest.mark.parametrize(
    "value",
    [
        "[1.0]",
        "[true, 2.0]",
        "[1.0, false]",
        "[nan, 1.0]",
        "[1.0, inf]",
        "[1.0, 2.0, 3.0]",
    ],
)
def test_a_non_finite_boolean_or_short_plate_centre_is_refused(
    tmp_path: Path, value: str
) -> None:
    path = _write(tmp_path / "p.toml", _holder(f"plate_centre_xy_um = {value}"))

    with pytest.raises(ProfileError) as info:
        Profile.load(path)

    assert "holder.plate_centre_xy_um" in str(info.value)


def test_a_boolean_plate_centre_says_it_is_not_a_number(tmp_path: Path) -> None:
    # pydantic would read `true` as 1.0 µm, a plausible position (FM-104).
    path = _write(tmp_path / "p.toml", _holder("plate_centre_xy_um = [true, 2.0]"))

    with pytest.raises(ProfileError) as info:
        Profile.load(path)

    assert "not a boolean" in str(info.value)


def test_an_unknown_holder_key_is_refused(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "p.toml",
        _holder("plate_center_xy_um = [1.0, 2.0]", measured=MEASURED),
    )

    with pytest.raises(ProfileError) as info:
        Profile.load(path)

    assert "plate_center_xy_um" in str(info.value)


def test_the_shipped_profiles_say_where_their_plate_lies() -> None:
    nikon = Profile.load(PROFILES / "nikon-ti2.toml").holder
    assert nikon.plate_centre_xy_um == (400.0, -325.0)
    # The provenance names the two sessions the mean comes from.
    assert "sessions 2 and 3" in nikon.measured

    sample = Profile.load(PROFILES / "demo-sample.toml").holder
    assert sample.plate_centre_xy_um == (0.0, 0.0)
    assert sample.measured.startswith("by construction")

    # A stand nobody measured says nothing: its plate is unplaced, not guessed.
    assert Profile.load(PROFILES / "demo.toml").holder.plate_centre_xy_um is None
    assert Profile.load(PROFILES / "nikon-tie.toml").holder.plate_centre_xy_um is None


# --- which stage direction is the back of the stand and the operator's right (m5 §16 D1)


def test_holder_directions_need_both_and_their_provenance(tmp_path: Path) -> None:
    both = _write(
        tmp_path / "both.toml",
        '[holder]\nback = "-X"\nright = "+Y"\n'
        'directions_measured = "2026-11-11, session 4 (#422)"\n',
    )
    holder = Profile.load(both).holder
    assert (holder.back, holder.right) == ("-X", "+Y")
    assert holder.directions_measured == "2026-11-11, session 4 (#422)"
    # The directions need no centre, and the centre's provenance is not theirs.
    assert holder.plate_centre_xy_um is None

    for body, words in (
        ('back = "+Y"\n', "both"),
        ('right = "+X"\n', "both"),
        ('back = "+Y"\nright = "+X"\n', "directions_measured"),
        (
            'back = "+Y"\nright = "+X"\ndirections_measured = "  "\n',
            "directions_measured",
        ),
    ):
        path = _write(tmp_path / "one.toml", f"[holder]\n{body}")
        with pytest.raises(ProfileError) as info:
            Profile.load(path)
        assert words in str(info.value), body

    # A provenance alone, with no direction, says nothing and is accepted.
    alone = _write(tmp_path / "alone.toml", '[holder]\ndirections_measured = "x"\n')
    assert Profile.load(alone).holder.back is None


def test_holder_directions_on_one_axis_are_refused(tmp_path: Path) -> None:
    for back, right in (("+Y", "-Y"), ("+X", "+X"), ("-X", "+X")):
        path = _write(
            tmp_path / "p.toml",
            f'[holder]\nback = "{back}"\nright = "{right}"\n'
            'directions_measured = "2026-11-11"\n',
        )
        with pytest.raises(ProfileError) as info:
            Profile.load(path)
        assert "different axes" in str(info.value), (back, right)

    # A word that is no stage direction is refused by its type.
    path = _write(
        tmp_path / "p.toml",
        '[holder]\nback = "up"\nright = "+X"\ndirections_measured = "x"\n',
    )
    with pytest.raises(ProfileError) as info:
        Profile.load(path)
    assert "holder.back" in str(info.value)


def test_the_ti2_holder_says_back_is_plus_y_and_right_is_plus_x() -> None:
    nikon = Profile.load(PROFILES / "nikon-ti2.toml").holder
    assert (nikon.back, nikon.right) == ("+Y", "+X")
    # Inferred from session 3, and says it is not checked at the stand.
    assert "session 3" in nikon.directions_measured
    assert "not checked at the stand" in nikon.directions_measured

    sample = Profile.load(PROFILES / "demo-sample.toml").holder
    assert (sample.back, sample.right) == ("+Y", "+X")
    assert sample.directions_measured.startswith("by construction")

    # A stand nobody looked at says neither: the page then speaks in stage terms.
    for name in ("demo.toml", "nikon-tie.toml"):
        holder = Profile.load(PROFILES / name).holder
        assert (holder.back, holder.right) == (None, None), name
