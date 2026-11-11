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
