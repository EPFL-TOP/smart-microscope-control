from __future__ import annotations

import os
from pathlib import Path

import pytest

from smc.hardware.errors import HardwareError, ProfileError
from smc.hardware.profile import Profile, list_profiles, search_paths
from smc.hardware.roles import Role

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_TOML = REPO_ROOT / "profiles" / "demo.toml"


def write_profile(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('[microscope]\nname = "t"\n' + body, encoding="utf-8")
    return path


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty working directory with no SMC_PROFILES, so the repo's profiles/ is not seen."""
    monkeypatch.delenv("SMC_PROFILES", raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


# -- demo ---------------------------------------------------------------------


def test_demo_toml_equals_profile_demo() -> None:
    loaded = Profile.load(DEMO_TOML)
    assert loaded.model_dump() == Profile.demo().model_dump()
    assert loaded.source == DEMO_TOML.resolve()


def test_load_demo_by_name_finds_the_repo_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SMC_PROFILES", raising=False)
    monkeypatch.chdir(REPO_ROOT)
    loaded = Profile.load("demo")
    assert loaded.source == DEMO_TOML.resolve()
    assert loaded.model_dump() == Profile.demo().model_dump()


def test_demo_name_without_file_falls_back_to_code(isolated: Path) -> None:
    loaded = Profile.load("demo")
    assert loaded.source is None
    assert loaded.model_dump() == Profile.demo().model_dump()


def test_demo_uses_the_shipped_demo_configuration() -> None:
    assert Profile.demo().config_path() is None


def test_source_is_not_serialised() -> None:
    assert "source" not in Profile.load(DEMO_TOML).model_dump()


def test_role_keys_dump_as_their_string_values(tmp_path: Path) -> None:
    path = write_profile(
        tmp_path / "p.toml",
        '[roles.assign]\nxy_stage = "XY"\n[roles.exclude]\nfocus = ["piezo"]\n',
    )
    profile = Profile.load(path)
    assert profile.roles.assign == {Role.xy_stage: "XY"}
    dumped = profile.model_dump(mode="json")
    assert dumped["roles"] == {
        "assign": {"xy_stage": "XY"},
        "exclude": {"focus": ["piezo"]},
    }


# -- validation ---------------------------------------------------------------


def test_profile_error_is_a_hardware_error() -> None:
    assert issubclass(ProfileError, HardwareError)


def test_unknown_role_key_lists_allowed_roles(tmp_path: Path) -> None:
    path = write_profile(tmp_path / "p.toml", '[roles.assign]\nxy_stag = "XY"\n')
    with pytest.raises(ProfileError) as info:
        Profile.load(path)
    msg = str(info.value)
    assert str(path) in msg
    assert "roles.assign.xy_stag" in msg
    for role in Role:
        assert role.value in msg


def test_unknown_section_key_is_an_error(tmp_path: Path) -> None:
    path = write_profile(tmp_path / "p.toml", "[safety]\nmax_jog = 10.0\n")
    with pytest.raises(ProfileError) as info:
        Profile.load(path)
    msg = str(info.value)
    assert str(path) in msg
    assert "safety.max_jog" in msg
    assert "max_jog_um" in msg  # the fix: the keys that do exist


def test_unknown_top_level_section_is_an_error(tmp_path: Path) -> None:
    path = write_profile(tmp_path / "p.toml", "[saftey]\nmax_jog_um = 10.0\n")
    with pytest.raises(ProfileError, match=r"saftey[\s\S]*safety"):
        Profile.load(path)


def test_source_cannot_be_set_from_the_file(tmp_path: Path) -> None:
    path = write_profile(tmp_path / "p.toml", "")
    path.write_text(
        'source = "elsewhere"\n' + path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    with pytest.raises(ProfileError, match="source"):
        Profile.load(path)


def test_missing_microscope_section_says_what_to_add(tmp_path: Path) -> None:
    path = tmp_path / "p.toml"
    path.write_text("[safety]\nmax_jog_um = 10.0\n", encoding="utf-8")
    with pytest.raises(ProfileError, match=r"microscope[\s\S]*add"):
        Profile.load(path)


@pytest.mark.parametrize(
    ("body", "key"),
    [
        ("[safety]\nz_soft_limits_um = [10.0, -10.0]\n", "safety.z_soft_limits_um"),
        ("[safety]\nz_soft_limits_um = [5.0, 5.0]\n", "safety.z_soft_limits_um"),
        (
            "[safety]\nxy_soft_limits_um = [[-1.0, 1.0], [3.0, 2.0]]\n",
            "safety.xy_soft_limits_um",
        ),
    ],
)
def test_unordered_limits_are_rejected(tmp_path: Path, body: str, key: str) -> None:
    path = write_profile(tmp_path / "p.toml", body)
    with pytest.raises(ProfileError) as info:
        Profile.load(path)
    msg = str(info.value)
    assert str(path) in msg
    assert key in msg
    assert "low < high" in msg


def test_ordered_limits_are_accepted(tmp_path: Path) -> None:
    path = write_profile(
        tmp_path / "p.toml",
        "[safety]\nz_soft_limits_um = [-5.0, 5.0]\n"
        "xy_soft_limits_um = [[-1.0, 1.0], [-2.0, 2.0]]\n",
    )
    safety = Profile.load(path).safety
    assert safety.z_soft_limits_um == (-5.0, 5.0)
    assert safety.xy_soft_limits_um == ((-1.0, 1.0), (-2.0, 2.0))


@pytest.mark.parametrize(
    ("body", "key"),
    [
        ("[safety]\nz_soft_limits_um = [nan, 1000.0]\n", "safety.z_soft_limits_um"),
        (
            "[safety]\nxy_soft_limits_um = [[-1.0, inf], [-2.0, 2.0]]\n",
            "safety.xy_soft_limits_um",
        ),
    ],
)
def test_non_finite_soft_limits_are_rejected(
    tmp_path: Path, body: str, key: str
) -> None:
    # inf passes "low < high" outright; nan is only rejected today because a
    # nan comparison is always False, which makes the message blame ordering
    # instead of the real problem. Either way this must name the real problem.
    path = write_profile(tmp_path / "p.toml", body)
    with pytest.raises(ProfileError) as info:
        Profile.load(path)
    msg = str(info.value)
    assert str(path) in msg
    assert key in msg
    assert "finite" in msg


@pytest.mark.parametrize("value", ["nan", "inf", "0", "-5"])
def test_profile_rejects_non_finite_or_non_positive_jog_limit(
    tmp_path: Path, value: str
) -> None:
    path = write_profile(tmp_path / "p.toml", f"[safety]\nmax_jog_um = {value}\n")
    with pytest.raises(ProfileError) as info:
        Profile.load(path)
    msg = str(info.value)
    assert str(path) in msg
    assert "safety.max_jog_um" in msg


def test_wrong_type_names_the_key(tmp_path: Path) -> None:
    path = write_profile(tmp_path / "p.toml", '[safety]\nmax_jog_um = "far"\n')
    with pytest.raises(ProfileError, match=r"safety\.max_jog_um"):
        Profile.load(path)


def test_invalid_toml_names_the_file(tmp_path: Path) -> None:
    path = tmp_path / "p.toml"
    path.write_text("[microscope\nname = 1\n", encoding="utf-8")
    with pytest.raises(ProfileError) as info:
        Profile.load(path)
    assert str(path) in str(info.value)
    assert "TOML" in str(info.value)


def test_missing_file_is_a_profile_error(tmp_path: Path) -> None:
    path = tmp_path / "nope.toml"
    with pytest.raises(ProfileError, match=r"nope\.toml"):
        Profile.load(path)


def test_quirks_are_free_form(tmp_path: Path) -> None:
    path = write_profile(
        tmp_path / "p.toml", '[quirks]\npfs_settle_ms = 300\nnested = { a = "b" }\n'
    )
    assert Profile.load(path).quirks == {"pfs_settle_ms": 300, "nested": {"a": "b"}}


@pytest.mark.parametrize("value", ["nan", "inf", "0.0", "-1.0"])
def test_profile_rejects_non_positive_pixel_size(tmp_path: Path, value: str) -> None:
    path = write_profile(
        tmp_path / "p.toml",
        f'[camera]\npixel_size_um = {{ "Nikon 10X S Fluor" = {value} }}\n',
    )
    with pytest.raises(ProfileError) as info:
        Profile.load(path)
    msg = str(info.value)
    assert str(path) in msg
    assert "pixel_size_um" in msg
    assert "Nikon 10X S Fluor" in msg


@pytest.mark.parametrize("value", ["0", "-1"])
def test_device_timeout_must_be_positive(tmp_path: Path, value: str) -> None:
    path = write_profile(
        tmp_path / "p.toml", f"[micromanager]\ndevice_timeout_ms = {value}\n"
    )
    with pytest.raises(ProfileError) as info:
        Profile.load(path)
    msg = str(info.value)
    assert str(path) in msg
    assert "micromanager.device_timeout_ms" in msg


# -- micromanager.config ------------------------------------------------------


def test_relative_config_resolves_against_the_profile_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stand = tmp_path / "stand"
    (stand / "cfg").mkdir(parents=True)
    (stand / "cfg" / "scope.cfg").write_text("# cfg\n", encoding="utf-8")
    path = write_profile(stand / "p.toml", '[micromanager]\nconfig = "cfg/scope.cfg"\n')
    monkeypatch.chdir(tmp_path)  # a different directory than the profile's
    profile = Profile.load(path)
    assert profile.config_path() == (stand / "cfg" / "scope.cfg").resolve()


def test_absolute_config_is_kept(tmp_path: Path) -> None:
    cfg = tmp_path / "abs.cfg"
    cfg.write_text("# cfg\n", encoding="utf-8")
    path = write_profile(
        tmp_path / "sub" / "p.toml", f"[micromanager]\nconfig = '{cfg.as_posix()}'\n"
    )
    assert Profile.load(path).config_path() == cfg


def test_missing_config_names_the_path(tmp_path: Path) -> None:
    path = write_profile(
        tmp_path / "p.toml", '[micromanager]\nconfig = "missing.cfg"\n'
    )
    with pytest.raises(ProfileError) as info:
        Profile.load(path)
    msg = str(info.value)
    assert str(path) in msg
    assert "micromanager.config" in msg
    assert str(tmp_path / "missing.cfg") in msg


# -- search paths -------------------------------------------------------------


def test_search_paths_default_to_profiles_then_cwd(isolated: Path) -> None:
    assert search_paths() == [isolated / "profiles", isolated]


def test_search_paths_honour_smc_profiles_env(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = isolated / "a", isolated / "b"
    monkeypatch.setenv("SMC_PROFILES", os.pathsep.join([str(first), "", str(second)]))
    assert search_paths() == [first, second, isolated / "profiles", isolated]


def test_bare_name_resolves_over_search_paths(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_dir = isolated / "site"
    write_profile(isolated / "profiles" / "stand.toml", "")
    env_profile = write_profile(env_dir / "stand.toml", "")
    local_only = write_profile(isolated / "profiles" / "local.toml", "")

    assert (
        Profile.load("stand").source == (isolated / "profiles" / "stand.toml").resolve()
    )
    monkeypatch.setenv("SMC_PROFILES", str(env_dir))
    assert Profile.load("stand").source == env_profile.resolve()  # env wins
    assert Profile.load("local").source == local_only.resolve()


def test_demo_toml_on_the_search_path_wins_over_code(isolated: Path) -> None:
    path = write_profile(isolated / "profiles" / "demo.toml", "")
    loaded = Profile.load("demo")
    assert loaded.source == path.resolve()
    assert loaded.microscope.name == "t"


def test_unknown_bare_name_lists_where_it_looked(isolated: Path) -> None:
    write_profile(isolated / "profiles" / "other.toml", "")
    with pytest.raises(ProfileError) as info:
        Profile.load("ghost")
    msg = str(info.value)
    assert "ghost" in msg
    assert str(isolated / "profiles") in msg
    assert "other" in msg
    assert "SMC_PROFILES" in msg


def test_list_profiles_finds_demo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SMC_PROFILES", raising=False)
    monkeypatch.chdir(REPO_ROOT)
    assert ("demo", (REPO_ROOT / "profiles" / "demo.toml").resolve()) in list_profiles()


def test_list_profiles_skips_toml_that_is_not_a_profile(isolated: Path) -> None:
    (isolated / "pyproject.toml").write_text(
        '[project]\nname = "x"\n', encoding="utf-8"
    )
    broken = isolated / "broken.toml"
    broken.write_text("[microscope\n", encoding="utf-8")
    assert list_profiles() == [("broken", broken.resolve())]


def test_list_profiles_first_search_path_wins(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_dir = isolated / "site"
    write_profile(isolated / "profiles" / "stand.toml", "")
    write_profile(isolated / "profiles" / "zeiss.toml", "")
    env_profile = write_profile(env_dir / "stand.toml", "")
    monkeypatch.setenv("SMC_PROFILES", str(env_dir))
    assert list_profiles() == [
        ("stand", env_profile.resolve()),
        ("zeiss", (isolated / "profiles" / "zeiss.toml").resolve()),
    ]
