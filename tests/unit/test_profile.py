from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

import smc.hardware.backends.mm as mm
import smc.hardware.profile as profile_module
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
    # The exact phrase from _check_ordered's finite check, not a bare "finite":
    # tmp_path itself contains "non_finite" (this test's own name), so a lone
    # "finite" substring would pass even if the isfinite check were removed.
    assert "must be finite numbers" in msg


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


@pytest.mark.parametrize("value", ["0.0", "-1.0", "nan", "inf"])
def test_max_z_jog_um_must_be_positive_and_finite(tmp_path: Path, value: str) -> None:
    path = write_profile(tmp_path / "p.toml", f"[safety]\nmax_z_jog_um = {value}\n")
    with pytest.raises(ProfileError) as info:
        Profile.load(path)
    msg = str(info.value)
    assert str(path) in msg
    assert "safety.max_z_jog_um" in msg
    # The exact phrase and the value, not a bare "finite" (FM-45).
    assert f"max_z_jog_um must be a positive, finite number of µm, got {value}" in msg


def test_omitted_max_z_jog_um_loads_as_none(tmp_path: Path) -> None:
    # "Omitted" stays visible in the profile; Safety alone applies the default.
    path = write_profile(tmp_path / "p.toml", "[safety]\nmax_jog_um = 10.0\n")
    assert Profile.load(path).safety.max_z_jog_um is None


def test_max_z_jog_um_is_read_from_the_profile(tmp_path: Path) -> None:
    path = write_profile(tmp_path / "p.toml", "[safety]\nmax_z_jog_um = 25.0\n")
    assert Profile.load(path).safety.max_z_jog_um == 25.0


def test_demo_profile_sets_max_z_jog_um() -> None:
    assert Profile.demo().safety.max_z_jog_um == 100.0


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


def test_device_timeout_must_fit_mmcore_on_windows(tmp_path: Path) -> None:
    # MMCore's setTimeoutMs takes a C long, which is 32-bit on Windows: a
    # larger value passes on Linux and macOS and overflows on the stand's PC.
    fits = write_profile(
        tmp_path / "fits.toml", "[micromanager]\ndevice_timeout_ms = 2147483647\n"
    )
    assert Profile.load(fits).micromanager.device_timeout_ms == 2_147_483_647
    path = write_profile(
        tmp_path / "p.toml", "[micromanager]\ndevice_timeout_ms = 2147483648\n"
    )
    with pytest.raises(ProfileError) as info:
        Profile.load(path)
    msg = str(info.value)
    assert "micromanager.device_timeout_ms: device_timeout_ms must be at most " in msg
    assert "2147483647 ms" in msg


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


@pytest.mark.parametrize(
    ("key", "line", "entry"),
    [
        ("config", 'config = "~nosuchuser/stand.cfg"', "~nosuchuser/stand.cfg"),
        (
            "adapter_search_paths",
            'adapter_search_paths = ["~nosuchuser/mm"]',
            "~nosuchuser/mm",
        ),
    ],
)
def test_a_home_directory_that_cannot_be_expanded_is_a_profile_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str, line: str, entry: str
) -> None:
    path = write_profile(tmp_path / "p.toml", f"[micromanager]\n{line}\n")

    def no_home(self: Path) -> Path:
        raise RuntimeError("Could not determine home directory.")

    # "~nosuchuser" does this on POSIX; Windows has no such failure to provoke.
    # A bare RuntimeError would end `smc profiles` at the first such file.
    monkeypatch.setattr(Path, "expanduser", no_home)
    with pytest.raises(ProfileError) as info:
        Profile.load(path)
    msg = str(info.value)
    assert f"{path}: micromanager.{key}: cannot expand '~' in {entry} " in msg
    assert "Could not determine home directory." in msg


# -- micromanager.adapter_search_paths (#77) ------------------------------------


def test_adapter_search_dirs_resolve_against_the_profile_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stand = tmp_path / "stand"
    (stand / "adapters").mkdir(parents=True)
    absolute = tmp_path / "mm-2.0"
    absolute.mkdir()
    path = write_profile(
        stand / "p.toml",
        "[micromanager]\n"
        f"adapter_search_paths = ['{absolute.as_posix()}', 'adapters', "
        f"'{absolute.as_posix()}']\n",
    )
    monkeypatch.chdir(tmp_path)  # a different directory than the profile's
    dirs = Profile.load(path).adapter_search_dirs()
    # The file's order is kept and nothing is de-duplicated here: that is
    # open_core's job, once it knows what the core already has.
    assert dirs == [absolute, (stand / "adapters").resolve(), absolute]


def test_adapter_search_dirs_expand_the_home_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    (home / "mm").mkdir(parents=True)
    # expanduser reads HOME on POSIX and USERPROFILE on Windows.
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    path = write_profile(
        tmp_path / "p.toml", "[micromanager]\nadapter_search_paths = ['~/mm']\n"
    )
    assert Profile.load(path).adapter_search_dirs() == [home / "mm"]


def test_no_adapter_search_paths_means_no_dirs() -> None:
    assert Profile.demo().adapter_search_dirs() == []


def test_missing_adapter_search_path_is_refused_at_load(tmp_path: Path) -> None:
    path = write_profile(
        tmp_path / "p.toml", '[micromanager]\nadapter_search_paths = ["nowhere"]\n'
    )
    with pytest.raises(ProfileError) as info:
        Profile.load(path)
    msg = str(info.value)
    assert f"{path}: micromanager.adapter_search_paths lists nowhere, " in msg
    assert "which is not a directory" in msg
    # A relative entry also says where it was looked for.
    assert str(tmp_path / "nowhere") in msg


def test_the_first_bad_adapter_search_path_is_the_one_reported(
    tmp_path: Path,
) -> None:
    (tmp_path / "fine").mkdir()
    path = write_profile(
        tmp_path / "p.toml",
        '[micromanager]\nadapter_search_paths = ["fine", "first-bad", "second-bad"]\n',
    )
    with pytest.raises(ProfileError) as info:
        Profile.load(path)
    msg = str(info.value)
    assert "micromanager.adapter_search_paths lists first-bad, " in msg
    assert "second-bad" not in msg


def test_adapter_search_path_that_is_a_file_is_refused(tmp_path: Path) -> None:
    (tmp_path / "mmgr_dal_DemoCamera.dll").write_text("not a dll\n", encoding="utf-8")
    path = write_profile(
        tmp_path / "p.toml",
        '[micromanager]\nadapter_search_paths = ["mmgr_dal_DemoCamera.dll"]\n',
    )
    with pytest.raises(ProfileError) as info:
        Profile.load(path)
    msg = str(info.value)
    assert "micromanager.adapter_search_paths lists mmgr_dal_DemoCamera.dll, " in msg
    assert "which is not a directory" in msg


@pytest.mark.parametrize("entry", ["", "   "])
def test_empty_adapter_search_path_entry_is_refused(tmp_path: Path, entry: str) -> None:
    # Path("") is ".", which would silently mean "the profile's own folder".
    path = write_profile(
        tmp_path / "p.toml", f'[micromanager]\nadapter_search_paths = ["{entry}"]\n'
    )
    with pytest.raises(ProfileError) as info:
        Profile.load(path)
    assert (
        'micromanager.adapter_search_paths: adapter_search_paths entries must not be empty; remove the "" entry'
        in str(info.value)
    )


def test_empty_adapter_search_path_entry_is_refused_in_code() -> None:
    from pydantic import ValidationError

    from smc.hardware.profile import MicroManagerSection

    with pytest.raises(ValidationError, match="entries must not be empty"):
        MicroManagerSection(adapter_search_paths=["C:/mm", ""])


# -- light source and channels (m2 §8.6) ------------------------------------------


def test_light_source_and_channels_default_to_unknown() -> None:
    profile = Profile.demo()
    assert profile.light_source.intensity_property == ""
    assert (profile.channels.group, profile.channels.exclude) == ("", [])


def test_light_source_and_channels_are_read_from_the_profile(tmp_path: Path) -> None:
    path = write_profile(
        tmp_path / "p.toml",
        '[light_source]\nintensity_property = "Intensity"\n\n'
        '[channels]\ngroup = "Channel"\nexclude = ["CondenserTurret"]\n',
    )
    profile = Profile.load(path)
    assert profile.light_source.intensity_property == "Intensity"
    assert (profile.channels.group, profile.channels.exclude) == (
        "Channel",
        ["CondenserTurret"],
    )


@pytest.mark.parametrize("group", ["a,b", " Channel", "Channel ", "Kanäle"])
def test_a_channel_group_that_cannot_go_into_a_cfg_is_refused(
    tmp_path: Path, group: str
) -> None:
    path = write_profile(tmp_path / "p.toml", f'[channels]\ngroup = "{group}"\n')
    with pytest.raises(ProfileError) as info:
        Profile.load(path)
    assert (
        f"channels.group: group {group!r} cannot be written to a configuration "
        "file: use printable ASCII with no comma and no leading or trailing space"
    ) in str(info.value)


@pytest.mark.parametrize("entry", ["", "  "])
def test_an_empty_channels_exclude_entry_is_refused(tmp_path: Path, entry: str) -> None:
    path = write_profile(tmp_path / "p.toml", f'[channels]\nexclude = ["{entry}"]\n')
    with pytest.raises(ProfileError) as info:
        Profile.load(path)
    assert (
        'channels.exclude: exclude entries must name a device label; remove the "" '
        "entry"
    ) in str(info.value)


def test_an_unknown_key_in_channels_lists_the_allowed_keys(tmp_path: Path) -> None:
    path = write_profile(tmp_path / "p.toml", '[channels]\ngroups = "Channel"\n')
    with pytest.raises(ProfileError) as info:
        Profile.load(path)
    assert "allowed keys: group, exclude" in str(info.value)


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


# -- text encoding (#93, FM-80) -----------------------------------------------

_BOM = b"\xef\xbb\xbf"


def test_a_profile_saved_as_cp1252_is_a_profile_error_naming_file_and_line(
    tmp_path: Path,
) -> None:
    # The bad byte is on line 4, so the line number is counted, not a default.
    path = tmp_path / "latin.toml"
    path.write_bytes('[microscope]\nname = "t"\n\n# café\n'.encode("cp1252"))
    with pytest.raises(ProfileError) as caught:
        Profile.load(path)
    message = str(caught.value)
    assert message.startswith(f"{path}: line 4 is not UTF-8 text (")
    assert "Save the file as UTF-8" in message
    assert isinstance(caught.value.__cause__, UnicodeDecodeError)


def test_a_profile_saved_as_utf16_is_a_profile_error(tmp_path: Path) -> None:
    path = tmp_path / "wide.toml"
    path.write_bytes('[microscope]\nname = "t"\n'.encode("utf-16"))
    with pytest.raises(ProfileError, match="line 1 is not UTF-8 text"):
        Profile.load(path)


def test_a_utf8_profile_with_a_byte_order_mark_loads(tmp_path: Path) -> None:
    path = tmp_path / "marked.toml"
    path.write_bytes(_BOM + '[microscope]\nname = "café"\n'.encode())
    assert Profile.load(path).microscope.name == "café"


def test_list_profiles_reads_a_file_with_a_byte_order_mark(isolated: Path) -> None:
    profile = isolated / "stand.toml"
    profile.write_bytes(_BOM + b'[microscope]\nname = "t"\n')
    (isolated / "pyproject.toml").write_bytes(_BOM + b'[project]\nname = "x"\n')
    assert list_profiles() == [("stand", profile.resolve())]


def test_a_marked_profile_that_then_is_not_utf8_names_the_right_byte_and_line(
    tmp_path: Path,
) -> None:
    # utf-8-sig reports the offset after the three bytes of the mark, so the
    # offending byte and its line come from the stripped data, not from the file.
    path = tmp_path / "marked-latin.toml"
    path.write_bytes(_BOM + '[microscope]\nname = "t"\n# café\n'.encode("cp1252"))
    with pytest.raises(ProfileError) as caught:
        Profile.load(path)
    message = str(caught.value)
    assert message.startswith(f"{path}: line 3 is not UTF-8 text (")
    assert "byte 0xe9" in message


# -- a file the profile names that cannot be looked at (#93, FM-77) ---------------


def _refuse_to_look_at(
    monkeypatch: pytest.MonkeyPatch, method: str, name: str, error: Exception
) -> None:
    """Make ``Path.<method>`` raise ``error`` for the path called ``name``.

    A patched method runs on every runner: Windows has no ``chmod 000``, and an
    unreachable network share cannot be built in a test.
    """
    real = getattr(Path, method)

    def refuse(self: Path, *args: object, **kwargs: object) -> object:
        if self.name == name:
            raise error
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, method, refuse)


def test_a_config_that_cannot_be_looked_at_is_a_profile_error_naming_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_profile(tmp_path / "p.toml", '[micromanager]\nconfig = "mm.cfg"\n')
    (tmp_path / "mm.cfg").write_text("", encoding="utf-8")
    denied = PermissionError(errno.EACCES, "Permission denied")
    _refuse_to_look_at(monkeypatch, "is_file", "mm.cfg", denied)
    with pytest.raises(ProfileError) as caught:
        Profile.load(path)
    message = str(caught.value)
    assert message.startswith(f"{path}: micromanager.config: cannot look at ")
    assert "Permission denied" in message
    assert caught.value.__cause__ is denied


def test_an_adapter_directory_that_cannot_be_looked_at_is_a_profile_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_profile(
        tmp_path / "p.toml", '[micromanager]\nadapter_search_paths = ["dlls"]\n'
    )
    (tmp_path / "dlls").mkdir()
    denied = PermissionError(errno.EACCES, "Permission denied")
    _refuse_to_look_at(monkeypatch, "is_dir", "dlls", denied)
    with pytest.raises(ProfileError) as caught:
        Profile.load(path)
    assert str(caught.value).startswith(
        f"{path}: micromanager.adapter_search_paths: cannot look at "
    )
    assert caught.value.__cause__ is denied


def test_a_relative_path_the_os_cannot_resolve_is_a_profile_error_naming_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An embedded NUL: resolve() raises ValueError, on POSIX; Windows differs.
    path = write_profile(
        tmp_path / "p.toml", '[micromanager]\nconfig = "a\\u0000b.cfg"\n'
    )
    real_resolve = Path.resolve

    def resolve(self: Path, *args: object, **kwargs: object) -> Path:
        if "\x00" in str(self):
            raise ValueError("embedded null character in path")
        return real_resolve(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "resolve", resolve)
    with pytest.raises(ProfileError) as caught:
        Profile.load(path)
    # The profile's resolved path: that is its ``source``, as in the ``~`` error.
    assert str(caught.value).startswith(
        f"{path.resolve()}: micromanager.config: cannot resolve "
    )
    assert isinstance(caught.value.__cause__, ValueError)


def test_a_directory_named_like_a_profile_is_listed_and_is_a_profile_error(
    isolated: Path,
) -> None:
    # Reading a directory is an OSError (IsADirectoryError, or PermissionError on
    # Windows): ``list_profiles`` keeps it listed, as it keeps any file that
    # does not load, and loading it says what is wrong.
    directory = isolated / "dir.toml"
    directory.mkdir()
    good = write_profile(isolated / "ok.toml", "")
    assert list_profiles() == [
        ("dir", directory.resolve()),
        ("ok", good.resolve()),
    ]
    with pytest.raises(ProfileError, match="cannot read profile"):
        Profile.load(directory)


# -- autofocus (m2 §3.9) ------------------------------------------------------


def _load_error(tmp_path: Path, body: str) -> str:
    path = write_profile(tmp_path / "p.toml", body)
    with pytest.raises(ProfileError) as info:
        Profile.load(path)
    msg = str(info.value)
    assert str(path) in msg
    return msg


def test_autofocus_section_defaults_to_unknown_everywhere(tmp_path: Path) -> None:
    profile = Profile.load(write_profile(tmp_path / "p.toml", ""))
    af = profile.autofocus
    assert (af.offset_unit, af.lock_timeout_s) == ("", None)
    assert (af.offset_per_um, af.in_range_property, af.in_range_values) == ({}, "", [])
    assert profile.safety.max_autofocus_offset_jog is None
    assert profile.safety.autofocus_offset_soft_limits is None
    # The demo measures nothing either: Profile.demo() sets none of them.
    assert Profile.demo().autofocus == af


@pytest.mark.parametrize("value", ["0", "-1.0", "nan", "inf", "600.5"])
def test_lock_timeout_must_be_positive_finite_and_at_most_600_s(
    tmp_path: Path, value: str
) -> None:
    msg = _load_error(tmp_path, f"[autofocus]\nlock_timeout_s = {value}\n")
    assert "autofocus.lock_timeout_s" in msg
    shown = str(float(value))
    assert (
        f"lock_timeout_s must be a finite number of seconds in (0, 600], got {shown}"
        in msg
    )


def test_lock_timeout_of_600_s_is_accepted(tmp_path: Path) -> None:
    path = write_profile(tmp_path / "p.toml", "[autofocus]\nlock_timeout_s = 600\n")
    assert Profile.load(path).autofocus.lock_timeout_s == 600.0


@pytest.mark.parametrize("value", ["0.0", "nan", "-inf"])
def test_offset_per_um_refuses_zero_and_non_finite_naming_the_objective(
    tmp_path: Path, value: str
) -> None:
    msg = _load_error(
        tmp_path,
        "[autofocus]\noffset_per_um = "
        f'{{ "Nikon 10X S Fluor" = -2.5, "Nikon 40X Plan Fluor ELWD" = {value} }}\n',
    )
    assert "autofocus.offset_per_um" in msg
    assert (
        "offset_per_um for objective 'Nikon 40X Plan Fluor ELWD' must be a finite, "
        "non-zero number (the offset change per µm of focus, measured with the "
        f"autofocus locked), got {value}"
    ) in msg
    assert "'Nikon 10X S Fluor'" not in msg


def test_offset_per_um_keeps_the_sign(tmp_path: Path) -> None:
    path = write_profile(
        tmp_path / "p.toml",
        '[autofocus]\noffset_per_um = { "4x" = -3.0, "40x" = 12 }\n',
    )
    assert Profile.load(path).autofocus.offset_per_um == {"4x": -3.0, "40x": 12.0}


@pytest.mark.parametrize(
    ("body", "phrase"),
    [
        (
            '[autofocus]\nin_range_property = "PFS in Range"\n',
            "in_range_property and in_range_values go together: set both",
        ),
        (
            '[autofocus]\nin_range_values = ["Within Range"]\n',
            "in_range_property and in_range_values go together: set both",
        ),
        (
            '[autofocus]\nin_range_property = "   "\n'
            'in_range_values = ["Within Range"]\n',
            "in_range_property must name a property of the autofocus device, or "
            "be empty; got '   '",
        ),
        (
            '[autofocus]\nin_range_property = "PFS in Range"\n'
            'in_range_values = ["Within Range", ""]\n',
            'in_range_values must not contain an empty value; remove the "" entry',
        ),
    ],
)
def test_in_range_property_and_values_come_together(
    tmp_path: Path, body: str, phrase: str
) -> None:
    msg = _load_error(tmp_path, body)
    assert f"  - autofocus: {phrase}" in msg


def test_in_range_property_and_values_load_together(tmp_path: Path) -> None:
    path = write_profile(
        tmp_path / "p.toml",
        '[autofocus]\nin_range_property = "PFS in Range"\n'
        'in_range_values = ["Within Range"]\noffset_unit = "PFS offset steps"\n',
    )
    af = Profile.load(path).autofocus
    assert (af.in_range_property, af.in_range_values) == (
        "PFS in Range",
        ["Within Range"],
    )
    assert af.offset_unit == "PFS offset steps"


@pytest.mark.parametrize("value", ["0.0", "-1.0", "nan", "inf"])
def test_max_autofocus_offset_jog_must_be_positive_and_finite(
    tmp_path: Path, value: str
) -> None:
    msg = _load_error(tmp_path, f"[safety]\nmax_autofocus_offset_jog = {value}\n")
    assert "safety.max_autofocus_offset_jog" in msg
    assert (
        "max_autofocus_offset_jog must be a positive, finite number in [autofocus] "
        f"offset_unit, got {value}"
    ) in msg


@pytest.mark.parametrize(
    ("limits", "phrase"),
    [
        ("[15000.0, 0.0]", "must be [low, high] with low < high"),
        ("[5.0, 5.0]", "must be [low, high] with low < high"),
        ("[0.0, inf]", "must be finite numbers"),
        ("[nan, 1.0]", "must be finite numbers"),
    ],
)
def test_autofocus_offset_soft_limits_must_be_ordered(
    tmp_path: Path, limits: str, phrase: str
) -> None:
    msg = _load_error(tmp_path, f"[safety]\nautofocus_offset_soft_limits = {limits}\n")
    assert "safety.autofocus_offset_soft_limits" in msg
    assert f"autofocus offset limits {phrase}" in msg


def test_autofocus_offset_soft_limits_load(tmp_path: Path) -> None:
    path = write_profile(
        tmp_path / "p.toml", "[safety]\nautofocus_offset_soft_limits = [0.0, 15000.0]\n"
    )
    assert Profile.load(path).safety.autofocus_offset_soft_limits == (0.0, 15000.0)


# -- stop_refused_by (m2 §11.3) --------------------------------------------------


def test_stop_refused_by_accepts_the_roles_with_a_stop(tmp_path: Path) -> None:
    path = write_profile(
        tmp_path / "p.toml",
        '[safety]\nstop_refused_by = ["focus", "xy_stage", "autofocus_offset"]\n',
    )
    assert Profile.load(path).safety.stop_refused_by == (
        Role.focus,
        Role.xy_stage,
        Role.autofocus_offset,
    )
    # Omitted, nothing is declared: the demo's stages take their stop.
    assert Profile.demo().safety.stop_refused_by == ()


def test_stop_refused_by_refuses_a_role_without_a_stop(tmp_path: Path) -> None:
    # "camera" is a role, so the enum alone would let it through.
    msg = _load_error(tmp_path, '[safety]\nstop_refused_by = ["camera"]\n')
    assert (
        "safety.stop_refused_by: 'camera' is not a role with a stop; stop_refused_by "
        "lists roles among xy_stage, focus, autofocus_offset"
    ) in msg


def test_stop_refused_by_refuses_a_misspelt_role_listing_the_three(
    tmp_path: Path,
) -> None:
    msg = _load_error(tmp_path, '[safety]\nstop_refused_by = ["xy"]\n')
    assert "safety.stop_refused_by: 'xy' is not a role with a stop" in msg
    assert "among xy_stage, focus, autofocus_offset" in msg


@pytest.mark.parametrize(
    ("value", "shown"),
    [('[["xy_stage"]]', "['xy_stage']"), ("[{a = 1}]", "{'a': 1}")],
)
def test_stop_refused_by_refuses_a_nested_array_naming_the_key(
    tmp_path: Path, value: str, shown: str
) -> None:
    # An unhashable entry once raised a bare TypeError out of Profile.load.
    msg = _load_error(tmp_path, f"[safety]\nstop_refused_by = {value}\n")
    assert f"safety.stop_refused_by: {shown} is not a role with a stop" in msg


def test_stop_refused_by_refuses_a_role_named_twice(tmp_path: Path) -> None:
    msg = _load_error(
        tmp_path, '[safety]\nstop_refused_by = ["xy_stage", "focus", "xy_stage"]\n'
    )
    assert (
        "safety.stop_refused_by: stop_refused_by names 'xy_stage' twice; name it once"
    ) in msg


@pytest.mark.parametrize(
    ("body", "key", "name"),
    [
        (
            "[autofocus]\nlock_timeout_s = true\n",
            "autofocus.lock_timeout_s",
            "lock_timeout_s",
        ),
        (
            '[autofocus]\noffset_per_um = { "4x" = true }\n',
            "autofocus.offset_per_um",
            "offset_per_um for objective '4x'",
        ),
        (
            "[safety]\nmax_autofocus_offset_jog = true\n",
            "safety.max_autofocus_offset_jog",
            "max_autofocus_offset_jog",
        ),
        (
            "[safety]\nautofocus_offset_soft_limits = [0.0, true]\n",
            "safety.autofocus_offset_soft_limits",
            "each bound of autofocus_offset_soft_limits",
        ),
    ],
)
def test_new_numeric_keys_refuse_a_toml_boolean_and_accept_an_integer(
    tmp_path: Path, body: str, key: str, name: str
) -> None:
    # pydantic's lax mode would read `true` as 1.0 (#106).
    msg = _load_error(tmp_path, body)
    assert key in msg
    assert f"{name} must be a number, not a boolean (got true)" in msg
    integers = write_profile(
        tmp_path / "ints.toml",
        "[safety]\nmax_autofocus_offset_jog = 50\n"
        "autofocus_offset_soft_limits = [0, 15000]\n"
        '[autofocus]\nlock_timeout_s = 5\noffset_per_um = { "4x" = -3 }\n',
    )
    profile = Profile.load(integers)
    assert profile.safety.max_autofocus_offset_jog == 50.0
    assert profile.safety.autofocus_offset_soft_limits == (0.0, 15000.0)
    assert profile.autofocus.lock_timeout_s == 5.0
    assert profile.autofocus.offset_per_um == {"4x": -3.0}


# -- engaging an autofocus that drives Z (m2 §12.3) --------------------------------


def test_engage_keys_default_to_today_s_behaviour(tmp_path: Path) -> None:
    profile = Profile.load(write_profile(tmp_path / "p.toml", ""))
    af = profile.autofocus
    assert (af.engage_drives_z_to_offset, af.settle_s, af.settle_tolerance_um) == (
        False,
        0.0,
        0.5,
    )
    assert profile.safety.max_engage_excursion_um is None


@pytest.mark.parametrize("value", ["0.0", "-1.0", "inf", "nan"])
def test_max_engage_excursion_um_must_be_positive_and_finite(
    tmp_path: Path, value: str
) -> None:
    msg = _load_error(tmp_path, f"[safety]\nmax_engage_excursion_um = {value}\n")
    assert "safety.max_engage_excursion_um" in msg
    assert (
        f"max_engage_excursion_um must be a positive, finite number of µm, got {value}"
    ) in msg


@pytest.mark.parametrize(
    ("key", "value", "phrase"),
    [
        ("settle_s", "-0.1", "settle_s must be a finite number of seconds in [0, 60]"),
        ("settle_s", "61.0", "settle_s must be a finite number of seconds in [0, 60]"),
        ("settle_s", "nan", "settle_s must be a finite number of seconds in [0, 60]"),
        (
            "settle_tolerance_um",
            "0.0",
            "settle_tolerance_um must be a positive, finite number of µm",
        ),
        (
            "settle_tolerance_um",
            "inf",
            "settle_tolerance_um must be a positive, finite number of µm",
        ),
    ],
)
def test_settle_keys_are_bounded(
    tmp_path: Path, key: str, value: str, phrase: str
) -> None:
    # 61 s is refused by its own bound, not by the lock timeout: set one above it.
    msg = _load_error(tmp_path, f"[autofocus]\nlock_timeout_s = 600\n{key} = {value}\n")
    assert f"autofocus.{key}" in msg
    assert f"{phrase}, got {value}" in msg


@pytest.mark.parametrize(
    ("body", "limit"),
    [
        ("lock_timeout_s = 1.0\nsettle_s = 2.0\n", "1 s"),
        ("settle_s = 5.0\n", "5 s, unset: assumed"),
    ],
)
def test_settle_s_must_be_below_the_lock_timeout(
    tmp_path: Path, body: str, limit: str
) -> None:
    msg = _load_error(tmp_path, f"[autofocus]\n{body}")
    assert (
        f"settle_s must be below lock_timeout_s ({limit}), which bounds the lock "
        "and the settling together"
    ) in msg
    accepted = write_profile(tmp_path / "ok.toml", "[autofocus]\nsettle_s = 4.9\n")
    assert Profile.load(accepted).autofocus.settle_s == 4.9


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("safety", "max_engage_excursion_um"),
        ("autofocus", "settle_s"),
        ("autofocus", "settle_tolerance_um"),
    ],
)
def test_new_numeric_engage_keys_refuse_a_toml_boolean(
    tmp_path: Path, section: str, key: str
) -> None:
    # pydantic's lax mode would read `true` as 1.0 (#106).
    msg = _load_error(tmp_path, f"[{section}]\n{key} = true\n")
    assert f"{section}.{key}" in msg
    assert f"{key} must be a number, not a boolean (got true)" in msg
    integers = write_profile(tmp_path / "ints.toml", f"[{section}]\n{key} = 1\n")
    loaded = Profile.load(integers)
    assert getattr(getattr(loaded, section), key) == 1.0


def test_the_default_lock_timeout_mirror_matches_the_backend() -> None:
    # A pure module imports no backend, so it keeps its own copy (m2 §12.3).
    assert profile_module._DEFAULT_LOCK_TIMEOUT_S == mm.DEFAULT_LOCK_TIMEOUT_S


def test_unknown_autofocus_key_is_refused(tmp_path: Path) -> None:
    msg = _load_error(tmp_path, "[autofocus]\nlock_timeout = 5.0\n")
    assert "autofocus.lock_timeout: unknown key in [autofocus]" in msg
    assert (
        "allowed keys: offset_unit, lock_timeout_s, offset_per_um, "
        "in_range_property, in_range_values"
    ) in msg


# -- [sample]: the synthetic plate of the demo-sample stand (#260) -------------


def test_a_sample_table_loads_with_its_defaults(tmp_path: Path) -> None:
    profile = Profile.load(write_profile(tmp_path / "p.toml", "[sample]\n"))

    sample = profile.sample
    assert sample is not None
    assert (sample.plate, sample.a1_center_xy_um) == ("96-well", None)
    assert (sample.rotation_deg, sample.camera_rotation_deg) == (0.0, 0.0)
    assert (sample.mirrored, sample.seed) == (False, 0)
    assert (sample.cells_per_well, sample.doublet_fraction) == (40, 0.3)
    assert sample.cell_radius_um == 5.0


def test_a_sample_table_loads_every_key_it_has(tmp_path: Path) -> None:
    body = (
        "[sample]\nplate = '384-well'\na1_center_xy_um = [100, -200.5]\n"
        "rotation_deg = -3.5\ncamera_rotation_deg = 90\nmirrored = true\n"
        "cells_per_well = 10\ndoublet_fraction = 0\ncell_radius_um = 2.5\nseed = 7\n"
    )
    sample = Profile.load(write_profile(tmp_path / "p.toml", body)).sample

    assert sample is not None
    assert sample.plate == "384-well"
    assert sample.a1_center_xy_um == (100.0, -200.5)
    assert (sample.rotation_deg, sample.camera_rotation_deg) == (-3.5, 90.0)
    assert sample.mirrored is True
    assert (sample.cells_per_well, sample.doublet_fraction) == (10, 0.0)
    assert (sample.cell_radius_um, sample.seed) == (2.5, 7)


def test_a_profile_without_a_sample_table_has_none(tmp_path: Path) -> None:
    assert Profile.load(write_profile(tmp_path / "p.toml", "")).sample is None


def test_profile_demo_has_no_sample() -> None:
    assert Profile.demo().sample is None
    assert Profile.load(DEMO_TOML).sample is None


def test_a_sample_table_is_refused_with_a_micromanager_config(tmp_path: Path) -> None:
    path = write_profile(
        tmp_path / "p.toml", '[micromanager]\nconfig = "stand.cfg"\n[sample]\n'
    )
    with pytest.raises(ProfileError) as info:
        Profile.load(path)

    lines = str(info.value).splitlines()
    problem = next(line for line in lines if line.startswith("  - sample:"))
    assert "config" in problem
    assert "stand.cfg" in problem


def test_a_sample_with_a_micromanager_that_failed_reports_only_that_error(
    tmp_path: Path,
) -> None:
    # `micromanager` is absent from `info.data` when it failed, so the sample
    # check has nothing to compare with and the real problem stands alone.
    path = write_profile(
        tmp_path / "p.toml", "[micromanager]\ndevice_timeout_ms = -1\n[sample]\n"
    )
    with pytest.raises(ProfileError) as info:
        Profile.load(path)

    msg = str(info.value)
    assert "1 problem(s)" in msg
    assert "micromanager.device_timeout_ms" in msg
    assert "  - sample:" not in msg


@pytest.mark.parametrize("plate", ["glass", "coverslip-22mm-square", "", "96-WELL"])
def test_a_sample_plate_that_is_not_a_well_plate_is_refused(
    tmp_path: Path, plate: str
) -> None:
    msg = _load_error(tmp_path, f'[sample]\nplate = "{plate}"\n')

    assert "  - sample.plate:" in msg
    assert "-well" in msg


def test_a_sample_plate_named_like_a_well_plate_that_useq_does_not_know_is_refused(
    tmp_path: Path,
) -> None:
    msg = _load_error(tmp_path, '[sample]\nplate = "97-well"\n')

    assert "  - sample.plate:" in msg
    assert "97-well" in msg
    assert "useq" in msg


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("rotation_deg", "46"),
        ("camera_rotation_deg", "181"),
        ("cells_per_well", "501"),
        ("cells_per_well", "-1"),
        ("cells_per_well", "true"),
        ("doublet_fraction", "1.5"),
        ("doublet_fraction", "-0.1"),
        ("cell_radius_um", "0"),
        ("cell_radius_um", "nan"),
        ("cell_radius_um", "51"),
        ("rotation_deg", "inf"),
        ("camera_rotation_deg", "true"),
        ("seed", "-1"),
        ("seed", "true"),
        ("a1_center_xy_um", "[1e7, 0]"),
        ("a1_center_xy_um", "[0, nan]"),
        ("a1_center_xy_um", "[true, 0]"),
    ],
)
def test_sample_numbers_out_of_bounds_or_boolean_are_refused(
    tmp_path: Path, key: str, value: str
) -> None:
    msg = _load_error(tmp_path, f"[sample]\n{key} = {value}\n")

    assert f"  - sample.{key}:" in msg


def test_sample_numbers_at_their_bounds_load(tmp_path: Path) -> None:
    body = (
        "[sample]\nrotation_deg = 45\ncamera_rotation_deg = -180\ncells_per_well = 500\n"
        "doublet_fraction = 1\ncell_radius_um = 50\nseed = 0\n"
        "a1_center_xy_um = [-1e6, 1e6]\nplate = '6-well'\n"
    )
    sample = Profile.load(write_profile(tmp_path / "p.toml", body)).sample

    assert sample is not None
    assert (sample.rotation_deg, sample.camera_rotation_deg) == (45.0, -180.0)
    assert sample.cells_per_well == 500


def test_an_unknown_sample_key_is_refused_naming_the_allowed_ones(
    tmp_path: Path,
) -> None:
    msg = _load_error(tmp_path, "[sample]\ncells = 3\n")

    assert "unknown key in [sample]" in msg
    assert "allowed keys: plate, a1_center_xy_um" in msg


def test_too_many_cells_for_the_plate_are_refused(tmp_path: Path) -> None:
    # 384 wells x 130 is 49 920 cells; x 131 is 50 304, over the bound.
    msg = _load_error(tmp_path, '[sample]\nplate = "384-well"\ncells_per_well = 131\n')

    assert "  - sample.cells_per_well:" in msg
    assert "50304" in msg
    assert "at most 130" in msg

    loaded = Profile.load(
        write_profile(
            tmp_path / "ok.toml", '[sample]\nplate = "384-well"\ncells_per_well = 130\n'
        )
    )
    assert loaded.sample is not None
    assert loaded.sample.cells_per_well == 130


def test_the_cell_bound_is_the_one_the_stand_imports() -> None:
    assert profile_module.MAX_SAMPLE_CELLS == 50_000


def test_an_unknown_plate_does_not_hide_behind_the_cell_count(tmp_path: Path) -> None:
    # `plate` failed, so it is absent from `info.data` when cells_per_well is
    # checked: the plate's own error is reported, and nothing else is.
    msg = _load_error(tmp_path, '[sample]\nplate = "glass"\ncells_per_well = 500\n')

    assert "1 problem(s)" in msg
    assert "  - sample.plate:" in msg


# -- the profiles that ship with the repository ---------------------------------

PROFILES_DIR = REPO_ROOT / "profiles"
SHIPPED = sorted(PROFILES_DIR.glob("*.toml"))


def test_the_profiles_directory_holds_the_profiles_this_repository_ships() -> None:
    # A glob that finds nothing would pass the parametrised test below by not running it.
    assert {path.stem for path in SHIPPED} >= {"demo", "nikon-ti2", "nikon-tie"}


@pytest.mark.parametrize("path", SHIPPED, ids=lambda path: path.stem)
def test_every_shipped_profile_loads(path: Path) -> None:
    loaded = Profile.load(path)

    assert loaded.microscope.name == path.stem
    config = loaded.config_path()
    assert config is None or config.is_file()


def test_nikon_ti2_profile_holds_the_owner_limits_and_filter_turret2() -> None:
    loaded = Profile.load(PROFILES_DIR / "nikon-ti2.toml")

    # What session 0 measured, and what the owner decided (design m2 §7.7).
    assert loaded.roles.assign == {Role.filter_turret: "FilterTurret2"}
    assert loaded.safety.z_soft_limits_um == (10.0, 4350.0)
    assert loaded.safety.xy_soft_limits_um == (
        (-49000.0, 49000.0),
        (-29000.0, 29000.0),
    )
    assert loaded.safety.max_z_jog_um == 100.0
    assert loaded.safety.max_jog_um == 5000.0
    assert loaded.safety.turret_requires_confirm is True
    assert loaded.micromanager.device_timeout_ms == 60_000
    assert loaded.config_path() == (PROFILES_DIR / "nikon-ti2.cfg").resolve()
    # What session 1 measured at 10x (#208), and the offset jog chosen from it.
    assert loaded.autofocus.offset_per_um == {"2-Plan Fluor 10x": 5.9}
    assert loaded.autofocus.offset_unit == "PFSOffset units"
    assert loaded.autofocus.in_range_property == "PFS in Range"
    assert loaded.autofocus.in_range_values == ["In Range"]
    assert loaded.safety.max_autofocus_offset_jog == 50.0
    # Every stop refused, still and moving (#208, m2 §11).
    assert loaded.safety.stop_refused_by == (
        Role.xy_stage,
        Role.focus,
        Role.autofocus_offset,
    )
    # What no session has measured stays unknown, never guessed: the offset
    # device exposes no limits, and the lock time is not a settling time.
    assert loaded.camera.pixel_size_um == {}
    assert loaded.autofocus.lock_timeout_s is None
    assert loaded.safety.autofocus_offset_soft_limits is None


def test_the_ti2_profile_declares_how_it_engages() -> None:
    # m2 §12.3: measured in run 1 (#208), and chosen for session 2 to confirm.
    loaded = Profile.load(PROFILES_DIR / "nikon-ti2.toml")
    assert loaded.autofocus.engage_drives_z_to_offset is True
    assert loaded.autofocus.settle_s == 1.0
    assert loaded.autofocus.settle_tolerance_um == 0.5
    assert loaded.safety.max_engage_excursion_um == 50.0


def test_nikon_tie_profile_sets_no_limit_beyond_the_layers_defaults() -> None:
    loaded = Profile.load(PROFILES_DIR / "nikon-tie.toml")

    assert loaded.config_path() == (PROFILES_DIR / "nikon-tie.cfg").resolve()
    assert loaded.roles.assign == {}
    assert loaded.safety.max_jog_um == 5000.0
    assert loaded.safety.turret_requires_confirm is True
    # Nothing was measured on a Ti-E: the Z jog default stays "assumed".
    assert loaded.safety.max_z_jog_um is None
    assert loaded.safety.z_soft_limits_um is None
    assert loaded.safety.xy_soft_limits_um is None
    assert loaded.safety.stop_refused_by == ()
    assert loaded.camera.pixel_size_um == {}
