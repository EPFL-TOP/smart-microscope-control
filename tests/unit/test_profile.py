from __future__ import annotations

import errno
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
