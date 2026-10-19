"""The demo configuration is the regression bed; make sure it behaves."""

import logging
import os
import re
import weakref
from pathlib import Path
from typing import Any

import pytest

from smc.hardware.core import CoreError, open_core, opened

pytestmark = pytest.mark.demo


@pytest.fixture
def need_mm(mm_available: bool) -> None:
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")


@pytest.fixture
def restored_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo what ``open_core`` does to ``PATH`` (FM-46).

    Search paths are put on ``PATH`` for the whole process, as pymmcore-plus does:
    without this, a test's ``tmp_path`` folder stays on ``PATH`` for every later test.
    """
    monkeypatch.setenv("PATH", os.environ["PATH"])


def _default_search_paths() -> list[str]:
    """What a fresh core searches, with no profile involved."""
    from pymmcore_plus import CMMCorePlus

    return [str(p) for p in CMMCorePlus().getDeviceAdapterSearchPaths()]


def _search_paths(core: Any) -> list[str]:
    return [str(p) for p in core.getDeviceAdapterSearchPaths()]


def _adapters_dir(tmp_path: Path, name: str = "adapters") -> Path:
    """An empty directory: enough to observe the order, never a vendor adapter (FM-40)."""
    path = tmp_path / name
    path.mkdir()
    return path


def test_demo_config_fills_the_core_roles(demo_core) -> None:
    # Everything above the hardware layer asks for these roles; the demo
    # configuration must provide them or no plugin can be tested.
    assert demo_core.getCameraDevice()
    assert demo_core.getXYStageDevice()
    assert demo_core.getFocusDevice()


def test_xy_stage_moves_where_it_is_told(demo_core) -> None:
    dev = demo_core.getXYStageDevice()
    demo_core.setXYPosition(123.0, -45.0)
    demo_core.waitForDevice(dev)
    x, y = demo_core.getXYPosition()
    assert x == pytest.approx(123.0, abs=0.5)
    assert y == pytest.approx(-45.0, abs=0.5)


def test_snap_returns_a_2d_frame(demo_core) -> None:
    frame = demo_core.snap()
    assert frame.ndim == 2
    assert frame.shape == (demo_core.getImageHeight(), demo_core.getImageWidth())


def test_opened_releases_devices(mm_available: bool) -> None:
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    with opened(None) as core:
        assert len(core.getLoadedDevices()) > 1
    assert list(core.getLoadedDevices()) == ["Core"]


def test_missing_config_is_a_clear_error(tmp_path) -> None:
    with pytest.raises(CoreError, match="configuration not found"):
        open_core(tmp_path / "does-not-exist.cfg")


# -- adapter search paths (#77) -----------------------------------------------


@pytest.mark.usefixtures("need_mm", "restored_path")
def test_open_core_puts_extra_adapter_dirs_first(tmp_path: Path) -> None:
    default = _default_search_paths()
    extra = _adapters_dir(tmp_path)
    core = open_core(None, adapter_search_paths=[extra])
    try:
        assert _search_paths(core) == [str(extra), *default]
        # The demo adapters are only in the install, behind the extra directory:
        # reaching them proves the directory that comes first does not hide them.
        assert len(core.getLoadedDevices()) > 1
    finally:
        core.unloadAllDevices()


@pytest.mark.usefixtures("need_mm", "restored_path")
def test_opened_passes_the_adapter_dirs_through(tmp_path: Path) -> None:
    extra = _adapters_dir(tmp_path)
    with opened(None, adapter_search_paths=[extra]) as core:
        assert _search_paths(core)[0] == str(extra)


@pytest.mark.usefixtures("need_mm", "restored_path")
@pytest.mark.parametrize("with_config", [False, True], ids=["demo", "config-file"])
def test_open_core_sets_search_paths_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_config: bool
) -> None:
    import pymmcore_plus

    seen: list[list[str]] = []

    # CMMCorePlus refuses attribute assignment, so a subclass stands in for it:
    # open_core imports the name when it runs.
    class Recording(pymmcore_plus.CMMCorePlus):
        def loadSystemConfiguration(  # noqa: N802 - MMCore API
            self, *args: Any, **kwargs: Any
        ) -> Any:
            seen.append(_search_paths(self))
            return super().loadSystemConfiguration(*args, **kwargs)

    monkeypatch.setattr(pymmcore_plus, "CMMCorePlus", Recording)
    extra = _adapters_dir(tmp_path)
    config = None
    if with_config:
        config = tmp_path / "scope.cfg"
        config.write_text("# no devices\n", encoding="utf-8")
    core = open_core(config, adapter_search_paths=[extra])
    core.unloadAllDevices()
    # Set after loading, the directory would be searched by nobody: the
    # adapters are found while the configuration loads.
    assert [paths[0] for paths in seen] == [str(extra)]


@pytest.mark.usefixtures("restored_path")
@pytest.mark.parametrize("kind", ["missing", "a-file"])
def test_open_core_refuses_a_bad_adapter_dir_before_building_a_core(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    import pymmcore_plus

    def no_core(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("a core was built before the directories were checked")

    # Both checks are false here, so the order shows: the CoreError can only
    # come from the directory check if it runs before the core is built.
    monkeypatch.setattr(pymmcore_plus, "CMMCorePlus", no_core)
    bad = tmp_path / "not-there"
    if kind == "a-file":
        bad.write_text("not a directory\n", encoding="utf-8")
    with pytest.raises(CoreError) as info:
        open_core(None, adapter_search_paths=[bad])
    assert f"adapter search path is not a directory: {bad}" in str(info.value)
    assert "[micromanager] adapter_search_paths" in str(info.value)
    # Control: the refusal above says nothing unless the stand-in is what
    # open_core builds. A valid call must reach it.
    with pytest.raises(RuntimeError, match="a core was built"):
        open_core(None)


@pytest.mark.usefixtures("need_mm", "restored_path")
def test_open_core_drops_duplicate_adapter_dirs(tmp_path: Path) -> None:
    default = _default_search_paths()
    extra = _adapters_dir(tmp_path)
    core = open_core(
        None, adapter_search_paths=[extra, Path(default[0]), extra, Path(default[0])]
    )
    try:
        assert _search_paths(core) == [str(extra), *default]
    finally:
        core.unloadAllDevices()


@pytest.mark.usefixtures("need_mm", "restored_path")
def test_open_core_sees_one_directory_spelled_two_ways(tmp_path: Path) -> None:
    extra = _adapters_dir(tmp_path)
    detour = tmp_path / "adapters" / ".." / "adapters"
    core = open_core(None, adapter_search_paths=[detour, extra])
    try:
        # The first spelling keeps the position; the second is the same folder.
        assert _search_paths(core) == [str(detour), *_default_search_paths()]
    finally:
        core.unloadAllDevices()


@pytest.mark.usefixtures("need_mm", "restored_path")
def test_open_core_makes_each_adapter_dir_an_exact_path_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # pymmcore-plus skips a directory whose string occurs anywhere in PATH, so
    # "mm-2.0" is skipped when "mm-2.0.3" is on it, and a Windows adapter then
    # loads its DLLs from nowhere.
    longer = _adapters_dir(tmp_path, "mm-2.0.3")
    extra = _adapters_dir(tmp_path, "mm-2.0")
    monkeypatch.setenv("PATH", str(longer))
    core = open_core(None, adapter_search_paths=[extra])
    core.unloadAllDevices()
    assert str(extra) in os.environ["PATH"].split(os.pathsep)


@pytest.mark.usefixtures("need_mm", "restored_path")
def test_open_core_puts_the_adapter_dirs_on_path_in_search_order(
    tmp_path: Path,
) -> None:
    # pymmcore-plus prepends them one by one, which reverses their order.
    first = _adapters_dir(tmp_path, "first")
    second = _adapters_dir(tmp_path, "second")
    core = open_core(None, adapter_search_paths=[first, second])
    core.unloadAllDevices()
    effective = _search_paths(core)
    assert effective[:2] == [str(first), str(second)]
    assert os.environ["PATH"].split(os.pathsep)[: len(effective)] == effective


@pytest.mark.usefixtures("need_mm", "restored_path")
def test_open_core_leaves_the_rest_of_path_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kept = [str(tmp_path / "tools"), str(tmp_path / "more tools")]
    monkeypatch.setenv("PATH", os.pathsep.join(kept))
    extra = _adapters_dir(tmp_path)
    core = open_core(None, adapter_search_paths=[extra])
    core.unloadAllDevices()
    entries = os.environ["PATH"].split(os.pathsep)
    assert entries[len(_search_paths(core)) :] == kept


@pytest.mark.usefixtures("need_mm", "restored_path")
def test_open_core_logs_the_effective_search_paths(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    default = _default_search_paths()
    extra = _adapters_dir(tmp_path)
    with caplog.at_level(logging.INFO, logger="smc.hardware.core"):
        core = open_core(None, adapter_search_paths=[extra])
    core.unloadAllDevices()
    expected = "Micro-Manager adapter search paths, first wins: " + "; ".join(
        [str(extra), *default]
    )
    assert [
        r.getMessage() for r in caplog.records if r.name == "smc.hardware.core"
    ] == [expected]


@pytest.mark.usefixtures("need_mm", "restored_path")
def test_open_core_without_extra_dirs_keeps_the_default(
    caplog: pytest.LogCaptureFixture,
) -> None:
    default = _default_search_paths()
    with caplog.at_level(logging.INFO, logger="smc.hardware.core"):
        core = open_core(None)
    core.unloadAllDevices()
    assert _search_paths(core) == default
    # The same line on every open, so a stand's log always says what was searched.
    assert "first wins: " + "; ".join(default) in caplog.text


@pytest.mark.usefixtures("need_mm")
def test_open_core_refuses_a_lone_path_instead_of_a_list(tmp_path: Path) -> None:
    # A str is a Sequence[str]: each character would become a "directory".
    with pytest.raises(TypeError, match=re.escape("sequence of directories")):
        open_core(None, adapter_search_paths=str(tmp_path))


@pytest.mark.usefixtures("need_mm", "restored_path")
def test_open_core_without_extras_still_makes_the_install_an_exact_path_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The common case, a profile with no extra directories: the substring rule
    # of pymmcore-plus would skip the install when "...-old" is on PATH.
    default = _default_search_paths()
    monkeypatch.setenv("PATH", default[0] + "-old")
    core = open_core(None)
    core.unloadAllDevices()
    assert default[0] in os.environ["PATH"].split(os.pathsep)


@pytest.mark.usefixtures("need_mm", "restored_path")
def test_open_core_makes_a_relative_adapter_dir_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _adapters_dir(tmp_path)
    monkeypatch.chdir(tmp_path)
    core = open_core(None, adapter_search_paths=["adapters"])
    core.unloadAllDevices()
    # A relative entry would follow the working directory at every later load.
    assert _search_paths(core)[0] == str(tmp_path / "adapters")


@pytest.mark.usefixtures("need_mm", "restored_path")
def test_open_core_expands_the_home_directory_of_an_adapter_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _adapters_dir(tmp_path)
    # expanduser reads HOME on POSIX and USERPROFILE on Windows.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    core = open_core(None, adapter_search_paths=["~/adapters"])
    core.unloadAllDevices()
    assert _search_paths(core)[0] == str(tmp_path / "adapters")


def test_adapter_dirs_are_the_same_when_only_the_case_differs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Windows folds case, POSIX does not: stand in for Windows' normcase so the
    # branch runs on every runner.
    monkeypatch.setattr(os.path, "normcase", str.lower)
    from smc.hardware import core as core_mod

    assert core_mod._merge_search_paths(["/Mm/Adapters"], ["/mm/adapters"]) == [
        "/Mm/Adapters"
    ]
    monkeypatch.setenv("PATH", os.pathsep.join(["/x/MM", "/y"]))
    # Different case on each side: both must be folded before they are compared.
    core_mod._put_on_path(["/X/Mm"])
    assert os.environ["PATH"].split(os.pathsep) == ["/X/Mm", "/y"]


@pytest.mark.parametrize("entry", ["", "   "])
def test_open_core_refuses_an_empty_adapter_dir(entry: str) -> None:
    # Path("") is ".": the working directory would silently become the first
    # place adapters and DLLs are taken from.
    with pytest.raises(CoreError) as info:
        open_core(None, adapter_search_paths=[entry])
    assert "adapter search path is empty" in str(info.value)
    assert "[micromanager] adapter_search_paths" in str(info.value)


def test_open_core_names_an_adapter_dir_it_cannot_expand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_home(self: Path) -> Path:
        raise RuntimeError("Could not determine home directory.")

    # "~nosuchuser" does this on POSIX; Windows has no such failure to provoke.
    monkeypatch.setattr(Path, "expanduser", no_home)
    with pytest.raises(CoreError) as info:
        open_core(None, adapter_search_paths=["~nosuchuser/mm"])
    msg = str(info.value)
    assert "adapter search path cannot be expanded: ~nosuchuser/mm" in msg
    assert "Could not determine home directory." in msg
    assert "[micromanager] adapter_search_paths" in msg


@pytest.mark.usefixtures("need_mm", "restored_path")
@pytest.mark.parametrize("with_config", [False, True], ids=["demo", "config-file"])
def test_a_failed_load_says_where_the_adapter_dirs_were_searched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_config: bool
) -> None:
    import pymmcore_plus

    # MMCore takes every adapter from the first directory that has it, so an
    # extra folder with a mismatched adapter fails a load that worked before.
    # The error must point at the folder that came first.
    class Failing(pymmcore_plus.CMMCorePlus):
        def loadSystemConfiguration(  # noqa: N802 - MMCore API
            self, *args: Any, **kwargs: Any
        ) -> Any:
            raise RuntimeError('Failed to load device adapter "DemoCamera"')

    monkeypatch.setattr(pymmcore_plus, "CMMCorePlus", Failing)
    default = _default_search_paths()
    extra = _adapters_dir(tmp_path)
    config = None
    if with_config:
        config = tmp_path / "scope.cfg"
        config.write_text("# no devices\n", encoding="utf-8")
    with pytest.raises(CoreError) as info:
        open_core(config, adapter_search_paths=[extra])
    msg = str(info.value)
    assert 'Failed to load device adapter "DemoCamera"' in msg
    assert (
        "(adapter search paths, first wins: " + "; ".join([str(extra), *default]) in msg
    )
    # Without extras nothing was searched first, so nothing is added.
    with pytest.raises(CoreError) as plain:
        open_core(config)
    assert "first wins" not in str(plain.value)


# -- PATH: a separator inside an entry, and what outlives a close (#110) -------


def _path_separator_warning(entry: str) -> str:
    """The WARNING decided in design section 7 (#110), whole.

    Its tail is what tells the operator that the folder is not refused and what
    still fails, so a prefix would let that part be reworded away (FM-45).
    """
    return (
        f"adapter search path {entry} contains {os.pathsep!r}, so it cannot be an "
        "entry of PATH; MMCore still searches it, but a vendor DLL beside an "
        "adapter in it will not be found through PATH"
    )


def test_put_on_path_leaves_out_an_entry_that_contains_the_separator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from smc.hardware import core as core_mod

    monkeypatch.setenv("PATH", "/keep")
    odd = f"/mm/a{os.pathsep}b"
    core_mod._put_on_path([odd, "/mm/c"])
    # Written as it is, the entry would read back as "/mm/a" and "b".
    assert os.environ["PATH"].split(os.pathsep) == ["/mm/c", "/keep"]
    # And every later open would add those two pieces again.
    core_mod._put_on_path([odd, "/mm/c"])
    assert os.environ["PATH"].split(os.pathsep) == ["/mm/c", "/keep"]


def test_put_on_path_warns_about_each_entry_it_cannot_write(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from smc.hardware import core as core_mod

    monkeypatch.setenv("PATH", "/keep")
    odd = f"/mm/a{os.pathsep}b"
    # A trailing separator reads back as an empty entry, which means the
    # current directory: it is as unwritable as one in the middle.
    trailing = f"/mm/d{os.pathsep}"
    with caplog.at_level(logging.WARNING, logger="smc.hardware.core"):
        core_mod._put_on_path(["/mm/c"])
        # Control: an ordinary entry says nothing, so the records below come
        # from the entries with the separator and not from the function's habit.
        assert [r for r in caplog.records if r.name == "smc.hardware.core"] == []
        # Ordinary entries on both sides, so a loop that stops at the first
        # odd one, or one that only looks at the first entry, shows.
        core_mod._put_on_path(["/mm/b", odd, trailing, "/mm/c"])
    records = [r for r in caplog.records if r.name == "smc.hardware.core"]
    # Once per entry, each naming its own entry, at WARNING (not ERROR).
    assert [(r.levelno, r.getMessage()) for r in records] == [
        (logging.WARNING, _path_separator_warning(odd)),
        (logging.WARNING, _path_separator_warning(trailing)),
    ]
    assert os.environ["PATH"].split(os.pathsep) == ["/mm/b", "/mm/c", "/keep"]


@pytest.mark.usefixtures("need_mm", "restored_path")
def test_open_core_does_not_grow_path_for_a_directory_with_the_path_separator(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    odd = tmp_path / f"mm{os.pathsep}odd"
    try:
        odd.mkdir()
    except OSError as exc:
        pytest.skip(f"cannot create a folder named with {os.pathsep!r} here: {exc}")
    after_open: list[str] = []
    warned: list[list[str]] = []
    for _ in range(2):
        caplog.clear()
        with (
            caplog.at_level(logging.WARNING, logger="smc.hardware.core"),
            opened(None, adapter_search_paths=[odd]) as core,
        ):
            after_open.append(os.environ["PATH"])
            # The list MMCore was handed starts with it, and the demo loads from
            # behind it: the directory was kept off PATH, not refused.
            assert _search_paths(core)[0] == str(odd)
            assert core.getCameraDevice()
        warned.append(
            [r.getMessage() for r in caplog.records if r.name == "smc.hardware.core"]
        )
    assert after_open[1] == after_open[0]
    # open_core is what reaches the operator: one warning per open, no more
    # (a second call of the helper) and no fewer (a directory filtered out
    # before it).
    assert warned == [[_path_separator_warning(str(odd))]] * 2


@pytest.mark.usefixtures("need_mm", "restored_path")
def test_open_core_leaves_an_earlier_opens_directories_on_path_behind_its_own(
    tmp_path: Path,
) -> None:
    # Decided in design §7 (#110): PATH is process-wide and close_core does not
    # restore it. What bounds the effect is the order, so that is pinned too.
    extra = _adapters_dir(tmp_path)
    with opened(None, adapter_search_paths=[extra]) as core:
        written = _search_paths(core)[0]
    assert written in os.environ["PATH"].split(os.pathsep)
    with opened(None):
        pass
    entries = os.environ["PATH"].split(os.pathsep)
    assert entries.index(_default_search_paths()[0]) < entries.index(written)


@pytest.mark.usefixtures("need_mm", "restored_path")
def test_reopening_with_the_same_extras_leaves_path_as_the_first_open_made_it(
    tmp_path: Path,
) -> None:
    # The normal case was always idempotent; the decision above relies on it,
    # since a process that opens the same stand again must not grow PATH.
    extra = _adapters_dir(tmp_path)
    with opened(None, adapter_search_paths=[extra]):
        pass
    after_first = os.environ["PATH"]
    with opened(None, adapter_search_paths=[extra]):
        pass
    assert os.environ["PATH"] == after_first


# -- a failed open (#132) -----------------------------------------------------

# Two demo devices, then (per test) a line that fails. The failing line is the
# last one of each file, so everything above it was loaded when it failed.
_CFG_HEAD = "Device,Cam,DemoCamera,DCam\nDevice,Z,DemoCamera,DStage\n"
_CFG_INIT = "Property,Core,Initialize,1\n"


def _devices_beyond_core(core: Any) -> list[str]:
    """The labels loaded besides the ``Core`` that every MMCore has."""
    return [d for d in core.getLoadedDevices() if d != "Core"]


def _write_cfg(path: Path, text: str) -> None:
    # Bytes, so that the lines stay "\n" on Windows (FM-21).
    path.write_bytes(text.encode("utf-8"))


@pytest.fixture
def recording_cores(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """The cores ``open_core`` builds, in order, so a test can look at one it never got back."""
    import pymmcore_plus

    built: list[Any] = []

    class Recording(pymmcore_plus.CMMCorePlus):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            built.append(self)

    monkeypatch.setattr(pymmcore_plus, "CMMCorePlus", Recording)
    return built


@pytest.mark.usefixtures("need_mm", "restored_path")
@pytest.mark.parametrize(
    ("text", "failing_line"),
    [
        (
            _CFG_HEAD + "Device,Bad,NoSuchAdapter,Nothing\n",
            "Device,Bad,NoSuchAdapter,Nothing",
        ),
        (
            _CFG_HEAD + "Device,Dz,Utilities,DA Z Stage\n" + _CFG_INIT,
            "Property,Core,Initialize,1",
        ),
        (
            _CFG_HEAD + _CFG_INIT + "Property,NoSuchLabel,Foo,1\n",
            "Property,NoSuchLabel,Foo,1",
        ),
    ],
    ids=[
        "adapter-missing",
        "device-fails-to-initialize",
        "label-unknown-after-initialize",
    ],
)
def test_a_config_that_fails_after_loading_devices_leaves_none_loaded(
    tmp_path: Path, recording_cores: list[Any], text: str, failing_line: str
) -> None:
    # MMCore unloads what it had loaded before it raises (design §7, #132), so
    # open_core has nothing to release. Measured, not assumed: a later MMCore
    # that stops doing it turns this red.
    cfg = tmp_path / "scope.cfg"
    _write_cfg(cfg, text)
    with pytest.raises(CoreError) as raised:
        open_core(cfg)
    message = str(raised.value)
    # Our phrase, and the line where the file failed (not MMCore's wording,
    # and not a word the path could supply, FM-45): the file failed where the
    # case says, so the devices above it had been loaded.
    assert message.startswith(f"could not load {cfg}: ")
    assert failing_line in message
    assert len(recording_cores) == 1
    assert _devices_beyond_core(recording_cores[0]) == []


@pytest.mark.usefixtures("need_mm", "restored_path")
def test_a_demo_configuration_that_fails_after_loading_devices_leaves_none_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recording_cores: list[Any]
) -> None:
    # The demo branch of open_core has its own handler. The demo file is not
    # ours to break, so the core is told to load a failing file of ours when
    # open_core asks for the demo (the same call into MMCore, with a file).
    import pymmcore_plus

    cfg = tmp_path / "scope.cfg"
    _write_cfg(cfg, _CFG_HEAD + _CFG_INIT + "Property,NoSuchLabel,Foo,1\n")

    # pymmcore_plus.CMMCorePlus is the recording subclass of the fixture here,
    # so the core this builds is appended to recording_cores too.
    class DemoThatFails(pymmcore_plus.CMMCorePlus):
        def loadSystemConfiguration(  # noqa: N802 - MMCore API
            self, *args: Any, **kwargs: Any
        ) -> Any:
            return super().loadSystemConfiguration(str(cfg))

    monkeypatch.setattr(pymmcore_plus, "CMMCorePlus", DemoThatFails)
    with pytest.raises(CoreError) as raised:
        open_core(None)
    message = str(raised.value)
    # Our phrase of the demo branch, and the line of the file the stand-in
    # loaded: the demo handler ran, after the devices above that line loaded.
    assert message.startswith("could not load the demo configuration: ")
    assert "Property,NoSuchLabel,Foo,1" in message
    assert len(recording_cores) == 1
    assert _devices_beyond_core(recording_cores[0]) == []


@pytest.mark.usefixtures("need_mm", "restored_path")
def test_the_devices_of_the_failing_configs_load_when_the_file_is_valid(
    tmp_path: Path,
) -> None:
    # Control for the test above: an empty list there means "unloaded", not
    # "never loaded".
    cfg = tmp_path / "scope.cfg"
    _write_cfg(cfg, _CFG_HEAD + _CFG_INIT)
    core = open_core(cfg)
    try:
        assert sorted(_devices_beyond_core(core)) == ["Cam", "Z"]
    finally:
        core.unloadAllDevices()


@pytest.mark.usefixtures("need_mm", "restored_path")
def test_the_leftover_check_sees_a_device_that_stays_loaded() -> None:
    # Without this, the failed-open test passes for a helper that always
    # returns [] (FM-43).
    from pymmcore_plus import CMMCorePlus

    core = CMMCorePlus()
    try:
        core.loadDevice("Cam", "DemoCamera", "DCam")
        assert _devices_beyond_core(core) == ["Cam"]
    finally:
        core.unloadAllDevices()


@pytest.mark.usefixtures("need_mm", "restored_path")
@pytest.mark.parametrize("with_config", [False, True], ids=["demo", "config-file"])
def test_an_interrupt_during_open_is_not_a_core_error_and_frees_the_core(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_cyclic_gc: None,
    caplog: pytest.LogCaptureFixture,
    with_config: bool,
) -> None:
    import pymmcore_plus

    refs: list[weakref.ReferenceType[Any]] = []
    loaded_at_interrupt: list[list[str]] = []

    # A Ctrl-C reaches Python when the call into MMCore returns: the devices
    # are loaded by then. Raised from the stub, not sent as a signal (FM-44).
    class Interrupted(pymmcore_plus.CMMCorePlus):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            refs.append(weakref.ref(self))

        def loadSystemConfiguration(  # noqa: N802 - MMCore API
            self, *args: Any, **kwargs: Any
        ) -> Any:
            super().loadSystemConfiguration(*args, **kwargs)
            loaded_at_interrupt.append(_devices_beyond_core(self))
            raise KeyboardInterrupt

    monkeypatch.setattr(pymmcore_plus, "CMMCorePlus", Interrupted)
    # pymmcore-plus logs "Initialized core %s" at DEBUG, and pytest keeps the
    # record with its arguments, which would hold the core (see
    # test_closing_the_demo_facade_frees_its_core_without_the_cyclic_collector).
    caplog.set_level(logging.INFO, logger="pymmcore-plus")
    config = None
    if with_config:
        config = tmp_path / "scope.cfg"
        _write_cfg(config, _CFG_HEAD + _CFG_INIT)
    # No `as`: the ExceptionInfo would hold the traceback, and so the core.
    with pytest.raises(KeyboardInterrupt):
        open_core(config)
    assert len(loaded_at_interrupt) == 1
    assert loaded_at_interrupt[0] != [], "no device was loaded when the interrupt came"
    assert len(refs) == 1
    # Freed by reference count once the exception is handled; MMCore's
    # destructor then unloads the devices.
    assert refs[0]() is None
