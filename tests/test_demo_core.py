"""The demo configuration is the regression bed; make sure it behaves."""

import logging
import os
import re
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
