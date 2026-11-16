"""The core opener builds a ``UniMMCore`` only for a ``.cfg`` with ``#py`` lines (m4 §3).

A plain ``CMMCorePlus`` reads ``#py`` lines as comments and loads nothing; a
``UniMMCore`` only logs a Python device whose ``initialize()`` failed. These
tests write a small Python device module into ``tmp_path`` so that they need
no Micro-Manager adapter.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pymmcore_plus import CMMCorePlus
from pymmcore_plus.experimental.unicore import UniMMCore

from smc.hardware.core import CoreError, declares_python_devices, open_core

_MODULE = "smc_test_unicore_devices"

_DEVICES = """
from pymmcore_plus.experimental.unicore import ShutterDevice


class GoodShutter(ShutterDevice):
    def __init__(self):
        super().__init__()
        self._open = False

    def get_open(self):
        return self._open

    def set_open(self, open):
        self._open = bool(open)


class BadShutter(GoodShutter):
    def initialize(self):
        raise RuntimeError("the stand is off")
"""


@pytest.fixture
def device_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """A module of Python devices importable for this test only."""
    (tmp_path / f"{_MODULE}.py").write_text(_DEVICES, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    return _MODULE


def _cfg(tmp_path: Path, lines: list[str], name: str = "stand.cfg") -> Path:
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_a_cfg_with_py_lines_opens_a_unimmcore(
    tmp_path: Path, device_module: str
) -> None:
    cfg = _cfg(
        tmp_path,
        [
            "Property,Core,Initialize,0",
            f"#py pyDevice,Sh,{device_module},GoodShutter",
            "Property,Core,Initialize,1",
            "#py Property,Core,Shutter,Sh",
        ],
    )
    core = open_core(cfg)
    try:
        assert isinstance(core, UniMMCore)
        assert "Sh" in core.getLoadedDevices()
        assert core.getShutterDevice() == "Sh"
    finally:
        core.unloadAllDevices()


def test_a_cfg_without_py_lines_opens_a_plain_cmmcoreplus(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, ["Property,Core,Initialize,0", "Property,Core,Initialize,1"])
    core = open_core(cfg)
    try:
        assert type(core) is CMMCorePlus
    finally:
        core.unloadAllDevices()


def test_py_detection_reads_bytes_so_a_cp1252_cfg_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "cp1252.cfg"
    # "# 10\xb5m objective" is cp1252 for "10 µm": a byte UTF-8 refuses.
    path.write_bytes(
        b"# 10\xb5m objective\r\n  \t#py pyDevice,Sh,mod,Cls\r\nProperty,Core,Initialize,1\r\n"
    )
    with pytest.raises(UnicodeDecodeError):
        path.read_text(encoding="utf-8")
    assert declares_python_devices(path)


def test_py_detection_needs_the_marker_at_a_line_start(tmp_path: Path) -> None:
    path = tmp_path / "plain.cfg"
    path.write_bytes(b"# a comment that says #py pyDevice\n#python is not #py\n")
    assert not declares_python_devices(path)


def test_a_py_device_that_fails_to_initialize_fails_the_open_and_names_it(
    tmp_path: Path, device_module: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed: list[object] = []
    import smc.hardware.core as core_module

    real_close = core_module.close_core

    def spy_close(core: CMMCorePlus) -> None:
        closed.append(core)
        real_close(core)

    monkeypatch.setattr(core_module, "close_core", spy_close)
    cfg = _cfg(
        tmp_path,
        [
            "Property,Core,Initialize,0",
            f"#py pyDevice,Zeta,{device_module},BadShutter",
            f"#py pyDevice,Fine,{device_module},GoodShutter",
            f"#py pyDevice,Alpha,{device_module},BadShutter",
            "Property,Core,Initialize,1",
        ],
    )
    with pytest.raises(CoreError) as caught:
        open_core(cfg)
    message = str(caught.value)
    assert "Zeta, Alpha" in message  # load order, not alphabetical
    assert "Fine" not in message
    assert "ERROR line" in message
    assert len(closed) == 1
    assert not closed[0].getLoadedDevices() or list(  # type: ignore[attr-defined]
        closed[0].getLoadedDevices()  # type: ignore[attr-defined]
    ) == ["Core"]
