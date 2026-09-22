"""The demo configuration is the regression bed; make sure it behaves."""

import pytest

from smc.hardware.core import CoreError, open_core, opened

pytestmark = pytest.mark.demo


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
