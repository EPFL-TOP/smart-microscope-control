"""A run's frames on the plate page's image (m3 §25.5).

A stub plugin hands the frames a test gives it, through ``ctx.show_frame``, and
waits on class-level ``threading.Event``s between them (the pattern of ``Slow`` in
``test_plate_page_controller.py``, whose helpers are copied here: ``tests/`` has
no packages). Everything runs on ``fake_microscope``; every wait is bounded
(FM-44).
"""

from __future__ import annotations

import logging
import sys
import threading
from collections.abc import Iterator
from datetime import timezone
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any, ClassVar, cast

import numpy as np
import numpy.typing as npt
import pytest

import smc.ui.controller as controller_module
from smc.hardware import Microscope
from smc.hardware.capabilities import Camera
from smc.plugins.api import PluginParams, PluginResult, RunContext
from smc.plugins.registry import GROUP
from smc.ui.controller import PlatePage
from smc.ui.frames import display_image

#: Bounded waits (FM-44): long enough for a slow CI runner, never forever.
WAIT_S = 10.0


class NoParams(PluginParams):
    pass


class StubResult(PluginResult):
    n: int = 0


class Hands:
    """Hands ``images`` in order, at ``(100 i, -50 i)`` µm and labelled ``F<i>``.

    After frame ``i`` it sets ``handed[i]`` and, when ``i`` is in ``pauses``, waits
    for ``go[i]`` (or a stop). It keeps its context in ``kept``: a plugin that left a
    thread behind calling ``show_frame`` after its run is what the page's ``clear()``
    guards against.
    """

    name = provides = "hands"
    summary = "A test plugin that hands frames."
    Params = NoParams
    Result = StubResult
    requires: tuple[type, ...] = ()
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()
    images: ClassVar[list[npt.NDArray[Any]]] = []
    pauses: ClassVar[frozenset[int]] = frozenset()
    handed: ClassVar[list[threading.Event]] = []
    go: ClassVar[list[threading.Event]] = []
    kept: ClassVar[list[RunContext]] = []

    def run(self, microscope: Any, params: NoParams, ctx: RunContext) -> StubResult:
        Hands.kept.append(ctx)
        for i, image in enumerate(Hands.images):
            ctx.show_frame(image, xy_um=(100.0 * i, -50.0 * i), label=f"F{i}")
            Hands.handed[i].set()
            if i in Hands.pauses:
                while not Hands.go[i].is_set():
                    ctx.wait_s(0.02)
        return StubResult(n=len(Hands.images))


class Quiet:
    """Hands nothing: another activity, which a run's frame must survive."""

    name = provides = "quiet"
    summary = "A test plugin that hands no frame."
    Params = NoParams
    Result = StubResult
    requires: tuple[type, ...] = ()
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()

    def run(self, microscope: Any, params: NoParams, ctx: RunContext) -> StubResult:
        return StubResult()


class FrameCamera:
    """The six methods of ``Camera``, giving the frame it is handed (copied from the controller tests)."""

    def __init__(self, frame: npt.NDArray[Any], *, exposure_ms: float = 5.0) -> None:
        self.frame = frame
        self._exposure_ms = exposure_ms

    def snap(self) -> npt.NDArray[Any]:
        return self.frame

    def exposure_ms(self) -> float:
        return self._exposure_ms

    def set_exposure_ms(self, value_ms: float) -> float:
        self._exposure_ms = value_ms
        return value_ms

    def image_shape(self) -> tuple[int, int]:
        return (int(self.frame.shape[0]), int(self.frame.shape[1]))

    def bit_depth(self) -> int:
        return 12

    def pixel_size_um(self) -> float:
        return 0.65


def _ep(cls: type) -> EntryPoint:
    return EntryPoint(
        name=cast("Any", cls).name, value=f"{__name__}:{cls.__name__}", group=GROUP
    )


def _image(i: int) -> npt.NDArray[np.uint16]:
    """A 4 x 6 ``uint16`` frame, non-square, whose min is ``1000 i`` and max ``1000 i + 23``."""
    return (np.arange(24, dtype=np.uint16).reshape(4, 6) + 1000 * i).astype(np.uint16)


def _hand(*images: npt.NDArray[Any], pauses: frozenset[int] = frozenset()) -> None:
    Hands.images = list(images)
    Hands.pauses = pauses


@pytest.fixture(autouse=True)
def _fresh_state() -> Iterator[None]:
    """The stub's frames and events are new for every test, and released after it."""
    Hands.images = []
    Hands.pauses = frozenset()
    Hands.handed = [threading.Event() for _ in range(8)]
    Hands.go = [threading.Event() for _ in range(8)]
    Hands.kept = []
    try:
        yield
    finally:
        for event in Hands.go:  # a worker a failed test left behind ends
            event.set()
        Hands.kept = []


@pytest.fixture
def page(fake_microscope: Microscope, tmp_path: Path) -> Iterator[PlatePage]:
    controller = PlatePage(
        fake_microscope,
        tmp_path / "run.json",
        entry_points=[_ep(Hands), _ep(Quiet)],
    )
    try:
        yield controller
    finally:
        controller.close()


def _ended(page: PlatePage) -> None:
    assert page.join(WAIT_S), "the activity did not end"
    assert page.status().running is None


# --- the frames ------------------------------------------------------------------------


def test_a_runs_frame_becomes_the_pages_frame_with_the_next_number(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    # A snap first: the run's frames are numbered after it, one counter for all.
    fake_microscope.override(Camera, FrameCamera(_image(9)))
    page.snap()
    _ended(page)
    snapped = page.frame()
    assert snapped is not None
    assert snapped.number == 1

    _hand(_image(0), _image(1), pauses=frozenset({0}))
    page.run("hands", [])
    assert Hands.handed[0].wait(WAIT_S)
    first = page.frame()
    assert first is not None
    assert first.number == 2
    assert (first.image == _image(0)).all()
    assert first.image.dtype == np.uint16
    assert first.bit_depth == 0  # not read, never guessed
    assert first.exposure_ms is None
    assert first.pixel_size_um == 0.0
    assert first.xy_um == (0.0, 0.0)
    assert first.taken_utc.tzinfo is timezone.utc
    assert first.summary == "hands F0: 4x6 uint16, min 0, max 23"
    # Nothing new handed: the same frame, so no session draws it again (FM-145).
    assert page.frame() is first

    Hands.go[0].set()
    _ended(page)
    second = page.frame()
    assert second is not None
    assert second.number == 3
    assert (second.image == _image(1)).all()
    assert second.xy_um == (100.0, -50.0)
    assert second.summary == "hands F1: 4x6 uint16, min 1000, max 1023"


def test_the_newest_of_several_handed_frames_is_the_one_kept(page: PlatePage) -> None:
    # Fields that come faster than the tick are replaced, never queued (m3 §25.2 D4).
    _hand(_image(0), _image(1), _image(2), pauses=frozenset({2}))
    page.run("hands", [])
    assert Hands.handed[2].wait(WAIT_S)
    frame = page.frame()
    assert frame is not None
    assert frame.number == 1
    assert (frame.image == _image(2)).all()
    assert frame.summary.startswith("hands F2: ")
    Hands.go[2].set()
    _ended(page)
    assert page.frame() is frame  # nothing was left behind in the slot


def test_the_runs_last_frame_is_kept_when_no_session_read_it(page: PlatePage) -> None:
    _hand(_image(0), _image(1))
    page.run("hands", [])
    _ended(page)
    # No session read a frame during the run, and another activity began since:
    # the run's last frame was kept at its end, not left in the slot to be cleared.
    page.run("quiet", [])
    _ended(page)
    frame = page.frame()
    assert frame is not None
    assert frame.number == 1
    assert (frame.image == _image(1)).all()


def test_a_snap_after_a_run_is_not_replaced_by_the_runs_frame(
    page: PlatePage, fake_microscope: Microscope
) -> None:
    _hand(_image(0))
    page.run("hands", [])
    _ended(page)
    # A frame handed after the run's end, by a thread the plugin left behind.
    Hands.kept[0].show_frame(_image(5), label="late")
    fake_microscope.override(Camera, FrameCamera(_image(7), exposure_ms=5.0))
    page.snap()
    _ended(page)
    frame = page.frame()
    assert frame is not None
    # The run's frame was 1 (kept at its end); the snap is 2, and stays the frame.
    assert frame.number == 2
    assert frame.exposure_ms == 5.0
    assert (frame.image == _image(7)).all()
    assert page.frame() is frame


def test_the_kept_frame_is_a_read_only_copy(page: PlatePage) -> None:
    handed = _image(0)
    _hand(handed, pauses=frozenset({0}))
    page.run("hands", [])
    assert Hands.handed[0].wait(WAIT_S)
    frame = page.frame()
    assert frame is not None
    # The plugin writes into its array after the page took it: the page's is its own.
    handed[:] = 4242
    assert (frame.image == _image(0)).all()
    assert not np.shares_memory(frame.image, handed)
    assert not frame.image.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        frame.image[0, 0] = 1
    Hands.go[0].set()
    _ended(page)


def test_a_frame_the_page_cannot_show_is_dropped_and_logged_once(
    page: PlatePage, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="smc.ui")
    _hand(
        np.zeros((4, 6), dtype=np.float32),
        np.zeros((2, 4, 6), dtype=np.uint16),
        np.zeros((4, 6), dtype=np.int16),
        _image(3),
    )
    page.run("hands", [])
    _ended(page)
    dropped = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and r.name == "smc.ui"
    ]
    assert dropped == [
        "hands handed a float32 frame of shape (4, 6); the page shows 2-D uint8 or "
        "uint16 frames"
    ]
    frame = page.frame()
    assert frame is not None
    assert frame.number == 1
    assert (frame.image == _image(3)).all()
    assert page.status().outcome is not None
    assert page.status().outcome.startswith("hands done in ")

    # A new activity logs its first dropped frame again.
    caplog.clear()
    _hand(np.zeros((4, 6), dtype=np.int32))
    page.run("hands", [])
    _ended(page)
    assert [r.getMessage() for r in caplog.records if r.name == "smc.ui"][-1:] == [
        "hands handed a int32 frame of shape (4, 6); the page shows 2-D uint8 or "
        "uint16 frames"
    ]
    assert page.frame() is frame


def test_the_frame_callback_is_not_a_method_of_the_page(
    page: PlatePage, monkeypatch: pytest.MonkeyPatch
) -> None:
    # FM-92: the context keeps its callback for the whole run; a bound method of
    # the page would reach the stand from it.
    received: list[object] = []
    real = controller_module.run_plugin

    def recording(*args: Any, **kwargs: Any) -> Any:
        received.append(kwargs.get("on_frame"))
        return real(*args, **kwargs)

    monkeypatch.setattr(controller_module, "run_plugin", recording)
    page.run("quiet", [])
    _ended(page)
    assert len(received) == 1
    on_frame = received[0]
    owner = getattr(on_frame, "__self__", None)
    assert isinstance(owner, controller_module._LatestShown)
    assert not isinstance(owner, PlatePage)


# --- beyond the plan: frames the tick would fail on --------------------------------


def test_an_empty_frame_is_dropped_before_the_tick_takes_its_minimum(
    page: PlatePage, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="smc.ui")
    _hand(np.zeros((0, 6), dtype=np.uint16))
    page.run("hands", [])
    _ended(page)
    assert page.frame() is None
    assert [r.getMessage() for r in caplog.records if r.name == "smc.ui"] == [
        "hands handed a uint16 frame of shape (0, 6); the page shows 2-D uint8 or "
        "uint16 frames"
    ]


def test_a_frame_of_the_other_byte_order_is_kept_in_the_native_one(
    page: PlatePage,
) -> None:
    # ``display_image`` reads ``uint8`` and ``uint16`` by equality, which a
    # big-endian ``>u2`` fails on a little-endian machine (and the reverse).
    other = ">" if sys.byteorder == "little" else "<"
    swapped = _image(2).astype(f"{other}u2")
    _hand(swapped)
    page.run("hands", [])
    _ended(page)
    frame = page.frame()
    assert frame is not None
    assert frame.image.dtype == np.uint16
    assert (frame.image == _image(2)).all()
    assert display_image(frame.image, bit_depth=frame.bit_depth).shape == (4, 6)
