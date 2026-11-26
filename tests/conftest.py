"""Loads the fixtures every test here may use (design §8).

The fixtures live in the pytest plugin ``smc.testing.fixtures``, so that a
plugin's own repository can load them the same way. It adds ``--profile``
and provides ``mm_available``, ``demo_core``, ``demo_microscope``,
``demo_microscope_dry``, ``demo_microscope_with_sample``, ``fake_core``,
``fake_microscope`` and ``hardware_microscope``.

Two kinds of test exist here (ADR-0005):

* **simulator tests** run against Micro-Manager's demo devices. They are the
  regression bed and run everywhere, including CI. When the adapters are not
  installed they *skip* locally, but *fail* when ``SMC_REQUIRE_MM=1`` (CI),
  so a broken install cannot hide behind a green run.
* **hardware tests** carry ``@pytest.mark.hardware`` and are deselected by
  default. They run only at a microscope: ``pytest -m hardware --profile X``.

``pytest_plugins`` is only allowed in this top-level conftest; ``pytester``
runs the plugin's own tests (``test_testing_fixtures.py``).
"""

from __future__ import annotations

import gc
import html
import logging
import math
import threading
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from smc.plugins.geometry import Matrix2
    from smc.plugins.slots import CameraCalibration

pytest_plugins = ["smc.testing.fixtures", "pytester"]

#: The Ti2-E at 10x, a camera at 180 degrees that is not mirrored (m3 §17.1, §19.1):
#: the image's left is the stage's +X and its bottom the stage's +Y.
TI2E_MATRIX: Matrix2 = ((-1.1, 0.0), (0.0, 1.1))


@pytest.fixture
def no_cyclic_gc() -> Iterator[None]:
    """Only reference counting frees objects: what a CLI command's end looks like (#142).

    The cyclic collector runs at an arbitrary moment, and at exit during
    interpreter finalization. A facade that is freed only by it destroys its
    core there, where a thread that is still inside MMCore aborts the process.
    A test that drops an object under this fixture and finds it alive has
    found a reference cycle.
    """
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        gc.enable()


@pytest.fixture(autouse=True)
def no_model_server_in_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """``SMC_MODEL_SERVER`` and ``SMC_MODEL_TIMEOUT_S`` are unset in every test (inference.md §8.9).

    A developer's PC that has them set, as the acquisition PC does, must not send a
    local test of ``cells`` over the network or wait on a timeout that is not the
    test's (FM-40). A test of the served path sets them itself, or has
    ``FakeModelServer.install`` do it.
    """
    monkeypatch.delenv("SMC_MODEL_SERVER", raising=False)
    monkeypatch.delenv("SMC_MODEL_TIMEOUT_S", raising=False)


@pytest.fixture(autouse=True)
def no_stand_read_outlives_its_test() -> Iterator[None]:
    """A background stand read ends before the next test starts (m3 §15.4, #333).

    ``PlatePage.status()`` during an activity starts a daemon thread
    (``smc-stand-read``) that reads the stand, and a test that ends while it is
    still reading leaves it to run in the next test, against the core the first
    test's fixture closed: its ``state: ... read failed`` DEBUG lines then fall
    inside the window a log-counting test measures, whenever a loaded runner starts
    the thread late (FM-44). The wait is bounded and the thread is asserted gone, so
    a read that a test forgot to release is a named error, not a hang.
    """
    yield
    for thread in threading.enumerate():
        if thread.name == "smc-stand-read":
            thread.join(10)
            assert not thread.is_alive(), "a stand read outlived its test"


@pytest.fixture
def handler_failures() -> Iterator[list[logging.LogRecord]]:
    """The failures of the plate page's handlers during the test; one the test did not expect fails it.

    ``PlatePageView._handler`` swallows what a handler raises (it becomes a line of
    the page), and a test that triggers a handler through a widget asserts effects
    that may all have happened before the failure: a warning escalated to an error
    (FM-147) on a handler's last line would pass every one of them (#277). The page's
    test modules ask for it with ``pytestmark``; a test that expects a failure
    asserts it and clears the list.
    """
    seen: list[logging.LogRecord] = []

    class Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if record.getMessage() == "a handler of the plate page failed":
                seen.append(record)

    collector = Collect()
    logger = logging.getLogger("smc.ui")
    logger.addHandler(collector)
    try:
        yield seen
    finally:
        logger.removeHandler(collector)
    assert not seen, [repr(r.exc_info[1]) if r.exc_info else None for r in seen]


@pytest.fixture
def browser_text() -> Callable[[Any], str]:
    """What the browser shows for a text pane of the plate page, read from the model the browser is sent (FM-271).

    A pane's type and its ``.object`` are Panel's and the page's, not the browser's: Panel
    builds a ``Str`` as ``'<pre>' + str(obj) + '</pre>'`` and the browser parses it, so a
    ``<b>`` in the text is an element and a ``<its run document>`` is an unknown one that
    swallows its words (#625). The Bokeh model of the pane (the pane's clone, which leaves
    the live pane with no model) holds what is sent; ``html.unescape`` **once** gives the
    HTML the browser's own unescape produces, and ``HTMLParser`` gives the elements it
    builds out of it. Unescaping twice would undo the page's own escape and pass a page
    that is still broken, so there is no second pass here. The helper asserts that the one
    element is a ``pre`` (an unknown element counts: it is what ate the words), and no
    comment, declaration or processing instruction either; it returns the concatenated
    text, which is what a reader sees.

    Measured on Panel 1.9.5, 1.10.0 and 1.11.1; the render path is the same on all three.
    """
    from html.parser import HTMLParser

    from bokeh.document import Document

    class Parsed(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.starts: list[str] = []
            self.other: list[str] = []
            self.data: list[str] = []

        def handle_starttag(self, tag: str, attrs: Any) -> None:
            self.starts.append(tag)

        def handle_data(self, data: str) -> None:
            self.data.append(data)

        def handle_comment(self, data: str) -> None:
            self.other.append(f"comment {data!r}")

        def handle_decl(self, decl: str) -> None:
            self.other.append(f"declaration {decl!r}")

        def handle_pi(self, data: str) -> None:
            self.other.append(f"processing instruction {data!r}")

        def unknown_decl(self, data: str) -> None:
            self.other.append(f"declaration {data!r}")

    def read(pane: Any) -> str:
        model = pane.clone().get_root(Document())
        parsed = Parsed()
        parsed.feed(html.unescape(model.text))
        parsed.close()
        assert parsed.starts == ["pre"], (
            f"the browser builds {parsed.starts} from {model.text!r}, not one <pre>"
        )
        assert not parsed.other, f"the browser reads {parsed.other} in {model.text!r}"
        return "".join(parsed.data)

    return read


@pytest.fixture
def seed_camera_calibration() -> Callable[..., CameraCalibration]:
    """A factory that writes a ``done`` ``camera-calibration`` record into a run document (#327).

    The plate page maps the walls an operator names on the image to the stage's
    with this record's matrix (m3 §19), so a test that captures a wall or drives to
    one first seeds a calibration, as ``tests/test_scan_plan_demo.py``'s ``_seed``
    seeds one. ``matrix`` is the camera's, the Ti2-E's at 10x by default; its pixel
    size, rotation and mirror are read from it, written here and not taken from the
    code under test (``M = ps * R(theta) * diag(s, -1)``, ``s`` -1 when mirrored, so
    a mirrored camera has a positive determinant). ``profile`` is the stand under
    test's ``microscope.name``: ``demo`` on the demo and the fake stands.
    """
    from smc.plugins.slots import CameraCalibration
    from smc.run import StageRecord, StageStatus, write_stage

    def seed(
        run_doc: Path,
        matrix: Matrix2 = TI2E_MATRIX,
        *,
        profile: str = "demo",
        dry_run: bool = False,
        objective: str | None = None,
    ) -> CameraCalibration:
        (m00, m01), (m10, m11) = matrix
        determinant = m00 * m11 - m01 * m10
        mirrored = determinant > 0.0
        sign = -1.0 if mirrored else 1.0
        calibration = CameraCalibration(
            objective=objective,
            image_shape_px=(384, 512),
            matrix_um_per_px=matrix,
            pixel_size_um=math.sqrt(abs(determinant)),
            rotation_deg=math.degrees(math.atan2(sign * m10, sign * m00)),
            mirrored=mirrored,
            residual_um=0.01,
            response=0.97,
            return_error_um=0.4,
            step_um=48.0,
        )
        then = datetime(2026, 11, 9, 9, 30, tzinfo=timezone.utc)
        write_stage(
            run_doc,
            StageRecord(
                plugin="camera-calibration",
                provides="camera-calibration",
                attempt="a1",
                status=StageStatus.done,
                started_utc=then,
                finished_utc=then,
                profile=profile,
                dry_run=dry_run,
                smc_version="0",
                params={},
                result=calibration.model_dump(mode="json"),
            ),
        )
        return calibration

    return seed
