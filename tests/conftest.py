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
import logging
import math
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

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
