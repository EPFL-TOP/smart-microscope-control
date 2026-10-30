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
from collections.abc import Iterator

import pytest

pytest_plugins = ["smc.testing.fixtures", "pytester"]


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
