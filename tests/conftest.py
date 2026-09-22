"""Shared fixtures.

Two kinds of test exist here (ADR-0005):

* **simulator tests** run against Micro-Manager's demo devices. They are the
  regression bed and run everywhere, including CI. When the adapters are not
  installed they *skip* locally, but *fail* when ``SMC_REQUIRE_MM=1`` (CI),
  so a broken install cannot hide behind a green run.
* **hardware tests** carry ``@pytest.mark.hardware`` and are deselected by
  default. They run only at a microscope: ``pytest -m hardware --profile X``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from smc.hardware.core import close_core, find_install, open_core


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--profile",
        default=None,
        help="Microscope profile for hardware tests (e.g. nikon-ti2).",
    )


@pytest.fixture(scope="session")
def mm_available() -> bool:
    """Whether the Micro-Manager device adapters are installed."""
    available = find_install() is not None
    if not available and os.environ.get("SMC_REQUIRE_MM") == "1":
        pytest.fail(
            "SMC_REQUIRE_MM=1 but Micro-Manager adapters are missing — "
            "run `mmcore install --test-adapters`."
        )
    return available


@pytest.fixture
def demo_core(mm_available: bool) -> Iterator[object]:
    """A fresh core loaded with the demo configuration, released afterwards."""
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    core = open_core(None)
    try:
        yield core
    finally:
        close_core(core)
