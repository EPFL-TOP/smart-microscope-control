"""The ``ObjectiveTurret`` contract, on every backend (m2 §8.3, §8.8).

**On ``hardware`` the turret never turns**: a turn can drive an objective
into a loaded plate. There, the only select is of the current objective,
which sends nothing; the cases that turn run on ``fake`` and ``demo``.
"""

from __future__ import annotations

import pytest

from smc.hardware import Microscope
from smc.hardware.capabilities import ObjectiveTurret
from smc.hardware.errors import SafetyRefusedError
from smc.testing import FakeCore


def _another(turret: ObjectiveTurret) -> str:
    current = turret.current()
    return next(label for label in turret.labels() if label != current)


def _never_on_hardware(backend: str) -> None:
    if backend == "hardware":
        pytest.skip("the turret never turns on hardware (m2 §8.8)")


def test_current_is_one_of_the_labels(objective_turret: ObjectiveTurret) -> None:
    labels = objective_turret.labels()
    assert labels
    assert objective_turret.current() in labels


def test_select_without_confirm_is_refused(
    objective_turret: ObjectiveTurret,
    microscope: Microscope,
    backend: str,
    request: pytest.FixtureRequest,
) -> None:
    if not microscope.profile.safety.turret_requires_confirm:
        # Such a stand would turn on this call.
        pytest.skip("the profile sets [safety] turret_requires_confirm = false")
    if backend == "fake":
        request.getfixturevalue("fake_core").log.clear()
    start = objective_turret.current()
    with pytest.raises(SafetyRefusedError) as refused:
        objective_turret.select(_another(objective_turret))
    assert "confirm=True" in refused.value.how_to_force
    assert objective_turret.current() == start
    if backend == "fake":
        fake: FakeCore = request.getfixturevalue("fake_core")
        assert fake.log == []


def test_an_unknown_label_is_refused(objective_turret: ObjectiveTurret) -> None:
    start = objective_turret.current()
    with pytest.raises(ValueError, match=r"has no position 'smc-no-such-objective'"):
        objective_turret.select("smc-no-such-objective", confirm=True)
    assert objective_turret.current() == start


def test_selecting_the_current_objective_returns_it(
    objective_turret: ObjectiveTurret,
) -> None:
    # The one select made on hardware: it sends nothing.
    start = objective_turret.current()
    assert objective_turret.select(start, confirm=True) == start


def test_selecting_another_objective_returns_it_and_reads_back(
    objective_turret: ObjectiveTurret, backend: str
) -> None:
    _never_on_hardware(backend)
    start = objective_turret.current()
    other = _another(objective_turret)
    assert objective_turret.select(other, confirm=True) == other
    assert objective_turret.current() == other
    assert objective_turret.select(start, confirm=True) == start


def test_dry_run_turns_nothing(
    dry_objective_turret: ObjectiveTurret, backend: str
) -> None:
    _never_on_hardware(backend)
    start = dry_objective_turret.current()
    other = _another(dry_objective_turret)
    assert dry_objective_turret.select(other, confirm=True) == other
    assert dry_objective_turret.current() == start


def test_pixel_size_is_unknown_or_the_profiles(
    objective_turret: ObjectiveTurret, microscope: Microscope, backend: str
) -> None:
    sizes = microscope.profile.camera.pixel_size_um
    for label in objective_turret.labels():
        assert objective_turret.pixel_size_um(label) == sizes.get(label, 0.0)
    if backend in ("fake", "demo"):
        assert objective_turret.pixel_size_um("Nikon 10X S Fluor") == 0.65
        assert objective_turret.pixel_size_um("Objective-2") == 0.0
