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

pytest_plugins = ["smc.testing.fixtures", "pytester"]
