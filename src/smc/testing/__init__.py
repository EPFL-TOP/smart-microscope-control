"""Test helpers that plugins and third parties can use too (design §8).

``FakeCore`` lives in the package, not under ``tests/``, so that a plugin's
own tests and M2's quirk tests can build a stand without Micro-Manager. The
pytest plugin, ``smc.testing.fixtures``, is not imported here: it needs
pytest, which is a dev dependency, and ``import smc.testing`` must work
without it.
"""

from smc.testing.fakes import FakeCore, FakeDevice, FakeProperty

__all__ = ["FakeCore", "FakeDevice", "FakeProperty"]
