"""Test helpers that plugins and third parties can use too (design §8).

``FakeCore`` lives in the package, not under ``tests/``, so that a plugin's
own tests and M2's quirk tests can build a stand without Micro-Manager.
``PlateSample`` and ``SampleCamera`` give a stand frames that follow its
stage, for code that looks at images to decide where to go. ``StubDetector``
is a scripted detector for a stage that asks a model for cells, and
``FakeModelServer`` a model server with no socket that records what it was
asked. The pytest
plugin, ``smc.testing.fixtures``, is not imported here: it needs pytest,
which is a dev dependency, and ``import smc.testing`` must work without it.
"""

from smc.testing.detectors import StubDetector, stub_model
from smc.testing.fake_model_server import FakeModelServer, ServedCall
from smc.testing.fakes import FakeCore, FakeDevice, FakeProperty
from smc.testing.synthetic import Blob, PlateSample, SampleCamera

__all__ = [
    "Blob",
    "FakeCore",
    "FakeDevice",
    "FakeModelServer",
    "FakeProperty",
    "PlateSample",
    "SampleCamera",
    "ServedCall",
    "StubDetector",
    "stub_model",
]
