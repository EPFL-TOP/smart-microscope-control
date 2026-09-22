import smc


def test_version_is_exposed() -> None:
    assert smc.__version__
    assert smc.__version__ != "0.0.0+unknown"
