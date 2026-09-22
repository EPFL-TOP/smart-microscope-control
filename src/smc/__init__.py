"""smart-microscope-control: one control layer, several microscopes, interchangeable tools."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("smart-microscope-control")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
