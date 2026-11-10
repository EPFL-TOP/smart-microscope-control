"""smart-microscope-control: one control layer, several microscopes, interchangeable tools."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("smart-microscope-control")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0+unknown"

# The lab's own plates (src/smc/plates.toml) are registered with useq here, before
# any lookup: every entry point imports smc first, so the CLI, the page, the
# plugins and a run document read back all see them by name (m3 §22 D3). A bad
# plates.toml fails the import, naming the file and the plate: a plate whose
# numbers are wrong must not be used silently.
from smc.plates import register_lab_plates as _register_lab_plates

_register_lab_plates()

__all__ = ["__version__"]
