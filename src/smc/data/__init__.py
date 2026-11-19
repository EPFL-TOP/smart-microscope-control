"""Load what a scan acquired on a laptop with no microscope (algorithms design §3).

Nothing in this package imports ``smc.hardware``, ``smc.testing``, ``smc.ui``,
``pymmcore_plus`` or ``pymmcore``, and no module here uses a relative import:
``tests/test_kit_imports.py`` holds both to it (design §8). A collaborator's
laptop has no Micro-Manager, and what is written here must load there.

``smc.data.cli`` (the ``smc dataset`` commands) is not imported here: the
package is a library first, and ``smc.cli`` is what loads the commands.
"""

from __future__ import annotations

from smc.data.scan import DatasetError, Scan, cells_table

__all__ = ["DatasetError", "Scan", "cells_table"]
