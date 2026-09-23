"""Child process that loads device adapters for ``smc discover``.

``python -m smc.discovery._mm_child (--adapters A,B | --all) [--probe NAME]``

It exists so a vendor DLL that hangs or crashes on load takes down this
process and not the survey (see :mod:`smc.discovery.mm_inventory`). The
result is the last line of stdout, JSON with ``ensure_ascii=True``: device
descriptions can hold non-ASCII characters, and a child's stdout on Windows
is the ANSI code page, which cannot encode them.
"""

from __future__ import annotations

import argparse
import json
import sys

from smc.discovery.mm_inventory import ChildResult, list_adapters, probe_adapter


def main(argv: list[str] | None = None) -> int:
    """List adapter devices (and probe one adapter); print the result as JSON."""
    parser = argparse.ArgumentParser(prog="smc.discovery._mm_child")
    parser.add_argument("--adapters", default="", help="comma-separated names")
    parser.add_argument("--all", action="store_true", help="every installed adapter")
    parser.add_argument("--probe", default=None, help="one adapter to initialise")
    args = parser.parse_args(argv)

    from pymmcore_plus import CMMCorePlus

    core = CMMCorePlus()
    names = [n for n in args.adapters.split(",") if n]
    if args.all:
        names = sorted(set(names) | {str(n) for n in core.getDeviceAdapterNames()})
    result = ChildResult(listed=list_adapters(core, names))
    if args.probe:
        result.probe = probe_adapter(args.probe)
    sys.stdout.write(json.dumps(result.model_dump(mode="json"), ensure_ascii=True))
    sys.stdout.write("\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
