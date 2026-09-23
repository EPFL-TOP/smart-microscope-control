"""Text rendering of an inventory, and the two files ``--out`` writes.

The files are written by the tool itself, in UTF-8, never through shell
redirection: a redirected stream on Windows is cp1252 and cannot encode the
device names a survey collects (#42). The text report is plain (no colour,
120 columns) so it reads the same pasted into a code block.
"""

from __future__ import annotations

import io
from collections.abc import Iterable
from pathlib import Path

from rich import box
from rich.console import Console, RenderableType
from rich.table import Table
from rich.text import Text

from smc.discovery.models import DeviceEntry, Inventory

__all__ = ["JSON_NAME", "TEXT_NAME", "render_text", "renderables", "write"]

JSON_NAME = "inventory.json"
TEXT_NAME = "inventory.txt"

_WIDTH = 120


def _t(value: object) -> Text:
    # Text, not str: a device name like "[COM3]" must not be read as markup.
    return Text("" if value is None else str(value))


def _ids(vendor_id: str | None, product_id: str | None) -> str:
    if vendor_id is None:
        return ""
    return f"{vendor_id}:{product_id or '????'}"


def _table(title: str, columns: Iterable[str]) -> Table:
    table = Table(title=title, title_justify="left", box=box.SIMPLE_HEAD)
    for column in columns:
        table.add_column(column, overflow="fold")
    return table


def _devices(title: str, entries: list[DeviceEntry]) -> Table:
    table = _table(
        f"{title} ({len(entries)})",
        ("name", "VID:PID", "manufacturer", "class", "status", "serial", "hints"),
    )
    for e in entries:
        table.add_row(
            _t(e.name),
            _t(_ids(e.vendor_id, e.product_id)),
            _t(e.manufacturer),
            _t(e.device_class),
            _t(e.status),
            _t(e.serial_number),
            _t(", ".join(e.hints)),
        )
    return table


def renderables(inv: Inventory) -> list[RenderableType]:
    """The report as rich renderables, for the CLI and for :func:`render_text`."""
    out: list[RenderableType] = []

    system = _table("System", ("", ""))
    system.show_header = False
    s = inv.system
    for key, value in (
        ("host", s.hostname),
        ("collected (UTC)", s.collected_at.isoformat()),
        ("os", s.os),
        ("python", s.python),
        ("smc", s.smc),
        ("pymmcore-plus", s.pymmcore_plus),
        ("Micro-Manager", s.mm_install or "NOT FOUND"),
        ("device API", s.device_api),
        ("other MM installs", ", ".join(s.other_mm_installs) or "none found"),
    ):
        system.add_row(_t(key), _t(value))
    out.append(system)

    installed = inv.adapters.installed
    out.append(
        _t(f"Installed adapters ({len(installed)}): {', '.join(installed) or 'none'}")
    )
    adapters = _table(
        f"Adapter devices ({len(inv.adapters.listed)} adapters listed)",
        ("adapter", "device", "type", "description"),
    )
    for a in inv.adapters.listed:
        if not a.installed:
            adapters.add_row(_t(a.name), _t("— not installed —"), _t(""), _t(""))
        elif a.error is not None:
            adapters.add_row(_t(a.name), _t("ERROR"), _t(""), _t(a.error))
        elif not a.devices:
            adapters.add_row(_t(a.name), _t("— no devices listed —"), _t(""), _t(""))
        for d in a.devices:
            adapters.add_row(_t(a.name), _t(d.name), _t(d.type), _t(d.description))
    out.append(adapters)

    serial = _table(
        f"Serial ports ({len(inv.serial)})",
        ("port", "VID:PID", "manufacturer", "description", "serial", "hints"),
    )
    for p in inv.serial:
        serial.add_row(
            _t(p.device),
            _t(_ids(p.vendor_id, p.product_id)),
            _t(p.manufacturer),
            _t(p.description),
            _t(p.serial_number),
            _t(", ".join(p.hints)),
        )
    out.append(serial)
    out.append(_devices("USB / PnP devices", inv.usb))
    out.append(_devices("PCI devices", inv.pci))

    hints = _table(
        f"Hints ({len(inv.hints)})", ("vendor", "matched", "adapters", "note")
    )
    for h in inv.hints:
        hints.add_row(
            _t(h.vendor), _t(h.matched), _t(", ".join(h.adapters)), _t(h.note)
        )
    out.append(hints)

    if inv.probe is not None:
        probe = _table(
            f"Probe of {inv.probe.adapter}", ("device", "type", "result", "error")
        )
        if inv.probe.error is not None:
            probe.add_row(_t("—"), _t(""), _t("ERROR"), _t(inv.probe.error))
        for pd in inv.probe.devices:
            probe.add_row(
                _t(pd.name), _t(pd.type), _t("ok" if pd.ok else "FAILED"), _t(pd.error)
            )
        out.append(probe)

    out.append(_t(f"Notes ({len(inv.notes)})"))
    out.extend(_t(f"  - {n}") for n in inv.notes)
    return out


def render_text(inv: Inventory) -> str:
    """The report as plain text, 120 columns, no colour."""
    buffer = io.StringIO()
    console = Console(
        file=buffer, width=_WIDTH, color_system=None, force_terminal=False
    )
    for item in renderables(inv):
        console.print(item)
    # Tables pad every cell to the column width; the file needs no padding.
    return "".join(line.rstrip() + "\n" for line in buffer.getvalue().splitlines())


def write(inv: Inventory, out_dir: Path) -> tuple[Path, Path]:
    """Write ``inventory.json`` and ``inventory.txt`` into ``out_dir`` (UTF-8).

    Creates the folder; overwrites both files. The folder name is the
    operator's (the stand name); host and UTC time are inside the files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / JSON_NAME
    text_path = out_dir / TEXT_NAME
    json_path.write_text(inv.model_dump_json(indent=2) + "\n", encoding="utf-8")
    text_path.write_text(render_text(inv), encoding="utf-8")
    return json_path, text_path
