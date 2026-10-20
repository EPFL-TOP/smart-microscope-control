"""Text rendering of an inventory, and the two files ``--out`` writes.

The files are written by the tool itself, in UTF-8, never through shell
redirection: a redirected stream on Windows is cp1252 and cannot encode the
device names a survey collects (#42). The text report is plain (no colour,
120 columns) so it reads the same pasted into a code block.
"""

from __future__ import annotations

import io
import tempfile
from collections.abc import Iterable
from pathlib import Path

from rich import box
from rich.console import Console, RenderableType
from rich.table import Table
from rich.text import Text

from smc.discovery.models import DeviceEntry, Inventory, ProbedDevice, ProbeResult

__all__ = [
    "JSON_NAME",
    "TEXT_NAME",
    "prepare",
    "probe_renderables",
    "render_text",
    "renderables",
    "write",
]

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
    # A Text title: an adapter name like "[x]" must not be read as markup (FM-36).
    table = Table(title=_t(title), title_justify="left", box=box.SIMPLE_HEAD)
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


#: What the ``result`` column says: three words, so that a device nobody
#: initialised is never one of the two that read as an answer.
_RESULT_WORDS = {"ok": "ok", "failed": "FAILED", "not-probed": "not probed"}

_INCOMPLETE = "INCOMPLETE: the adapter child stopped before the last device"


def _detail(device: ProbedDevice) -> str:
    """What a row says beyond its result.

    The error and what it usually means, the reason nothing was tried, or the
    port that was given.
    """
    if device.status == "failed":
        error = device.error or ""
        if device.hint:
            return f"{error} — {device.hint}" if error else device.hint
        return error
    if device.status == "not-probed":
        return device.reason or ""
    return f"port {device.port}" if device.port else ""


def probe_renderables(probe: ProbeResult) -> list[RenderableType]:
    """The probe table, then its error, notes and numbered causes (design m2 §9.6).

    Public since ``smc config build`` prints the walk of its stand and of its
    camera adapter as ``smc discover --probe-adapter`` does (design m2 §10.6).
    """
    title = f"Probe of {probe.adapter}"
    if not probe.complete:
        title += f" — {_INCOMPLETE}"
    table = _table(title, ("device", "type", "from", "result", "detail"))
    if probe.error is not None:
        table.add_row(_t("—"), _t(""), _t(""), _t("ERROR"), _t(probe.error))
    for pd in probe.devices:
        table.add_row(
            _t(pd.name),
            _t(pd.type),
            _t(pd.source),
            _t(_RESULT_WORDS[pd.status]),
            _t(_detail(pd)),
        )
    out: list[RenderableType] = [table]
    out.extend(_t(f"- {note}") for note in probe.notes)
    if probe.causes:
        out.append(_t("In order of likelihood:"))
        out.extend(_t(f"{n}. {cause}") for n, cause in enumerate(probe.causes, 1))
    return out


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
        ("os build", s.os_build),
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
            if a.load_diagnosis is not None:
                adapters.add_row(
                    _t(""), _t(""), _t(""), _t(f"probe: {a.load_diagnosis}")
                )
        elif not a.devices:
            adapters.add_row(_t(a.name), _t("— no devices listed —"), _t(""), _t(""))
        for d in a.devices:
            adapters.add_row(_t(a.name), _t(d.name), _t(d.type), _t(d.description))
        if a.list_meaning is not None:
            # What the list says about the stand, where the list is read.
            adapters.add_row(_t(""), _t(""), _t(""), _t(f"meaning: {a.list_meaning}"))
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
        out.extend(probe_renderables(inv.probe))

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


def prepare(out_dir: Path) -> None:
    """Create ``out_dir`` and prove a file can be written there.

    Called before the survey, so a typo or a read-only share fails in one
    line before minutes of adapter loading rather than after.

    Raises:
        OSError: When the folder cannot be created or written to.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=out_dir):
        pass


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
