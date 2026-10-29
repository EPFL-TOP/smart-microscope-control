"""The cells slot: what was detected, in a file the result names (m3 §3.5).

The cells live in a file and not in the run document: a 96-well well at 40x is
about 300 frames, and thousands of cells per well would make a document of
megabytes that every checkpoint rewrites. The result keeps only the counts, so
a reader that needs only counts never opens the file, and ``load_cells``
refuses a file that no longer matches them (it was replaced behind the result).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from smc.plugins.api import PluginResult
from smc.plugins.slots.base import SlotValue
from smc.run.files import resolve_in_run, retry_on_permission_error

__all__ = ["Cell", "CellsFile", "CellsResult", "load_cells"]

#: The one version of the cells file this smc reads and writes.
_SCHEMA_VERSION = 1


class Cell(SlotValue):
    """One detected cell: where it is, what it is called and how sure the detector was.

    ``label`` is the issue's ``class``, which is a Python keyword. ``frame`` is
    the index of the frame in ``ScanResult.frames``; ``box_px`` is ``(row0,
    col0, row1, col1)`` in that frame's pixels, for the report's overlays.
    ``x_um``, ``y_um`` are the centre of the box in stage µm (see
    ``smc.plugins.geometry``).
    """

    well: str
    x_um: float
    y_um: float
    label: str
    score: float = Field(ge=0.0, le=1.0)
    frame: int = Field(ge=0)
    box_px: tuple[float, float, float, float]


class CellsFile(BaseModel):
    """The content of a cells file: a schema version and every cell, as ``SlotValue``'s rules."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: int = _SCHEMA_VERSION
    cells: tuple[Cell, ...]


class CellsResult(PluginResult):
    """What the cells stage detected: the counts, and the file that holds the cells.

    ``file`` is ``cells/<token>.json`` relative to the run folder. ``counts``
    maps well to label to count with every label of ``classes`` present, zeros
    included. ``model`` is the weights' file name, never its full path (run
    documents are attached to issues: FM-50), or ``blobs`` for the blob
    detector of the synthetic plate (m3 §9.1); ``model_sha256`` its SHA-256
    (of the detector's identity string for ``blobs``) and ``classes`` the
    detector's, background excluded.
    """

    # ``model`` and ``model_sha256`` are this result's field names; pydantic's
    # default would reserve the ``model_`` prefix. The strict config of
    # ``PluginResult`` is merged, not replaced (a test pins it).
    model_config = ConfigDict(protected_namespaces=())

    file: str
    model: str
    model_sha256: str
    classes: tuple[str, ...]
    device: str
    score_threshold: float
    merge_radius_um: float
    frames: int = Field(ge=0)
    counts: dict[str, dict[str, int]]
    merged: int = Field(ge=0)
    seconds_per_frame: float | None = Field(ge=0.0)


class _NonFiniteConstantError(ValueError):
    """``json.loads`` met ``NaN`` or ``Infinity``, which smc never writes."""


def _refuse_constant(name: str) -> Any:
    raise _NonFiniteConstantError(name)


def load_cells(run_dir: Path, result: CellsResult) -> tuple[Cell, ...]:
    """Read the cells file that ``result`` names, and check it is the one the result counted.

    The file is decoded as ``utf-8-sig`` (a byte-order mark from Notepad is
    accepted, FM-80), its ``schema_version`` is checked on the raw JSON before
    the model sees it (so a newer file is reported as newer, not as a list of
    unknown fields), and its cells are counted per well and label against
    ``result.counts``: ``counts`` carries zeros and the file holds no cell for
    them, which is no difference.

    Raises:
        ValueError: The file is missing, cannot be read, is not UTF-8, is not
            JSON, is of another schema, does not validate, or holds other
            counts than the result. The message starts ``cells file <file> ``
            and ends with the reason; the cause is chained. A reader turns it
            into a ``PluginRunError`` that says to run ``cells`` again.
    """
    prefix = f"cells file {result.file} "
    try:
        path = resolve_in_run(run_dir, result.file)
    except ValueError as exc:
        raise ValueError(f"{prefix}cannot be used: {exc}") from exc
    try:
        data = retry_on_permission_error(path.read_bytes)
    except FileNotFoundError as exc:
        raise ValueError(f"{prefix}is missing from the run folder {run_dir}") from exc
    except OSError as exc:
        raise ValueError(f"{prefix}cannot be read ({exc.strerror or exc})") from exc
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"{prefix}is not UTF-8 text ({exc.reason} at byte {exc.start}); "
            f"save it as UTF-8, not ANSI or UTF-16"
        ) from exc
    try:
        raw = json.loads(text, parse_constant=_refuse_constant)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{prefix}is not valid JSON ({exc.msg} at line {exc.lineno}, "
            f"column {exc.colno})"
        ) from exc
    except _NonFiniteConstantError as exc:
        raise ValueError(f"{prefix}contains {exc}, which JSON does not allow") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{prefix}is not a JSON object")
    version = raw.get("schema_version", _SCHEMA_VERSION)
    # `type is int`: a bool is an int to Python, and 1.0 is not a version.
    if type(version) is not int or version != _SCHEMA_VERSION:
        raise ValueError(
            f"{prefix}has schema {version!r}; this smc reads schema {_SCHEMA_VERSION}"
        )
    try:
        loaded = CellsFile.model_validate(raw)
    except ValidationError as exc:
        problems = [
            f"{'.'.join(str(part) for part in error['loc']) or 'file'}: {error['msg']}"
            for error in exc.errors()
        ]
        more = f" (and {len(problems) - 1} more)" if len(problems) > 1 else ""
        raise ValueError(f"{prefix}does not validate: {problems[0]}{more}") from exc
    _check_counts(prefix, loaded, result)
    return loaded.cells


def _check_counts(prefix: str, loaded: CellsFile, result: CellsResult) -> None:
    """Refuse a file whose (well, label) counts differ from the result's.

    A pair that one side lacks counts as 0, so the zeros that ``counts``
    carries equal the cells the file does not hold. Both sides are compared in
    sorted order, so the first difference named is the same on every run: a
    cell removed from the file, a count changed in the result, or a label the
    counts do not know.
    """
    found: dict[tuple[str, str], int] = {}
    for cell in loaded.cells:
        key = (cell.well, cell.label)
        found[key] = found.get(key, 0) + 1
    counted = {
        (well, label): count
        for well, labels in result.counts.items()
        for label, count in labels.items()
    }
    for key in sorted(found.keys() | counted.keys()):
        have, want = found.get(key, 0), counted.get(key, 0)
        if have != want:
            raise ValueError(
                f"{prefix}differs from the result at {key[0]}/{key[1]} "
                f"(the file holds {have}, the result counted {want})"
            )
