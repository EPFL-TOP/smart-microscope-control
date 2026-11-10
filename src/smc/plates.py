"""The lab's own plate definitions, registered with ``useq`` when ``smc`` is imported (m3 §22).

Every plate tool names its plate by a ``useq`` key, and a run document stores
that name, not the geometry. A plate whose real pitch is not ``useq``'s (the
Ti2-E's Greiner 24-well, 19.45 mm against ``24-well``'s 19.0, #400) needs a name
of its own, seen by every entry point: the CLI, the page, the plugins and a run
document read back. So the definitions live in one versioned data file shipped
in the package, ``plates.toml``, and ``smc/__init__`` registers them.

This module imports nothing from ``smc``: it runs while ``smc`` is being
imported.
"""

from __future__ import annotations

import logging
import re
import sys
from importlib.resources import files
from typing import Annotated, Any, Literal

import useq
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    ValidationError,
    model_validator,
)

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on the 3.10 CI job
    import tomli as tomllib

__all__ = [
    "PLATES_FILE",
    "LabPlate",
    "PlateSource",
    "load_lab_plates",
    "register_lab_plates",
]

logger = logging.getLogger(__name__)

#: The data file, in the ``smc`` package next to this module.
PLATES_FILE = "plates.toml"

#: ``useq``-style names (D6): the profile's ``[sample] plate``, the page's menu
#: and ``plate-finding`` select well plates by the ``-well`` suffix.
_NAME = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*-well")

#: A length in mm: finite and positive. TOML has ``nan`` and ``inf``, and a
#: ``nan`` pitch would compare false with every bound downstream.
_Mm = Annotated[FiniteFloat, Field(gt=0, strict=True)]

#: What :func:`register_lab_plates` registered, once it has.
_registered: tuple[str, ...] | None = None


class PlateSource(BaseModel):
    """Where a number comes from (D2).

    ``assumed`` is a number taken from elsewhere and not checked on this plate:
    it is never called ``measured`` (anything assumed is labelled as such).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["measured", "maker", "assumed"]
    note: Annotated[str, Field(min_length=1, strict=True)]


class LabPlate(BaseModel):
    """One plate of ``plates.toml``; ``(x, y)`` in mm, the units ``useq`` keeps.

    Rows and columns are counted, so they carry no source; the two measures do.
    Strict types: a ``rows = "4"`` or a ``circular_wells = 1`` is a typo in the
    file, not a value to coerce.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    description: Annotated[str, Field(min_length=1, strict=True)]
    rows: Annotated[int, Field(ge=1, le=64, strict=True)]
    columns: Annotated[int, Field(ge=1, le=96, strict=True)]
    well_spacing_mm: tuple[_Mm, _Mm]
    well_size_mm: tuple[_Mm, _Mm]
    circular_wells: Annotated[bool, Field(strict=True)]
    well_spacing_source: PlateSource
    well_size_source: PlateSource

    @model_validator(mode="after")
    def _wells_fit_their_pitch(self) -> LabPlate:
        """A well as wide as its pitch overlaps its neighbour: the size or the pitch is wrong."""
        if any(
            size >= spacing
            for size, spacing in zip(
                self.well_size_mm, self.well_spacing_mm, strict=True
            )
        ):
            raise ValueError(
                f"well_size_mm {list(self.well_size_mm)} must be below the spacing "
                f"{list(self.well_spacing_mm)} on each axis, or the wells overlap"
            )
        return self


def _validation_message(exc: ValidationError) -> str:
    parts = []
    for error in exc.errors():
        where = ".".join(str(part) for part in error["loc"])
        message = str(error["msg"]).removeprefix("Value error, ")
        parts.append(f"{where}: {message}" if where else message)
    return "; ".join(parts)


def load_lab_plates(text: str | None = None) -> dict[str, LabPlate]:
    """The lab's plates by name, from the packaged ``plates.toml`` or from ``text``.

    ``text`` is for tests. Every refusal names the file and the plate: it is
    raised while ``smc`` is imported, where nothing else says where to look.

    Raises:
        ValueError: The file is not TOML, holds anything but a ``[plates]``
            table, or a plate has a bad name, an unknown key or a bad value.
    """
    if text is None:
        text = files("smc").joinpath(PLATES_FILE).read_text(encoding="utf-8")
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{PLATES_FILE} is not TOML: {exc}") from exc
    unknown = sorted(set(data) - {"plates"})
    if unknown:
        raise ValueError(
            f"{PLATES_FILE}: unknown {', '.join(repr(k) for k in unknown)}; "
            f"the file holds one [plates] table, a sub-table per plate"
        )
    table: Any = data.get("plates", {})
    if not isinstance(table, dict):
        raise ValueError(f"{PLATES_FILE}: [plates] must be a table of plates")
    plates: dict[str, LabPlate] = {}
    for name, fields in table.items():
        if not _NAME.fullmatch(name):
            raise ValueError(
                f"{PLATES_FILE}: plate {name!r}: a name is lower case letters, "
                f"digits and hyphens, and must end in '-well' (such as "
                f"'greiner-24-well')"
            )
        try:
            plates[name] = LabPlate.model_validate(fields)
        except ValidationError as exc:
            raise ValueError(
                f"{PLATES_FILE}: plate {name!r}: {_validation_message(exc)}"
            ) from None
    return plates


def register_lab_plates() -> tuple[str, ...]:
    """Register the lab's plates with ``useq``; the sorted names. Idempotent.

    ``useq`` overwrites a key it holds without a word, and a run document stores
    the name only: a lab file naming ``24-well`` would change the geometry of
    every run that holds it (D4). So a name ``useq`` already has is refused
    before anything is registered, all or none. ``useq`` gets the geometry
    only; its ``name`` stays the key.

    Raises:
        ValueError: ``plates.toml`` is bad (see :func:`load_lab_plates`), or one
            of its names is already in ``useq``'s registry.
    """
    global _registered
    if _registered is not None:
        return _registered
    lab = load_lab_plates()
    taken = sorted(set(lab) & useq.registered_well_plate_keys())
    if taken:
        raise ValueError(
            f"{PLATES_FILE}: {', '.join(repr(name) for name in taken)} already "
            f"named in useq's registry; a lab plate takes a name of its own, or "
            f"every run document that names it would change geometry (m3 §22 D4)"
        )
    useq.register_well_plates(
        {
            name: {
                "rows": plate.rows,
                "columns": plate.columns,
                "well_spacing": plate.well_spacing_mm,
                "well_size": plate.well_size_mm,
                "circular_wells": plate.circular_wells,
            }
            for name, plate in lab.items()
        }
    )
    _registered = tuple(sorted(lab))
    logger.debug("registered the lab's plates with useq: %s", _registered)
    return _registered
