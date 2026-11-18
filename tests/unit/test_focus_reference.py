"""Focus references survive the run document (m4 §11.5, §11.8).

A tool that visits positions keeps one ``FocusReference`` per position in its
result, so that a later run restores each plane. ``FocusReference`` is a frozen
``slots`` dataclass, not a pydantic model: pydantic's handling of one was
measured on 2.13.5 only, and CI's oldest job resolves the oldest pydantic
smc allows, so the round trip is proven here, through the file.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from smc.hardware.capabilities import FocusReference
from smc.run import RunDocument, StageRecord, StageStatus, write_stage

#: DF2's references are 9 bytes on the stand; these have bytes above 0x7F, so
#: that only base64 (ASCII) can carry them through a JSON file.
BLOB = bytes((0xDF, 0x02, 0xF0, 0x80, 0xFF, 0x00, 0x01, 0x7F, 0xC3))


class _VisitResult(BaseModel):
    """A visiting tool's result, as #559's will hold its references."""

    references: dict[str, FocusReference]


def _encoded(blob: bytes) -> str:
    return base64.b64encode(blob).decode("ascii")


def test_references_round_trip_through_the_run_document(tmp_path: Path) -> None:
    result = _VisitResult(
        references={
            "A1": FocusReference("DefiniteFocus", _encoded(BLOB), "Fluar 10x/0.50"),
            "B2": FocusReference("DefiniteFocus", _encoded(BLOB[::-1]), None),
        }
    )
    now = datetime.now(timezone.utc)
    record = StageRecord(
        plugin="visit",
        provides="visits",
        attempt="attempt-1",
        status=StageStatus.done,
        started_utc=now,
        finished_utc=now,
        profile="zeiss-observer7",
        dry_run=False,
        smc_version="0",
        params={},
        result=result.model_dump(mode="json"),
    )
    path = tmp_path / "run.json"
    write_stage(path, record)
    path.read_bytes().decode("ascii")  # the document stays ASCII
    stored = RunDocument.load(path).stages["visit"].result
    assert stored is not None
    back = _VisitResult.model_validate(stored)
    assert back == result
    assert isinstance(back.references["A1"], FocusReference)
    assert base64.b64decode(back.references["A1"].data, validate=True) == BLOB
    assert back.references["B2"].objective is None
