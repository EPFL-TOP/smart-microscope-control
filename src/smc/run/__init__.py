"""The run document: what plugins write and read between two invocations (m3 §1.7)."""

from smc.run.document import (
    SCHEMA_VERSION,
    ReadRecord,
    RunDocument,
    RunDocumentError,
    StageRecord,
    StageStatus,
    write_stage,
)

__all__ = [
    "SCHEMA_VERSION",
    "ReadRecord",
    "RunDocument",
    "RunDocumentError",
    "StageRecord",
    "StageStatus",
    "write_stage",
]
