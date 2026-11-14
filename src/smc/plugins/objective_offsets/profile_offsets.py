"""What ``objective-offsets`` measured, as the profile's ``[objectives]`` would hold it, and the write (m3 §20.11).

``smc objective offset --from-run-doc`` and the plate page's offsets block (m5 §18)
show the same preview and make the same write. The logic lives here, once, so that
the page can never show lines the command would not print, or write lines it did not
show (FM-172):

* ``plan_offsets`` opens no file and no stand. It takes a profile and a run document
  already loaded and returns the new section with the lines that say what it
  replaces, in the command's order and words (m3 §20.7).
* ``write_offsets`` writes a plan into the profile's own file, atomically, and says in
  one line why it could not (FM-38).

Every refusal is an ``OffsetsWriteError`` whose text is the text of a ``✗`` line.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from smc.hardware.errors import ProfileError
from smc.hardware.profile import ObjectivesSection, Profile, profile_text
from smc.plugins.slots import ObjectiveOffsetsResult
from smc.run import RunDocument

__all__ = ["OffsetsPlan", "OffsetsWriteError", "plan_offsets", "write_offsets"]


class OffsetsWriteError(Exception):
    """The offsets cannot be planned or written; ``str(exc)`` is the text of a ``✗`` line, with no prefix."""


@dataclass(frozen=True, slots=True)
class OffsetsPlan:
    """What a write would put into the profile, and the lines that say so.

    ``profile`` is the profile's name (``[microscope] name``) and ``source`` the file
    a write replaces. ``lines`` are plain text (FM-36), in the order of m3 §20.7:
    per target its shift, per written label what it was, per dropped Z the reason,
    the reference and the ``measured`` text. ``new_reference`` is ``True`` when the
    profile named no reference before.
    """

    profile: str
    source: Path
    section: ObjectivesSection
    lines: tuple[str, ...]
    new_reference: bool


def plan_offsets(
    profile: Profile, document: RunDocument, run_doc: Path, *, session: str | None
) -> OffsetsPlan:
    """The section ``document``'s ``objective-offsets`` result would write into ``profile``, with the preview's lines (m3 §20.7, steps 1 to 5).

    Entries are replaced, not merged (FM-99): each label the run measured gets a new
    entry, and one it held a Z for loses it, which a line says. ``run_doc`` is only
    the name the texts use: its ``as_posix()`` in ``in <where>: ...`` and its
    ``name`` alone in ``measured``, never its folder (FM-50). A profile with no
    ``source`` is refused here too, though the command and the page each check it
    first and say it in their own place.

    Raises:
        OffsetsWriteError: The profile has no file; the session is more than one
            line; the document has no usable result of this profile, or one this
            smc does not read; the profile's reference is no longer the one the run
            measured from; or the section would not be valid in the profile.
    """
    name = profile.microscope.name
    source = profile.source
    if source is None:
        raise OffsetsWriteError(
            f"{name} is not loaded from a file: give the path of its .toml to -p"
        )
    if session is not None and any(ord(char) < 0x20 for char in session):
        raise OffsetsWriteError(
            "the session is one line of text: no tab, no line break"
        )

    where = run_doc.as_posix()
    reason = document.unusable("objective-offsets", profile=name, dry_run=False)
    record = document.provider("objective-offsets")
    if reason is not None or record is None or record.result is None:
        raise OffsetsWriteError(f"in {where}: {reason or 'it holds no result'}")
    result: ObjectiveOffsetsResult | None = None
    try:
        result = ObjectiveOffsetsResult.model_validate(record.result)
    except ValidationError:
        result = None
    if result is None or record.finished_utc is None:
        raise OffsetsWriteError(
            f"the objective-offsets result in {where} is not one this smc "
            "reads; run objective-offsets again"
        )

    # The reference now: the offsets are the run's only if the profile's reference is
    # still the one it measured them from.
    now = profile.objectives.reference.strip() or None
    measured_from = result.reference if result.reference_in_profile else None
    if now != measured_from:
        raise OffsetsWriteError(
            f"the profile's offsets are now from {now or '(none)'}, and the run "
            f"measured them from {measured_from or '(none)'}: run "
            "objective-offsets again"
        )

    date = record.finished_utc.astimezone(timezone.utc).date().isoformat()
    by = f"{session}, " if session else ""
    measured = f"{date}, {by}objective-offsets in {run_doc.name}"
    old = profile.objectives.offsets
    entries: dict[str, Any] = {
        label: entry.model_dump() for label, entry in old.items()
    }
    written: dict[str, tuple[float, float]] = {}
    for offset in result.offsets:
        x_um, y_um = (round(value, 2) + 0.0 for value in offset.xy_offset_um)
        written[offset.objective] = (x_um, y_um)
        entries[offset.objective] = {
            "xy_offset_um": [x_um, y_um],
            "measured": measured,
        }
    problem = ""
    try:
        section = ObjectivesSection.model_validate(
            {"reference": result.reference, "offsets": entries}
        )
    except ValidationError as exc:
        problem = "; ".join(
            str(error["msg"]).removeprefix("Value error, ") for error in exc.errors()
        )
    if problem:
        # Text, never the error: nothing here keeps it (FM-98).
        raise OffsetsWriteError(
            f"the offsets would not be valid in the profile: {problem}"
        )

    lines = _preview_lines(result, profile, written, measured, now is None)
    return OffsetsPlan(
        profile=name,
        source=source,
        section=section,
        lines=lines,
        new_reference=now is None,
    )


def _preview_lines(
    result: ObjectiveOffsetsResult,
    profile: Profile,
    written: dict[str, tuple[float, float]],
    measured: str,
    new_reference: bool,
) -> tuple[str, ...]:
    """The lines of the preview, in the order the design fixes: targets, labels, dropped Z, reference, text (m3 §20.7).

    Plain text, never markup: a label or a measured text holds brackets and quotes
    that a console would read as styles (FM-36). A value is signed and rounded as the
    read-out prints it, and ``+ 0.0`` keeps a ``-0.0`` from showing.
    """
    old = profile.objectives.offsets
    lines: list[str] = []
    for measure in result.measures:
        x_um, y_um = (value + 0.0 for value in measure.shift_xy_um)
        wells = ", ".join(well.well for well in measure.wells)
        lines.append(
            f"{measure.objective!r}  shift from {result.plate_objective!r} "
            f"({x_um:+.2f}, {y_um:+.2f}) µm, spread {measure.spread_um:.2f} µm over "
            f"{wells} (bound {measure.max_spread_um:.2f} µm)"
        )
    for label, (x_um, y_um) in written.items():
        before = old.get(label)
        if before is None or before.xy_offset_um is None:
            was = "was unknown"
        else:
            old_x, old_y = (value + 0.0 for value in before.xy_offset_um)
            was = f"was xy ({old_x:+.2f}, {old_y:+.2f}) µm"
            if before.measured:
                was += f", measured {before.measured}"
        lines.append(f"{label!r}  xy ({x_um:+.2f}, {y_um:+.2f}) µm  {was}")
    for label in written:
        before = old.get(label)
        if before is not None and before.z_offset_um is not None:
            lines.append(
                f"{label!r}: its z_offset_um {before.z_offset_um + 0.0:+.2f} µm "
                f"(measured {before.measured}) is dropped: objective-offsets "
                "measures XY only"
            )
    new = "  (new: the profile named none)" if new_reference else ""
    lines.append(f"reference {result.reference!r}{new}")
    lines.append(f'measured = "{measured}"')
    return tuple(lines)


def write_offsets(plan: OffsetsPlan) -> None:
    """Replace the ``[objectives]`` block of ``plan.source`` by ``plan.section``, atomically (m3 §20.7, step 6).

    Only the write needs ``tomli-w``: a PC updated with ``git pull`` and not
    reinstalled still shows the preview, and is told here what is missing (as
    ``smc presets save`` is). The writer is imported at the call, and so are the
    atomic writer and its error, so that ``smc --help`` and the preview load no more
    than they did. Nothing is written when ``with_objectives`` refuses.

    Raises:
        OffsetsWriteError: ``tomli-w`` is missing; the profile's text cannot be read
            or the block cannot be put into it (a damaged block, an ``[objectives]``
            table outside it); or the file could not be replaced (it is as it was,
            and the whole new text is kept when only the rename failed).
    """
    from smc.run.files import AtomicWriteError, write_atomically

    missing = False
    try:
        from smc.hardware.profile_write import with_objectives
    except ImportError:
        missing = True
    if missing:
        raise OffsetsWriteError(
            "writing the profile needs the tomli-w package, which this install "
            "lacks: run pip install -e . in the smc folder, then write again"
        )
    source = plan.source
    try:
        text = with_objectives(profile_text(source), plan.section, where=str(source))
    except ProfileError as exc:
        raise OffsetsWriteError(str(exc)) from exc
    failed = ""
    try:
        write_atomically(source, text.encode("utf-8"), make_parents=False)
    except AtomicWriteError as exc:
        # The profile is as it was. When only the rename failed (Windows: an editor or a
        # sync client holds the file), the whole new text is kept.
        failed = f"{source} was not changed: {exc.strerror or exc}" + (
            f"; the whole new text is kept in {exc.kept}: close what holds the "
            "profile and write again, then delete that file"
            if exc.kept
            else ""
        )
    if failed:
        raise OffsetsWriteError(failed)
