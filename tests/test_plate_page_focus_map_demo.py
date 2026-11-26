"""The page keeps the 10x scan as the focus map and gives it to a map scan, on the demo-sample stand, through the view's own handlers (m5 §25 D1 to D4, #634).

The stand is ``demo-sample`` with the demo stage's ``Velocity`` raised on its own core, the camera
calibrated and the plate registered from the sample's references at the 10x. A module fixture builds
``run.json`` once through the view (the calibration, the registration and *Lay out the fields* on
``B2`` at ``coverage`` 0.4: 36 fields); each test copies it into its own folder, opens its own stand
and drives the Scan screen: a click is ``button.clicks += 1``, the autofocus is the drive bar's *Autofocus
on* (the demo's locks at once, so every frame of the scan reads ``locked``, measured on the demo).

What is checked is what the owner will check at the Ti2-E: the copy is the run document **as it was at
the end of the scan** (a byte for byte copy taken before anything else runs), the kept file is replaced
by the next scan taken locked and left alone by one that was not, the pair ``focus_map=<the kept file>``
is the one the click sends and the command shows, and a copy that cannot be written costs the scan
nothing but a tail on its outcome. The texts are the design's, written out whole (FM-45).
"""

from __future__ import annotations

import os
import shlex
import shutil
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

if os.environ.get("SMC_REQUIRE_UI"):
    import panel as pn  # noqa: F401  (CI must fail, not skip, when the extra is missing)
else:
    pytest.importorskip("panel")

import smc.ui.controller as controller_module
from smc.hardware import Microscope
from smc.hardware.profile import Profile
from smc.run import RunDocument, StageStatus
from smc.ui.controller import PlatePage
from smc.ui.guide import FOCUS_MAP_NONE
from smc.ui.plate_page import PlatePageView

pytestmark = [
    pytest.mark.demo,
    # A failure a handler swallowed fails the test that did not expect it (#277).
    pytest.mark.usefixtures("handler_failures"),
]

DEMO_SAMPLE = Path(__file__).resolve().parents[1] / "profiles" / "demo-sample.toml"
TEN = "Nikon 10X S Fluor"
#: B2 at ``coverage`` 0.4 is 36 fields, all of them locked with the demo's autofocus on.
FIELDS = 36
KEPT_NAME = "run.focus-map.json"
ANCHOR_AT_B2 = (
    "Before you start: with Live, focus by eye on the cells near the centre of B2. "
    "That focus is the scan's starting point; each well is then anchored near its centre."
)
#: What ``scan`` itself says to ``cell_focus=map`` with no ``focus_map`` (its validator's words).
TOOLS_OWN_REFUSAL = (
    "set focus_map to a run document whose scan was taken with the autofocus locked"
)


def _click(button: Any) -> None:
    button.clicks += 1


@dataclass(frozen=True)
class Rig:
    """One test's page, its view, its stand and the run document it writes."""

    page: PlatePage
    view: PlatePageView
    stand: Microscope
    run_doc: Path

    @property
    def kept(self) -> Path:
        return self.run_doc.parent / KEPT_NAME

    def finish(self) -> None:
        """Let the run end as a session would: ticks, then the redraw the end of a run asks for."""
        started = time.monotonic()
        while not self.page.join(0.01):
            self.view.tick()
            assert time.monotonic() - started < 300, "the run did not end"
        self.view.tick()
        self.view.refresh()

    def autofocus(self, on: bool) -> None:
        """*Autofocus on* or *off*, as the drive bar's buttons do, and the tick that draws it."""
        _click(self.view.autofocus_on_button if on else self.view.autofocus_off_button)
        self.finish()

    def scan(self) -> str:
        """*Start the scan* on the Scan screen; the outcome the strip says when it ends."""
        self.view.show_step("scan")
        self.view.refresh()
        self.view.tick()
        _click(self.view.scan_button)
        self.finish()
        outcome = self.page.status().outcome
        assert outcome is not None
        return outcome

    def record(self) -> Any:
        return RunDocument.load(self.run_doc).stages["scan"]

    def type_into_the_scan(self, **fields: str) -> None:
        """Type into the scan's own fields, then the redraw the page makes at each change."""
        for key, text in fields.items():
            self.view.tool_fields["scan"][key].value = text
        self.view.tick()


@pytest.fixture(scope="module")
def run_json(mm_available: bool, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The camera calibrated, the plate registered and B2's 36 fields laid out at the 10x, all through the view's handlers."""
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    folder = tmp_path_factory.mktemp("page-focus-map")
    run_doc = folder / "run.json"
    stand = Microscope.open(Profile.load(DEMO_SAMPLE))
    stand.core.setProperty("XY", "Velocity", "100000")
    page = PlatePage(stand, run_doc)
    try:
        view = PlatePageView(page)
        rig = Rig(page, view, stand, run_doc)
        for step, button in (
            ("stand", view.calibrate_button),
            ("plate", view.register_button),
        ):
            view.show_step(step)
            view.refresh()
            view.tick()
            _click(button)
            rig.finish()
        view.show_step("wells")
        page.select_text("B2")
        view.tool_fields["scan-plan"]["coverage"].value = "0.4"
        view.refresh()
        view.tick()
        _click(view.layout_button)
        rig.finish()
        stages = RunDocument.load(run_doc).stages
        assert [
            stages[name].status for name in ("camera-calibration", "scan-plan")
        ] == [
            StageStatus.done,
            StageStatus.done,
        ]
        assert "plate-registration" in stages
    finally:
        page.close()
        stand.close()
    return run_doc


@pytest.fixture
def make_rig(run_json: Path, tmp_path: Path) -> Iterator[Callable[..., Rig]]:
    """A stand, a page and a view over a copy of ``run.json``; what it opened is closed afterwards."""
    opened: list[tuple[PlatePage, Microscope]] = []

    def make(*, dry_run: bool = False) -> Rig:
        run_doc = tmp_path / "run.json"
        shutil.copy(run_json, run_doc)
        stand = Microscope.open(Profile.load(DEMO_SAMPLE), dry_run=dry_run)
        # The demo stage moves at its `Velocity` property, 10 mm/s by default.
        stand.core.setProperty("XY", "Velocity", "100000")
        page = PlatePage(stand, run_doc)
        opened.append((page, stand))
        view = PlatePageView(page)
        view.show_step("scan")
        view.refresh()
        view.tick()
        return Rig(page, view, stand, run_doc)

    try:
        yield make
    finally:
        for page, stand in opened:
            page.close()
            stand.close()


def _locked(record: Any) -> list[str]:
    return [frame["focus"] for frame in record.result["frames"]]


def test_a_scan_with_locked_frames_keeps_a_copy_of_the_run_document(
    make_rig: Callable[..., Rig],
) -> None:
    rig = make_rig()
    assert not rig.kept.exists()
    rig.autofocus(True)

    outcome = rig.scan()

    # The scan ended `done`, every frame was taken with the autofocus locked, and nothing else
    # ran on the page since: the kept file is the run document's own bytes, plan and scan together.
    assert rig.record().status is StageStatus.done, outcome
    assert _locked(rig.record()) == ["locked"] * FIELDS
    assert rig.kept.read_bytes() == rig.run_doc.read_bytes()
    assert rig.page.focus_map_path == rig.kept
    # Nothing is said of it on the strip: a copy that was made needs no word.
    assert "focus map" not in outcome
    # The copy is a run document a map scan reads: its scan and its plan are the 10x's.
    kept = RunDocument.load(rig.kept)
    assert kept.stages["scan"].attempt == rig.record().attempt
    assert kept.stages["scan-plan"].status is StageStatus.done


def test_a_later_locked_scan_replaces_the_kept_map(
    make_rig: Callable[..., Rig],
) -> None:
    rig = make_rig()
    rig.autofocus(True)
    rig.scan()
    first = rig.kept.read_bytes()
    first_attempt = rig.record().attempt

    rig.scan()

    # One file, replaced: the second scan's run document, not the first's.
    assert rig.record().attempt != first_attempt
    assert rig.kept.read_bytes() == rig.run_doc.read_bytes()
    assert rig.kept.read_bytes() != first
    assert sorted(path.name for path in rig.run_doc.parent.glob("*.focus-map*")) == [
        KEPT_NAME
    ]


def test_a_scan_with_no_locked_frame_leaves_the_kept_map_as_it_was(
    make_rig: Callable[..., Rig],
) -> None:
    rig = make_rig()
    # The autofocus is off from the start: the frames read `off`, and there is no map to keep.
    outcome = rig.scan()
    assert rig.record().status is StageStatus.done, outcome
    assert set(_locked(rig.record())) == {"off"}
    assert not rig.kept.exists()

    # A map is kept by a locked scan, and a later scan that was not locked leaves it alone: the
    # kept file is the 10x's, and what the 40x scan wrote is a different run document.
    rig.autofocus(True)
    rig.scan()
    kept = rig.kept.read_bytes()
    rig.autofocus(False)
    rig.scan()
    assert set(_locked(rig.record())) == {"off"}
    assert rig.kept.read_bytes() == kept
    assert rig.run_doc.read_bytes() != kept


def test_a_dry_run_scan_keeps_nothing(
    make_rig: Callable[..., Rig], monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = make_rig(dry_run=True)
    # A dry run walks the moves and takes no frame, which already leaves nothing to keep; the page's
    # own guard is shown with a record that does hold locked frames (a third party's scan, or a
    # change in what a dry run records), so that it is the guard that is tested and not the tool.
    monkeypatch.setattr(controller_module, "_holds_a_locked_frame", lambda record: True)

    outcome = rig.scan()

    assert rig.record().status is StageStatus.done, outcome
    assert rig.record().dry_run is True
    assert not rig.kept.exists()


def test_a_copy_that_cannot_be_written_leaves_the_scan_done_and_says_so(
    make_rig: Callable[..., Rig], monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = make_rig()
    rig.autofocus(True)

    def refused(path: Path, data: bytes) -> None:
        raise PermissionError(13, "Permission denied")

    # Windows refuses the rename onto a file another program holds open (FM-24): the name the page
    # imported, so that this is the one place the copy is written through.
    monkeypatch.setattr(controller_module, "write_atomically", refused)

    outcome = rig.scan()

    assert rig.record().status is StageStatus.done
    assert len(rig.record().result["frames"]) == FIELDS
    assert outcome.startswith("Scan done in ")
    assert outcome.endswith(
        "; the focus map was not kept: [Errno 13] Permission denied"
    )
    assert not rig.kept.exists()
    # The strip says it too, and the activity ended as a done one (the next scan is not blocked).
    assert rig.page.status().running is None
    assert rig.view.scan_button.disabled is False


def test_a_rename_refused_names_the_kept_file_and_where_the_whole_copy_is(
    make_rig: Callable[..., Rig], monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = make_rig()
    rig.autofocus(True)
    real_replace = os.replace

    def locked(src: Any, dst: Any) -> None:
        # What CPython raises on Windows for a replace onto a file another program holds open:
        # its `filename` is the SOURCE, the temporary, which is not what is locked (FM-24).
        if str(dst).endswith(".focus-map.json"):
            raise PermissionError(13, "Access is denied", src, None, dst)
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", locked)

    outcome = rig.scan()

    # The scan is done; the line names the kept file and says where the whole copy is, so that it
    # can be renamed by hand, and the temporary is the copy of the run document as it was.
    assert rig.record().status is StageStatus.done
    left = sorted(rig.run_doc.parent.glob(".run.focus-map.json.*.smc-tmp"))
    assert len(left) == 1
    assert left[0].read_bytes() == rig.run_doc.read_bytes()
    assert outcome.endswith(
        "; the focus map was not kept: Access is denied (run.focus-map.json); the whole "
        f"copy is in {left[0].name}: rename it to run.focus-map.json once nothing holds "
        "that file open"
    )
    assert not rig.kept.exists()


def test_the_copy_reads_the_run_document_with_the_retry_a_busy_file_needs(
    make_rig: Callable[..., Rig], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Antivirus or a sync client may hold the run document for a moment on Windows (FM-24): the
    # page's own read of it for the copy goes through the retry the run document's reads use.
    rig = make_rig()
    rig.autofocus(True)
    seen: list[str] = []
    real = controller_module.retry_on_permission_error

    def spying(action: Callable[[], Any]) -> Any:
        seen.append(getattr(action, "__name__", "?"))
        return real(action)

    monkeypatch.setattr(controller_module, "retry_on_permission_error", spying)

    rig.scan()

    assert seen == ["read_bytes"]
    assert rig.kept.read_bytes() == rig.run_doc.read_bytes()


def test_map_with_no_typed_focus_map_is_given_the_kept_file_before_the_parse(
    make_rig: Callable[..., Rig],
) -> None:
    rig = make_rig()
    rig.autofocus(True)
    rig.scan()
    kept = rig.kept.read_bytes()
    rig.autofocus(False)
    rig.type_into_the_scan(cell_focus="map", autofocus="off")

    outcome = rig.scan()

    # The tool refuses `cell_focus=map` with no `focus_map`, so the pair had to be there before the
    # parse: the scan ran, and its record holds the page's kept file as typed on the command line.
    record = rig.record()
    assert record.status is StageStatus.done, outcome
    assert record.params["cell_focus"] == "map"
    assert os.path.samefile(record.params["focus_map"], rig.kept)
    assert Path(record.params["focus_map"]).is_absolute()
    assert record.result["focus_map"]["scan_attempt"] is not None
    # The map scan was taken with the autofocus off, so it kept nothing of its own.
    assert set(_locked(record)) == {"off"}
    assert rig.kept.read_bytes() == kept


def test_a_typed_focus_map_wins_and_null_is_none(
    make_rig: Callable[..., Rig], tmp_path: Path
) -> None:
    rig = make_rig()
    rig.autofocus(True)
    rig.scan()
    typed = tmp_path / "ten.json"
    shutil.copy(rig.kept, typed)
    rig.autofocus(False)

    # A typed `focus_map` is the one the click sends, and the page adds no second pair.
    rig.type_into_the_scan(cell_focus="map", autofocus="off", focus_map=str(typed))
    assert shlex.split(rig.view.scan_command.object).count("--param") == 3
    outcome = rig.scan()
    assert rig.record().status is StageStatus.done, outcome
    assert os.path.samefile(rig.record().params["focus_map"], typed)

    # `focus_map=null` is none as the parser reads it: replaced by the kept file, never met by a
    # second `focus_map` pair (``'focus_map' given twice``).
    rig.type_into_the_scan(focus_map="null")
    outcome = rig.scan()
    assert rig.record().status is StageStatus.done, outcome
    assert os.path.samefile(rig.record().params["focus_map"], rig.kept)


def test_map_with_no_kept_file_waits_with_the_tools_own_words(
    make_rig: Callable[..., Rig],
) -> None:
    rig = make_rig()
    rig.type_into_the_scan(cell_focus="map", autofocus="off")
    assert not rig.kept.exists()

    # Nothing is added: the command is the call as typed, and the click is the tool's own refusal.
    # The page does not wait for a map it has not got (`_blocked` reads no pair): the button is
    # there and says nothing, and it is the click that is refused, in the tool's words.
    assert rig.view.scan_button.disabled is False
    assert rig.view.scan_wait_text.visible is False
    assert "focus_map=" not in rig.view.scan_command.object
    assert rig.view.focus_map_text.object.splitlines() == [
        FOCUS_MAP_NONE,
        ANCHOR_AT_B2,
    ]
    before = RunDocument.load(rig.run_doc).stages.keys()
    _click(rig.view.scan_button)
    rig.finish()
    outcome = rig.page.status().outcome
    assert outcome is not None
    assert outcome.startswith("scan: ParamsError: ")
    assert TOOLS_OWN_REFUSAL in outcome
    # Refused before anything was written, and nothing was kept.
    assert RunDocument.load(rig.run_doc).stages.keys() == before
    assert not rig.kept.exists()


def test_the_wait_and_the_command_name_the_focus_map_the_click_gives(
    make_rig: Callable[..., Rig],
) -> None:
    rig = make_rig()
    rig.autofocus(True)
    rig.scan()
    view = rig.view
    rig.autofocus(False)

    rig.type_into_the_scan(cell_focus="map", autofocus="off")

    # Before the click the Scan screen names the file the click will give, and *Start the scan*
    # does not wait for a map it has: the tool's refusal would be the wait only with none.
    record = RunDocument.load(rig.kept).stages["scan"]
    assert record.finished_utc is not None
    assert view.focus_map_text.object.splitlines() == [
        f"Focus map: the scan of B2 at {TEN}, "
        f"{record.finished_utc.strftime('%Y-%m-%d %H:%M')} UTC, kept by the page in "
        f"{KEPT_NAME}.",
        ANCHOR_AT_B2,
    ]
    assert view.scan_button.disabled is False
    assert view.scan_wait_text.visible is False
    # The command is the call the click makes: the kept file once, as an absolute path.
    words = shlex.split(view.scan_command.object)
    given = [word for word in words if word.startswith("focus_map=")]
    assert len(given) == 1
    assert os.path.samefile(given[0][len("focus_map=") :], rig.kept)
    assert Path(given[0][len("focus_map=") :]).is_absolute()

    # The click gives the same file: what ran is the command that was shown.
    outcome = rig.scan()
    assert rig.record().status is StageStatus.done, outcome
    assert os.path.samefile(
        rig.record().params["focus_map"], given[0][len("focus_map=") :]
    )

    # `per-field` is no map: no pair, and the lines are gone.
    rig.type_into_the_scan(cell_focus="per-field")
    assert "focus_map=" not in view.scan_command.object
    assert view.focus_map_text.object == ""
    assert view.focus_map_text.visible is False
