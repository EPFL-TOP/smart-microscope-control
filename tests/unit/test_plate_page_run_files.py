"""The page's ``/run-files/`` route: the files a run document names open from the page, and nothing else (m5 §17.12 D21, #475).

Through tornado's own test server over the handler the page passes to ``pn.serve`` as
``extra_patterns``, with the controller the page really has: the route answers what
``PlatePage.run_file`` finds. ``pn.serve(..., static_dirs=...)`` would serve the whole run folder
and refuses a folder that does not exist yet, so the allowlist is the controller's, computed at each
request (FM-175's cousin: a route that serves files from a folder serves the folder). What this cannot
show is a browser opening the link in a new tab: that is ``/ui-check``'s (FM-161).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any

import pytest

if os.environ.get("SMC_REQUIRE_UI"):
    import panel  # noqa: F401  # CI must fail, not skip, when the extra is missing
else:
    pytest.importorskip("panel")

import tornado.web
from tornado.testing import AsyncHTTPTestCase

from smc.hardware.profile import Profile
from smc.plugins.api import PluginParams, PluginResult, RunContext
from smc.plugins.registry import GROUP
from smc.run import StageRecord, StageStatus, write_stage
from smc.ui.controller import PlatePage
from smc.ui.plate_page import RunFileHandler

T0 = datetime(2026, 11, 13, 9, 0, tzinfo=timezone.utc)


class NoParams(PluginParams):
    pass


class NoResult(PluginResult):
    pass


class _Tool:
    """A registered tool, so that the page lists its record's card; it is never run."""

    Params = NoParams
    Result = NoResult
    requires: tuple[type, ...] = ()
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()
    summary = "A test stand-in."
    name: str
    provides: str

    def run(self, microscope: Any, params: NoParams, ctx: RunContext) -> NoResult:
        return NoResult()


class Report(_Tool):
    name = provides = "report"


class NisExport(_Tool):
    name = provides = "nis-export"


class Route(_Tool):
    name = provides = "route"


class Cells(_Tool):
    name = provides = "cells"


def _entry_points() -> list[EntryPoint]:
    return [
        EntryPoint(name=cls.name, value=f"{__name__}:{cls.__name__}", group=GROUP)
        for cls in (Report, NisExport, Route, Cells)
    ]


def _record(
    run_doc: Path,
    plugin: str,
    result: dict[str, Any],
    *,
    status: StageStatus = StageStatus.done,
) -> None:
    write_stage(
        run_doc,
        StageRecord(
            plugin=plugin,
            provides=plugin,
            attempt="a0",
            status=status,
            started_utc=T0,
            finished_utc=T0,
            profile="demo",
            dry_run=False,
            smc_version="0",
            params={},
            result=result,
        ),
    )


class RunFilesTest(AsyncHTTPTestCase):
    """The handler over a page whose run document names a report, an export and a result."""

    def setUp(self) -> None:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self.run_dir = base / "run"
        (self.run_dir / "report").mkdir(parents=True)
        (self.run_dir / "export").mkdir()
        self.run_doc = self.run_dir / "run.json"
        # A report whose name has a space and a non-ASCII letter, and bytes that are not ASCII:
        # served as they are, whatever the PC's encoding (CLAUDE.md: bytes, never a guess).
        self.report = self.run_dir / "report" / "rapport é 1.html"
        self.report.write_bytes("<p>résumé</p>".encode())
        self.export = self.run_dir / "export" / "list.xml"
        self.export.write_bytes(b"<?xml version='1.0'?><x/>")
        self.other = self.run_dir / "report" / "not-listed.html"
        self.other.write_bytes(b"not listed")
        self.data = self.run_dir / "export" / "cells.json"
        self.data.write_bytes(b'{"n": 3}')
        self.odd = self.run_dir / "export" / "result.weirdext"
        self.odd.write_bytes(b"\x00\x01\x02")
        self.secret = base / "secret.txt"
        self.secret.write_bytes(b"outside the run folder")
        _record(self.run_doc, "report", {"file": "report/rapport é 1.html"})
        _record(self.run_doc, "nis-export", {"file": "export/list.xml"})
        _record(self.run_doc, "cells", {"file": "export/cells.json"})
        _record(self.run_doc, "route", {"file": "export/result.weirdext"})
        self.page = PlatePage.closed(
            Profile.demo(), self.run_doc, entry_points=_entry_points()
        )
        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()
        self.page.close()
        self._tmp.cleanup()

    def get_app(self) -> tornado.web.Application:
        # Built by `setUp` before `super().setUp()` asks for it: the route as `serve` passes it.
        return tornado.web.Application(
            [(r"/run-files/(.*)", RunFileHandler, {"page": self.page})]
        )

    # -- what is served -------------------------------------------------------------------

    def test_the_route_serves_a_listed_file_with_its_type(self) -> None:
        report = self.fetch("/run-files/report/rapport%20%C3%A9%201.html")
        assert report.code == 200
        # The bytes as written (UTF-8 here, as the layer writes), and a type that says so.
        assert report.body == "<p>résumé</p>".encode()
        assert report.headers["Content-Type"] == "text/html; charset=utf-8"
        # Never sniffed into another type: an export is not HTML.
        assert report.headers["X-Content-Type-Options"] == "nosniff"

        export = self.fetch("/run-files/export/list.xml")
        assert export.code == 200
        assert export.body == b"<?xml version='1.0'?><x/>"
        assert export.headers["Content-Type"].split(";")[0] in (
            "application/xml",
            "text/xml",
        )

        data = self.fetch("/run-files/export/cells.json")
        assert (data.code, data.body) == (200, b'{"n": 3}')
        assert data.headers["Content-Type"] == "application/json"

        # A type `mimetypes` does not know is a download, and the bytes are intact.
        odd = self.fetch("/run-files/export/result.weirdext")
        assert (odd.code, odd.body) == (200, b"\x00\x01\x02")
        assert odd.headers["Content-Type"] == "application/octet-stream"

    def test_the_route_answers_404_for_an_unlisted_file_a_parent_and_an_absolute_path(
        self,
    ) -> None:
        for refused in (
            # A sibling of a listed file, and the run document itself.
            "/run-files/report/not-listed.html",
            "/run-files/run.json",
            # A folder, with and without its slash, and the folder of the run: no listing.
            "/run-files/report",
            "/run-files/report/",
            "/run-files/",
            "/run-files",
            # A parent, as written and as a client encodes it.
            "/run-files/../secret.txt",
            "/run-files/%2e%2e/secret.txt",
            "/run-files/report/../../secret.txt",
            "/run-files/report/%2E%2E%2F%2E%2E%2Fsecret.txt",
            # An absolute path, as a POSIX one and as the drive of a Windows one.
            f"/run-files/{self.secret}",
            f"/run-files//{str(self.secret).lstrip('/')}",
            "/run-files/C:/Windows/win.ini",
            "/run-files/C%3A%5CWindows%5Cwin.ini",
            # A backslash that a Windows path reads as a parent.
            "/run-files/..%5Csecret.txt",
            "/run-files/report%5C..%5C..%5Csecret.txt",
            # A name nothing lists.
            "/run-files/nothing/here.html",
        ):
            response = self.fetch(refused)
            assert response.code == 404, (refused, response.code)
            # Whatever was asked for is never in the answer: no path is echoed into a page.
            assert b"outside the run folder" not in response.body, refused
            assert b"not listed" not in response.body, refused

    def test_a_listed_file_that_is_gone_is_404_and_the_list_is_read_at_each_request(
        self,
    ) -> None:
        assert self.fetch("/run-files/export/list.xml").code == 200
        self.export.unlink()
        assert self.fetch("/run-files/export/list.xml").code == 404
        # A file the document gains after the page started is served at once (a tool run
        # again): the allowlist is the controller's, computed per request.
        late = self.run_dir / "export" / "late.json"
        late.write_bytes(b"{}")
        assert self.fetch("/run-files/export/late.json").code == 404
        _record(self.run_doc, "cells", {"file": "export/late.json"})
        assert self.fetch("/run-files/export/late.json").code == 200

    def test_only_a_file_of_a_record_that_ended_done_is_served(self) -> None:
        (self.run_dir / "export" / "partial.json").write_bytes(b"{}")
        _record(
            self.run_doc,
            "cells",
            {"file": "export/partial.json"},
            status=StageStatus.failed,
        )
        # A failed record has no result to name a file (FM-154): nothing is served for it.
        assert self.fetch("/run-files/export/partial.json").code == 404

    def test_a_file_outside_the_run_folder_that_a_result_names_is_not_served(
        self,
    ) -> None:
        # `nis-export --param out=` may write anywhere: the card shows the path, the page links
        # none of it, and the route serves only what is inside the run folder.
        _record(self.run_doc, "nis-export", {"file": str(self.secret)})
        assert self.fetch(f"/run-files/{self.secret}").code == 404
        assert self.fetch("/run-files/../secret.txt").code == 404
        assert self.fetch("/run-files/secret.txt").code == 404

    def test_a_post_is_not_answered(self) -> None:
        response = self.fetch(
            "/run-files/export/list.xml",
            method="POST",
            body=b"",
            allow_nonstandard_methods=True,
        )
        assert response.code == 405
