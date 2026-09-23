"""scripts/dev/point.py feeds /point; pin the parsing it relies on."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def point():
    spec = importlib.util.spec_from_file_location(
        "point_script", ROOT / "scripts" / "dev" / "point.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_depends_on_reads_the_plan_line(point) -> None:
    plan = "## Plan\n\n**Depends on**: #5, #6, #7 merged into `main`\n**Risk**: pure"
    assert point.depends_on(plan) == [5, 6, 7]


def test_depends_on_none_is_empty(point) -> None:
    assert (
        point.depends_on("**Depends on**: none — uses only `smc.hardware.core`") == []
    )
    assert point.depends_on("no such line") == []


def _c(body: str, at: str) -> dict:
    return {"body": body, "createdAt": at}


def test_verdict_takes_the_latest_design_review(point) -> None:
    texts = [
        _c(
            "## Review (design session, 2026-09-23)\n\n**Verdict: changes requested**",
            "2026-09-23T08:00:00Z",
        ),
        _c(
            "## Review (design session, 2026-09-24)\n\n**Verdict: ready to merge**",
            "2026-09-24T08:00:00Z",
        ),
    ]
    assert point.verdict(texts) == "ready to merge"


def test_verdict_notices_an_addressed_review(point) -> None:
    texts = [
        _c(
            "## Review (design session, 2026-09-23)\n\n**Verdict: changes requested**",
            "2026-09-23T08:00:00Z",
        ),
        _c(
            "## Review addressed (develop session, 2026-09-23)\n- 1. fixed",
            "2026-09-23T12:00:00Z",
        ),
    ]
    assert (
        point.verdict(texts) == "changes requested, then addressed: needs a re-review"
    )


def test_verdict_without_review(point) -> None:
    assert (
        point.verdict([_c("some note", "2026-09-23T08:00:00Z")])
        == "no design review yet"
    )


def test_report_field_collects_its_bullets(point) -> None:
    body = (
        "## Report (develop session, 2026-09-23)\n\n"
        "**Not verified**: real Get-PnpDevice output\n"
        "**Follow-ups**:\n- macOS SPUSBHostDataType\n- setup guide paths\n\n"
        "**Friction**: none\n"
    )
    assert point.field(body, "Follow-ups") == [
        "macOS SPUSBHostDataType",
        "setup guide paths",
    ]
    assert point.field(body, "Not verified") == ["real Get-PnpDevice output"]
    assert point.field(body, "Friction") == []


def test_checks_summary(point) -> None:
    green = [{"name": "lint", "status": "COMPLETED", "conclusion": "SUCCESS"}]
    failing = [
        *green,
        {"name": "tests (windows)", "status": "COMPLETED", "conclusion": "FAILURE"},
    ]
    running = [{"name": "lint", "status": "IN_PROGRESS", "conclusion": ""}]
    assert point.checks(green) == "green"
    assert point.checks(failing) == "failing: tests (windows)"
    assert point.checks(running) == "running"
