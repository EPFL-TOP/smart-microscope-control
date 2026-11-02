"""A newcomer's first run of the plate workflow, by the page's next steps alone (m5 §8, #293).

The owner followed ``docs/ui/plate-page.md`` on ``demo-sample`` and found it hard: a terminal
command with a JSON ``refs`` to register the plate, values to type that only the docs
knew, and no word on where the files were. This runs the page's controller the way the
page's button does (the step's own pairs, never a form), from Connect to the exported
position list, with one choice left to the operator: the well to scan. The real plugins
run on Micro-Manager's demo devices, so what a step suggests is what the plugins accept.

One well keeps the chain short (two wells took 21 s, measured 2026-11-02, Python 3.13.0;
the scan is nearly all of it).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from smc.hardware.profile import Profile
from smc.ui.controller import PlatePage
from smc.ui.guide import SELECT_WELLS

pytestmark = [
    pytest.mark.demo,
    # A failure a handler swallowed fails the test that did not expect it (#277).
    pytest.mark.usefixtures("handler_failures"),
]

DEMO_SAMPLE = Path(__file__).resolve().parents[1] / "profiles" / "demo-sample.toml"
#: The workflow's tools, in the order the run document asks for them.
WORKFLOW = [
    "plate-registration",
    "scan-plan",
    "scan",
    "cells",
    "density",
    "route",
    "report",
    "nis-export",
]
#: Eight steps and the operator's one choice, with room for one more.
MAX_WALKS = 12


def test_a_newcomer_reaches_the_export_by_next_steps_alone(
    mm_available: bool, tmp_path: Path
) -> None:
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    run_doc = tmp_path / "run.json"
    page = PlatePage.closed(Profile.load(DEMO_SAMPLE), run_doc)
    try:
        page.connect()
        assert page.join(60.0), "connect did not end"
        assert page.connected, page.status().outcome

        ran: list[tuple[str, tuple[str, ...]]] = []
        last_why: tuple[str, ...] = ()
        for _walk in range(MAX_WALKS):
            step = page.view().next_step
            assert step is not None
            if step.plugin is None:
                last_why = step.why
                break
            if step.blocked == SELECT_WELLS:
                # The one choice the page cannot make: which well to scan.
                page.select_text("B2")
                continue
            assert step.blocked is None, f"{step.plugin} is blocked: {step.blocked}"
            page.run(step.plugin, list(step.params))
            assert page.join(120.0), f"{step.plugin} did not end"
            outcome = page.status().outcome
            assert outcome is not None
            assert outcome.startswith(f"{step.plugin} done in "), outcome
            ran.append((step.plugin, step.params))
        else:
            pytest.fail(f"the next steps did not end in {MAX_WALKS} walks: {ran}")

        assert [plugin for plugin, _pairs in ran] == WORKFLOW
        pairs = dict(ran)
        # What the profile suggests reached the steps it names, and only those.
        assert "coverage=0.4" in pairs["scan-plan"]
        assert "detector=blobs" in pairs["cells"]
        assert not any(p.startswith(("coverage=", "detector=")) for p in pairs["route"])
        # Registered from the sample's geometry: three references, no capture on the page.
        assert page.refs == ()
        assert pairs["plate-registration"][0] == "plate=96-well"

        # Every file the workflow wrote is named by the last step, and is there.
        assert last_why[0] == "every stage of the workflow is done; what it wrote:"
        written = [tuple(line.partition(": ")[::2]) for line in last_why[1:]]
        # scan's folder, then the file of cells, of report and of nis-export, in tool order.
        assert [key for key, _path in written] == ["folder", "file", "file", "file"]
        for key, path in written:
            assert Path(path).is_relative_to(run_doc.parent), path
            assert Path(path).is_dir() if key == "folder" else Path(path).is_file(), (
                path
            )
        suffixes = [Path(path).suffix for _key, path in written[1:]]
        assert suffixes[1:] == [".html", ".xml"]  # the report, then the export
    finally:
        page.close()
        page.release()
