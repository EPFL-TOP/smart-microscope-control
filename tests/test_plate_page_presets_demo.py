"""An assay repeated by a preset, on ``demo-sample``'s real plugins (m5 §10, #319).

The owner's feedback was that the page asks for the same choices at every run. This is
the walk of ``test_plate_page_first_run_demo`` with the checked-in example preset
(``profiles/presets/weekly.toml``) loaded first: the plate, the wells and the fields
come from it, so no field is typed and the page never asks which wells to scan. Then
the fields are saved as another preset and loaded again on a second tab, which holds
the same fields.

The profile and the preset are copied into ``tmp_path`` and the working directory is
there: the page saves into the first folder of the profile's presets, and a profile
loaded from the repository would put the test's preset into ``profiles/presets``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import pytest

if os.environ.get("SMC_REQUIRE_UI"):
    import panel as pn
else:
    pn = pytest.importorskip("panel")

from smc.hardware.profile import Profile
from smc.run import RunDocument
from smc.ui.controller import PlatePage
from smc.ui.guide import SELECT_WELLS
from smc.ui.plate_page import PlatePageView

pytestmark = [
    pytest.mark.demo,
    pytest.mark.filterwarnings(
        "error::DeprecationWarning", "error::PendingDeprecationWarning"
    ),
    # A failure a handler swallowed fails the test that did not expect it (#277).
    pytest.mark.usefixtures("handler_failures"),
]

REPO = Path(__file__).resolve().parents[1]
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
#: Eight steps, with room for two more: no step is blocked on a choice.
MAX_WALKS = 10


def _fields(view: PlatePageView) -> dict[str, dict[str, str]]:
    return {
        tool: {key: str(widget.value) for key, widget in fields.items()}
        for tool, fields in view.tool_fields.items()
    }


def test_the_weekly_preset_reaches_the_export_by_next_steps_without_typing(
    mm_available: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    profiles = tmp_path / "profiles"
    (profiles / "presets").mkdir(parents=True)
    shutil.copy(REPO / "profiles" / "demo-sample.toml", profiles)
    shutil.copy(REPO / "profiles" / "presets" / "weekly.toml", profiles / "presets")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SMC_PROFILES", raising=False)
    run_doc = tmp_path / "run.json"
    page = PlatePage.closed(Profile.load(profiles / "demo-sample.toml"), run_doc)
    try:
        page.connect()
        assert page.join(60.0), "connect did not end"
        assert page.connected, page.status().outcome

        # The preset is chosen on the page, the way the operator does.
        first = PlatePageView(page)
        first.preset_select.value = "weekly"
        first.preset_load_button.clicks += 1
        assert first.message.object == ""
        loaded = page.preset
        assert loaded is not None
        assert loaded.mismatches == ()
        assert page.selection == ("B2", "B11")
        assert first.preset_note.object.endswith("the page holds the preset's values")

        ran: list[str] = []
        for _walk in range(MAX_WALKS):
            step = page.view().next_step
            assert step is not None
            if step.plugin is None:
                break
            # No wells to choose: the preset gave them.
            assert step.blocked != SELECT_WELLS
            assert step.blocked is None, f"{step.plugin} is blocked: {step.blocked}"
            page.run(step.plugin, list(step.params))
            assert page.join(120.0), f"{step.plugin} did not end"
            outcome = page.status().outcome
            assert outcome is not None
            assert outcome.startswith(f"{step.plugin} done in "), outcome
            ran.append(step.plugin)
        else:
            pytest.fail(f"the next steps did not end in {MAX_WALKS} walks: {ran}")

        assert ran == WORKFLOW
        stages = RunDocument.load(run_doc).stages
        assert {name: stages[name].preset for name in WORKFLOW} == dict.fromkeys(
            WORKFLOW, "weekly"
        )
        assert stages["scan-plan"].params["wells"] == "B2,B11"
        assert stages["scan-plan"].params["coverage"] == 0.4
        assert stages["cells"].params["detector"] == "blobs"

        # Save what the page holds, with two fields the profile does not suggest.
        first.tool_fields["scan-plan"]["overlap"].value = "0.15"
        first.tool_fields["cells"]["score_threshold"].value = "0.6"
        first.preset_name.value = "again"
        first.preset_save_button.clicks += 1
        assert first.message.object.startswith("saved again to "), first.message.object
        assert (profiles / "presets" / "again.toml").is_file()
        saved_fields = _fields(first)

        # Forget it, and load it on a second tab of the same page.
        page.forget_preset()
        assert page.preset is None
        second = PlatePageView(page)
        assert _fields(second) != saved_fields  # a fresh tab starts from the profile
        second.preset_select.value = "again"
        second.preset_load_button.clicks += 1

        assert second.message.object == ""
        assert _fields(second) == saved_fields
        assert page.selection == ("B2", "B11")
        again: Any = page.preset
        assert again is not None
        assert again.file.name == "again"
        assert again.mismatches == ()
    finally:
        page.close()
        page.release()
