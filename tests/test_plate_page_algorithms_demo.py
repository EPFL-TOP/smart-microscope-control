"""A selector chosen on the page gives scored stops that the map, the report and the export carry (m5 §21, #569).

``demo-sample``'s real plugins and Micro-Manager's demo devices, from Connect to the exported position
list, with one thing the operator does that the next steps do not: at the *Route* row of the Results
screen, choose a selector other than the default and one of its parameters. The selector is a
throwaway class of this module registered in place of the installed ones (``registry.entry_points``,
with the built-ins kept): ``most-cells``' own choice with a score and a measure on every field, which
is what a collaborator's algorithm does (algorithms §5.2). What is checked is what the owner will
check at a stand with a real one: the run document records the name and the typed parameters, every
stop carries its score, the map's hover names them, the report prints them, and the export lists the
same stops in the same order and carries no score (algorithms §5.6).

The map's hover itself is a browser's: the source it reads is asserted here (FM-161).
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import replace
from html import escape
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any

import pytest
from pydantic import Field

if os.environ.get("SMC_REQUIRE_UI"):
    import panel as pn  # noqa: F401  (CI must fail, not skip, when the extra is missing)
else:
    pytest.importorskip("panel")

from smc.hardware.profile import Profile
from smc.plugins.nis_export.format import decode
from smc.plugins.slots import NisExportResult, ReportResult, RouteResult
from smc.run import RunDocument
from smc.selection import registry
from smc.selection.api import Choice, SelectionInput, SelectorParams
from smc.selection.builtin import MostCells, MostCellsParams
from smc.ui.controller import PlatePage
from smc.ui.guide import SELECT_WELLS, tool_title
from smc.ui.plate_page import PlatePageView

pytestmark = [
    pytest.mark.demo,
    # A failure a handler swallowed fails the test that did not expect it (#277).
    pytest.mark.usefixtures("handler_failures"),
]

DEMO_SAMPLE = Path(__file__).resolve().parents[1] / "profiles" / "demo-sample.toml"
#: The steps before ``route`` and the operator's one choice, with room for one more.
MAX_WALKS = 12
SCALE = 20.0


class ScoredParams(SelectorParams):
    """One parameter, so that the page's field for it and the run document's record of it are both seen."""

    scale: float = Field(10.0, gt=0, description="Cells per point of score.")


class Scored:
    """``most-cells``' fields, each with the number of cells it holds over ``scale`` as its score and that number as its measure."""

    name = "scored"
    summary = "The fields most-cells chooses, scored by the cells they hold."
    Params = ScoredParams

    def select(self, data: SelectionInput, params: ScoredParams) -> list[Choice]:
        chosen = MostCells().select(data, MostCellsParams())
        found = []
        for choice in chosen:
            held = float(len(data.cells_in(choice.x_um, choice.y_um, choice.well)))
            found.append(
                replace(
                    choice,
                    score=held / params.scale,
                    measures={"cells_in_field": held},
                )
            )
        return found


def _register(monkeypatch: pytest.MonkeyPatch) -> None:
    """The built-ins and ``scored``, in place of what is installed: nothing was read of the registry yet."""
    groups = {
        registry.SELECTOR_GROUP: [
            EntryPoint(
                "most-cells", "smc.selection.builtin:MostCells", registry.SELECTOR_GROUP
            ),
            EntryPoint("scored", f"{__name__}:Scored", registry.SELECTOR_GROUP),
        ],
        registry.ORDERER_GROUP: [
            EntryPoint(
                "nearest", "smc.selection.builtin:Nearest", registry.ORDERER_GROUP
            ),
            EntryPoint(
                "as-chosen", "smc.selection.builtin:AsChosen", registry.ORDERER_GROUP
            ),
        ],
    }
    monkeypatch.setattr(registry, "entry_points", lambda *, group: list(groups[group]))


def _walk(page: PlatePage, until: str | None, ran: list[str]) -> None:
    """The page's next steps, run as its button does, until ``until`` is next (not run) or none is left."""
    for _walk in range(MAX_WALKS):
        step = page.view().next_step
        assert step is not None
        if step.plugin is None or step.plugin == until:
            return
        if step.blocked == SELECT_WELLS:
            page.select_text("B2")
            continue
        assert step.blocked is None, f"{step.plugin} is blocked: {step.blocked}"
        page.run(step.plugin, list(step.params))
        assert page.join(120.0), f"{step.plugin} did not end"
        outcome = page.status().outcome
        assert outcome is not None
        assert outcome.startswith(f"{tool_title(step.plugin)} done in "), outcome
        ran.append(step.plugin)
    pytest.fail(f"the next steps did not end in {MAX_WALKS} walks: {ran}")


def test_a_selector_chosen_on_the_page_gives_scored_stops_that_the_map_the_report_and_the_export_carry(
    mm_available: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    _register(monkeypatch)
    run_doc = tmp_path / "run.json"
    page = PlatePage.closed(Profile.load(DEMO_SAMPLE), run_doc)
    try:
        page.connect()
        assert page.join(60.0), "connect did not end"
        assert page.connected, page.status().outcome
        view = PlatePageView(page)
        ran: list[str] = []
        _walk(page, "route", ran)
        assert ran == ["plate-registration", "scan-plan", "scan", "cells", "density"]

        # The row's two choices, as a session makes them: the selector by name, its parameter
        # as a field. The default orderer stays.
        route = view.result_rows["route"]
        fields = view.tool_fields["route"]
        assert fields["selector"].value == ""
        fields["selector"].value = "scored"
        (scale,) = view._algorithms["selector"].fields["scored"].values()
        assert scale.label == "scale"
        scale.value = str(int(SCALE))
        assert fields["selector_params"].value == '{"scale": 20.0}'
        # The row's command is the call the CLI takes, split as a POSIX shell does.
        assert shlex.split(route.command.object)[-4:] == [
            "--param",
            "selector=scored",
            "--param",
            'selector_params={"scale": 20.0}',
        ]
        route.button.clicks += 1
        assert page.join(120.0), "route did not end"
        outcome = page.status().outcome
        assert outcome is not None
        assert outcome.startswith(f"{tool_title('route')} done in "), outcome

        # The run document records what was chosen and typed, as `smc run route --param ...` would.
        record = RunDocument.load(run_doc).stages["route"]
        assert record.params["selector"] == "scored"
        assert record.params["selector_params"] == {"scale": SCALE}
        assert record.result is not None
        routed = RouteResult.model_validate(record.result)
        assert len(routed.positions) > 0
        assert any("scored" in line for line in routed.assumed)
        # Every stop carries the selector's score and measure, and they agree with the parameter
        # the page sent (the score is the cells held over `scale`).
        for stop in routed.positions:
            assert stop.score is not None
            assert set(stop.measures) == {"cells_in_field"}
            assert stop.score == pytest.approx(stop.measures["cells_in_field"] / SCALE)

        # The map's hover reads one text per stop, naming the score and the measure.
        view.refresh()
        stops = sorted(routed.positions, key=lambda stop: stop.order)
        assert list(view.route_source.data["tip"]) == [
            f"{stop.name} (visit {stop.order + 1}), score {stop.score:.3g}; "
            f"cells_in_field {stop.measures['cells_in_field']:.3g}"
            for stop in stops
        ]
        assert view.route_points in view.hover.renderers
        assert view.frame_renderer not in view.hover.renderers

        # The last two steps are the page's: the report and the export.
        _walk(page, None, ran)
        assert ran[-2:] == ["report", "nis-export"]
        stages = RunDocument.load(run_doc).stages
        assert stages["route"].params["selector"] == "scored"

        # The report prints the scores, once per scored well and in a stop's hover.
        assert stages["report"].result is not None
        report = ReportResult.model_validate(stages["report"].result)
        html = (tmp_path / report.file).read_text(encoding="utf-8")
        lines = re.findall(r'<p class="counts scores">(.*?)</p>', html)
        wells = sorted({stop.well for stop in stops})
        assert len(lines) == len(wells)
        for well in wells:
            mine = [stop for stop in stops if stop.well == well]
            shown = ", ".join(f"{s.name} {s.score:.3g}" for s in mine[:12])
            more = f", and {len(mine) - 12} more" if len(mine) > 12 else ""
            assert f"Selector scores: {shown}{more}" in lines
        for stop in stops[:3]:
            held = f"{stop.measures['cells_in_field']:.3g}"
            assert f", score {stop.score:.3g} (cells_in_field {held})</title>" in html
            assert escape(stop.name) in html

        # The export lists the same stops in the same order, and no score reaches it.
        assert stages["nis-export"].result is not None
        exported = NisExportResult.model_validate(stages["nis-export"].result)
        assert exported.positions == len(stops)
        written = tmp_path / exported.file
        points: Any = decode(written.read_bytes())
        assert [point.name for point in points] == [stop.name for stop in stops]
        text = written.read_bytes().decode("utf-16")
        assert "score" not in text
        assert "cells_in_field" not in text
    finally:
        page.close()
        page.release()
