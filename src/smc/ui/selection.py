"""Choosing how ``route`` selects and orders the fields, as the page needs it: pure text and rules (m5 §21).

``route`` takes a selector and an orderer by name and their parameters as one JSON
object each (algorithms §5.6). The page offers the installed algorithms as choices and
a chosen algorithm's parameters as fields, but the **JSON text is the truth**: a preset,
a run and a command carry it as the one pair ``selector_params={...}`` (m5 §10.1), so
what is here turns fields into that text and back and says what is on offer. It reads no
registry (the controller does, once), and it holds no Panel, no Bokeh and no device, so
every sentence the operator reads is tested as text (FM-139: plain text, never markup).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel

from smc.plugins.errors import ParamsError
from smc.plugins.params import format_param, parse_params
from smc.plugins.slots import measures_text, score_text

if TYPE_CHECKING:
    from smc.plugins.api import PluginParams

__all__ = [
    "LABEL_SUMMARY_CHARS",
    "NOT_A_JSON_OBJECT",
    "NOT_INSTALLED",
    "ORDER_LABEL",
    "ROUTE_OWN_PARAMS",
    "SELECT_LABEL",
    "UNKNOWN_KEYS",
    "AlgorithmChoice",
    "algorithm_params_text",
    "option_label",
    "options",
    "params_from_text",
    "stop_tip",
    "summary_text",
]

#: The algorithms whose parameters are fields of ``RouteParams`` itself (algorithms §5.6):
#: the page shows ``route``'s own fields for them, the same instances as under *Advanced*,
#: so that a preset and a profile fill them as they fill every field.
ROUTE_OWN_PARAMS: Mapping[str, tuple[str, ...]] = {
    "most-cells": ("min_cells", "candidate_step_fraction")
}
SELECT_LABEL = "Select the fields with"
ORDER_LABEL = "Visit the fields in this order"
#: A closed ``<select>`` clips what it holds, so the label's summary is cut here and the
#: pane under the Select shows it whole.
LABEL_SUMMARY_CHARS = 80
NOT_INSTALLED = "{name} is not installed on this PC; installed: {names}."
UNKNOWN_KEYS = (
    "{carrier} also holds {keys}, which {name} does not take: Run will refuse it."
)
NOT_A_JSON_OBJECT = "{carrier} is not a JSON object ({problem}): Run will refuse it."

_NOT_AN_OBJECT = "a JSON object is expected"


@dataclass(frozen=True, slots=True)
class AlgorithmChoice:
    """One installed selector or orderer as the page lists it: its name, its summary and its parameters.

    ``params`` are ``describe_params`` rows (name, type, default, notes) of ``model``, read
    once when the page starts; ``model`` is the algorithm's ``Params`` class, which the page
    validates a group's fields against so that the text it writes is the text ``route``
    accepts.
    """

    name: str
    summary: str
    params: tuple[tuple[str, str, str, str], ...]
    model: type[BaseModel]


def _collapsed(text: str) -> str:
    """``text`` on one line with its whitespace runs as one space: a summary is one line, but a third party wrote it."""
    return " ".join(text.split())


def option_label(name: str, summary: str, *, default: bool) -> str:
    """``name (default) - summary`` or ``name - summary``, the summary cut at ``LABEL_SUMMARY_CHARS`` with an ellipsis.

    The cut keeps the label's length at the limit, ellipsis included. Plain text: the
    label is a ``Select``'s, never HTML.
    """
    shown = _collapsed(summary)
    if len(shown) > LABEL_SUMMARY_CHARS:
        shown = shown[: LABEL_SUMMARY_CHARS - 1].rstrip() + "\N{HORIZONTAL ELLIPSIS}"
    mark = " (default)" if default else ""
    return f"{name}{mark} \N{EM DASH} {shown}"


def options(choices: Sequence[AlgorithmChoice], default_name: str) -> dict[str, str]:
    """A ``Select``'s options, ``{label: value}``: the default first with the value ``""``, then the others by name.

    An empty value is how the page says "the tool's own default" everywhere (an empty
    field sends nothing), so the default is never sent by name. A ``default_name`` that is
    not among ``choices`` gives no default entry: the caller's fallback (m5 §21, D7).
    """
    found: dict[str, str] = {}
    by_name = sorted(choices, key=lambda choice: choice.name)
    for choice in by_name:
        if choice.name == default_name:
            found[option_label(choice.name, choice.summary, default=True)] = ""
    for choice in by_name:
        if choice.name != default_name:
            found[option_label(choice.name, choice.summary, default=False)] = (
                choice.name
            )
    return found


def summary_text(
    value: str, choices: Sequence[AlgorithmChoice], default_name: str
) -> str:
    """The summary, whole, of the chosen algorithm; for a name that is not installed, the installed ones.

    ``value`` is the Select's: ``""`` is the default. A name a preset or a profile gave that
    this PC does not have is said, with what it has, because ``route`` would refuse it with
    the same names (D6).
    """
    name = value or default_name
    for choice in choices:
        if choice.name == name:
            return _collapsed(choice.summary)
    return NOT_INSTALLED.format(
        name=name, names=", ".join(sorted(choice.name for choice in choices))
    )


def algorithm_params_text(
    model: type[BaseModel], values: Mapping[str, str]
) -> tuple[str, str | None]:
    """The JSON text of a group's filled fields, and the problem the model finds in them, if any.

    Blank fields are left out, so the text holds what the operator chose and nothing else;
    none filled is ``("", None)``. The rest go through ``parse_params``, the command line's
    own rule, and an accepted set is written typed (``20.0``, ``true``, ``["a", "b"]``), as
    ``smc run --param 'selector_params={"x": 20}'`` would record it. A value the model
    refuses is **kept as typed** (``{"keep": "abc"}``) with the problem returned, so that
    the run's refusal names the same key as the note under the fields.
    """
    filled = {key: text.strip() for key, text in values.items() if text.strip()}
    if not filled:
        return "", None
    try:
        # `parse_params` is typed for PluginParams and reads only `model_fields` and
        # `model_validate` (measured, #569), which a SelectorParams has too.
        parsed = parse_params(
            cast("type[PluginParams]", model), [f"{k}={v}" for k, v in filled.items()]
        )
    except ParamsError as error:
        return json.dumps(filled), "; ".join(error.problems)
    dumped = parsed.model_dump(mode="json")
    return json.dumps(
        {key: dumped[key] for key in model.model_fields if key in filled}
    ), None


def params_from_text(
    text: str, model: type[BaseModel]
) -> tuple[dict[str, str], tuple[str, ...], str | None]:
    """A JSON text as a group's fields: each key the model has as the text the command takes, the keys it lacks, a problem.

    The other direction of ``algorithm_params_text``. A blank text is no field and no
    problem. A key the model lacks is returned (in the text's order) and left in the text
    by the caller: ``route`` refuses it in its own words, which a rewrite would hide. A
    text that is not JSON, or not an object, is the third item and no field.
    """
    if not text.strip():
        return {}, (), None
    try:
        loaded = json.loads(text)
    except ValueError as error:
        return {}, (), str(error)
    except RecursionError:
        # A text nested too deep for the parser is a text that is not JSON.
        return {}, (), "the text is nested too deep"
    if not isinstance(loaded, dict):
        return {}, (), _NOT_AN_OBJECT
    fields = model.model_fields
    known = {
        key: format_param(fields[key], value)
        for key, value in loaded.items()
        if key in fields
    }
    unknown = tuple(key for key in loaded if key not in fields)
    return known, unknown, None


def stop_tip(
    *, name: str, order: int, score: float | None, measures: Mapping[str, float]
) -> str:
    """What the map says of a stop on hover: ``A1_r002 (visit 3)``, its score, its measures.

    ``order`` is the stop's 0-based place in the route, written as the visit number the
    report prints (``order + 1``). ``score`` and ``measures`` are the selector's own
    (algorithms §5.2) and are said only when it gave them; a score of ``0.0`` is a score.
    The texts are ``score_text`` and ``measures_text``, the report's.
    """
    tip = f"{name} (visit {order + 1})"
    if score is not None:
        tip += f", score {score_text(score)}"
    if measures:
        tip += f"; {measures_text(measures)}"
    return tip
