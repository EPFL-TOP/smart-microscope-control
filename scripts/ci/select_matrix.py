"""Choose the test jobs of a CI run: the light set or the full matrix (ADR-0009).

``main`` requires three checks before a merge (``lint & types`` and the two
jobs of the *light set*); the other three jobs of the *full matrix* run every
night and on a pull request that carries the label ``ci: full``. The job
names are the contract with the required checks of ``main``: ``ci.yml`` keeps
the ``tests (<os>, py<python>)`` template and this file keeps the entries, and
``tests/test_ci_select_matrix.py`` pins both against literals.

    python3 scripts/ci/select_matrix.py

reads ``GITHUB_EVENT_NAME``, ``GITHUB_REPOSITORY`` and ``PR_NUMBER`` (the
workflow passes the pull request's number), asks the API for the pull
request's labels, and writes ``matrix=<json>`` and ``serial=true|false`` to
``$GITHUB_OUTPUT`` and the reason to ``$GITHUB_STEP_SUMMARY``.

The labels are read from the API when the run starts, not from the event
payload: ``gh pr create --label`` may attach the label after ``opened`` has
fired, and a re-run of a run replays its original payload. The cost is that a
label added to a PR takes effect on its next run (a push or *Re-run all jobs*).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

# The label that asks for the full matrix on a pull request (ADR-0009).
FULL_LABEL = "ci: full"

Matrix = dict[str, list[dict[str, str]]]

# Python versions are strings: "3.10" as a number is 3.1.
_UBUNTU_BASELINE = {"os": "ubuntu-latest", "python": "3.13"}
# What the microscope PCs run (surveys, 2026-09).
_WINDOWS_MICROSCOPE_PC = {"os": "windows-latest", "python": "3.14"}

LIGHT: Matrix = {"include": [_UBUNTU_BASELINE, _WINDOWS_MICROSCOPE_PC]}

FULL: Matrix = {
    "include": [
        {"os": "ubuntu-latest", "python": "3.10"},  # the minimum Python (pyproject)
        _UBUNTU_BASELINE,
        {
            "os": "windows-latest",
            "python": "3.11",
        },  # the oldest Python seen on a lab Windows PC
        _WINDOWS_MICROSCOPE_PC,
        {"os": "macos-latest", "python": "3.13"},  # the owner's machine
    ]
}

# `gh` must answer within this; a hang must not hold the lint job for ever.
_GH_TIMEOUT_S = 60


def job_name(entry: dict[str, str]) -> str:
    """The name GitHub gives the test job of ``entry`` (the template of ``ci.yml``)."""
    return f"tests ({entry['os']}, py{entry['python']})"


def select(event: str, labels: list[str] | None) -> tuple[Matrix, str]:
    """The matrix for ``event`` and the reason, for the step summary.

    ``labels`` is the pull request's label list, or ``None`` when it could not
    be read: more testing is the safe direction, so an unreadable list runs the
    full matrix. An event without a rule is refused: a new trigger in
    ``ci.yml`` must come with a decision about what it runs.
    """
    if event == "push":
        return LIGHT, "push to main: the light set"
    if event == "pull_request":
        if labels is None:
            return FULL, "pull request, the labels could not be read: the full matrix"
        if FULL_LABEL in labels:
            return FULL, f"pull request carrying `{FULL_LABEL}`: the full matrix"
        return LIGHT, "pull request: the light set"
    if event in ("schedule", "workflow_dispatch"):
        return FULL, f"{event}: the full matrix"
    raise ValueError(
        f"no rule for the event {event!r}: decide what it runs in scripts/ci/select_matrix.py"
    )


def is_serial(event: str) -> bool:
    """Whether the suite runs in order, with coverage (the nightly and a dispatch).

    A pull request and a push run it in parallel (``pytest -n auto``) with the
    wall-clock tests apart, which is what makes the light set light. The nightly
    is where a failure that parallelism hides would show, so it keeps the whole
    suite in order and the coverage report (two invocations cannot share one
    coverage file, so the parallel runs drop it).
    """
    return event in ("schedule", "workflow_dispatch")


def pr_labels(repo: str, number: int) -> list[str] | None:
    """The labels of pull request ``number``, or ``None`` when ``gh`` cannot say.

    ``None`` is not ``[]``: a pull request with no label is a good read, a
    failed call (no token, rate limit, no ``gh``) is not.
    """
    cmd = [
        "gh",
        "api",
        f"repos/{repo}/issues/{number}/labels",
        "--paginate",
        "--jq",
        ".[].name",
    ]
    try:
        done = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=_GH_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return [line.strip() for line in done.stdout.splitlines() if line.strip()]


def _append(path: str, text: str) -> None:
    # newline="\n": the runner reads LF, and a text-mode write must not add CR on Windows.
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> int:
    event = os.environ.get("GITHUB_EVENT_NAME", "")
    labels: list[str] | None = []
    if event == "pull_request":
        repo = os.environ.get("GITHUB_REPOSITORY", "")
        raw_number = os.environ.get("PR_NUMBER", "")
        labels = (
            pr_labels(repo, int(raw_number)) if repo and raw_number.isdigit() else None
        )
        if labels is None:
            print(
                "::warning::could not read the labels of this pull request:"
                " running the full matrix (more testing is the safe direction)"
            )
    try:
        matrix, reason = select(event, labels)
    except ValueError as error:
        print(f"::error::{error}")
        return 1

    line = f"matrix={json.dumps(matrix)}\nserial={str(is_serial(event)).lower()}\n"
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        _append(output, line)
    else:
        print(line, end="")
    names = ", ".join(f"`{job_name(entry)}`" for entry in matrix["include"])
    summary = f"**CI matrix** ({reason}): {names}\n"
    target = os.environ.get("GITHUB_STEP_SUMMARY")
    if target:
        _append(target, summary)
    else:
        print(summary, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
