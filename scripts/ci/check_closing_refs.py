"""Fail a pull request that would close an issue it does not implement.

GitHub closes an issue for *close*, *fix* or *resolve* before ``#N``
anywhere in a PR's body, even after "not" or inside a quoted sentence:
#127 closed #16, #136 closed #13, and #179 closed #48 with its plan still
unimplemented. CLAUDE.md asks for ``Refs #N``; this check enforces it, from
the branch name (CLAUDE.md, workflow step 3):

- ``docs/…`` and ``adr/…`` close nothing: a design PR is followed by code;
- ``<type>/<N>-<slug>`` closes ``#N`` at most;
- a branch without a number closes nothing.

    python scripts/ci/check_closing_refs.py <head-branch> [<closed number>...]

The numbers are GitHub's own parse (``closingIssuesReferences``).
"""

from __future__ import annotations

import re
import sys

_BRANCH = re.compile(r"(?P<type>[a-z]+)/(?P<number>\d+)(?:-|$)")
_CLOSES_NOTHING = {"docs", "adr"}


def allowed(branch: str) -> set[int]:
    """The issues a PR from ``branch`` may close."""
    m = _BRANCH.match(branch)
    if m is None or m["type"] in _CLOSES_NOTHING:
        return set()
    return {int(m["number"])}


def unexpected(branch: str, closed: list[int]) -> list[int]:
    """The issues GitHub would close that the branch does not implement."""
    return sorted(set(closed) - allowed(branch))


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    branch, *numbers = argv
    closed = [int(n.lstrip("#")) for n in numbers]
    bad = unexpected(branch, closed)
    may = ", ".join(f"#{n}" for n in sorted(allowed(branch))) or "no issue"
    if not bad:
        print(f"closing references: ok (branch {branch!r} may close {may})")
        return 0
    for n in bad:
        print(
            f"::error::This PR would close #{n} when it merges, but branch"
            f" {branch!r} may close {may}. GitHub reads close/fix/resolve before"
            f" #{n} anywhere in the body, even after 'not' or inside a quote:"
            f" write 'Refs #{n}' (CLAUDE.md, workflow step 6)."
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
