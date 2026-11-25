"""scripts/dev/tracks.py: whose is a path, a pull request, an issue or a lane (ADR-0010, #653).

The map itself is pinned by ``test_two_tracks_repository.py``; these tests pin the reader that
``point.py``, ``lane.py`` and ``/review`` share, and the small command line the skills run.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tracks() -> Any:
    name = "smc_dev_tracks"  # the name point.py and lane.py give it: one instance
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / "dev" / "tracks.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rules(tracks: Any) -> Any:
    return tracks.read_rules()


def _pr(*paths: str, changed: int | None = None, **extra: Any) -> dict[str, Any]:
    return {
        "number": 1,
        "files": [{"path": p} for p in paths],
        "changedFiles": len(paths) if changed is None else changed,
        **extra,
    }


def test_a_path_is_the_analysis_tracks_only_on_a_line_with_no_owner(
    tracks: Any, rules: Any
) -> None:
    for path in ("src/smc/selection/builtin.py", "tests/unit/test_selector_x.py"):
        assert tracks.path_track(path, rules) == "analysis", path
    for path in (
        "src/smc/selection/api.py",
        "src/smc/selection/registry.py",
        "pyproject.toml",
        "scripts/dev/point.py",
        "src/smc/hardware/bus.py",
    ):
        assert tracks.path_track(path, rules) == "stand", path


def test_a_path_with_no_matching_line_is_the_stands(tracks: Any) -> None:
    only_analysis = [(tracks.pattern_regex("/docs/algorithms/"), False)]
    assert tracks.path_track("docs/algorithms/README.md", only_analysis) == "analysis"
    assert tracks.path_track("src/smc/ui/app.py", only_analysis) == "stand"
    assert tracks.path_track("anything", []) == "stand"


def test_the_last_matching_line_wins(tracks: Any) -> None:
    rules = [
        (tracks.pattern_regex("*"), True),
        (tracks.pattern_regex("/a/"), False),
        (tracks.pattern_regex("/a/b.py"), True),
    ]
    assert tracks.path_track("a/c.py", rules) == "analysis"
    assert tracks.path_track("a/b.py", rules) == "stand"
    assert tracks.path_track("z.py", rules) == "stand"


def test_a_path_written_with_backslashes_or_a_leading_slash_is_read_as_posix(
    tracks: Any, rules: Any
) -> None:
    assert tracks.path_track("src\\smc\\selection\\builtin.py", rules) == "analysis"
    assert tracks.path_track("/docs/algorithms/README.md", rules) == "analysis"
    assert tracks.path_track("src\\smc\\selection\\api.py", rules) == "stand"


@pytest.mark.parametrize(
    "spelling",
    [
        "./src/smc/selection/builtin.py",
        ".\\src\\smc\\selection\\builtin.py",
        "src//smc/selection/builtin.py",
        "src/smc/../smc/selection/builtin.py",
        "/./src/smc/selection/builtin.py",
    ],
)
def test_one_file_is_one_path_whatever_the_shell_wrote_in_front_of_it(
    tracks: Any, rules: Any, spelling: str
) -> None:
    # PowerShell completes with `.\`: the file fell to the stand without a word (FM-24).
    assert tracks.path_track(spelling, rules) == "analysis"
    assert tracks.path_track(spelling.replace("builtin", "api"), rules) == "stand"


def test_a_star_of_a_pattern_does_not_cross_a_slash(tracks: Any) -> None:
    # `/tests/**/test_selector_*.py` owns files, and a directory that merely looks like one
    # is not matched on GitHub: a `.*` here would hand it to the analysis track.
    regex = tracks.pattern_regex("/tests/**/test_selector_*.py")
    assert regex.match("tests/test_selector_x.py")
    assert regex.match("tests/unit/test_selector_x.py")
    assert not regex.match("tests/unit/test_selector_x/y.py")


def test_a_pr_is_the_analysis_tracks_only_when_every_path_is(
    tracks: Any, rules: Any
) -> None:
    both = ("src/smc/selection/builtin.py", "docs/algorithms/README.md")
    assert tracks.pr_track(_pr(*both), rules) == "analysis"
    assert tracks.pr_track(_pr(*both, "pyproject.toml"), rules) == "stand"
    assert tracks.pr_track(_pr(), rules) == "stand"
    assert tracks.pr_track({"number": 2}, rules) == "stand"


def test_a_pr_whose_file_list_is_cut_is_the_stands(tracks: Any, rules: Any) -> None:
    two = ("src/smc/selection/builtin.py", "docs/algorithms/README.md")
    assert tracks.pr_track(_pr(*two, changed=3), rules) == "stand"
    assert tracks.pr_track(_pr(*two, changed=2), rules) == "analysis"
    # a pull request that gh gave no count for is read by what it listed
    assert tracks.pr_track({"files": [{"path": two[0]}]}, rules) == "analysis"


def test_an_issue_is_the_analysis_tracks_only_with_its_label(tracks: Any) -> None:
    labelled = {"labels": [{"name": "type: feature"}, {"name": "track: analysis"}]}
    assert tracks.issue_track(labelled) == "analysis"
    assert tracks.issue_track({"labels": [{"name": "area: algorithms"}]}) == "stand"
    assert tracks.issue_track({"labels": []}) == "stand"
    assert tracks.issue_track({}) == "stand"
    assert tracks.other("stand") == "analysis"
    assert tracks.other("analysis") == "stand"


def test_a_lane_names_its_track_by_its_letter(tracks: Any) -> None:
    assert tracks.lane_track("S1") == "stand"
    assert tracks.lane_track("S13") == "stand"
    assert tracks.lane_track("A1") == "analysis"
    assert tracks.lane_track("A2") == "analysis"
    for bad in ("X1", "S", "s1", "A1.2", "S1 ", ""):
        with pytest.raises(ValueError, match=r"S<k>.*A<k>"):
            tracks.lane_track(bad)


def test_a_prs_issues_are_what_it_closes_then_its_branch(tracks: Any) -> None:
    closing = {
        "headRefName": "feat/636-x",
        "closingIssuesReferences": [{"number": 636}],
    }
    assert tracks.pr_issues(closing) == [636]
    design = {"headRefName": "docs/654-x", "closingIssuesReferences": []}
    assert tracks.pr_issues(design) == [654]
    assert tracks.pr_issues({"headRefName": "main"}) == []
    # GitHub's own list comes first; the branch's number is added once, after it
    both = {"headRefName": "feat/9-x", "closingIssuesReferences": [{"number": 5}]}
    assert tracks.pr_issues(both) == [5, 9]


def test_the_branch_reader_is_the_ci_checks_one_instance(tracks: Any) -> None:
    assert tracks.closing_refs is sys.modules["smc_ci_check_closing_refs"]


def test_a_missing_codeowners_or_a_form_the_reader_does_not_model_ends_with_the_file_named(
    tracks: Any, tmp_path: Path
) -> None:
    missing = tmp_path / "CODEOWNERS"
    with pytest.raises(SystemExit) as gone:
        tracks.read_rules(missing)
    assert str(missing) in str(gone.value)
    missing.write_text("/docs/** @x\n", encoding="utf-8")
    with pytest.raises(SystemExit) as unmodelled:
        tracks.read_rules(missing)
    assert "does not model" in str(unmodelled.value)
    assert str(missing) in str(unmodelled.value)


def test_a_codeowners_that_is_not_plain_utf8_ends_with_the_file_named_and_what_to_do(
    tracks: Any, tmp_path: Path
) -> None:
    # A Windows editor saves a BOM, UTF-16 or cp1252: the BOM made the first comment line a
    # pattern of a form "the matcher does not model", the others a raw UnicodeDecodeError.
    text = tracks.CODEOWNERS.read_text(encoding="utf-8")
    for name, data in {
        "bom": b"\xef\xbb\xbf" + text.encode("utf-8"),
        "utf16": text.encode("utf-16"),
        "cp1252": ("# caf\N{LATIN SMALL LETTER E WITH ACUTE}\n" + text).encode(
            "cp1252"
        ),
    }.items():
        path = tmp_path / f"CODEOWNERS-{name}"
        path.write_bytes(data)
        with pytest.raises(SystemExit) as stop:
            tracks.read_rules(path)
        assert str(path) in str(stop.value), name
        assert "UTF-8" in str(stop.value), name
    clean = tmp_path / "CODEOWNERS"
    clean.write_bytes(text.encode("utf-8"))
    assert tracks.read_rules(clean)


def test_the_reader_keeps_whether_a_line_names_an_owner(
    tracks: Any, tmp_path: Path
) -> None:
    path = tmp_path / "CODEOWNERS"
    path.write_text("# c\n*  @x\n\n/docs/algorithms/\n", encoding="utf-8")
    (star, owned), (docs, free) = tracks.read_rules(path)
    assert (owned, free) == (True, False)
    assert star.match("x.py")
    assert docs.match("docs/algorithms/README.md")
    assert not docs.match("docs/other.md")


def test_the_real_codeowners_is_read_from_the_repository_root(tracks: Any) -> None:
    assert tracks.CODEOWNERS == ROOT / ".github" / "CODEOWNERS"
    assert tracks.CODEOWNERS.is_file()


def test_the_two_tracking_issues_have_their_own_titles(tracks: Any) -> None:
    assert tracks.TRACKING_TITLES == {
        "stand": "Point: next steps",
        "analysis": "Point (analysis): next steps",
    }
    assert set(tracks.TRACKING_TITLES) == set(tracks.TRACKS)


def test_the_tracks_tool_starts_no_process_of_its_own() -> None:
    # Every `gh` child goes through point.gh: no console, a timeout, utf-8 (FM-22, FM-31).
    source = (ROOT / "scripts" / "dev" / "tracks.py").read_text(encoding="utf-8")
    assert "subprocess" not in source


# -- the command line -----------------------------------------------------------


def test_the_paths_mode_prints_the_track_of_each_path(
    tracks: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    assert tracks.main(["src/smc/selection/builtin.py", "pyproject.toml"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "analysis  src/smc/selection/builtin.py",
        "stand     pyproject.toml",
    ]


def test_the_paths_mode_prints_ascii_whatever_the_path_holds(
    tracks: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        tracks.main(["docs/algorithms/caf\N{LATIN SMALL LETTER E WITH ACUTE}.md"]) == 0
    )
    out = capsys.readouterr().out
    assert out.isascii()
    assert out.startswith("analysis  docs/algorithms/caf")


@pytest.mark.parametrize("argv", [[], ["--pr", "5", "pyproject.toml"]])
def test_the_command_line_takes_paths_or_a_pr_and_not_both_or_neither(
    tracks: Any, argv: list[str]
) -> None:
    with pytest.raises(SystemExit) as stop:
        tracks.main(argv)
    assert stop.value.code == 2


@pytest.mark.parametrize(
    "wild",
    [
        "src/smc/selection/*.py",
        "tests/test_selector_?.py",
        "tests/test_selector_[ab].py",
    ],
)
def test_a_wildcard_is_refused_for_it_stands_for_several_tracks(
    tracks: Any, capsys: pytest.CaptureFixture[str], wild: str
) -> None:
    # `src/smc/selection/*.py` holds api.py and registry.py, which are the stand's, and a
    # shell that does not expand it (cmd.exe, PowerShell) would have it read as analysis.
    with pytest.raises(SystemExit) as stop:
        tracks.main(["pyproject.toml", wild])
    assert stop.value.code == 2
    assert wild in capsys.readouterr().err


def _fake_gh(
    monkeypatch: pytest.MonkeyPatch,
    tracks: Any,
    pr: dict[str, Any],
    issue: dict[str, Any],
    diff: str,
) -> list[tuple[str, ...]]:
    seen: list[tuple[str, ...]] = []
    point = tracks.load_script("smc_dev_point", tracks.HERE / "point.py")

    def gh(*args: str) -> str:
        seen.append(args)
        assert args[:3] == ("pr", "diff", str(pr["number"]))
        return diff

    def gh_json(*args: str) -> Any:
        seen.append(args)
        if args[:2] == ("pr", "view"):
            return pr
        if args[:2] == ("issue", "view"):
            return issue
        raise AssertionError(f"unexpected gh call: {args}")

    monkeypatch.setattr(point, "gh", gh)
    monkeypatch.setattr(point, "gh_json", gh_json)
    return seen


def test_the_pr_mode_names_the_issues_track_and_the_paths_of_the_other_track(
    tracks: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen = _fake_gh(
        monkeypatch,
        tracks,
        {
            "number": 712,
            "headRefName": "feat/636-x",
            "closingIssuesReferences": [{"number": 636}],
        },
        {"number": 636, "labels": [{"name": "track: analysis"}]},
        "src/smc/selection/builtin.py\npyproject.toml\nsrc/smc/selection/registry.py\n",
    )
    assert tracks.main(["--pr", "712"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "PR #712 | issue #636, analysis track | 3 paths: 1 analysis, 2 stand",
        "other track: pyproject.toml",
        "other track: src/smc/selection/registry.py",
    ]
    # the whole diff is read, not a listing that gh may cut
    assert ("pr", "diff", "712", "--name-only") in seen
    assert ("issue", "view", "636", "--json", "number,labels") in seen


def test_the_pr_mode_says_none_when_every_path_is_on_the_issues_track(
    tracks: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _fake_gh(
        monkeypatch,
        tracks,
        {"number": 5, "headRefName": "feat/7-x", "closingIssuesReferences": []},
        {"number": 7, "labels": []},
        "src/smc/ui/app.py\r\nsrc/smc/hardware/bus.py\r\n",
    )
    assert tracks.main(["--pr", "5"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "PR #5 | issue #7, stand track | 2 paths: 0 analysis, 2 stand",
        "other track: none",
    ]


def test_the_pr_mode_takes_the_first_issue_of_a_pr_that_names_two(
    tracks: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # GitHub says it closes #5 and the branch is named for #9: the issue GitHub lists is
    # the first, and the paths are checked against its track.
    seen = _fake_gh(
        monkeypatch,
        tracks,
        {
            "number": 8,
            "headRefName": "feat/9-x",
            "closingIssuesReferences": [{"number": 5}],
        },
        {"number": 5, "labels": [{"name": "track: analysis"}]},
        "src/smc/selection/builtin.py\n",
    )
    assert tracks.main(["--pr", "8"]) == 0
    assert (
        capsys.readouterr()
        .out.splitlines()[0]
        .startswith("PR #8 | issue #5, analysis track |")
    )
    assert ("issue", "view", "5", "--json", "number,labels") in seen
    assert not any(a[:3] == ("issue", "view", "9") for a in seen)


def test_the_pr_mode_without_an_issue_takes_the_track_of_its_paths(
    tracks: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _fake_gh(
        monkeypatch,
        tracks,
        {"number": 9, "headRefName": "main", "closingIssuesReferences": []},
        {},
        "src/smc/selection/builtin.py\ndocs/algorithms/README.md\n",
    )
    assert tracks.main(["--pr", "9"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "PR #9 | no issue found, track by its paths: analysis | 2 paths: 2 analysis, 0 stand",
        "other track: none",
    ]
