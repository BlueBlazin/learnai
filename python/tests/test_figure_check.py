"""The figure checker (`tools/figures/check.py`): the rules, and the catalog against them."""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tools" / "figures" / "check.py"
POLICY = REPO_ROOT / "src" / "components" / "prose" / "svg-policy.ts"


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, check=False
    )


@pytest.fixture(scope="module")
def check():
    spec = importlib.util.spec_from_file_location("figure_check", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_catalog_has_no_new_findings():
    result = run()
    assert result.returncode == 0, result.stdout


def test_known_findings_are_still_reported():
    # A line that no longer applies must be removed from known.txt.
    result = run()
    assert "no longer reported" not in result.stdout


def ts_list(name: str) -> list[str]:
    source = POLICY.read_text(encoding="utf-8")
    match = re.search(rf"{name}[^=]*=\s*(?:new Set\()?\[(.*?)\]", source, re.DOTALL)
    assert match, f"{name} not found in svg-policy.ts"
    return re.findall(r'"([^"]+)"', match.group(1))


def test_the_python_policy_matches_svg_policy_ts(check):
    assert set(check.SVG_ELEMENTS) == set(ts_list("SVG_ELEMENTS"))
    assert check.GEOMETRY == ts_list("GEOMETRY")
    assert check.PRESENTATION == ts_list("PRESENTATION")
    assert check.REFERENCING == ts_list("REFERENCING")
    assert check.ACCESSIBILITY == ts_list("ACCESSIBILITY")


GOOD = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 480 100">
  <title>A figure</title>
  <desc>One sentence. Two sentences.</desc>
  <text x="10" y="20" font-size="13">left</text>
  <text x="240" y="60" font-size="13" text-anchor="middle" style="fill: var(--figure-muted)">middle</text>
  <rect x="10" y="70" width="40" height="20" fill="none" stroke="currentColor"/>
</svg>
"""


def problem(tmp_path: Path, svg: str, statement: str = "", *, name: str = "fig.svg") -> Path:
    directory = tmp_path / "problems" / "demo" / "one"
    (directory / "assets").mkdir(parents=True, exist_ok=True)
    (directory / "problem.toml").write_text("", encoding="utf-8")
    (directory / "assets" / name).write_text(svg, encoding="utf-8")
    if not statement:
        statement = f"![Shows it.](assets/{name})\n"
    (directory / "statement.md").write_text(statement, encoding="utf-8")
    return directory


def messages(directory: Path) -> list[str]:
    result = run("--known", "/dev/null", str(directory))
    return [line.split(": ", 1)[1] for line in result.stdout.splitlines() if ": " in line]


def test_a_good_figure_passes(tmp_path):
    directory = problem(tmp_path, GOOD)
    result = run("--known", "/dev/null", str(directory))
    assert result.returncode == 0, result.stdout
    assert result.stdout == ""


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (lambda s: s.replace(' viewBox="0 0 480 100"', ""), "no viewBox"),
        (lambda s: s.replace("<title>A figure</title>", ""), "no <title>"),
        (lambda s: s.replace("<desc>One sentence. Two sentences.</desc>", ""), "no <desc>"),
        (lambda s: s.replace("Two sentences.", "Two. Three."), "3 sentences"),
        (lambda s: s.replace("viewBox", 'width="480" viewBox'), "a width"),
        (lambda s: s.replace("</svg>", "<script/></svg>"), "<script> is not kept"),
        (lambda s: s.replace("<rect", '<rect class="x"'), "'class'"),
        (lambda s: s.replace('stroke="currentColor"', 'stroke="#000"'), "stroke is '#000'"),
        (lambda s: s.replace('fill="none"', 'style="fill: red"'), "fill is 'red'"),
        (lambda s: s.replace('font-size="13">left', 'font-size="12">left'), "font-size 12"),
        (lambda s: s.replace('x="10" y="20" font-size="13">left', 'x="10" y="20">left'), "not set"),
        (lambda s: s.replace('x="10" y="20"', 'x="-30" y="20"'), "leaves the viewBox"),
        (lambda s: s.replace('x="240" y="60"', 'x="30" y="20"'), "overlap"),
        (
            lambda s: s.replace(
                "left</text>", "left</text><text x='9' y='9' font-size='13' fill='blue'>b</text>"
            ),
            "fill is 'blue'",
        ),
    ],
)
def test_a_rule_that_is_broken_is_reported(tmp_path, change, expected):
    directory = problem(tmp_path, change(GOOD))
    found = messages(directory)
    assert any(expected in line for line in found), found
    assert run("--known", "/dev/null", str(directory)).returncode == 1


def test_a_group_may_carry_the_font_size(tmp_path):
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 480 100">'
        "<title>T</title><desc>D.</desc>"
        '<g font-size="13"><text x="10" y="20">inherits</text></g></svg>'
    )
    assert messages(problem(tmp_path, svg)) == []
    assert any(
        "font-size 14" in line for line in messages(problem(tmp_path, svg.replace("13", "14")))
    )


def test_the_root_must_be_an_svg(tmp_path):
    directory = problem(tmp_path, "<html/>")
    assert any("not an svg" in line for line in messages(directory))
    directory = problem(tmp_path, "<svg", name="broken.svg")
    assert any("does not parse" in line for line in messages(directory))


STEPPED = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 480 100">
  <title>Steps</title>
  <desc>Two frames.</desc>
  <text x="10" y="90" font-size="13">always</text>
  <g id="step-1">
    <desc>First step.</desc>
    <text x="10" y="20" font-size="13">same place</text>
  </g>
  <g id="step-2">
    <desc>Second step.</desc>
    <text x="10" y="20" font-size="13">same place</text>
  </g>
</svg>
"""


def test_stepped_frames_are_compared_one_at_a_time(tmp_path):
    assert messages(problem(tmp_path, STEPPED)) == []


def test_stepped_figures_have_ordered_ids_and_one_sentence_notes(tmp_path):
    gap = problem(tmp_path, STEPPED.replace('id="step-2"', 'id="step-3"'))
    assert any("step-2 missing" in line for line in messages(gap))
    two = problem(tmp_path, STEPPED.replace("First step.", "First. Step."))
    assert any("step-1: the <desc> has 2 sentences" in line for line in messages(two))
    none = problem(tmp_path, STEPPED.replace("<desc>Second step.</desc>", ""))
    assert any("step-2: the first child is not a <desc>" in line for line in messages(none))
    clash = problem(
        tmp_path,
        STEPPED.replace(
            'x="10" y="20" font-size="13">same', 'x="10" y="90" font-size="13">same', 1
        ),
    )
    assert any("overlap" in line for line in messages(clash))


def test_the_statement_must_name_real_figures_in_two_sentences_at_most(tmp_path):
    statement = "\n\n".join(  # noqa: FLY002
        [
            "![One. Two. Three.](assets/fig.svg)",
            "![Gone.](assets/gone.svg)",
            "![Nope.](interactive:no-such-figure)",
            "![Real.](interactive:sampler-order)",
            "```\n![In a fence. A. B.](assets/fenced.svg)\n```",
        ]
    )
    found = messages(problem(tmp_path, GOOD, statement))
    assert any("caption of assets/fig.svg has 3 sentences" in line for line in found)
    assert any("assets/gone.svg does not exist" in line for line in found)
    assert any("interactive:no-such-figure" in line for line in found)
    assert not any("sampler-order" in line or "fenced" in line for line in found)


def test_an_unreferenced_figure_is_a_warning_only(tmp_path):
    directory = problem(tmp_path, GOOD, "No figure here.\n")
    result = run("--known", "/dev/null", str(directory))
    assert "warning" in result.stdout
    assert result.returncode == 0


def test_known_findings_do_not_fail_the_run(tmp_path):
    directory = problem(tmp_path, GOOD.replace('font-size="13">left', 'font-size="12">left'))
    svg = (directory / "assets" / "fig.svg").resolve()
    line = f"{svg.as_posix()}: the text 'left' has font-size 12, not 13"
    known = tmp_path / "known.txt"
    known.write_text(f"# a comment\n{line}\n", encoding="utf-8")
    result = run("--known", str(known), str(directory))
    assert result.returncode == 0, result.stdout
    assert result.stdout.startswith("known: ")
