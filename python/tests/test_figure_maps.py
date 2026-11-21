"""The architecture maps under `problems/` match their specs in `tools/figures/maps/`."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tools" / "figures" / "maps.py"
SPECS = sorted((REPO_ROOT / "tools" / "figures" / "maps").glob("*.json"))
SVG = "{http://www.w3.org/2000/svg}"


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, check=False
    )


@pytest.fixture(scope="module")
def maps():
    spec = importlib.util.spec_from_file_location("figure_maps", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_written_maps_are_up_to_date():
    result = run("--check")
    assert result.returncode == 0, result.stderr


def test_check_fails_on_a_stale_or_missing_file(tmp_path):
    spec = json.loads(SPECS[0].read_text(encoding="utf-8"))
    for problem in spec["problems"]:
        directory = tmp_path / "problems" / spec["track"] / problem
        directory.mkdir(parents=True)
        (directory / "problem.toml").write_text("", encoding="utf-8")
    root = ["--root", str(tmp_path)]

    missing = run("--check", *root)
    assert missing.returncode == 1
    assert "missing" in missing.stderr

    assert run(*root).returncode == 0
    assert run("--check", *root).returncode == 0

    written = next(tmp_path.rglob(spec["file"]))
    first = written.read_text(encoding="utf-8")
    written.write_text(first.replace("RMSNorm", "LayerNorm"), encoding="utf-8")
    stale = run("--check", *root)
    assert stale.returncode == 1
    assert "out of date" in stale.stderr

    # Writing again restores the same bytes.
    assert run(*root).returncode == 0
    assert written.read_text(encoding="utf-8") == first
    shutil.rmtree(tmp_path / "problems")


@pytest.mark.parametrize("path", SPECS, ids=lambda path: path.stem)
def test_maps_follow_the_figure_rules(maps, path):
    spec = json.loads(path.read_text(encoding="utf-8"))
    assert maps.problems_of(spec) == []
    for problem, focus in spec["problems"].items():
        root = ET.fromstring(maps.render(spec, problem))
        assert root.tag == f"{SVG}svg"
        assert root.get("viewBox") == f"0 0 480 {spec['height']}"
        assert "width" not in root.attrib
        assert [child.tag for child in root][:2] == [f"{SVG}title", f"{SVG}desc"]
        for element in root.iter():
            assert "class" not in element.attrib
            for name in ("fill", "stroke"):
                assert element.get(name) in (None, "none", "currentColor")
        sizes = {text.get("font-size") for text in root.iter(f"{SVG}text")}
        assert sizes == {"13"}
        # Only this problem's parts keep their full strength.
        dimmed = sum(1 for element in root.iter() if element.get("stroke-opacity"))
        assert (dimmed > 0) == (focus.get("parts") is not None)


def test_a_label_that_does_not_fit_is_refused(maps):
    spec = json.loads(SPECS[0].read_text(encoding="utf-8"))
    box = next(block for block in spec["blocks"] if block.get("shape", "box") == "box")
    box["label"] = "A label far too long for the box it is in"
    assert any("does not fit" in problem for problem in maps.problems_of(spec))


def small_spec(problems: dict) -> dict:
    return {
        "track": "demo",
        "file": "map.svg",
        "title": "A map",
        "desc": "Two boxes.",
        "width": 480,
        "height": 100,
        "blocks": [
            {
                "part": "a",
                "role": "attn",
                "label": "Attention",
                "x": 10,
                "y": 10,
                "width": 120,
                "height": 20,
            },
            {"part": "b", "label": "Input", "x": 10, "y": 50, "width": 120, "height": 20},
        ],
        "problems": problems,
    }


def test_a_box_without_a_role_is_drawn_unfilled(maps):
    spec = small_spec({"one": {"parts": ["a"]}})
    assert maps.problems_of(spec) == []
    root = ET.fromstring(maps.render(spec, "one"))
    boxes = list(root.iter(f"{SVG}rect"))
    assert [b.get("fill") for b in boxes] == [None, "none"]
    # The unfilled box is dimmed by its outline alone.
    assert boxes[1].get("stroke-opacity") is not None
    assert boxes[1].get("fill-opacity") is None


def test_a_wrong_role_is_still_refused(maps):
    spec = small_spec({"one": {"parts": ["a"]}})
    spec["blocks"][1]["role"] = "purple"
    assert any("unknown role" in found for found in maps.problems_of(spec))


def test_a_problem_may_belong_to_another_module(maps, tmp_path):
    spec = small_spec({"one": {"parts": ["a"]}, "other/two": {"parts": None}})
    assert maps.place(spec, "one") == ("demo", "one")
    assert maps.place(spec, "other/two") == ("other", "two")
    with pytest.raises(maps.SpecError):
        maps.place(spec, "a/b/c")
    for track, name in (("demo", "one"), ("other", "two")):
        directory = tmp_path / "problems" / track / name
        directory.mkdir(parents=True)
        (directory / "problem.toml").write_text("", encoding="utf-8")
    files = maps.outputs([spec], tmp_path)
    assert sorted(path.relative_to(tmp_path).as_posix() for path in files) == [
        "problems/demo/one/assets/map.svg",
        "problems/other/two/assets/map.svg",
    ]
    assert "other/two" in files[tmp_path / "problems/other/two/assets/map.svg"]
    missing = small_spec({"other/nope": {"parts": None}})
    with pytest.raises(maps.SpecError, match="other/nope"):
        maps.outputs([missing], tmp_path)
