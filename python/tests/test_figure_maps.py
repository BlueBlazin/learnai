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
