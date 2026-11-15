"""Milestone 5: case stages for capstones, one-time library costs kept off a case's
clock, and the speed judge's reuse of its sizing measurement."""

import io
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
from conftest import CATALOG, SOLUTIONS, run_harness

from whetstone import cli, render
from whetstone.judge import case, registry, speed, warmup
from whetstone.judge.registry import load_cases, stages_of

STAGED_TESTS = """
from whetstone.judge import case, check


@case("sums a short list", public=True, stage="1: small lists")
def short(ctx):
    check.equal(ctx.call("total", [1, 2, 3]), 6, label="total")


@case("sums an empty list", public=True, stage="1: small lists")
def empty(ctx):
    check.equal(ctx.call("total", []), 0, label="total")


@case("negative numbers", stage="2: signs")
def negatives(ctx):
    xs = [-5, 2, -7]
    check.equal(ctx.call("total", xs), ctx.expected("total", xs), label="total")


@case("a long list", public=True, stage="1: small lists")
def long_list(ctx):
    xs = [int(v) for v in ctx.rng.integers(0, 100, size=50)]
    check.equal(ctx.call("total", xs), ctx.expected("total", xs), label="total")


@case("a list of one negative number", stage="2: signs")
def one_negative(ctx):
    check.equal(ctx.call("total", [-4]), -4, label="total")
"""


@pytest.fixture
def scratch_root(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "root"
    shutil.copytree(CATALOG / "problems" / "basics", root / "problems" / "basics")
    monkeypatch.setenv("WHETSTONE_ROOT", str(root))
    return root


@pytest.fixture
def staged(scratch_root) -> Path:
    problem = scratch_root / "problems" / "basics" / "sum-list"
    (problem / "tests.py").write_text(STAGED_TESTS)
    return problem


def ws(capsys, *argv: str) -> tuple[int, str, str]:
    code = cli.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# -- stages -------------------------------------------------------------------------------


def test_stage_is_recorded_on_the_case_and_ordered_by_first_appearance(staged):
    cases = load_cases(staged / "tests.py")
    assert [c.stage for c in cases] == [
        "1: small lists", "1: small lists", "2: signs", "1: small lists", "2: signs",
    ]  # fmt: skip
    assert stages_of(cases) == ["1: small lists", "2: signs"]
    assert stages_of([{"id": "a"}, {"id": "b", "stage": "x"}]) == ["x"]


@pytest.mark.parametrize("bad", ["", "   ", "two\nlines", 3, "x" * 61])
def test_stage_must_be_a_short_one_line_label(bad):
    with pytest.raises(ValueError, match="stage="):
        case("name", stage=bad)
    assert registry.MAX_STAGE_LENGTH == 60


def test_start_event_carries_the_stage_also_for_hidden_cases(staged, sum_list):
    run = run_harness(staged, SOLUTIONS / "wrong_on_negatives.py")
    cases = run.start["cases"]
    assert [(c["id"], c["name"], c["stage"]) for c in cases] == [
        ("short", "sums a short list", "1: small lists"),
        ("empty", "sums an empty list", "1: small lists"),
        ("hidden-1", "Hidden case 1", "2: signs"),
        ("long_list", "a long list", "1: small lists"),
        ("hidden-2", "Hidden case 2", "2: signs"),
    ]
    # Nothing else changes: the events of a case do not repeat the stage.
    assert all("stage" not in e for e in run.of("case_end") + run.of("end"))
    assert set(cases[0]) == {"id", "name", "public", "stage"}
    # Run mode lists the public cases only, each with its stage.
    public = run_harness(staged, SOLUTIONS / "good.py", mode="run")
    assert [c["stage"] for c in public.start["cases"]] == ["1: small lists"] * 3
    # A problem without stages has no `stage` key at all.
    plain = run_harness(sum_list, SOLUTIONS / "good.py")
    assert all("stage" not in c for c in plain.start["cases"])


def test_text_output_groups_cases_by_stage_and_counts_each(capsys, staged):
    code, out, _ = ws(capsys, "submit", "sum-list", "--solution",
                      str(SOLUTIONS / "wrong_on_negatives.py"))  # fmt: skip
    assert code == 1
    lines = out.splitlines()
    headings = [line.strip() for line in lines if line.startswith("  Stage")]
    # A heading whenever the stage changes, in the order the cases ran.
    assert headings == [
        "Stage 1: small lists", "Stage 2: signs", "Stage 1: small lists", "Stage 2: signs",
    ]  # fmt: skip
    assert lines.index("  Stage 2: signs") < next(i for i, x in enumerate(lines) if "hidden-1" in x)
    # After the verdict: one line per stage, in order of first appearance.
    verdict = next(i for i, line in enumerate(lines) if line.startswith("Wrong answer"))
    assert "3/5 passed" in lines[verdict]
    assert lines[verdict + 1 : verdict + 3] == [
        "  3/3 passed  Stage 1: small lists",
        "  0/2 passed  Stage 2: signs",
    ]

    code, out, _ = ws(capsys, "run", "sum-list", "--reference")
    assert code == 0
    assert "  3/3 passed  Stage 1: small lists" in out and "Stage 2" not in out


def test_renderer_counts_skipped_cases_of_a_stage_and_keeps_a_stage_prefix():
    out = io.StringIO()
    renderer = render.Renderer(out, color=False)
    renderer.handle({"event": "start", "problem": "p", "framework": "python", "mode": "submit",
                     "cases": [{"id": "a", "name": "A", "public": True, "stage": "Stage 1: load"},
                               {"id": "b", "name": "B", "public": True, "stage": "Stage 1: load"}],
                     "env": {}})  # fmt: skip
    renderer.handle({"event": "case_end", "id": "a", "status": "pass", "duration_ms": 1.0})
    renderer.handle({"event": "case_end", "id": "b", "status": "skipped", "duration_ms": 0,
                     "message": "skipped: needs an asset"})  # fmt: skip
    renderer.handle({"event": "end", "status": "assets_missing", "passed": 1, "total": 2,
                     "skipped": 1, "duration_ms": 5.0})  # fmt: skip
    text = out.getvalue()
    assert text.count("Stage Stage") == 0 and "  Stage 1: load\n" in text
    assert "  1/2 passed, 1 skipped  Stage 1: load\n" in text
    assert render.stage_title("2: paged KV cache") == "Stage 2: paged KV cache"


def test_validate_requires_every_case_to_have_a_stage_or_none(capsys, staged):
    code, out, _ = ws(capsys, "validate", "sum-list")
    assert code == 0, out
    tests = staged / "tests.py"
    tests.write_text(tests.read_text().replace(', stage="2: signs")', ")", 1))
    code, out, _ = ws(capsys, "validate", "sum-list")
    assert code == 1
    assert "tests.py: 1 of 5 cases have no stage (negatives)" in out
    assert out.count("have no stage") == 1
    assert "give every @case a stage= or none" in out


# -- one-time library costs are off the clock ---------------------------------------------

SLOW_SETUP_CASE = """

@case("a case that needs a slow one-time setup", public=True, id="slow_setup")
def slow_setup(ctx):
    import time
    from whetstone.judge import warmup

    with warmup.off_the_clock():
        time.sleep(1.6)            # longer than the case's limit of 1 s
    warmup.once("test-setup", lambda: time.sleep(0.2))
    warmup.once("test-setup", lambda: time.sleep(5.0))      # already done: not run
    check.equal(ctx.call("total", [1, 2]), 3, label="total")


@case("the limit still applies to the rest of a case", public=True, id="after_pause")
def after_pause(ctx):
    import time
    from whetstone.judge import warmup

    with warmup.off_the_clock():
        time.sleep(0.3)
    while True:
        pass
"""


def test_time_off_the_clock_counts_neither_toward_the_limit_nor_the_duration(scratch_root):
    problem = scratch_root / "problems" / "basics" / "sum-list"
    tests = problem / "tests.py"
    tests.write_text(tests.read_text() + SLOW_SETUP_CASE)
    started = time.perf_counter()
    run = run_harness(problem, SOLUTIONS / "good.py", mode="run")
    assert time.perf_counter() - started > 1.8  # the setup did take its time
    slow = run.case("slow_setup")
    assert slow["status"] == "pass", slow
    assert slow["duration_ms"] < 400  # 1.8 s of setup are not in it
    # The soft limit resumed with what was left of it: 1 s of spinning, not 0.7 s.
    after = run.case("after_pause")
    assert after["status"] == "timeout" and after["message"] == "time limit exceeded (1 s)"
    assert 1000 <= after["duration_ms"] < 1400
    assert run.end["status"] == "time_limit"


def test_off_the_clock_and_once_work_without_a_harness():
    seen = []

    @contextmanager
    def pause():
        seen.append("pause")
        yield
        seen.append("resume")

    with warmup.off_the_clock():  # nothing installed: just runs
        seen.append("work")
    warmup.install(pause)
    try:
        with warmup.off_the_clock():
            seen.append("work")
        warmup.once("m5-unit", lambda: seen.append("once"))
        warmup.once("m5-unit", lambda: seen.append("twice"))
        warmup.once("m5-unit-raises", lambda: 1 / 0)  # swallowed
    finally:
        warmup.install(None)
        warmup._done.difference_update({"m5-unit", "m5-unit-raises"})
    assert seen == ["work", "pause", "work", "resume", "pause", "once", "resume", "pause", "resume"]


def test_a_slow_import_is_executed_off_the_clock(tmp_path, monkeypatch):
    (tmp_path / "m5_slow_module.py").write_text("VALUE = 41 + 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(warmup, "SLOW_IMPORTS", frozenset({"m5_slow_module"}))
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    paused = []

    @contextmanager
    def pause():
        paused.append("in")
        yield
        paused.append("out")

    warmup.install(pause)
    try:
        warmup.watch_imports()
        warmup.watch_imports()  # idempotent
        assert sum(isinstance(f, warmup._SlowImportFinder) for f in sys.meta_path) == 1
        import m5_slow_module

        assert m5_slow_module.VALUE == 42 and paused == ["in", "out"]
        import json  # noqa: F401 - any other import is left alone

        assert paused == ["in", "out"]
    finally:
        warmup.install(None)
        sys.modules.pop("m5_slow_module", None)


# -- the speed judge does not throw its sizing measurement away -----------------------------


def test_slow_callables_are_measured_rounds_times_and_fast_ones_sized_first():
    calls = {"s": 0, "r": 0}

    def slow(name):
        def run():
            calls[name] += 1
            time.sleep(0.006)  # longer than MIN_MEASURE_S: one call per measurement

        return run

    result = speed.compare(slow("s"), slow("r"), rounds=3, warmup=1)
    assert result.rounds == 3
    assert calls == {"s": 4, "r": 4}  # 1 warm-up + 3 measured, the sizing call among them

    order = []
    speed.compare(lambda: order.append("s"), lambda: order.append("r"), rounds=2, warmup=0)
    # Fast callables: one sizing call each, then two rounds of many repeats.
    assert order[:2] == ["r", "s"] and len(order) > 2 + 2 * 2
