"""Fixes from the milestone 2 authors, and the judge-side additions of milestone 3."""

import ast
import json
import shutil
import sys
import textwrap
import types

import numpy as np
import pytest
from conftest import CATALOG, SOLUTIONS, run_harness

from whetstone.judge import check, forbidden, frameworks, speed
from whetstone.judge.errors import CheckFailure
from whetstone.judge.forbidden import ForbiddenCall, Guard
from whetstone.kit import Counters


def fails(fn, *args, **kwargs) -> CheckFailure:
    with pytest.raises(CheckFailure) as info:
        fn(*args, **kwargs)
    return info.value


# -- A1: a message never shows what the test expects ---------------------------------------

#: Values chosen so that none of them can appear in a message by accident.
SECRET = 7919


@pytest.mark.parametrize(
    ("call", "secrets"),
    [
        (lambda: check.equal([1, 2], list(range(SECRET, SECRET + 13))), ["13", str(SECRET)]),
        (lambda: check.equal([1, 2, 3], [1, 2, SECRET]), [str(SECRET)]),
        (lambda: check.equal(5, SECRET), [str(SECRET)]),
        (lambda: check.equal(np.zeros(4, dtype=np.int64), np.full(4, SECRET)), [str(SECRET)]),
        (lambda: check.equal(np.zeros((2, 2)), np.zeros((13, 17))), ["13", "17"]),
        (lambda: check.allclose(np.zeros(3), np.zeros((13, 17))), ["13", "17"]),
        (lambda: check.allclose(np.zeros(3, np.float32), np.full(3, 7919.25, np.float32)),
         ["7919", "7.919"]),
        (lambda: check.allclose(np.zeros(3, np.float64), np.zeros(3, np.float16), dtype=True),
         ["float16"]),
        (lambda: check.shape(np.zeros((2, 2)), (13, 17)), ["13", "17"]),
        (lambda: check.match({"a": 1}, {"zebra": 1, "yak": 2}), ["zebra", "yak"]),
        (lambda: check.match([1.0], [np.float32(1.0)] * 13), ["13"]),
        (lambda: check.match(3, [np.int64(1)] * 13), ["13"]),
        (lambda: check.match((1, 2), (np.zeros(2),) * 13), ["13"]),
        (lambda: check.raises(ZeroDivisionError, int, "3"), ["ZeroDivisionError"]),
        (lambda: check.raises(ZeroDivisionError, int, "x"), ["ZeroDivisionError"]),
        (lambda: check.distribution([0, 1, 40], np.full(13, 1 / 13)), ["13"]),
        (lambda: check.distribution(counts=[5000, 0], probs=[0.25, 0.75]), ["5000", "0.25", "0.75"]),
        (lambda: check.mean([0.5, 2.0], 0.5, low=-13.0, high=1.0), ["13"]),
        (lambda: check.mean(np.ones(SECRET), 0.25, low=0.0, high=1.0), [str(SECRET), "0.25"]),
    ],
)  # fmt: skip
def test_no_check_message_reveals_what_was_expected(call, secrets):
    failure = fails(call)
    assert "\n" not in failure.message
    for secret in secrets:
        assert secret not in failure.message, failure.message


def test_the_expected_side_is_in_the_detail_for_public_cases():
    failure = fails(check.equal, [1, 2], [5, 6, 7])
    assert failure.message == "output: got 2 items, which is not the expected number"
    assert "expected 3 items, got 2" in failure.detail and "[5, 6, 7]" in failure.detail
    failure = fails(check.match, {"a": 1}, {"zebra": 1})
    assert failure.message == "output: got a dict whose keys are not the expected keys"
    assert "expected keys: ['zebra']" in failure.detail and "got keys: ['a']" in failure.detail
    assert fails(check.match, [1], {"zebra": 1}).message == "output: expected a dict, got a list"


def test_allclose_states_only_the_order_of_magnitude_of_the_error():
    expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    failure = fails(check.allclose, expected + np.float32(0.004), expected)
    assert "largest error on the order of 1e-03" in failure.message
    assert "max abs error: 0.00" in failure.detail
    got = expected.copy()
    got[1] = np.inf
    assert "got infinity where a finite number is expected" in fails(check.allclose, got, expected).message


def test_a_hidden_case_reports_a_wrong_length_without_the_expected_count(tmp_path):
    """End to end: the report from the milestone 2 authors, on a hidden case."""
    problem = tmp_path / "problems" / "basics" / "sum-list"
    shutil.copytree(CATALOG / "problems" / "basics" / "sum-list", problem)
    tests = problem / "tests.py"
    tests.write_text(
        tests.read_text()
        + textwrap.dedent(
            """

            @case("a list comes back")
            def hidden_length(ctx):
                check.equal(ctx.call("total", [1, 2]), [3, 0, 0, 0, 0, 0, 0], label="total")
            """
        )
    )
    solution = tmp_path / "python.py"
    solution.write_text("def total(xs):\n    return [sum(xs)]\n")
    run = run_harness(problem, solution)
    hidden = run.case("hidden-3")
    assert hidden["message"] == "total: got 1 item, which is not the expected number"
    assert hidden["detail"] is None and "7" not in hidden["message"]


# -- A3: naming the metric of a cost check ---------------------------------------------------


def test_cost_metrics_can_be_named_so_two_kit_objects_do_not_collide(ctx):
    draft, target = Counters("tokens_processed"), Counters("tokens_processed")
    draft.add("tokens_processed", 40)
    target.add("tokens_processed", 12)
    check.cost(draft, "tokens_processed", at_most=40, metric="draft_tokens", unit="tokens")
    check.cost(target, "tokens_processed", exactly=12, metric="target_tokens", unit="tokens")
    assert ctx.metrics == [
        {"name": "draft_tokens", "value": 40, "unit": "tokens", "limit": 40},
        {"name": "target_tokens", "value": 12, "unit": "tokens", "limit": 12},
    ]
    failure = fails(check.cost, target, "tokens_processed", at_most=5, metric="target_tokens")
    assert failure.message == "target_tokens: 12 exceeds the limit of 5"
    # Without a name the old behavior stands: the second report replaces the first.
    check.cost(draft, "tokens_processed", at_most=40)
    check.cost(target, "tokens_processed", at_most=12)
    assert [m["value"] for m in ctx.metrics if m["name"] == "tokens_processed"] == [12]


# -- A4: guarding C types, and names that only the source can show ------------------------------


def test_a_c_type_that_constructs_in_new_is_refused_and_left_intact():
    """`json.scanner.make_scanner` takes its arguments in a C `__new__`. Guarding it
    (the old code put an `__init__` on it) broke `json.loads` for every caller."""
    import json.scanner

    scanner_type = json.scanner.make_scanner
    assert isinstance(scanner_type, type) and scanner_type.__init__ is object.__init__
    before = dict(vars(scanner_type))
    problems = forbidden.check_names(["json.scanner.make_scanner", "builtins.int", "math.pi"])
    assert problems["json.scanner.make_scanner"] == (
        "is a type implemented in C that takes its arguments in `__new__`; a guard on it "
        "would break it for every caller. Rule out its identifier with "
        '`forbidden_source = ["make_scanner"]` instead'
    )
    assert "forbidden_source" in problems["builtins.int"]
    assert "list its identifier in `forbidden_source`" in problems["math.pi"]
    # Nothing was touched: decoders with non-default hooks build a new scanner.
    assert dict(vars(scanner_type)) == before
    assert json.loads('{"a": [1, 2.5]}') == {"a": [1, 2.5]}
    assert json.JSONDecoder(parse_float=str).decode("[1.5]") == ["1.5"]
    assert isinstance(json.JSONDecoder(parse_int=str).scan_once, scanner_type)
    assert int("3") == 3


def test_classes_that_construct_in_a_python_new_or_take_no_arguments_are_guarded(tmp_path):
    module = types.ModuleType("ws_fake_new")
    exec(  # noqa: S102 - builds the fixture module
        textwrap.dedent(
            """
            class Point:
                def __new__(cls, x):
                    self = super().__new__(cls)
                    self.x = x
                    return self

            class Empty:
                pass
            """
        ),
        module.__dict__,
    )
    sys.modules["ws_fake_new"] = module
    solution = tmp_path / "solution.py"
    solution.write_text(
        "import ws_fake_new\n\n\ndef point():\n    return ws_fake_new.Point(2)\n\n\n"
        "def empty():\n    return ws_fake_new.Empty()\n"
    )
    namespace: dict = {}
    exec(compile(solution.read_text(), str(solution), "exec"), namespace)  # noqa: S102
    guard = Guard(["ws_fake_new.Point", "ws_fake_new.Empty"], solution)
    guard.install()
    try:
        assert guard.unresolved == {}
        assert module.Point(3).x == 3 and isinstance(module.Empty(), module.Empty)
        for name in ("point", "empty"):
            with pytest.raises(ForbiddenCall):
                namespace[name]()
    finally:
        guard.uninstall()
        del sys.modules["ws_fake_new"]
    assert namespace["point"]().x == 2 and "__init__" not in vars(module.Empty)


def test_python_classes_are_still_guarded_through_init(tmp_path):
    module = types.ModuleType("ws_fake_classes")
    exec("class Layer:\n    def __init__(self, scale):\n        self.scale = scale\n", module.__dict__)  # noqa: S102
    sys.modules["ws_fake_classes"] = module
    solution = tmp_path / "solution.py"
    solution.write_text("import ws_fake_classes\n\n\ndef make():\n    return ws_fake_classes.Layer(2)\n")
    namespace: dict = {}
    exec(compile(solution.read_text(), str(solution), "exec"), namespace)  # noqa: S102
    guard = Guard(["ws_fake_classes.Layer"], solution)
    guard.install()
    try:
        assert module.Layer(3).scale == 3
        with pytest.raises(ForbiddenCall):
            namespace["make"]()
    finally:
        guard.uninstall()
        del sys.modules["ws_fake_classes"]


@pytest.mark.parametrize(
    ("source", "line"),
    [
        ("import torch\nx = torch.zeros(2, dtype=torch.bfloat16)\n", 2),
        ("from torch import bfloat16\n", 1),
        ("from torch import bfloat16 as half\n", 1),
        ("import torch\n\n\ndef f(x):\n    return x.bfloat16()\n", 5),
        ("import torch\nd = getattr(torch, 'bfloat16')\n", 2),
        ("import mlx.core as mx\ny = mx.zeros(2).astype(dtype='bfloat16')\n", 2),
        ("import numpy as np\nbfloat16 = np.uint16\n", 2),
        ("def f(x, bfloat16=None):\n    return f(x, bfloat16=1)\n", 2),
    ],
)
def test_forbidden_source_finds_an_identifier_in_every_spelling(source, line):
    assert forbidden.find_identifiers(ast.parse(source), ["bfloat16"])[0] == (line, "bfloat16")


def test_forbidden_source_ignores_comments_docstrings_and_other_names():
    source = '"""Decode bfloat16 by hand."""\n# bfloat16 is not used\nx = "a bfloat16 value"\ny = 16\n'
    assert forbidden.find_identifiers(ast.parse(source), ["bfloat16"]) == []
    assert forbidden.find_identifiers(ast.parse("import safetensors.torch\n"), ["safetensors"])


def test_forbidden_source_fails_every_case_before_anything_runs(tmp_path):
    problem = tmp_path / "problems" / "basics" / "sum-list"
    shutil.copytree(CATALOG / "problems" / "basics" / "sum-list", problem)
    toml = problem / "problem.toml"
    toml.write_text(toml.read_text() + 'forbidden_source = ["sum"]\n')
    run = run_harness(problem, SOLUTIONS / "good.py")
    assert run.end["status"] == "wrong_answer"
    assert set(run.statuses.values()) == {"fail"}
    public, hidden = run.case("short"), run.case("hidden-1")
    assert public["message"] == "`sum` is not allowed in this problem (used at good.py line 2)"
    assert "These identifiers may not appear in the solution at all:\n  sum\n" in public["detail"]
    assert hidden["message"] == public["message"] and hidden["detail"] is None
    # The reference adds with a loop and is unaffected.
    assert run_harness(problem, problem / "reference" / "python.py").end["status"] == "accepted"


def test_validate_checks_forbidden_source_entries(tmp_path, monkeypatch):
    from whetstone import validate as gate

    root = tmp_path / "root"
    shutil.copytree(CATALOG / "problems" / "basics", root / "problems" / "basics")
    monkeypatch.setenv("WHETSTONE_ROOT", str(root))
    problem = root / "problems" / "basics" / "sum-list"
    toml = problem / "problem.toml"
    toml.write_text(toml.read_text() + 'forbidden_source = ["torch.bfloat16", "sum", "sum"]\n')
    report, _ = gate.check_metadata(problem, {"sum-list": [problem]})
    assert "problem.toml: forbidden_source contains duplicates" in report.errors
    assert any("'torch.bfloat16' must be a single identifier" in e for e in report.errors)


def test_bf16_decode_uses_forbidden_source_and_no_hand_rolled_scan():
    from conftest import REPO_ROOT

    problem = REPO_ROOT / "problems" / "foundations" / "bf16-decode"
    if not problem.is_dir():
        pytest.skip("the catalog is not next to the package")
    from whetstone.problem import load_problem

    assert load_problem(problem).forbidden_source == ("bfloat16",)
    assert "ast" not in (problem / "tests.py").read_text().split("import numpy")[0]


# -- B4: speed, throughput, and forcing lazy results ---------------------------------------------


def test_require_reports_throughput_when_told_how_many_tokens(ctx):
    result = speed.SpeedResult(solution_s=0.002, reference_s=0.001, rounds=5)
    speed.require(result, max_ratio=3.0, label="generate", ctx=ctx, tokens=24)
    assert ctx.metrics == [
        {"name": "slowdown", "value": 2, "unit": "x", "limit": 3},
        {"name": "tokens_per_s", "value": 12000, "unit": "tok/s", "higher_is_better": True},
    ]


def test_compare_speed_times_prepared_callables(ctx):
    calls = {"solution": 0, "reference": 0}

    def run(name):
        def body():
            calls[name] += 1
            return sum(range(2000))

        return body

    ratio = ctx.compare_speed(run("solution"), run("reference"), max_ratio=50.0, label="step")
    assert 0.02 < ratio < 50 and min(calls.values()) > 7
    assert [m["name"] for m in ctx.metrics] == ["slowdown"]
    with pytest.raises(CheckFailure, match="step: .* slower than the reference"):
        ctx.compare_speed(lambda: sum(range(200000)), lambda: None, max_ratio=1.5, label="step")


def test_mlx_sync_forces_arrays_hidden_in_objects_and_waits_for_async_work():
    mx = pytest.importorskip("mlx.core")
    from whetstone.kit import EvalSpy

    fw = frameworks.get("mlx")

    class Cache:
        def __init__(self):
            self.keys = [mx.ones((4, 4)) * 2]
            self.meta = {"values": mx.ones((4,)) + 1}

    cache = Cache()
    leaves = [t for t in frameworks._leaves((cache, 3), deep=True) if fw.is_tensor(t)]
    assert len(leaves) == 2
    fw.sync((mx.zeros(2) + 1, cache))  # forces everything; must not raise or hang
    pending = mx.ones((8, 8)) @ mx.ones((8, 8))
    mx.async_eval(pending)
    fw.sync()  # no arrays passed: still waits for the stream
    assert float(pending[0, 0]) == 8.0
    assert EvalSpy(file=__file__).evals == 0


def test_check_memory_reports_mebibytes_with_the_limit(ctx):
    measurement = types.SimpleNamespace(peak_bytes=3 * 2**20, active_bytes=2**20)
    check.memory(measurement, at_most_mib=4)
    check.memory(measurement, at_most_mib=4, what="active", metric="result_memory")
    assert ctx.metrics == [
        {"name": "peak_memory", "value": 3, "unit": "MiB", "limit": 4},
        {"name": "result_memory", "value": 1, "unit": "MiB", "limit": 4},
    ]
    failure = fails(check.memory, measurement, at_most_mib=2)
    assert failure.message == "peak_memory: 3 MiB exceeds the limit of 2 MiB"
    with pytest.raises(ValueError):
        check.memory(measurement, at_most_mib=2, what="cache")


def test_json_is_importable_for_event_checks():
    # Guards the test module itself against an unused import after edits.
    assert json.dumps({"ok": True}) == '{"ok": true}'
