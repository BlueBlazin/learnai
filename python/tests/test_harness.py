"""The harness as a black box: spawn it, read the event stream, check the contract."""

import json
import shutil
import subprocess

import pytest
from conftest import REPO_ROOT, SOLUTIONS, fixture_problem, run_harness

# -- protocol shape ---------------------------------------------------------------


def test_event_stream_shape_and_exit_code(sum_list, tmp_path):
    # Run from an unrelated working directory with absolute paths.
    run = run_harness(sum_list, SOLUTIONS / "good.py", cwd=tmp_path)
    assert run.returncode == 0
    assert [e["event"] for e in run.events] == (
        ["start"] + ["case_start", "case_end"] * 4 + ["end"]
    )
    start = run.start
    assert start["problem"] == "sum-list"
    assert start["framework"] == "python"
    assert start["mode"] == "submit"
    assert set(start["env"]) >= {"python", "device"}
    assert all(isinstance(v, str) for v in start["env"].values())
    for event in run.of("case_end"):
        assert set(event) == {
            "event", "id", "status", "duration_ms", "message", "detail", "stdout", "metrics",
        }  # fmt: skip
        assert event["status"] == "pass"
        assert event["message"] is None and event["detail"] is None
    end = run.end
    assert end["status"] == "accepted"
    assert (end["passed"], end["total"]) == (4, 4)
    assert end["duration_ms"] > 0
    assert end["metrics"] == []
    assert run.logs == []


def test_every_stdout_line_is_protocol_or_log(sum_list):
    run = run_harness(sum_list, SOLUTIONS / "prints.py")
    for line in run.stdout.splitlines():
        if line.startswith("@@WS@@ "):
            json.loads(line[len("@@WS@@ ") :])


def test_run_mode_executes_public_cases_only(sum_list):
    run = run_harness(sum_list, SOLUTIONS / "good.py", mode="run")
    assert [c["id"] for c in run.start["cases"]] == ["short", "empty"]
    assert all(c["public"] for c in run.start["cases"])
    assert run.end["total"] == 2


def test_single_case_selection(sum_list):
    run = run_harness(sum_list, SOLUTIONS / "good.py", case="empty")
    assert list(run.statuses) == ["empty"]
    assert run.end["total"] == 1
    # A hidden case is selected by the id it has on the wire...
    run = run_harness(sum_list, SOLUTIONS / "good.py", case="hidden-2")
    assert list(run.statuses) == ["hidden-2"]
    assert run.start["cases"] == [{"id": "hidden-2", "name": "Hidden case 2", "public": False}]
    # ...its real id selects nothing (it would confirm a guess), except for an author...
    run = run_harness(sum_list, SOLUTIONS / "good.py", case="long_list")
    assert run.end["status"] == "harness_error"
    run = run_harness(sum_list, SOLUTIONS / "good.py", case="long_list", extra=["--reveal"])
    assert list(run.statuses) == ["long_list"]
    # ...and a hidden case cannot be selected in run mode at all.
    run = run_harness(sum_list, SOLUTIONS / "good.py", mode="run", case="hidden-1")
    assert run.end["status"] == "harness_error"
    assert run.returncode != 0


# -- hidden cases -------------------------------------------------------------------


def test_hidden_cases_are_redacted(sum_list):
    run = run_harness(sum_list, SOLUTIONS / "wrong_on_negatives.py")
    assert run.start["cases"] == [
        {"id": "short", "name": "sums a short list", "public": True},
        {"id": "empty", "name": "sums an empty list", "public": True},
        {"id": "hidden-1", "name": "Hidden case 1", "public": False},
        {"id": "hidden-2", "name": "Hidden case 2", "public": False},
    ]
    assert [e["id"] for e in run.of("case_start")] == ["short", "empty", "hidden-1", "hidden-2"]
    failed = run.case("hidden-1")
    assert failed["status"] == "fail"
    assert failed["message"] == "total: differs from the expected result"
    assert failed["detail"] is None  # the detail would have shown the expected value
    assert "-10" not in json.dumps(failed)
    assert run.end["status"] == "wrong_answer"
    assert (run.end["passed"], run.end["total"]) == (3, 4)
    # Neither the function names nor the descriptions of hidden tests leave the harness.
    for leak in ("negatives", "long_list", "negative numbers", "a long list"):
        assert leak not in run.stdout


def test_hidden_cases_report_no_metrics_pass_or_fail():
    # A cost metric carries its limit, and the limit of a hidden case is derived from
    # its hidden input (here 2 + 40 - 1 = 41 tokens). It must not leave the harness,
    # whether the case passes (the leak this test was written for) or fails.
    problem = fixture_problem("greedy-decode")
    run = run_harness(problem, problem / "reference" / "torch.py", "torch")
    assert run.end["status"] == "accepted", run.events
    assert run.case("hidden-1")["status"] == "pass"
    assert run.case("hidden-1")["metrics"] == []
    # Public cases keep theirs, and the run-level metrics are the worst over them.
    public = {m["name"]: m for m in run.case("linear_cost")["metrics"]}
    assert public["tokens_processed"] == {
        "name": "tokens_processed", "value": 15, "unit": "tokens", "limit": 15,
    }  # fmt: skip
    assert {m["name"]: m["value"] for m in run.end["metrics"]} == {
        "tokens_processed": 15, "forward_calls": 12,
    }  # fmt: skip
    # The hidden limit (41 tokens) is nowhere in the stream, as a value or as a limit.
    reported = [m for e in run.events for m in e.get("metrics") or []]
    assert all(m["value"] != 41 and m.get("limit") != 41 for m in reported)
    assert "41 tokens" not in run.stdout

    failing = run_harness(problem, problem / "mutants" / "no_cache.py", "torch")
    hidden = failing.case("hidden-1")
    assert hidden["status"] == "fail" and hidden["metrics"] == [] and hidden["detail"] is None
    # The one-line message still says how far over the limit the solution is.
    assert "exceeds the limit of 41 tokens" in hidden["message"]
    assert failing.case("linear_cost")["metrics"][0]["value"] == 114
    assert {m["name"]: m["value"] for m in failing.end["metrics"]}["tokens_processed"] == 114


def test_reveal_reports_the_metrics_of_hidden_cases():
    problem = fixture_problem("greedy-decode")
    run = run_harness(problem, problem / "reference" / "torch.py", "torch", extra=["--reveal"])
    (metric,) = run.case("longer")["metrics"]
    assert (metric["name"], metric["value"], metric["limit"]) == ("tokens_processed", 41, 41)
    assert {m["name"]: m["value"] for m in run.end["metrics"]}["tokens_processed"] == 41


def test_hidden_ids_are_stable_whatever_else_runs(sum_list):
    # hidden-2 is the second hidden case of the file, also when it runs alone.
    alone = run_harness(sum_list, SOLUTIONS / "wrong_on_negatives.py", case="hidden-1")
    assert alone.statuses == {"hidden-1": "fail"}
    alone = run_harness(sum_list, SOLUTIONS / "wrong_on_negatives.py", case="hidden-2")
    assert alone.statuses == {"hidden-2": "pass"}


def test_a_real_case_id_cannot_look_like_an_opaque_one(sum_list, tmp_path):
    problem = tmp_path / "sum-list"
    shutil.copytree(sum_list, problem)
    tests = (problem / "tests.py").read_text()
    (problem / "tests.py").write_text(
        tests.replace("public=True)", 'public=True, id="hidden-1")', 1)
    )
    run = run_harness(problem, SOLUTIONS / "good.py")
    assert run.end["status"] == "harness_error"
    assert "'hidden-1' is reserved for hidden cases" in run.end["message"]


def test_reveal_shows_authors_the_real_hidden_cases(sum_list):
    run = run_harness(sum_list, SOLUTIONS / "wrong_on_negatives.py", extra=["--reveal"])
    names = {c["id"]: c["name"] for c in run.start["cases"]}
    assert names["negatives"] == "negative numbers" and names["long_list"] == "a long list"
    failed = run.case("negatives")
    assert failed["status"] == "fail"
    assert "expected: -10" in failed["detail"]


def test_hidden_errors_say_what_and_where_in_the_learners_file_only(sum_list, tmp_path):
    solution = tmp_path / "python.py"
    solution.write_text(
        "def helper(xs):\n"
        "    return {}[min(xs)]\n"
        "\n"
        "\n"
        "def total(xs):\n"
        "    if any(x < 0 for x in xs):\n"
        "        return helper(xs)\n"
        "    if len(xs) > 10:\n"
        "        raise ValueError('x' * 500)\n"
        "    return sum(xs)\n"
    )
    run = run_harness(sum_list, solution)
    case = run.case("hidden-1")
    assert case["status"] == "error"
    # The learner's exception type and message, and the line in their own file.
    assert case["message"] == "KeyError: -7 (python.py line 2, in helper)"
    assert case["detail"] is None  # no traceback: it could show the test's source
    long = run.case("hidden-2")
    assert long["status"] == "error" and long["detail"] is None
    assert long["message"].startswith("ValueError: xxxx")
    assert long["message"].endswith("... (python.py line 9, in total)")
    assert len(long["message"]) < 250  # the exception text is cut to 200 characters
    assert "tests.py" not in run.stdout and "ctx." not in run.stdout
    assert run.end["status"] == "runtime_error"
    # An author gets the full traceback.
    revealed = run_harness(sum_list, solution, extra=["--reveal"])
    assert 'File "python.py", line 7, in total' in revealed.case("negatives")["detail"]


def test_hidden_error_outside_the_learners_file_has_no_location(sum_list, tmp_path):
    # The solution returns something the test then trips over: the exception is raised
    # in tests.py, so there is no line of the learner's to point at, and no test source.
    problem = tmp_path / "sum-list"
    shutil.copytree(sum_list, problem)
    tests = (problem / "tests.py").read_text()
    tests = tests.replace(
        'check.equal(ctx.call("total", xs), ctx.expected("total", xs), label="total")',
        'check.equal(ctx.call("total", xs) + 0, ctx.expected("total", xs), label="total")',
    )
    (problem / "tests.py").write_text(tests)
    solution = tmp_path / "python.py"
    solution.write_text("def total(xs):\n    return None if min(xs, default=0) < 0 else sum(xs)\n")
    run = run_harness(problem, solution)
    case = run.case("hidden-1")
    assert case["status"] == "error" and case["detail"] is None
    assert case["message"] == "TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'"


def test_public_failures_show_expected_and_got(sum_list, tmp_path):
    solution = tmp_path / "python.py"
    solution.write_text("def total(xs):\n    return sum(xs) + 1\n")
    run = run_harness(sum_list, solution)
    public = run.case("short")
    assert public["status"] == "fail"
    assert public["message"] == "total: differs from the expected result"
    assert "expected: 6" in public["detail"] and "got: 7" in public["detail"]
    assert run.case("hidden-1")["detail"] is None


# -- output capture -----------------------------------------------------------------


def test_capture_is_fd_level_and_per_case(sum_list):
    run = run_harness(sum_list, SOLUTIONS / "prints.py")
    assert run.end["status"] == "accepted"
    assert (run.end["passed"], run.end["total"]) == (4, 4)  # the spoofed end line was inert
    out = run.case("short")["stdout"]
    assert "python print 3\n" in out  # sys.stdout
    assert "stderr write 3\n" in out  # sys.stderr
    assert "raw fd write 3\n" in out  # os.write(1, ...)
    assert "c puts\n" in out  # C stdio from an extension
    assert '@@WS@@ {"event":"end"' in out  # captured as data, not parsed as protocol
    assert "python print 0\n" in run.case("empty")["stdout"]
    assert "python print 3\n" not in run.case("empty")["stdout"]
    assert "python print 50\n" in run.case("hidden-2")["stdout"]
    # Import-time output belongs to no case: it arrives as a diagnostic log line.
    assert run.logs == ["printed at import time"]
    assert len(run.of("end")) == 1


def test_captured_output_is_truncated_to_64_kb(sum_list):
    run = run_harness(sum_list, SOLUTIONS / "floods.py", mode="run")
    out = run.case("empty")["stdout"]
    assert out.startswith("x" * 65536)
    assert "output truncated: 200001 bytes in total" in out
    assert len(out) < 66_000
    assert run.case("short")["stdout"] == ""


# -- errors and tracebacks ------------------------------------------------------------


def test_runtime_error_has_a_traceback_of_user_frames_only(sum_list):
    run = run_harness(sum_list, SOLUTIONS / "raises_on_empty.py")
    case = run.case("empty")
    assert case["status"] == "error"
    assert case["message"] == "IndexError: list index out of range"
    detail = case["detail"]
    assert detail.startswith("Traceback (most recent call last):")
    assert 'File "raises_on_empty.py", line 6, in total' in detail
    assert 'File "raises_on_empty.py", line 2, in helper' in detail
    assert "src/learnai" not in detail and "tests.py" not in detail
    assert detail.rstrip().endswith("IndexError: list index out of range")
    assert run.end["status"] == "runtime_error"
    assert run.returncode == 0  # the harness worked; the solution did not


def test_sys_exit_in_a_solution_is_an_error_not_a_crash(sum_list):
    run = run_harness(sum_list, SOLUTIONS / "calls_sys_exit.py")
    assert run.case("empty")["status"] == "error"
    assert run.case("empty")["message"] == "SystemExit: 3"
    assert run.case("hidden-1")["status"] == "pass"
    assert run.end["status"] == "runtime_error"


def test_the_learners_own_assert_is_an_error(sum_list):
    run = run_harness(sum_list, SOLUTIONS / "own_assert.py", mode="run")
    case = run.case("empty")
    assert case["status"] == "error"
    assert case["message"] == "AssertionError: xs must not be empty"


@pytest.mark.parametrize(
    ("solution", "message", "in_detail"),
    [
        (
            "import_raises.py",
            "the solution failed to import: RuntimeError: bad configuration",
            ['File "import_raises.py", line 10, in <module>', "in configure", "RuntimeError"],
        ),
        (
            "syntax_error.py",
            "the solution failed to import: SyntaxError: '(' was never closed",
            ["syntax_error.py", "line 2", "return sum(xs"],
        ),
        ("missing_entry.py", "the solution does not define `total`", ["must define: total"]),
    ],
)
def test_import_errors(sum_list, solution, message, in_detail):
    run = run_harness(sum_list, SOLUTIONS / solution)
    assert run.returncode == 0
    end = run.end
    assert end["status"] == "import_error"
    assert end["message"] == message
    for text in in_detail:
        assert text in end["detail"]
    assert "src/learnai" not in end["detail"]
    assert (end["passed"], end["total"]) == (0, 4)
    # Every case is reported, as skipped, with the one-line reason. The traceback is
    # on the `end` event only; repeating it on every case would be noise.
    assert set(run.statuses.values()) == {"skipped"}
    assert len(run.statuses) == 4
    for case in run.of("case_end"):
        assert case["message"] == "not run: " + message
        assert case["detail"] is None
    assert "hidden-1" in run.statuses and "negatives" not in run.statuses
    assert run.of("case_start") == []


# -- time limits ------------------------------------------------------------------------


def test_soft_time_limit_stops_a_python_loop_and_continues(sum_list):
    run = run_harness(sum_list, SOLUTIONS / "hangs_on_negatives.py")
    case = run.case("hidden-1")
    assert case["status"] == "timeout"
    assert case["message"] == "time limit exceeded (1 s)"
    assert 1000 <= case["duration_ms"] < 1400
    assert case["stdout"] == "about to spin\n"
    assert run.case("hidden-2")["status"] == "pass"  # later cases still ran
    assert run.end["status"] == "time_limit"
    assert (run.end["passed"], run.end["total"]) == (3, 4)
    assert run.returncode == 0


def test_hard_time_limit_kills_a_solution_that_swallows_the_timeout(sum_list):
    run = run_harness(sum_list, SOLUTIONS / "swallows_timeout.py")
    case = run.case("hidden-1")
    assert case["status"] == "timeout"
    assert "time limit exceeded (1 s)" in case["message"]
    assert case["stdout"] == "about to spin\n"  # recovered from the capture file
    assert run.case("hidden-2")["status"] == "skipped"
    assert run.statuses["short"] == "pass"
    assert run.end["status"] == "time_limit"
    assert (run.end["passed"], run.end["total"]) == (2, 4)
    assert run.end["duration_ms"] < 5000
    assert run.returncode == 0


def test_import_that_never_finishes(sum_list):
    run = run_harness(sum_list, SOLUTIONS / "import_hangs.py", env={"LEARNAI_IMPORT_LIMIT_S": "1"})
    assert run.end["status"] == "import_error"
    assert run.end["message"] == "importing the solution did not finish within 1 s"
    assert set(run.statuses.values()) == {"skipped"}


def test_verdict_precedence_with_mixed_outcomes(sum_list):
    run = run_harness(sum_list, SOLUTIONS / "mixed_failures.py")
    assert run.statuses == {
        "short": "pass", "empty": "error", "hidden-1": "timeout", "hidden-2": "fail",
    }  # fmt: skip
    assert run.end["status"] == "runtime_error"
    run = run_harness(sum_list, SOLUTIONS / "mixed_failures.py", case="hidden-2")
    assert run.end["status"] == "wrong_answer"


# -- never without an end event -----------------------------------------------------------


def test_process_death_still_produces_an_end_event(sum_list):
    run = run_harness(sum_list, SOLUTIONS / "exits_process.py")
    case = run.case("hidden-1")
    assert case["status"] == "error"
    assert case["message"] == "the process exited with code 7 while running this case"
    assert case["stdout"] == "dying now\n"
    assert run.case("hidden-2")["status"] == "skipped"
    assert run.end["status"] == "runtime_error"
    assert (run.end["passed"], run.end["total"]) == (2, 4)
    assert run.returncode == 0


def test_usage_and_setup_failures_end_with_harness_error(sum_list, tmp_path):
    missing = run_harness(sum_list, tmp_path / "nope.py")
    assert missing.end["status"] == "harness_error"
    assert "solution file not found" in missing.end["message"]
    assert missing.returncode != 0

    not_a_problem = run_harness(tmp_path, SOLUTIONS / "good.py")
    assert not_a_problem.end["status"] == "harness_error"
    assert "no problem.toml" in not_a_problem.end["message"]
    assert not_a_problem.returncode != 0

    unsupported = run_harness(sum_list, SOLUTIONS / "good.py", framework="torch")
    assert unsupported.end["status"] == "harness_error"
    assert "does not support framework 'torch'" in unsupported.end["message"]

    bad_flag = run_harness(sum_list, SOLUTIONS / "good.py", mode="bogus")
    assert bad_flag.end["status"] == "harness_error"
    assert bad_flag.returncode == 2


def test_broken_tests_file_is_a_harness_error(sum_list, tmp_path):
    problem = tmp_path / "sum-list"
    shutil.copytree(sum_list, problem)
    (problem / "tests.py").write_text("raise RuntimeError('author bug')\n")
    run = run_harness(problem, SOLUTIONS / "good.py")
    assert run.end["status"] == "harness_error"
    assert "author bug" in run.end["message"]
    assert run.returncode != 0


def test_working_directory_cannot_shadow_frameworks(tmp_path):
    # Solutions are named torch.py; running from their directory must stay safe.
    problem = fixture_problem("unit-normalize")
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    shutil.copy(problem / "reference" / "torch.py", workdir / "torch.py")
    (workdir / "numpy.py").write_text("raise RuntimeError('shadowed numpy')\n")
    run = run_harness(problem, workdir / "torch.py", framework="torch", cwd=workdir)
    assert run.end["status"] == "accepted", run.end


# -- the documented command, verbatim --------------------------------------------------------


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not installed")
def test_architecture_command_verbatim_from_another_directory(tmp_path):
    problem = REPO_ROOT / "problems" / "foundations" / "stable-softmax"
    command = [
        "uv", "run", "--project", str(REPO_ROOT / "python"), "--extra", "jax", "--extra", "mlx",
        "--", "python", "-m", "learnai.harness",
        "--problem", str(problem), "--solution", str(problem / "reference" / "torch.py"),
        "--framework", "torch", "--mode", "submit",
    ]  # fmt: skip
    proc = subprocess.run(
        command, check=False, capture_output=True, text=True, cwd=tmp_path, timeout=300
    )
    assert proc.returncode == 0, proc.stderr
    lines = [line for line in proc.stdout.splitlines() if line.startswith("@@WS@@ ")]
    first, last = json.loads(lines[0][7:]), json.loads(lines[-1][7:])
    assert first["event"] == "start" and first["problem"] == "stable-softmax"
    assert last["event"] == "end" and last["status"] == "accepted"
