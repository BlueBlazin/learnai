"""The ``learnai`` CLI, including the validate gate."""

import shutil
from pathlib import Path

import pytest
from conftest import CATALOG, FIXTURES, REPO_ROOT, SOLUTIONS, fixture_problem

from learnai import cli, protocol
from learnai import validate as gate


@pytest.fixture
def fixture_root(monkeypatch):
    monkeypatch.setenv("LEARNAI_ROOT", str(CATALOG))
    return CATALOG


def learnai(capsys, *argv: str) -> tuple[int, str, str]:
    code = cli.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


@pytest.fixture
def scratch_root(tmp_path, monkeypatch) -> Path:
    """An empty content root with one copy of the sum-list fixture to break."""
    root = tmp_path / "root"
    shutil.copytree(CATALOG / "problems" / "basics", root / "problems" / "basics")
    monkeypatch.setenv("LEARNAI_ROOT", str(root))
    return root


def errors_for(root: Path, name: str = "sum-list") -> list[str]:
    (directory,) = list((root / "problems").glob(f"*/{name}"))
    (report,) = gate.validate(root, [directory], jobs=2)
    return report.errors


# -- list, run, submit ----------------------------------------------------------------


def test_list(capsys, fixture_root):
    code, out, _ = learnai(capsys, "list")
    assert code == 0
    lines = out.splitlines()
    assert lines[0].split() == ["TRACK", "ID", "DIFFICULTY", "FRAMEWORKS", "JUDGES", "TITLE"]
    # Ordered by track order, then by position within the track.
    assert lines[1].split()[:2] == ["basics", "sum-list"]
    row = next(line for line in lines if " greedy-decode " in line)
    assert "torch,jax,mlx" in row and "numeric,cost" in row and "Greedy Decoding" in row


def test_run_renders_public_cases(capsys, fixture_root):
    code, out, _ = learnai(capsys, "run", "sum-list", "--solution", str(SOLUTIONS / "good.py"))
    assert code == 0
    assert "sum-list / python / run" in out
    assert "PASS  short" in out and "sums a short list" in out
    assert "negatives" not in out
    assert "Accepted  2/2 passed" in out


def test_submit_renders_failures_from_the_same_events(capsys, fixture_root):
    code, out, _ = learnai(capsys, "submit", "sum-list", "--mutant", "absolute_values")
    assert code == 1
    assert "FAIL  hidden-1" in out and "Hidden case 1" in out
    assert "total: differs from the expected result" in out
    assert "Wrong answer  3/4 passed" in out
    # What the hidden tests are called and what they check is not printed...
    assert "negatives" not in out and "negative numbers" not in out and "expected:" not in out

    # ...unless an author asks.
    code, out, _ = learnai(capsys, "submit", "sum-list", "--mutant", "absolute_values", "--reveal")
    assert code == 1
    assert "FAIL  negatives" in out and "negative numbers" in out and "expected: -10" in out
    code, out, _ = learnai(capsys, "submit", "sum-list", "--reference", "--case", "hidden-2")
    assert code == 0 and "PASS  hidden-2" in out and "1/1 passed" in out

    code, out, _ = learnai(
        capsys, "submit", "sum-list", "--solution", str(SOLUTIONS / "raises_on_empty.py")
    )
    assert code == 1
    assert "ERROR empty" in out
    assert "| IndexError: list index out of range" in out
    assert "Runtime error" in out


def test_metrics_and_captured_output_are_shown(capsys, fixture_root):
    problem = fixture_problem("greedy-decode")
    code, out, _ = learnai(
        capsys, "run", "greedy-decode", "-f", "torch",
        "--solution", str(problem / "mutants" / "no_cache.py"),
    )  # fmt: skip
    assert code == 1
    assert "tokens_processed: 114 tokens (limit 15 tokens)" in out
    assert "tokens_processed: 114 tokens exceeds the limit of 15 tokens" in out
    code, out, _ = learnai(capsys, "run", "sum-list", "--solution", str(SOLUTIONS / "prints.py"))
    assert code == 0
    assert "log: printed at import time" in out
    assert "| python print 3" in out


def test_reference_starter_and_default_solution(capsys, fixture_root):
    assert learnai(capsys, "submit", "sum-list", "--reference")[0] == 0
    code, out, _ = learnai(capsys, "run", "sum-list", "--starter")
    assert code == 1 and "NotImplementedError" in out
    # No workspace file yet: the starter runs and the CLI says so.
    code, out, err = learnai(capsys, "run", "sum-list")
    assert code == 1
    assert "running the unmodified starter" in err


def test_workspace_solution_is_the_default(capsys, scratch_root):
    workspace = scratch_root / "workspace" / "sum-list"
    workspace.mkdir(parents=True)
    shutil.copy(SOLUTIONS / "good.py", workspace / "python.py")
    code, out, err = learnai(capsys, "submit", "sum-list")
    assert code == 0 and "Accepted  4/4 passed" in out and err == ""


def test_events_flag_prints_the_raw_protocol(capsys, fixture_root):
    code, out, _ = learnai(capsys, "run", "sum-list", "--reference", "--events")
    assert code == 0
    events = [protocol.decode(line) for line in out.splitlines()]
    assert [e["event"] for e in events] == [
        "start", "case_start", "case_end", "case_start", "case_end", "end",
    ]  # fmt: skip


def test_usage_errors(capsys, fixture_root):
    code, _, err = learnai(capsys, "run", "no-such-problem")
    assert code == 2 and "no problem with id 'no-such-problem'" in err
    code, _, err = learnai(capsys, "run", "sum-list", "-f", "torch")
    assert code == 2 and "supports python" in err
    code, _, err = learnai(capsys, "run", "sum-list", "--mutant", "nope")
    assert code == 2 and "available: absolute_values" in err
    code, _, err = learnai(capsys, "validate")
    assert code == 2


# -- validate: the quality gate ----------------------------------------------------------


def test_fixture_catalog_passes_validation(capsys, fixture_root):
    code, out, _ = learnai(capsys, "validate", "--all", "-v")
    assert code == 0, out
    assert "7 of 7 problems passed validation" in out
    for line in (
        "reference (jax) accepted",
        "starter (mlx) rejected (runtime_error)",
        "mutant no_cache rejected (wrong_answer)",
        "mutant python_loops rejected (wrong_answer)",
        "mutant never_times_out rejected (wrong_answer)",
        "mutant no_terminator rejected (wrong_answer)",
        "mutant slightly_biased rejected (wrong_answer)",
        "mutant uses_builtin rejected (wrong_answer)",
        "forbidden: 7 names resolve and can be guarded",
    ):
        assert line in out


def test_the_exemplar_problem_passes_validation(capsys, monkeypatch):
    monkeypatch.setenv("LEARNAI_ROOT", str(REPO_ROOT))
    code, out, _ = learnai(capsys, "validate", "stable-softmax", "-v")
    assert code == 0, out
    assert out.count("rejected (") == 3 + 7  # three starters, seven mutants
    assert "mutant uses_builtin rejected (wrong_answer)" in out
    assert "forbidden: 24 names resolve and can be guarded" in out


def test_deliberately_broken_fixture_fails_validation(capsys, monkeypatch):
    monkeypatch.setenv("LEARNAI_ROOT", str(FIXTURES / "broken"))
    code, out, _ = learnai(capsys, "validate", "--all")
    assert code == 1
    assert "FAIL basics/weak-tests" in out
    assert "error: tests.py: no hidden case" in out
    assert "error: mutant absolute_values: was accepted; no test catches this bug" in out
    assert "0 of 1 problems passed validation" in out


def test_validate_rejects_an_accepted_starter(scratch_root):
    problem = scratch_root / "problems" / "basics" / "sum-list"
    (problem / "starter" / "python.py").write_text("def total(xs):\n    return sum(xs)\n")
    assert errors_for(scratch_root) == ["starter (python): must not be accepted, but it was"]


def test_validate_rejects_a_failing_reference(scratch_root):
    problem = scratch_root / "problems" / "basics" / "sum-list"
    (problem / "reference" / "python.py").write_text("def total(xs):\n    return 6\n")
    (error,) = [e for e in errors_for(scratch_root) if e.startswith("reference")]
    assert "must be accepted, got wrong_answer" in error
    assert "first failing case empty" in error


def test_validate_rejects_starters_and_mutants_that_do_not_even_import(scratch_root):
    problem = scratch_root / "problems" / "basics" / "sum-list"
    (problem / "starter" / "python.py").write_text("def total(xs:\n")
    (problem / "mutants" / "absolute_values.py").write_text("# BUG: typo.\ndef totl(xs): ...\n")
    errors = errors_for(scratch_root)
    assert any(e.startswith("starter (python): must run and be rejected") for e in errors)
    assert any("mutant absolute_values: must run and be rejected" in e for e in errors)
    assert any("import_error" in e for e in errors)


def test_validate_reports_hidden_cases_by_their_real_ids(scratch_root):
    problem = scratch_root / "problems" / "basics" / "sum-list"
    # Wrong on negative input only when judged as the solution (the oracle is this file too).
    (problem / "reference" / "python.py").write_text(
        "def total(xs):\n    return abs(sum(xs)) if __name__ == 'solution' else sum(xs)\n"
    )
    (error,) = [e for e in errors_for(scratch_root) if e.startswith("reference")]
    assert "first failing case negatives" in error  # not "hidden-1": authors need the name


def test_validate_checks_that_forbidden_names_resolve(scratch_root):
    problem = scratch_root / "problems" / "basics" / "sum-list"
    edit_toml(
        problem,
        "time_limit_s = 1",
        'time_limit_s = 1\nforbidden = ["math.fsum", "math.fsun", "math.pi", "statistics.NoSuch.mean"]',
    )
    errors = errors_for(scratch_root)
    assert errors == [
        "problem.toml: forbidden name `math.fsun`: `fsun` does not exist",
        (
            "problem.toml: forbidden name `math.pi`: is not callable, so there is no call to "
            "guard (a dtype or a constant is only mentioned); list its identifier in "
            "`forbidden_source` instead"
        ),
        "problem.toml: forbidden name `statistics.NoSuch.mean`: `NoSuch` does not exist",
    ]


def test_validate_runs_the_forbidden_guard_against_reference_and_mutants(scratch_root):
    problem = scratch_root / "problems" / "basics" / "sum-list"
    edit_toml(problem, "time_limit_s = 1", 'time_limit_s = 1\nforbidden = ["math.fsum"]')
    assert errors_for(scratch_root) == []
    # A reference that needs the forbidden call is a broken problem, and validate says so.
    (problem / "reference" / "python.py").write_text(
        "import math\n\n\ndef total(xs):\n    return int(math.fsum(xs))\n"
    )
    (error,) = errors_for(scratch_root)
    assert "reference (python): must be accepted, got wrong_answer" in error
    assert "`math.fsum` is not allowed in this problem" in error


class _Recorder:
    """Stands in for runner.run: records when each harness run starts and ends."""

    def __init__(self):
        import threading

        self.lock = threading.Lock()
        self.active = 0
        self.log = []  # (problem id, runs active while this one ran)

    def __call__(self, command, on_event=None):
        import time

        problem = Path(command[command.index("--problem") + 1]).name
        solution = Path(command[command.index("--solution") + 1])
        with self.lock:
            self.active += 1
            peak = self.active
        time.sleep(0.02)
        with self.lock:
            peak = max(peak, self.active)
            self.active -= 1
            self.log.append((problem, peak))
        accepted = solution.parent.name == "reference"
        cases = [{"id": "a", "public": True}, {"id": "b", "public": False}]
        return gate.runner.RunResult(
            start={"cases": cases},
            cases=[{"id": c["id"], "status": "pass", "duration_ms": 1.0} for c in cases],
            end={"status": "accepted" if accepted else "wrong_answer", "duration_ms": 2.0},
        )


def test_timing_sensitive_problems_validate_alone_after_the_parallel_batch(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(gate.runner, "run", recorder)
    monkeypatch.setattr(gate, "check_forbidden", lambda names: {})
    directories = gate.problem_dirs(CATALOG)
    reports = gate.validate(CATALOG, directories, jobs=4)
    assert all(report.ok for report in reports), [r.errors for r in reports]
    assert len(reports) == 7  # every problem is still judged, and reported in catalog order
    assert [r.problem_id for r in reports] == [d.name for d in directories]

    sensitive = {"pairwise-distances", "sse-tokens"}  # the speed and the service fixture
    order = [problem for problem, _ in recorder.log]
    first_serial = min(i for i, problem in enumerate(order) if problem in sensitive)
    assert set(order[first_serial:]) == sensitive  # nothing else runs once they start
    assert set(order[:first_serial]).isdisjoint(sensitive)
    assert all(peak == 1 for problem, peak in recorder.log if problem in sensitive)
    assert max(peak for problem, peak in recorder.log if problem not in sensitive) > 1


def test_which_judges_count_as_timing_sensitive():
    from learnai.problem import load_problem

    verdicts = {
        d.name: gate.is_timing_sensitive(load_problem(d)) for d in gate.problem_dirs(CATALOG)
    }
    assert verdicts == {
        "sum-list": False, "dynamic-batcher": False, "greedy-decode": False,
        "pairwise-distances": True, "sample-categorical": False, "sse-tokens": True,
        "unit-normalize": False,
    }  # fmt: skip


def edit_toml(problem: Path, old: str, new: str) -> None:
    path = problem / "problem.toml"
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new))


@pytest.mark.parametrize(
    ("break_it", "expected"),
    [
        (lambda p: edit_toml(p, 'id = "sum-list"', 'id = "sum-lists"'),
         "id 'sum-lists' must equal the directory name 'sum-list'"),
        (lambda p: edit_toml(p, 'track = "basics"', 'track = "advanced"'),
         "track 'advanced' must equal the parent directory name 'basics'"),
        (lambda p: edit_toml(p, "prerequisites = []", 'prerequisites = ["ghost"]'),
         "prerequisite 'ghost' does not exist"),
        (lambda p: edit_toml(p, 'difficulty = "easy"', 'difficulty = "trivial"'),
         "difficulty must be one of"),
        (lambda p: edit_toml(p, 'judges = ["numeric"]', 'judges = ["vibes"]'),
         "unknown judge 'vibes'"),
        (lambda p: edit_toml(p, 'frameworks = ["python"]', 'frameworks = ["python", "torch"]'),
         '"python" cannot be combined with tensor frameworks'),
        (lambda p: edit_toml(p, 'frameworks = ["python"]', 'frameworks = ["jax"]'),
         'tensor problems must support "torch"'),
        (lambda p: edit_toml(p, 'entry = ["total"]', "entry = []"), "entry is empty"),
        (lambda p: edit_toml(p, "time_limit_s = 1", ""), "missing key `time_limit_s`"),
        (lambda p: edit_toml(p, "order = 10", 'order = "first"'), "`order` has the wrong type"),
        (lambda p: edit_toml(p, "order = 10", 'order = 10\nforbidden = "math.fsum"'),
         "`forbidden` must be a list of dotted names"),
        (lambda p: edit_toml(p, "order = 10", 'order = 10\nforbidden = ["fsum"]'),
         "forbidden name 'fsum' must be a dotted name"),
        (lambda p: edit_toml(p, "order = 10", 'order = 10\nforbidden = ["math.fsum", "math.fsum"]'),
         "forbidden contains duplicates"),
        (lambda p: (p / "starter" / "python.py").unlink(), "starter/python.py: missing or empty"),
        (lambda p: (p / "reference" / "jax.py").write_text("x = 1\n"),
         "reference/jax.py: framework 'jax' is not in `frameworks`"),
        (lambda p: shutil.copy(p / "reference" / "python.py", p / "starter" / "python.py"),
         "starter/python.py is identical to the reference"),
        (lambda p: (p / "mutants" / "absolute_values.py").unlink(),
         "at least one mutant is required"),
        (lambda p: (p / "mutants" / "absolute_values.py").write_text("def total(xs): ...\n"),
         "line 1 must be `# BUG: <what is wrong>`"),
        (lambda p: (p / "editorial.md").write_text("  \n"), "editorial.md: missing or empty"),
        (lambda p: (p / "hints.md").write_text("One.\n\n---\n\n---\n\nThree.\n"),
         "hints.md: contains an empty hint"),
        (lambda p: (p / "statement.md").write_text("TODO(author): write me\n"),
         "statement.md: still contains TODO(author)"),
        (lambda p: (p.parent / "track.toml").unlink(), "track.toml: cannot be read"),
    ],
)  # fmt: skip
def test_validate_metadata_checks(scratch_root, break_it, expected):
    break_it(scratch_root / "problems" / "basics" / "sum-list")
    errors = errors_for(scratch_root)
    assert any(expected in error for error in errors), errors


def test_validate_requires_globally_unique_ids(scratch_root):
    shutil.copytree(scratch_root / "problems" / "basics", scratch_root / "problems" / "extras")
    edit_toml(scratch_root / "problems" / "extras" / "sum-list", '"basics"', '"extras"')
    (scratch_root / "problems" / "extras" / "track.toml").write_text(
        'id = "extras"\ntitle = "Extras"\ndescription = "More."\norder = 20\n'
    )
    directory = scratch_root / "problems" / "extras" / "sum-list"
    (report,) = gate.validate(scratch_root, [directory])
    assert any("is not globally unique" in error for error in report.errors)


# -- new ----------------------------------------------------------------------------------


@pytest.mark.parametrize("frameworks", ["torch", "python"])
def test_new_scaffolds_a_problem_that_validates_once_filled_in(capsys, scratch_root, frameworks):
    code, _, _ = learnai(capsys, "new", "decoding/top-k-filter", "--frameworks", frameworks,
                      "--title", "Top-k Filtering", "--difficulty", "medium")  # fmt: skip
    assert code == 0
    problem = scratch_root / "problems" / "decoding" / "top-k-filter"
    for name in ("problem.toml", "statement.md", "hints.md", "editorial.md", "tests.py",
                 f"starter/{frameworks}.py", f"reference/{frameworks}.py",
                 "mutants/ignores_factor.py"):  # fmt: skip
        assert (problem / name).is_file(), name
    assert (scratch_root / "problems" / "decoding" / "track.toml").is_file()
    toml = (problem / "problem.toml").read_text()
    assert 'title = "Top-k Filtering"' in toml and 'difficulty = "medium"' in toml

    # Unfinished scaffolds are refused...
    errors = errors_for(scratch_root, "top-k-filter")
    assert errors and all("TODO(author)" in e or "track.toml" in e for e in errors)
    # ...and once every marker is replaced, the scaffold is a valid problem.
    for path in [*problem.rglob("*"), problem.parent / "track.toml"]:
        if path.is_file():
            path.write_text(path.read_text().replace("TODO(author)", "done"))
    assert errors_for(scratch_root, "top-k-filter") == []


def test_new_refuses_bad_targets(capsys, scratch_root):
    assert learnai(capsys, "new", "Not A Path")[0] == 2
    assert learnai(capsys, "new", "basics/sum-list")[0] == 2  # the id already exists
    assert learnai(capsys, "new", "x/y", "--frameworks", "jax")[0] == 2
    code, _, err = learnai(capsys, "new", "x/y", "--frameworks", "python,torch")
    assert code == 2 and "cannot be combined" in err
