"""The ``ws`` command-line interface (docs/ARCHITECTURE.md section 3.7)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from whetstone import __version__, protocol, render, runner, scaffold
from whetstone import validate as gate
from whetstone.problem import Problem, ProblemError, discover, find_problem, find_root, problem_dirs


def _solution_for(args: argparse.Namespace, root: Path, problem: Problem, framework: str) -> Path:
    if args.solution:
        return Path(args.solution)
    if args.reference:
        return problem.reference_path(framework)
    if args.starter:
        return problem.starter_path(framework)
    if args.mutant:
        path = problem.path / "mutants" / f"{args.mutant.removesuffix('.py')}.py"
        if not path.is_file():
            names = ", ".join(p.stem for p in problem.mutant_paths()) or "none"
            raise ProblemError(f"no mutant {args.mutant!r} (available: {names})")
        return path
    workspace = root / "workspace" / problem.id / f"{framework}.py"
    if workspace.is_file():
        return workspace
    print(
        f"note: no solution at {workspace}; running the unmodified starter. "
        "Copy the starter there and edit it.",
        file=sys.stderr,
    )
    return problem.starter_path(framework)


def cmd_run(args: argparse.Namespace, mode: str) -> int:
    root = find_root()
    try:
        problem = find_problem(root, args.problem)
        framework = args.framework or (problem.frameworks[0] if problem.frameworks else "")
        if framework not in problem.frameworks:
            raise ProblemError(
                f"{problem.id} supports {', '.join(problem.frameworks)}; got {framework!r}"
            )
        if args.mutant and framework != problem.oracle_framework:
            raise ProblemError(f"mutants are written in {problem.oracle_framework}")
        solution = _solution_for(args, root, problem, framework)
        if not solution.is_file():
            raise ProblemError(f"solution file not found: {solution}")
    except ProblemError as exc:
        print(f"ws: {exc}", file=sys.stderr)
        return 2

    command = runner.harness_command(problem.path, solution, framework, mode, args.case)
    if args.events:

        def show(event: dict) -> None:
            if event.get("event") == "log":
                print(event.get("text", ""), flush=True)
            else:
                print(protocol.encode(event), flush=True)
    else:
        show = render.Renderer().handle
    result = runner.run(command, show)
    if result.verdict == "accepted":
        return 0
    return 2 if result.verdict == "harness_error" else 1


def cmd_list(args: argparse.Namespace) -> int:
    root = find_root()
    problems = discover(root)
    if not problems:
        print(f"no problems under {root / 'problems'}")
        return 0
    rows = [("TRACK", "ID", "DIFFICULTY", "FRAMEWORKS", "JUDGES", "TITLE")]
    rows += [
        (p.track, p.id, p.difficulty, ",".join(p.frameworks), ",".join(p.judges), p.title)
        for p in problems
    ]
    widths = [max(len(row[i]) for row in rows) for i in range(5)]
    for row in rows:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row[:5])) + "  " + row[5])
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    root = find_root()
    if args.all == bool(args.problems):
        print("ws validate: give problem ids or --all (not both)", file=sys.stderr)
        return 2
    try:
        if args.all:
            directories = problem_dirs(root)
        else:
            directories = [_problem_dir(root, name) for name in args.problems]
    except ProblemError as exc:
        print(f"ws: {exc}", file=sys.stderr)
        return 2
    if not directories:
        print(f"ws validate: no problems under {root / 'problems'}", file=sys.stderr)
        return 2

    reports = gate.validate(root, directories, jobs=args.jobs)
    paint = render.Renderer().paint
    for report in reports:
        status = paint("ok  ", "pass") if report.ok else paint("FAIL", "fail")
        print(f"{status} {report.path.parent.name}/{report.problem_id}")
        if args.verbose or not report.ok:
            for check in report.checks:
                print(f"       {paint(check, 'dim')}")
        for warning in report.warnings:
            print(f"       warning: {warning}")
        for error in report.errors:
            print(f"       error: {error}")
    failed = sum(1 for report in reports if not report.ok)
    print(f"{len(reports) - failed} of {len(reports)} problems passed validation")
    return 1 if failed else 0


def _problem_dir(root: Path, name: str) -> Path:
    """Resolve an id or path to a directory without requiring a valid problem.toml."""
    as_path = Path(name)
    if as_path.is_dir() and (as_path / "problem.toml").exists():
        return as_path.resolve()
    matches = [p for p in root.glob("problems/*/*") if p.is_dir() and p.name == name.split("/")[-1]]
    if "/" in name:
        matches = [p for p in matches if p.parent.name == name.split("/")[0]]
    if len(matches) != 1:
        raise ProblemError(f"cannot find exactly one problem directory for {name!r}")
    return matches[0]


def cmd_new(args: argparse.Namespace) -> int:
    try:
        path = scaffold.create(
            find_root(),
            args.target,
            title=args.title,
            frameworks=[f.strip() for f in args.frameworks.split(",") if f.strip()],
            difficulty=args.difficulty,
        )
    except scaffold.ScaffoldError as exc:
        print(f"ws new: {exc}", file=sys.stderr)
        return 2
    print(f"created {path}")
    print(f"every {gate.PLACEHOLDER} marker must be replaced before `ws validate` passes;")
    print("see docs/PROBLEM_AUTHORING.md")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ws", description="Whetstone problems from the terminal.")
    parser.add_argument("--version", action="version", version=f"whetstone {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    sub.add_parser("list", help="list the problem catalog")

    for name, text in (
        ("run", "run the public cases against a solution"),
        ("submit", "run all cases (public and hidden) against a solution"),
    ):
        cmd = sub.add_parser(name, help=text)
        cmd.add_argument("problem", help="problem id")
        cmd.add_argument("-f", "--framework", help="default: the problem's first framework")
        which = cmd.add_mutually_exclusive_group()
        which.add_argument("--solution", help="path of the solution file "
                           "(default: workspace/<id>/<framework>.py)")  # fmt: skip
        which.add_argument("--reference", action="store_true", help="judge the reference")
        which.add_argument("--starter", action="store_true", help="judge the starter")
        which.add_argument("--mutant", metavar="NAME", help="judge mutants/NAME.py")
        cmd.add_argument("--case", help="run only this case id")
        cmd.add_argument("--events", action="store_true", help="print raw protocol events")

    cmd = sub.add_parser("validate", help="the quality gate for problems")
    cmd.add_argument("problems", nargs="*", metavar="id", help="problem ids")
    cmd.add_argument("--all", action="store_true", help="validate the whole catalog")
    cmd.add_argument("-j", "--jobs", type=int, default=None, help="parallel harness runs")
    cmd.add_argument("-v", "--verbose", action="store_true", help="also list passed checks")

    cmd = sub.add_parser("new", help="scaffold a problem directory")
    cmd.add_argument("target", metavar="<track>/<id>")
    cmd.add_argument("--title", default=None)
    cmd.add_argument("--frameworks", default="torch,jax,mlx", help='comma separated, or "python"')
    cmd.add_argument("--difficulty", default="easy", choices=["easy", "medium", "hard"])
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _dispatch(parser, args)
    except BrokenPipeError:  # output piped into `head` and the like
        sys.stderr.close()
        return 1


def _dispatch(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if args.command == "list":
        return cmd_list(args)
    if args.command in ("run", "submit"):
        return cmd_run(args, args.command)
    if args.command == "validate":
        return cmd_validate(args)
    if args.command == "new":
        return cmd_new(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
