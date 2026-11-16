"""The ``learnai`` command-line interface (docs/ARCHITECTURE.md section 3.7)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from learnai import __version__, assets, protocol, render, runner, scaffold
from learnai import validate as gate
from learnai.problem import Problem, ProblemError, discover, find_problem, find_root, problem_dirs


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
        print(f"learnai: {exc}", file=sys.stderr)
        return 2

    command = runner.harness_command(
        problem.path, solution, framework, mode, args.case, reveal=args.reveal
    )
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
        print("learnai validate: give problem ids or --all (not both)", file=sys.stderr)
        return 2
    try:
        if args.all:
            directories = problem_dirs(root)
        else:
            directories = [_problem_dir(root, name) for name in args.problems]
    except ProblemError as exc:
        print(f"learnai: {exc}", file=sys.stderr)
        return 2
    if not directories:
        print(f"learnai validate: no problems under {root / 'problems'}", file=sys.stderr)
        return 2

    reports = gate.validate(root, directories, jobs=args.jobs, require_assets=args.require_assets)
    paint = render.Renderer().paint
    tags = {
        "ok": paint("ok  ", "pass"),
        "fail": paint("FAIL", "fail"),
        "skipped": paint("SKIP", "incomplete"),
    }
    for report in reports:
        print(f"{tags[report.status]} {report.path.parent.name}/{report.problem_id}")
        if args.verbose or not report.ok:
            for check in report.checks:
                print(f"       {paint(check, 'dim')}")
        for skip in report.skips:
            print(f"       skipped: {skip}")
        for warning in report.warnings:
            print(f"       warning: {warning}")
        for error in report.errors:
            print(f"       error: {error}")
    failed = sum(1 for report in reports if report.status == "fail")
    skipped = sum(1 for report in reports if report.status == "skipped")
    summary = f"{len(reports) - failed - skipped} of {len(reports)} problems passed validation"
    if skipped:
        summary += f"; {skipped} skipped (not validated on this machine, see above)"
    partial = sum(
        1
        for report in reports
        if report.status == "ok" and any("NOT validated" in w for w in report.warnings)
    )
    if partial:
        summary += (
            f"; {partial} passed with cases that could NOT be validated on this machine "
            "(see the warnings above)"
        )
    print(summary)
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
        print(f"learnai new: {exc}", file=sys.stderr)
        return 2
    print(f"created {path}")
    print(f"every {gate.PLACEHOLDER} marker must be replaced before `learnai validate` passes;")
    print("see docs/PROBLEM_AUTHORING.md")
    return 0


def cmd_assets(args: argparse.Namespace) -> int:
    root = find_root()
    if args.action == "list":
        return _assets_list(root, as_json=args.json)
    if args.action == "fetch":
        return _assets_fetch(root, args)
    print("learnai assets: give `list` or `fetch <name>`", file=sys.stderr)
    return 2


def _assets_list(root: Path, *, as_json: bool) -> int:
    listing = assets.listing(root)
    if as_json:
        print(json.dumps(listing, indent=2))
        return 0
    if not listing["assets"]:
        print("no problem in the catalog declares a model asset")
    rows = [("ASSET", "STATE", "SIZE", "REPOSITORY", "USED BY")]
    for item in listing["assets"]:
        on_disk = item["bytes_on_disk"] if item["present"] else item["size_mb"] * 1e6
        size = ("" if item["present"] else "~") + assets.format_size(on_disk)
        used = ", ".join(item["problems"][:4])
        if len(item["problems"]) > 4:
            used += f", and {len(item['problems']) - 4} more"
        rows.append((item["name"], item["state"], size, item["repo"], used))
    if len(rows) > 1:
        widths = [max(len(row[i]) for row in rows) for i in range(4)]
        for row in rows:
            print(
                "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row[:4])) + "  " + row[4]
            )
    for item in listing["assets"]:
        if not item["present"]:
            why = f" ({item['reason']})" if item["reason"] else ""
            print(f"{item['name']} is {item['state']}{why}: {item['fetch_command']}")
    for issue in listing["issues"]:
        print(f"warning: {issue['problem']}: {issue['message']}")
    print(f"assets are stored under {listing['models_dir']}")
    return 0


def _assets_fetch(root: Path, args: argparse.Namespace) -> int:
    merged, _ = assets.catalog(root)
    # The house models can be fetched before any problem declares them (authors need
    # the weights to write the problem). A catalog declaration wins when there is one.
    from learnai.kit.realmodel import HOUSE_MODELS

    for name, spec in HOUSE_MODELS.items():
        merged.setdefault(name, assets.CatalogAsset(spec))
    if args.all == bool(args.names):
        print("learnai assets fetch: give asset names or --all (not both)", file=sys.stderr)
        return 2
    names = sorted(merged) if args.all else args.names
    unknown = [name for name in names if name not in merged]
    if unknown:
        known = ", ".join(sorted(merged)) or "none"
        print(
            f"learnai assets fetch: no asset named {', '.join(unknown)} in the catalog (known: {known})",
            file=sys.stderr,
        )
        return 2

    def show(event: dict) -> None:
        kind = event["event"]
        if args.json:
            print(protocol.encode(event), flush=True)
        elif kind == "asset_start":
            total = assets.format_size(event["total_bytes"])
            print(f"{event['asset']}: {len(event['files'])} files, {total}, from {event['repo']}")
        elif kind == "asset_progress":
            percent = 100 * event["done_bytes"] / max(event["total_bytes"], 1)
            done, total = (assets.format_size(event[k]) for k in ("done_bytes", "total_bytes"))
            print(f"  {percent:5.1f}%  {done} of {total}  {event['file']}", flush=True)
        elif kind == "asset_file":
            note = " (already downloaded)" if event["cached"] else ""
            print(f"  done   {event['file']}{note}", flush=True)
        elif kind == "asset_end" and event["ok"]:
            print(f"{event['asset']}: ready at {event['path']}")
        elif kind == "asset_end":
            print(f"{event['asset']}: failed: {event['message']}", file=sys.stderr)

    failed = 0
    for name in names:
        try:
            assets.fetch(
                merged[name].spec, assets.models_dir(root), show, verify=not args.no_verify
            )
        except assets.AssetError:
            failed += 1  # reported through the asset_end event
    return 1 if failed else 0


#: Distributions `learnai doctor` reports, by group. Versions are read from the installed
#: metadata, so nothing is imported to find them.
DOCTOR_FRAMEWORKS = ("torch", "jax", "mlx")
DOCTOR_PACKAGES = (
    "numpy", "triton", "huggingface_hub", "safetensors", "tokenizers", "transformers",
    "mlx-lm",
)  # fmt: skip


def doctor_report(root: Path) -> dict:
    """The document ``learnai doctor --json`` prints (PROBLEM_AUTHORING section 19)."""
    import os
    import platform
    from importlib import metadata

    from learnai import hardware

    def version(name: str) -> str | None:
        try:
            return metadata.version(name)
        except Exception:  # noqa: BLE001 - not installed, or broken metadata
            return None

    override = os.environ.get("LEARNAI_HARDWARE")
    return {
        "schema": 1,
        "learnai": __version__,
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "release": platform.release(),
        },
        "frameworks": {name: version(name) for name in DOCTOR_FRAMEWORKS},
        "packages": {name: version(name) for name in DOCTOR_PACKAGES},
        "hardware": hardware.report(),
        "hardware_override": (
            None
            if override is None
            else sorted({item.strip() for item in override.split(",") if item.strip()} | {"cpu"})
        ),
        "root": str(root),
        "models_dir": str(assets.models_dir(root)),
    }


def cmd_doctor(args: argparse.Namespace) -> int:
    report = doctor_report(find_root())
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    system = report["platform"]
    print(f"learnai {report['learnai']}, python {report['python']} ({report['executable']})")
    print(f"platform   {system['system']} {system['machine']} {system['release']}")
    for group in ("frameworks", "packages"):
        for name, found in report[group].items():
            print(f"{name:<16} {found or 'not installed'}")
    hw = report["hardware"]
    print(f"cpu        {hw['cpu']['cores']} cores")
    for key, label in (("mlx-gpu", "mlx-gpu"), ("cuda", "cuda")):
        item = hw[key]
        if item["present"]:
            memory = item["memory_bytes"]
            size = f", {assets.format_size(memory)}" if memory else ""
            print(f"{label:<10} {item['name'] or 'present'}{size}")
        else:
            print(f"{label:<10} not available")
    if report["hardware_override"] is not None:
        print(f"LEARNAI_HARDWARE pretends: {', '.join(report['hardware_override'])}")
    print(f"models     {report['models_dir']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="learnai", description="Learn AI problems from the terminal."
    )
    parser.add_argument("--version", action="version", version=f"learnai {__version__}")
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
        cmd.add_argument(
            "--case", help="run only this case id (hidden cases: hidden-N, or use --reveal)"
        )
        cmd.add_argument(
            "--reveal",
            action="store_true",
            help="for authors: show hidden cases with their real ids, names, and details",
        )
        cmd.add_argument("--events", action="store_true", help="print raw protocol events")

    cmd = sub.add_parser("validate", help="the quality gate for problems")
    cmd.add_argument("problems", nargs="*", metavar="id", help="problem ids")
    cmd.add_argument("--all", action="store_true", help="validate the whole catalog")
    cmd.add_argument("-j", "--jobs", type=int, default=None, help="parallel harness runs")
    cmd.add_argument("-v", "--verbose", action="store_true", help="also list passed checks")
    cmd.add_argument(
        "--require-assets",
        action="store_true",
        help="fail, instead of skipping, a problem whose model asset is not downloaded",
    )

    cmd = sub.add_parser("assets", help="model weights that real-model problems need")
    actions = cmd.add_subparsers(dest="action", metavar="<action>")
    action = actions.add_parser("list", help="what each problem needs and what is downloaded")
    action.add_argument("--json", action="store_true", help="print one JSON document")
    action = actions.add_parser(
        "fetch", help="download assets into .learnai/models (needs the `models` extra)"
    )
    action.add_argument("names", nargs="*", metavar="name", help="asset names")
    action.add_argument("--all", action="store_true", help="every asset in the catalog")
    action.add_argument("--json", action="store_true", help="print progress as @@WS@@ JSON lines")
    action.add_argument(
        "--no-verify", action="store_true", help="skip the SHA-256 check after downloading"
    )

    cmd = sub.add_parser("doctor", help="what this environment has: versions and hardware")
    cmd.add_argument("--json", action="store_true", help="print one JSON document")

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
    if args.command == "assets":
        return cmd_assets(args)
    if args.command == "doctor":
        return cmd_doctor(args)
    if args.command == "new":
        return cmd_new(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
