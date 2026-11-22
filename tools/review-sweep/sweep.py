"""Automated sweep of problems for the defect kinds that a script can see.

It changes no problem. For each problem it submits every mutant with hidden
cases revealed, submits one mutant as a learner sees it, submits the reference,
and reads the problem's files. The flags it raises are leads for a reviewer,
not verdicts: see README.md for what each flag means and what it cannot show.

Run from the repository root:

    python3 tools/review-sweep/sweep.py --out <dir> [--modules a,b] [--ids x,y] [-j 3]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import pathlib
import re
import subprocess
import time
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[2]
LEARNAI = [
    "uv", "run", "--project", "python",
    "--extra", "jax", "--extra", "mlx", "--extra", "models", "--extra", "editor",
    "learnai",
]
MARK = "@@WS@@ "
SERIAL_JUDGES = {"speed", "service"}
NAN = re.compile(r"\bnan\b|\binf(inity)?\b", re.IGNORECASE)
NAN_EXPECTED = re.compile(r"nan|inf|overflow|zero|divide|0 ?/ ?0|underflow", re.IGNORECASE)
EXPECTED_VALUE = re.compile(r"expected[ :=]+[-+\[(]?\d", re.IGNORECASE)
BANNED = re.compile(r"is not allowed in this problem")
NO_BUILTIN = re.compile(
    r"built-?in|do not (call|use)|must not (call|use)|may not (call|use)|"
    r"without (calling|using)|not allowed|off limits|forbidden",
    re.IGNORECASE,
)
NOT_MODIFIED = re.compile(
    r"not (be )?modif|must not mutate|not mutated|left unchanged|"
    r"(is|are|stays?|remains?) unchanged|in place",
    re.IGNORECASE,
)
MUTATES = re.compile(r"in.?place|mutat|modif|overwrit|alias|shares? (the )?(storage|memory)", re.IGNORECASE)
TIE = re.compile(r"\bties?\b|\btied\b|equal (scores|logits|values|priorit)", re.IGNORECASE)


def submit(problem: str, *args: str, timeout: int = 600) -> dict:
    """One `learnai submit --events` run, reduced to what the sweep reads."""
    started = time.monotonic()
    try:
        proc = subprocess.run(
            [*LEARNAI, "submit", problem, "--events", *args],
            cwd=ROOT, capture_output=True, text=True, timeout=timeout, check=False,
        )
        out, err = proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return {"status": "sweep_timeout", "cases": [], "wall_s": timeout, "stderr": ""}
    cases: dict[str, dict] = {}
    order: list[str] = []
    end: dict = {}
    for line in out.splitlines():
        if not line.startswith(MARK):
            continue
        try:
            event = json.loads(line[len(MARK):])
        except json.JSONDecodeError:
            continue
        kind = event.get("event")
        if kind == "start":
            for case in event.get("cases", []):
                cases[case["id"]] = {
                    "id": case["id"], "name": case.get("name"),
                    "public": bool(case.get("public")), "status": None,
                }
                order.append(case["id"])
        elif kind == "case_end":
            case = cases.setdefault(event["id"], {"id": event["id"], "public": None})
            if event["id"] not in order:
                order.append(event["id"])
            case.update(
                status=event.get("status"),
                ms=event.get("duration_ms"),
                message=event.get("message"),
                has_detail=event.get("detail") is not None,
                stdout=(event.get("stdout") or "")[:300],
            )
        elif kind == "end":
            end = event
    return {
        "status": end.get("status", "no_end_event"),
        "passed": end.get("passed"), "total": end.get("total"),
        "skipped": end.get("skipped"), "message": end.get("message"),
        "cases": [cases[i] for i in order],
        "wall_s": round(time.monotonic() - started, 2),
        "stderr": err[-600:] if not end else "",
    }


def bug_line(path: pathlib.Path) -> str:
    for line in path.read_text().splitlines()[:8]:
        if line.startswith("# BUG:"):
            return line[len("# BUG:"):].strip()
    return ""


def discover(modules: list[str] | None, ids: list[str] | None) -> list[dict]:
    if modules is None:
        area = tomllib.loads((ROOT / "areas/inference-and-serving.toml").read_text())
        modules = area["modules"]
    found = []
    for module in modules:
        track = tomllib.loads((ROOT / "problems" / module / "track.toml").read_text())
        core = set(track.get("core", []))
        metas = []
        for directory in (ROOT / "problems" / module).iterdir():
            if (directory / "problem.toml").exists():
                meta = tomllib.loads((directory / "problem.toml").read_text())
                metas.append((meta.get("order", 0), directory, meta))
        for _, directory, meta in sorted(metas, key=lambda m: (m[0], m[1].name)):
            if ids and directory.name not in ids:
                continue
            found.append({
                "id": directory.name, "module": module, "dir": directory, "meta": meta,
                "core": directory.name in core,
            })
    return found


def static_flags(problem: dict) -> tuple[dict, list[dict]]:
    directory, meta = problem["dir"], problem["meta"]
    statement = (directory / "statement.md").read_text()
    mutant_files = sorted((directory / "mutants").glob("*.py"))
    bugs = {m.stem: bug_line(m) for m in mutant_files}
    mutant_text = " ".join(f"{name} {text}" for name, text in bugs.items())
    hints = (directory / "hints.md").read_text() if (directory / "hints.md").exists() else ""
    info = {
        "mutants": len(mutant_files),
        "hints": len(re.findall(r"^#{1,3} ", hints, re.MULTILINE)),
        "forbidden": len(meta.get("forbidden", [])),
        "forbidden_source": meta.get("forbidden_source", []),
        "assets": bool(meta.get("assets")),
    }
    flags = []
    if info["forbidden_source"]:
        flags.append({"flag": "forbidden_source", "weight": 2,
                      "note": "source words banned: " + ", ".join(map(str, info["forbidden_source"]))})
    if NO_BUILTIN.search(statement) and not info["forbidden"] and not info["forbidden_source"]:
        flags.append({"flag": "ban_not_enforced", "weight": 2,
                      "note": "statement rules something out; problem.toml has no forbidden list"})
    if NOT_MODIFIED.search(statement) and not MUTATES.search(mutant_text):
        flags.append({"flag": "no_mutation_mutant", "weight": 1,
                      "note": "statement speaks of inputs or state left unchanged; no mutant is about it"})
    if TIE.search(statement) and not TIE.search(mutant_text):
        flags.append({"flag": "no_tie_mutant", "weight": 1,
                      "note": "statement has a tie rule; no mutant is about ties"})
    if any(not text for text in bugs.values()):
        flags.append({"flag": "mutant_without_bug_line", "weight": 1,
                      "note": ", ".join(n for n, t in bugs.items() if not t)})
    return {**info, "bugs": bugs}, flags


def sweep_problem(problem: dict) -> dict:
    pid, meta = problem["id"], problem["meta"]
    info, flags = static_flags(problem)
    names = sorted(info["bugs"])
    runs = {name: submit(pid, "--mutant", name, "--reveal") for name in names}
    reference = submit(pid, "--reference", "--reveal")
    learner = submit(pid, "--mutant", names[0]) if names else None

    def add(flag: str, weight: int, note: str) -> None:
        flags.append({"flag": flag, "weight": weight, "note": note})

    if reference["status"] in ("hardware_missing", "assets_missing"):
        if any(c["status"] not in (None, "pass", "skipped") for c in reference["cases"]):
            add("reference_not_accepted", 5, f"{reference['status']}, and a case that ran did not pass")
    elif reference["status"] != "accepted":
        add("reference_not_accepted", 5, f"{reference['status']}: {reference.get('message')}")
    slow = [(c["id"], round(c["ms"] / 1000, 1)) for c in reference["cases"]
            if c.get("ms") and c["ms"] > 3000]
    if slow:
        add("slow_case", 1, ", ".join(f"{i} {s} s" for i, s in slow))

    killed_by: dict[str, list[str]] = {}
    for name, run in runs.items():
        cases = [c for c in run["cases"] if c["status"] not in (None, "skipped")]
        bad = [c for c in cases if c["status"] != "pass"]
        for case in bad:
            killed_by.setdefault(case["id"], []).append(name)
        if run["status"] in ("accepted", "hardware_missing", "assets_missing") and not bad:
            add("mutant_not_rejected", 5, f"{name}: ended {run['status']}")
            continue
        if run["status"] in ("no_end_event", "sweep_timeout", "import_error") or not cases:
            add("mutant_did_not_run", 3, f"{name}: {run['status']} {run.get('message') or run['stderr'][-160:]}")
            continue
        fails = [c for c in bad if c["status"] == "fail"]
        if bad and not fails:
            kinds = sorted({c["status"] for c in bad})
            add("crash_only", 3, f"{name}: rejected only by {', '.join(kinds)}: "
                + (bad[0].get("message") or "")[:140])
        elif fails and all(NAN.search(c.get("message") or "") for c in fails) \
                and not NAN_EXPECTED.search(info["bugs"][name]):
            add("nan_only", 2, f"{name}: every failing case is a NaN or infinity; BUG line: "
                + info["bugs"][name][:120])
        banned = all(BANNED.search(c.get("message") or "") for c in bad)
        if len(cases) >= 4 and len(bad) == len(cases) and not banned:
            add("fails_every_case", 1, f"{name}: fails all {len(cases)} cases")
        if bad and all(c["public"] is False for c in bad):
            add("hidden_only", 2, f"{name}: caught only by hidden cases "
                + ", ".join(c["id"] for c in bad[:4]))
        if len(bad) == 1:
            add("single_case", 0, f"{name}: caught by one case, {bad[0]['id']}")

    all_cases = [c for c in reference["cases"] if c["status"] not in (None, "skipped")]
    idle = [c["id"] for c in all_cases if c["id"] not in killed_by]
    if idle:
        add("idle_case", 0, "no mutant fails: " + ", ".join(idle))
    skipped = [c["id"] for c in reference["cases"] if c["status"] == "skipped"]
    if skipped:
        add("skipped_case", 0, "not run here: " + ", ".join(skipped))

    if learner:
        hidden = [c for c in learner["cases"] if c["public"] is False]
        for case in hidden:
            if not re.fullmatch(r"Hidden case \d+", case.get("name") or ""):
                add("hidden_leak", 4, f"hidden case shown with name {case.get('name')!r}")
            if case.get("has_detail"):
                add("hidden_leak", 4, f"{case['id']} shows detail to the learner")
            if case.get("stdout"):
                add("hidden_leak", 3, f"{case['id']} shows stdout: {case['stdout'][:100]!r}")
    messages = sorted({
        (c.get("message") or "")[:200]
        for run in runs.values() for c in run["cases"]
        if c["public"] is False and c["status"] not in (None, "pass", "skipped")
        and EXPECTED_VALUE.search(c.get("message") or "")
    })
    if messages:
        add("hidden_message_values", 1, f"{len(messages)} hidden failure messages state an expected value, e.g. "
            + messages[0][:150])

    return {
        "id": pid, "module": problem["module"], "core": problem["core"],
        "difficulty": meta.get("difficulty"), "judges": meta.get("judges", []),
        "framework": (meta.get("frameworks") or ["?"])[0],
        "info": {k: v for k, v in info.items() if k != "bugs"},
        "bugs": info["bugs"], "flags": flags,
        "score": sum(f["weight"] for f in flags),
        "reference": reference, "mutants": runs, "learner_view": learner,
        "hidden_messages": messages,
    }


def report(results: list[dict]) -> str:
    lines = ["# Review sweep", "",
             f"{len(results)} problems, {sum(len(r['mutants']) for r in results)} mutant runs.", ""]
    counts: dict[str, set] = {}
    for r in results:
        for f in r["flags"]:
            counts.setdefault(f["flag"], set()).add(r["id"])
    lines += ["## Flags", "", "| flag | problems |", "|---|---|"]
    lines += [f"| `{flag}` | {len(ids)} |" for flag, ids in sorted(counts.items(), key=lambda kv: -len(kv[1]))]
    lines += ["", "## By module", "", "| module | problems | with a weighted flag | score |", "|---|---|---|---|"]
    modules: dict[str, list[dict]] = {}
    for r in results:
        modules.setdefault(r["module"], []).append(r)
    for module, rs in modules.items():
        lines.append(f"| `{module}` | {len(rs)} | {sum(1 for r in rs if r['score'])} | {sum(r['score'] for r in rs)} |")
    lines += ["", "## Problems, highest score first", ""]
    for r in sorted(results, key=lambda r: (-r["score"], r["module"], r["id"])):
        weighted = [f for f in r["flags"] if f["weight"]]
        if not weighted:
            continue
        core = ", core" if r["core"] else ""
        lines.append(f"### `{r['id']}` ({r['module']}{core}), score {r['score']}")
        lines += [f"- `{f['flag']}`: {f['note']}" for f in weighted]
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--modules")
    parser.add_argument("--ids")
    parser.add_argument("-j", "--jobs", type=int, default=3)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    problems = discover(args.modules.split(",") if args.modules else None,
                        set(args.ids.split(",")) if args.ids else None)
    parallel = [p for p in problems if not SERIAL_JUDGES & set(p["meta"].get("judges", []))]
    serial = [p for p in problems if SERIAL_JUDGES & set(p["meta"].get("judges", []))]
    results: dict[str, dict] = {}

    def done(result: dict) -> None:
        results[result["id"]] = result
        (args.out / "problems").mkdir(exist_ok=True)
        (args.out / "problems" / f"{result['id']}.json").write_text(json.dumps(result, indent=1, default=str))
        print(f"[{len(results)}/{len(problems)}] {result['id']}: score {result['score']}", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for result in pool.map(sweep_problem, parallel):
            done(result)
    for problem in serial:
        done(sweep_problem(problem))
    ordered = [results[p["id"]] for p in problems]
    (args.out / "results.json").write_text(json.dumps(ordered, indent=1, default=str))
    (args.out / "report.md").write_text(report(ordered))
    print(f"wrote {args.out / 'report.md'}")


if __name__ == "__main__":
    main()
