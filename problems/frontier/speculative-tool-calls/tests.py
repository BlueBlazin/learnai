"""Tests for speculative-tool-calls.

`play` runs the solution on a simulated clock against a scripted model that streams
one piece every 1/8 s, and tools whose latencies are odd multiples of 1/16 s (so a
tool never finishes at the instant a piece arrives: there are no ties). It compares
four things with what the statement prescribes:

1. the transcript, and the protocol violations the model recorded;
2. the tool executions: for every tool the exact set of runs, each with its
   arguments, start time, status (done, failed, or canceled), and end time;
3. that no tool with side effects was started before the end of its turn;
4. the time at which the run ends, which must not exceed the prescribed time. The
   time of the non-speculative loop of `tool-call-loop` is reported next to it.

The expected side is computed by `prescribed`, a synchronous restatement of the
rules that walks the pieces of each turn; it does not come from the reference.
"""

import copy
import json

from learnai.judge import case, check
from learnai.kit import ScriptedChatModel, ScriptedTool, reply, tool_call

PIECE_S = 0.125  # one stream piece every 1/8 s
CHUNK = 8  # characters of content or arguments per piece
WEATHER = {"Paris": 21, "Oslo": 4, "Cairo": 33, "Lima": 18}
SENT = []


def get_weather(city):
    if city not in WEATHER:
        raise ValueError(f"no weather station in {city}")
    return f"{WEATHER[city]} C in {city}"


def get_time(zone):
    return {"zone": zone, "time": "14:05"}


def search(q, limit=2):
    return [f"{q} result {i}" for i in range(limit)]


def send_email(to, body=""):
    if to == "nobody":
        raise RuntimeError("no such recipient")
    return f"sent to {to}"


def book_table(name, seats):
    return {"booked": name, "seats": seats}


# name -> (function, latency in seconds, has side effects)
TOOLBOX = {
    "get_weather": (get_weather, 0.6875, False),
    "get_time": (get_time, 0.3125, False),
    "search": (search, 1.0625, False),
    "send_email": (send_email, 0.4375, True),
    "book_table": (book_table, 0.8125, True),
}


def make_tools(names):
    return {
        n: ScriptedTool(n, TOOLBOX[n][0], delay_s=TOOLBOX[n][1], side_effects=TOOLBOX[n][2])
        for n in names
    }


# -- predictors ----------------------------------------------------------------------------


def never(name, text):
    return None


def closes_object(name, text):
    """Predicts that the object ends here: the text so far plus a closing brace."""
    try:
        guess = json.loads(text + "}")
    except json.JSONDecodeError:
        return None
    return guess if isinstance(guess, dict) else None


def habits(table):
    """Predicts a tool's usual arguments as soon as its name is known."""

    def predictor(name, text):
        return copy.deepcopy(table.get(name))

    return predictor


def habits_then_close(table):
    def predictor(name, text):
        if name in table:
            return copy.deepcopy(table[name])
        return closes_object(name, text)

    return predictor


# -- the rules, restated -------------------------------------------------------------------


def error(text):
    return json.dumps({"error": text})


def parse(text):
    try:
        arguments = json.loads(text)
    except json.JSONDecodeError:
        return None
    return arguments if isinstance(arguments, dict) else None


def piece_times(message, start):
    """(time of the finish piece, [(header time, [fragment times], completion time)])."""
    content = message.get("content") or ""
    now = start + PIECE_S * -(-len(content) // CHUNK)
    calls = []
    for call in message.get("tool_calls", ()):
        now += PIECE_S
        header = now
        fragments = []
        for _ in range(-(-len(call["function"]["arguments"]) // CHUNK)):
            now += PIECE_S
            fragments.append(now)
        calls.append([header, fragments, None])
    finish = now + PIECE_S
    for i, entry in enumerate(calls):
        entry[2] = calls[i + 1][0] if i + 1 < len(calls) else finish
    return finish, calls


def run_of(name, arguments, start):
    """(outcome, run record) of a tool run that is allowed to finish."""
    fn, delay, _ = TOOLBOX[name]
    try:
        result = fn(**arguments)
    except Exception as exc:  # noqa: BLE001  a tool may raise anything
        return error(f"tool failed: {exc}"), (arguments, start, "failed", start + delay)
    outcome = result if isinstance(result, str) else json.dumps(result)
    return outcome, (arguments, start, "done", start + delay)


def prescribed(script, names, messages, predictor, max_steps):
    """What the statement prescribes.

    Returns (transcript, runs, turn_ends, elapsed, baseline): `runs` maps a tool's name
    to its run records (arguments, start, status, end); `turn_ends` are the times of the
    finish pieces; `baseline` is when the non-speculative loop would end.
    """
    model = ScriptedChatModel(script)
    transcript = copy.deepcopy(messages)
    runs = {name: [] for name in names}
    now = baseline = 0.0
    turn_ends = []
    for _ in range(max_steps):
        message = model.complete(transcript)
        transcript.append(message)
        finish, times = piece_times(message, now)
        turn_ends.append(finish)
        end, longest = finish, 0.0
        for call, (header, fragments, completion) in zip(
            message.get("tool_calls", ()), times, strict=True
        ):
            name, text = call["function"]["name"], call["function"]["arguments"]
            final = parse(text)
            guess = None  # (arguments, start time) of a speculative run
            if name in names and not TOOLBOX[name][2]:
                seen = ""
                for when, fragment in zip(
                    [header, *fragments],
                    ["", *(text[i : i + CHUNK] for i in range(0, len(text), CHUNK))],
                    strict=True,
                ):
                    seen += fragment
                    arguments = predictor(name, seen)
                    if arguments is not None:
                        guess = (arguments, when)
                        break
            confirmed = guess is not None and final is not None and guess[0] == final
            if guess is not None and not confirmed:
                # Discarded at completion: canceled, unless it had already finished.
                _, record = run_of(name, guess[0], guess[1])
                if record[3] > completion:
                    record = (guess[0], guess[1], "canceled", completion)
                runs[name].append(record)
            if name not in names:
                outcome = error(f"unknown tool: {name}")
            elif final is None:
                outcome = error("invalid arguments")
            else:
                if confirmed:
                    start = guess[1]
                else:
                    start = finish if TOOLBOX[name][2] else completion
                outcome, record = run_of(name, final, start)
                runs[name].append(record)
                end = max(end, record[3])
                longest = max(longest, TOOLBOX[name][1])
            transcript.append({"role": "tool", "tool_call_id": call["id"], "content": outcome})
        baseline += (finish - now) + longest
        now = end
        if not message.get("tool_calls"):
            break
    return transcript, runs, turn_ends, now, baseline


def show(messages):
    lines = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            lines.append(f"  [{i}] {m!r}")
        elif m.get("role") == "assistant" and m.get("tool_calls"):
            calls = ", ".join(
                f"{c['id']}: {c['function']['name']}({c['function']['arguments']})"
                for c in m["tool_calls"]
            )
            lines.append(f"  [{i}] assistant requests {calls}")
        elif m.get("role") == "tool":
            lines.append(f"  [{i}] tool {m.get('tool_call_id')!r}: {m.get('content')!r}")
        else:
            lines.append(f"  [{i}] {m.get('role')}: {m.get('content')!r}")
    return "\n".join(lines) + "\n"


def show_runs(records):
    if not records:
        return "    (none)\n"
    return "".join(
        f"    {json.dumps(args)}  started {start:g}  {status} at {end:g}\n"
        for args, start, status, end in records
    )


def by_time(records):
    return sorted(records, key=lambda r: (r[1], json.dumps(r[0], sort_keys=True), r[2]))


def play(ctx, script, messages, predictor, max_steps, *, names=None):
    names = list(TOOLBOX) if names is None else names
    want, want_runs, turn_ends, want_elapsed, baseline = prescribed(
        script, names, messages, predictor, max_steps
    )

    sim = ctx.simclock()
    model = ScriptedChatModel(script, delay_s=PIECE_S, chunk_chars=CHUNK)
    tools = make_tools(names)
    given = copy.deepcopy(messages)
    try:
        got = sim.run_until_complete(
            ctx.solution.run_agent(model, tools, given, predictor, max_steps)
        )
    except NotImplementedError:
        raise
    except Exception as exc:  # noqa: BLE001  reported as a failed check, with a pointer
        check.that(
            False,
            f"run_agent raised {type(exc).__name__}: {str(exc)[:120]}",
            "A bad call or a failing tool is an outcome for the model, never an exception "
            "of the loop. That includes a speculative run that fails or is canceled.\n",
        )
    elapsed = sim.now()

    check.that(
        isinstance(got, list) and all(isinstance(m, dict) for m in got),
        "run_agent: must return the transcript, a list of message dicts",
        f"got: {type(got).__name__}",
    )
    check.that(
        given == messages and got is not given,
        "run_agent: the `messages` list it was given was modified (or returned itself)",
    )
    if model.violations:
        first = model.violations[0]
        check.that(
            False,
            f"the model rejected a conversation it was given ({first.code})",
            f"{first}\nthe transcript returned:\n{show(got)}",
        )

    # 1. The transcript.
    both = f"expected transcript:\n{show(want)}got:\n{show(got)}"
    for i, (g, w) in enumerate(zip(got, want, strict=False)):
        if g != w:
            check.that(
                False,
                f"message {i} of the transcript (role {g.get('role')!r}) is not what the "
                "rules prescribe",
                f"expected: {w!r}\ngot:      {g!r}\n\n{both}",
            )
    check.that(
        len(got) == len(want),
        f"the transcript has {len(got)} messages, expected {len(want)}",
        both,
    )

    # 2. Side effects never start before the end of their turn.
    for name, tool in tools.items():
        if not tool.side_effects:
            continue
        early = [r for r in tool.runs if r.started_s not in turn_ends]
        check.that(
            not early,
            f"{name!r} has side effects and was started at t={early[0].started_s:g} s, "
            "which is not the end of a turn"
            if early
            else "",
            f"the turns ended at {turn_ends}\nruns of {name!r}:\n"
            + show_runs([(r.arguments, r.started_s, r.status, r.finished_s) for r in tool.runs]),
        )

    # 3. The exact set of tool executions.
    total = sum(len(tool.runs) for tool in tools.values())
    for name, tool in tools.items():
        got_runs = by_time((r.arguments, r.started_s, r.status, r.finished_s) for r in tool.runs)
        expected = by_time(want_runs[name])
        if got_runs == expected:
            continue
        detail = (
            f"expected runs of {name!r}:\n{show_runs(expected)}got:\n{show_runs(got_runs)}\n{both}"
        )
        unfinished = [r for r in got_runs if r[2] == "running"]
        check.that(
            not unfinished,
            f"a run of {name!r} started at t={unfinished[0][1]:g} s was still running when "
            "run_agent returned: a speculative run that is not used must be canceled"
            if unfinished
            else "",
            detail,
        )
        check.that(
            len(got_runs) == len(expected),
            f"{name!r} was started {len(got_runs)} times, expected {len(expected)} (the runs "
            "the rules prescribe)",
            detail,
        )
        check.that(
            [r[0] for r in got_runs] == [r[0] for r in expected]
            and [r[1] for r in got_runs] == [r[1] for r in expected],
            f"the runs of {name!r} do not have the prescribed arguments and start times "
            f"(they started at {[r[1] for r in got_runs]}, expected "
            f"{[r[1] for r in expected]})",
            detail,
        )
        check.that(
            False,
            f"the runs of {name!r} did not end as prescribed (their statuses are "
            f"{[r[2] for r in got_runs]}, expected {[r[2] for r in expected]})",
            detail,
        )
    ctx.metric("tool_runs", total, unit="runs")

    # 4. The time.
    ctx.metric("non_speculative", baseline, unit="s")
    check.at_most("elapsed", elapsed, want_elapsed, unit="s")
    return got, tools, elapsed, baseline


USER = [{"role": "user", "content": "Weather in Paris? Then mail Ann."}]
EXAMPLE = [
    reply(
        tool_calls=[
            tool_call("get_weather", {"city": "Paris"}),
            tool_call("send_email", {"to": "ann"}),
        ]
    ),
    reply(tool_calls=[tool_call("search", '{"q":"x","limit":1}')]),
    "Done.",
]


@case("the worked example: a confirmed guess, a side effect, a wrong guess; 3.375 s", public=True)
def worked_example(ctx):
    got, tools, elapsed, baseline = play(ctx, EXAMPLE, USER, closes_object, 8)
    check.equal(
        [m.get("content") for m in got if m["role"] == "tool"],
        ["21 C in Paris", "sent to ann", '["x result 0"]'],
        label="contents of the tool messages",
    )
    runs = [(r.arguments, r.started_s, r.status) for r in tools["search"].runs]
    check.equal(
        runs,
        [({"q": "x"}, 1.6875, "canceled"), ({"q": "x", "limit": 1}, 2.0625, "done")],
        label="runs of search",
    )
    if (elapsed, baseline) != (3.375, 3.625):
        raise RuntimeError(f"authoring error: the example takes {elapsed} s, baseline {baseline}")


@case("no predictions: a read-only call still starts when its arguments are complete", public=True)
def starts_at_completion(ctx):
    # Three calls in one turn; the first two are complete long before the turn ends.
    script = [
        reply(
            tool_calls=[
                tool_call("search", {"q": "bistro", "limit": 3}),
                tool_call("get_weather", {"city": "Lima"}),
                tool_call("get_time", {"zone": "PET"}),
            ]
        ),
        "Lima: 18 C at 14:05.",
    ]
    play(ctx, script, USER, never, 4)


@case("tools with side effects wait for the end of the turn, whatever is predicted", public=True)
def side_effects_wait(ctx):
    # The predictor knows both calls in advance. Only the read-only one may use that.
    table = {
        "book_table": {"name": "Lo", "seats": 2},
        "send_email": {"to": "ann", "body": "booked"},
        "get_time": {"zone": "CET"},
    }
    script = [
        reply(
            tool_calls=[
                tool_call("book_table", {"name": "Lo", "seats": 2}),
                tool_call("get_time", {"zone": "CET"}),
                tool_call("send_email", {"to": "ann", "body": "booked"}),
            ]
        ),
        reply(tool_calls=[tool_call("send_email", {"to": "bob"})]),
        "Booked and sent.",
    ]
    play(ctx, script, USER, habits(table), 5)


@case("a guess that is almost right is wrong: it is canceled and the real call runs", public=True)
def near_miss(ctx):
    # Same tool, same first argument; the prediction lacks `limit`, or has another city.
    table = {"get_weather": {"city": "Oslo"}}
    script = [
        reply(
            tool_calls=[
                tool_call("search", '{"q":"x","limit":3}'),
                tool_call("get_weather", {"city": "Cairo"}),
            ]
        ),
        reply(tool_calls=[tool_call("get_weather", {"city": "Oslo"})]),
        "33 C in Cairo, 4 C in Oslo.",
    ]
    play(ctx, script, USER, habits_then_close(table), 5)


@case("arguments are compared as values: spacing and key order do not matter")
def compared_as_values(ctx):
    table = {"search": {"q": "llm", "limit": 1}, "get_weather": {"city": "Paris"}}
    script = [
        reply(
            tool_calls=[
                tool_call("search", '{ "limit" : 1, "q" : "llm" }'),
                tool_call("get_weather", '{"city":"Paris"}'),
            ]
        ),
        "ok",
    ]
    play(ctx, script, USER, habits(table), 3)


@case("a wrong guess that has already finished, or already failed, is dropped silently")
def finished_before_completion(ctx):
    # get_time takes 0.3125 s; these arguments stream for longer than that.
    table = {"get_time": {"zone": "CET"}, "get_weather": {"city": "Atlantis"}}
    script = [
        reply(
            tool_calls=[
                tool_call("get_time", {"zone": "America/Argentina/Buenos_Aires"}),
                tool_call("get_weather", '{"city":                                "Lima"}'),
                tool_call("get_time", {"zone": "CET"}),
            ]
        ),
        "ok",
    ]
    play(ctx, script, USER, habits(table), 3)


@case("bad calls: unknown tools, arguments that are not a JSON object, a failing tool")
def bad_calls(ctx):
    table = {"get_weather": {"city": "Paris"}, "search": {"q": "a"}}
    script = [
        reply(
            tool_calls=[
                tool_call("get_weather", '{"city": '),
                tool_call("get_stock", {"symbol": "X"}),
                tool_call("search", "[1, 2]"),
                tool_call("send_email", '{"to": "ann"'),
                tool_call("get_weather", {"city": "Atlantis"}),
                tool_call("send_email", {"to": "nobody"}),
            ]
        ),
        "Nothing worked.",
    ]
    play(ctx, script, USER, habits(table), 3)
    play(ctx, script, USER, closes_object, 3, names=["get_weather", "send_email"])


@case("the same call twice in one turn; text before the calls; a predictor that never hits")
def duplicates_and_content(ctx):
    table = {"get_weather": {"city": "Oslo"}, "get_time": {"zone": "XYZ"}}
    script = [
        reply(
            "Let me check both.",
            tool_calls=[
                tool_call("get_weather", {"city": "Oslo"}),
                tool_call("get_weather", {"city": "Oslo"}),
                tool_call("get_time", {"zone": "CET"}),
            ],
        ),
        "4 C.",
    ]
    play(ctx, script, USER, habits(table), 3)


@case("the step limit: the last turn's calls are answered, and nothing is left running")
def step_limit(ctx):
    table = {"search": {"q": "a"}}
    script = [
        reply(tool_calls=[tool_call("search", {"q": "a"})]),
        reply(tool_calls=[tool_call("search", {"q": "b"}), tool_call("send_email", {"to": "al"})]),
        reply(tool_calls=[tool_call("search", {"q": "a"})]),
        "never reached",
    ]
    for max_steps in (1, 2, 3):
        got, _, _, _ = play(ctx, script, USER, habits(table), max_steps)
        check.that(got[-1]["role"] == "tool", "the run must end with the last turn's tool results")


@case("a six-step session: confirmed, wrong, and absent predictions, and side effects")
def long_session(ctx):
    table = {"get_time": {"zone": "CET"}, "book_table": {"name": "Lo", "seats": 2}}
    script = [
        reply(tool_calls=[tool_call("get_time", {"zone": "CET"})]),
        reply(
            "Searching.",
            tool_calls=[
                tool_call("search", '{"q":"a","limit":2}'),
                tool_call("get_weather", {"city": "Cairo"}),
                tool_call("get_time", {"zone": "EET"}),
            ],
        ),
        reply(tool_calls=[tool_call("book_table", {"name": "Lo", "seats": 2})]),
        reply(
            tool_calls=[
                tool_call("send_email", {"to": "ann", "body": "see you at eight"}),
                tool_call("get_weather", {"city": "Lima"}),
                tool_call("get_time", {"zone": "CET"}),
            ]
        ),
        reply(tool_calls=[tool_call("search", {"q": "taxi"})]),
        "All set.",
    ]
    _, _, elapsed, baseline = play(ctx, script, USER, habits_then_close(table), 8)
    if not elapsed < baseline - 1.0:
        raise RuntimeError(f"authoring error: {elapsed} s against {baseline} s")
