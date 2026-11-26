"""Tests for tool-call-loop.

`play` runs the solution on a simulated clock against a scripted model (one model
call takes 1 s) and a fixed set of tools with latencies, then compares four things
with what the statement prescribes: the transcript, the protocol violations the
model recorded, what was run (model calls, tool runs and their arguments), and when
(every tool of a step starts when the model's message arrives; the run ends after
the sum of model latency plus the longest tool of each step). The expected side is
computed here by `prescribed`, a plain synchronous restatement of the rules, so it
does not come from the reference. All latencies are multiples of 1/64 s: every time
is exact.

The model and the tools are handed over behind the kit's `Guarded` (`Model`, `Tool`).
A solution may call the model and await a tool, and read their names and latencies.
The model's script, counters, and logs, and a tool's function, run log, and counters
are refused, so a reply cannot be read before its call and a run cannot go unrecorded.
"""

import copy
import json

from learnai.judge import case, check
from learnai.kit import (
    Counters,
    Guarded,
    ScriptedChatModel,
    ScriptedTool,
    check_guard,
    guard,
    reply,
    tool_call,
)

MODEL_S = 1.0


class Model(Guarded, ScriptedChatModel):
    """The model a solution gets: the call; not the script, the counters, or the logs."""

    label = "model"
    refused = Guarded.refused | {"script", "calls", "replies", "violations"}


class Tool(Guarded, ScriptedTool):
    """The tool a solution gets: `await tool(**arguments)`; not its function or its log."""

    label = "tool"
    refused = Guarded.refused | {"fn", "runs"}


WEATHER = {"Paris": 21, "Oslo": 4, "Cairo": 33, "Lima": 18}


def get_weather(city):
    if city not in WEATHER:
        raise ValueError(f"no weather station in {city}")
    return f"{WEATHER[city]} C in {city}"


def get_time(zone):
    return {"zone": zone, "time": "14:05"}


def add(a, b):
    return a + b


def search(q, limit=2):
    return [{"title": f"{q} result {i}", "rank": i} for i in range(limit)]


def book_table(name, seats):
    if seats > 6:
        raise RuntimeError("no table for more than 6")
    return f'booked "{name}" for {seats}'


# name -> (function, latency in seconds)
TOOLBOX = {
    "get_weather": (get_weather, 0.5),
    "get_time": (get_time, 0.25),
    "add": (add, 0.0),
    "search": (search, 2.0),
    "book_table": (book_table, 0.75),
}


def make_tools(names=None):
    names = TOOLBOX if names is None else names
    return {n: Tool(n, TOOLBOX[n][0], delay_s=TOOLBOX[n][1]) for n in names}


def error(text):
    return json.dumps({"error": text})


def outcome_of(call, names):
    """(outcome, the tool run as (name, arguments) or None) by the statement's table."""
    name, text = call["function"]["name"], call["function"]["arguments"]
    if name not in names:
        return error(f"unknown tool: {name}"), None
    try:
        arguments = json.loads(text)
    except json.JSONDecodeError:
        return error("invalid arguments"), None
    if not isinstance(arguments, dict):
        return error("invalid arguments"), None
    try:
        result = TOOLBOX[name][0](**arguments)
    except Exception as exc:  # noqa: BLE001  a tool may raise anything
        return error(f"tool failed: {exc}"), (name, arguments)
    return (result if isinstance(result, str) else json.dumps(result)), (name, arguments)


def prescribed(script, names, messages, max_steps):
    """What the statement prescribes: (transcript, steps, runs, elapsed).

    steps: one (time the model's message arrives, [(name, arguments), ...]) per step.
    """
    model = ScriptedChatModel(script)
    transcript = copy.deepcopy(messages)
    now, steps = 0.0, []
    for _ in range(max_steps):
        message = model.complete(transcript)
        now += MODEL_S
        transcript.append(message)
        calls = message.get("tool_calls")
        if not calls:
            break
        runs = []
        for call in calls:
            outcome, run = outcome_of(call, names)
            if run is not None:
                runs.append(run)
            transcript.append({"role": "tool", "tool_call_id": call["id"], "content": outcome})
        steps.append((now, runs))
        now += max((TOOLBOX[name][1] for name, _ in runs), default=0.0)
    return transcript, model.counters.calls, steps, now


def show(messages):
    lines = []
    for i, m in enumerate(messages):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = ", ".join(
                f"{c['id']}: {c['function']['name']}({c['function']['arguments']})"
                for c in m["tool_calls"]
            )
            lines.append(f"  [{i}] assistant requests {calls}")
        elif isinstance(m, dict) and m.get("role") == "tool":
            lines.append(f"  [{i}] tool {m.get('tool_call_id')!r}: {m.get('content')!r}")
        elif isinstance(m, dict):
            lines.append(f"  [{i}] {m.get('role')}: {m.get('content')!r}")
        else:
            lines.append(f"  [{i}] {m!r}")
    return "\n".join(lines) + "\n"


def play(ctx, script, messages, max_steps, *, names=None):
    names = list(TOOLBOX) if names is None else names
    want, want_calls, want_steps, want_elapsed = prescribed(script, names, messages, max_steps)

    sim = ctx.simclock()
    model = Model(script, delay_s=MODEL_S)
    tools = make_tools(names)
    given = copy.deepcopy(messages)
    guard(ctx)  # the solution's code runs from here to the end of `run_until_complete`
    got = sim.run_until_complete(ctx.solution.run_agent(model, tools, given, max_steps))
    check_guard()
    elapsed = sim.now()

    check.that(
        isinstance(got, list) and all(isinstance(m, dict) for m in got),
        "run_agent: must return the transcript, a list of message dicts",
        f"got: {type(got).__name__}",
    )
    check.that(
        given == messages and got is not given,
        "run_agent: the `messages` list it was given was modified (or returned itself)",
        f"messages before:\n{show(messages)}messages after:\n{show(given)}",
    )
    if model.violations:
        first = model.violations[0]
        check.that(
            False,
            f"the model rejected a conversation it was given ({first.code})",
            f"{first}\nthe transcript returned:\n{show(got)}",
        )

    both = f"expected transcript:\n{show(want)}got:\n{show(got)}"
    for i, (g, w) in enumerate(zip(got, want, strict=False)):
        if g != w:
            role = g.get("role")
            check.that(
                False,
                f"message {i} of the transcript (role {role!r}) is not what the rules prescribe",
                f"expected: {w!r}\ngot:      {g!r}\n\n{both}",
            )
    check.that(
        len(got) == len(want),
        f"the transcript has {len(got)} messages, expected {len(want)}",
        both,
    )

    # What was run.
    counts = Counters("model_calls", "tool_runs")
    counts.add("model_calls", model.counters.calls)
    counts.add("tool_runs", sum(t.counters.started for t in tools.values()))
    check.cost(counts, "model_calls", exactly=want_calls, unit="calls")
    check.cost(counts, "tool_runs", exactly=sum(len(runs) for _, runs in want_steps), unit="runs")
    for name, tool in tools.items():
        want_args = [args for _, runs in want_steps for n, args in runs if n == name]
        check.that(
            [run.arguments for run in tool.runs] == want_args,
            f"tool {name!r} was not run with the arguments of its calls, in call order",
            f"expected runs: {want_args}\ngot: {[run.arguments for run in tool.runs]}",
        )

    # When. Every tool of a step starts when the model's message arrives.
    for when, runs in want_steps:
        for name in {n for n, _ in runs}:
            n_runs = sum(1 for n, _ in runs if n == name)
            started = [r.started_s for r in tools[name].runs if r.started_s == when]
            check.that(
                len(started) == n_runs,
                f"a call to {name!r} was not started at the instant the model's message "
                "arrived: the calls of a step must run concurrently",
                f"the model's message arrived at t={when:g}; {name!r} runs started at "
                f"{[r.started_s for r in tools[name].runs]}",
            )
    check.that(
        abs(elapsed - want_elapsed) <= 1e-9,
        f"the run ended at t={elapsed:g} s, expected t={want_elapsed:g} s (model latency plus "
        "the longest tool of each step)",
        f"expected t={want_elapsed:g} s\n{both}",
    )
    ctx.metric("elapsed", elapsed, unit="s")
    return got, model, tools


USER = [{"role": "user", "content": "Weather and time in Paris?"}]
EXAMPLE = [
    reply(
        tool_calls=[
            tool_call("get_weather", {"city": "Paris"}),
            tool_call("get_time", {"zone": "CET"}),
        ]
    ),
    reply(
        tool_calls=[tool_call("get_weather", '{"city": '), tool_call("get_stock", {"symbol": "X"})]
    ),
    "It is 21 C in Paris at 14:05.",
]


# Rejects first_call_only, completion_order_results (get_time finishes first),
# sequential_calls (3.75 s), unknown_tool_raises, malformed_arguments_raise.
@case("the worked example: two parallel calls, two bad calls, an answer; 3.5 s", public=True)
def worked_example(ctx):
    names = ["get_weather", "get_time"]
    got, _, _ = play(ctx, EXAMPLE, USER, 8, names=names)
    check.equal(
        [m.get("content") for m in got if m["role"] == "tool"],
        [
            "21 C in Paris",
            '{"zone": "CET", "time": "14:05"}',
            '{"error": "invalid arguments"}',
            '{"error": "unknown tool: get_stock"}',
        ],
        label="contents of the tool messages",
    )


# Rejects stops_without_answering, no_step_limit, limit_off_by_one.
@case("the worked example with max_steps 1 and 2: the last step's calls are answered", public=True)
def step_limit(ctx):
    names = ["get_weather", "get_time"]
    got, _, _ = play(ctx, EXAMPLE, USER, 1, names=names)
    check.equal(len(got), 4, label="number of messages with max_steps=1")
    play(ctx, EXAMPLE, USER, 2, names=names)
    play(ctx, EXAMPLE, USER, 3, names=names)


# Rejects tool_exception_crashes and one_failure_fails_step: the failing call sits
# between two that succeed.
@case("a tool that raises becomes an error result; its siblings are unaffected", public=True)
def tool_failure(ctx):
    script = [
        reply(
            tool_calls=[
                tool_call("get_weather", {"city": "Oslo"}),
                tool_call("get_weather", {"city": "Atlantis"}),
                tool_call("add", {"a": 2, "b": 3}),
            ]
        ),
        reply("Let me try another city.", tool_calls=[tool_call("get_weather", {"city": "Lima"})]),
        "Oslo 4 C, Lima 18 C, and 2 + 3 = 5.",
    ]
    play(ctx, script, [{"role": "user", "content": "Compare the weather."}], 5)


# Rejects encodes_strings and mutates_messages.
@case(
    "no tool calls at all; a system prompt; results that are strings, numbers, and lists",
    public=True,
)
def plain_and_typed_results(ctx):
    play(ctx, ["Hello."], [{"role": "user", "content": "Hi"}], 1)
    messages = [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Book a table and look things up."},
    ]
    script = [
        reply(
            tool_calls=[
                tool_call("book_table", {"name": 'Chez "Max"', "seats": 4}),
                tool_call("add", {"a": 1.5, "b": 2}),
                tool_call("search", {"q": "bistro"}),
            ]
        ),
        "Done.",
    ]
    play(ctx, script, messages, 4)


# Rejects numbered_ids: the ids are not call_0, call_1, ...
@case("ids chosen by the model are echoed, also when they do not follow a pattern")
def custom_ids(ctx):
    script = [
        reply(
            tool_calls=[
                tool_call("add", {"a": 1, "b": 1}, id="toolu_01A"),
                tool_call("get_time", {"zone": "UTC"}, id="x"),
            ]
        ),
        reply(tool_calls=[tool_call("get_weather", {"city": "Cairo"}, id="call_0")]),
        reply(
            tool_calls=[
                tool_call("add", {"a": 2, "b": 2}, id="call_7"),
                tool_call("add", {"a": 3, "b": 3}, id="call_3"),
            ]
        ),
        "ok",
    ]
    play(ctx, script, USER, 6)


# Rejects completion_order_results and sequential_calls: the slow tool is called first
# and the same tool twice in one step.
@case("slow and fast tools in one step, the same tool twice: results stay in call order")
def call_order(ctx):
    script = [
        reply(
            tool_calls=[
                tool_call("search", {"q": "a", "limit": 1}),
                tool_call("get_weather", {"city": "Lima"}),
                tool_call("add", {"a": 1, "b": 2}),
                tool_call("get_weather", {"city": "Oslo"}),
                tool_call("book_table", {"name": "Lo", "seats": 2}),
                tool_call("get_time", {"zone": "PET"}),
            ]
        ),
        reply(
            tool_calls=[
                tool_call("get_time", {"zone": "A"}),
                tool_call("get_time", {"zone": "B"}),
                tool_call("book_table", {"name": "N", "seats": 9}),
                tool_call("get_weather", {"city": "Cairo"}),
            ]
        ),
        "done",
    ]
    play(ctx, script, USER, 5)


# Rejects non_object_arguments, malformed_arguments_raise, unknown_tool_raises.
@case("arguments that are not JSON, JSON that is not an object, wrong argument names")
def bad_arguments(ctx):
    script = [
        reply(
            tool_calls=[
                tool_call("add", "[1, 2]"),
                tool_call("add", '"a=1"'),
                tool_call("add", "null"),
                tool_call("add", "{'a': 1, 'b': 2}"),
                tool_call("add", ""),
                tool_call("add", {"a": 1, "b": 2}),
            ]
        ),
        reply(
            tool_calls=[
                tool_call("add", {"x": 1, "y": 2}),
                tool_call("nope", "not json"),
                tool_call("add", {"a": 1}),
            ]
        ),
        reply(tool_calls=[tool_call("Add", {"a": 1, "b": 2})]),
        "I give up.",
    ]
    play(ctx, script, USER, 6)


# Rejects no_step_limit, limit_off_by_one, stops_without_answering.
@case("a model that never stops asking: exactly max_steps model calls")
def never_stops(ctx):
    def endless(limit):
        def script(messages, index):
            if index >= limit + 3:  # a loop without a limit must not hang the judge
                raise RuntimeError(f"the model was called {index + 1} times with max_steps={limit}")
            return [tool_call("add", {"a": index, "b": 1})]

        return script

    for max_steps in (1, 2, 7):
        play(ctx, endless(max_steps), USER, max_steps)

    def answers_at_limit(messages, index):
        if index > 6:
            raise RuntimeError(f"the model was called {index + 1} times")
        return "done" if index == 3 else [tool_call("get_time", {"zone": f"Z{index}"})]

    play(ctx, answers_at_limit, USER, 4)
    play(ctx, answers_at_limit, USER, 3)


# Rejects mutates_messages (the first transcript would grow) and anything that
# leaves the transcript invalid at the step limit.
@case("a transcript stopped by the step limit is continued by a second run")
def resume(ctx):
    script = [
        reply(tool_calls=[tool_call("get_weather", {"city": "Oslo"})]),
        reply(
            tool_calls=[
                tool_call("get_weather", {"city": "Paris"}),
                tool_call("get_time", {"zone": "CET"}),
            ]
        ),
        "Oslo is colder.",
    ]
    first, _, _ = play(ctx, script[:2], USER, 2)
    check.that(first[-1]["role"] == "tool", "the first run should end with tool messages")
    play(ctx, script[2:], first, 3)


def random_script(rng, steps):
    names = [*TOOLBOX, "missing"]
    cities = [*WEATHER, "Nowhere"]
    out = []
    for step in range(steps):
        calls = []
        for _ in range(int(rng.integers(1, 6))):
            name = names[int(rng.integers(len(names)))]
            arguments = {
                "get_weather": {"city": cities[int(rng.integers(len(cities)))]},
                "get_time": {"zone": f"Z{int(rng.integers(9))}"},
                "add": {"a": int(rng.integers(9)), "b": int(rng.integers(9))},
                "search": {"q": f"q{int(rng.integers(9))}", "limit": int(rng.integers(0, 3))},
                "book_table": {"name": "R", "seats": int(rng.integers(1, 10))},
                "missing": {},
            }[name]
            roll = rng.random()
            if roll < 0.1:
                arguments = json.dumps(arguments)[:-1]
            elif roll < 0.15:
                arguments = "[]"
            call_id = None if rng.random() < 0.5 else f"id-{step}-{len(calls)}-{rng.integers(99)}"
            calls.append(tool_call(name, arguments, id=call_id))
        out.append(reply(tool_calls=calls))
    return out


@case("random conversations: 1 to 5 calls per step, good and bad, with and without a limit")
def random_conversations(ctx):
    rng = ctx.rng
    for _ in range(12):
        steps = int(rng.integers(0, 6))
        script = [*random_script(rng, steps), "final answer"]
        max_steps = int(rng.integers(1, steps + 3))
        play(ctx, script, USER, max_steps)
