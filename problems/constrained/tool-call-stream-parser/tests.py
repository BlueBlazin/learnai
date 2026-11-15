"""Tests for tool-call-stream-parser.

A stream is written here as a script: a list of content strings and tool calls. From
the script the tests build the text and, independently of any parser, the merged
event list that every chunking must produce. The exact events of each step (what is
held back, what is released when) are compared with the oracle reference, and the
worked example pins those down by hand.

States are opaque and must not be modified by `parse_step`, so the functions are
called directly on the solution module: `ctx.call` would copy the state on every call
and hide a solution that mutates it.
"""

import json

from whetstone.judge import case, check

OPEN = "<tool_call>"
CLOSE = "</tool_call>"


class Call:
    """One tool call of a script: its name, its arguments as written, and its spacing."""

    def __init__(self, name, args, style="spaced", name_literal=None):
        self.name = name
        self.args = args
        self.name_literal = name_literal if name_literal is not None else json.dumps(name)
        self.style = style
        if json.loads(self.name_literal) != name or not isinstance(json.loads(args), dict):
            raise RuntimeError("authoring error: a script call is not well formed")

    def text(self):
        name, args = self.name_literal, self.args
        if self.style == "compact":
            return f'{OPEN}{{"name":{name},"arguments":{args}}}{CLOSE}'
        if self.style == "lines":  # the layout of the Hermes chat template
            return f'{OPEN}\n{{"name": {name}, "arguments": {args}}}\n{CLOSE}'
        if self.style == "airy":
            return f'{OPEN} \t{{\n  "name" : {name} ,\r\n  "arguments" :\t{args}\n}} \n{CLOSE}'
        return f'{OPEN}{{"name": {name}, "arguments": {args}}}{CLOSE}'


def stream_text(script):
    return "".join(part.text() if isinstance(part, Call) else part for part in script)


def merged_events(script):
    """What the whole stream means, straight from the script."""
    events, index = [], 0
    for part in script:
        if isinstance(part, Call):
            events += [
                ("tool_call_start", index, part.name),
                ("arguments", index, part.args),
                ("tool_call_end", index),
            ]
            index += 1
        elif part:
            if events and events[-1][0] == "content":
                events[-1] = ("content", events[-1][1] + part)
            else:
                events.append(("content", part))
    return events


def merge(events):
    """Join adjacent content events, and adjacent arguments events of one call."""
    out = []
    for event in events:
        if (
            out
            and event[0] in ("content", "arguments")
            and out[-1][0] == event[0]
            and out[-1][1:-1] == event[1:-1]
        ):
            out[-1] = (*event[:-1], out[-1][-1] + event[-1])
        else:
            out.append(event)
    return out


def check_shape(events, where):
    check.that(isinstance(events, list), f"{where}: the events must be a list", f"got {events!r}")
    for event in events:
        ok = isinstance(event, tuple) and len(event) >= 2 and isinstance(event[0], str)
        if ok and event[0] == "content":
            ok = len(event) == 2 and isinstance(event[1], str) and event[1] != ""
        elif ok and event[0] in ("tool_call_start", "arguments"):
            ok = len(event) == 3 and type(event[1]) is int and isinstance(event[2], str)
            ok = ok and (event[0] == "tool_call_start" or event[2] != "")
        elif ok and event[0] == "tool_call_end":
            ok = len(event) == 2 and type(event[1]) is int
        else:
            ok = False
        check.that(
            ok,
            f"{where}: an event does not have one of the four specified forms",
            f"the event: {event!r}\nall events of this step: {events!r}",
        )


def run(module, pieces, *, end=True):
    """Feed the pieces. Returns the list of per-step event lists (parse_end's last)."""
    state = module.parse_start()
    steps = []
    for piece in pieces:
        result = module.parse_step(state, piece)
        check.that(
            isinstance(result, tuple) and len(result) == 2,
            "parse_step: must return a tuple (state, events)",
            f"got {result!r}",
        )
        state, events = result
        check_shape(events, "parse_step")
        steps.append(events)
    if end:
        events = module.parse_end(state)
        check_shape(events, "parse_end")
        steps.append(events)
    return steps


def show_steps(pieces, steps, expected=None):
    lines = []
    for i, events in enumerate(steps):
        piece = repr(pieces[i]) if i < len(pieces) else "(parse_end)"
        lines.append(f"  step {i + 1}, piece {piece}:")
        lines.append(f"    got      {events!r}")
        if expected is not None:
            lines.append(f"    expected {expected[i]!r}")
    return "\n".join(lines)


def check_chunking(ctx, text, pieces, expected_merged=None):
    """Every step's events equal the oracle's; the merged result is the stream's meaning."""
    if "".join(pieces) != text:
        raise RuntimeError("authoring error: the pieces do not spell the text")
    got = run(ctx.solution, pieces)
    want = run(ctx.oracle, pieces)
    flat_got = [event for events in got for event in events]
    flat_want = [event for events in want for event in events]
    if expected_merged is not None and merge(flat_want) != expected_merged:
        raise RuntimeError("authoring error: the oracle disagrees with the script")
    detail = f"text: {text!r}\npieces: {pieces!r}\n"
    if merge(flat_got) != merge(flat_want):
        check.that(
            False,
            "the events of the whole stream, merged, are not what the text contains",
            detail + f"merged events, got:      {merge(flat_got)!r}\n"
            f"merged events, expected: {merge(flat_want)!r}\n",
        )
    for i, (mine, theirs) in enumerate(zip(got, want, strict=True)):
        if mine == theirs:
            continue
        where = "parse_end" if i == len(pieces) else f"step {i + 1}"
        emitted = sum(len(e[-1]) for e in mine if e[0] == "content")
        due = sum(len(e[-1]) for e in theirs if e[0] == "content")
        if emitted > due:
            why = "content was emitted that may still be the beginning of a tag"
        elif emitted < due:
            why = "content that is certain was not emitted"
        else:
            why = "the events of this step are not the specified ones"
        check.that(False, f"{where}: {why}", detail + show_steps(pieces, got, want))


def cut(text, points):
    points = sorted(set(points))
    return [text[a:b] for a, b in zip([0, *points], [*points, len(text)], strict=True)]


def chunkings(ctx, text, random=6):
    """A fixed family of ways to cut `text`, plus seeded random ones."""
    n = len(text)
    yield [text]
    yield list(text)
    for size in (2, 3, 7):
        yield cut(text, range(size, n, size))
    for trial in range(random):
        rng = ctx.child_rng(trial)
        count = int(rng.integers(1, max(2, n // 3)))
        pieces = cut(text, [int(p) for p in rng.integers(1, max(2, n), size=count)])
        if trial % 2:  # empty pieces must change nothing
            pieces.insert(int(rng.integers(len(pieces) + 1)), "")
        yield pieces


def check_script(ctx, script, random=6):
    text, expected = stream_text(script), merged_events(script)
    for pieces in chunkings(ctx, text, random):
        check_chunking(ctx, text, pieces, expected)
    return text


WEATHER = Call("get_weather", '{"city": "Oslo", "note": "a \\"}\\" b"}')
NOOP = Call("noop", "{}")
EXAMPLE_PIECES = [
    "Checking <",
    "b>now</b>.<tool",
    '_call>{"name": "get_wea',
    'ther", "arguments": {"city": "Os',
    'lo", "note": "a \\',
    '"}\\" b"}}</tool_call',
    '>\n<tool_call>{"name": "noop", "arguments": {}}</tool_call> Done <to',
]
EXAMPLE_EVENTS = [
    [("content", "Checking ")],
    [("content", "<b>now</b>.")],
    [],
    [("tool_call_start", 0, "get_weather"), ("arguments", 0, '{"city": "Os')],
    [("arguments", 0, 'lo", "note": "a \\')],
    [("arguments", 0, '"}\\" b"}')],
    [
        ("tool_call_end", 0),
        ("content", "\n"),
        ("tool_call_start", 1, "noop"),
        ("arguments", 1, "{}"),
        ("tool_call_end", 1),
        ("content", " Done "),
    ],
    [("content", "<to")],
]


@case("the worked example: seven pieces, two calls, and held text at the end", public=True)
def worked_example(ctx):
    script = ["Checking <b>now</b>.", WEATHER, "\n", NOOP, " Done <to"]
    if stream_text(script) != "".join(EXAMPLE_PIECES):
        raise RuntimeError("authoring error: the example's pieces do not spell its script")
    got = run(ctx.solution, EXAMPLE_PIECES)
    for i, (mine, expected) in enumerate(zip(got, EXAMPLE_EVENTS, strict=True)):
        where = "parse_end" if i == len(EXAMPLE_PIECES) else f"step {i + 1}"
        check.that(
            mine == expected,
            f"{where}: the events are not the ones in the statement's table",
            show_steps(EXAMPLE_PIECES, got, EXAMPLE_EVENTS),
        )


@case("plain text streams through at once; only a possible tag start waits", public=True)
def content_only(ctx):
    # Rejects buffers_content (content withheld) and end_drops_held (a final "<").
    for text in ["Hello, world!", "a < b and b > a", "2 <tool 3", "x <tool_call y", "ends with <",
                 "</tool_call> is not special here", "é€😀 <t", "", "<", "<tool_call"]:  # fmt: skip
        check_script(ctx, [text])


@case("the opening tag split across pieces at every position", public=True)
def tag_split_everywhere(ctx):
    # Rejects leaks_split_tag.
    script = ["Sure.", Call("search", '{"q": "fjords"}'), " ok"]
    text, expected = stream_text(script), merged_events(script)
    for point in range(1, len(text)):
        check_chunking(ctx, text, cut(text, [point]), expected)
    for first in range(5, 5 + len(OPEN) + 1):
        for second in range(first + 1, 5 + len(OPEN) + 2):
            check_chunking(ctx, text, cut(text, [first, second]), expected)


@case("argument deltas are the new characters only, exactly as written", public=True)
def argument_deltas(ctx):
    # Rejects cumulative_arguments.
    args = '{ "a" : [1, 2.5e3, {"b": null}],\n  "c": "x y",\t"d": {} }'
    for style in ("spaced", "compact", "lines", "airy"):
        check_script(ctx, [Call("f", args, style)])
        check_script(ctx, ["before ", Call("f", "{}", style), " after"])


@case("two and three calls: indices count up, text between calls is content", public=True)
def several_calls(ctx):
    # Rejects index_not_advanced.
    a, b, c = Call("first", '{"n": 1}'), Call("second", '{"n": 2}', "lines"), Call("third", "{}")
    check_script(ctx, [a, b])
    check_script(ctx, ["One: ", a, "\n", b, "\n", c, "\nDone."])
    check_script(ctx, [a, b, c], random=10)


@case("held text that is not a tag after all, followed directly by a real tag")
def restart_after_mismatch(ctx):
    # Rejects no_restart: in "<<tool_call>" the first "<" is content and the second
    # begins the tag.
    call = Call("f", '{"x": 1}')
    for before in ["<", "<<", "a<tool_<", "<tool_call<", "<t<to<too", "x <tool_cal"]:
        check_script(ctx, [before, call, "<"])


@case("strings in the arguments that contain braces, quotes, backslashes, and tags")
def strings_are_data(ctx):
    # Rejects braces_in_strings and end_tag_by_search.
    for args in [
        '{"code": "if (x) { y(); }"}',
        '{"a": "}", "b": "{", "c": "]["}',
        '{"say": "he said \\"}\\" and left"}',
        '{"path": "C:\\\\", "next": "}"}',
        '{"doc": "close with </tool_call> please", "n": 1}',
        '{"doc": "<tool_call>{\\"name\\": \\"x\\"}</tool_call>"}',
        '{"k}": {"{": ["}", {"]": "\\\\\\""}]}}',
        '{"emoji": "😀 é €", "u": "\\u007d"}',
    ]:
        check_script(ctx, ["go ", Call("run", args), " done"])


@case("a piece boundary between a backslash and the character it escapes")
def escape_split(ctx):
    # Rejects escape_not_carried: every single cut, so one falls after each backslash.
    for args in ['{"s": "a \\"}\\" b"}', '{"s": "\\\\"}', '{"s": "x\\\\\\"}"}']:
        script = [Call("f", args), "!"]
        text, expected = stream_text(script), merged_events(script)
        for point in range(1, len(text)):
            check_chunking(ctx, text, cut(text, [point]), expected)
    name = Call('a"b', "{}", name_literal='"a\\"b"')
    text = stream_text([name])
    for point in range(1, len(text)):
        check_chunking(ctx, text, cut(text, [point]), merged_events([name]))


@case("the name is the value of its JSON string")
def name_is_decoded(ctx):
    # Rejects name_not_decoded.
    for name, literal in [("get_x", '"get\\u005fx"'), ('say "hi"', '"say \\"hi\\""'),
                          ("a\\b", '"a\\\\b"'), ("café", '"caf\\u00e9"'), ("naïve/2", '"naïve\\/2"'),
                          ("arguments", '"arguments"'), ("name", '"name"')]:  # fmt: skip
        check_script(ctx, [Call(name, '{"name": "arguments"}', name_literal=literal)])


@case("an old state can be stepped again: parse_step does not modify its argument")
def states_are_values(ctx):
    # Rejects mutates_state.
    module = ctx.solution
    script = ["Hi <", Call("f", '{"a": "b"}'), " bye"]
    text = stream_text(script)
    for point in (3, 4, 9, 25, 37, len(text) - 6):
        state, _ = module.parse_step(module.parse_start(), text[:point])
        first = module.parse_step(state, text[point:])[1]
        module.parse_step(state, "<tool_call>{")  # a different continuation of the same state
        module.parse_end(state)
        again = module.parse_step(state, text[point:])[1]
        check.that(
            first == again,
            "parse_step gave different events when the same state was stepped a second time",
            f"text: {text!r}\nthe state after {text[:point]!r} was stepped with {text[point:]!r}, "
            f"then with another piece, then with {text[point:]!r} again\n"
            f"first:  {first!r}\nsecond: {again!r}\n",
        )


@case("a stream that ends early, at every character of a reply with two calls")
def truncated(ctx):
    script = ["Let me <look.", Call("a", '{"q": "x}"}', "lines"), "\n", Call("b", "{}"), " <to"]
    text = stream_text(script)
    for end in range(len(text) + 1):
        prefix = text[:end]
        for pieces in ([prefix], list(prefix), cut(prefix, range(4, end, 4))):
            check_chunking(ctx, prefix, pieces)


@case("property: every chunking of a random stream gives the same merged events")
def chunking_invariance(ctx):
    contents = ["", "ok ", "a < b", "<", "<tool", "<tool_call", "\n", "é€ 😀", "</tool_call>",
                "x<<y", "{\"name\": 1}", "done."]  # fmt: skip
    argument_sets = ["{}", '{"a": 1}', '{"s": "}{"}', '{"t": "\\"\\\\"}', '{"l": [[], {}, "]"]}',
                     '{"x": "</tool_call>"}', '{ "p" : { "q" : [ 1 , 2 ] } }']  # fmt: skip
    names = ["f", "get_weather", "a.b-c", "名前"]
    styles = ["spaced", "compact", "lines", "airy"]

    def prop(rng):
        script = []
        for _ in range(int(rng.integers(1, 5))):
            script.append(contents[int(rng.integers(len(contents)))])
            if rng.random() < 0.75:
                script.append(
                    Call(
                        names[int(rng.integers(len(names)))],
                        argument_sets[int(rng.integers(len(argument_sets)))],
                        styles[int(rng.integers(len(styles)))],
                    )
                )
        script.append(contents[int(rng.integers(len(contents)))])
        text = stream_text(script)
        if rng.random() < 0.25:
            text = text[: int(rng.integers(len(text) + 1))]
            expected = None
        else:
            expected = merged_events(script)
        whole = merge([e for events in run(ctx.oracle, [text]) for e in events])
        if expected is not None and expected != whole:
            raise RuntimeError("authoring error: the oracle disagrees with the script")
        for _ in range(4):
            count = int(rng.integers(0, max(1, len(text) // 2) + 1))
            points = [int(p) for p in rng.integers(0, len(text) + 1, size=count)]
            pieces = cut(text, [p for p in points if 0 < p < len(text)])
            got = merge([e for events in run(ctx.solution, pieces) for e in events])
            check.that(
                got == whole,
                "two chunkings of the same text gave different merged events",
                f"text: {text!r}\npieces: {pieces!r}\nmerged events, got: {got!r}\n"
                f"merged events of the text in one piece: {whole!r}\n",
            )
            check_chunking(ctx, text, pieces)

    check.forall(prop, trials=40)
