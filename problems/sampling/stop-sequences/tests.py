"""Tests for stop-sequences.

A stream is driven the way a server drives it: `stop_step` once per piece, starting
from state None, until it reports a stop; `stop_flush` only if the stream ends
without one. Every case checks the text emitted at **each step**, against the
definition written out below (`expected_emits`), which works character by character
on the whole text and shares no code with the reference.

The property case states the two things the function is for: whatever the chunking,
the emitted pieces concatenate to the text cut before the first completed stop string,
and nothing is held back unless it could still become a stop string.
"""

import numpy as np

from whetstone.judge import case, check
from whetstone.kit import ToyVocab


def first_stop(text, stops, checked=0):
    """(start, end) of the stop match that completes first, or None.

    Among matches that end at the same character, the one that starts first.
    `checked` is a length of `text` already known to contain no complete match.
    """
    for end in range(checked + 1, len(text) + 1):
        starts = [end - len(s) for s in stops if len(s) <= end and text[end - len(s) : end] == s]
        if starts:
            return min(starts), end
    return None


def held_length(text, stops):
    """Length of the longest suffix of `text` that is a proper prefix of a stop string."""
    for n in range(min(len(text), max(map(len, stops), default=1) - 1), 0, -1):
        if any(len(s) > n and s[:n] == text[len(text) - n :] for s in stops):
            return n
    return 0


def expected_emits(pieces, stops):
    """The definition: per-step emits, the stop flag, and the flushed text (or None)."""
    text, shown, emits = "", 0, []
    for piece in pieces:
        checked = len(text)
        text += piece
        match = first_stop(text, stops, checked)
        if match is not None:
            emits.append(text[shown : match[0]])
            return emits, True, None
        visible = len(text) - held_length(text, stops)
        emits.append(text[shown:visible])
        shown = visible
    return emits, False, text[shown:]


def run_stream(ctx, pieces, stops):
    """Drive the solution over the stream. Returns (emits, stopped, flushed or None)."""
    state, emits = None, []
    for piece in pieces:
        result = ctx.call("stop_step", state, piece, list(stops))
        check.that(
            isinstance(result, tuple)
            and len(result) == 3
            and type(result[1]) is str
            and type(result[2]) is bool,
            "stop_step: must return a tuple (state, emit, stopped) with a str and a bool",
            f"got {result!r}",
        )
        state, emit, stopped = result
        emits.append(emit)
        if stopped:
            return emits, True, None
    flushed = ctx.call("stop_flush", state)
    check.that(type(flushed) is str, "stop_flush: must return a str", f"got {flushed!r}")
    return emits, False, flushed


def check_stream(ctx, pieces, stops):
    pieces = list(pieces)
    want_emits, want_stopped, want_flushed = expected_emits(pieces, stops)
    emits, stopped, flushed = run_stream(ctx, pieces, stops)
    where = (
        f"stops = {list(stops)!r}\npieces = {pieces!r}\n"
        f"expected emits = {want_emits!r}, stopped = {want_stopped}, flush = {want_flushed!r}\n"
        f"got      emits = {emits!r}, stopped = {stopped}, flush = {flushed!r}"
    )
    total = "".join(emits) + (flushed or "")
    want_total = "".join(want_emits) + (want_flushed or "")
    check.that(
        stopped == want_stopped and len(emits) == len(want_emits),
        "stop_step: reported a stop at the wrong step"
        if stopped or want_stopped
        else "stop_step: wrong number of steps",
        where,
    )
    check.that(
        total == want_total,
        "the emitted text, concatenated, is not the text cut before the first stop string",
        where,
    )
    check.that(
        emits == want_emits,
        "stop_step: the right text overall, but emitted at the wrong step (text must be "
        "released as soon as it can no longer be part of a stop string, and not before)",
        where,
    )
    check.that(flushed == want_flushed, "stop_flush: wrong text", where)


def chunkings(rng, text, n):
    """The whole text, single characters, and `n` random splits (some with empty pieces)."""
    yield [text]
    yield list(text)
    for _ in range(n):
        cuts = sorted(rng.integers(0, len(text) + 1, size=rng.integers(1, 7)).tolist())
        yield [text[a:b] for a, b in zip([0, *cuts], [*cuts, len(text)], strict=True)]


EXAMPLE_STOPS = ["\nUser:", "###"]


@case("the worked example: a stop string that arrives in two pieces", public=True)
def worked_example(ctx):
    pieces = ["Sure", ".\n", "Us", "e #", "#1", ".\nUser", ":", " next"]
    emits, stopped, _ = run_stream(ctx, pieces, EXAMPLE_STOPS)
    check.equal(emits, ["Sure", ".", "", "\nUse ", "##1", ".", ""], label="emitted pieces")
    check.that(stopped, "stop_step: did not report the stop")


@case("the stream ends while text is held back: stop_flush releases it", public=True)
def flush_releases(ctx):
    emits, stopped, flushed = run_stream(ctx, ["Sure", ".\nUser"], EXAMPLE_STOPS)
    check.equal(emits, ["Sure", "."], label="emitted pieces")
    check.that(not stopped, "stop_step: reported a stop that is not in the text")
    check.equal(flushed, "\nUser", label="stop_flush")
    check_stream(ctx, ["no stop here", "##"], EXAMPLE_STOPS)
    check_stream(ctx, [], EXAMPLE_STOPS)


@case("a stop inside one piece: the text after it is dropped, the text before it kept", public=True)
def stop_inside_piece(ctx):
    check_stream(ctx, ["one ### two ### three"], EXAMPLE_STOPS)
    check_stream(ctx, ["abc", "###"], EXAMPLE_STOPS)
    check_stream(ctx, ["###", "abc"], EXAMPLE_STOPS)
    check_stream(ctx, ["a\nUs", "er:###"], EXAMPLE_STOPS)


@case("a stop string spread over three or more pieces, one character at a time", public=True)
def character_stream(ctx):
    check_stream(ctx, list("ok\nUser: hi"), EXAMPLE_STOPS)
    check_stream(ctx, list("a#b##c###d"), EXAMPLE_STOPS)
    check_stream(ctx, ["a", "", "#", "", "#", "", "#"], EXAMPLE_STOPS)


@case("two stops in one piece: the one that completes first wins, whatever the list order")
def two_stops_one_piece(ctx):
    # Catches taking the first stop string of the list that occurs anywhere.
    for stops in (["END", "\n"], ["\n", "END"]):
        check_stream(ctx, ["line one\nline two END"], stops)
        check_stream(ctx, ["an END, then\n"], stops)
        check_stream(ctx, ["an E", "ND, then\n"], stops)


@case("a failed partial match that overlaps the next one: stop 'aab' in 'aaab'")
def overlapping_restart(ctx):
    # After "aa" + "a" the held text is no longer a prefix of "aab", but its last two
    # characters still are. Catches releasing all of it.
    check_stream(ctx, ["a", "a", "a", "b", "c"], ["aab"])
    check_stream(ctx, ["ab", "ab", "ac"], ["abac"])
    check_stream(ctx, ["x-", "-", "-", ">", "y"], ["-->"])
    check_stream(ctx, ["abab", "abc"], ["ababc", "bx"])


@case("ordinary text is not delayed: only a possible beginning of a stop is held back")
def no_needless_delay(ctx):
    # Catches holding back a fixed window of max(len(stop)) - 1 characters.
    check_stream(ctx, ["Hello", ", ", "world", "!"], ["<|end_of_turn|>"])
    check_stream(ctx, ["a < b", " and <", "| is held", " <|en"], ["<|end_of_turn|>", "\n\n\n"])


@case("a short stop inside a longer one: every chunking gives the same text")
def nested_stops(ctx):
    # In "xabcd" the stop "bc" completes before "abcd" does: the output is "xa", whether
    # the text arrives at once or in pieces. Catches preferring the match that starts
    # first among those visible in the buffer.
    for stops in (["abcd", "bc"], ["bc", "abcd"]):
        for pieces in (["xabcd"], ["xab", "cd"], ["x", "a", "b", "c", "d"], ["xabc", "d"]):
            check_stream(ctx, pieces, stops)
    # Two stops that end at the same character: the cut is before the longer one.
    for pieces in (["12abc"], ["12a", "bc"], ["12", "ab", "c"]):
        check_stream(ctx, pieces, ["bc", "abc"])
        check_stream(ctx, pieces, ["abc", "bc"])


@case("no stop strings, one-character stops, and stops that are prefixes of each other")
def degenerate_stops(ctx):
    check_stream(ctx, ["anything", " at all"], [])
    check_stream(ctx, ["no newline", " yet\nmore"], ["\n"])
    check_stream(ctx, ["sto", "p it"], ["stop", "st", "stopper"])
    check_stream(ctx, ["stopp", "ed"], ["stopper", "stopped"])
    check_stream(ctx, ["stoppe"], ["stopper", "stopped"])
    check_stream(ctx, ["é€", "😀 fi", "n."], ["😀 fin", "€€"])


@case("pieces from a tokenizer: a stop string that ends in the middle of a token")
def tokenizer_pieces(ctx):
    vocab = ToyVocab(size=64, seed=0)
    text = '{"a": true, "e": false}, {"u": null}'
    pieces = [vocab.bytes(token).decode() for token in vocab.encode(text)]
    if "".join(pieces) != text or '{"' not in pieces or "true" not in pieces:
        raise RuntimeError("authoring error: the token pieces changed")
    # "}, {" needs the tokens "}", ", ", and the first character of the token '{"'.
    check_stream(ctx, pieces, ["}, {"])
    check_stream(ctx, pieces, ['e, "', "null"])
    check_stream(ctx, pieces, ['"u": nul!', "}}"])


@case("every chunking of a text gives the same output: the text cut before the first stop")
def chunking_invariance(ctx):
    alphabets = ["ab", "abc", "a\n#"]

    def prop(rng):
        alphabet = alphabets[int(rng.integers(len(alphabets)))]

        def word(low, high):
            return "".join(rng.choice(list(alphabet), size=rng.integers(low, high)).tolist())

        stops = [word(1, 5) for _ in range(rng.integers(1, 4))]
        text = word(0, 25)
        match = first_stop(text, stops)
        truncated = text if match is None else text[: match[0]]
        for pieces in chunkings(rng, text, 4):
            emits, stopped, flushed = run_stream(ctx, pieces, stops)
            where = f"stops = {stops!r}\npieces = {pieces!r}\nemits = {emits!r}, flush = {flushed!r}"
            check.that(
                "".join(emits) + (flushed or "") == truncated,
                "the emitted text, concatenated, is not the text cut before the first stop string",
                where,
            )
            check.that(
                stopped == (match is not None),
                "stop_step: the stopped flag is wrong at the end of the stream",
                where,
            )
            check_stream(ctx, pieces, stops)

    check.forall(prop, trials=150)


@case("long streams: 2000 characters without a stop, then one at the very end")
def long_stream(ctx):
    rng = np.random.default_rng(ctx.seed)
    text = "".join(rng.choice(list("abc<|"), size=2000).tolist()).replace("<|c", "<|a")
    check_stream(ctx, list(text), ["<|c"])
    check_stream(ctx, [*list(text), "<", "|", "c", "tail"], ["<|c"])
