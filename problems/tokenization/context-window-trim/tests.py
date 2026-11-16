"""Tests for context-window-trim.

The solution is called directly (not through `ctx.call`, which rebuilds containers),
so that "does not modify its argument" and "returns a new list" can be checked. The
token counters are defined here: `words_counter` is the one of the statement's example,
and `TemplateCounter` imitates a ChatML template, whose count is deliberately not the
sum of per-message counts (a fixed overhead per prompt, a default system turn when the
list has no system message, and consecutive tool results sharing one turn).
"""

import copy

from learnai.judge import case, check

WORDS = ["the", "model", "reads", "prompt", "and", "writes", "next", "token"]
WORDS += ["while", "cache", "stays", "warm", "for", "every", "turn"]


def msg(role, content):
    return {"role": role, "content": content}


def words_counter(msgs):
    return 3 + sum(4 + len(m["content"].split()) for m in msgs)


class TemplateCounter:
    """Token count of a ChatML-like prompt; about one token per four characters."""

    def __init__(self, per_prompt=4, per_message=4, default_system=11):
        self.per_prompt = per_prompt  # BOS and the generation prompt
        self.per_message = per_message  # <|im_start|>, the role, newline, <|im_end|>
        self.default_system = default_system  # the turn added when there is no system message

    def __call__(self, msgs):
        n = self.per_prompt
        if not msgs or msgs[0]["role"] != "system":
            n += self.default_system
        previous = None
        for m in msgs:
            # Consecutive tool results are rendered into one turn: less overhead.
            n += 2 if m["role"] == "tool" and previous == "tool" else self.per_message
            n += -(-len(m["content"]) // 4)
            previous = m["role"]
        return n


EXAMPLE = [
    msg("system", "Be brief."),
    msg("user", "What is BPE?"),
    msg("assistant", "A subword tokenizer built from merges."),
    msg("user", "And WordPiece?"),
    msg("assistant", "Greedy longest match over a vocabulary."),
    msg("user", "Which one does BERT use?"),
]


def text(rng, low=1, high=12):
    return " ".join(str(w) for w in rng.choice(WORDS, size=int(rng.integers(low, high))))


def conversation(rng, turns, system=True, tools=0.3, open_last=0.3):
    """A random conversation: optional system message, then `turns` turns."""
    out = [msg("system", text(rng, 3, 20))] if system else []
    for t in range(turns):
        out.append(msg("user", text(rng)))
        last = t == turns - 1
        if last and rng.random() >= open_last:
            break  # the usual request: the last turn is the user's message alone
        if rng.random() < tools:
            out.append(msg("assistant", text(rng, 1, 4)))
            out.extend(msg("tool", text(rng, 1, 30)) for _ in range(int(rng.integers(1, 4))))
            if last:
                break  # called again in the middle of a tool loop
        out.append(msg("assistant", text(rng, 1, 25)))
    return out


def candidates(messages):
    """Every candidate of the statement, oldest history first (k = 0, 1, ...)."""
    head = messages[:1] if messages[0]["role"] == "system" else []
    body = messages[len(head) :]
    return [head + body[i:] for i, m in enumerate(body) if m["role"] == "user"]


def call(ctx, messages, counter, max_context, max_new_tokens):
    """Call the solution on a private copy. Returns (result or None, raised or None).

    Fails if the solution modified its argument, returned the very list it was given,
    or returned something that is not a list of messages.
    """
    given = copy.deepcopy(messages)
    result, raised = None, None
    try:
        result = ctx.solution.trim_messages(given, counter, max_context, max_new_tokens)
    except ValueError as error:
        raised = error
    check.that(
        given == messages,
        "trim_messages modified its `messages` argument",
        f"it now has {len(given)} messages:\n{given!r}",
    )
    if raised is None:
        check.that(
            isinstance(result, list) and all(isinstance(m, dict) for m in result),
            "trim_messages must return a list of message dicts",
            f"got {result!r}",
        )
        check.that(
            result is not given, "trim_messages returned the list it was given, not a new one"
        )
    return result, raised


def trim(ctx, messages, counter, max_context, max_new_tokens, want=None):
    """Compare with the oracle; `want` is a list of indices into messages, or ValueError."""
    got, raised = call(ctx, messages, counter, max_context, max_new_tokens)
    try:
        expected = ctx.oracle.trim_messages(
            copy.deepcopy(messages), counter, max_context, max_new_tokens
        )
    except ValueError:
        expected = None
    if want is ValueError:
        assert expected is None
    elif want is not None:
        assert expected == [messages[i] for i in want]
    if expected is None:
        check.that(
            raised is not None,
            "trim_messages returned a result where the statement specifies an exception",
            f"expected ValueError: the system message and the last turn do not fit\ngot {got!r}",
        )
        return None
    check.that(
        raised is None,
        "trim_messages raised ValueError although a candidate fits",
        f"expected {len(expected)} messages:\n{expected!r}",
    )
    check.equal(got, expected, label="trim_messages")
    return got


@case("the worked example: all six messages, four, two, and ValueError", public=True)
def worked_example(ctx):
    trim(ctx, EXAMPLE, words_counter, 67, 16, want=[0, 1, 2, 3, 4, 5])
    trim(ctx, EXAMPLE, words_counter, 64, 16, want=[0, 3, 4, 5])
    trim(ctx, EXAMPLE, words_counter, 40, 16, want=[0, 5])
    trim(ctx, EXAMPLE, words_counter, 33, 16, want=ValueError)


# Rejects no_reservation: every prompt here is below max_context on its own.
@case("room is reserved for the answer: max_new_tokens counts against the window", public=True)
def output_is_reserved(ctx):
    trim(ctx, EXAMPLE, words_counter, 60, 0, want=[0, 1, 2, 3, 4, 5])
    trim(ctx, EXAMPLE, words_counter, 60, 9, want=[0, 1, 2, 3, 4, 5])
    trim(ctx, EXAMPLE, words_counter, 60, 10, want=[0, 3, 4, 5])
    trim(ctx, EXAMPLE, words_counter, 60, 26, want=[0, 3, 4, 5])
    trim(ctx, EXAMPLE, words_counter, 60, 27, want=[0, 5])
    trim(ctx, EXAMPLE, words_counter, 60, 42, want=[0, 5])
    trim(ctx, EXAMPLE, words_counter, 60, 43, want=ValueError)


# Rejects drops_system and drops_newest_first.
@case("the system message is kept, and the oldest turns go first", public=True)
def system_is_kept(ctx):
    counter = TemplateCounter()
    messages = [
        msg("system", "You answer questions about tokenizers in one sentence."),
        msg("user", "first question " * 6),
        msg("assistant", "first answer " * 8),
        msg("user", "second question " * 6),
        msg("assistant", "second answer " * 8),
        msg("user", "third question " * 6),
        msg("assistant", "third answer " * 8),
        msg("user", "fourth question"),
    ]
    sizes = [counter(c) for c in candidates(messages)]
    trim(ctx, messages, counter, sizes[0] + 50, 50, want=list(range(8)))
    trim(ctx, messages, counter, sizes[1] + 50, 50, want=[0, 3, 4, 5, 6, 7])
    trim(ctx, messages, counter, sizes[2] + 50, 50, want=[0, 5, 6, 7])
    trim(ctx, messages, counter, sizes[3] + 50, 50, want=[0, 7])
    # Without a system message nothing is invented, and the rule is the same.
    trim(ctx, messages[1:], counter, counter(messages[5:]) + 50, 50, want=[4, 5, 6])


# Rejects orphans_assistant and drops_two_at_a_time.
@case("a turn with tool calls is dropped whole, never cut", public=True)
def tool_turns_are_whole(ctx):
    counter = TemplateCounter()
    messages = [
        msg("system", "Use the tools."),
        msg("user", "What is the weather in Oslo and in Rome?"),
        msg("assistant", "calling weather twice"),
        msg("tool", "Oslo: 4 degrees, rain, wind from the north west " * 3),
        msg("tool", "Rome: 19 degrees, clear"),
        msg("assistant", "Oslo is cold and wet, Rome is mild and clear."),
        msg("user", "And tomorrow?"),
        msg("assistant", "calling forecast"),
        msg("tool", "Oslo: 6 degrees. Rome: 21 degrees."),
        msg("assistant", "Slightly warmer in both."),
        msg("user", "Thanks. Which one should I visit?"),
    ]
    sizes = [counter(c) for c in candidates(messages)]
    # One token too few for everything: the whole first turn goes, five messages.
    trim(ctx, messages, counter, sizes[0] + 20 - 1, 20, want=[0, 6, 7, 8, 9, 10])
    trim(ctx, messages, counter, sizes[1] + 20, 20, want=[0, 6, 7, 8, 9, 10])
    trim(ctx, messages, counter, sizes[1] + 20 - 1, 20, want=[0, 10])
    trim(ctx, messages, counter, sizes[2] + 20, 20, want=[0, 10])


# Rejects never_raises.
@case("ValueError when the system message and the last turn alone do not fit", public=True)
def mandatory_part_too_large(ctx):
    counter = TemplateCounter()
    messages = [
        msg("system", "You are a careful assistant. " * 10),
        msg("user", "hello"),
        msg("assistant", "hello"),
        msg("user", "Summarize this document: " + "lorem ipsum " * 60),
    ]
    need = counter([messages[0], messages[3]])
    trim(ctx, messages, counter, need + 100, 100, want=[0, 3])
    trim(ctx, messages, counter, need + 99, 100, want=ValueError)
    trim(ctx, messages, counter, need, 100, want=ValueError)
    trim(ctx, messages, counter, 50, 100, want=ValueError)  # the reservation exceeds the window
    trim(ctx, [msg("user", "hi " * 40)], counter, 20, 0, want=ValueError)


# Rejects strict_inequality: equality fits; one token less does not.
@case("a prompt that fills the window exactly fits")
def exact_fit(ctx):
    counter = TemplateCounter()
    for seed_turns, reserve in ((3, 0), (5, 64), (8, 256)):
        messages = conversation(ctx.rng, seed_turns)
        for size in (counter(c) for c in candidates(messages)):
            trim(ctx, messages, counter, size + reserve, reserve)
            trim(ctx, messages, counter, max(size + reserve - 1, 1), reserve)
            trim(ctx, messages, counter, size + reserve + 1, reserve)


# Rejects sums_per_message: each per-message count pays the per-prompt overhead again.
@case("the count of a prompt is not the sum of the counts of its messages")
def count_is_not_additive(ctx):
    for counter in (TemplateCounter(per_prompt=40), TemplateCounter(default_system=60)):
        for system in (True, False):
            messages = conversation(ctx.rng, 6, system=system, tools=0.6)
            for size in (counter(c) for c in candidates(messages)):
                trim(ctx, messages, counter, size + 32, 32)


@case("the last turn is kept whole when it continues past the user message")
def open_last_turn(ctx):
    counter = TemplateCounter()
    messages = [
        msg("system", "Use the tools."),
        msg("user", "old question " * 10),
        msg("assistant", "old answer " * 10),
        msg("user", "Find the three largest files."),
        msg("assistant", "calling list_files"),
        msg("tool", "a.bin 4 GB, b.bin 2 GB, c.bin 1 GB, d.txt 3 kB"),
        msg("tool", "e.log 900 MB"),
    ]
    last = [messages[0], *messages[3:]]
    trim(ctx, messages, counter, counter(messages) + 10, 10, want=list(range(7)))
    trim(ctx, messages, counter, counter(last) + 10, 10, want=[0, 3, 4, 5, 6])
    trim(ctx, messages, counter, counter(last) + 9, 10, want=ValueError)


# Rejects mutates_input: here nothing is dropped, and the result must still be a copy.
@case("the result is a new list, and the argument is left as it was")
def new_list(ctx):
    counter = TemplateCounter()
    messages = conversation(ctx.rng, 4)
    got = trim(ctx, messages, counter, 100_000, 512)
    check.equal(got, messages, label="trim_messages")
    trim(ctx, messages, counter, counter(candidates(messages)[2]) + 512, 512)
    only_user = [msg("user", "hi")]
    check.equal(trim(ctx, only_user, counter, 100, 10), only_user, label="trim_messages")
    check.equal(trim(ctx, EXAMPLE[:2], counter, 100, 10), EXAMPLE[:2], label="trim_messages")


# What trimming means, without the oracle.
@case("the result fits, is a subsequence, keeps what it must, and drops no more than needed")
def result_properties(ctx):
    counter = TemplateCounter()

    def prop(rng):
        messages = conversation(rng, int(rng.integers(1, 9)), system=bool(rng.random() < 0.7))
        cands = candidates(messages)
        reserve = int(rng.integers(0, 200))
        low, high = counter(cands[-1]), counter(cands[0])
        max_context = int(rng.integers(max(low - 20, 1), high + 20)) + reserve
        got, raised = call(ctx, messages, counter, max_context, reserve)
        if raised is not None:
            check.that(
                low + reserve > max_context,
                "trim_messages raised ValueError although the system message and last turn fit",
            )
            return
        check.that(
            counter(got) + reserve <= max_context,
            "trim_messages: the result does not fit",
            f"count_tokens(result) = {counter(got)}, max_new_tokens = {reserve}, "
            f"max_context = {max_context}",
        )
        check.that(
            got in cands, "trim_messages: the result is not the conversation minus its oldest turns"
        )
        k = cands.index(got)
        check.that(
            k == 0 or counter(cands[k - 1]) + reserve > max_context,
            "trim_messages dropped a turn that would still have fit",
        )

    check.forall(prop, trials=60)


@case("long conversations, with and without a system message")
def long_conversations(ctx):
    counter = TemplateCounter()
    for system in (True, False):
        messages = conversation(ctx.rng, 60, system=system)[:200]
        if messages[-1]["role"] == "system":
            messages.append(msg("user", "hi"))
        sizes = [counter(c) for c in candidates(messages)]
        for k in (0, 1, 17, len(sizes) // 2, len(sizes) - 2, len(sizes) - 1):
            trim(ctx, messages, counter, sizes[k] + 128, 128)
        trim(ctx, messages, counter, sizes[-1] + 127, 128)


@case("random conversations, counters, and budgets")
def random_sweep(ctx):
    for _ in range(40):
        counter = TemplateCounter(
            per_prompt=int(ctx.rng.integers(0, 12)),
            per_message=int(ctx.rng.integers(2, 8)),
            default_system=int(ctx.rng.integers(0, 30)),
        )
        turns = int(ctx.rng.integers(1, 10))
        messages = conversation(ctx.rng, turns, system=bool(ctx.rng.random() < 0.6), tools=0.4)
        sizes = [counter(c) for c in candidates(messages)]
        reserve = int(ctx.rng.integers(0, 300))
        max_context = int(ctx.rng.integers(max(sizes[-1] - 10, 1), sizes[0] + 10)) + reserve
        trim(ctx, messages, counter, max_context, reserve)
