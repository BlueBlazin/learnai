"""Tests for jump-forward-decode.

Every case builds a model, a vocabulary, and a constraint, and checks the solution's
tokens in three steps: against the constraint itself (replayed here: every token
permitted, EOS only in an accepting state and only at the end), against constrained
greedy decoding done here without a cache and with one call per step (the
definition), and then the cost. The definition also yields the two numbers the limits
are stated in: `f`, the number of steps with two or more permitted tokens, and `L`,
the number of tokens generated before the last of them. The limits are `f` forward
calls and `P + L` tokens processed (0 when `f` is 0).
"""

from whetstone.judge import case, check
from whetstone.kit import CharDFA, DFAConstraint, TinyLM, ToyVocab

FN = "jump_forward_decode"
PROMPT = [0, 17, 42, 5]

SAFE = r'\{"safe": (true|false)\}'
WORD = r"(true|false|null)"
LIST = r"\[\d(, \d)*\]"
RECORD = r'\{"user": "(sue|stu|nat)", "rate": \d, "safe": (true|false)\}'
TALLY = r'\{"users": \d\d?, "rest": \[(true|false)(, (true|false))*\]\}'
# After the closing bracket the output is complete and exactly one token ("." ) may
# still follow: one allowed token, two permitted ones.
OPTIONAL_DOT = r"\[\d\]\.?"


class Template:
    """A fixed sequence of slots, each a set of token ids: not a DFAConstraint.

    A slot with one id is a forced token, a slot with several is a choice. The state
    is a tuple (slots filled, last token), so a solution cannot assume that states
    are ints, and nothing here has a `.dfa` or a `.vocab`.
    """

    def __init__(self, slots):
        self.slots = [sorted(slot) for slot in slots]

    def with_framework(self, fw):
        return self

    def start(self):
        return (0, None)

    def allowed(self, state):
        return list(self.slots[state[0]]) if state[0] < len(self.slots) else []

    def advance(self, state, token):
        if token not in self.allowed(state):
            raise ValueError(f"token {token!r} is not allowed in state {state!r}")
        return (state[0] + 1, token)

    def is_accepting(self, state):
        return state[0] == len(self.slots)


def permitted(constraint, state, eos_id):
    tokens = list(constraint.allowed(state))
    if constraint.is_accepting(state):
        tokens.append(eos_id)
    return tokens


def definition(ctx, seed, prompt, constraint, max_new_tokens, eos_id):
    """Constrained greedy decoding, the whole sequence through the model at every step.

    Returns (tokens, f, L): the tokens, the number of free steps, and the number of
    tokens generated before the last free step (None when there is no free step).
    """
    lm = TinyLM(ctx.oracle_fw, seed=seed)
    state, sequence, generated = constraint.start(), list(prompt), []
    free, last_free = 0, None
    while len(generated) < max_new_tokens:
        choices = permitted(constraint, state, eos_id)
        if not choices:
            break
        row = ctx.oracle_fw.numpy(lm(sequence)[0])[-1]
        ranked = sorted(choices, key=lambda token: -row[token])
        if len(ranked) > 1:
            if row[ranked[0]] - row[ranked[1]] < 1e-3:
                raise RuntimeError("authoring error: the two best permitted logits are too close")
            free, last_free = free + 1, len(generated)
        generated.append(ranked[0])
        sequence.append(ranked[0])
        if ranked[0] == eos_id:
            break
        state = constraint.advance(state, ranked[0])
    return generated, free, last_free


def replay(constraint, tokens, eos_id, describe):
    """Check `tokens` against the constraint alone, with messages that name the rule."""
    state = constraint.start()
    for position, token in enumerate(tokens):
        where = f"position {position} of {tokens}\n{describe(tokens[:position])}"
        if token == eos_id:
            check.that(
                constraint.is_accepting(state),
                "EOS was generated although the output is not complete",
                where,
            )
            check.that(position == len(tokens) - 1, "tokens were generated after EOS", where)
            return
        check.that(
            token in constraint.allowed(state),
            "a generated token is not permitted by the constraint at its position",
            where,
        )
        state = constraint.advance(state, token)


def decode(ctx, seed, constraint, max_new_tokens, eos_id, *, prompt=PROMPT, expected=None,
           vocab=None, free=None):  # fmt: skip
    """Run the solution; check its tokens, then its cost. Returns (tokens, f, L)."""
    reference, f, last_free = definition(ctx, seed, prompt, constraint, max_new_tokens, eos_id)
    if expected is not None and expected != reference:
        raise RuntimeError(f"authoring error: the definition gives {reference}")
    if free is not None and free != f:
        raise RuntimeError(f"authoring error: this generation has {f} free steps")

    def describe(tokens):
        if vocab is None:
            return ""
        return f"text so far: {b''.join(vocab.pieces[t] for t in tokens)!r}\n"

    lm = TinyLM(ctx.fw, seed=seed)
    got = ctx.call(FN, lm, list(prompt), constraint, max_new_tokens, eos_id)
    check.that(
        isinstance(got, list) and all(type(token) is int for token in got),
        f"{FN}: must return a list of Python ints",
        f"got {got!r}",
    )
    check.that(len(got) <= max_new_tokens, "more than max_new_tokens tokens were generated")
    replay(constraint, got, eos_id, describe)
    check.equal(got, reference, label="generated tokens")
    check.cost(lm.counters, "forward_calls", at_most=f)
    limit = len(prompt) + last_free if f else 0
    check.cost(lm.counters, "tokens_processed", at_most=limit, unit="tokens")
    return got, f, last_free


def regex(pattern, vocab_seed=1):
    vocab = ToyVocab(64, seed=vocab_seed)
    return DFAConstraint(CharDFA.from_regex(pattern), vocab), vocab


def text(vocab, tokens):
    return b"".join(vocab.pieces[token] for token in tokens).decode()


@case("the worked example: '{\"safe\": false}' and EOS, 12 tokens in 3 calls", public=True)
def worked_example(ctx):
    constraint, vocab = regex(SAFE)
    got, f, last_free = decode(ctx, 5, constraint, 24, 1, vocab=vocab,
                               expected=[52, 12, 46, 25, 29, 37, 43, 50, 46, 37, 40, 1])  # fmt: skip
    if text(vocab, got) != '{"safe": false}' or (f, last_free) != (3, 7):
        raise RuntimeError("authoring error: the statement's numbers no longer hold")


@case("forced text is not a forced token: the model picks how 'true' is spelled", public=True)
def tokenization_is_a_choice(ctx):
    # Rejects jumps_forced_text: at the start t, tr, tru, true (and the pieces of the
    # other words) are all permitted, and these models do not pick the longest piece.
    for seed in (3, 5, 11):
        constraint, vocab = regex(WORD)
        decode(ctx, seed, constraint, 12, 1, vocab=vocab)
    constraint, vocab = regex(SAFE)
    for seed in (2, 4, 9):
        decode(ctx, seed, constraint, 24, 1, vocab=vocab, free=3)


@case("the output begins with a forced token: it goes into the model with the prompt", public=True)
def forced_at_the_start(ctx):
    # Rejects separate_prompt_call: "[" is forced, so a call for the prompt alone
    # produces logits nobody uses.
    for seed in (5, 7, 9):
        constraint, vocab = regex(LIST)
        got, _, last_free = decode(ctx, seed, constraint, 14, 1, vocab=vocab)
        if text(vocab, got[:1]) != "[" or last_free == 0:
            raise RuntimeError("authoring error: expected a forced '[' first")


@case("max_new_tokens cuts a run of forced tokens", public=True)
def limit_inside_a_run(ctx):
    # Rejects run_ignores_limit. With the worked example's model the forced run
    # '"safe' covers generated tokens 2 to 6.
    constraint, vocab = regex(SAFE)
    for limit in (4, 1, 6, 7, 10):
        got, _, _ = decode(ctx, 5, constraint, limit, 1, vocab=vocab)
        if len(got) != limit:
            raise RuntimeError("authoring error: expected the limit to end this generation")


@case("one allowed token in an accepting state is a choice between it and EOS", public=True)
def eos_competes_with_one_token(ctx):
    # Rejects eos_not_counted. After "[d]" the only allowed token is "." and the
    # state is accepting: some of these models stop there, the others continue.
    endings = set()
    for seed in (1, 2, 4, 6, 10, 12):
        constraint, vocab = regex(OPTIONAL_DOT)
        got, _, _ = decode(ctx, seed, constraint, 12, 1, vocab=vocab)
        endings.add(text(vocab, got).endswith("."))
    if endings != {True, False}:
        raise RuntimeError(f"authoring error: expected both endings, got {endings}")


@case("an output that is forced from start to end costs no forward call")
def everything_forced(ctx):
    # s, a, f, e, then EOS: five tokens and the model is never run.
    constraint, vocab = regex(r"safe")
    got, f, _ = decode(ctx, 3, constraint, 12, 1, vocab=vocab)
    if text(vocab, got) != "safe" or got[-1] != 1 or f != 0:
        raise RuntimeError("authoring error: expected 'safe', EOS, and no free step")
    decode(ctx, 3, constraint, 3, 4, vocab=vocab)


@case("a dead end after forced tokens, and one at the very start")
def dead_end(ctx):
    # The 64-token vocabulary has no "z": "ta" is forced and then nothing can follow.
    constraint, vocab = regex(r"t(a|z)z")
    got, f, _ = decode(ctx, 3, constraint, 12, 1, vocab=vocab)
    if text(vocab, got) != "ta" or f != 0:
        raise RuntimeError("authoring error: expected the dead end after a forced 'ta'")
    constraint, vocab = regex(r"z+")
    if decode(ctx, 3, constraint, 12, 1, vocab=vocab)[0] != []:
        raise RuntimeError("authoring error: expected an empty result")
    # A free choice first, then the dead end.
    constraint, vocab = regex(r"(true|null)z")
    decode(ctx, 4, constraint, 12, 1, vocab=vocab)


@case("a record with three keys: long forced runs between a handful of choices")
def long_forced_runs(ctx):
    # Rejects forced_not_fed (the choices after a run depend on it),
    # one_call_per_forced_token, and feeds_trailing_run.
    for seed in (2, 5, 8, 13):
        for pattern in (RECORD, TALLY):
            constraint, vocab = regex(pattern)
            got, f, _ = decode(ctx, seed, constraint, 60, 1, vocab=vocab)
            if 2 * f > len(got):
                raise RuntimeError("authoring error: expected a mostly forced output")


@case("a constraint that is not a DFA: a token template with tuple states")
def other_constraint(ctx):
    # Slots with one id are forced; the template ends accepting with nothing allowed.
    slots = [[7], [9, 21, 33], [8], [8], [40, 41], [12], [5, 6, 7, 8], [30], [31], [32]]
    for seed in (4, 8, 12):
        decode(ctx, seed, Template(slots), 20, 1)
        decode(ctx, seed, Template(slots[1:6]), 4, 1)
    # A template of forced slots only.
    decode(ctx, 4, Template([[9], [8], [7]]), 20, 1)


@case("other models, another vocabulary, eos_id = 4, and other prompts")
def other_models(ctx):
    for seed in (2, 6, 7, 13):
        for pattern in (SAFE, LIST, RECORD, OPTIONAL_DOT, WORD):
            constraint, vocab = regex(pattern, vocab_seed=3)
            decode(ctx, seed, constraint, 40, 4, vocab=vocab, prompt=[0, 9, seed, 30, 2])


@case("a single-token prompt, and a long prompt that goes in with the first forced run")
def prompt_lengths(ctx):
    constraint, vocab = regex(SAFE)
    decode(ctx, 9, constraint, 24, 1, vocab=vocab, prompt=[0])
    constraint, vocab = regex(LIST)
    decode(ctx, 9, constraint, 10, 1, vocab=vocab, prompt=[0, *range(20, 59)])
    constraint, vocab = regex(RECORD)
    decode(ctx, 6, constraint, 60, 1, vocab=vocab, prompt=[0, *range(10, 40)])
