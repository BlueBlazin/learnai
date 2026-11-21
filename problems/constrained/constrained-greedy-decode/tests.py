"""Tests for constrained-greedy-decode.

Every case builds a model, a vocabulary, and a constraint, and checks the solution's
tokens in three steps: against the constraint itself (replayed here on a fresh copy:
every token permitted, EOS only in an accepting state and only at the end), against
constrained greedy decoding done here without a cache, and then the cost. For `g`
returned tokens from a prompt of `P` tokens the limits are `g` forward calls and
`P + g - 1` tokens processed, and nothing at all when `g` is 0.

The model is handed over as `LM`, a `TinyLM` behind the kit's `Guarded`: a solution may
call it and read its sizes. Its counters, its weights, its seed, and `with_framework`
are refused, so the counts cannot be written and no second, uncounted model can do
the work. The constraint is not judged on a count and is handed over as it is.
"""

from learnai.judge import case, check
from learnai.kit import CharDFA, DFAConstraint, Guarded, TinyLM, ToyVocab, check_guard, guard

FN = "constrained_greedy"
PROMPT = [0, 17, 42, 5]

LIST = r"\[\d(, \d)*\]"
WORD = r"(true|false|null)"
NUMBER = r"-?\d+"
DECIMAL = r"-?\d+(\.\d+)?"
OBJECT = r'\{"[a-u]+": (true|false|null|-?\d+)\}'


class LM(Guarded, TinyLM):
    """The model the solution gets: it is called, and its sizes are read; nothing else."""

    label = "lm"
    refused = Guarded.refused | {"seed"}

    def __repr__(self):  # TinyLM's shows the seed, which would rebuild the model
        return f"TinyLM(vocab_size={self.vocab_size}, max_context={self.max_context})"


class Brackets:
    """Balanced square brackets around digits, at most `depth` deep: not a DFAConstraint.

    The state is a tuple (open brackets, whether anything has been generated), so a
    solution cannot assume states are ints, and nothing here has a `.dfa`.
    """

    def __init__(self, vocab, depth):
        self.open, self.close = vocab.id_of(b"["), vocab.id_of(b"]")
        self.digits = sorted(vocab.id_of(str(d).encode()) for d in range(10))
        self.depth = depth

    def with_framework(self, fw):
        return self

    def start(self):
        return (0, False)

    def allowed(self, state):
        level, begun = state
        if begun and level == 0:
            return []
        tokens = []
        if level < self.depth:
            tokens.append(self.open)
        if level > 0:
            tokens += [self.close, *self.digits]
        return sorted(tokens)

    def advance(self, state, token):
        if token not in self.allowed(state):
            raise ValueError(f"token {token!r} is not allowed in state {state!r}")
        level, _ = state
        return (level + (token == self.open) - (token == self.close), True)

    def is_accepting(self, state):
        return state == (0, True)


def permitted(constraint, state, eos_id):
    tokens = list(constraint.allowed(state))
    if constraint.is_accepting(state):
        tokens.append(eos_id)
    return tokens


def uncached_constrained_greedy(ctx, seed, prompt, constraint, max_new_tokens, eos_id):
    """The definition: the whole sequence through the model at every step."""
    lm = TinyLM(ctx.oracle_fw, seed=seed)
    state, sequence, generated = constraint.start(), list(prompt), []
    while len(generated) < max_new_tokens:
        choices = permitted(constraint, state, eos_id)
        if not choices:
            break
        row = ctx.oracle_fw.numpy(lm(sequence)[0])[-1]
        ranked = sorted(choices, key=lambda token: -row[token])
        if len(ranked) > 1 and row[ranked[0]] - row[ranked[1]] < 1e-3:
            raise RuntimeError("authoring error: the two best permitted logits are too close")
        generated.append(ranked[0])
        sequence.append(ranked[0])
        if ranked[0] == eos_id:
            break
        state = constraint.advance(state, ranked[0])
    return generated


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
            check.that(
                position == len(tokens) - 1, "tokens were generated after EOS", where
            )
            return
        check.that(
            token in constraint.allowed(state),
            "a generated token is not permitted by the constraint at its position",
            where,
        )
        state = constraint.advance(state, token)


def decode(ctx, seed, constraint, max_new_tokens, eos_id, *, prompt=PROMPT, expected=None,
           vocab=None):  # fmt: skip
    """Run the solution; check its tokens, then its cost. Returns the tokens."""
    reference = uncached_constrained_greedy(ctx, seed, prompt, constraint, max_new_tokens, eos_id)
    if expected is not None and expected != reference:
        raise RuntimeError(f"authoring error: the definition gives {reference}")

    def describe(tokens):
        if vocab is None:
            return ""
        return f"text so far: {b''.join(vocab.pieces[t] for t in tokens)!r}\n"

    lm = LM(ctx.fw, seed=seed)
    guard(ctx)
    got = ctx.call(FN, lm, list(prompt), constraint, max_new_tokens, eos_id)
    check_guard()
    check.that(
        isinstance(got, list) and all(type(token) is int for token in got),
        f"{FN}: must return a list of Python ints",
        f"got {got!r}",
    )
    check.that(len(got) <= max_new_tokens, "more than max_new_tokens tokens were generated")
    replay(constraint, got, eos_id, describe)
    check.equal(got, reference, label="generated tokens")
    produced = len(reference)
    check.cost(lm.counters, "forward_calls", at_most=produced)
    limit = len(prompt) + produced - 1 if produced else 0
    check.cost(lm.counters, "tokens_processed", at_most=limit, unit="tokens")
    return got


def regex(pattern, vocab_seed=1):
    vocab = ToyVocab(64, seed=vocab_seed)
    return DFAConstraint(CharDFA.from_regex(pattern), vocab), vocab


def text(vocab, tokens):
    return b"".join(vocab.pieces[token] for token in tokens).decode()


@case("the worked example: a list of digits, '[5, 1, 5]' and EOS in 9 calls", public=True)
def worked_example(ctx):
    constraint, vocab = regex(LIST)
    got = decode(ctx, 5, constraint, 12, 1, vocab=vocab,
                 expected=[51, 54, 7, 32, 39, 45, 54, 61, 1])  # fmt: skip
    if text(vocab, got) != "[5, 1, 5]":
        raise RuntimeError("authoring error: the statement says the text is '[5, 1, 5]'")


@case("one of three words: the first token decides, the rest is forced, then EOS", public=True)
def one_of_three_words(ctx):
    for seed in (3, 5, 11):
        constraint, vocab = regex(WORD)
        decode(ctx, seed, constraint, 12, 1, vocab=vocab)


@case("EOS competes with continuing: a number may stop after any digit", public=True)
def eos_is_a_choice(ctx):
    # After each digit the state is accepting and ten digits are allowed as well.
    for seed, eos_id, length in [(3, 4, 2), (7, 1, 4), (3, 1, 8), (9, 1, 12)]:
        constraint, vocab = regex(NUMBER)
        got = decode(ctx, seed, constraint, 12, eos_id, vocab=vocab)
        if len(got) != length:
            raise RuntimeError(f"authoring error: seed {seed} generates {len(got)} tokens")


@case("max_new_tokens ends generation inside an object, without EOS", public=True)
def cut_by_limit(ctx):
    constraint, vocab = regex(OBJECT)
    got = decode(ctx, 3, constraint, 12, 1, vocab=vocab)
    if len(got) != 12 or got[-1] == 1:
        raise RuntimeError("authoring error: this generation was expected to hit the limit")
    decode(ctx, 3, constraint, 1, 1, vocab=vocab)


@case("a dead end: nothing is permitted, so generation stops without EOS and without a call")
def dead_end(ctx):
    # The 64-token vocabulary has no "z". After "ta" nothing can follow; for "z+"
    # nothing can even start, and the model must not be called at all.
    constraint, vocab = regex(r"t(a|z)z")
    got = decode(ctx, 3, constraint, 12, 1, vocab=vocab)
    if text(vocab, got) != "ta":
        raise RuntimeError("authoring error: expected the dead end after 'ta'")
    constraint, vocab = regex(r"z+")
    if decode(ctx, 3, constraint, 12, 1, vocab=vocab) != []:
        raise RuntimeError("authoring error: expected an empty result")


@case("EOS as the very first token: the empty output is complete")
def eos_first(ctx):
    for seed, eos_id in [(14, 1), (6, 4)]:
        constraint, vocab = regex(r"\d*")
        if decode(ctx, seed, constraint, 6, eos_id, vocab=vocab) != [eos_id]:
            raise RuntimeError("authoring error: this model no longer starts with EOS")


@case("a constraint that is not a DFA: nested brackets with tuple states")
def other_constraint(ctx):
    vocab = ToyVocab(64, seed=2)
    for seed, depth in [(4, 2), (8, 3), (12, 1)]:
        decode(ctx, seed, Brackets(vocab, depth), 20, 1, vocab=vocab)


@case("decimals and objects on other models and another vocabulary, with eos_id = 4")
def other_models(ctx):
    for seed in (2, 6, 7, 13):
        for pattern in (DECIMAL, OBJECT, LIST):
            constraint, vocab = regex(pattern, vocab_seed=3)
            decode(ctx, seed, constraint, 16, 4, vocab=vocab, prompt=[0, 9, seed, 30, 2])


@case("60 tokens of letters: every token goes through the model once")
def long_generation(ctx):
    # The closing "z" has no token, so the letters never end. Re-feeding the prefix
    # would cost 2010 tokens instead of 63.
    constraint, vocab = regex(r"[a-u]+z")
    got = decode(ctx, 7, constraint, 60, 1, vocab=vocab)
    if len(got) != 60:
        raise RuntimeError("authoring error: expected 60 tokens")


@case("a single-token prompt, and a long prompt that goes in as one call")
def prompt_lengths(ctx):
    constraint, vocab = regex(WORD)
    decode(ctx, 9, constraint, 8, 1, vocab=vocab, prompt=[0])
    constraint, vocab = regex(LIST)
    decode(ctx, 9, constraint, 10, 1, vocab=vocab, prompt=[0, *range(20, 59)])
