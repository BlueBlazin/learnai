"""Tests for streaming-detokenizer.

Solutions are called directly (`ctx.solution.detok_step`), not through `ctx.call`, so
that a state object reaches the next call exactly as the solution returned it. The
expected texts come from the oracle, fed from a twin vocabulary with its own counters;
the property cases restate the contract without the oracle, with Python's full decode
and with the standard library's incremental UTF-8 decoder.

A solution gets the vocabulary as `Vocab`, a `ToyVocab` behind the kit's `Guarded` that
shares the counters of the test's own (`handed`): it may use `bytes`, `decode`,
`is_special`, the size, and the special ids. The counters, `pieces`, `id_of`, `encode`,
the seed, and `with_framework` are refused, so the count cannot be written and the
token bytes cannot be had without it.

The byte ED is left out of random token sequences: after ED, a second byte in A0..BF is
the one corner the statement leaves open (see its note on surrogates).
"""

import codecs

from learnai.judge import case, check
from learnai.kit import Guarded, ToyVocab, check_guard, guard

FFFD = "�"
TEXT = 'naïve café costs 5 €, 한국어 日本語 😀 {"a": true} '


class Vocab(Guarded, ToyVocab):
    """The vocabulary a solution gets: counted lookups, the size, the special ids."""

    label = "vocab"
    refused = Guarded.refused | {"seed", "pieces", "id_of", "encode"}

    def __repr__(self):  # ToyVocab's shows the seed, which would rebuild the table
        return f"ToyVocab(size={self.size})"


_handed = {}


def handed(vocab):
    """The guarded vocabulary for one of the test's: always the same one, sharing its counters."""
    if id(vocab) not in _handed:
        _handed[id(vocab)] = (vocab, Vocab.of(vocab))  # keeps `vocab`, and so its id, alive
    return _handed[id(vocab)][1]


def call(ctx, fn, *args):
    """Call `fn` of the solution (or the oracle) directly, with the guard armed."""
    args = [handed(a) if isinstance(a, ToyVocab) else a for a in args]
    guard(ctx)
    try:
        return fn(*args)
    finally:
        check_guard()


def stream(ctx, vocab, ids, module=None):
    """Feed `ids` one by one; return the per-step texts and the flushed text."""
    module = module or ctx.solution
    state, texts = None, []
    for step, token in enumerate(ids):
        out = call(ctx, module.detok_step, state, token, vocab)
        check.that(
            isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], str),
            "detok_step must return a tuple (state, text) with text a str",
            f"step {step}: got {out!r}",
        )
        state, text = out
        texts.append(text)
    tail = call(ctx, module.detok_flush, state)
    check.that(isinstance(tail, str), "detok_flush must return a str", f"got {tail!r}")
    return texts, tail


def same_as_oracle(ctx, vocab, ids, label):
    """Per-step texts and the flush equal the oracle's; returns the full streamed text."""
    texts, tail = stream(ctx, vocab, ids)
    want_texts, want_tail = stream(ctx, vocab.with_framework(None), ids, ctx.oracle)
    context = f"token bytes: {[vocab.pieces[t] for t in ids]!r}"
    check.equal(texts, want_texts, label=f"{label}: text per step", context=context)
    check.equal(tail, want_tail, label=f"{label}: detok_flush", context=context)
    return "".join(texts) + tail


def full_decode(vocab, ids):
    return b"".join(vocab.pieces[t] for t in ids).decode("utf-8", errors="replace")


def single_bytes(vocab, data):
    """`data` as one single-byte token per byte (every byte has a token at size 282)."""
    return [vocab.id_of(bytes([b])) for b in data]


def random_ids(rng, vocab, n, special_rate=0.05):
    """Random token ids without the byte ED, with a few special tokens mixed in."""
    pool = [i for i, piece in enumerate(vocab.pieces) if piece and b"\xed" not in piece]
    ids = [int(t) for t in rng.choice(pool, size=n)]
    return [int(rng.integers(0, 5)) if rng.random() < special_rate else t for t in ids]


@case("the worked example: 'é' in one token, '😀' in two, then a cut-off '€'", public=True)
def worked_example(ctx):
    vocab = ToyVocab(size=64, seed=0)
    pieces = [b'{"', "é".encode(), b'": ', b"\xf0\x9f", b"\x98\x80", b'"}']
    ids = [vocab.id_of(p) for p in pieces] + [vocab.eos_id]
    texts, tail = stream(ctx, vocab, ids)
    check.equal(texts, ['{"', "é", '": ', "", "😀", '"}', ""], label="text per step")
    check.equal(tail, "", label="detok_flush")

    ids = [vocab.id_of(b"5"), vocab.id_of(b" "), vocab.id_of(b"\xe2\x82")]
    texts, tail = stream(ctx, vocab, ids)
    check.equal(texts, ["5", " ", ""], label="text per step (cut-off '€')")
    check.equal(tail, FFFD, label="detok_flush (cut-off '€')")


@case("characters split byte by byte, also with a special token in between", public=True)
def split_characters(ctx):
    # Rejects per_token_decode and special_resets_state.
    vocab = ToyVocab(size=64, seed=1)
    ids = single_bytes(vocab, "€😀é".encode())
    texts, tail = stream(ctx, vocab, ids)
    check.equal(texts, ["", "", "€", "", "", "", "😀", "", "é"], label="text per step")
    check.equal(tail, "", label="detok_flush")

    e2, x82, ac = single_bytes(vocab, "€".encode())
    ids = [e2, vocab.pad_id, x82, vocab.eos_id, ac, vocab.id_of(b"1")]
    texts, tail = stream(ctx, vocab, ids)
    check.equal(texts, ["", "", "", "", "€", "1"], label="text per step (specials inside '€')")
    check.equal(tail, "", label="detok_flush (specials inside '€')")


@case("invalid bytes come out as U+FFFD at once; an unfinished tail on flush", public=True)
def invalid_bytes(ctx):
    # Rejects errors_ignore, flush_drops_tail, holds_back_replacement_char.
    vocab = ToyVocab(size=64, seed=2)
    ac, x80 = vocab.id_of(b"\xac"), vocab.id_of(b"\x80")
    e2_82, f0_9f, a = vocab.id_of(b"\xe2\x82"), vocab.id_of(b"\xf0\x9f"), vocab.id_of(b"a")
    # A continuation byte with no lead byte is invalid right away.
    texts, tail = stream(ctx, vocab, [a, ac, a, x80])
    check.equal(texts, ["a", FFFD, "a", FFFD], label="text per step (stray continuation bytes)")
    check.equal(tail, "", label="detok_flush (stray continuation bytes)")
    # 'a' ends the wait for the third byte of the first character: E2 82 is invalid now.
    texts, tail = stream(ctx, vocab, [e2_82, a, f0_9f])
    check.equal(texts, ["", FFFD + "a", ""], label="text per step (interrupted character)")
    check.equal(tail, FFFD, label="detok_flush (interrupted character)")
    # A new lead byte also ends the wait; the new character then waits itself.
    texts, tail = stream(ctx, vocab, [f0_9f, e2_82, ac])
    check.equal(texts, ["", FFFD, "€"], label="text per step (lead byte after lead byte)")
    check.equal(tail, "", label="detok_flush (lead byte after lead byte)")

    check.equal(call(ctx, ctx.solution.detok_flush, None), "", label="detok_flush(None)")


@case("cost: each token's bytes are looked up a bounded number of times", public=True)
def linear_cost(ctx):
    # Rejects full_redecode.
    vocab = ToyVocab(size=64, seed=3)
    ids = random_ids(ctx.rng, vocab, 400)
    total = sum(len(vocab.pieces[t]) for t in ids)
    same_as_oracle(ctx, vocab, ids, "400 random tokens")
    check.cost(vocab.counters, "bytes_decoded", at_most=4 * total, unit="bytes")


@case("a genuine U+FFFD character and other three-byte characters are not delayed")
def real_replacement_character(ctx):
    # Rejects holds_back_replacement_char: EF BF BD is the valid encoding of U+FFFD.
    vocab = ToyVocab(size=282, seed=4)
    for text in ["a�b", "�", "��!", "x￾￿�"]:
        same_as_oracle(ctx, vocab, single_bytes(vocab, text.encode()), f"bytes of {text!r}")
    fffd = single_bytes(vocab, FFFD.encode())
    texts, tail = stream(ctx, vocab, [*fffd, vocab.id_of(b"a")])
    check.equal(tail, "", label="detok_flush after a valid U+FFFD and 'a'")
    check.that(
        texts[2] == FFFD,
        "the last byte of a valid U+FFFD character must release it in the same step",
        f"texts per step: {texts!r}",
    )


@case("second bytes outside the lead byte's range are not waited for")
def second_byte_ranges(ctx):
    vocab = ToyVocab(size=282, seed=5)
    for data in [
        b"\xe0\x80\x80",
        b"\xe0\xa0\x80",
        b"\xf0\x80\x80\x80",
        b"\xf0\x90\x80\x80",
        b"\xf4\x90\x80\x80",
        b"\xf4\x8f\xbf\xbf",
        b"\xc0\x80",
        b"\xc1\xbf",
        b"\xf5\x80\x80\x80",
        b"\xff\xfe",
        b"\xc3\xa9\xa9\xa9",
        b"\xe2\x82\xe2\x82\xac",
    ]:
        same_as_oracle(ctx, vocab, single_bytes(vocab, data), f"bytes {data!r}")
        same_as_oracle(ctx, vocab, single_bytes(vocab, data[:-1]), f"bytes {data[:-1]!r}")


@case("a state can be continued more than once: detok_step does not modify it")
def states_are_values(ctx):
    # Rejects mutates_state. At every step a decoy token is fed from the same state first.
    vocab = ToyVocab(size=64, seed=6)
    oracle_vocab = vocab.with_framework(None)
    ids = [*single_bytes(vocab, "a😀€é1".encode()), *random_ids(ctx.rng, vocab, 60)]
    decoys = random_ids(ctx.rng, vocab, len(ids), special_rate=0.0)
    state = want_state = None
    for step, (token, decoy) in enumerate(zip(ids, decoys, strict=True)):
        call(ctx, ctx.solution.detok_step, state, decoy, vocab)  # a branch that is thrown away
        state, text = call(ctx, ctx.solution.detok_step, state, token, vocab)
        want_state, want = ctx.oracle.detok_step(want_state, token, oracle_vocab)
        check.that(
            text == want,
            "after continuing from the same state twice, a step returned the wrong text",
            f"step {step}: expected {want!r}, got {text!r}\n"
            "detok_step must not modify the state it is given\n",
        )
    flush = ctx.solution.detok_flush
    check.equal(call(ctx, flush, state), ctx.oracle.detok_flush(want_state), label="flush")
    check.equal(call(ctx, flush, state), ctx.oracle.detok_flush(want_state), label="flush")


@case("two streams advanced in turn do not disturb each other")
def interleaved_streams(ctx):
    vocab = ToyVocab(size=64, seed=7)
    ids_a = single_bytes(vocab, '😀€é{"a": true}'.encode())
    ids_b = random_ids(ctx.rng, vocab, len(ids_a))
    want_a, tail_a = stream(ctx, vocab.with_framework(None), ids_a, ctx.oracle)
    want_b, tail_b = stream(ctx, vocab.with_framework(None), ids_b, ctx.oracle)
    state_a = state_b = None
    got_a, got_b = [], []
    for a, b in zip(ids_a, ids_b, strict=True):
        state_a, text = call(ctx, ctx.solution.detok_step, state_a, a, vocab)
        got_a.append(text)
        state_b, text = call(ctx, ctx.solution.detok_step, state_b, b, vocab)
        got_b.append(text)
    check.equal(got_a, want_a, label="first stream: text per step")
    check.equal(got_b, want_b, label="second stream: text per step")
    flush = ctx.solution.detok_flush
    check.equal(call(ctx, flush, state_a), tail_a, label="first stream: detok_flush")
    check.equal(call(ctx, flush, state_b), tail_b, label="second stream: detok_flush")


@case("random token sequences from vocabularies of 64 and 282 tokens")
def random_sequences(ctx):
    for size, seed in [(64, 8), (64, 9), (100, 10), (282, 11), (282, 12)]:
        vocab = ToyVocab(size=size, seed=seed)
        for trial in range(12):
            ids = random_ids(ctx.rng, vocab, int(ctx.rng.integers(0, 40)))
            same_as_oracle(ctx, vocab, ids, f"vocabulary of {size}, sequence {trial}")


@case("property: the streamed text is the full decode, step by step as early as possible")
def equals_full_decode(ctx):
    def prop(rng):
        vocab = ToyVocab(size=(64, 282)[int(rng.integers(2))], seed=int(rng.integers(100)))
        ids = random_ids(rng, vocab, int(rng.integers(0, 60)))
        texts, tail = stream(ctx, vocab, ids)
        detail = f"token bytes: {[vocab.pieces[t] for t in ids]!r}\ntext per step: {texts!r}\n"
        check.that(
            "".join(texts) + tail == full_decode(vocab, ids),
            "the pieces do not concatenate to the decode of the whole sequence",
            detail + f"flush: {tail!r}\nfull decode: {full_decode(vocab, ids)!r}\n",
        )
        # An incremental decoder emits each character at the first moment it is certain.
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        want = [decoder.decode(vocab.pieces[t]) for t in ids]
        check.that(
            texts == want,
            "a step returned text too early or too late",
            detail + f"expected:      {want!r}\n",
        )

    check.forall(prop, trials=60)


@case("property: valid text cut into arbitrary tokens never yields U+FFFD")
def valid_text_never_replaced(ctx):
    def prop(rng):
        vocab = ToyVocab(size=282, seed=int(rng.integers(100)))
        start = int(rng.integers(0, len(TEXT)))
        text = (TEXT + TEXT)[start : start + int(rng.integers(0, 40))]
        ids = []
        for token in vocab.encode(text):
            # Break about half of the merged tokens back into single bytes.
            ids += single_bytes(vocab, vocab.pieces[token]) if rng.random() < 0.5 else [token]
        texts, tail = stream(ctx, vocab, ids)
        detail = f"text: {text!r}\ntoken bytes: {[vocab.pieces[t] for t in ids]!r}\n"
        check.that(
            all(FFFD not in piece for piece in texts) and tail == "",
            "valid text produced a U+FFFD or left something for the flush",
            detail + f"text per step: {texts!r}\nflush: {tail!r}\n",
        )
        check.that("".join(texts) == text, "the pieces do not concatenate to the text", detail)
        total = sum(len(vocab.pieces[t]) for t in ids)
        check.that(
            vocab.counters.bytes_decoded <= 4 * total,
            "too many bytes decoded for the length of the stream",
            f"{vocab.counters.bytes_decoded} bytes decoded for tokens of {total} bytes in total\n",
        )

    check.forall(prop, trials=60)


@case("cost: a long stream of multi-byte characters, one byte per token")
def linear_cost_multibyte(ctx):
    vocab = ToyVocab(size=282, seed=13)
    ids = single_bytes(vocab, (TEXT * 12).encode())
    total = len(ids)
    text = same_as_oracle(ctx, vocab, ids, "one byte per token")
    check.equal(text, TEXT * 12, label="streamed text")
    check.cost(vocab.counters, "bytes_decoded", at_most=4 * total, unit="bytes")
