"""Tests for token-healing.

Expected results are computed here from the statement's definitions by brute force
(every suffix start k against every token; every token against the remaining bytes),
so they do not depend on the reference. Failure details show the prompt as pieces.
"""

from whetstone.judge import case, check
from whetstone.kit import ToyVocab


def expected_heal(ids, vocab):
    """The longest healable suffix, straight from the definition."""
    pieces = vocab.pieces
    for k in range(len(ids)):
        suffix = list(ids[k:])
        if any(vocab.is_special(t) for t in suffix):
            continue
        tail = b"".join(pieces[t] for t in suffix)
        if any(pieces[q].startswith(tail) and suffix != [q] for q in range(vocab.size)):
            return list(ids[:k]), tail
    return list(ids), b""


def expected_allowed(required, generated, vocab):
    remaining = required[len(generated) :]
    if not remaining:
        return list(range(vocab.size))
    return [
        t
        for t, piece in enumerate(vocab.pieces)
        if piece and (remaining.startswith(piece) or piece.startswith(remaining))
    ]


def show(vocab, ids):
    return "[" + ", ".join(f"{t}={vocab.pieces[t]!r}" for t in ids) + "]"


def check_heal(ctx, ids, vocab):
    want_ids, want_prefix = expected_heal(ids, vocab)
    got = ctx.call("heal", list(ids), vocab)
    check.that(
        isinstance(got, tuple) and len(got) == 2,
        "heal: must return a tuple (trimmed_ids, required_prefix)",
        f"got {got!r}",
    )
    trimmed, prefix = got
    check.that(
        isinstance(trimmed, list) and all(type(t) is int for t in trimmed),
        "heal: trimmed_ids must be a list of Python ints",
        f"got {trimmed!r}",
    )
    check.that(isinstance(prefix, bytes), "heal: required_prefix must be bytes", f"got {prefix!r}")
    detail = (
        f"vocabulary: {vocab!r}\nprompt: {show(vocab, ids)}\n"
        f"expected: keep {show(vocab, want_ids)}, required_prefix {want_prefix!r}\n"
        f"got:      keep {show(vocab, trimmed) if all(0 <= t < vocab.size for t in trimmed) else trimmed}"
        f", required_prefix {prefix!r}\n"
    )
    check.that(
        len(trimmed) == len(want_ids),
        f"heal: removed {len(ids) - len(trimmed)} of the prompt's tokens, "
        "which is not the longest healable suffix",
        detail,
    )
    check.that(trimmed == want_ids, "heal: trimmed_ids is not a prefix of the prompt", detail)
    check.that(
        prefix == want_prefix,
        "heal: required_prefix is not the bytes of the removed tokens",
        detail,
    )
    return trimmed, prefix


def check_allowed(ctx, required, generated, vocab):
    want = expected_allowed(required, generated, vocab)
    got = ctx.call("healing_allowed", required, generated, vocab)
    check.that(
        isinstance(got, list) and all(type(t) is int for t in got),
        "healing_allowed: must return a list of Python ints",
        f"got {got!r}",
    )
    missing = sorted(set(want) - set(got))
    extra = sorted(set(got) - set(want))
    known = [t for t in extra if 0 <= t < vocab.size]

    def some(tokens):
        head = ", ".join(f"{t}={vocab.pieces[t]!r}" for t in tokens[:12])
        return (head + (", ..." if len(tokens) > 12 else "")) or "none"

    detail = (
        f"vocabulary: {vocab!r}\nrequired_prefix: {required!r}\n"
        f"generated_bytes: {generated!r}\n"
        f"allowed but missing from your list: {some(missing)}\n"
        f"in your list but not allowed: {some(known)}\n"
    )
    lifted = len(generated) >= len(required)
    what = "after the required prefix is covered" if lifted else "while text is still missing"
    check.that(not missing and not extra, f"healing_allowed: wrong set of tokens {what}", detail)
    check.that(got == want, "healing_allowed: the ids are not in increasing order", detail)
    return got


def seam(vocab, *parts):
    """Token ids of text fragments that are encoded separately and concatenated."""
    return [t for part in parts for t in vocab.encode(part)]


def allowed_along(ctx, required, vocab):
    """healing_allowed for every way of having regenerated a prefix of `required`."""
    for done in range(len(required) + 1):
        check_allowed(ctx, required, required[:done], vocab)


@case("the worked example: 't' is taken back, '{' and '\"' are taken back together", public=True)
def worked_example(ctx):
    vocab = ToyVocab(size=64, seed=0)
    everything = list(range(64))
    check.equal(
        ctx.call("heal", [11, 7, 57, 6], vocab), ([11, 7, 57], b"t"), label="heal('{\"a\": t')"
    )
    check.equal(
        ctx.call("healing_allowed", b"t", b"", vocab), [6, 46, 49, 54],
        label="healing_allowed(b't', b'')",
    )  # fmt: skip
    check.equal(
        ctx.call("healing_allowed", b"t", b"tr", vocab), everything,
        label="healing_allowed(b't', b'tr')",
    )  # fmt: skip
    check.equal(
        ctx.call("heal", [51, 27, 48, 15, 5], vocab), ([51, 27, 48], b'{"'),
        label="heal('[1, ' + '{' + '\"')",
    )  # fmt: skip
    check.equal(
        ctx.call("healing_allowed", b'{"', b"", vocab), [11, 15],
        label="healing_allowed(b'{\"', b'')",
    )  # fmt: skip
    check.equal(
        ctx.call("healing_allowed", b'{"', b"{", vocab), [5, 13, 36, 57],
        label="healing_allowed(b'{\"', b'{')",
    )  # fmt: skip
    check.equal(
        ctx.call("heal", [11, 7, 57, 27, 21], vocab), ([11, 7, 57, 27, 21], b""),
        label="heal('{\"a\": 1}')",
    )  # fmt: skip
    check.equal(
        ctx.call("healing_allowed", b"", b"", vocab), everything,
        label="healing_allowed(b'', b'')",
    )  # fmt: skip


@case("a last token that longer tokens extend is removed; one that nothing extends stays", public=True)  # fmt: skip
def last_token(ctx):
    # Rejects always_removes_last ('}', '1', 'e' stay) and only_removed_token.
    vocab = ToyVocab(size=64, seed=0)
    for text in ['{"a": t', '{"a": tr', '{"a": tru', '{"a": true', '{"a": 1}', '{"a": nul',
                 "[1,", "[1", '{"a"', '{"a":', "false", "fals", "fa", "[true, f"]:  # fmt: skip
        _, required = check_heal(ctx, vocab.encode(text), vocab)
        check_allowed(ctx, required, b"", vocab)


@case("text that the prompt spells in several tokens and the vocabulary in one", public=True)
def several_tokens(ctx):
    # Rejects single_token_backup, shortest_suffix, and stops_at_first_failure: the
    # suffix 'r' of ('t', 'r') is not healable, but ('t', 'r') is.
    vocab = ToyVocab(size=64, seed=0)
    for parts in [("[1, ", "{", '"'), ('{"a"', ":"), ('{"a"', ":", " "), ("[", "t", "r"),
                  ("[", "t", "r", "u"), ("tr", "u", "e"), ("1, ", "n", "u", "l", "l"),
                  ("[1", ",", " "), ("f", "a", "l", "s"), ("[", "tr", "ue"), ('"', "}")]:  # fmt: skip
        check_heal(ctx, seam(vocab, *parts), vocab)


@case("the next token may stop short of the missing text, match it, or run past it", public=True)
def prefixes_and_extensions(ctx):
    # Rejects only_extensions ('t' and 'tr' while 'tru' is missing) and
    # only_removed_token ('true' while 'tru' is missing).
    vocab = ToyVocab(size=64, seed=0)
    for required in [b"tru", b"fals", b'{"', b'":', b"nul", b", ", b"t", b"null"]:
        allowed_along(ctx, required, vocab)


@case("once the generated bytes cover the required prefix, every token is allowed", public=True)
def lifted(ctx):
    # Rejects never_lifted.
    vocab = ToyVocab(size=64, seed=1)
    for required, generated in [(b"t", b"true"), (b"tru", b"true, "), (b'{"', b'{"a": 1'),
                                (b"t", b"t"), (b"fa", b"false"), (b"", b"1"), (b"nu", b"null")]:  # fmt: skip
        check_allowed(ctx, required, generated, vocab)


@case("special tokens: never removed from the prompt, never allowed while text is missing")
def specials(ctx):
    # Rejects heals_specials and specials_allowed_while_healing.
    vocab = ToyVocab(size=64, seed=2)
    brace, quote, t = vocab.id_of(b"{"), vocab.id_of(b'"'), vocab.id_of(b"t")
    for ids in [[0, 7, 4], [0, 7, 3], [1], [0], [0, 3, brace, quote], [brace, 4, quote],
                [0, brace, quote, 2], [3, t], [t, 3, t], [0, 4, 3], [brace, quote, 4, t]]:  # fmt: skip
        check_heal(ctx, ids, vocab)
    for required in (b"t", b'{"', b"fal"):
        allowed_along(ctx, required, vocab)


@case("characters split across tokens: the bytes of € and 😀")
def multibyte(ctx):
    vocab = ToyVocab(size=64, seed=0)
    for raw in [(b"\xe2",), (b"\xe2", b"\x82"), (b"\xe2\x82",), (b"\xf0", b"\x9f"), (b"\xc3",),
                (b"\xf0\x9f", b"\x98"), (b"\x98", b"\x80"), (b"\xe2\x82\xac",), (b"\xc3", b"\xa9")]:  # fmt: skip
        ids = [vocab.id_of(b"["), *(vocab.id_of(piece) for piece in raw)]
        _, required = check_heal(ctx, ids, vocab)
        allowed_along(ctx, required, vocab)


@case("generated bytes that disagree with the required prefix raise ValueError")
def disagreement(ctx):
    # Rejects no_agreement_check.
    vocab = ToyVocab(size=64, seed=0)
    for required, generated in [(b"tru", b"f"), (b"tru", b"tra"), (b'{"', b"{a"), (b"t", b"false"),
                                (b"nul", b"nu1"), (b"t", b"r")]:  # fmt: skip
        check.raises(
            ValueError, ctx.call, "healing_allowed", required, generated, vocab,
            label="healing_allowed with generated bytes that are not the removed text",
        )  # fmt: skip
    check_allowed(ctx, b"tru", b"tr", vocab)


@case("an empty prompt, one-token prompts, and prompts that are removed entirely")
def small_prompts(ctx):
    vocab = ToyVocab(size=64, seed=3)
    check_heal(ctx, [], vocab)
    for token in range(vocab.size):
        check_heal(ctx, [token], vocab)
    for parts in [("t", "r"), ("n", "u", "l"), ("{", '"'), ("tru",), ("f", "a")]:
        check_heal(ctx, seam(vocab, *parts), vocab)


@case("neither function modifies its arguments")
def arguments_untouched(ctx):
    vocab = ToyVocab(size=64, seed=0)
    ids = [51, 27, 48, 15, 5]
    ctx.solution.heal(ids, vocab)
    check.that(
        ids == [51, 27, 48, 15, 5],
        "heal: modified the prompt list it was given",
        f"the list is now {ids!r}",
    )


@case("every pair of ordinary tokens as the end of a prompt, three vocabularies")
def all_pairs(ctx):
    for size, seed in [(64, 4), (64, 5), (90, 6)]:
        vocab = ToyVocab(size=size, seed=seed)
        ordinary = [t for t in range(vocab.size) if not vocab.is_special(t)]
        # Pairs that can matter: the first token starts some longer token.
        heads = [a for a in ordinary
                 if any(p.startswith(vocab.pieces[a]) and p != vocab.pieces[a] for p in vocab.pieces)]  # fmt: skip
        lead = vocab.id_of(b"1")
        for a in heads:
            for b in ordinary:
                check_heal(ctx, [lead, a, b], vocab)


@case("random prompts stitched from fragments, healed and regenerated token by token")
def random_prompts(ctx):
    fragments = ["t", "r", "u", "e", "tr", "tru", "true", "f", "a", "l", "s", "fa", "fal", "fals",
                 "n", "nu", "nul", "null", '"', ":", " ", ",", "{", "}", '{"', '":', '": ', ", ",
                 '"}', "1", "[", "]", "é", "€", "😀"]  # fmt: skip
    for trial in range(60):
        rng = ctx.child_rng(trial)
        vocab = ToyVocab(size=int(rng.choice([64, 72, 282])), seed=int(rng.integers(50)))
        parts = [fragments[int(i)] for i in rng.integers(len(fragments), size=rng.integers(1, 9))]
        ids = seam(vocab, *parts)
        if rng.random() < 0.3:
            ids.insert(int(rng.integers(len(ids) + 1)), int(rng.integers(5)))
        _, required = check_heal(ctx, ids, vocab)
        # Regenerate: pick allowed tokens at random until the prefix is covered.
        generated = b""
        for _ in range(len(required) + 2):
            allowed = check_allowed(ctx, required, generated, vocab)
            if len(generated) >= len(required):
                break
            generated += vocab.pieces[int(allowed[int(rng.integers(len(allowed)))])]
