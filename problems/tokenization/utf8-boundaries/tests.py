"""Tests for utf8-boundaries.

The worked examples are compared with literal values and everything else with the
oracle. Three property cases state what the function means without the oracle: cutting
valid text, feeding a decoder in chunks, and the definition of "pending" by brute force
(try every way to complete the tail).
"""

from learnai.judge import case, check

TEXT = "naïve café: 5 €, 日本語, 😀 ok"
ALPHABET = ["a", " ", "é", "ß", "€", "語", "�", "😀", "𝄞", "\x00", "\x7f", "\u0080", "߿"]
# Bytes at the edges of every range in the UTF-8 table, for random and exhaustive cases.
EDGES = bytes.fromhex("41 7f 80 8f 90 9f a0 bf c0 c1 c2 df e0 e1 ec ed ee ef f0 f1 f3 f4 f5 ff")


def decode(data):
    return data.decode("utf-8", errors="replace")


def got_len(ctx, data, label=None):
    """Call the solution and check the type and range of what it returns."""
    label = label or f"complete_prefix_len({data!r})"
    n = ctx.call("complete_prefix_len", data)
    check.that(type(n) is int, f"{label}: the result must be an int", f"got {n!r}")
    check.that(
        len(data) - 3 <= n <= len(data) and n >= 0,
        f"{label}: the result must be between len(data) - 3 and len(data)",
        f"len(data) = {len(data)}, got {n}",
    )
    return n


def expect(ctx, data, want, label=None):
    label = label or f"complete_prefix_len({data!r})"
    check.equal(got_len(ctx, data, label), want, label=label)


def same_as_oracle(ctx, data, label=None):
    expect(ctx, data, ctx.expected("complete_prefix_len", data), label)


def is_pending(tail):
    """By brute force: can `tail` be extended to exactly one valid character?"""
    for second in (0x80, 0x8F, 0x90, 0x9F, 0xA0, 0xBF):
        for more in range(3):
            try:
                text = (tail + bytes([second]) + b"\x80" * more).decode("utf-8")
            except UnicodeDecodeError:
                continue
            if len(text) == 1:
                return True
    return False


def by_definition(data):
    for k in (1, 2, 3):
        if k <= len(data) and is_pending(data[len(data) - k :]):
            return len(data) - k
    return len(data)


@case("the worked examples from the statement", public=True)
def worked_example(ctx):
    expect(ctx, b"", 0)
    expect(ctx, b"abc", 3)
    expect(ctx, b"caf\xc3", 3)
    expect(ctx, b"caf\xc3\xa9", 5)
    expect(ctx, b"\xe2\x82", 0)
    expect(ctx, b"ok \xf0\x9f\x98", 3)
    expect(ctx, b"ok \xf0\x9f\x98\x80", 7)


@case("every cut of 'aé€😀': one-, two-, three-, and four-byte characters", public=True)
def every_cut(ctx):
    # Rejects holds_back_any_high_byte, ignores_four_byte, last_byte_only.
    data = "aé€😀".encode()  # 61 | c3 a9 | e2 82 ac | f0 9f 98 80
    for cut, want in enumerate([0, 1, 1, 3, 3, 3, 6, 6, 6, 6, 10]):
        expect(ctx, data[:cut], want)


@case("bytes that can never start a character are complete", public=True)
def invalid_leads(ctx):
    # Rejects invalid_lead_held and ignore_roundtrip.
    expect(ctx, b"a\xff", 2)
    expect(ctx, b"a\xffb", 3)
    expect(ctx, b"\xc0", 1)
    expect(ctx, b"\xc1", 1)
    expect(ctx, b"\xf5", 1)
    expect(ctx, b"ab\x80", 3)
    for lead in (0xC0, 0xC1, *range(0xF5, 0x100)):
        for tail in (b"", b"\x80", b"\xbf\x80"):
            same_as_oracle(ctx, b"x" + bytes([lead]) + tail)


@case("a second byte outside its lead's range ends the wait", public=True)
def second_byte_ranges(ctx):
    # Rejects no_second_byte_check.
    expect(ctx, b"\xe0\xa0", 0)
    expect(ctx, b"\xe0\x80", 2)
    expect(ctx, b"\xed\x9f", 0)
    expect(ctx, b"\xed\xa0", 2)
    expect(ctx, b"\xf0\x90\x80", 0)
    expect(ctx, b"\xf0\x8f\x80", 3)
    expect(ctx, b"\xf4\x8f", 0)
    expect(ctx, b"\xf4\x90", 2)


@case("all 256 single bytes and all 65536 two-byte strings")
def exhaustive_short(ctx):
    for a in range(256):
        same_as_oracle(ctx, bytes([a]))
    fn = ctx.solution.complete_prefix_len
    oracle = ctx.oracle.complete_prefix_len
    for a in range(256):
        for b in range(256):
            data = bytes([a, b])
            if fn(data) != oracle(data):
                same_as_oracle(ctx, data)  # reports the first difference


@case("three- and four-byte strings over the edges of every byte range")
def exhaustive_edges(ctx):
    fn = ctx.solution.complete_prefix_len
    oracle = ctx.oracle.complete_prefix_len
    for a in EDGES:
        for b in EDGES:
            for c in EDGES:
                for data in (bytes([a, b, c]), bytes([a, b, c, 0x80]), bytes([0xE2, a, b, c])):
                    if fn(data) != oracle(data):
                        same_as_oracle(ctx, data)


@case("an unfinished character right after junk or another unfinished character")
def lead_after_junk(ctx):
    for data in [
        b"\xe2\xe2\x82",
        b"\xf0\x9f\xf0\x9f",
        b"\xc3\xc3",
        b"\xe2\x82\xc3",
        b"\xff\xf0\x9f\x98",
        b"\x80\x80\xe2",
        b"\xf0\x9f\x98\xe2\x82",
    ]:
        same_as_oracle(ctx, data)


@case("more continuation bytes than the lead byte calls for")
def extra_continuation_bytes(ctx):
    for data in [
        b"\xc3\xa9\xa9",
        b"\xe2\x82\xac\x82",
        b"\xf0\x9f\x98\x80\x80",
        b"\x80\x80\x80\x80",
        b"a\x80\x80",
        b"\xc3\xa9\xa9\xa9\xa9",
    ]:
        same_as_oracle(ctx, data)


@case("long inputs: only the end matters")
def long_inputs(ctx):
    body = (TEXT * 2000).encode()
    for tail in [b"", b"\xe2", b"\xe2\x82", b"\xf0\x9f\x98", b"\xff", b"\xe0\x80", "é".encode()]:
        same_as_oracle(ctx, body + tail, f"48 kB of valid text, then {tail!r}")
        same_as_oracle(ctx, b"\xff\xfe" * 5000 + tail, f"10 kB of invalid bytes, then {tail!r}")


@case("property: cutting valid text gives the last character boundary at or before the cut")
def cuts_of_valid_text(ctx):
    def prop(rng):
        text = "".join(str(c) for c in rng.choice(ALPHABET, size=int(rng.integers(0, 12))))
        data = text.encode()
        boundaries = {len(text[:i].encode()) for i in range(len(text) + 1)}
        cut = int(rng.integers(0, len(data) + 1))
        want = max(b for b in boundaries if b <= cut)
        got = got_len(ctx, data[:cut])
        check.that(
            got == want,
            f"a cut of valid text: got {got}, expected {want} (the last character boundary "
            "before the cut)",
            f"text = {text!r}\ndata[:cut] = {data[:cut]!r}\nexpected {want}, got {got}\n",
        )

    check.forall(prop, trials=200)


@case("property: decoding chunk by chunk gives the same text as decoding at once")
def chunked_decoding(ctx):
    def prop(rng):
        if rng.random() < 0.5:
            text = "".join(str(c) for c in rng.choice(ALPHABET, size=int(rng.integers(0, 16))))
            data = text.encode()
        else:
            data = bytes(rng.choice(list(EDGES), size=int(rng.integers(0, 16))).tolist())
        out, pending, pos = [], b"", 0
        while pos < len(data):
            step = int(rng.integers(1, 5))
            pending += data[pos : pos + step]
            pos += step
            n = got_len(ctx, pending)
            out.append(decode(pending[:n]))
            pending = pending[n:]
        streamed = "".join(out) + decode(pending)
        check.that(
            streamed == decode(data),
            "decoding the complete prefix of each chunk does not reproduce the full decode",
            f"data = {data!r}\nexpected {decode(data)!r}\ngot      {streamed!r}\n",
        )

    check.forall(prop, trials=200)


@case("property: exactly the bytes that some continuation could complete are held back")
def matches_definition(ctx):
    def prop(rng):
        data = bytes(rng.choice(list(EDGES), size=int(rng.integers(0, 7))).tolist())
        got, want = got_len(ctx, data), by_definition(data)
        check.that(
            got == want,
            f"random bytes: got {got}, expected {want} (the pending sequence of the definition)",
            f"data = {data!r}\nexpected {want}, got {got}\n",
        )

    check.forall(prop, trials=400)
