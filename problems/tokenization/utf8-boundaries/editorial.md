## The idea

UTF-8 is self-synchronizing: the top bits of a byte say what it is. `0xxxxxxx` is a
whole character, `10xxxxxx` continues one, and `110xxxxx`, `1110xxxx`, `11110xxx` start
a character of two, three, or four bytes. So the question "is something unfinished at
the end?" never needs the whole buffer. A pending sequence is a lead byte plus at most
two continuation bytes; walking back from the end over continuation bytes, the first
other byte within three positions is the only candidate for its lead.

What that byte is settles the answer:

- ASCII, or no such byte within three positions: nothing is pending.
- A lead byte that already has all its continuation bytes: the character is complete.
  (If it has too many, the surplus ones are stray continuation bytes: invalid, and
  therefore complete.)
- A lead byte with fewer than it needs: pending, **if** what is there can still become
  a valid character.

The last condition is where the table's narrower rows matter. `E0 80` looks like two
thirds of a three-byte character, but `E0` must be followed by `A0..BF`, so a decoder
has already given up on `E0` when it sees `80`. Holding these two bytes back delays
two replacement characters for no reason, and in a stream "delay" can mean "until the
end of the response".

Why is the returned prefix safe to decode? A UTF-8 decoder with `errors="replace"`
reads left to right and, at every character or error, returns to its initial state.
The cut is placed right before a lead byte, which always starts a new character or a
new error, whatever precedes it. So decoding the prefix now and the rest later gives
the same text as decoding everything at once, and that is the property the tests check
on random bytes.

## Reference solution (Python)

```python
def complete_prefix_len(data):
    n = len(data)
    for back in range(1, min(3, n) + 1):
        byte = data[n - back]
        if byte & 0xC0 == 0x80:
            continue
        if back < _sequence_length(byte) and _second_byte_ok(data, n - back, back):
            return n - back
        return n
    return n


def _sequence_length(lead):
    if 0xC2 <= lead <= 0xDF:
        return 2
    if 0xE0 <= lead <= 0xEF:
        return 3
    if 0xF0 <= lead <= 0xF4:
        return 4
    return 1


def _second_byte_ok(data, start, have):
    if have < 2:
        return True
    lead, second = data[start], data[start + 1]
    if lead == 0xE0:
        return second >= 0xA0
    if lead == 0xED:
        return second <= 0x9F
    if lead == 0xF0:
        return second >= 0x90
    if lead == 0xF4:
        return second <= 0x8F
    return True
```

The loop returns at the first non-continuation byte, so it never looks past it: in
`E2 E2 82` it stops at the second `E2` and reports two pending bytes, and the first
`E2` is someone else's problem (it decodes to one U+FFFD).

## Common mistakes

- **Holding back every high byte.** Stripping all trailing bytes `>= 0x80` withholds
  characters that are complete: `"café"` would lose its `é` until more text arrives,
  and a response that ends in `é` would never show it.
- **Forgetting four-byte characters.** Code written against accented Latin and CJK
  text works for two and three bytes. Every emoji is four bytes, and most are split
  across tokens.
- **Trusting the bit pattern for the length.** `110xxxxx` covers `C0` and `C1`, and
  `11110xxx` and above covers `F5..FF`, none of which can start a character. Treating
  them as lead bytes holds an invalid byte back; if it is the last byte of the stream
  it is never shown as the U+FFFD it is.
- **Not checking the second byte.** See `E0 80` above; also `ED A0..BF` (surrogates),
  `F0 80..8F` (overlong), and `F4 90..BF` (too large).
- **Looking only at the last byte.** A lead byte at the very end is the easy case.
  `E2 82` ends in a continuation byte and is just as unfinished.
- **`errors="ignore"` and measure.** `len(data.decode("utf-8", "ignore").encode())`
  drops the unfinished tail, but it drops every invalid byte too, anywhere in the
  data, so the "prefix length" is not even a position in the buffer.
- **Using the library's incremental decoder.** It is the right tool in production
  Python, and not the exercise. It also differs in one corner: CPython's incremental
  decoder buffers `ED A0`, the start of an encoded surrogate, although the final
  decode turns it into two U+FFFD.

## Complexity

Constant time and memory: at most three bytes are examined, whatever the length of the
buffer. That matters, because the function runs once per generated token.

## In the wild

The WHATWG Encoding Standard's UTF-8 decoder, which every browser implements for
`TextDecoder` with `stream: true`, keeps exactly this state: bytes needed, bytes seen,
and a lower and upper bound for the next byte, which are the narrower rows of the
table. Rust's `str::from_utf8` reports the distinction directly: `Utf8Error::error_len()`
is `None` for an unfinished sequence at the end of the input and a length for bytes
that are invalid. Inference servers tend to use a cruder rule on text instead of bytes:
vLLM, Hugging Face TGI, and mlx-lm decode the recent tokens and hold the output back
while the decoded string ends in U+FFFD. That handles unfinished characters, but it
also holds back invalid bytes, and a genuine U+FFFD in the model's output, until a
later token arrives. The `streaming-detokenizer` problem builds the exact version on
top of this function.
