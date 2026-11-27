## The idea

UTF-8 is self-synchronizing because the top bits of each byte identify its role. `0xxxxxxx` is a single-byte ASCII character, `10xxxxxx` is a continuation byte, and `110xxxxx`, `1110xxxx`, and `11110xxx` start two-, three-, and four-byte characters. Deciding whether a character is unfinished at the end of a buffer therefore never requires scanning the whole buffer. A pending sequence consists of one lead byte plus at most two continuation bytes. Walking backward from the end over continuation bytes, the first non-continuation byte within three positions is the only possible lead byte.

That candidate byte determines the answer:

- **ASCII, or no non-continuation byte within three positions.** Nothing is pending.
- **A lead byte that already has enough continuation bytes.** The character is complete. Any extra continuation bytes after it are stray, invalid, and therefore also complete.
- **A lead byte with fewer bytes than its length requires.** The suffix is pending only if the bytes seen so far can still form a valid character.

The third case is where the narrower second-byte ranges matter. `E0 80` looks like two-thirds of a three-byte character, but `E0` must be followed by `A0..BF`. A decoder already rejects `E0` as soon as it sees `80`. Holding those two bytes back delays two replacement characters unnecessarily, and in a streaming server that delay can stall output until the end of the response.

Why is the returned prefix safe to decode immediately? A UTF-8 decoder with `errors="replace"` scans left to right and returns to its initial state after every valid character or error. Placing the cut directly before a lead byte guarantees that the suffix starts fresh regardless of what precedes it. Decoding the prefix now and the remainder later produces the exact same string as decoding all bytes at once, which is the property the tests verify on random byte sequences.

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

The loop returns as soon as it encounters a non-continuation byte, so it never inspects earlier bytes. For `E2 E2 82`, it stops at the second `E2` and reports two pending bytes, leaving the first `E2` in the prefix where it decodes to one U+FFFD.

## Common mistakes

- **Holding back every high byte.** Stripping all trailing bytes `>= 0x80` withholds complete multi-byte characters. `"café"` would hide its `é` until more text arrives, and a response ending in `é` would never emit it.
- **Forgetting four-byte characters.** Logic tested only on accented Latin and CJK text covers two- and three-byte sequences. Every emoji takes four bytes and is often split across tokens.
- **Trusting the bit pattern for the length.** `110xxxxx` includes `C0` and `C1`, and `11110xxx` and above include `F5..FF`. None of those bytes can start a valid character. Treating them as lead bytes holds invalid bytes back instead of emitting U+FFFD.
- **Not checking the second byte.** Treating any continuation byte as valid after a lead byte fails to reject `E0 80..9F` (overlong), `ED A0..BF` (surrogates), `F0 80..8F` (overlong), and `F4 90..BF` (above U+10FFFF).
- **Looking only at the last byte.** Checking only whether the final byte is a lead byte misses sequences like `E2 82`, which end in a continuation byte and are still unfinished.
- **Using `errors="ignore"` to measure length.** Computing `len(data.decode("utf-8", "ignore").encode())` drops the unfinished tail, but it also drops every invalid byte earlier in `data`. The resulting length does not even correspond to a valid cut position in the buffer.
- **Using `codecs` incremental decoders.** An incremental decoder works in production Python, but it is forbidden here. CPython's incremental decoder also differs in one edge case. It buffers `ED A0`, the start of an encoded surrogate, even though final decoding turns it into two U+FFFD characters.

## Complexity

Both time and memory are $O(1)$. The function examines at most three bytes regardless of buffer length, which matters because it runs on every generated token.

## In the wild

The WHATWG Encoding Standard's UTF-8 decoder, which browsers use for `TextDecoder` with `stream: true`, tracks this exact state with bytes needed, bytes seen, and lower and upper bounds for the next byte. Rust's `str::from_utf8` exposes the same distinction through `Utf8Error::error_len()`, which returns `None` for an incomplete trailing sequence and a byte count for invalid bytes.

Inference servers such as vLLM, Hugging Face TGI, and mlx-lm often use a simpler text-level heuristic. They decode recent tokens and hold back output while the decoded string ends in U+FFFD. That handles incomplete characters, but it also delays genuinely invalid bytes and real U+FFFD characters until a later token arrives. The `streaming-detokenizer` problem builds the exact byte-level version on top of this function.
