## The idea

UTF-8 decoding runs left to right and returns to its initial state after every valid
character and every error. At any point, the byte stream splits into a settled prefix
and at most one unfinished character at the end. Only that trailing tail, at most three
bytes long, can still change when more bytes arrive.

This gives the whole design:

- **State** holds only the pending tail. You do not need earlier tokens or previously
  emitted text.
- **Step** appends the new token's bytes to the pending tail, splits off the new tail,
  and decodes and returns everything before it.
- **Flush** decodes the pending tail. Because it was never completed, it decodes to one
  `�` (U+FFFD), or `""` if empty.

Correctness follows from where the cut is placed. The new tail begins at a valid lead
byte, and a lead byte always starts a new character or a new error regardless of what
came before it. Decoding the bytes before the cut therefore produces the exact same
characters as decoding the entire stream up to that point, including replacement
characters for any invalid sequences. Concatenating all step outputs and the flush
matches the full decode.

Timeliness follows from holding back *only* bytes that can still become a valid
character. `E2 82` can still become `€`, so it waits. By contrast, `E0 80` is overlong,
`FF` cannot start any character, and `E2 82` followed by `a` is already abandoned. All
of these are settled immediately and returned in the same step. In a live stream,
holding back settled bytes can stall output indefinitely if generation stops.

Special tokens need no special branch. Their bytes are `b""`, so appending them leaves
the tail unchanged and returns `""`.

## Reference solution (Python)

```python
def detok_step(state, token_id, vocab):
    pending = (state or b"") + vocab.bytes(token_id)
    n = _complete_prefix_len(pending)
    return pending[n:], pending[:n].decode("utf-8", errors="replace")


def detok_flush(state):
    return (state or b"").decode("utf-8", errors="replace")


def _complete_prefix_len(data):
    n = len(data)
    for back in range(1, min(3, n) + 1):
        byte = data[n - back]
        if byte & 0xC0 == 0x80:
            continue
        if 0xC2 <= byte <= 0xDF:
            need = 2
        elif 0xE0 <= byte <= 0xEF:
            need = 3
        elif 0xF0 <= byte <= 0xF4:
            need = 4
        else:
            return n
        if back >= need:
            return n
        if back >= 2:
            second = data[n - back + 1]
            low = {0xE0: 0xA0, 0xF0: 0x90}.get(byte, 0x80)
            high = {0xED: 0x9F, 0xF4: 0x8F}.get(byte, 0xBF)
            if not low <= second <= high:
                return n
        return n - back
    return n
```

Because `state` is an immutable `bytes` object, leaving old states unmodified is free.
Every step returns a new `bytes` object, so callers can safely branch or retry from any
earlier state.

Python's standard library can also track the state for you via
`codecs.getincrementaldecoder("utf-8")("replace")` with `getstate()` and `setstate()`.
It produces the same outputs except for the surrogate corner noted in the statement. The
decoder instance itself is mutable, so do not return the decoder as `state`—store and
return its `getstate()` tuple instead.

## Common mistakes

- **Decoding each token separately.** Calling `vocab.bytes(t).decode(errors="replace")`
  turns every split character into two or more `�` characters. This bug passes every
  ASCII-only test.
- **Re-decoding the full sequence at every step.** Storing all token ids, calling
  `vocab.decode(ids)` each step, and returning the new suffix can be made functionally
  correct, but step $k$ costs $O(k)$ work and makes the stream quadratic. The
  `bytes_decoded` counter catches this immediately, reaching 130,000 bytes on a
  400-token stream whose tokens contain only 634 bytes.
- **Never flushing.** If the stream ends inside an unfinished character, the trailing
  bytes still correspond to one `�`. Returning `""` from `detok_flush` makes the
  streamed output shorter than a full non-streaming decode.
- **Holding back whenever the decoded text ends in `�`.** Decoding the buffer and
  waiting whenever the result ends in `�` conflates three distinct cases: an unfinished
  character (should wait), an invalid byte (should not wait, since more bytes cannot
  fix it), and a literal `�` validly encoded as `EF BF BD` (should not wait).
- **Using `errors="ignore"`.** This hides split characters by dropping them, and also
  drops genuinely invalid bytes, so the stream no longer matches `errors="replace"`.
- **Using a mutable state.** Extending a `bytearray` or `list` in place works for a
  single linear stream, but breaks as soon as the caller branches (for beam search,
  speculative decoding rollbacks, or retries from a checkpoint) because both branches
  mutate the same buffer.
- **Resetting state on special tokens.** A special token in the middle of a character is
  rare but legal. Treating it as a boundary discards the held-back bytes.

## Complexity

Each step takes $O(\text{bytes of the token})$ time and stores at most 3 bytes of state.
Over a full stream, every token is looked up once and every byte is decoded once. Total
time is linear in the output length, and `bytes_decoded` equals the exact total bytes of
the tokens.

## In the wild

Production detokenizers handle more than raw UTF-8 bytes. SentencePiece-style
vocabularies encode spaces as a prefix marker, and cleanup rules can alter a token's
text depending on its neighbors. Hugging Face TGI and vLLM solve this by maintaining two
offsets into the token list (`prefix_offset` and `read_offset`), decoding a small
sliding window of recent tokens with and without the newest token, and emitting the
difference. Keeping the window bounded keeps per-step cost constant.

For unfinished characters, both TGI and vLLM use the `�` heuristic described above,
holding back output if the newly decoded text ends in `�`. `mlx-lm`'s streaming
detokenizers do the same on their unflushed bytes. That heuristic is simple and usually
works, but it delays genuinely invalid bytes until the next token arrives. `llama.cpp`'s
server instead inspects the trailing bytes for an incomplete UTF-8 sequence, matching
the approach in this problem. Keeping the pending bytes as explicit state also composes
cleanly with other streaming buffers, such as holding back partial matches of a stop
string (`stop-sequences`).
