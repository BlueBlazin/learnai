## The idea

Decoding is a left-to-right process that returns to its initial state after every
character and after every error. So at any moment the byte stream splits into a part
whose decoding is settled, and at most one unfinished character at the end. Only that
tail, at most three bytes, can still change. This gives the whole design:

- **State** = the pending tail. Nothing else is needed: not the earlier tokens, not
  the text already sent.
- **Step** = append the token's bytes to the tail, cut off the new tail, decode and
  return the rest.
- **Flush** = decode the tail. It was never completed, so it is one U+FFFD.

Correctness follows from where the cut is placed. The new tail starts at a lead byte,
and a lead byte always begins a new character or a new error, whatever precedes it. So
decoding the part before the cut now gives exactly the characters that a decode of the
entire stream produces for those bytes, including the replacement characters for
invalid sequences. Summing over the steps, the returned pieces concatenate to the full
decode.

Timeliness follows from holding back *only* what can still become valid. `E2 82` can
become `€`, so it waits. `E0 80` cannot become anything, `FF` cannot start anything,
and `E2 82` followed by `a` has been abandoned: all of these are as final as ASCII,
and are returned in the same step. In a stream, "later" may mean never: if generation
stops, the client is left with whatever was sent.

Special tokens need no special case. Their bytes are empty, so the step appends
nothing, the tail is unchanged, and the returned text is `""`.

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

The state is an immutable `bytes` object, so "do not modify the state" costs nothing:
every step builds a new one, and a caller that kept an old state can continue from it.

In Python the standard library can carry the state for you:
`codecs.getincrementaldecoder("utf-8")("replace")` with `getstate()` and `setstate()`
gives the same texts (up to the surrogate corner noted in the statement). The decoder
object itself is mutable, so it must not *be* the state; its `getstate()` value can.

## Common mistakes

- **Decoding each token separately.** `vocab.bytes(t).decode(errors="replace")` turns
  every split character into two or more U+FFFD. It passes every test with ASCII text.
- **Decoding everything again at each step.** Keeping all token ids and calling
  `vocab.decode(ids)` each time, then returning the new suffix, can be made correct,
  but step $k$ costs $O(k)$: quadratic over the stream. The counter
  `bytes_decoded` makes this visible: 130,000 bytes for a 400-token stream whose
  tokens have 634.
- **Never flushing.** If the stream ends inside a character, the tail is still text:
  one U+FFFD. Returning `""` from `detok_flush` makes the streamed text shorter than
  what a non-streaming request would return for the same tokens.
- **Holding back whenever the text ends in U+FFFD.** Decode the buffer, and wait if
  the result ends in `�`. This confuses three different things: an unfinished
  character (wait), an invalid byte (do not wait: it will never get better), and a
  genuine U+FFFD character, validly encoded as `EF BF BD` (certainly do not wait).
- **`errors="ignore"`.** It hides the split-character problem by deleting it, and
  deletes every genuinely invalid byte as well, so the stream no longer matches the
  full decode.
- **A mutable state.** A `bytearray` or list that is extended in place works for one
  linear stream. As soon as the caller branches (beam search, speculative decoding
  that rolls back, a retry from a checkpoint), both branches write into the same
  buffer.
- **Resetting on special tokens.** A special token in the middle of a character is
  unusual but legal; treating it as a boundary throws away the held-back bytes.

## Complexity

Each step is $O(\text{bytes of the token})$ time, and the state is at most three bytes.
Over a stream, every token is looked up once and every byte is decoded once: linear in
the output, with `bytes_decoded` equal to the total bytes of the tokens.

## In the wild

Production detokenizers have more to deal with than bytes: SentencePiece-style
vocabularies encode spaces as a prefix marker, and "cleanup" rules change the text of
a token depending on its neighbors. The common answer, used by Hugging Face TGI and
vLLM, is to keep two offsets into the token list (`prefix_offset` and `read_offset`),
decode a window of a few recent tokens with and without the newest ones, and emit the
difference; the window keeps the cost per step bounded. For unfinished characters
both apply the heuristic from the mistakes above: if the new text ends in U+FFFD,
emit nothing and wait. mlx-lm's streaming detokenizers do the same on their unflushed
bytes. That is simple and nearly always right, but it does delay genuinely invalid
output until the next token. `llama.cpp`'s server instead inspects the trailing bytes
of the generated piece for an incomplete UTF-8 sequence, which is the approach of this
problem. The tail-of-bytes state also composes with the other things a streaming loop
holds back, such as a partial match of a stop string (`stop-sequences`).
