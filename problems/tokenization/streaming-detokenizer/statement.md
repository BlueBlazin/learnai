A streaming server sends text to the client while the model is still generating.
Implement the detokenizer for that loop. It receives one token at a time and returns the
text that can be sent immediately. Because tokens are raw byte strings, a single UTF-8
character can span multiple tokens. Your detokenizer must hold back only an unfinished
character at the end of the stream, without ever decoding the whole sequence again.

![This problem is the Detokenize step, turning one token at a time into the text that can be sent.](assets/map.svg)

## Specification

```python
def detok_step(state: Any, token_id: int, vocab: ToyVocab) -> tuple[Any, str]: ...
def detok_flush(state: Any) -> str: ...
```

`vocab` is a [`ToyVocab`](../../../python/src/learnai/kit/vocab.py), the kit's small
token table. Calling `vocab.bytes(token_id)` returns the byte string for one token.
Special tokens (ids 0 to 4, such as `<|eos|>`) have empty bytes `b""`.

**The stream.** A stream feeds a sequence of tokens $t_1, \dots, t_K$ one per call.

```python
state = None
for token_id in tokens:
    state, text = detok_step(state, token_id, vocab)
    send(text)
send(detok_flush(state))
```

The first call passes `state = None`. Every later call passes the `state` returned by
the previous `detok_step`. You may choose any type for `state`, and the tests never
inspect its contents.

**What each step returns.** Let $B_k$ be the concatenation of the bytes from the first
$k$ tokens, and let $D$ denote UTF-8 decoding with `errors="replace"` (which replaces
invalid byte sequences with the replacement character `�`, U+FFFD). Split $B_k$ into a
*final part* $F_k$ and a *pending tail*. The pending tail is the unfinished character at
the very end of $B_k$, if one exists, and $F_k$ is everything before it.

Here, an "unfinished character" has the same definition as in `utf8-boundaries`. It is a
UTF-8 lead byte followed by fewer continuation bytes than it requires, where the bytes
present are still valid according to the UTF-8 byte table so that more bytes could
complete one valid character. A pending tail is at most three bytes long.

With this split (and empty $F_0$), $D(F_{k-1})$ is always a prefix of $D(F_k)$:

- Step $k$ returns the newly finalized text, which is $D(F_k)$ with its first
  $|D(F_{k-1})|$ characters removed.
- `detok_flush` returns $D$ applied to the pending tail of $B_K$. That is `""` when no
  bytes are pending, and `"�"` otherwise, because an unfinished character at the end of
  the stream is invalid.

Under these rules, every character is returned at the earliest step where it is settled.
The concatenation of all step outputs and the flush equals $D(B_K)$, and no step ever
emits half a character.

Your implementation must also satisfy these edge cases:

- **Invalid bytes are final immediately.** A byte that cannot start a character (such as
  a stray continuation byte or `FF`) is never held back. The same applies when an
  unfinished character is followed by a byte that cannot continue it. Both are returned
  as `�` in the step that settles them, matching $D(B_K)$. Only a tail that can still
  become a valid character waits.
- **A step can return `""`.** This happens when a token only starts or continues a
  pending character, and for every special token. Because a special token has empty
  bytes, it leaves any pending tail unchanged.
- **A step can return multiple characters.** For example, a single token can settle an
  abandoned character as `�` and also contribute its own text.
- **Flushing is idempotent.** `detok_flush(None)` returns `""`. Calling `detok_flush`
  does not mutate or invalidate `state`, so calling it twice returns the same string.
- **States are immutable values.** `detok_step` must not mutate the `state` it receives,
  and must not store stream progress in globals or attributes on `vocab`. A caller may
  branch from the same state multiple times (as in beam search) or interleave multiple
  streams.

> [!NOTE]
> One corner case is left open. After the lead byte `ED`, a second byte in `A0..BF` is
> invalid UTF-8 (it would encode a UTF-16 surrogate), so by the rule above it does not
> wait. CPython's incremental decoder does hold it back. Either behavior is accepted,
> and the tests never place a byte in `A0..BF` after `ED`.

**Cost.** `vocab` counts every byte it returns in `vocab.counters.bytes_decoded` (both
`vocab.bytes` and `vocab.decode` increment this counter). Across an entire stream,
`bytes_decoded` must not exceed **four times** the total byte length of the tokens fed.
A UTF-8 character is at most four bytes long, so no token ever needs to be looked up in
more than four steps, and looking each token up once is sufficient. Re-decoding the
entire prefix at each step costs $O(K^2)$ bytes and exceeds this limit.

## Example

With `vocab = ToyVocab(size=64, seed=0)`, consider the tokens whose bytes are `{"`, `é`
(`C3 A9`, one token), `": `, the first half of `😀` (`F0 9F`), its second half
(`98 80`), `"}`, and `<|eos|>`.

| step | token bytes | returned text | pending after the step |
|---|---|---|---|
| 1 | `7B 22` | `{"` | |
| 2 | `C3 A9` | `é` | |
| 3 | `22 3A 20` | `": ` | |
| 4 | `F0 9F` | `""` | `F0 9F` |
| 5 | `98 80` | `😀` | |
| 6 | `22 7D` | `"}` | |
| 7 | (special) | `""` | |

`detok_flush` returns `""`, and the outputs concatenate to `{"é": 😀"}`.

![Steps 3 to 6 of the table: the four bytes of `😀` arrive in two tokens, and the first two wait for the rest.](assets/stream-steps.svg)

Now consider a stream that stops mid-character. Feeding the tokens `5`, a space, and the
first two bytes of `€` (`E2 82`) returns `"5"`, `" "`, and `""`. Calling `detok_flush`
then returns `"�"`.

Invalid bytes do not wait. With the tokens `E2 82`, `a`, and `F0 9F`:

| step | token bytes | returned text | pending after the step |
|---|---|---|---|
| 1 | `E2 82` | `""` | `E2 82` |
| 2 | `61` | `"�a"` | |
| 3 | `F0 9F` | `""` | `F0 9F` |

In step 2, `a` (`61`) cannot continue `E2 82`. That settles `E2 82` as invalid, so step
2 emits one `�` followed by `a`. Finally, `detok_flush` returns `"�"` for the
unfinished `F0 9F`.

## Constraints

- Streams contain at most 5000 tokens. Vocabularies have 64 to 282 tokens with various
  seeds, so do not hard-code token ids. At size 282, every single byte value is a token
  and streams include arbitrary invalid byte sequences.
- Every step's output and the flush output are compared for exact string equality.
- `vocab.counters.bytes_decoded` must be at most four times the total bytes of the
  tokens fed for every stream.
- Use `vocab.bytes` or `vocab.decode` to read token bytes. `vocab.pieces` is an
  uncounted view reserved for the test suite.

## Why this matters

This function runs once per generated token on every streaming request. Decoding each
token independently breaks any multi-byte character split across token boundaries,
turning emoji and CJK text into `�`. Re-decoding the full output on every step fixes
the text, but turns a 4000-token response into millions of byte copies on the server's
hot path. A third common shortcut is to hold output back whenever the decoded string
ends in `�`, which stalls on genuinely invalid bytes until a later token happens to
flush them. The exact rule is small, yet all three buggy shortcuts appear in production
systems.
