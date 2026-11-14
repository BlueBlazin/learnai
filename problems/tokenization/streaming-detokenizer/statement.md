A streaming server sends text to the client while the model is still generating.
Implement the detokenizer for that loop: it receives one token at a time and returns
the text that can be sent now. Tokens are byte strings, and a character's bytes may be
spread over several tokens, so the function must hold back an unfinished character,
and nothing else, without ever decoding the whole sequence again.

## Specification

```python
def detok_step(state, token_id: int, vocab: ToyVocab) -> tuple[state, str]: ...
def detok_flush(state) -> str: ...
```

`vocab` is a [`ToyVocab`](../../../python/src/whetstone/kit/vocab.py), the kit's small
token table. `vocab.bytes(token_id)` returns the bytes of one token. Special tokens
(ids 0 to 4, such as `<|eos|>`) have the bytes `b""`.

**The stream.** A stream is a sequence of tokens $t_1, \dots, t_K$ fed one per call:

```python
state = None
for token_id in tokens:
    state, text = detok_step(state, token_id, vocab)
    send(text)
send(detok_flush(state))
```

The first call receives `state = None`. After that, `state` is whatever your
`detok_step` returned; its type is your choice, and the tests never look inside it.

**What each step returns.** Let $B_k$ be the bytes of the first $k$ tokens,
concatenated, and let $D$ be UTF-8 decoding with `errors="replace"`. Split $B_k$ into
a *final part* $F_k$ and a *pending tail*: the pending tail is the unfinished
character at the very end of $B_k$, if there is one, and $F_k$ is everything before
it. "Unfinished character" has the precise meaning of the `utf8-boundaries` problem: a
UTF-8 lead byte followed by fewer continuation bytes than it calls for, where the
bytes present are still allowed by the UTF-8 table, so that more bytes could complete
it to one valid character. The tail is at most three bytes long.

Then:

- step $k$ returns the text that became final in this step: $D(F_k)$ without its
  first $|D(F_{k-1})|$ characters ($D(F_{k-1})$ is always a prefix of $D(F_k)$, and
  $F_0$ is empty);
- `detok_flush` returns $D$ of the pending tail of $B_K$: `""` if there is none, and
  otherwise `"�"`, since a character that was never finished is invalid.

So every character is returned by the first step at which it is certain, the
concatenation of all returned texts and the flush equals $D(B_K)$, the decode of the
whole sequence, and no step ever returns half a character.

Consequences that the functions must get right:

- **Invalid bytes are final immediately.** A byte that cannot start a character (a
  stray continuation byte, `FF`), and an unfinished character followed by a byte that
  cannot continue it, are returned as U+FFFD in the step that makes them certain,
  exactly as they appear in $D(B_K)$. Only a tail that can still become a valid
  character waits.
- **A step often returns `""`**: when the token only starts or continues a character,
  and for every special token. A special token has no bytes, so it changes nothing: an
  unfinished character before it is still pending after it.
- **A step can return several characters**, for example a U+FFFD for an abandoned
  character followed by the text of the current token.
- `detok_flush(None)` is `""`. `detok_flush` does not end the state's life: calling it
  twice returns the same text.
- **States are values.** `detok_step` must not modify the state it is given, and must
  not keep the stream's progress anywhere else (a global, an attribute of `vocab`). The
  caller may continue from the same state more than once, as beam search does, and may
  advance several streams in turn.

> [!NOTE]
> One corner is left open. After the lead byte `ED`, a second byte in `A0..BF` is not
> valid UTF-8 (it would encode a UTF-16 surrogate), so by the rule above it does not
> wait. CPython's incremental decoder does hold it back. Either behavior is accepted:
> the tests never put a byte `A0..BF` after `ED`.

**Cost.** `vocab` counts every byte it hands out in `vocab.counters.bytes_decoded`
(both `vocab.bytes` and `vocab.decode` count). Over a whole stream, that counter must
not exceed **four times** the total number of bytes of the tokens fed. A character has
at most four bytes, so no token ever needs to be looked up in more than four steps;
looking each token up once is enough. Decoding the sequence from the start at every
step costs on the order of $K^2$ bytes and fails this limit.

## Example

With `vocab = ToyVocab(size=64, seed=0)`, take the tokens whose bytes are `{"`, `é`
(`C3 A9`, one token), `": `, the first half of `😀` (`F0 9F`), its second half
(`98 80`), `"}`, and finally `<|eos|>`:

| step | token bytes | returned text | pending after the step |
|---|---|---|---|
| 1 | `7B 22` | `{"` | |
| 2 | `C3 A9` | `é` | |
| 3 | `22 3A 20` | `": ` | |
| 4 | `F0 9F` | `""` | `F0 9F` |
| 5 | `98 80` | `😀` | |
| 6 | `22 7D` | `"}` | |
| 7 | (special) | `""` | |

`detok_flush` returns `""`, and the pieces concatenate to `{"é": 😀"}`.

A stream that stops in the middle of a character: the tokens `5`, a space, and the
first two bytes of `€` (`E2 82`) return `"5"`, `" "`, `""`, and `detok_flush` returns
`"�"`.

Invalid bytes do not wait. With the tokens `E2 82`, `a`, `F0 9F`:

| step | token bytes | returned text | pending after the step |
|---|---|---|---|
| 1 | `E2 82` | `""` | `E2 82` |
| 2 | `61` | `"�a"` | |
| 3 | `F0 9F` | `""` | `F0 9F` |

Step 2 settles the first character: `a` cannot continue it, so `E2 82` is invalid and
comes out as one U+FFFD together with the `a`. `detok_flush` returns `"�"` for
the unfinished `F0 9F`.

## Constraints

- Streams have at most 5000 tokens. Vocabularies have 64 to 282 tokens with various
  seeds, so token ids must not be hard-coded; at size 282 every byte value is a token,
  and streams contain arbitrary invalid byte sequences.
- The text of every step and of the flush is compared exactly.
- `vocab.counters.bytes_decoded` is at most four times the total bytes of the tokens
  fed, for every stream.
- Use `vocab.bytes` or `vocab.decode` to get at the token bytes; `vocab.pieces` is an
  uncounted view that exists for the tests.

## Why this matters

This function runs once per generated token for every streaming request. Decoding each
token on its own is the classic bug: multi-byte characters arrive as `�` in the middle
of a word, most visibly with emoji and CJK text. Decoding the whole output again at
every step fixes the text and makes a 4000-token response cost millions of byte copies
on the server's hot path. And the common heuristic of holding output back whenever the
decoded text ends in `�` stalls on bytes that really are invalid: the client sees
nothing until some later token happens to release it. The exact rule is small, but
each of the three shortcuts ships in real systems.
