A model emits tokens, tokens are byte strings, and a character of text can take up to
four bytes that need not arrive together. Implement the function that says how many of
the bytes received so far can be turned into text now, and how many must wait for the
next token. The difficulty is telling a character that is merely unfinished from bytes
that are broken for good: the first must wait, the second must not.

![This problem is the Detokenize step, deciding how many of the received bytes can become text.](assets/map.svg)

## Specification

```python
def complete_prefix_len(data: bytes) -> int: ...
```

**UTF-8.** A character is encoded as one to four bytes. The first byte, the *lead
byte*, fixes the length; the others are *continuation bytes*, in the range `80..BF`
(hexadecimal throughout). For some lead bytes the second byte is restricted further:

| lead byte | length | second byte | third and fourth byte |
|---|---|---|---|
| `00..7F` | 1 | | |
| `C2..DF` | 2 | `80..BF` | |
| `E0` | 3 | `A0..BF` | `80..BF` |
| `E1..EC`, `EE..EF` | 3 | `80..BF` | `80..BF` |
| `ED` | 3 | `80..9F` | `80..BF` |
| `F0` | 4 | `90..BF` | `80..BF` |
| `F1..F3` | 4 | `80..BF` | `80..BF` |
| `F4` | 4 | `80..8F` | `80..BF` |

The bytes `C0`, `C1`, and `F5..FF` never appear in UTF-8, and a continuation byte
`80..BF` cannot start a character. This is the table of the Unicode standard and of
RFC 3629; the narrower second-byte ranges exclude overlong encodings, UTF-16
surrogates, and code points above U+10FFFF.

**Pending sequence.** A non-empty suffix of `data` is *pending* if it is a row of the
table cut short: a lead byte from the table, followed by zero or more of the bytes its
row allows in those positions, and strictly fewer bytes in total than the row's length.
Equivalently: the suffix is not a whole character yet, but some bytes exist that would
complete it to exactly one valid character. At most one suffix of `data` is pending, and
it is at most three bytes long.

**`complete_prefix_len(data)`** returns `len(data)` minus the length of the pending
suffix, or `len(data)` if no suffix is pending. The bytes before that point are final:
no later byte can change how they decode.

Consequences that the function must get right:

- A complete character at the end is not pending, whatever its length.
- **Invalid bytes are not pending.** A byte that can never start a character (`FF`, a
  stray continuation byte), and a lead byte whose next byte is outside its row (`E0 80`,
  `C3 41`), will decode to U+FFFD replacement characters whatever comes next, so they
  count as complete.
- Only the end of `data` matters. Invalid or unfinished sequences earlier in `data` do
  not change the result, and a pending sequence may directly follow one of them: in
  `E2 E2 82` the last two bytes are pending.
- `data` may be empty; the result is `0`.
- The result is a Python `int`.

Work with the byte values yourself. Do not use the incremental decoders of the
`codecs` module (`codecs.getincrementaldecoder`, `codecs.iterdecode`, and the classes
behind them), which do this bookkeeping internally.

## Example

`é` is `C3 A9`, `€` is `E2 82 AC`, and `😀` is `F0 9F 98 80`.

```python
complete_prefix_len(b"")                      # 0
complete_prefix_len(b"abc")                   # 3
complete_prefix_len(b"caf\xc3")               # 3   the lead byte of "é" waits
complete_prefix_len(b"caf\xc3\xa9")           # 5   "é" is complete
complete_prefix_len(b"\xe2\x82")              # 0   two of the three bytes of "€"
complete_prefix_len(b"ok \xf0\x9f\x98")       # 3   three of the four bytes of "😀"
complete_prefix_len(b"ok \xf0\x9f\x98\x80")   # 7
```

Invalid bytes do not wait:

```python
complete_prefix_len(b"a\xff")                 # 2   FF starts nothing
complete_prefix_len(b"ab\x80")                # 3   a continuation byte without a lead
complete_prefix_len(b"\xe0\xa0")              # 0   E0 A0 can still become U+0800..U+083F
complete_prefix_len(b"\xe0\x80")              # 2   E0 80 cannot become anything
complete_prefix_len(b"\xed\xa0")              # 2   a surrogate: not valid UTF-8
complete_prefix_len(b"\xf4\x90")              # 2   beyond U+10FFFF
```

The typical use is a streaming loop: append each token's bytes to a buffer, decode
`buffer[:n]` with `n = complete_prefix_len(buffer)`, and keep `buffer[n:]` for the next
round. With `errors="replace"` the pieces concatenate to exactly the text that decoding
all the bytes at once would give.

## Constraints

- `len(data)` is at most 100,000. `data` is a `bytes` object and may contain any byte
  values in any order.
- The result is compared exactly.
- There is no speed requirement beyond the time limit, but note that only the last
  three bytes can influence the answer.
- `codecs` incremental decoders are not allowed; the judge rejects a solution that
  calls them.

## Why this matters

Byte-level tokenizers split characters freely: an emoji is often two or three tokens,
and a CJK character can be split after its first byte. A streaming server that decodes
each token's bytes as they arrive sends `�` to the client in the middle of a word, and
the correct character never appears, because the stream cannot be taken back. A server
that instead waits whenever the buffer "looks odd" has the opposite bug: when the model
really does emit an invalid byte, output stalls until something else flushes it. This
function is the exact rule between the two, and every streaming detokenizer contains a
version of it.
