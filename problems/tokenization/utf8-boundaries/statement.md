In a byte-level tokenizer, a single Unicode character can take up to four UTF-8 bytes and arrive split across multiple tokens. Implement `complete_prefix_len`, which returns how many leading bytes of a buffer can be decoded into text now while holding back any incomplete character at the end. Invalid bytes that can never form a valid character must not be held back.

![This problem is the Detokenize step, deciding how many of the received bytes can become text.](assets/map.svg)

## Specification

```python
def complete_prefix_len(data: bytes) -> int: ...
```

**UTF-8.** A Unicode character is encoded as one to four bytes. The first byte, called the *lead byte*, determines the sequence length. The remaining bytes are *continuation bytes* in the hexadecimal range `80..BF`. For some lead bytes, the second byte has a narrower valid range:

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

The bytes `C0`, `C1`, and `F5..FF` never appear in valid UTF-8, and a continuation byte (`80..BF`) cannot start a character. This table matches the Unicode standard and RFC 3629. Its narrower second-byte ranges exclude overlong encodings, UTF-16 surrogates, and code points above U+10FFFF.

**Pending sequence.** A non-empty suffix of `data` is *pending* if it is a valid row of the table cut short. It starts with a lead byte from the table, continues with zero or more bytes allowed in those positions by that row, and contains strictly fewer bytes than the row's length. Equivalently, the suffix is not a complete character yet, but appending one or more bytes can complete it into a single valid character. At most one suffix of `data` is pending, and its length is at most three bytes.

**`complete_prefix_len(data)`** returns `len(data)` minus the length of the pending suffix, or `len(data)` if no suffix is pending. The bytes before that cutoff are final because no later bytes can change how they decode.

Edge cases and rules:

- A complete character at the end of `data` is not pending, regardless of its length.
- **Invalid bytes are not pending.** A byte that can never start a character (such as `FF` or a stray continuation byte) and a lead byte followed by a byte outside its row (such as `E0 80` or `C3 41`) will decode to U+FFFD replacement characters no matter what follows. They count as complete.
- Only the end of `data` affects the result. Invalid or truncated sequences earlier in `data` do not change the answer, and a pending sequence may directly follow one. For example, in `E2 E2 82`, the last two bytes are pending.
- If `data` is empty, return `0`.
- Return a Python `int`.

Work with the byte values directly. Do not use the incremental decoders from the `codecs` module (`codecs.getincrementaldecoder`, `codecs.iterdecode`, `codecs.IncrementalDecoder`, or `codecs.getreader`).

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

In a typical streaming loop, you append each token's bytes to a buffer, compute `n = complete_prefix_len(buffer)`, decode `buffer[:n]`, and keep `buffer[n:]` for the next step. With `errors="replace"`, the decoded chunks concatenate to the exact string produced by decoding all bytes at once.

## Constraints

- `len(data)` is at most 100,000. `data` is a `bytes` object and may contain any byte values in any order.
- The result is compared for exact equality.
- There is no speed requirement beyond the time limit, though only the last three bytes can influence the answer.
- `codecs` incremental decoders are forbidden and rejected by the judge.

## Why this matters

Byte-level tokenizers split multi-byte characters freely. An emoji often spans two or three tokens, and a CJK character can split right after its first byte. A streaming server that decodes each token's bytes as they arrive sends `�` to the client in the middle of a word, and the real character never appears because streamed output cannot be taken back.

A server that holds back bytes whenever the buffer looks invalid has the opposite bug. When the model emits a genuinely invalid byte, output stalls until later bytes flush it. This function defines the exact boundary between waiting for an unfinished character and flushing broken bytes.
