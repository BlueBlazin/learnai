What is the minimum information you need to carry from one step to the next? You do not
need the previous tokens or the text already sent. Think about which bytes at the end of
the stream can still change what they decode to when more bytes arrive.

---

Make `state` the pending bytes themselves, stored as a `bytes` object of length 0 to 3
(treating `None` as `b""`). On each step, append the new token's bytes to `state`, find
the length of the settled prefix of that small buffer, decode and return that prefix
with `errors="replace"`, and return the remaining suffix as the new state. Because
`bytes` objects are immutable, creating a new one each step automatically keeps old
states intact. `detok_flush` simply decodes whatever bytes `state` holds.

---

The split point is the same helper as in `utf8-boundaries`. Walk backward from the end
of the buffer over at most three bytes, skipping continuation bytes
(`byte & 0xC0 == 0x80`). If the first non-continuation byte is a valid lead byte
(`C2..DF` needs 2 bytes, `E0..EF` needs 3, `F0..F4` needs 4) with fewer bytes than it
needs, and its second byte (if present) falls in the lead byte's valid range (`E0`:
`A0..BF`, `ED`: `80..9F`, `F0`: `90..BF`, `F4`: `80..8F`), then the suffix starting at
that lead byte is pending. Otherwise, nothing is pending. The full step is:

```python
pending = (state or b"") + vocab.bytes(token_id)
n = complete_prefix_len(pending)
return pending[n:], pending[:n].decode("utf-8", errors="replace")
```
