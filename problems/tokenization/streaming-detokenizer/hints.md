What is the least you need to remember between two steps? Not the tokens, and not the
text already sent. Think about which bytes of the stream can still change what they
decode to when more bytes arrive.

---

Let the state be the pending bytes themselves: a `bytes` object of length 0 to 3
(`None` at the start counts as empty). In each step, append the new token's bytes to
it, find how long the final part of that small buffer is, decode the final part with
`errors="replace"` and return it, and keep the rest as the new state. Because `bytes`
objects are immutable, building a new one each step also satisfies the rule that old
states stay usable. `detok_flush` decodes whatever the state holds.

---

The split point is the function from `utf8-boundaries`: walk back from the end of the
buffer over at most three bytes, skipping continuation bytes (`b & 0xC0 == 0x80`). If
the first other byte is a lead byte (`C2..DF` needs 2 bytes, `E0..EF` 3, `F0..F4` 4)
that has fewer bytes than it needs, and its second byte, if present, is inside the
lead's range (`E0`: `A0..BF`, `ED`: `80..9F`, `F0`: `90..BF`, `F4`: `80..8F`), the
tail starting at that lead byte is pending. Otherwise nothing is. The whole step:

```python
pending = (state or b"") + vocab.bytes(token_id)
n = complete_prefix_len(pending)
return pending[n:], pending[:n].decode("utf-8", errors="replace")
```
