The four encoding steps nest cleanly. Added tokens split the text into stretches, the regex splits each stretch into words, and BPE runs inside each word. Think about which earlier solutions correspond to each layer, and which lookup tables you should build *once* in `load_tokenizer` rather than rebuilding on every `encode` call.

---

Precompute these structures in `load_tokenizer` and close over them in `encode` and `decode`:

- `rank`: a dict mapping `(left, right)` to its index in `merges`.
- `added`: a dict mapping `content` to `id` for **all** entries in `added_tokens`, plus a compiled alternation regex of their escaped contents sorted longest-first (at the same position, regex alternation picks the first matching branch).
- `word_re`: `re.compile(translate_pattern(pattern))`.
- `alphabet`: a list of 256 single-character strings for the ByteLevel mapping, and its inverse dict for `decode`.
- `token_bytes`: a dict mapping each id to `bytes`, built from every vocabulary token (via the inverse alphabet) and overwritten for every added token with `content.encode("utf-8")`.

In `encode`, iterate over `added_re.finditer(text)`. Encode the stretch before each match with the ordinary path, append the matched added token's id, and encode any trailing stretch at the end. The ordinary path iterates over `word_re.finditer`, converts each word's UTF-8 bytes into alphabet characters, runs the BPE merge loop from `bpe-encode` on a list of strings, and maps each final string through `vocab`.

---

Here is the ByteLevel alphabet construction along with the two lines that most often cause bugs:

```python
keep = [*range(0x21, 0x7F), *range(0xA1, 0xAD), *range(0xAE, 0x100)]
chars = {byte: chr(byte) for byte in keep}
spare = 0x100
for byte in range(256):
    if byte not in chars:
        chars[byte] = chr(spare)
        spare += 1

# inside the BPE loop: lowest rank among the pairs that are present
best = min(pairwise(parts), key=lambda pair: rank.get(pair, len(rank)))

# decode: join bytes first, decode once
return b"".join(token_bytes[i] for i in ids).decode("utf-8", errors="replace")
```

Look up each merged token's id with `vocab[part]`. Never compute it from the merge rank.
