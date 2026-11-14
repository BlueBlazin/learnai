The four steps nest: added tokens cut the text into stretches, the pattern cuts a
stretch into words, and BPE runs inside one word. Which of your earlier solutions is
each layer, and which data structure does each one need to be built *once*, in
`load_tokenizer`, rather than on every call to `encode`?

---

Build these in `load_tokenizer` and close over them:

- `rank`: a dict from `(left, right)` to the pair's position in `merges`;
- `added`: a dict from `content` to `id`, for **all** entries of `added_tokens`, and a
  compiled regex that alternates their escaped contents, longest first (at one
  position, alternation takes the first branch that matches);
- `word_re = re.compile(translate_pattern(pattern))`;
- `alphabet`: a list of 256 one-character strings, and its inverse for `decode`;
- `token_bytes`: a dict from id to `bytes`, for every vocabulary entry (each character
  through the inverse alphabet) and every added token (`content.encode("utf-8")`).

`encode` walks `added_re.finditer(text)`, encoding the text before each match with the
ordinary path and appending the match's id. The ordinary path is: for each match of
`word_re`, UTF-8 bytes, alphabet characters, the BPE loop of `bpe-encode` on a list of
strings, then `vocab[...]`.

---

The alphabet, and the two lines people get wrong:

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

The id of a merged token is `vocab[left + right]`. Do not compute it from the rank.
