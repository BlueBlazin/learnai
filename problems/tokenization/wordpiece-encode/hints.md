At each position you are looking for the longest substring starting there that is in
the vocabulary. In which order should you try the possible end positions so that the
first hit is the answer? And what string do you actually look up when the piece does
not start the word?

---

Keep an index `start`. For `end` from `len(word)` down to `start + 1`, build the
candidate: `word[start:end]` if `start == 0`, otherwise `"##" + word[start:end]`. The
first candidate that is in `vocab` is the piece: record its id and set `start = end`.
Two exits need care. If the inner loop ends without a hit, the answer is
`[vocab[unk]]` for the whole word, so return immediately instead of appending. And the
length check `len(word) > max_chars` comes before any of this.

---

Python's `for ... else` expresses the inner loop exactly:

```python
for end in range(len(word), start, -1):
    piece = word[start:end] if start == 0 else "##" + word[start:end]
    if piece in vocab:
        ids.append(vocab[piece])
        start = end
        break
else:  # no break: nothing matched at this position
    return [vocab[unk]]
```

Wrap it in `while start < len(word)`. An empty word never enters the loop and returns
the empty list.
