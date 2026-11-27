At each position, you want the longest substring starting there that exists in `vocab`. In what order should you test candidate end positions so the first hit is guaranteed to be the longest? And how does the lookup string change once you move past index 0?

---

Track the current index `start`. For `end` from `len(word)` down to `start + 1`, form the candidate as `word[start:end]` when `start == 0`, or `"##" + word[start:end]` otherwise. On the first candidate found in `vocab`, append its ID and update `start = end`.

Two edge cases need care. First, check `len(word) > max_chars` before the loop starts. Second, if the inner loop finishes without a match, return `[vocab[unk]]` immediately so any earlier pieces are discarded.

---

Python's `for ... else` handles the inner loop cleanly:

```python
for end in range(len(word), start, -1):
    piece = word[start:end] if start == 0 else "##" + word[start:end]
    if piece in vocab:
        ids.append(vocab[piece])
        start = end
        break
else:  # no break means nothing matched at this position
    return [vocab[unk]]
```

Wrap this in `while start < len(word):`. An empty word skips the loop and returns `[]`.
