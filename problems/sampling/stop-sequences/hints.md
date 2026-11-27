Think about what you are allowed to emit. Suppose the text ends in `"\nUs"` and one stop string is `"\nUser:"`. You cannot emit those three characters because the stop string might complete, and emitted text cannot be taken back. You also cannot drop them because the next piece might be `"ually"`. What is the general rule for which trailing characters must wait, and what is the minimum state needed between calls?

---

The state only needs to store the held-back text. On each call, form `buffer = held + piece` and check two things in order:

1. Does `buffer` contain a complete stop string? Every match that can complete now starts inside `buffer`, because `held` already contains every possible partial prefix. If a match exists, emit `buffer` up to the match's start and stop.
2. Otherwise, find the longest suffix of `buffer` that is a proper prefix of some stop string (a prefix shorter than the stop string itself). Hold that suffix and emit the rest.

Compute this suffix from the entire `buffer`, not just `piece`. With stop `"aab"` and pieces `"a"`, `"a"`, `"a"`, the buffer `"aaa"` is not a prefix of `"aab"`, but its last two characters are.

---

For step 1, `start = buffer.find(stop)` returns the first occurrence of each stop string, which is also the occurrence that ends earliest. Compare matches by `(start + len(stop), start)` and take the smallest pair (earliest end, then earliest start). For step 2, iterate `n` from `min(len(stop) - 1, len(buffer))` down to 1, test `buffer.endswith(stop[:n])`, and keep the largest `n` across all stops.

```
if a match was found:  return "", buffer[:start], True
else:                  return buffer[len(buffer) - n:], buffer[:len(buffer) - n], False
```

`stop_flush` returns the held text, or `""` when `state` is `None`.
