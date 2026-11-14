Think about what you are allowed to show. After some pieces the text ends in `"\nUs"`
and one stop string is `"\nUser:"`. You cannot emit those four characters (the stop
may complete, and emitted text cannot be taken back) and you cannot drop them (the
next piece may be `"ually"`). So they wait. What is the general rule for which
characters at the end of the text have to wait, and what is the least the state has
to remember between calls?

---

The state can be just the held-back text. In each call, form
`buffer = held + piece` and ask two questions in this order:

1. Does `buffer` contain a complete stop string? Every match that could complete now
   begins inside `buffer`, because the held text was chosen to contain every possible
   beginning. If so, emit `buffer` up to the start of the match and stop.
2. Otherwise, what is the longest suffix of `buffer` that is a prefix of a stop
   string, but not a whole one? Hold that; emit the rest.

Compute the suffix from the whole `buffer`, not from `piece`: with the stop `"aab"`
and the pieces `"a"`, `"a"`, `"a"`, the buffer `"aaa"` is not a prefix of the stop,
but its last two characters are.

---

For question 1, `buffer.find(stop)` gives the first occurrence of each stop string,
which is also the occurrence that ends first. Compare the stop strings by
`(start + len(stop), start)` and take the smallest: earliest end, then earliest
start. For question 2, try the lengths `n = min(len(stop) - 1, len(buffer))` down to
1 and test `buffer.endswith(stop[:n])`; keep the largest `n` over all stops.

```
if a match was found:  return "", buffer[:start], True
else:                  return buffer[len(buffer) - n:], buffer[:len(buffer) - n], False
```

`stop_flush` returns the held text (and `""` for a state of `None`).
