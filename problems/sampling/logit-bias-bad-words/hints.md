Separate the two kinds of effect. A bias changes a logit by a finite amount; the
allow-list and the banned sequences remove a token outright. If you decide the
removals as a set of token ids and apply them last, can a bias ever undo one? And for
a banned sequence `[a, b, c]`: which single token does it remove, and what must the
last two tokens of the history be at that moment?

---

Build two per-token tables in Python, then touch the tensor once. `delta[v]` is the
bias of token `v` (0 if it has none). `removed[v]` starts as "is there an allow-list"
for every token and is set to false for the listed ones. Then, for each banned
sequence, split it into `prefix = word[:-1]` and `last = word[-1]`; if the history
ends with `prefix`, set `removed[last]`. The result is `-inf` where `removed`, and
`logits + delta` elsewhere.

---

"The history ends with `prefix`" needs care when the prefix is empty or longer than
the history:

```
start = len(history) - len(prefix)
matches = start >= 0 and history[start:] == prefix
```

Do not write `history[-len(prefix):]`: for a one-token sequence the prefix is empty
and `history[-0:]` is the entire history, not an empty list. Compare lists, in order;
comparing sets or sorted lists accepts the wrong order, and searching the whole
history accepts a prefix that is no longer at the end. Finally:

```
out = where(removed, -inf, logits + delta)
```
