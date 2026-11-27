Separate the two kinds of controls. A bias shifts a logit by a finite amount, whereas the allow-list and banned sequences remove a token outright. If you compute the removed token ids as a mask and apply that mask last, a bias can never undo a removal. For a banned sequence `[a, b, c]`, think about which single token gets removed and what the last two tokens of `history` must be for that to happen.

---

Build two per-token Python lists first, then touch the tensor once. Let `delta[v]` hold the bias for token `v` (or `0.0` if absent). Initialize `removed[v]` to `allowed_token_ids is not None` for every token, and set it to `False` for each id in `allowed_token_ids`. Then, for each sequence `word` in `bad_words`, split it into `prefix = word[:-1]` and `last = word[-1]`. Whenever `history` ends with `prefix`, mark `removed[last] = True`. Finally, return `-inf` where `removed` is true and `logits + delta` elsewhere.

---

Checking whether `history` ends with `prefix` requires care when `prefix` is empty or longer than `history`:

```
start = len(history) - len(prefix)
matches = start >= 0 and history[start:] == prefix
```

Avoid `history[-len(prefix):]`, because for a one-token sequence `len(prefix)` is `0` and `history[-0:]` evaluates to the entire history rather than `[]`. Compare the slices as ordered lists rather than sets or sorted lists, and only check the tail of `history`. Then combine everything in one select:

```
out = where(removed, -inf, logits + delta)
```
