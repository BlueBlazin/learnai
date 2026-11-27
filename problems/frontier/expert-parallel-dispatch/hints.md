Start from `moe-sparse-dispatch`, where `(token, expert, weight)` assignments were grouped by expert. Here, each assignment also has a source device `home[t]` and a destination device `placement[e]`. Ask which grouping determines a message payload and which determines an expert call. If a token has two selected experts on the same remote device, how many assignments is that, and how many rows should cross?

---

Loop over each destination device `d`. For each source device `s`, find the tokens with `home == s` that have at least one chosen expert on `d` (`np.flatnonzero` returns them in increasing order, each once). Send their rows via `devs.send` unless `s == d`. Concatenate all local and received rows on `d` into one tensor and track which token ID each row belongs to. For each expert placed on `d`, find the rows routed to it, call the expert once, and accumulate `weight * output` into a `partial` tensor aligned with the rows on `d` (make sure to accumulate, since two experts on `d` may contribute to the same row). Finally, slice `partial` back into per-source segments, send each remote segment back to `s`, and add it into `y` at that segment's token indices.

---

```
target = placement[chosen]                         # (N, k): device of each chosen expert
for d in devices:
    batches = []
    for s in devices:
        tokens = flatnonzero((home == s) & (target == d).any(axis=1))
        if tokens.size:
            rows = x[tokens]
            batches.append((s, tokens, rows if s == d else devs.send(s, d, rows)))
    rows, tokens = concat of the batches' rows, concat of their token ids
    partial = zeros_like(rows)
    for e in experts placed on d:
        at, slot = nonzero(chosen[tokens] == e)    # row on d, and which of its k slots
        if at.size:
            partial[at] += weights[tokens[at], slot][:, None] * experts(e, rows[at])
    for (s, sent, _), part in zip(batches, segments of partial):
        y[sent] += part if s == d else devs.send(d, s, part)
```

`partial[at] += ...` must be an accumulating add (`index_add` or `.at[].add`). Within a single expert the indices in `at` are distinct, but across multiple experts on the same device the same row can be updated more than once.
