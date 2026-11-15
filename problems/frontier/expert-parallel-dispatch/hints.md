Start from `moe-sparse-dispatch`: there the assignments (token, expert, weight)
were grouped by expert. Here each assignment also has a source device, `home[t]`,
and a destination device, `placement[e]`. Which of the two groupings decides what a
message contains, and which decides what an expert call contains? And if a token
has two experts on the same remote device, how many assignments is that, and how
many rows should cross?

---

Loop over the destination device `d`. For each source `s`, the tokens to bring to
`d` are those with `home == s` that have at least one chosen expert placed on `d`
(`np.flatnonzero` gives them in increasing order, each once); send their rows
unless `s == d`. Concatenate what is now on `d` into one tensor and remember which
token each row is. For each expert on `d`, find the rows routed to it, call it
once, and add `weight * output` into a "partial" tensor with one row per row on
`d` (accumulate: two experts may add into the same row). Then cut the partial
tensor back into the per-source segments, send each segment home unless `s == d`,
and add it into `y` at the tokens of that segment.

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

`partial[at] += ...` must be an accumulating add (`index_add`, `.at[].add`): within
one expert the rows are distinct, and across experts the same row is hit again.
