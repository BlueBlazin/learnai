The routing gives every token a list of `k` experts and `k` weights. Turn that table
on its side: for each expert, which tokens chose it, and with which weight? Once you
have that list for an expert, the three steps of the layer are a gather (`x[tokens]`),
one call, and a scatter-add back to `tokens`. How do you get all the lists without a
Python loop over tokens, and without a boolean-mask read (which MLX does not have)?

---

Flatten the routing into `N * k` assignments: `flat_expert = chosen.reshape(-1)`, the
token of each assignment `flat_token = repeat(arange(N), k)`, and
`flat_weight = weights.reshape(-1)`. Sort the assignments by expert
(`order = argsort(flat_expert)`). Now every expert owns one contiguous run of `order`,
and the run lengths are the counts of each expert in `flat_expert` (a `bincount`).
Bring the counts to Python ints: they decide how many rows each call gets, and an
expert with a count of 0 must be skipped, not called.

---

```
y = zeros_like(x)
start = 0
for expert, count in enumerate(counts):
    if count == 0:
        continue
    run = order[start : start + count]
    start += count
    tokens = flat_token[run]
    out = experts(expert, x[tokens])
    y[tokens] += flat_weight[run, None] * out      # scatter-add
```

The scatter must accumulate, since each token receives `k` contributions, and it must
use `tokens` as indices, not a slice: the rows of one expert are spread over `x`. In
torch the accumulation is `y.index_add(0, tokens, ...)`; in JAX and MLX,
`y.at[tokens].add(...)`. Within one run the token indices are distinct (a token does
not select an expert twice), so an indexed `+=` is safe here as well. For the weights,
a softmax over just the `k` kept logits equals the full softmax renormalized over
them.
