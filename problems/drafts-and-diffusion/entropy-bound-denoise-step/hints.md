Do the six steps in order on the worked example by hand before coding the acceptance:
write the four entropies, sort them, and write next to each the sum of the entropies
*above it in the sorted list*. Which positions have a sum of at most 0.5? Why is the
first one in the list accepted whatever the bound is?

---

Sampling and entropy read the same distribution, so compute the log-probabilities of
`logits / temperature` once and exponentiate them. Think about what
$-\sum_v p_v \ln p_v$ evaluates to in float32 when some $p_v$ is exactly 0: it depends
on whether the logarithm you multiply by came from `log(p)` or from a `log_softmax`,
which is finite there.

The candidates are the cumulative-sum-and-count of `categorical-sample`, one row per
position.

---

The acceptance needs, for every position, the total entropy of the positions ranked
before it. Two ways to get it:

- sort the entropies ascending, take a cumulative sum in rank order, remove each
  position's own entropy from its entry, and carry the comparison with the bound back
  from rank order to position order (the inverse of a permutation is its `argsort`);
- or compare every pair of positions: $j$ ranks before $i$ when $H_j < H_i$, or
  $H_j = H_i$ and $j < i$, and the sum over those $j$ is an $L \times L$ masked sum.

With a sort, the tie rule is the sort's stability. `torch.sort` and `torch.argsort`
are stable only with `stable=True` (without it, short inputs happen to come back in
index order and long ones do not); `jnp.argsort` is stable by default. If you are not
sure what your framework's sort does with equal keys, the pairwise form does not
depend on it.

---

Check your three results against the example's second call (`entropy_bound = 0.3`):
exactly one position is accepted although no entropy is below 0.3, the rejected
positions hold `floor(u_noise * V)` and not their old tokens, and `argmax_canvas` is
the same as in the first call. Nothing you return depends on `current`, and nothing
should be written into it.
