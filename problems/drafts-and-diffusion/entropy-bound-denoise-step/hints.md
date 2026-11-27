Work through the six steps on the example by hand before writing code. Compute the four entropies, sort them in ascending order, and write next to each position the sum of the entropies *strictly above it* in the sorted list. Which positions have a sum of at most 0.5? Why is the first position in the list always accepted regardless of the bound?

---

Both sampling and entropy use the same temperature-scaled distribution, so compute `log_p = log_softmax(logits / temperature)` once and take `p = exp(log_p)`. Think carefully about $-\sum_v p_v \ln p_v$ in float32 when some $p_v$ underflows to 0. Multiplying `p * log(p)` produces `0 * -inf = NaN`, whereas multiplying `p * log_p` stays finite because `log_p` from `log_softmax` is finite.

For the candidate tokens, apply the same cumulative-sum-and-count pattern from `categorical-sample` independently to each row.

---

Acceptance requires, for every position, the sum of entropies of all positions ranked before it. There are two clean ways to compute this:

- Sort the entropies in ascending order, take their cumulative sum, subtract each position's own entropy from its cumulative sum, compare against `entropy_bound`, and map the boolean mask back to original position order using the inverse permutation (`argsort(order)`).
- Compare every pair of positions directly: position $j$ ranks before position $i$ if $H_j < H_i$, or if $H_j = H_i$ and $j < i$. Summing $H_j$ over those $j$ is an $L \times L$ masked sum.

If you sort, tie-breaking depends on sort stability. `torch.sort` and `torch.argsort` are stable only when you pass `stable=True` (without it, short arrays happen to keep index order while longer ones do not). `jnp.argsort` is stable by default. If you are unsure whether your framework's sort is stable, the pairwise comparison avoids sorting altogether.

---

Check all three outputs against the example's second call (`entropy_bound = 0.3`). Exactly one position should be accepted even though every position has entropy above 0.3, rejected positions should contain `floor(u_noise * V)` rather than their old tokens, and `argmax_canvas` should match the first call. None of the outputs depend on `current`, and `current` must not be modified.
