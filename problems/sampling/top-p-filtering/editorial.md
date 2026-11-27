## The idea

Because the nucleus is defined on the ranked tokens, sort each row's probabilities in
descending order first. In sorted order, the nucleus is a contiguous prefix, and
checking membership becomes a single vectorized comparison on a cumulative sum across
the whole batch.

The subtle part is choosing which cumulative sum to compare against `p`. Let $c_k$ be
the cumulative probability of the top $k$ tokens, with $c_0 = 0$. The nucleus ends at
$k^\ast = \min\{k : c_k \ge p\}$. For every rank $k \le k^\ast$, we have $c_{k-1} < p$
(otherwise the prefix would have stopped earlier). For every rank $k > k^\ast$, we have
$c_{k-1} \ge c_{k^\ast} \ge p$. Therefore,

$$
\text{rank } k \text{ is kept} \iff c_{k-1} < p .
$$

Comparing the *exclusive* cumulative sum against `p` handles both boundary rules
automatically. The crossing token satisfies $c_{k-1} < p \le c_k$ and is kept. The top
rank has $c_0 = 0 < p$ and is always kept, so no special cases, clamping, or fallback
checks are needed.

After masking out tokens outside the nucleus, renormalize the kept probabilities and
scatter them back to their original token positions.

## Reference solution (PyTorch)

```python
def top_p_filter(logits, p):
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, order = torch.sort(probs, dim=-1, descending=True, stable=True)
    mass_before = torch.cumsum(sorted_probs, dim=-1) - sorted_probs
    kept = torch.where(mass_before < p, sorted_probs, 0.0)
    kept = kept / kept.sum(dim=-1, keepdim=True)
    return torch.zeros_like(probs).scatter(-1, order, kept)
```

Passing `stable=True` enforces the tie-breaking rule by keeping equal probabilities in
ascending index order. JAX's `argsort` is stable by default (sort `-probs` for
descending order), and MLX's `argsort` also preserves index order for ties. For the
top-ranked token, `mass_before[..., 0]` is `x - x`, which is exactly `0.0`, so the top
token is kept for any positive `p` regardless of floating-point rounding.

The final line inverts the sort. Because `order[b, k]` holds the original token index at
rank `k`, the kept probability at rank `k` must be written to index `order[b, k]`.
Scattering along the last axis performs that inverse permutation.

## Common mistakes

- **Dropping the crossing token.** Checking `cumsum(sorted) <= p` keeps a token only if
  the cumulative mass including it stays within `p`. The retained mass then falls short
  of `p`, and when the top token alone exceeds `p`, no tokens survive. Patching that
  case by forcing the first token hides the crash while leaving every other row one
  token short.
- **Shifting twice.** Another common formulation computes the inclusive mask
  `cumsum > p` and shifts it right by one position, which is equivalent to using the
  exclusive sum. Applying both an exclusive sum and a right shift keeps one token too
  many.
- **Forgetting to renormalize.** After zeroing out removed tokens, each row sums to
  $c_{k^\ast}$ rather than 1. Some samplers renormalize internally and mask this bug,
  whereas an inverse-CDF sampler fails silently.
- **Not undoing the sort, or using `gather(kept, order)`.** Calling `gather` reads from
  position `order[k]`, applying the permutation a second time instead of inverting it.
  The result is still a valid probability distribution, just over the wrong tokens. It
  even produces the right answer whenever the permutation is its own inverse (such as
  already-sorted rows or a single swap), so small hand tests often miss it.
- **Filtering across the batch.** Reducing the cutoff to a single scalar threshold or
  token count makes every row depend on the other rows in the batch. Each row has its
  own $k^\ast$.
- **Unstable or reversed tie order.** Sorting in ascending order and reversing the
  result flips the order of tied elements. This shows up only when a tie straddles the
  cutoff boundary, which is rare for arbitrary logits but common for masked, quantized,
  or padded inputs.

## Complexity

Sorting takes $O(B \cdot V \log V)$ time and $O(B \cdot V)$ memory. For large
vocabularies, inference engines often run a partial top-$k$ sort in $O(V + k \log k)$
time first and then apply top-p filtering inside those $k$ candidates.

## In the wild

Nucleus sampling was introduced by Holtzman et al. in "The Curious Case of Neural Text
Degeneration" (ICLR 2020), using this exact definition of the smallest set whose
probability mass is at least `p`. Hugging Face's `TopPLogitsWarper`, llama.cpp's top-p
sampler, and vLLM's sampler all sort, compute a cumulative sum, and keep the crossing
token. Most of them operate directly on logits and set filtered tokens to `-inf` so
additional logit processors can be chained before a final softmax handles
renormalization. None of them specify a tie-breaking order, which is one reason exact
token-for-token reproducibility across inference engines is hard.
