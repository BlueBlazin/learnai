## The idea

The nucleus is defined on the ranking of the tokens, so sort first. In sorted order
the nucleus is a prefix, and membership becomes a comparison on a cumulative sum,
which is one vectorized operation for the whole batch.

The only delicate point is which cumulative sum. Let $c_k$ be the mass of the top $k$
tokens. The nucleus ends at $k^\ast = \min\{k : c_k \ge p\}$. For a rank
$k \le k^\ast$ we have $c_{k-1} < p$ (otherwise the prefix would have stopped
earlier), and for $k > k^\ast$ we have $c_{k-1} \ge c_{k^\ast} \ge p$. So

$$
\text{rank } k \text{ is kept} \iff c_{k-1} < p .
$$

The *exclusive* cumulative sum decides. It gives both boundary rules for free: the
crossing token has $c_{k-1} < p \le c_k$ and is kept, and the first rank has
$c_0 = 0 < p$ and is always kept. No special case, no clamp, no "keep at least one"
patch.

Then renormalize the kept probabilities, and scatter them back to token positions.

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

`stable=True` implements the tie rule: equal probabilities keep their original order,
lowest index first. JAX's `argsort` is stable by default (sort `-probs` for descending
order), and MLX's `argsort` keeps equal elements in index order as well. `mass_before[..., 0]` is `x - x`, exactly zero, so the top
token is kept for any positive `p` regardless of rounding.

The last line inverts the sort. `order[b, k]` is the token at rank `k`, so the value
at rank `k` must be *written to* index `order[b, k]`; that is what scatter does.

## Common mistakes

- **Dropping the crossing token.** `cumsum(sorted) <= p` keeps a token only if the
  mass including it is within `p`. The kept mass is then below `p`, and when the top
  token alone exceeds `p` nothing is kept at all, which people then patch by forcing
  the first token. The patch hides the symptom; every other row is still one token
  short.
- **Shifting twice.** The widespread formulation computes the inclusive mask
  `cumsum > p` and shifts it right by one position. That is the same rule as the
  exclusive sum. Doing both, an exclusive sum and a shift, keeps one token too many.
- **Forgetting to renormalize.** The zeroed vector sums to $c_{k^\ast}$, not 1. Some
  samplers normalize internally and hide this; an inverse-CDF sampler does not.
- **Not undoing the sort**, or undoing it with `gather(kept, order)`. Gather reads
  from position `order[k]`; it applies the permutation again instead of inverting it.
  The output is a valid distribution over the wrong tokens, and it is even correct
  whenever the permutation happens to be its own inverse (already sorted rows, or a
  single swap), which is why small hand tests miss it.
- **Filtering across the batch.** Reducing the cutoff to one scalar (a global
  threshold probability, a global count) makes every row's result depend on the other
  rows. Each row has its own $k^\ast$.
- **Unstable or reversed tie order.** Sorting ascending and flipping reverses the
  order of equal elements. It only matters when a tie straddles the boundary, which
  is rare with real logits and common with masked, quantized, or padded ones.

## Complexity

$O(B \cdot V \log V)$ time for the sort and $O(B \cdot V)$ memory. For large
vocabularies engines often take a top-k (a partial sort, $O(V + k \log k)$) first and
apply top-p inside it.

## In the wild

Nucleus sampling was introduced by Holtzman et al., "The Curious Case of Neural Text
Degeneration" (ICLR 2020), with exactly this definition: the smallest set whose mass
is at least `p`. Hugging Face's `TopPLogitsWarper`, llama.cpp's top-p sampler, and
vLLM's sampler all sort, take a cumulative sum, and keep the crossing token. Most of
them work on logits and set removed tokens to `-inf` instead of returning
probabilities, so that further processors can be chained; the renormalization then
happens in the final softmax. None of them specify the order of ties, which is one
reason token-for-token reproducibility across engines is hard.
