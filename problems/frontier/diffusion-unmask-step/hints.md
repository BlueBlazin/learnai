This step uses two different probability distributions, and it is easy to confuse them. One distribution decides *which candidate token* a position draws (using `temperature` and `uniforms`). The other measures *how confident* the model is in that candidate. Which distribution ranks the positions, and which token's probability do you take from it?

---

Process all rows at once. Start by setting the `mask_id` column of `logits` to `-inf` so every later step ignores it automatically. To draw candidates, take `argmax` when `temperature == 0`. Otherwise, take the cumulative sum of `softmax(logits / temperature)` and count how many entries in each row are `<= uniforms[:, None] * cdf[:, -1:]`. For confidence, gather each position's candidate probability from the unscaled `softmax(logits)`, then overwrite unmasked positions with `-inf` so they can never rank in the top `n`.

---

Sort positions by descending confidence with a *stable* sort so ties stay in index order, then invert the permutation to get each position's rank:

```
n     = min(n_unmask, number of masked positions)
order = stable argsort of -confidence
rank  = argsort(order)            # rank[i]: place of position i in the order
new   = where(rank < n, candidates, tokens)
```

Because unmasked positions have confidence `-inf` and `n` never exceeds the number of masked positions, `rank < n` selects only masked positions. For `schedule`, use `divmod(n_masked, steps)` to give every step the quotient and add 1 to the first `remainder` steps.
