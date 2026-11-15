There are two different probabilities in this step, and it is easy to merge them.
One distribution decides *which token* a position would get (it depends on the
temperature and on the uniform number). The other says *how sure the model is of
that token*. Which one do you rank positions by, and of which token?

---

Work on all rows at once. Set the mask token's column to `-inf` first; everything
after that then ignores it automatically. Candidates: `argmax` at temperature 0,
otherwise the cumulative sum of `softmax(logits / temperature)` compared with
`uniforms[:, None] * cdf[:, -1:]`, counting the entries that do not exceed the
threshold. Confidence: gather the candidate's entry from `softmax(logits)`, then
overwrite it with `-inf` wherever the position is not masked, so that such a
position can never rank among the first `n`.

---

For the commit, sort positions by descending confidence with a *stable* sort (so
that equal confidences stay in index order), and turn the order into a rank:

```
n     = min(n_unmask, number of masked positions)
order = stable argsort of -confidence
rank  = argsort(order)            # rank[i]: place of position i in the order
new   = where(rank < n, candidates, tokens)
```

Because unmasked positions have confidence `-inf` and `n` never exceeds the number
of masked positions, `rank < n` selects masked positions only. The schedule is
`divmod(n_masked, steps)`: the quotient for every step, plus one for each of the
first `remainder` steps.
