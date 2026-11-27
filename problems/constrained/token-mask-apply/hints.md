What value can you write into a logit so that its probability is exact `0` after any softmax, regardless of the other logits or any later temperature scaling? And what happens if a softmax receives a row containing only that value?

---

`mask_logits` is a single elementwise selection (`where` in every framework) that keeps `logits` where `allowed` is `True` and writes `-inf` elsewhere. Before selecting, check each row separately by reducing `allowed` with `any` along the last axis and raising `ValueError` if any row is `False`. Reducing over the entire tensor at once misses a single empty row.

---

For `masked_probs`, apply the mask **before** running softmax, not after.

```python
probs = softmax(mask_logits(logits, allowed), axis=-1)
```

A stable softmax subtracts the row maximum, which is now the largest *allowed* logit, preserving full precision for allowed tokens while $e^{-\infty} = 0$ gives exact zero to disallowed tokens. Running softmax first and renormalizing afterward fails when allowed probabilities underflow to zero.
