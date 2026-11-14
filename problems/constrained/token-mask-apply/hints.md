What value can you put into a logit so that the token's probability is exactly zero
after *any* softmax, whatever the other logits are and whatever temperature is applied
later? And what does a softmax do with a row that contains only that value?

---

`mask_logits` is one elementwise selection: keep the logit where `allowed` is true and
write `-inf` elsewhere (`where` in every framework). Before that, look at each row of
the mask separately: reduce `allowed` with "any" along the last axis, and if some row
comes out false, raise `ValueError`. Reducing over the whole tensor at once is not the
same test.

---

For `masked_probs`, apply the mask **before** the softmax, not after:

```python
probs = softmax(mask_logits(logits, allowed), axis=-1)
```

A stable softmax subtracts the row maximum, which is now the largest *allowed* logit,
so the allowed tokens keep full precision, and $e^{-\infty} = 0$ gives the disallowed
ones exactly zero. Taking the softmax first and renormalizing afterward fails when the
allowed tokens' probabilities have already underflowed to zero.
