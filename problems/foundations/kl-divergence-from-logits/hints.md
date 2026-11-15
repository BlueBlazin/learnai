The formula needs $\log P_i$ and $\log Q_i$. You already know a way to get the
logarithm of a softmax that stays finite when the probability itself underflows. What
does that leave as the only place where an infinity or a NaN can still come from?

---

Take `log_softmax` of both logit tensors (the built-in one, or your own: shift each
row by its maximum first, then subtract the log of the sum of exponentials) and form `exp(log_p) * (log_p - log_q)`. For
finite logits every factor is finite, so the product is right even when `exp(log_p)`
underflows to 0. What remains are the masked entries, and those you can recognize from
the *inputs*: `p_logits == -inf` and `q_logits == -inf`. Do not test `exp(log_p) == 0`:
that is also true for a finite logit whose probability underflowed, and rule 2 treats
the two differently.

---

Overwrite the terms with two `where` calls, in this order, so that rule 1 wins where
both are masked:

```python
terms = exp(log_p) * (log_p - log_q)
terms = where(q_logits == -inf, inf, terms)  # rule 2
terms = where(p_logits == -inf, 0.0, terms)  # rule 1, applied last
```

Sum along the last axis and clamp the result at 0 from below. `where` only selects: a
NaN in a term that gets overwritten never reaches the sum.
