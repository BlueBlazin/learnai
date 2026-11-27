The formula needs $\log P_i$ and $\log Q_i$. You already know how to compute the logarithm of a softmax so it stays finite even when the probability itself underflows. Once both log-probabilities are finite for every unmasked token, what is the only remaining source of infinities or NaNs?

---

Compute `log_softmax` on both logit tensors (either using the built-in function, or by shifting each row by its maximum first and then subtracting `logsumexp`), and form `exp(log_p) * (log_p - log_q)`. For finite logits, every factor is finite, so the product is accurate even when `exp(log_p)` underflows to `0.0`. The only remaining cases are masked entries, which you should detect from the *input* logits (`p_logits == -inf` and `q_logits == -inf`). Do not test `exp(log_p) == 0`, because that also triggers for a finite logit whose probability underflowed, and rule 2 treats those two cases differently.

---

Overwrite the masked terms with two `where` calls in this order so that rule 1 wins when both tokens are masked:

```python
terms = exp(log_p) * (log_p - log_q)
terms = where(q_logits == -inf, inf, terms)  # rule 2
terms = where(p_logits == -inf, 0.0, terms)  # rule 1, applied last
```

Sum along the last axis and clamp the result below at `0.0`. Because `where` selects elementwise, any NaN in a masked entry that gets overwritten never reaches the sum.
