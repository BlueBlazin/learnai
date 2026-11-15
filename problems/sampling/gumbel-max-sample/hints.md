If $U$ is uniform on $(0, 1)$, then $-\log U$ is exponential and $-\log(-\log U)$ is
standard Gumbel. Write the noise for one entry and evaluate it by hand at the two
values a float32 uniform number can take at the ends: `u = 0.0` and `u = 1.0`. What
is the score of a masked logit in each case?

---

Clamp the uniforms into `[1.1754944e-38, 0.99999994]` before the logarithms. Both
bounds are float32 values, the smallest positive normal number and the largest number
below 1, so after the clamp `log(u)` lies in `[-87.34, -5.96e-8]`: finite and strictly
negative, which makes the second logarithm finite too. Then the noise is finite for
every entry, and that alone settles the masked tokens: `-inf` plus a finite number is
`-inf`.

---

```
u = clip(uniforms, 1.1754944e-38, 0.99999994)
gumbel = -log(-log(u))
token = argmax(logits + gumbel, axis=-1)
```

Add the noise to the logits themselves, not to probabilities and not to
log-probabilities you normalized first: the softmax normalizer is the same for every
entry of a row and cannot change an argmax. An epsilon inside the logarithms
(`log(u + 1e-10)`) does not replace the clamp: in float32, `1 - 1e-10` is `1`.
