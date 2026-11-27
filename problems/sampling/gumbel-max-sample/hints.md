If $U$ is uniform on $(0, 1)$, then $-\log U$ is exponential and $-\log(-\log U)$ is standard Gumbel. Write out this noise formula for a single entry and evaluate it by hand at the two boundary values a float32 uniform can hit: `u = 0.0` and `u = 1.0`. Check what happens to the score of a masked `-inf` logit in each case.

---

Clamp `uniforms` into `[1.1754944e-38, 0.99999994]` before taking the logarithms. Those bounds are the smallest positive normal float32 and the largest float32 below `1.0`. After clamping, `log(u)` stays in `[-87.34, -5.96e-8]`, which is finite and strictly negative, so the outer logarithm is also finite. Because the noise is always finite, adding it to a masked `-inf` logit always gives `-inf`.

---

```
u = clip(uniforms, 1.1754944e-38, 0.99999994)
gumbel = -log(-log(u))
token = argmax(logits + gumbel, axis=-1)
```

Add the noise directly to the raw logits rather than probabilities or log-probabilities. Subtracting the softmax log-normalizer shifts every entry in a row by the same constant and cannot change the argmax. Also avoid replacing the clamp with an epsilon inside the logarithm (such as `log(u + 1e-10)`), because `1.0 - 1e-10` rounds to `1.0` in float32.
