Break the computation into three stages on tensors of shape `(B, H, T, d)`.

First, RMSNorm acts on one vector at a time, so decide which single axis the mean square runs over and how many root mean squares you compute in total. Next, check which axes of `(B, H, T, d/2)` align with the `cos` and `sin` tables of shape `(T, d/2)`. Finally, check which two axes form the attention score matrix when mixing tokens.

---

Follow these three steps in order:

1. RMS-normalize `q` with `q_weight` and `k` with `k_weight`. Take the mean of squares over the **last axis only**, keeping that dimension so it broadcasts cleanly (one root mean square per `(b, h, t)`). Leave `v` untouched.
2. Rotate the *normalized* `q` and `k`. Split the even-indexed and odd-indexed components (`x[..., 0::2]` and `x[..., 1::2]`, each of shape `(B, H, T, d/2)`), apply the 2D rotation with `cos` and `sin` (which broadcast from the right), and interleave the rotated components back into alternating order.
3. Compute `scores = q @ k^T / sqrt(d)`, mask future positions with `-inf`, apply softmax over the last axis, and multiply by `v`.

Steps 1 and 2 do not commute because of the learned gains. Rotation preserves the root mean square, but the gain scales each dimension independently before rotation mixes each adjacent pair.

---

For a tensor `x` of shape `(B, H, T, d)` and gain `g` of shape `(d,)`, the normalization and rotation formulas are:

$$
\hat{x} = \frac{x}{\sqrt{\operatorname{mean}_{\text{last axis}}(x^2) + \varepsilon}} \odot g,
\qquad
\tilde{x}_{2i} = \hat{x}_{2i} \cos - \hat{x}_{2i+1} \sin, \quad
\tilde{x}_{2i+1} = \hat{x}_{2i} \sin + \hat{x}_{2i+1} \cos.
$$

To interleave the rotated even and odd halves, stack them along a new last axis to get shape `(B, H, T, d/2, 2)`, then reshape back to `(B, H, T, d)`. Then compute attention as:

$$
y = \operatorname{softmax}\!\Big(\frac{\tilde{q}\,\tilde{k}^\top}{\sqrt{d}} + M\Big)\, v,
$$

where $M$ is $-\infty$ strictly above the diagonal and `0` elsewhere. As a sanity check on the worked example, both normalized keys should have comparable magnitudes ($[2.83, 0]$ and $[2, 1]$) even though the raw keys differ by a factor of 20.
