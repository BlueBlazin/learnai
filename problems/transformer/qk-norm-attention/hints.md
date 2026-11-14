There are three stages, and each one works on a different part of the shape
`(B, H, T, d)`. The normalization looks at one vector at a time: which single axis does
its mean run over, and how many root mean squares are there in total? The rotation
needs a table that depends on the token: which axis must `cos` of shape `(T, d/2)` line
up with? The attention mixes tokens: which two axes does the score matrix have at the
end?

---

In order:

1. RMS-normalize `q` with `q_weight` and `k` with `k_weight`. The mean of the squares
   is over the **last axis only**, keeping that axis so it broadcasts: one root mean
   square per `(b, h, t)`. Leave `v` alone.
2. Rotate the *normalized* `q` and `k`. Take the even-indexed and odd-indexed
   components (`x[..., 0::2]` and `x[..., 1::2]`, each `(B, H, T, d/2)`), apply the
   two-line rotation with `cos` and `sin` (which broadcast from the right), and put
   the two results back in alternating order.
3. `scores = q @ k^T / sqrt(d)`, hide the future with `-inf`, softmax over the last
   axis, multiply by `v`.

The order of steps 1 and 2 matters only because of the gains. A rotation does not
change the root mean square of a vector, but the gain multiplies each *dimension* by
its own number, and rotating mixes the two dimensions of a pair.

---

The normalization and the rotation, for a tensor `x` of shape `(B, H, T, d)`:

$$
\hat{x} = \frac{x}{\sqrt{\operatorname{mean}_{\text{last axis}}(x^2) + \varepsilon}} \odot g,
\qquad
\tilde{x}_{2i} = \hat{x}_{2i} \cos - \hat{x}_{2i+1} \sin, \quad
\tilde{x}_{2i+1} = \hat{x}_{2i} \sin + \hat{x}_{2i+1} \cos .
$$

To interleave the two rotated halves, stack them on a new last axis (shape
`(B, H, T, d/2, 2)`) and reshape to `(B, H, T, d)`. Then

$$
y = \operatorname{softmax}\!\Big(\frac{\tilde{q}\,\tilde{k}^\top}{\sqrt{d}} + M\Big)\, v,
$$

with $M$ equal to $-\infty$ above the diagonal and 0 elsewhere. Check yourself on the
example: both normalized keys must come out with similar lengths, $[2.83, 0]$ and
$[2, 1]$, although the raw keys differ by a factor of twenty.
