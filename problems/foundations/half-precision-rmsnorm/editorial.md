## The idea

`float16` has 5 exponent bits. Its finite range ends at 65504, its normal range starts at
$2^{-14} \approx 6.1 \times 10^{-5}$, and below that only multiples of $2^{-24}$ exist.
Squaring halves the usable exponent range on both ends. Any entry with $|x| > 255.9$
overflows when squared, and any entry with $|x| < 2^{-7} \approx 7.8 \times 10^{-3}$
drops into the subnormals. The inputs and output of RMSNorm fit comfortably in `float16`,
but the intermediate $x^2$ does not.

Without epsilon, RMSNorm is scale-invariant, so dividing $x$ by any positive scalar $c$
leaves the output unchanged. With epsilon, dividing both numerator and denominator by $c$
gives

$$
\frac{x}{\sqrt{\operatorname{mean}(x^2) + \varepsilon}}
= \frac{x / c}{\sqrt{\operatorname{mean}\big((x/c)^2\big) + \varepsilon / c^2}} .
$$

We can therefore choose a per-vector scale $c > 0$ so that the rescaled vector $u = x / c$
stays well inside the normal range of `float16`.

**Choosing the scale.** Setting $c = \max_j |x_j|$ guarantees $|u_j| \le 1$ for every
entry, with the largest entry equal to $\pm 1$. As a result, $\operatorname{mean}(u^2)$
lies in $[1/d, 1]$, where $d$ is the vector length. Nothing overflows, and the dominant
terms in the mean sit far above the subnormals. Entries much smaller than the maximum do
lose precision when divided by $c$ or squared, but they contribute negligibly to the
mean.

**Handling epsilon.** After rescaling, $\varepsilon / c^2$ becomes the remaining hazard.
When $c = 10^{-4}$ and $\varepsilon = 10^{-2}$, $\varepsilon / c^2 = 10^{6}$ overflows.
Worse, $c \cdot c$ underflows to zero for $c < 2.4 \times 10^{-4}$, making
$\varepsilon / 0$ infinite. Bounding $c$ from below by $\sqrt{\varepsilon}$ prevents both
failures. Defining $r = \sqrt{\varepsilon} / c$ and multiplying by the gain $g$ gives

$$
c = \max\!\big(\max_j |x_j|,\ \sqrt{\varepsilon}\big), \qquad
r = \frac{\sqrt{\varepsilon}}{c} \le 1, \qquad
y = \frac{u}{\sqrt{\operatorname{mean}(u^2) + r^2}} \cdot g .
$$

When the vector's largest magnitude exceeds $\sqrt{\varepsilon}$, we have
$c = \max_j |x_j|$ and $\operatorname{mean}(u^2) \ge 1/d$. Otherwise,
$c = \sqrt{\varepsilon}$ and $r = 1$. In both cases, the expression under the square root
lies in $[1/d, 2]$. An all-zero vector yields $u = 0$ and $r = 1$, so $y = 0$ with no
special case. Computing $\sqrt{\varepsilon}$ in Python also avoids subnormal loss because
$\sqrt{10^{-6}} = 10^{-3}$ is a normal `float16`, whereas $10^{-6}$ itself is a subnormal
with a 1.3% representation error.

## Reference solution (PyTorch)

```python
def rms_norm_f16(x, weight, eps=1e-5):
    root_eps = math.sqrt(eps)
    c = torch.clamp(x.abs().amax(dim=-1, keepdim=True), min=root_eps)
    u = x / c
    r = root_eps / c
    return u / torch.sqrt(u.square().mean(dim=-1, keepdim=True) + r * r) * weight
```

Because `root_eps` is a Python float, `clamp(min=root_eps)` and `root_eps / c` remain
`float16`. The JAX and MLX implementations are identical, using `jnp.maximum` or
`mx.maximum` in place of `torch.clamp`.

## What accuracy to expect

Each `float16` operation rounds with a relative error of up to $2^{-11} \approx 4.9 \times 10^{-4}$,
and the reference solution performs about six operations in sequence. Across several
thousand random vectors spanning scales from $5 \times 10^{-8}$ to 65504 with $d \le 512$
in all three frameworks, its worst error is 0.56 of the tolerance (`5e-3` relative plus
`4e-6` absolute).

Two limits come from `float16` itself rather than this algorithm:

- **Dynamic range within one vector.** An entry more than $2^{14}$ times smaller than the
  vector's maximum becomes subnormal when divided by $c$ and keeps only a few bits. After
  dividing by the root mean square, its absolute error is at most about
  $\sqrt{d} \cdot 2^{-25}$ times the gain, which stays within the absolute tolerance for
  $d \le 512$.
- **Long sums.** A `float16` running sum of $d$ numbers in $[0, 1]$ loses unit precision
  once it passes 2048, where the spacing between adjacent `float16` values reaches 2.
  Framework behavior also differs here. On torch 2.15, JAX 0.12, and MLX 0.32.4,
  `torch.mean` and `jnp.mean` of four `float16` entries equal to 60000 return 60000 even
  though their `float16` `sum` is `inf` (JAX's traced program explicitly promotes to
  `float32` inside `mean`). By contrast, `mx.mean` returns `inf` because MLX accumulates
  in `float16`. This is why the problem caps $d$ at 512. A `float16` kernel for $d$ in
  the thousands must sum in blocks or use a wider accumulator.

## Common mistakes

- **Squaring first.** Any entry above 255 squares to `inf`, and `x / inf` silently
  returns `0` without raising an error.
- **Upcasting.** Computing `x.float()`, normalizing, and casting back with `.half()`
  produces accurate numbers and is common in production, but it violates the pure
  `float16` requirement of this problem.
- **Copying the `float32` recipe.** Using $s = \max(1, \max_j |x_j|)$ only rescales
  large vectors. Small vectors are still squared directly, pushing both their squares and
  epsilon into the subnormals and causing up to 2% error when $\varepsilon = 10^{-6}$.
- **Forgetting to rescale epsilon.** Computing `mean(u^2) + eps` changes the formula. For
  a vector of magnitude $10^{-3}$, the rescaled epsilon term should be ten times the mean
  square, not $10^{-5}$ of it.
- **Computing $\varepsilon / (s \cdot s)$ with $s = \max_j |x_j|$.** This overflows or
  divides by an underflowed zero on small vectors, and produces `0 / 0` on an all-zero
  vector.
- **Rescaling by the mean of $|x|$.** When one large activation sits among zeros, the
  mean of $|x|$ can be $d$ times smaller than the maximum. An entry of $u$ can then reach
  $d$, and $u^2$ overflows again once $d > 255$. Only the maximum bounds every entry.
- **Letting the dtype drift.** Expressions such as `x * torch.ones(1)` or
  `mx.maximum(c, mx.array(1.0))` silently promote to `float32` because a default-dtype
  tensor is not a Python scalar. Use Python floats instead.

## Complexity

Compared with the naive formula, rescaling adds one reduction (the maximum) and one
division per vector. Time complexity is $O(d)$ and extra memory is $O(d)$, all in
`float16`.

## In the wild

Meta's reference Llama implementation takes the upcasting route. In `llama/model.py` of
`meta-llama/llama`, `RMSNorm.forward` runs `self._norm(x.float()).type_as(x)`, computing
the norm in `float32` and casting back. On the framework versions tested here,
`torch.nn.functional.rms_norm` and `mx.fast.rms_norm` also return accurate results for a
`float16` vector containing 60000, where naive `float16` arithmetic returns zeros.

Rescaling by the largest magnitude before squaring is the classical way to compute a
Euclidean norm without spurious overflow or underflow. It is the same strategy used by
the reference BLAS routine `nrm2` and by `hypot`. In `bfloat16`, the tradeoff reverses.
Its exponent range matches `float32`, so squaring rarely overflows, but it only has 8 bits
of precision.
