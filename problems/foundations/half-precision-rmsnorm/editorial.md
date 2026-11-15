## The idea

Float16 has 5 exponent bits. Its finite range ends at 65504, its normal range starts at
$2^{-14} \approx 6.1 \times 10^{-5}$, and below that only multiples of $2^{-24}$ exist.
Squaring halves the usable range on both sides: $x^2$ overflows for $|x| > 255.9$ and
drops below the normal range for $|x| < 2^{-7} \approx 7.8 \times 10^{-3}$. The inputs
and the output of RMSNorm are fine; the intermediate $x^2$ is not.

RMSNorm without epsilon is scale invariant: dividing $x$ by any $c > 0$ changes
nothing. With epsilon, the identity is

$$
\frac{x}{\sqrt{\operatorname{mean}(x^2) + \varepsilon}}
= \frac{x / c}{\sqrt{\operatorname{mean}\big((x/c)^2\big) + \varepsilon / c^2}} ,
$$

so we may choose $c$ per vector to put the squares where float16 is comfortable.

**Which $c$.** With $c = \max_j |x_j|$ every $|u_j| = |x_j| / c \le 1$ and the largest
is exactly 1. Hence $1/d \le \operatorname{mean}(u^2) \le 1$: no overflow, and the sum
is dominated by terms that are nowhere near the subnormals. Entries much smaller than
the maximum do lose bits when they are divided by $c$ or squared, but they are then
negligible in the mean; this is the one error the method keeps, discussed below.

**Epsilon.** $\varepsilon / c^2$ is the dangerous expression now. For $c = 10^{-4}$ and
$\varepsilon = 10^{-2}$ it is $10^{6}$: overflow. Worse, $c \cdot c$ underflows to zero
for $c < 2.4 \times 10^{-4}$, and $\varepsilon / 0$ is `inf`. Both are avoided by
bounding the scale from below by $\sqrt{\varepsilon}$:

$$
c = \max\!\big(\max_j |x_j|,\ \sqrt{\varepsilon}\big), \qquad
r = \frac{\sqrt{\varepsilon}}{c} \le 1, \qquad
y = \frac{u}{\sqrt{\operatorname{mean}(u^2) + r^2}} \cdot g .
$$

If the vector is larger than $\sqrt{\varepsilon}$, then $c$ is its maximum and
$\operatorname{mean}(u^2) \ge 1/d$. If it is smaller, then $c = \sqrt{\varepsilon}$ and
$r = 1$. Either way the radicand lies in $[1/d, 2]$. The all-zero vector gives $u = 0$
and $r = 1$, so $y = 0$, with no special case. And $\sqrt{\varepsilon}$, computed in
Python, is an ordinary float16 number for every epsilon in use ($10^{-3}$ for
$\varepsilon = 10^{-6}$), whereas $\varepsilon = 10^{-6}$ itself is a subnormal that
float16 stores with a 1.3% error.

## Reference solution (PyTorch)

```python
def rms_norm_f16(x, weight, eps=1e-5):
    root_eps = math.sqrt(eps)
    c = torch.clamp(x.abs().amax(dim=-1, keepdim=True), min=root_eps)
    u = x / c
    r = root_eps / c
    return u / torch.sqrt(u.square().mean(dim=-1, keepdim=True) + r * r) * weight
```

`root_eps` is a Python float, so `clamp(min=root_eps)` and `root_eps / c` stay float16.
The JAX and MLX versions are the same lines with `jnp.maximum` / `mx.maximum`.

## What accuracy to expect

Each float16 operation rounds with a relative error of up to $2^{-11} \approx 4.9
\times 10^{-4}$. The reference performs about six in sequence. Measured against the
exact result over several thousand random vectors of every scale from $5 \times
10^{-8}$ to 65504, with $d$ up to 512 and in all three frameworks, its worst error was
0.56 of the tolerance (`5e-3` relative plus `4e-6` absolute).

Two limits are worth knowing, because they are limits of float16 and not of this
implementation:

- **Dynamic range within one vector.** An entry more than $2^{14}$ times smaller than
  the vector's maximum becomes subnormal when divided by $c$ and keeps only a few bits.
  After the division by the root mean square its absolute error is at most about
  $\sqrt{d} \cdot 2^{-25}$ times the gain, which the absolute tolerance covers for
  $d \le 512$.
- **Long sums.** A float16 running sum of $d$ numbers in $[0, 1]$ stops being exact
  once it passes 2048, where the spacing is 2. What actually happens depends on the
  framework. On the versions used here (torch 2.15, JAX 0.12, MLX 0.32.4),
  `torch.mean` and `jnp.mean` of four float16 entries equal to 60000 return 60000
  although the float16 `sum` of the same entries is `inf` (JAX visibly converts to
  float32 inside `mean`; its traced program shows it), while `mx.mean` returns `inf`:
  MLX accumulates in float16. That is why this problem stops at $d = 512$. A float16
  kernel for $d$ in the thousands has to sum in blocks or carry a wider accumulator.

## Common mistakes

- **Squaring first.** `inf` above 255, and the output is `x / inf = 0`. Nothing raises.
- **Upcasting.** `x.float()`, compute, `.half()` gives the right numbers and is what
  several production implementations do (see below). It is ruled out here because the
  point is the technique for when you cannot or do not want to widen.
- **Copying the float32 recipe**, $s = \max(1, \max|x|)$. It only rescales large
  vectors. Small ones are squared as they are, and both their squares and epsilon sit
  in the subnormals: up to 2% error with $\varepsilon = 10^{-6}$.
- **Forgetting to rescale epsilon.** `mean(u^2) + eps` normalizes a different vector:
  for a vector of size $10^{-3}$ the correct epsilon term after rescaling is ten times
  the mean square, not $10^{-5}$ of it.
- **$\varepsilon / (s \cdot s)$ with $s = \max|x|$.** Overflows, or divides by an
  underflowed zero, for small vectors; and $s = 0$ for the zero vector gives `0 / 0`.
- **Rescaling by the mean of $|x|$.** The mean can be $d$ times smaller than the
  maximum (one massive activation among zeros), so an entry of $u$ can be as large as
  $d$, and $u^2$ overflows again once $d > 255$. Only the maximum bounds every entry.
- **Letting the dtype drift.** `x * torch.ones(1)` or `mx.maximum(c, mx.array(1.0))`
  silently produce float32, since a default-dtype tensor is not a scalar. Python floats
  do not widen.

## Complexity

One extra reduction (the maximum) and one extra division per vector compared with the
naive formula: $O(d)$ time, $O(d)$ extra memory, all in the storage type.

## In the wild

Meta's reference Llama code takes the other route: its `RMSNorm.forward` is
`self._norm(x.float()).type_as(x)`, that is, compute in float32 and cast back
(`llama/model.py` in `meta-llama/llama`). On the versions used here, the built-in
`torch.nn.functional.rms_norm` and `mx.fast.rms_norm` also return the correct result
for a float16 vector containing 60000, where the naive float16 formula returns zeros
(measured; how they get there is their business). Rescaling by the largest magnitude
before squaring is the classical way to compute a Euclidean norm without spurious
overflow or underflow, the approach of the reference BLAS routine `nrm2` and of
`hypot`. The same reasoning applies to bfloat16 with the roles reversed: its exponent
range is float32's, so nothing overflows, but it has only 8 bits of precision.
