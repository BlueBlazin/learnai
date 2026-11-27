Trace one vector through the layer and write down its width at each step. It enters
with width `d`, expands to width `h` on each of the two branches, stays at width `h`
after elementwise multiplication, and leaves with width `d_out`. The weights are stored
in `(out, in)` layout. For a batch of row vectors `x` of shape `(..., d)`, which
product maps width `d` to width `h`, `x @ w_gate` or `x @ w_gate.T`? Check which one
even has valid shapes when `d = 6` and `h = 10`.

---

The layer consists of four operations:

1. Compute `gate = x @ w_gate.T` and `up = x @ w_up.T`, both of shape `(..., h)`. Both
   branches read the same input `x`, and neither feeds into the other.
2. Apply SiLU to `gate` only.
3. Multiply the activated gate and `up` elementwise. This is the only place where the
   two branches combine.
4. Project the product down with `w_down.T`.

SiLU is $z \cdot \sigma(z)$. Use your framework's built-in `silu`, or compute it from
`sigmoid`. Do not substitute plain sigmoid or GELU, since both look similar but produce
different numbers.

---

The complete formula for row vectors is:

$$
y = \big(\operatorname{silu}(x\,W_{\text{gate}}^\top) \odot (x\,W_{\text{up}}^\top)\big)\, W_{\text{down}}^\top .
$$

Check your implementation on the worked example. With $g = [1, -2, 3]$ and
$u = [3, 2, 1]$, the gated hidden vector is
$\operatorname{silu}(g) \odot u = [0.7311 \cdot 3,\; -0.2384 \cdot 2,\; 2.8577 \cdot 1]$.
If your second entry is $-2 \cdot \operatorname{silu}(2) = -3.52$, you put the
activation on the wrong branch. If it is $0.1192 \cdot 2$, you used plain sigmoid
without multiplying by $z$.
