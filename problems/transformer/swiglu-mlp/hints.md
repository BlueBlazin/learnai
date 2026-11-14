Follow one vector through the layer and write down its width at every step: `d` going
in, `h` on each of the two branches, `h` after they are multiplied, `d_out` coming out.
The weights are stored as `(out, in)`. For a batch of row vectors `x` of shape
`(..., d)`, which product turns width `d` into width `h`: `x @ w_gate` or
`x @ w_gate.T`? Which of the two would even run for `d = 6`, `h = 10`?

---

The layer is four operations:

1. `gate = x @ w_gate.T` and `up = x @ w_up.T`, both of shape `(..., h)`. The two
   branches read the same input; neither feeds the other.
2. Apply SiLU to `gate`, and to nothing else.
3. Multiply the activated gate and `up` elementwise. This is the only place where the
   two branches meet.
4. Project the product with `w_down.T`.

SiLU is $z \cdot \sigma(z)$: your framework has it (`silu`), or build it from
`sigmoid`. Do not confuse it with the plain sigmoid or with GELU; both have a similar
curve and give different numbers.

---

The whole function:

$$
y = \big(\operatorname{silu}(x\,W_{\text{gate}}^\top) \odot (x\,W_{\text{up}}^\top)\big)\, W_{\text{down}}^\top .
$$

Check it against the example: with $g = [1, -2, 3]$ and $u = [3, 2, 1]$,
$\operatorname{silu}(g) \odot u = [0.7311 \cdot 3,\; -0.2384 \cdot 2,\; 2.8577 \cdot 1]$.
If your second entry is $-2 \cdot \operatorname{silu}(2) = -3.52$, the activation is on
the wrong branch; if it is $0.1192 \cdot 2$, you used the sigmoid without the factor
$z$.
