Implement the SwiGLU feed-forward layer, the MLP used in every block of Llama, Qwen,
Granite, and most other modern decoder-only models. It uses three weight matrices
instead of two, and applies the activation function to only one of its two parallel
branches.

![This problem is the SwiGLU MLP, the feed-forward layer of every block.](assets/map.svg)

## Specification

```python
def swiglu(x, w_gate, w_up, w_down): ...
```

For a single input vector $x \in \mathbb{R}^d$ (treated as a column vector), the layer
computes an output vector $y \in \mathbb{R}^{d_{\text{out}}}$ in three steps using
hidden width $h$:

1. **Project** into two parallel branches of hidden width $h$:

$$
g = W_{\text{gate}}\, x, \qquad u = W_{\text{up}}\, x .
$$

2. **Gate** by applying SiLU to the gate branch $g$ only, and multiply the result
   elementwise by the up branch $u$:

$$
m = \operatorname{silu}(g) \odot u, \qquad \operatorname{silu}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}} ,
$$

   where $\sigma$ is the logistic sigmoid function.

3. **Project down** to the output width $d_{\text{out}}$:

$$
y = W_{\text{down}}\, m .
$$

Combined into one expression, $y = W_{\text{down}} \big( \operatorname{silu}(W_{\text{gate}}\, x) \odot W_{\text{up}}\, x \big)$.
There are no bias terms.

![Only the gate branch passes through SiLU, and the two branches are then multiplied.](assets/swiglu.svg)

The weight matrices use `(out, in)` layout, matching `nn.Linear.weight` and standard
model checkpoints. Entry `w_gate[i, j]` multiplies input component `j` and contributes
to output component `i`.

The input `x` has one to three dimensions. The layer applies independently to every
vector along the last axis, treating all leading axes as batch axes. You may use your
framework's `silu` or `sigmoid`.

## Shapes

| name | shape | dtype |
|---|---|---|
| `x` | `(..., d)`, one to three dimensions | float32 |
| `w_gate` | `(h, d)` | float32 |
| `w_up` | `(h, d)` | float32 |
| `w_down` | `(d_out, h)` | float32 |
| result | `(..., d_out)` | float32 |

## Example

Consider one input vector with $d = 2$, $h = 3$, and $d_{\text{out}} = 2$:

```python
x = [1.0, 2.0]
w_gate = [[1.0, 0.0], [0.0, -1.0], [1.0, 1.0]]  # shape (3, 2)
w_up = [[1.0, 1.0], [2.0, 0.0], [0.0, 0.5]]  # shape (3, 2)
w_down = [[1.0, 0.0, 1.0], [0.0, 1.0, -1.0]]  # shape (2, 3)

swiglu(x, w_gate, w_up, w_down)  # [5.050899, -3.334534]
```

Step by step:

- Project into both branches to get $g = W_{\text{gate}}\,x = [1, -2, 3]$ and
  $u = W_{\text{up}}\,x = [3, 2, 1]$.
- Apply SiLU to $g$ to get $\operatorname{silu}(g) = [0.731059, -0.238406, 2.857723]$.
  A negative gate value does not zero out the unit because
  $\operatorname{silu}(-2) = -2\,\sigma(-2) = -0.2384$.
- Multiply elementwise to get $m = \operatorname{silu}(g) \odot u = [2.193176, -0.476812, 2.857723]$.
- Project down to get $y = W_{\text{down}}\,m = [2.193176 + 2.857723,\; -0.476812 - 2.857723] = [5.050899, -3.334534]$.

## Constraints

- Results are compared against the reference with a relative tolerance of `1e-4` and an
  absolute tolerance of `1e-5`. A NaN is always wrong.
- $1 \le d, h, d_{\text{out}} \le 64$, and `x` has at most 4,096 elements. The three
  widths are independent, so the weights need not be square and $d_{\text{out}}$ need
  not equal $d$.
- The entries of $g$ and $u$ lie in $[-100, 100]$. At the ends of this range the gate
  saturates, with $\operatorname{silu}(z) \approx z$ for large positive $z$ and
  $\operatorname{silu}(z) \approx 0$ for very negative $z$. Your output must remain
  finite and accurate there.
- No Python loops over vectors or components are needed.

## Why this matters

The feed-forward layer holds about two thirds of a decoder's parameters and runs once
per layer for every token, making it a major share of per-token compute. Checkpoints
store its three matrices under names like `gate_proj`, `up_proj`, and `down_proj`,
without recording which branch gets the activation or which activation function was
used.

Because `gate_proj` and `up_proj` have the same shape, swapping them raises no error.
Loading a GELU-gated model like Gemma with SiLU code also runs without failing, and the
model still generates fluent text at degraded quality. Matching the exact formula is the
only way to catch these silent bugs.
