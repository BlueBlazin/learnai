Implement the SwiGLU feed-forward layer, the MLP inside every block of Llama, Qwen,
Granite, and most other current decoder-only models. It has three weight matrices where
the classic MLP has two, and the activation goes on exactly one of its two parallel
branches; putting it anywhere else still returns a tensor of the right shape.

## Specification

```python
def swiglu(x, w_gate, w_up, w_down): ...
```

For one input vector $x \in \mathbb{R}^d$ (a column vector):

1. **Two parallel projections** into the hidden width $h$:

$$
g = W_{\text{gate}}\, x, \qquad u = W_{\text{up}}\, x .
$$

2. **Gate.** Apply SiLU to the gate branch only, and multiply the two branches
   elementwise:

$$
m = \operatorname{silu}(g) \odot u, \qquad \operatorname{silu}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}} .
$$

3. **Project down:**

$$
y = W_{\text{down}}\, m .
$$

In one line: $y = W_{\text{down}} \big( \operatorname{silu}(W_{\text{gate}}\, x) \odot W_{\text{up}}\, x \big)$.
There are no biases.

The weights are stored in `(out, in)` layout, the layout of `nn.Linear.weight` and of
model checkpoints: `w_gate[i, j]` multiplies component `j` of the input and contributes
to component `i` of the output.

- `x`: float32, shape `(..., d)`, with one to three dimensions. The layer is applied
  independently to every vector along the last axis; all other axes are batch axes.
- `w_gate`, `w_up`: float32, shape `(h, d)`.
- `w_down`: float32, shape `(d_out, h)`.
- Returns float32, shape `(..., d_out)`.

You may use your framework's `silu` or `sigmoid`.

## Shapes

| name | shape | dtype |
|---|---|---|
| `x` | `(..., d)`, one to three dimensions | float32 |
| `w_gate` | `(h, d)` | float32 |
| `w_up` | `(h, d)` | float32 |
| `w_down` | `(d_out, h)` | float32 |
| result | `(..., d_out)` | float32 |

## Example

One vector with $d = 2$, $h = 3$, $d_{out} = 2$:

```python
x = [1.0, 2.0]
w_gate = [[1.0, 0.0], [0.0, -1.0], [1.0, 1.0]]  # shape (3, 2)
w_up = [[1.0, 1.0], [2.0, 0.0], [0.0, 0.5]]  # shape (3, 2)
w_down = [[1.0, 0.0, 1.0], [0.0, 1.0, -1.0]]  # shape (2, 3)

swiglu(x, w_gate, w_up, w_down)  # [5.050899, -3.334534]
```

Step by step:

- $g = W_{\text{gate}}\,x = [1, -2, 3]$ and $u = W_{\text{up}}\,x = [3, 2, 1]$.
- $\operatorname{silu}(g) = [0.731059, -0.238406, 2.857723]$. A negative gate value does
  not close the gate completely: $\operatorname{silu}(-2) = -2\,\sigma(-2) = -0.2384$.
- $m = \operatorname{silu}(g) \odot u = [2.193176, -0.476812, 2.857723]$.
- $y = W_{\text{down}}\,m = [2.193176 + 2.857723,\; -0.476812 - 2.857723] = [5.050899, -3.334534]$.

## Constraints

- Results are compared with the reference to a relative tolerance of `1e-4` and an
  absolute tolerance of `1e-5`. A NaN is always wrong.
- $1 \le d, h, d_{out} \le 64$, and `x` has at most 4,096 elements. The three widths
  are independent: the weights are not necessarily square, and $d_{out}$ is not
  necessarily equal to $d$.
- The entries of $g$ and $u$ lie in $[-100, 100]$. At the ends of that range the gate is
  saturated: $\operatorname{silu}(z)$ is $z$ for large $z$ and $0$ for very negative $z$,
  and the result must still be finite and correct there.
- No Python loops over vectors or components are needed.

## Why this matters

The feed-forward layer holds about two thirds of a decoder's parameters and runs once
per layer for every token, so it is a large share of the time spent on each generated
token. A checkpoint stores its three matrices under names such as `gate_proj`, `up_proj`, and
`down_proj`, and nothing in the file says which branch carries the activation or which
activation it is. The two hidden matrices have the same shape, so swapping them, or
loading a GELU-gated model (Gemma) with SiLU code, raises no error: the model runs and
produces fluent text that is worse than it should be. The only defense is to match the
formula exactly, which is what this problem checks.
