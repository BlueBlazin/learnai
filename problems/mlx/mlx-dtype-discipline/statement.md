Implement the feed-forward block of a transformer whose weights are float16, so that
**every array it computes is float16**. Nothing in MLX stops a float32 value from
entering the graph: one constant created the wrong way promotes everything after it,
the numbers stay right, and the block quietly takes twice the memory.

## Specification

```python
def forward_f16(params, x): ...
```

`params` is a dict of float16 arrays: `"gain"` of shape `(D,)`, `"w_in"` of shape
`(H, D)`, and `"w_out"` of shape `(D, H)`, both weights in `(out, in)` layout. For
each row $x \in \mathbb{R}^D$ of the input:

1. **RMS norm with a gain.** With $\epsilon = 10^{-5}$ and $g$ the gain,

$$
n = \frac{x}{\sqrt{\frac{1}{D}\sum_{j} x_j^2 + \epsilon}} \odot g .
$$

2. **Up-projection and GELU**, in the tanh form used by GPT-2 and Gemma:

$$
h = W_{in}\, n, \qquad
a = \tfrac{1}{2}\, h \left(1 + \tanh\!\left(\sqrt{2/\pi}\,\left(h + 0.044715\, h^3\right)\right)\right) .
$$

3. **Down-projection and residual.**

$$
y = x + W_{out}\, a .
$$

- `x`: a float16 array of shape `(N, D)`. A row may be all zeros (a padding position);
  its output is all zeros.
- Returns `y`: a **float16** array of shape `(N, D)`.

Two requirements beyond the formula:

- **Half precision throughout.** The result is float16, and the hidden activations
  (everything of shape `(N, H)`) are float16 while they are computed. Casting a
  float32 result to float16 at the end does not meet this, and the judge measures it:
  see the memory limit under Constraints.
- **No NaN.** $\epsilon$ exists so that a zero row divides by a positive number. It
  has to survive in float16, whose smallest positive value is about $6 \times 10^{-8}$.

How MLX 0.32.4 decides the dtype of a result:

| operand combined with a float16 array | result |
|---|---|
| a Python `float` or `int` (`0.5`, `math.sqrt(2)`) | float16 |
| a NumPy scalar or NumPy array, of any float dtype | float16 |
| an integer or bool `mx.array` (`mx.arange(4)`, `x > 0`) | float16 |
| `mx.array(0.5)`, `mx.array(numpy_float64_array)` | **float32** |
| `mx.zeros(shape)`, `mx.ones(shape)`, `mx.arange(4.0)`, `mx.linspace(...)`, `mx.tri(...)` | **float32** |
| `mx.where(cond, 0.0, -math.inf)`: two Python scalars | **float32** |

The constructors take a `dtype` argument (`mx.zeros(shape, dtype=x.dtype)`), any
array can be cast with `.astype`, and `mx.where(cond, x, 0.0)` with one float16 branch
stays float16.

## Shapes

| name | shape | dtype |
|---|---|---|
| `params["gain"]` | `(D,)` | float16 |
| `params["w_in"]` | `(H, D)` | float16 |
| `params["w_out"]` | `(D, H)` | float16 |
| `x` | `(N, D)` | float16 |
| result | `(N, D)` | float16 |

## Example

```python
params = {
    "gain": [1.0, 0.5],
    "w_in": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],  # H = 3, D = 2
    "w_out": [[1.0, 0.0, 0.0], [0.0, 1.0, -1.0]],
}
x = [[3.0, 4.0], [0.0, 0.0]]

forward_f16(params, x)
# [[3.680, 3.102],
#  [0.000, 0.000]]        dtype float16
```

Row 0: the mean of squares is $(9 + 16)/2 = 12.5$, the RMS is $3.5355$, and
$n = [0.8485, 1.1314] \odot [1, 0.5] = [0.8485, 0.5657]$. Then
$h = [0.8485, 0.5657, 1.4142]$, $a = [0.6804, 0.4040, 1.3028]$, and
$y = [3 + 0.6804,\; 4 + 0.4040 - 1.3028] = [3.6804, 3.1012]$, which float16 stores as
`[3.680, 3.102]`.

Row 1: the mean of squares is 0, the divisor is $\sqrt{10^{-5}} = 0.00316$, and
$n = 0$, so $y = 0$. With an $\epsilon$ that is 0 in float16 the row would be
$0 / 0$.

## Constraints

- $1 \le N \le 4096$, $1 \le D \le 512$, $1 \le H \le 2048$. Nonzero rows of `x` have
  an RMS between 0.05 and 16, and the test weights keep $|h| < 16$, so nothing
  overflows float16 (largest value 65,504).
- The result must be float16. Its values are compared with a float64 computation to a
  relative tolerance of `1e-2` and an absolute tolerance of `2e-2`, which is what
  float16 arithmetic (about 3 significant digits) can deliver here. A NaN is always
  wrong.
- **Memory limit: 45 MiB.** Two cases run the block with hidden activations of 8 MiB
  in float16 ($N = 4096$, $D = 256$, $H = 1024$, and $N = 2048$, $D = 512$,
  $H = 2048$) and measure the peak MLX memory above the inputs
  (`mx.get_peak_memory`). The reference peaks at 30 MiB; a block whose hidden
  activations are float32 peaks at 64 MiB or more, whatever dtype it returns. These
  cases need the Apple GPU.
- No Python loops over rows are needed.

## Why this matters

Checkpoints for Apple silicon are float16 or bfloat16 (or quantized, with float16
scales), because unified memory is the budget: a 3.66 B-parameter model is 7.3 GB in
half precision and 14.6 GB in float32, and activations and the KV cache scale the same
way. A stray float32 constant does not raise and does not change the output beyond
rounding. It doubles the memory of every activation downstream and moves twice the
bytes through every kernel, and if it reaches the KV cache it doubles that for the
rest of the conversation. The opposite habit carried over from float32 code is as
quiet: an epsilon of `1e-8` is exactly zero in float16, so the guard it was written to
be is not there on the day a padding row arrives.
