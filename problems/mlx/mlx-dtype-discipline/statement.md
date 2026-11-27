Implement a transformer feed-forward block whose weights and activations stay in `float16` throughout. MLX silently promotes a `float16` array to `float32` whenever it interacts with a `float32` array. A single stray `float32` constant doubles the memory of every downstream activation while leaving the numerical values nearly unchanged.

## Specification

```python
def forward_f16(params, x): ...
```

`x` is a `float16` input array with `N` rows and `D` features. `params` is a dictionary of three `float16` arrays: the normalization gain `"gain"`, the up-projection weight `"w_in"`, and the down-projection weight `"w_out"`. Both weight matrices use `(out, in)` layout with hidden dimension `H`.

For each row $x \in \mathbb{R}^D$ of the input, let $g = \text{params}[\text{"gain"}]$, $W_{\text{in}} = \text{params}[\text{"w\_in"}]$, and $W_{\text{out}} = \text{params}[\text{"w\_out"}]$. Compute the output row $y \in \mathbb{R}^D$ in three steps:

1. **RMS normalization.** Divide the row by its root-mean-square magnitude and multiply elementwise ($\odot$) by $g$, using $\epsilon = 10^{-5}$ to prevent division by zero:

$$
n = \frac{x}{\sqrt{\frac{1}{D}\sum_{j=1}^{D} x_j^2 + \epsilon}} \odot g .
$$

2. **Up-projection and GELU.** Project $n$ to the hidden dimension $H$ and apply the tanh GELU approximation used by GPT-2 and Gemma:

$$
h = W_{\text{in}}\, n, \qquad
a = \tfrac{1}{2}\, h \left(1 + \tanh\!\left(\sqrt{2/\pi}\,\left(h + 0.044715\, h^3\right)\right)\right) .
$$

3. **Down-projection and residual connection.** Project $a$ back to dimension $D$ and add the original input row:

$$
y = x + W_{\text{out}}\, a .
$$

A row of `x` may be all zeros (for example, a padding position). Its output must be all zeros, with no `NaN`.

Two requirements apply beyond the mathematical formula:

- **Half precision throughout.** Both the returned array and the intermediate hidden activations of shape `(N, H)` must be `float16` while they are computed. Computing in `float32` and casting back to `float16` at the end does not satisfy this requirement and exceeds the memory limit under Constraints.
- **No `NaN` on zero rows.** The constant $\epsilon = 10^{-5}$ ensures that a zero row divides by a positive number. It must remain nonzero in `float16`, whose smallest positive value is about $6 \times 10^{-8}$.

The table below shows how MLX 0.32.4 chooses the output dtype when combining another operand with a `float16` array:

| operand combined with a `float16` array | result dtype |
|---|---|
| Python `float` or `int` (`0.5`, `math.sqrt(2)`) | `float16` |
| NumPy scalar or NumPy array of any float dtype | `float16` |
| integer or boolean `mx.array` (`mx.arange(4)`, `x > 0`) | `float16` |
| `mx.array(0.5)`, `mx.array(numpy_float64_array)` | **`float32`** |
| `mx.zeros(shape)`, `mx.ones(shape)`, `mx.arange(4.0)`, `mx.linspace(...)`, `mx.tri(...)` | **`float32`** |
| `mx.where(cond, 0.0, -math.inf)` (two Python scalars) | **`float32`** |

Array constructors accept a `dtype` argument, such as `mx.zeros(shape, dtype=x.dtype)`. Any array can be cast with `.astype`, and `mx.where(cond, x, 0.0)` stays `float16` because one branch is a `float16` array.

## Shapes

| name | shape | dtype |
|---|---|---|
| `params["gain"]` | `(D,)` | `float16` |
| `params["w_in"]` | `(H, D)` | `float16` |
| `params["w_out"]` | `(D, H)` | `float16` |
| `x` | `(N, D)` | `float16` |
| result | `(N, D)` | `float16` |

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

For row 0, the mean of squares is $(9 + 16)/2 = 12.5$, so the RMS is $3.5355$. Multiplying by the gain gives $n = [0.8485, 1.1314] \odot [1, 0.5] = [0.8485, 0.5657]$. Next, $h = [0.8485, 0.5657, 1.4142]$ and $a = [0.6804, 0.4040, 1.3028]$. Adding the down-projection to the input produces $y = [3 + 0.6804,\; 4 + 0.4040 - 1.3028] = [3.6804, 3.1012]$, which `float16` stores as `[3.680, 3.102]`.

For row 1, the mean of squares is 0, so the divisor is $\sqrt{10^{-5}} = 0.00316$. That yields $n = 0$ and $y = 0$. If $\epsilon$ underflowed to 0 in `float16`, this row would compute $0 / 0$ and produce `NaN`.

## Constraints

- $1 \le N \le 4096$, $1 \le D \le 512$, $1 \le H \le 2048$. Nonzero rows of `x` have an RMS between 0.05 and 16, and the test weights keep $|h| < 16$, so no value overflows `float16` (maximum 65,504).
- The returned array must be `float16`. Values are checked against a `float64` reference with relative tolerance `1e-2` and absolute tolerance `2e-2`, matching the roughly 3 significant digits of `float16`. Any `NaN` fails.
- **Memory limit: 45 MiB.** Two test cases run the block with 8 MiB `float16` hidden activations ($N = 4096, D = 256, H = 1024$ and $N = 2048, D = 512, H = 2048$) and measure peak MLX memory above the inputs using `mx.get_peak_memory`. The reference peaks at 30 MiB. A block whose hidden activations promote to `float32` peaks at 64 MiB or more regardless of its return dtype. These cases require an Apple GPU.
- No Python loops over rows are needed.

## Why this matters

Checkpoints on Apple silicon use `float16`, `bfloat16`, or quantized formats with `float16` scales because unified memory is tightly budgeted. A 3.66B-parameter model occupies 7.3 GB in half precision versus 14.6 GB in `float32`, and activations and the KV cache scale by the same factor.

A stray `float32` constant raises no error and barely alters the output values, yet it doubles the memory of every downstream activation and halves effective memory bandwidth. If the promotion reaches the KV cache, the model pays that 2× penalty for the rest of the conversation. The opposite habit from `float32` code is equally silent. An epsilon of `1e-8` rounds to `0` in `float16`, removing the zero-division guard until a padding row arrives and injects `NaN`.
