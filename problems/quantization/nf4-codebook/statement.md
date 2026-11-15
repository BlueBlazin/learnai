Implement NF4, the 4-bit data type of QLoRA: sixteen levels placed at quantiles of a
normal distribution instead of at equal distances, applied block by block after each
block is divided by its largest magnitude. The codes are indices into a table, so the
quantizer is a nearest-neighbor search and the dequantizer a lookup; the table itself
has to contain an exact zero, which is why it is not symmetric.

## Specification

```python
def nf4_levels(): ...  # -> the 16 levels, ascending
def nf4_quantize(w, block_size): ...  # -> (codes, absmax)
def nf4_dequantize(codes, absmax, block_size): ...  # -> float32 weights
```

**`nf4_levels()`** returns the table as a float32 tensor of shape `(16,)`, in ascending
order. With $\Phi^{-1}$ the inverse of the standard normal distribution function and
$\delta = 0.9677083$:

- **Eight positive levels**: $\Phi^{-1}(p)$ at the 8 probabilities
  $p = \delta - k \cdot \dfrac{\delta - 0.5}{8}$, $k = 0, \dots, 7$.
- **Seven negative levels**: $-\Phi^{-1}(p)$ at the 7 probabilities
  $p = \delta - k \cdot \dfrac{\delta - 0.5}{7}$, $k = 0, \dots, 6$.
- **Zero.**

The 16 numbers are sorted and divided by the largest of them, $\Phi^{-1}(\delta)
\approx 1.848$, so that they span $[-1, 1]$. The result is the table that
bitsandbytes ships:

| code | level | code | level |
|---|---|---|---|
| 0 | $-1.0$ | 8 | $0.0795803$ |
| 1 | $-0.6961928$ | 9 | $0.1609302$ |
| 2 | $-0.5250731$ | 10 | $0.2461123$ |
| 3 | $-0.3949175$ | 11 | $0.3379152$ |
| 4 | $-0.2844414$ | 12 | $0.4407098$ |
| 5 | $-0.1847734$ | 13 | $0.5626170$ |
| 6 | $-0.0910500$ | 14 | $0.7229568$ |
| 7 | $0.0$ | 15 | $1.0$ |

Level 7 must be **exactly** $0.0$. The other levels are compared with this table to
the tolerance given below, so a function that derives them and a function that returns
them as constants are both accepted.

**`nf4_quantize(w, block_size)`**. `w` is a weight matrix of shape `(out, in)`. Each
row is cut into `in / block_size` consecutive blocks of `block_size` inputs; block $b$
of row $i$ holds $w_{ij}$ for $b \cdot \text{block\_size} \le j < (b + 1) \cdot
\text{block\_size}$. For every block:

$$
\text{absmax}_{ib} = \begin{cases}
\max_{j \in b} |w_{ij}| & \text{if that maximum is positive}, \\
1 & \text{if the block is all zeros},
\end{cases}
\qquad
\text{code}_{ij} = \operatorname*{arg\,min}_{k \in \{0, \dots, 15\}} \Big| \frac{w_{ij}}{\text{absmax}_{ib}} - \ell_k \Big| ,
$$

where $\ell_k$ is level $k$: the code is the index of the level **nearest in value**
to the normalized weight. It returns the pair `(codes, absmax)`: `codes` as a
**uint8** tensor of shape `(out, in)` with values 0 to 15, and `absmax` as a float32
tensor of shape `(out, in / block_size)`.

No test contains a normalized weight within $0.001$ of the midpoint between two
adjacent levels, so the search has one answer and float32 noise cannot change it.

**`nf4_dequantize(codes, absmax, block_size)`** returns the float32 tensor of shape
`(out, in)` with

$$
\hat w_{ij} = \ell_{\,\text{code}_{ij}} \cdot \text{absmax}_{ib}, \qquad b = \lfloor j / \text{block\_size} \rfloor .
$$

`codes` is a uint8 tensor with values 0 to 15 and `absmax` a float32 tensor, shaped as
`nf4_quantize` returns them; they need not come from `nf4_quantize`.

## Shapes

| name | shape | dtype |
|---|---|---|
| result of `nf4_levels` | `(16,)` | float32 |
| `w` | `(out, in)` | float32 |
| `block_size` | Python `int` that divides `in` | |
| `codes` | `(out, in)` | uint8 |
| `absmax` | `(out, in / block_size)` | float32 |
| result of `nf4_dequantize` | `(out, in)` | float32 |

## Example

```python
w = [
    [0.0, 0.1, -0.25, 0.52, -1.0, 0.33, 0.75, -0.6],
    [0.01, -0.04, 0.0, 0.018, -0.013, 0.033, -0.003, 0.004],
]

codes, absmax = nf4_quantize(w, block_size=8)
# absmax = [[1.0], [0.04]]
# codes = [[ 7, 8, 4, 13, 0, 11, 14, 2],
#          [10, 0, 7, 12, 4, 14,  6, 8]]

nf4_dequantize(codes, absmax, 8)
# [[0.0,      0.079580, -0.284441, 0.562617, -1.0,      0.337915,  0.722957, -0.525073],
#  [0.009844, -0.04,     0.0,      0.017628, -0.011378, 0.028918, -0.003642,  0.003183]]
```

Row 0 is one block with absmax $1.0$, so the weights are already normalized. $0.1$
lies between levels 8 ($0.0796$) and 9 ($0.1609$) and is closer to the first. $-0.6$
lies between levels 1 ($-0.6962$) and 2 ($-0.5251$): the distances are $0.096$ and
$0.075$, so the code is 2. The largest magnitude, $-1.0$, lands exactly on level 0,
and $0.0$ exactly on level 7.

Row 1 is fifty times smaller. Its absmax is $0.04$, set by $-0.04$, and after the
division its weights use the table just as row 0 does: $0.01 / 0.04 = 0.25$ is nearest
to level 10 ($0.2461$) and comes back as $0.2461 \cdot 0.04 = 0.009844$.

The levels are dense near zero and sparse near $\pm 1$: the gap between levels 7 and 8
is $0.080$, the gap between levels 14 and 15 is $0.277$.

## Constraints

- $1 \le \text{out} \le 64$, $1 \le \text{in} \le 1024$, and `block_size` divides
  `in`. Every weight is finite with $|w_{ij}| \le 100$, and a block that is not all
  zeros has an absmax of at least $10^{-6}$.
- The levels, `absmax`, and the dequantized weights are compared to a relative
  tolerance of `1e-4` and an absolute tolerance of `1e-6`. Level 7 and every weight
  dequantized from code 7 must be exactly $0.0$.
- `codes` are compared **exactly** and must have dtype uint8.
- No Python loops over elements, blocks, or rows are needed.

## Why this matters

Weights of a trained network are distributed roughly like a normal: most are small
and few are large. A uniform 4-bit grid spends its 16 codes evenly over the range and
therefore mostly on values that rarely occur. NF4 places the levels so that each of
them is used about equally often by normally distributed weights, which is why 4-bit
models loaded through bitsandbytes (`load_in_4bit`, `bnb_4bit_quant_type="nf4"`) lose
less than uniform 4-bit rounding would. The format is a table and a per-block scale,
and each has a way to go wrong that still produces a model that loads. A table without
an exact zero turns every zero weight, and every padding row, into a small nonzero
number. A code treated as a position on a uniform grid, instead of an index into the
table, decodes to the wrong value for all but a few codes. And a scale taken from the
standard deviation, which the "normal" in the name suggests, leaves the largest
weights of every block outside $[-1, 1]$, where the table ends.
