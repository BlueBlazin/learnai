Encode and decode weights in the layout of GGUF's `Q4_K`, the 4-bit "K-quant" of
llama.cpp: 256 weights form a superblock of 8 blocks of 32, every weight is a 4-bit
code on its block's grid, and the 8 block scales and 8 block minimums are themselves
quantized to 6 bits against two float16 numbers per superblock. The decoder is two
multiplications and a subtraction. The encoder is where the work is: a grid whose
bottom can never be positive, two rounding steps on the block parameters, and codes
that must be computed against the grid the decoder will reconstruct, not the one
you wanted.

## Specification

```python
def q4k_encode(w): ...  # -> (codes, scales, mins, d, dmin)
def q4k_decode(codes, scales, mins, d, dmin): ...  # -> float32 weights
```

`w` has shape `(S, 256)`: `S` superblocks. Superblock $s$ is cut into 8 consecutive
blocks of 32 weights; block $b$ holds $w_{s,k}$ for $32 b \le k < 32 (b + 1)$.

**The format.** A superblock stores

| field | shape | dtype | meaning |
|---|---|---|---|
| `codes` | `(S, 256)` | uint8, values 0..15 | 4 bits per weight |
| `scales` | `(S, 8)` | uint8, values 0..63 | 6 bits per block: the block's step, in units of `d` |
| `mins` | `(S, 8)` | uint8, values 0..63 | 6 bits per block: the block's offset, in units of `dmin` |
| `d` | `(S,)` | float16 | the unit of the steps |
| `dmin` | `(S,)` | float16 | the unit of the offsets |

That is $256 \cdot 4 + 8 \cdot 6 + 8 \cdot 6 + 2 \cdot 16 = 1152$ bits for 256
weights: 4.5 bits per weight. This problem keeps each field in a tensor of its own;
llama.cpp packs the same fields into 144 bytes.

**`q4k_decode`.** With `d` and `dmin` widened to float32, the weight at position $k$
of block $b$ is

$$
\hat w = (d \cdot \text{scales}_b) \cdot \text{code}_k \;-\; (d_{\min} \cdot \text{mins}_b),
$$

evaluated in float32 in that order. The offset is **subtracted**, and it is never
negative, so the lowest value of a block's grid, $-d_{\min} \cdot \text{mins}_b$, is
zero or negative. The fields need not come from `q4k_encode`.

**`q4k_encode`**, for every superblock separately:

1. **Block grids.** For each block, with its smallest and largest weight:
   $$
   lo_b = \min\big(\min_k w_k,\ 0\big), \qquad
   \text{scale}_b = \frac{\max_k w_k - lo_b}{15}, \qquad
   \text{min}_b = -lo_b \;\ge\; 0 .
   $$
   The minimum is clamped to be non-positive because the format cannot store a
   positive one. A block of positive weights therefore gets a grid that starts at 0.
2. **Units.** $\text{max\_scale} = \max_b \text{scale}_b$ and
   $\text{max\_min} = \max_b \text{min}_b$ over the 8 blocks, and
   $$
   d = \operatorname{float16}\!\Big(\frac{\text{max\_scale}}{63}\Big), \qquad
   d_{\min} = \operatorname{float16}\!\Big(\frac{\text{max\_min}}{63}\Big),
   $$
   computed in float32 and then rounded to the nearest float16.
3. **6-bit fields.**
   $$
   \text{scales}_b = \operatorname{round}\!\Big(\frac{63 \cdot \text{scale}_b}{\text{max\_scale}}\Big), \qquad
   \text{mins}_b = \operatorname{round}\!\Big(\frac{63 \cdot \text{min}_b}{\text{max\_min}}\Big),
   $$
   with $\text{scales}_b = 0$ for every block if $\text{max\_scale} = 0$, and
   $\text{mins}_b = 0$ for every block if $\text{max\_min} = 0$.
4. **Codes, against the stored grid.** Let $D_b = d \cdot \text{scales}_b$ and
   $M_b = d_{\min} \cdot \text{mins}_b$, computed in float32 from the **float16**
   values of step 2 and the integers of step 3: exactly what a decoder will use.
   Then
   $$
   \text{code}_k = \begin{cases}
   \operatorname{clamp}\!\Big(\operatorname{round}\!\Big(\dfrac{w_k + M_b}{D_b}\Big),\ 0,\ 15\Big) & D_b > 0, \\[1.5ex]
   0 & D_b = 0 .
   \end{cases}
   $$

It returns `(codes, scales, mins, d, dmin)` with the shapes and dtypes of the table.

$\operatorname{round}$ is rounding to the nearest integer, ties to even, as in
PyTorch, JAX, and MLX.

> [!NOTE]
> The decoder and the layout are llama.cpp's. The choice of each block's scale and
> minimum in step 1 is plain min-max; llama.cpp instead runs a weighted
> least-squares search per block (`make_qkx2_quants`), which this problem does not
> reproduce. Steps 2 to 4 follow its reference quantizer.

## Shapes

| name | shape | dtype |
|---|---|---|
| `w` | `(S, 256)` | float32 |
| `codes` | `(S, 256)` | uint8 |
| `scales`, `mins` | `(S, 8)` | uint8 |
| `d`, `dmin` | `(S,)` | float16 |
| result of `q4k_decode` | `(S, 256)` | float32 |

## Example

One superblock. Only the listed weights are non-zero; blocks 3 to 7 are all zeros.

```python
w[0, 0:4] = [-1.0, 0.25, 2.0, 0.66]  # block 0 (the rest of the block is 0)
w[0, 32:36] = [0.52, 1.0, 0.32, 0.75]  # block 1 (the rest of the block is 0)
w[0, 64:96] = -0.4  # block 2, constant

codes, scales, mins, d, dmin = q4k_encode(w)
# d      = [0.003173828125]          dmin = [0.015869140625]
# scales = [[63, 21, 0, 0, 0, 0, 0, 0]]
# mins   = [[63,  0, 25, 0, 0, 0, 0, 0]]
# codes[0, 0:4]   = [0,  6, 15,  8]
# codes[0, 32:36] = [8, 15,  5, 11]
# codes[0, 64:96] = 0

q4k_decode(codes, scales, mins, d, dmin)
# [0, 0:4]   = [-0.99976, 0.19995, 1.99951, 0.59985]
# [0, 32:36] = [ 0.53320, 0.99976, 0.33325, 0.73315]
# [0, 64:96] = -0.39673
```

**Step 1.** Block 0 spans $[-1, 2]$: scale $3 / 15 = 0.2$, minimum $1$. Block 1 has
only non-negative weights, so its $lo$ is clamped to $0$: scale $1 / 15$, minimum
$0$. Block 2 is constant at $-0.4$: its range is 0, so its scale is 0 and its
minimum $0.4$. The zero blocks have scale 0 and minimum 0.

**Steps 2 and 3.** $\text{max\_scale} = 0.2$ and $\text{max\_min} = 1$. The units are
$0.2 / 63 = 0.0031746$ and $1 / 63 = 0.015873$, which float16 rounds to
$0.0031738$ and $0.015869$. The fields: $63 \cdot (1/15) / 0.2 = 21$ for block 1,
and $63 \cdot 0.4 / 1 = 25.2 \to 25$ for the minimum of block 2.

**Step 4.** Block 0 is decoded with $D_0 = 63 d = 0.19995$ and $M_0 = 63 d_{\min} =
0.99976$, not with $0.2$ and $1$: the weight $0.66$ sits at
$(0.66 + 0.99976) / 0.19995 = 8.30$ and gets code 8. Block 2 has $D_2 = 0$: its
codes are 0, and every weight decodes to $-M_2 = -25\, d_{\min} = -0.39673$. The
constant block is carried by its 6-bit minimum alone and comes back off by $0.0033$.

## Constraints

- $1 \le S \le 16$. Entries of `w` are finite with $|w| \le 10^3$; a superblock that
  is not all zeros has a largest magnitude of at least $10^{-3}$.
- All five outputs of `q4k_encode` are compared **exactly**, including dtypes, and
  `d` and `dmin` bit for bit. So that float32 rounding cannot decide a result, in
  every test the two quotients of step 3 are at least $0.01$ from a tie, the two
  float16 conversions are unaffected by a relative change of $10^{-5}$, and no
  $(w_k + M_b) / D_b$ between 0 and 15 is within $0.02$ of a tie. One test about the
  float16 units lowers that last margin to $0.003$.
- `q4k_decode` is compared to a relative tolerance of `1e-5` plus `1e-6` of the
  largest decoded magnitude.
- **Error bound.** For weights encoded by `q4k_encode`, every decoded weight
  satisfies
  $$ |w_k - \hat w_k| \;\le\; \tfrac12 D_b + \tfrac{15}{2}\, d + \tfrac12\, d_{\min} $$
  up to 5 percent (the float16 rounding of the units): half a step of the block's
  grid, plus what the 6-bit rounding of the block's step and offset can move the
  ends of that grid.
- No Python loops over superblocks, blocks, or weights are needed.

## Why this matters

`Q4_K_M` is the file most people actually run: it is the default recommendation for
GGUF models, and its tensors are mostly `Q4_K` blocks. The format is a study in
where bits go. A flat group-wise scheme with a float16 scale and minimum per 32
weights costs 5 bits per weight; quantizing those 16 parameters to 6 bits each and
sharing two float16 units across 256 weights brings it to 4.5, and the saved half
bit is spent nowhere, the grid inside a block is the same. The price is a second
layer of rounding, and every bug of this problem lives there. Codes computed against
the ideal block scale decode on a slightly different grid. A positive block minimum
silently wraps in an unsigned field. A unit taken from the first block instead of
the largest saturates every bigger block at 63. None of these breaks the file
format, so the model loads, and only a perplexity measurement shows that the
quantizer threw away precision it had paid for.
