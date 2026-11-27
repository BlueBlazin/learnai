Implement `q4k_encode` and `q4k_decode` for GGUF's `Q4_K` 4-bit quantization format from llama.cpp. Each superblock of 256 weights is split into 8 blocks of 32 weights, where every weight gets a 4-bit code and each block's scale and minimum are quantized to 6 bits against two float16 superblock units.

## Specification

```python
def q4k_encode(w): ...  # -> (codes, scales, mins, d, dmin)
def q4k_decode(codes, scales, mins, d, dmin): ...  # -> float32 weights
```

The input `w` holds $S$ superblocks of 256 weights. Each superblock $s$ is split into 8 consecutive blocks of 32 weights, where block $b \in \{0, \dots, 7\}$ holds weights $w_{s,k}$ for $32b \le k < 32(b + 1)$.

A `Q4_K` superblock stores five fields:

- `codes` holds 4-bit integers in $[0, 15]$, one per weight.
- `scales` holds 6-bit integers in $[0, 63]$, one per block, giving each block's step size in units of `d`.
- `mins` holds 6-bit integers in $[0, 63]$, one per block, giving each block's offset in units of `dmin`.
- `d` and `dmin` are float16 scalars per superblock that set the units for `scales` and `mins`.

Together, these fields use $256 \cdot 4 + 8 \cdot 6 + 8 \cdot 6 + 2 \cdot 16 = 1152$ bits per 256 weights, or 4.5 bits per weight. This problem stores each field in its own tensor rather than bit-packing them into llama.cpp's 144-byte struct.

### `q4k_decode(codes, scales, mins, d, dmin)`

Convert all inputs to float32. For each superblock $s$, block $b$, and weight position $k$ in block $b$, compute the decoded weight in float32 in this order:

$$
\hat w = (d \cdot \text{scales}_b) \cdot \text{code}_k \;-\; (d_{\min} \cdot \text{mins}_b)
$$

Because $d_{\min}$ and $\text{mins}_b$ are non-negative and the offset is subtracted, the lowest point on any block's grid, $-d_{\min} \cdot \text{mins}_b$, is always zero or negative. Inputs to `q4k_decode` do not have to come from `q4k_encode`.

### `q4k_encode(w)`

Process each superblock independently in four steps.

1. **Block grids.** For each block $b$ of 32 weights $w_k$, compute its lower bound, ideal step size, and non-negative offset:
   $$
   lo_b = \min\big(\min_k w_k,\ 0\big), \qquad
   \text{scale}_b = \frac{\max_k w_k - lo_b}{15}, \qquad
   \text{min}_b = -lo_b \;\ge\; 0 .
   $$
   Clamping $lo_b$ to at most $0$ is required because the format cannot store a positive grid minimum. A block containing only positive weights therefore uses a grid starting at $0$.

2. **Superblock units.** Take $\text{max\_scale} = \max_b \text{scale}_b$ and $\text{max\_min} = \max_b \text{min}_b$ over the 8 blocks. Compute the superblock units in float32 and round them to the nearest float16:
   $$
   d = \operatorname{float16}\!\Big(\frac{\text{max\_scale}}{63}\Big), \qquad
   d_{\min} = \operatorname{float16}\!\Big(\frac{\text{max\_min}}{63}\Big) .
   $$

3. **6-bit block fields.** Quantize the per-block scales and minimums to integers in $[0, 63]$:
   $$
   \text{scales}_b = \operatorname{round}\!\Big(\frac{63 \cdot \text{scale}_b}{\text{max\_scale}}\Big), \qquad
   \text{mins}_b = \operatorname{round}\!\Big(\frac{63 \cdot \text{min}_b}{\text{max\_min}}\Big) .
   $$
   If $\text{max\_scale} = 0$, set $\text{scales}_b = 0$ for all 8 blocks. If $\text{max\_min} = 0$, set $\text{mins}_b = 0$ for all 8 blocks.

4. **4-bit weight codes.** Reconstruct each block's actual step $D_b = d \cdot \text{scales}_b$ and offset $M_b = d_{\min} \cdot \text{mins}_b$ in float32 using the **float16** values of `d` and `dmin` from step 2 and the integers from step 3. This matches the exact grid `q4k_decode` will reconstruct. Then quantize every weight in block $b$ against that grid:
   $$
   \text{code}_k = \begin{cases}
   \operatorname{clamp}\!\Big(\operatorname{round}\!\Big(\dfrac{w_k + M_b}{D_b}\Big),\ 0,\ 15\Big) & D_b > 0, \\[1.5ex]
   0 & D_b = 0 .
   \end{cases}
   $$

Return `(codes, scales, mins, d, dmin)` with the shapes and dtypes listed below. Throughout, $\operatorname{round}$ means round-to-nearest with ties to even, matching PyTorch, JAX, and MLX.

> [!NOTE]
> The layout and decoder match llama.cpp exactly. Step 1 uses plain min-max to pick each block's initial scale and minimum, whereas llama.cpp runs a weighted least-squares search per block (`make_qkx2_quants`). Steps 2 through 4 follow llama.cpp's reference quantizer.

## Shapes

| name | shape | dtype |
|---|---|---|
| `w` | `(S, 256)` | float32 |
| `codes` | `(S, 256)` | uint8 |
| `scales`, `mins` | `(S, 8)` | uint8 |
| `d`, `dmin` | `(S,)` | float16 |
| result of `q4k_decode` | `(S, 256)` | float32 |

## Example

Consider a single superblock where only the listed weights are non-zero (blocks 3 through 7 are all zeros):

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

![The encoded superblock of the example: the float16 pair, the 6-bit scale and minimum of each block, and the 32 4-bit codes of each block.](assets/superblock.svg)

**Step 1.** Block 0 spans $[-1, 2]$, giving $\text{scale}_0 = 3 / 15 = 0.2$ and $\text{min}_0 = 1$. Block 1 has only non-negative weights, so $lo_1$ is clamped to $0$, giving $\text{scale}_1 = 1 / 15$ and $\text{min}_1 = 0$. Block 2 is constant at $-0.4$, so its range is $0$, $\text{scale}_2 = 0$, and $\text{min}_2 = 0.4$. The all-zero blocks have scale $0$ and minimum $0$.

**Steps 2 and 3.** Over the 8 blocks, $\text{max\_scale} = 0.2$ and $\text{max\_min} = 1$. Dividing by $63$ gives $0.0031746$ and $0.015873$, which round in float16 to $d = 0.0031738$ and $d_{\min} = 0.015869$. The 6-bit fields are $63 \cdot (1/15) / 0.2 = 21$ for block 1's scale and $63 \cdot 0.4 / 1 = 25.2 \to 25$ for block 2's minimum.

**Step 4.** Block 0 is quantized against $D_0 = 63 d = 0.19995$ and $M_0 = 63 d_{\min} = 0.99976$ rather than the ideal $0.2$ and $1$. For weight $0.66$, $(0.66 + 0.99976) / 0.19995 = 8.30$, which rounds to code $8$. Block 2 has $D_2 = 0$, so all its codes are $0$, and every weight decodes to $-M_2 = -25\, d_{\min} = -0.39673$, off from $-0.4$ by $0.0033$.

## Constraints

- $1 \le S \le 16$. All entries of `w` are finite with $|w| \le 10^3$. Every superblock that is not all zeros has a largest magnitude of at least $10^{-3}$.
- All five outputs of `q4k_encode` are compared **exactly**, including dtypes, and `d` and `dmin` must match bit for bit.
- To prevent float32 rounding from deciding a tie, every test keeps the step-3 quotients at least $0.01$ from a half-integer, ensures the two float16 conversions are invariant to a relative perturbation of $10^{-5}$, and keeps every $(w_k + M_b) / D_b$ in $[0, 15]$ at least $0.02$ from a half-integer (lowered to $0.003$ in one test focused on the float16 units).
- `q4k_decode` is checked to a relative tolerance of `1e-5` plus `1e-6` times the largest decoded magnitude.
- **Error bound.** For weights encoded by `q4k_encode`, every decoded weight satisfies
  $$ |w_k - \hat w_k| \;\le\; \tfrac12 D_b + \tfrac{15}{2}\, d + \tfrac12\, d_{\min} $$
  within a 5 percent margin (accounting for float16 rounding of `d` and `dmin`). This bound sums half a grid step with the maximum shift of the grid's endpoints from 6-bit rounding.
- No Python loops over superblocks, blocks, or weights are needed.

## Why this matters

`Q4_K_M` is the default GGUF quantization preset for most local LLMs, and most of its tensors use `Q4_K` superblocks. Storing a float16 scale and minimum per 32 weights costs 5 bits per weight. Quantizing those 16 block parameters to 6 bits each and sharing two float16 units across 256 weights cuts the cost to 4.5 bits per weight while keeping the same 32-weight block size.

That second level of quantization introduces subtle bugs that never break the file format. Computing codes against the ideal block scale instead of the reconstructed grid shifts every weight by a fraction of a step. Failing to clamp positive block minimums wraps unsigned integers. Taking the superblock unit from the wrong block saturates larger blocks at 63. In every case the model loads and generates text, and only a perplexity evaluation reveals the lost precision.
