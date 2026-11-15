## The idea

A quantizer with $2^k$ levels wastes nothing when every level is used equally often.
For values drawn from a known distribution that means placing the levels at its
quantiles: equal steps in probability, not in value. For a standard normal, equal
steps in probability are small steps in value near the mean, where the density is
high, and large steps in the tails.

NF4 is that idea with two practical adjustments.

**The range is cut off.** The quantile function is unbounded: $\Phi^{-1}(1) =
\infty$. The construction stops at $\delta = 0.9677083$, where $\Phi^{-1}(\delta)
\approx 1.848$, and divides by that value, so the outermost levels are $\pm 1$.

**Zero is a level.** Evenly spaced probabilities that are symmetric about $0.5$ never
include $0.5$ itself when their number is even, so a symmetric 16-level table has no
zero: its two innermost levels straddle it. Zero matters more than any other value:
padding, pruned weights, and freshly initialized LoRA matrices are exact zeros, and
they must stay zero. The fix is an asymmetric construction: $2^{k-1} + 1 = 9$
probabilities from $\delta$ down to $0.5$ on the positive side and $2^{k-1} = 8$ on
the negative side. Each side ends at $0.5$, whose quantile is $0$; the two zeros are
merged into one. That leaves 8 positive levels, 7 negative levels, and one zero: 16.
The positive side is slightly denser than the negative one, which is the price.

**Blocks.** Weights are normal only up to a scale, and the scale varies across a
matrix. Dividing each block of 64 by its absolute maximum maps it into $[-1, 1]$, the
range of the table, and guarantees that nothing is clipped. One float per block is
stored next to the codes.

## Reference solution (PyTorch)

```python
OFFSET = 0.9677083


def nf4_levels():
    positive = torch.special.ndtri(torch.linspace(OFFSET, 0.5, 9)[:-1])
    negative = -torch.special.ndtri(torch.linspace(OFFSET, 0.5, 8)[:-1])
    levels = torch.cat([negative, torch.zeros(1), positive.flip(0)])
    return levels / levels.max()


def nf4_quantize(w, block_size):
    out_dim, in_dim = w.shape
    blocks = w.reshape(out_dim, in_dim // block_size, block_size)
    absmax = blocks.abs().amax(dim=-1)
    absmax = torch.where(absmax > 0, absmax, torch.ones_like(absmax))
    normalized = blocks / absmax[..., None]
    codes = (normalized[..., None] - nf4_levels()).abs().argmin(dim=-1)
    return codes.reshape(out_dim, in_dim).to(torch.uint8), absmax


def nf4_dequantize(codes, absmax, block_size):
    values = nf4_levels()[codes.long()]
    return values * absmax.repeat_interleave(block_size, dim=1)
```

`negative` is already ascending (the most negative level comes from $\delta$), and
`positive` is descending, hence the flip. JAX has `jax.scipy.special.ndtri`; MLX has
no `ndtri`, and the reference uses $\Phi^{-1}(p) = \sqrt{2}\,\operatorname{erfinv}(2p
- 1)$. In float32 the three constructions agree with the published table to about
$2 \cdot 10^{-7}$. In float64, with `scipy.stats.norm.ppf`, the construction
reproduces the published table exactly; that was checked when this problem was
written.

The search compares every element with all 16 levels, which costs 16 times the
memory of the matrix for a moment. Because the table is sorted, a binary search over
the 15 midpoints between adjacent levels (`searchsorted`, `bucketize`) gives the same
codes with one comparison tensor; the kernels in bitsandbytes hard-code those
midpoints as a small decision tree.

## Common mistakes

- **A uniform table.** `linspace(-1, 1, 16)` is 4-bit uniform quantization with
  per-block absmax scaling. It has no zero level either: its innermost levels are
  $\pm 1/15$.
- **A symmetric quantile table.** Eight probabilities on each side give 16 levels
  without a zero. The table looks right in a plot, and every exact zero in the model
  becomes $\pm 0.04 \cdot \text{absmax}$.
- **The table not normalized.** Without the division by $\Phi^{-1}(\delta)$ the
  levels span $\pm 1.848$ while the normalized weights span $\pm 1$: the outermost
  levels on each side are never or hardly ever the nearest one, so part of the
  16-entry table is wasted, and dequantized weights are too large by the same factor.
- **Nearest by index.** `round((x + 1) / 2 * 15)` treats the code as a position on a
  uniform ruler. It agrees with the table at codes 0 and 15 and almost nowhere else,
  because the levels are not evenly spaced.
- **Normalizing by the standard deviation.** The levels are quantiles of a normal, so
  dividing by the standard deviation of the block seems natural. But the table ends at
  $1$, not at $1.848$ standard deviations, and the block's largest weights, which
  matter most, are then clipped to the last level.
- **One absmax for the tensor.** The paper's description of the data type normalizes
  "the tensor"; the implementation works in blocks, and a single outlier would
  otherwise set the scale of every weight in the matrix.
- **No guard for an all-zero block.** `0 / 0` is NaN, and the nearest level to NaN is
  whatever `argmin` returns. With absmax replaced by 1, the block gets code 7
  throughout.
- **Tiling the scales.** In `nf4_dequantize`, each absmax must be repeated
  `block_size` times in a row (`repeat_interleave`, `repeat` along the axis). Tiling
  the row of scales (`[a0, a1, a0, a1, ...]`) has the right shape and assigns every
  block the wrong scale.

## Complexity

Building the table is constant work. Quantizing is $O(16 \cdot \text{out} \cdot
\text{in})$ with the direct comparison, $O(\text{out} \cdot \text{in} \cdot \log 16)$
with a binary search. Dequantizing is one lookup and one multiplication per weight.
Storage is 4 bits per weight plus one float32 per block: $4 + 32 / 64 = 4.5$ bits per
weight at the default block size.

## In the wild

The data type is defined in *QLoRA: Efficient Finetuning of Quantized LLMs*
(arXiv:2305.14314). The paper describes the levels as averages of adjacent quantiles,
$q_i = \tfrac{1}{2}\big(Q(\tfrac{i}{2^k + 1}) + Q(\tfrac{i + 1}{2^k + 1})\big)$, and
describes the asymmetric construction with $2^{k-1}$ quantiles for the negative part
and $2^{k-1} + 1$ for the positive part. The table that is actually used is the one
computed by `create_normal_map` in bitsandbytes (`bitsandbytes/functional.py`), with
evenly spaced probabilities from the offset $0.9677083$ to $0.5$; this problem follows
that function, and the 16 values it yields are hard-coded in the library.

bitsandbytes flattens a weight tensor and cuts it into blocks of 64 consecutive
elements (the default; the paper uses 64 as well). For a matrix whose row length is a
multiple of the block size that is the row-wise blocking of this problem. QLoRA adds
"double quantization" on top: the per-block scales are themselves quantized to 8 bits
in blocks of 256, which brings the overhead down from $0.5$ to about $0.127$ bits per
weight. NF4 is a storage format: for the forward pass the weights are dequantized to
a floating-point compute dtype and multiplied as usual, so it saves memory, not time.
