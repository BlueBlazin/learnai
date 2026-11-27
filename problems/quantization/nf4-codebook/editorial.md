## The idea

A $k$-bit quantizer makes the best use of its $2^k$ levels when every level is used equally often. For values drawn from a known distribution, that means placing levels at equal steps in probability (quantiles) rather than equal steps in value. For a standard normal distribution, equal probability steps produce tight spacing near zero, where density is high, and wide spacing in the tails.

NF4 makes two adjustments to the codebook and adds per-block scaling.

**Bounded range.** The normal quantile function is unbounded because $\Phi^{-1}(1) = \infty$. NF4 stops at $\delta = 0.9677083$, where $\Phi^{-1}(\delta) \approx 1.848$, and divides all levels by that maximum so the codebook spans $[-1, 1]$.

**Exact zero.** An even number of probabilities placed symmetrically around $0.5$ never includes $0.5$ itself, so a symmetric 16-level table has no zero. Exact zeros matter because padding, pruned weights, and freshly initialized LoRA matrices must stay zero after quantization. NF4 solves this with an asymmetric split of $2^{k-1} + 1 = 9$ probabilities from $\delta$ down to $0.5$ on the positive side and $2^{k-1} = 8$ on the negative side. Both halves end at probability $0.5$, whose quantile is $0$, and the two zeros are merged into one. That leaves 8 positive levels, 7 negative levels, and one exact zero, at the small cost of slightly denser positive spacing.

**Per-block scaling.** Real weight matrices are normal only up to a local scale that varies across rows and columns. Dividing each block by its absolute maximum maps all weights in the block into $[-1, 1]$ so nothing is clipped, at the cost of storing one float32 scale per block.

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

In `nf4_levels`, `negative` is already in ascending order because the largest probability $\delta$ produces the most negative value, while `positive` is descending and needs `.flip(0)`. JAX provides `jax.scipy.special.ndtri`. MLX has no `ndtri`, so the MLX reference computes $\Phi^{-1}(p) = \sqrt{2}\,\operatorname{erfinv}(2p - 1)$. In float32, all three match the published bitsandbytes table to within $2 \cdot 10^{-7}$, and in float64 with `scipy.stats.norm.ppf` the formula reproduces the published table exactly.

Comparing every element against all 16 levels temporarily allocates a tensor 16 times the size of `w`. Because the table is sorted, a binary search over the 15 midpoints between adjacent levels (`searchsorted` or `bucketize`) produces the same codes without that temporary expansion. The CUDA kernels in bitsandbytes hard-code those 15 midpoints as a small decision tree.

## Common mistakes

- **Using a uniform table.** `linspace(-1, 1, 16)` gives uniform 4-bit quantization with per-block scaling, and it also lacks an exact zero because its innermost levels are $\pm 1/15$.
- **Using a symmetric quantile table.** Taking 8 probabilities on each side omits zero, turning every exact zero weight into $\pm 0.04 \cdot \text{absmax}$.
- **Forgetting to normalize the table.** Without dividing by $\Phi^{-1}(\delta) \approx 1.848$, the table spans $[-1.848, 1.848]$ while normalized weights lie in $[-1, 1]$. The outer levels are rarely selected, and dequantized weights are scaled up by $1.848$.
- **Rounding by uniform index.** Computing `round((x + 1) / 2 * 15)` treats the code as a position on a uniform grid. It matches NF4 only at codes 0 and 15 because the codebook levels are unevenly spaced.
- **Normalizing by standard deviation.** Dividing each block by its standard deviation instead of its `absmax` leaves the largest weights outside $[-1, 1]$, clipping them to the outermost levels.
- **Using one `absmax` for the whole tensor.** Although the paper informally refers to normalizing the tensor, real implementations normalize per block so that a single outlier does not crush the precision of the entire matrix.
- **Skipping the all-zero block guard.** Dividing `0 / 0` produces `NaN`, which turns `argmin` into an arbitrary code. Replacing a zero `absmax` with `1` maps every element in the block to code 7.
- **Tiling the scales in `nf4_dequantize`.** Each block scale must be repeated `block_size` times consecutively (`repeat_interleave`). Tiling the row of scales (`[a0, a1, a0, a1, ...]`) produces the right shape while pairing blocks with the wrong scales.

## Complexity

Building the 16-entry table takes $O(1)$ time. Quantizing takes $O(16 \cdot \text{out} \cdot \text{in})$ time with direct distance comparisons, or $O(\text{out} \cdot \text{in} \cdot \log 16)$ with binary search. Dequantizing performs one table lookup and one multiplication per weight. Storage is 4 bits per weight plus one float32 scale per block, which comes to $4 + 32 / 64 = 4.5$ bits per weight at the default block size of 64.

## In the wild

NF4 was introduced in *QLoRA: Efficient Finetuning of Quantized LLMs* (arXiv:2305.14314). The paper describes levels as averages of adjacent quantiles, $q_i = \tfrac{1}{2}\big(Q(\tfrac{i}{2^k + 1}) + Q(\tfrac{i + 1}{2^k + 1})\big)$ where $Q = \Phi^{-1}$, with $2^{k-1}$ quantiles on the negative side and $2^{k-1} + 1$ on the positive side. In practice, models use the table produced by `create_normal_map` in bitsandbytes (`bitsandbytes/functional.py`), which spaces probabilities linearly from $0.9677083$ to $0.5$ and hard-codes the resulting 16 values.

bitsandbytes flattens each weight tensor and splits it into blocks of 64 consecutive elements, which matches row-wise blocking whenever `in` is a multiple of 64. QLoRA also adds double quantization, quantizing the per-block float32 scales to 8 bits in groups of 256 to reduce scale overhead from $0.5$ to about $0.127$ bits per weight. Because NF4 is a storage format, weights are dequantized back to a floating-point compute dtype during the forward pass, saving memory rather than compute.
