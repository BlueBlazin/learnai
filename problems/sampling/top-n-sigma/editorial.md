## The idea

Tang et al. observe that language model logits typically form a large, roughly Gaussian cloud of noisy tokens plus a small set of informative tokens far above it. Probability-based filters like top-p and min-p view those logits through softmax, which changes shape with temperature. Top-n-sigma filters in logit space instead by keeping every token $i$ whose logit $x_i$ lies within $n$ standard deviations $\sigma$ of the row maximum $M$:

$$
x_i \ge M - n\,\sigma .
$$

This rule is invariant to both temperature scaling and constant shifts. Dividing every logit by a positive temperature $T$ scales $x_i$, $M$, and $\sigma$ by the same factor $1/T$, leaving the inequality unchanged. Adding a constant to every logit in a row shifts $x_i$ and $M$ equally without changing $\sigma$. Top-p and min-p are shift-invariant, but they are not temperature-invariant.

Two details in the definition of $\sigma$ are essential for correctness:

- **Compute over finite logits only.** A masked `-inf` token is absent, not merely low-probability. Including `-inf` in arithmetic produces a mean of `-inf` and deviations of `-inf - (-inf)`, which evaluate to NaN.
- **Use population standard deviation, not sample standard deviation.** Divide by the count of finite entries $|F|$, not $|F| - 1$. Across a 100,000-token vocabulary the difference is negligible, but after a grammar masks all but four tokens it shifts the cutoff band by 15%. With only one finite token left, dividing by $|F| - 1$ divides by zero.

## Reference solution (PyTorch)

```python
def top_n_sigma_filter(logits, n):
    if n <= 0:
        return logits
    finite = torch.isfinite(logits)
    count = finite.sum(dim=-1, keepdim=True)
    mean = torch.where(finite, logits, 0.0).sum(dim=-1, keepdim=True) / count
    squares = torch.where(finite, (logits - mean) ** 2, 0.0)
    std = torch.sqrt(squares.sum(dim=-1, keepdim=True) / count)
    cutoff = torch.amax(logits, dim=-1, keepdim=True) - n * std
    return torch.where(logits >= cutoff, logits, float("-inf"))
```

- `torch.where` selects values before summation, so the `inf` produced by `(logits - mean) ** 2` at a masked entry never enters the sum.
- Using `>=` handles degenerate rows without extra branching. When a row has one finite logit, `std` is `0`, the cutoff equals that logit, and the token passes.
- `-inf >= cutoff` is always false, so previously masked entries stay `-inf`.

The JAX and MLX implementations use the same operations with `keepdims=True`.

## Common mistakes

- **Calling a library `std` over the whole row.** A single `-inf` turns `std` into NaN. Every `x >= NaN` check is false, so the filter masks the entire row to `-inf`. This bug passes any test without masked inputs.
- **Zeroing masked entries but dividing by the full vocabulary size `V`.** Computing `where(finite, x, 0).mean()` avoids NaN, but it pulls the mean toward `0` and distorts the standard deviation.
- **Using sample standard deviation.** `torch.std` and `numpy.std(ddof=1)` divide by $|F| - 1$, whereas `numpy.std`, `jnp.std`, and `mx.std` divide by $|F|$ by default. Porting code across libraries without checking `ddof` or `correction` changes the result on small candidate sets.
- **Measuring the band below the mean $\mu$ (cutoff $\mu - n\sigma$).** That textbook rule filters from the mean instead of the maximum $M$, which keeps most of the noise cloud.
- **Taking the standard deviation of probabilities.** This pattern feels natural after implementing top-p or min-p, but it breaks temperature invariance.
- **Computing one standard deviation for the whole batch.** Mixing rows lets a sharp row and a flat row distort each other's cutoff.
- **Using `>` instead of `>=`.** This looks harmless until a row has zero spread, where it masks out the only valid token.
- **Applying the cutoff when `n <= 0`.** The formula gives greedy decoding at `n = 0` and masks everything when `n < 0`, whereas inference engines treat `n <= 0` as disabled.

## Complexity

For a batch of `B` rows and vocabulary size `V`, the filter runs in $O(B \cdot V)$ time and $O(B \cdot V)$ memory. It uses three reductions and an elementwise comparison, with no sorting and no exponentials.

## In the wild

Top-n-sigma was introduced by Tang, Liu, Xu, and Huang in "Top-nσ: Not All Logits Are You Need" (arXiv 2411.07641, 2024). The authors report that `n = 1` works well and maintains generation quality at high temperatures where top-p and min-p degrade.

This problem matches `llama_sampler_top_n_sigma_apply` in `llama.cpp` (`src/llama-sampler.cpp`). That implementation returns early for `n <= 0` or a single candidate, skips `-INFINITY` logits when computing the mean and standard deviation, divides by the count of kept logits, and masks any logit strictly below `max - n * std`. The CLI flag is `--top-nsigma` with a default of `-1.0` (disabled). In the default sampler chain, it runs first among truncation samplers (penalties, DRY, top-n-sigma, top-k, typical-p, top-p, min-p, XTC, temperature).

Implementations can differ in two subtle ways. The paper applies the rule to temperature-scaled logits, which selects the same tokens as applying it before temperature scaling in `llama.cpp` because of temperature invariance. However, implementations that rely on a tensor library's default `std` inherit its divisor convention, and implementations that run after other filters must explicitly exclude `-inf` tokens from the statistics.
