## The idea

Tang et al. observe that the logits of a language model look like a large cloud of
roughly Gaussian "noise" tokens plus a few informative tokens far above it. Probability
based filters see that picture through the softmax, which temperature distorts. Top-nσ
cuts in logit space instead: keep what lies within $n$ standard deviations of the top,

$$
x_i \ge M - n\,\sigma .
$$

The useful property follows in one line. Dividing the logits by a temperature $T$
divides $M$, every $x_i$, and $\sigma$ by $T$, so the inequality is unchanged: the
kept set does not depend on the temperature. Adding a constant to a row changes
neither side's difference, so it is shift invariant as well. Top-p and min-p have the
second property and not the first.

Everything else is bookkeeping about what $\sigma$ is:

- **Over the finite logits.** A masked token is not a very unlikely token, it is
  absent. Arithmetic on `-inf` gives a mean of `-inf` and deviations of
  `-inf - (-inf)`, which is NaN.
- **Population, not sample.** The divisor is the number of finite logits. For a
  vocabulary of 100,000 the difference is invisible; after a grammar has masked all
  but four tokens it is 15 percent of the band, and with one token left the sample
  formula divides by zero.

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

- `where` is a selection, not arithmetic: the `inf` that `(logits - mean) ** 2`
  produces at a masked entry is never added to anything.
- `>=` makes the degenerate rows work without a branch. With one finite logit the
  standard deviation is 0, the cutoff is that logit, and it passes.
- `-inf >= cutoff` is false, so masked entries stay masked.

The JAX and MLX versions are the same lines with `keepdims=True`.

## Common mistakes

- **A library `std` over the whole row.** One `-inf` makes it NaN, `x >= NaN` is false
  for every `x`, and the filter returns a row of `-inf`. It passes every test without
  masked entries.
- **Zeroing the masked entries but dividing by `V`** (`where(finite, x, 0).mean()`).
  No NaN, and a mean pulled toward 0 and a standard deviation of the wrong size.
- **The sample standard deviation.** `torch.std` and `numpy.std(ddof=1)` divide by
  $|F| - 1$; `numpy.std`, `jnp.std`, and `mx.std` divide by $|F|$ by default. Porting
  a NumPy line to torch changes the answer.
- **The band below the mean** ($\mu - n\sigma$). That is the textbook "n sigma" rule,
  and a different filter: it keeps most of the noise cloud.
- **Standard deviation of probabilities.** It reads naturally after writing top-p and
  min-p, and it gives up the reason for the method: the kept set moves with the
  temperature.
- **One standard deviation for the batch.** A confident row and a flat row then share
  a band.
- **`>` in place of `>=`.** Invisible until a row has zero spread, where it removes
  the only candidate.
- **Applying the rule for `n <= 0`.** The formula with $n = 0$ is greedy decoding and
  with $n < 0$ it removes everything; engines treat both as "off".

## Complexity

$O(B \cdot V)$ time and memory: three reductions and a comparison. No sort, and no
exponential.

## In the wild

The method is from Tang, Liu, Xu, and Huang, "Top-nσ: Not All Logits Are You Need"
(arXiv 2411.07641, 2024), which reports that `n = 1` works well and that quality stays
stable at temperatures where top-p and min-p degrade.

llama.cpp implements it as `llama_sampler_top_n_sigma_apply` in
`src/llama-sampler.cpp`, and this problem follows it: it returns early for `n <= 0`
(and for a single candidate), skips `-INFINITY` logits when computing the mean and the
standard deviation, divides by the count of the logits it kept, and sets a logit to
`-INFINITY` when it is `< max - n * std`. The parameter is `--top-nsigma`, with a
default of `-1.0` (disabled); `top_n_sigma` comes first among the truncation samplers
in the default order (penalties, DRY, top-n-sigma, top-k, typical-p, top-p, min-p,
XTC, temperature).

Where implementations differ: the paper states the rule on temperature-scaled logits,
which by the invariance above selects the same tokens as the unscaled rule llama.cpp
applies before its temperature stage. An implementation built on a tensor library's
default `std` inherits that library's divisor, so two ports of "the same" sampler can
disagree on short candidate lists; and an implementation that runs after other
filters must decide, as llama.cpp did, that removed tokens do not count.
