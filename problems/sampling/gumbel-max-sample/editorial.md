## The idea

Let $G_1, \ldots, G_V$ be independent standard Gumbel random variables with cumulative distribution function $P(G \le g) = e^{-e^{-g}}$. For any vector of logits $x$ of length $V$, adding Gumbel noise and taking the argmax samples directly from the softmax distribution:

$$
P\Big(\operatorname*{argmax}_v\,(x_v + G_v) = i\Big)
= \frac{e^{x_i}}{\sum_v e^{x_v}} .
$$

The proof takes one integral. For any competitor $v \ne i$, the event $x_v + G_v \le t$ has probability $\exp(-e^{x_v - t})$, so all competitors stay below $t$ with probability $\exp(-e^{-t} \sum_{v \ne i} e^{x_v})$. Multiply by the density of $x_i + G_i$ at $t$, which is $e^{x_i - t} \exp(-e^{x_i - t})$, and integrate over $t$. Writing $Z = \sum_v e^{x_v}$ and substituting $s = e^{-t}$ gives

$$
\int_{-\infty}^{\infty} e^{x_i - t}\, e^{-Z e^{-t}}\, dt
= e^{x_i} \int_0^\infty e^{-Z s}\, ds = \frac{e^{x_i}}{Z} .
$$

Thus, the argmax of noisy logits is an exact sample from the softmax without ever computing the normalizer $Z$. A masked token with $x_v = -\infty$ has $e^{x_v} = 0$, so it contributes nothing to $Z$ and has zero probability, which corresponds to a score of `-inf` in code.

To generate $G$ from a uniform random variable $U$, invert the CDF: $G = -\log(-\log U)$. Equivalently, $-\log U$ is a unit-rate exponential waiting time, token $v$'s clock runs at rate $e^{x_v}$, and the token whose clock rings first wins.

## Reference solution (PyTorch)

```python
U_MIN = 1.1754944e-38  # the smallest positive normal float32
U_MAX = 0.99999994  # the largest float32 below 1


def gumbel_sample(logits, uniforms):
    u = torch.clamp(uniforms, U_MIN, U_MAX)
    gumbel = -torch.log(-torch.log(u))
    return torch.argmax(logits + gumbel, dim=-1)
```

The main numerical hazard is at the endpoints of $[0, 1]$. The map $-\log(-\log u)$ sends $(0, 1)$ onto the entire real line, and float32 uniforms hit both boundaries:

| `u` | `-log(u)` | noise | effect without the clamp |
|---|---|---|---|
| `0.0` | `+inf` | `-inf` | a finite logit scores `-inf`, like a masked one |
| `1.0` | `0.0` | `+inf` | a finite logit always wins; a masked one scores NaN |

Clamping `u` to `[U_MIN, U_MAX]` keeps the noise inside $[-4.47, 16.64]$. The clamping only alters events of probability $2^{-24}$ per entry, which is undetectable in practice, while preventing NaNs and illegal masked-token outputs.

## Common mistakes

- **Using only one logarithm.** Computing `logits - log(u)` adds exponential noise instead of Gumbel noise. It looks random and stays fair for equal logits, but it does not follow the softmax distribution when logits differ.
- **Adding noise to probabilities.** Computing `softmax(logits) + g` adds noise with standard deviation `1.28` to probabilities in $[0, 1]$, making sampling nearly uniform regardless of the logits.
- **Flipping the sign.** Computing `logits + log(-log(u))` negates the Gumbel noise. Because the Gumbel distribution is skewed, $-G$ is not Gumbel and produces the wrong sampling distribution.
- **Omitting the upper clamp or using a too-small epsilon.** Expressions like `log(u + 1e-10)` or `clamp(u, 1e-10, 1 - 1e-10)` fail in float32 because `1.0 - 1e-10` rounds to `1.0`. At `u = 1.0`, the noise is `+inf`, a masked entry becomes NaN, and `torch.argmax` can return that NaN index.
- **Omitting the lower clamp.** At `u = 0.0`, the noise is `-inf`. If that entry is the only unmasked token in the row, every score becomes `-inf` and `argmax` returns index `0` even if token `0` is masked.
- **Sharing noise across the batch.** Broadcasting one row of Gumbel noise across all rows in the batch gives every sequence the same noise draw, so identical prompts produce identical tokens.
- **Generating randomness inside the function.** Drawing fresh uniforms internally makes the function impossible to replay, unit-test deterministically, or compare across frameworks.

## Complexity

For a batch of `B` rows and vocabulary size `V`, the sampler takes $O(B \cdot V)$ time and $O(B \cdot V)$ memory. It computes two logarithms per entry and consumes `V` uniform numbers per sampled token, whereas inverse-CDF sampling needs only one uniform number per token. In exchange, it avoids softmax, sorting, and sequential cumulative sums. Every step except the final argmax reduction is elementwise, which makes it fast on GPUs and inside compiled graphs.

## In the wild

`jax.random.categorical` computes `argmax(gumbel(...) + logits)` directly on unnormalized logits. MLX's `mx.random.categorical` uses the same Gumbel-max approach when sampling across a batch of rows, and switches to inverse-CDF sampling when drawing many samples from a single distribution. By contrast, `torch.multinomial` operates on probabilities rather than logits, while `torch.nn.functional.gumbel_softmax` builds its noise as `-log(Exponential(1))` to avoid the double logarithm of a uniform.

Frameworks guard the endpoints differently. JAX's default Gumbel sampler draws uniforms from `[tiny, 1)`, so the generator avoids both `0.0` and `1.0`. MLX computes `-log(-log(uniform))` without its own clamp. When a function accepts external uniforms, clamping both ends is required. Float32 uniform precision also limits tail accuracy. JAX notes that its default mode introduces bias for events with probability below roughly $10^{-7}$, and provides `mode="high"` using two random words per entry and `log1p` near the endpoints to push that threshold to around $10^{-14}$.

The Gumbel-max trick originates in extreme-value theory (Gumbel, 1954) and entered modern machine learning through Maddison, Tarlow, and Minka's "A* Sampling" (2014). Later work relaxed the argmax into Gumbel-softmax (Jang et al., 2017; Maddison et al., 2017) and extended it to top-k sampling without replacement (Kool et al., 2019).
