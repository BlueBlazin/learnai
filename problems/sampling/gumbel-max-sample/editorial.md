## The idea

Let $G_1, \ldots, G_V$ be independent standard Gumbel variables, with distribution
function $P(G \le g) = e^{-e^{-g}}$. Then for any logits $x$,

$$
P\Big(\operatorname*{argmax}_v\,(x_v + G_v) = i\Big)
= \frac{e^{x_i}}{\sum_v e^{x_v}} .
$$

The proof is one integral. $x_v + G_v \le t$ has probability $\exp(-e^{x_v - t})$, so
all competitors of $i$ are below $t$ with probability $\exp(-e^{-t} \sum_{v \ne i}
e^{x_v})$. Multiply by the density of $x_i + G_i$ at $t$, which is
$e^{x_i - t} \exp(-e^{x_i - t})$, and integrate: with $Z = \sum_v e^{x_v}$ and the
substitution $s = e^{-t}$,

$$
\int_{-\infty}^{\infty} e^{x_i - t}\, e^{-Z e^{-t}}\, dt
= e^{x_i} \int_0^\infty e^{-Z s}\, ds = \frac{e^{x_i}}{Z} .
$$

So the argmax of noisy logits is a sample from the softmax, and the normalizer $Z$ is
never computed. A masked token has $e^{x_v} = 0$: it contributes nothing to $Z$ and
has probability 0, which in code is "its score is `-inf`".

The noise is made by inverting the distribution function: if $U$ is uniform,
$G = -\log(-\log U)$. An equivalent picture: $-\log U$ is an exponential clock, token
$v$'s clock runs at rate $e^{x_v}$, and the token whose clock rings first wins.

## Reference solution (PyTorch)

```python
U_MIN = 1.1754944e-38  # the smallest positive normal float32
U_MAX = 0.99999994  # the largest float32 below 1


def gumbel_sample(logits, uniforms):
    u = torch.clamp(uniforms, U_MIN, U_MAX)
    gumbel = -torch.log(-torch.log(u))
    return torch.argmax(logits + gumbel, dim=-1)
```

The clamp is the whole numerical content. $-\log(-\log u)$ maps $(0, 1)$ onto the
real line, and float32 uniforms hit both ends:

| `u` | `-log(u)` | noise | effect without the clamp |
|---|---|---|---|
| `0.0` | `+inf` | `-inf` | a finite logit scores `-inf`, like a masked one |
| `1.0` | `0.0` | `+inf` | a finite logit always wins; a masked one scores NaN |

After the clamp the noise lies in $[-4.47, 16.64]$. The distortion is confined to
events of probability $2^{-24}$ per entry, far below anything a statistical test or a
user can see; the alternative is an occasional token that had probability zero.

## Common mistakes

- **One logarithm.** `logits - log(u)` adds exponential noise. The result looks
  random and plausible, and follows a different distribution: with two equal logits
  it is still fair, with unequal ones it is not the softmax.
- **Noise added to probabilities.** `softmax(logits) + g` perturbs numbers in
  $[0, 1]$ with noise of standard deviation 1.28: nearly uniform sampling, whatever
  the logits.
- **The wrong sign**, `logits + log(-log(u))`. $-G$ is not Gumbel (the distribution
  is skewed), so this is not a relabeling of the uniforms; it is a different sampler.
- **No upper clamp**, or an epsilon that vanishes in float32
  (`log(u + 1e-10)`, `clamp(u, eps, 1 - eps)` with `eps = 1e-10`). At `u = 1` the
  noise is `+inf`; on a masked entry the score is NaN, and `torch.argmax` returns the
  position of a NaN.
- **No lower clamp.** At `u = 0` the score is `-inf`. If that was the only unmasked
  token of the row, every score is `-inf` and argmax returns index 0, masked or not.
- **Noise shared across the batch.** Broadcasting one row of noise makes every
  request in a batch draw the same Gumbels: identical prompts produce identical
  continuations.
- **Drawing the noise inside.** Then the function cannot be tested or replayed, and
  two frameworks can never be compared.

## Complexity

$O(B \cdot V)$ time and memory, two logarithms per entry, and $V$ random numbers per
sampled token where inverse-CDF sampling needs one. In exchange there is no softmax,
no sort, and no sequential cumulative sum: everything is elementwise except a final
reduction, which is why this is the form that runs well on a GPU and inside compiled
graphs.

## In the wild

`jax.random.categorical` is `argmax(gumbel(...) + logits)` on unnormalized logits,
and MLX's `mx.random.categorical` adds `mx.random.gumbel` noise and takes the argmax
whenever it samples from a batch of rows (it switches to an inverse CDF when many
samples are drawn from a single distribution). `torch.multinomial` takes
probabilities and does not use this trick; `torch.nn.functional.gumbel_softmax`
builds its noise as `-log(Exponential(1))`, which is the same variable without the
double logarithm of a uniform.

Frameworks differ in how they guard the ends. JAX's default Gumbel sampler draws its
uniforms from `[tiny, 1)`, so the generator handles the lower end and excludes the
upper one; MLX's `gumbel` is `-log(-log(uniform))` with no clamp of its own. Code
that receives uniforms from elsewhere, as here, has to clamp both. The resolution of
float32 uniforms also limits the trick itself: JAX documents that with its default
mode, float32 sampling is biased for events with probability below about $10^{-7}$,
and offers `mode="high"`, which spends two random words per entry and switches to
`log1p` near the ends to push that limit to about $10^{-14}$.

The trick is from the extreme-value literature (Gumbel, 1954) and entered machine
learning through Maddison, Tarlow, and Minka, "A* Sampling" (2014); Jang et al. and
Maddison et al. (2017) relaxed the argmax into the Gumbel-softmax, and Kool et al.
(2019) showed that taking the top $k$ noisy logits samples $k$ tokens without
replacement.
