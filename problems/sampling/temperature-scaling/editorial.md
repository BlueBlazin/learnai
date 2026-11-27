## The idea

Let $x_i$ and $x_j$ be two logits in a row, and let $p_i$ and $p_j$ be their softmax probabilities after dividing by a positive temperature $T$. Softmax depends only on differences between logits, and dividing by $T$ scales every difference by $1/T$:

$$
\frac{p_i}{p_j} = e^{(x_i - x_j)/T}.
$$

When $T < 1$, the probability ratios move away from 1 and the distribution sharpens. When $T > 1$, the ratios move toward 1 and the distribution flattens. Because $x \mapsto x / T$ is strictly increasing for $T > 0$, the token ranking never changes. Unless all finite logits are equal, the entropy of $\operatorname{softmax}(x / T)$ is strictly increasing in $T$ because its derivative with respect to $T$ is $\operatorname{Var}_p(x) / T^3$, which is positive.

As $T \to 0$, every ratio $p_i / p_{\max}$ with $x_i < x_{\max}$ approaches 0, so the limit is greedy selection. Plugging $T = 0$ directly into $x / T$ divides by zero, so greedy rows need a separate branch. Representing a greedy row as `0` at the winning token and `-inf` elsewhere lets it flow through the same softmax and sampler as every other row while assigning probability `1` to the argmax.

Because a batch can mix zero and positive temperatures, compute both branches for the full batch and select per row with `where`. Make sure the unused branch does not produce `inf` or NaN along the way.

## Reference solution (PyTorch)

```python
def apply_temperature(logits, temperature):
    t = temperature[:, None]
    greedy = t == 0
    scaled = logits / torch.where(greedy, 1.0, t)
    top = torch.argmax(logits, dim=-1, keepdim=True)
    is_top = torch.arange(logits.shape[-1]) == top
    one_hot = torch.where(is_top, 0.0, float("-inf"))
    return torch.where(greedy, one_hot, scaled)
```

- `temperature[:, None]` reshapes `temperature` to `(B, 1)` so it broadcasts across the vocabulary axis. Without that extra axis, a `(B,)` vector aligns with the last axis of `logits`. That raises an error when `B != V` and produces silently wrong values when `B == V`.
- `torch.where(greedy, 1.0, t)` keeps the division safe in every row. Even though the final `where` discards `scaled` on greedy rows, computing an `inf` or NaN in a discarded branch still poisons gradients and triggers NaN checks.
- `argmax` returns the lowest index among tied maxima in PyTorch, JAX, and MLX. Because every row has at least one finite entry, a masked `-inf` can never win.
- Building `one_hot` with `where` on a boolean mask avoids arithmetic on `-inf`, so masked entries cannot turn into NaN.

The JAX and MLX implementations use the same structure with `jnp.where` / `mx.where` and `keepdims=True`.

## Common mistakes

- **Dividing by zero.** Computing `logits / temperature` on a zero-temperature row gives `+inf` for positive logits, `-inf` for negative logits, and NaN for `0`. Taking the softmax of a row containing `+inf` produces NaN.
- **Clamping the temperature (`max(T, 1e-6)`).** Clamping avoids NaN and approximates greedy sampling, but it violates the specification. It scales the logits by $10^6$ instead of replacing them with `0` and `-inf`, and tied maxima split probability instead of selecting the lowest index.
- **Multiplying instead of dividing.** It is easy to flip the sign of the effect. As a sanity check, a higher temperature must flatten the distribution and increase entropy.
- **Turning `-inf` into NaN.** Constructing the greedy row arithmetically (such as `logits * 0 + mask` or `logits - logits`) evaluates `-inf * 0` or `-inf - (-inf)` on masked entries, both of which produce NaN. Use `where` instead.
- **Keeping all tied maxima.** Checking `logits == logits.max()` marks every tied maximum. The result is no longer one-hot, and different engines can emit different tokens for the same logits. Tie-breaking must pick the lowest index.
- **Keeping the original logit at the argmax.** Preserving `logits[b, v*]` yields the same softmax, but the specification requires `0` so the greedy output does not depend on the scale of the input logits.
- **Using one temperature for the whole batch.** Reading `temperature[0]` or `temperature.item()` from single-sequence code forces every request in the batch to use the first request's setting.

## Complexity

$O(B \cdot V)$ time and $O(B \cdot V)$ memory for one division, one `argmax`, and two `where` operations across the batch.

## In the wild

llama.cpp uses the same branch in `llama_sampler_temp_impl`. For `temp <= 0`, it finds the first token with the highest logit and sets all other logits to `-INFINITY`. Otherwise it divides by `temp`. In its default sampler chain, temperature runs last (after top-k, top-p, and min-p) so those filters see the unscaled distribution.

Hugging Face's `TemperatureLogitsWarper` rejects `temperature = 0` and requires `do_sample=False` for greedy decoding. In contrast, the OpenAI API and vLLM accept `temperature = 0` as greedy decoding, and batched servers track one temperature per request using a `(B,)` vector just like this problem.
