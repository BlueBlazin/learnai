## The idea

Dividing logits by $T$ multiplies every gap between two logits by $1/T$, and softmax
depends only on those gaps:

$$
\frac{p_i}{p_j} = e^{(x_i - x_j)/T}.
$$

For $T < 1$ the ratios move away from 1 (the distribution sharpens), for $T > 1$ they
move toward 1 (it flattens), and the order of the tokens never changes because
$x \mapsto x / T$ is increasing. The entropy of $\operatorname{softmax}(x / T)$ is a
strictly increasing function of $T$ unless all finite logits are equal: its derivative
with respect to $T$ is $\operatorname{Var}_p(x) / T^3$, a variance, which is positive.

As $T \to 0$ every ratio $p_i / p_{\max}$ with $x_i < x_{\max}$ goes to 0: the limit is
the greedy choice. The limit cannot be computed by plugging in $T = 0$, since that
evaluates `x / 0`, so greedy is a separate branch. The specification writes the greedy
row as logits (0 at the chosen token, `-inf` elsewhere) so that it can flow through
the same softmax and sampler as every other row and still give the argmax with
probability exactly 1.

With per-row temperatures both branches are computed for the whole batch and a
`where` selects per row. The only care needed is that the branch that is not selected
must not produce garbage on the way.

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

- `temperature[:, None]` has shape `(B, 1)` and broadcasts over the vocabulary.
  Without the extra axis the `(B,)` vector lines up with the *last* axis of the
  logits: an error when `B != V`, and silently wrong numbers when `B == V`.
- `torch.where(greedy, 1.0, t)` makes the division safe everywhere. The final `where`
  would discard an `inf` or NaN from a greedy row anyway, but a value that is computed
  and discarded still poisons gradients and trips NaN checks, so do not compute it.
- `argmax` returns the first maximal index in PyTorch and JAX (both document it) and
  in MLX. A masked entry can never win, because every row has a finite logit.
- The one-hot row comes from `where` on a boolean table. Nothing is multiplied by 0 or
  subtracted from itself, so a masked `-inf` cannot turn into NaN.

The JAX and MLX versions are the same lines with `jnp.where` / `mx.where` and
`keepdims=True`.

## Common mistakes

- **Dividing by zero.** `logits / temperature` for a zero-temperature row gives
  `+inf` for positive logits, `-inf` for negative ones, and NaN for a logit of exactly
  0. Softmax of a row with `+inf` in it is NaN.
- **Clamping the temperature** (`max(T, 1e-6)`). It avoids the NaN and is close to
  greedy, but it is not the specified result: the logits are multiplied by a million
  instead of being replaced by 0 and `-inf`, and tied maxima share the probability
  instead of the lowest index winning.
- **Multiplying instead of dividing.** The convention is easy to flip. The check is
  the entropy: a higher temperature must give a flatter distribution.
- **`-inf` turned into NaN.** Building the greedy row arithmetically, for instance
  `logits * 0 + mask` or `logits - logits`, hits `-inf * 0` or `-inf - (-inf)` at
  masked entries. Both are NaN. Use `where`.
- **Keeping every tied maximum.** `logits == logits.max()` marks all tied entries.
  The result is no longer one-hot, and two engines given the same logits then emit
  different tokens. Tie-breaking has to be specified and implemented: here, the lowest
  index.
- **Keeping the original logit** at the argmax. It gives the same softmax, but the
  specification asks for 0, which keeps the row independent of the scale of the
  logits.
- **One temperature for the whole batch**, typically `temperature[0]` or
  `temperature.item()` left over from single-sequence code. Every request then decodes
  with the first request's setting.

## Complexity

$O(B \cdot V)$ time and memory: one division, one argmax, and two selects over the
batch.

## In the wild

llama.cpp's temperature sampler has the same branch: for `temp <= 0` it finds the
token with the highest logit and sets every other logit to `-INFINITY`, keeping the
first one among ties, and otherwise divides (`llama_sampler_temp_impl`). It runs
temperature *last* in its default sampler chain, after top-k, top-p, and min-p, so the
filters see the model's unscaled distribution. Hugging Face's
`TemperatureLogitsWarper` rejects a temperature of 0 outright and expects greedy
decoding to be requested with `do_sample=False`. The OpenAI API and vLLM both accept
`temperature = 0` and document it as greedy, and a batched server carries one
temperature per request: exactly the `(B,)` vector of this problem.
