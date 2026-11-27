## The idea

A constraint does not change what the model computes. It changes which outcomes the sampler may pick. For a single row, let $A$ be the set of allowed token indices, $\ell_v$ the logit of token $v$, and $p_v$ its unmasked softmax probability. The target distribution is the model's distribution conditioned on the next token belonging to $A$.

$$
P(v \mid v \in A) = \frac{p_v}{\sum_{w \in A} p_w}
= \frac{e^{\ell_v}}{\sum_{w \in A} e^{\ell_w}} \quad (v \in A).
$$

The normalizing denominator of the full softmax cancels out, leaving a plain softmax over the allowed logits. Writing `-inf` into every disallowed position implements this directly because $e^{-\infty} = 0$, so disallowed tokens contribute nothing to the sum and receive exact probability `0`. The `-inf` value also survives every downstream sampler step, including temperature scaling, logit bias, and top-p filtering.

The order of operations is critical for numerical stability. A stable softmax subtracts the row maximum before exponentiating. When you mask first, that maximum is the largest allowed logit, so allowed tokens keep full float32 precision. If you run softmax first, the maximum can belong to a disallowed token. Any allowed token more than about 100 units below it underflows to zero in float32, and renormalizing divides zero by zero to produce NaN. This happens routinely when a model strongly prefers prose while a grammar demands `{`.

A row with no allowed tokens has no valid conditioned distribution. Its masked logits are all `-inf`, whose softmax is NaN. Letting NaN reach the sampler causes out-of-range indices or crashes far from the root cause, so checking every row before computing the mask fails fast and cleanly.

## Reference solution (PyTorch)

```python
def mask_logits(logits, allowed):
    if not bool(allowed.any(dim=-1).all()):
        raise ValueError("a row of `allowed` has no allowed token")
    return torch.where(allowed, logits, float("-inf"))


def masked_probs(logits, allowed):
    return torch.softmax(mask_logits(logits, allowed), dim=-1)
```

The JAX and MLX versions use the same logic with `jnp.where` or `mx.where` and each framework's `softmax`. In JAX, `bool(...)` reads a concrete value, which works here because the function is not traced. Inside `jax.jit`, you would instead return a per-row dead-end flag and let the caller raise.

## Common mistakes

- **Using a large negative constant.** A value like `-1e9` may look right on simple inputs, but downstream logit biases or second masks can shift it, half precision cannot represent it, and checks using `isinf` miss it. `-inf` requires no tuning.
- **Zeroing probabilities without renormalizing.** Multiplying unmasked probabilities by `allowed` leaves each row summing to the allowed mass instead of 1. A sampler drawing `u` in `[0, 1)` against the cumulative sum runs off the end whenever `u` exceeds that mass.
- **Running softmax before masking.** This is mathematically equivalent in exact arithmetic, but produces NaN whenever the allowed probabilities underflow to zero.
- **Skipping the empty-row check or checking the whole batch at once.** Omitting the check returns NaN, and calling `allowed.any()` across the entire tensor only triggers when every row is empty. In a batched server, one dead-ended request must be caught even when other rows are valid.
- **Broadcasting one mask across the batch.** Each sequence has its own constraint state. A `(V,)` mask only works when every sequence happens to be in the exact same state.
- **Building an additive mask with multiplication.** Computing `logits + (~allowed) * -inf` evaluates `0 * -inf = NaN` at every allowed position.

## Complexity

Selecting entries and running softmax each take one pass over all `B * V` elements, for $O(BV)$ time and memory. In practice, the bottleneck in constrained decoding is computing `allowed` and transferring it to the device. A boolean mask for a 128,000-token vocabulary takes 128 kB per sequence per step, which is why serving engines pack masks into bitsets.

## In the wild

- Hugging Face `transformers` implements constraints as logits processors that write `-inf` into disallowed token scores before sampling. Top-k and top-p filters use the same convention so that filters compose cleanly.
- llama.cpp's grammar sampler sets the logit of each rejected candidate to `-INFINITY`.
- XGrammar and llguidance pack each mask into a bitset of 32 tokens per integer and apply it to logits in place. vLLM gathers one bitmask row per constrained request in a batch and applies them in a single call.
- Mask placement in the sampler chain matters. Applied before top-k and top-p, those filters pick among allowed tokens. Applied after them, the surviving top-k set can contain no allowed tokens, creating the empty-row dead end this problem detects.
