Start with the formula, because it tells you what the count is for. One term is
proportional to the count; the other depends only on whether the count is zero. How do
you turn "the count is positive" into a number you can multiply by a penalty? And what
type does an integer tensor times a Python float have in your framework?

---

For `apply_penalties`: `token_counts * frequency_penalty + (token_counts > 0) *
presence_penalty`, subtracted from the logits. For `count_generated` you need a
histogram per row. First build a `(B, T)` mask of the generated positions by comparing
`arange(T)` with `prompt_lengths[:, None]`. Then add that mask, as integers, into a
`(B, V)` table of zeros at the columns given by `tokens`: a generated position adds 1
to its token's count and a prompt position adds 0.

---

The adding has to *accumulate*: a token that occurs at three generated positions must
receive three additions. That is a scatter-add:

- PyTorch: `torch.zeros(B, V, dtype=torch.int64).scatter_add(1, tokens, generated.to(torch.int64))`
- JAX: `jnp.zeros((B, V), jnp.int32).at[rows, tokens].add(generated.astype(jnp.int32))`
- MLX: `mx.zeros((B, V), dtype=mx.int32).at[rows, tokens].add(generated.astype(mx.int32))`

with `rows = arange(B)[:, None]`. An indexed assignment such as
`counts[rows, tokens] += 1` does not accumulate in PyTorch: when an index repeats,
only one of the writes survives. The boundary is `arange(T) >= prompt_lengths[:, None]`:
position `prompt_lengths[b]` is the first generated one.
