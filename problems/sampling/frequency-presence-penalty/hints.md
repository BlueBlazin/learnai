Start with the penalty formula because it shows what the count is used for. One term is
proportional to the count, and the other depends only on whether the count is positive.
How do you convert "the count is positive" into a value you can multiply by a penalty?
Also check what dtype your framework produces when multiplying an integer or boolean
tensor by a Python float.

---

For `apply_penalties`, compute
`token_counts * frequency_penalty + (token_counts > 0) * presence_penalty` and subtract
it from `logits`. For `count_generated`, you need a per-row histogram of generated
tokens. Build a `(B, T)` mask of generated positions by comparing `arange(T)` with
`prompt_lengths[:, None]`. Then add that mask, cast to integers, into a `(B, V)` table
of zeros at the columns given by `tokens`. Each generated position adds `1` to its
token's count, and each prompt position adds `0`.

---

The addition must accumulate so that a token appearing at three generated positions
receives three additions. Use a scatter-add operation:

- PyTorch: `torch.zeros(B, V, dtype=torch.int64).scatter_add(1, tokens, generated.to(torch.int64))`
- JAX: `jnp.zeros((B, V), jnp.int32).at[rows, tokens].add(generated.astype(jnp.int32))`
- MLX: `mx.zeros((B, V), dtype=mx.int32).at[rows, tokens].add(generated.astype(mx.int32))`

Here `rows = arange(B)[:, None]`. In PyTorch, plain indexed assignment such as
`counts[rows, tokens] += 1` does not accumulate duplicate indices, so only one write
survives. Use `arange(T) >= prompt_lengths[:, None]` for the boundary because position
`prompt_lengths[b]` is the first generated token.
