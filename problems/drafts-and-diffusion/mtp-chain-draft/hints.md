Before writing any tensor code, write down the eight token ids that each of the three stages embeds in the worked example. Stage 0 never sees `tokens[0]`, and its last id is `next_token`. What is the last id of stage 1?

---

Track two tensors that update at every stage: `ids` (a window of `T` token ids) and `h` (a `(T, D)` hidden-state tensor). Initialize `ids = tokens[1:] + [next_token]` and `h = hidden`. After each stage, set `h` to the stage's full post-normed output, and roll `ids` left by one by dropping its first id and appending the newly drafted token. Always build a new tensor for `ids` instead of mutating `tokens`.

---

Compute the block input with one projection over a concatenated tensor:

```python
e = ops.rms_norm(embedding[ids], params[p + "enorm.weight"], cfg.norm_eps)
g = ops.rms_norm(h, params[p + "hnorm.weight"], cfg.norm_eps)
x = concat((e, g), axis=-1) @ params[p + "eh_proj.weight"].T  # embedding first
```

Here `p = f"mtp.layers.{j}."`, and the first `D` columns of `eh_proj.weight` multiply the embedding half.

---

The decoder block matches a single layer of `decoder-forward` on one sequence. Add a leading batch axis of 1 so `ops.attention` receives `(1, n_heads, T, head_dim)` queries and `(1, n_kv_heads, T, head_dim)` keys and values. To get that shape, reshape `(T, n * head_dim)` to `(T, n, head_dim)`, swap the first two axes, and add the leading axis. Make sure you pass a causal mask. Even though only the last row produces logits, the next stage reads every row.

After the block, apply `h = ops.rms_norm(x, post_norm, cfg.norm_eps)`. Then compute `h[-1] @ head.T`, using the embedding matrix as `head` when `cfg.tie_embeddings` is `True`.
