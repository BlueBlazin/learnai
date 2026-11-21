Write down, for the example, the eight token ids that each of the three stages embeds
before you write any tensor code. Stage 0 never sees `tokens[0]`, and the last id of
stage 0 is `next_token`. What is the last id of stage 1?

---

Keep two things that change from stage to stage: `ids`, a window of `T` token ids, and
`h`, a `(T, D)` tensor. Start with `ids = tokens[1:] + [next_token]` and `h = hidden`.
After each stage, `h` becomes the stage's post-normed output (all rows), and the window
rolls by one: drop its first id, append the token just drafted. Build a new tensor
for the window instead of writing into `tokens`.

---

The input of the block is one matrix product on a concatenation:

```python
e = ops.rms_norm(embedding[ids], params[p + "enorm.weight"], cfg.norm_eps)
g = ops.rms_norm(h, params[p + "hnorm.weight"], cfg.norm_eps)
x = concat((e, g), axis=-1) @ params[p + "eh_proj.weight"].T  # embedding first
```

with `p = f"mtp.layers.{j}."`. The first `D` columns of `eh_proj.weight` multiply the
embedding half.

---

The block is the layer loop body of `decoder-forward` for a single sequence. Give the
tensors a leading axis of 1 so that `ops.attention` sees
`(1, n_heads, T, head_dim)` queries and `(1, n_kv_heads, T, head_dim)` keys and values:
reshape `(T, n * head_dim)` to `(T, n, head_dim)`, swap the first two axes, add the
axis. Do not forget the causal mask: the last row attends to everything either way, but
the next stage reads every row.

After the block: `h = ops.rms_norm(x, post_norm, cfg.norm_eps)`, the logits are
`h[-1] @ head.T`, and the head is the embedding matrix unless the checkpoint is untied.
