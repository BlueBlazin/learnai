Start by targeting stage 2 with `state=None` and ignore the returned state at first, since stage 2 checks only `logits`. The outer block structure matches `decoder-forward`, except `cfg.layers[i]` picks the mixer for each layer. For the attention mixer, trace the tensor shape at every step between `q_proj` and `o_proj` before coding. Check how the $2 d H$ columns of `q_proj` split into queries and gates, and note which tensors are normalized, which are rotated, and which are left untouched.

---

Reshape the `q_proj` output to `(T, H, 2 * d)` first, then split along the last axis to get `q` and `gate`. Splitting the flat `(T, 2 * d * H)` tensor in half puts all of the first $H / 2$ heads into `q`. Normalize `q` and `k` over the last axis, transpose heads to the front (`(H, T, d)` and `(G, T, d)`), and call `ops.partial_rope` with both `cfg.rope_base` and `cfg.partial_rotary_factor` (its defaults use a different base and rotate every channel). Reshape `gate` to `(T, H * d)` in head order and pass it through a sigmoid before multiplying the merged attention output.

For a linear-attention layer, collect the nine tensors under `model.layers.{i}.linear_attn.` into a dict matching `ops.gated_delta_layer` and pass both `cfg.n_key_heads` and `cfg.norm_eps`, which cannot be inferred from tensor shapes.

---

For stage 3, ask what each layer needs from earlier calls to match a single full-sequence pass. A linear-attention layer already returns the `(conv_state, S)` pair it needs. A full-attention layer needs all earlier keys and values plus the number of prior tokens $P$, which is not passed separately. Read $P$ from the sequence dimension of the cached keys (`past[0].shape[1]`), and use it for both the new token positions and the attention mask width.

Rotate only the new keys at positions `P .. P + T - 1`, then concatenate them after the cached keys. The cached keys were already normalized and rotated when they were created.

---

With $P$ cached tokens and $T$ new tokens, the causal mask has shape `(T, P + T)`. Row $t$ is `True` for columns `0 .. P + t`, which you can build by comparing `arange(P + T)[None, :] <= positions[:, None]`. When $T = 1$ this mask is all `True`, so omitting the mask passes single-token decode steps but fails on multi-token chunked prefills.

Return a new list for `state` with one pair per layer in layer order, and never mutate the `state` list passed in by the caller.
