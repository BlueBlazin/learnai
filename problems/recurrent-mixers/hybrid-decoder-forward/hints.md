Start with `state=None` and ignore what you return as state: stage 2 only reads the
logits. The block structure is `decoder-forward` with the mixer chosen per layer by
`cfg.layers[i]`. For the attention mixer, write down the shape of every tensor between
`q_proj` and `o_proj` before writing code. Where do the $2dH$ columns of the query
projection go, and which of the resulting tensors are normalized, which rotated, and
which neither?

---

The query projection is reshaped to `(T, H, 2 * d)` first and cut along the last axis
second; cutting the flat `(T, 2 * d * H)` output in the middle gives the first $H / 2$
heads everything. Norm, then heads to the front (`(H, T, d)` and `(G, T, d)`), then
`ops.partial_rope` with both `cfg.rope_base` and `cfg.partial_rotary_factor`: its
defaults are another base and "rotate every channel". The gate is reshaped back to
`(T, H * d)`, in the same head order as the merged attention output, and goes through a
sigmoid.

For a linear-attention layer the only work is collecting the nine tensors under the
names `ops.gated_delta_layer` expects and passing it the head count and the epsilon,
which it cannot read from the shapes.

---

For stage 3, ask of each layer: what does a second call need in order to compute
exactly what one long call would have computed? A linear-attention layer answers with
the pair it already returns. An attention layer needs every earlier key and value, and
one number that is nowhere in the arguments: how many tokens came before. That number
decides two things, the positions the new queries and keys are rotated with and the
width of the mask, and it can be read off the cached keys.

Rotate only the new keys, then put them after the cached ones. The cached keys were
rotated when they were new.

---

The mask of a call with $P$ cached tokens and $T$ new ones has $T$ rows and $P + T$
columns, and row $t$ is `True` up to and including column $P + t$: a comparison of
`arange(P + T)` against the new positions. With $T = 1$ it is all `True`, which is why
a decode step that passes no mask works and a chunked prefill that copies it does not.

Build the returned state as a new list, one pair per layer in layer order, and leave
the list you were given alone: the tests continue twice from one state.
