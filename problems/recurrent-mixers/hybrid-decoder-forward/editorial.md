## The idea

The model is `decoder-forward` with two changes, and the state is where they meet.

**The mixer depends on the layer.** `cfg.layers[i]` picks it. A linear-attention layer
is the pipeline of `gated-delta-layer`, taken from the kit. A full-attention layer is
ordinary grouped-query attention with three additions: an RMSNorm on queries and keys
over the head dimension (so every score is a dot product of vectors of controlled
length), a rotation of only the first quarter of each head, and an output gate. The
gate is computed by the same projection as the query, which is why `q_proj` is twice as
wide as `o_proj` is deep: each head's slice of its output is `[query | gate]`, and the
attention output of that head is multiplied by `sigmoid(gate)` before `o_proj` mixes the
heads.

**The state is a list with two kinds of entries.** A linear-attention layer hands back
`(conv_state, S)`, whose size does not depend on how many tokens it has seen. An
attention layer hands back `(k, v)`, which grow by one column per token. The loop does
not need to know more than that: it passes entry `i` to layer `i` and collects what the
layer returns.

What the attention layer needs beyond its own entry is its place in the sequence, in
two spots. The new queries and keys are rotated with the positions $P, \dots, P + T - 1$,
and the mask has $P + T$ columns. $P$ is not an argument; it is the length of the cached
keys. Keys enter the cache finished (normalized and rotated at their own position) and
are never touched again, which is what makes a cached key equal to the key a single
pass would have computed.

The invariant to test against is the statement's property: after any prefix, the list
holds exactly what a single pass had in its hands at that point. For a linear-attention
layer that is the last $K - 1$ projected inputs and the memory; for an attention layer,
every key and value so far.

## Reference solution (PyTorch)

```python
DELTA = (
    "in_proj_qkv.weight", "in_proj_z.weight", "in_proj_a.weight", "in_proj_b.weight",
    "conv1d.weight", "A_log", "dt_bias", "norm.weight", "out_proj.weight",
)  # fmt: skip


def layer_types(n_layers: int, interval: int) -> list[str]:
    # Layers are counted from 1 here: the last layer of every group of `interval` attends.
    return [
        "full_attention" if (i + 1) % interval == 0 else "linear_attention"
        for i in range(n_layers)
    ]


def gated_attention(
    params: dict[str, torch.Tensor],
    prefix: str,
    cfg: HybridConfig,
    u: torch.Tensor,
    past: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Gated attention on T new rows `u`, after the `past[0].shape[1]` cached tokens."""
    new, heads, kv_heads, d = u.shape[0], cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    seen = 0 if past is None else past[0].shape[1]

    # Each head owns 2 * d consecutive outputs of q_proj: its query, then its gate.
    both = (u @ params[prefix + "q_proj.weight"].T).reshape(new, heads, 2 * d)
    q, gate = both[..., :d], both[..., d:]
    k = (u @ params[prefix + "k_proj.weight"].T).reshape(new, kv_heads, d)
    v = (u @ params[prefix + "v_proj.weight"].T).reshape(new, kv_heads, d)

    # Norm over the head dimension, then heads first, then the rotation at the
    # positions these tokens have in the whole sequence.
    q = ops.rms_norm(q, params[prefix + "q_norm.weight"], cfg.norm_eps).transpose(0, 1)
    k = ops.rms_norm(k, params[prefix + "k_norm.weight"], cfg.norm_eps).transpose(0, 1)
    v = v.transpose(0, 1)
    positions = torch.arange(seen, seen + new)
    q = ops.partial_rope(q, positions, cfg.rope_base, cfg.partial_rotary_factor)
    k = ops.partial_rope(k, positions, cfg.rope_base, cfg.partial_rotary_factor)

    # The cache holds finished keys: normalized and rotated once, never again.
    if past is not None:
        k = torch.cat((past[0], k), dim=1)
        v = torch.cat((past[1], v), dim=1)

    # Row t is the token at position seen + t: it sees keys 0 .. seen + t.
    mask = torch.arange(seen + new)[None, :] <= positions[:, None]
    attended = ops.attention(q, k, v, mask).transpose(0, 1).reshape(new, heads * d)
    gated = attended * torch.sigmoid(gate.reshape(new, heads * d))
    return gated @ params[prefix + "o_proj.weight"].T, (k, v)


def hybrid_forward(
    params: dict[str, torch.Tensor],
    cfg: HybridConfig,
    tokens: torch.Tensor,
    state: list[tuple[torch.Tensor, torch.Tensor]] | None,
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
    past = [None] * cfg.n_layers if state is None else state
    new_state = []  # a new list: the caller's is left as it was
    x = params["model.embed_tokens.weight"][tokens]
    for i, kind in enumerate(cfg.layers):
        layer = f"model.layers.{i}."
        u = ops.rms_norm(x, params[layer + "input_layernorm.weight"], cfg.norm_eps)
        if kind == "linear_attention":
            delta = {name: params[f"{layer}linear_attn.{name}"] for name in DELTA}
            mixed, kept = ops.gated_delta_layer(u, delta, past[i], cfg.n_key_heads, cfg.norm_eps)
        else:
            mixed, kept = gated_attention(params, layer + "self_attn.", cfg, u, past[i])
        new_state.append(kept)
        x = x + mixed
        u = ops.rms_norm(x, params[layer + "post_attention_layernorm.weight"], cfg.norm_eps)
        x = x + ops.swiglu(
            u,
            params[layer + "mlp.gate_proj.weight"],
            params[layer + "mlp.up_proj.weight"],
            params[layer + "mlp.down_proj.weight"],
        )
    x = ops.rms_norm(x, params["model.norm.weight"], cfg.norm_eps)
    return x @ params["model.embed_tokens.weight"].T, new_state
```

`gated_attention` is 20 lines because `ops.attention` already does the grouped-query
pairing, the $1/\sqrt d$, and the masked softmax. With a single new token the mask is
one row of `True`.

## Common mistakes

**In the schedule**

- **Counting from zero.** `i % interval == 0` makes layers 0, 4, 8 attend. The
  checkpoint's tensor names then do not match (`layers.0.linear_attn.*` exists,
  `layers.0.self_attn.*` does not).
- **Recomputing the schedule in the forward pass.** `Qwen3_5TextConfig` stores
  `layer_types` and applies the interval only when the list is absent; a config may
  carry any list.

**In one pass**

- **Splitting query and gate across the flat projection.** `chunk(2)` on the
  `(T, 2dH)` output gives the first half of the heads' slices to the queries. The
  split is on the last axis after the reshape to `(T, H, 2d)`.
- **No output gate**, or the wrong one. `silu(gate)` is the gate of the delta layer's
  norm; here it is a sigmoid.
- **No QK-norm, or the norm after the rotation.** The gains differ per channel, and
  the rotation mixes channel $i$ with channel $i + r/2$, so the two orders differ.
- **The rotation's defaults.** `ops.partial_rope(x, positions)` rotates every channel
  with base 10000. Both `cfg.rope_base` and `cfg.partial_rotary_factor` have to be
  passed. `ops.rope` on the first channels pairs `(0, 1), (2, 3)`, which is the layout
  of `decoder-forward` and not this model's.
- **Epsilons left at a default.** `ops.rms_norm` defaults to `1e-5` and
  `ops.gated_delta_layer` to `1e-6`; the model has one `cfg.norm_eps` for every norm.
- **The key-head count left out.** `ops.gated_delta_layer` cannot read it from the
  shapes and assumes one key head per value head.
- **No final norm.** As in `decoder-forward`.

**With state**

- **A recurrent layer restarted on every call**, or an attention layer called without
  its cache. Each is right for any single call.
- **Positions that restart at 0.** The new keys are then rotated as if they opened the
  sequence, while the cached ones keep their true positions.
- **Cached keys rotated again**, by concatenating before the rotation.
- **A cache that is not grown**: returning only the new keys and values.
- **No mask when there is a cache.** True for a decode step of one token, wrong for a
  chunked prefill, where the first token of a chunk would see the rest of it.
- **Writing into the caller's list.** `state[i] = kept` gives the right logits and
  changes the state the caller kept, which breaks every use that continues twice from
  one state (two samples from one prompt, a retry, a shared prefix).
- **Calling the kit's forward pass.** `hybrid_forward_ref` is the answer; the statement
  rules it out and the harness rejects the call.

## Complexity

For $T$ new tokens after $P$ cached ones: a linear-attention layer costs
$O(T \cdot H_v d_v d_k)$ for the recurrence whatever $P$ is, and an attention layer
$O(T (P + T) H d)$. The state is constant for the first kind and
$2\,G\,d\,(P + T)$ numbers for the second. In Qwen3.5-0.8B that is 280,576 numbers
per delta layer (18 layers) against 1,024 per token per attention layer (6 layers): at
4,096 tokens of context the six attention caches hold 25 million numbers and the 18
recurrent states 5 million.

## In the wild

Checked on 2026-11-21 by reading and running `transformers` 5.23.0, reading `mlx-lm`
0.32.2, and from the `config.json` of `Qwen/Qwen3.5-0.8B` as recorded in this
project's curriculum notes (24 layers, attention at layers 3, 7, 11, 15, 19, 23; 8
query heads on 2 key-value heads of width 256 at model width 1024; `rope_theta` 1e7 and
`partial_rotary_factor` 0.25; tied embeddings; `rms_norm_eps` 1e-6).

- **The layer.** `Qwen3_5DecoderLayer` in
  `transformers/models/qwen3_5/modeling_qwen3_5.py` is the block of step 2, and
  `Qwen3_5Attention.forward` the five steps of the attention mixer: `torch.chunk(...,
  2, dim=-1)` on the projection viewed as `(..., heads, 2 * head_dim)`, `q_norm` and
  `k_norm` on the head dimension, `apply_rotary_pos_emb`, the cache update, and
  `attn_output * torch.sigmoid(gate)` before `o_proj`.
- **The schedule.** `Qwen3_5TextConfig.__post_init__` fills `layer_types` from
  `full_attention_interval` (default 4) only when no list is given. `mlx-lm`'s
  `DecoderLayer` applies the interval rule directly.
- **The state.** `mlx-lm`'s `make_cache` returns
  `[ArraysCache(size=2) if l.is_linear else KVCache() for l in self.layers]`, the list
  of this problem. `transformers` puts both kinds in one `DynamicCache` whose layers
  hold `keys`/`values` or `conv_states`/`recurrent_states`.
- **The fixture.** `fixtures/make.py` builds `Qwen3_5ForCausalLM` from a tiny
  `Qwen3_5TextConfig`, copies the kit's weights into it, and records logits for one
  call and for two chunked runs with a `DynamicCache`, and the final cache, for three
  layer schedules. The reference agrees with them to `4e-5` on logits of magnitude 7.
  The library runs a multi-token call through a chunked form of the delta rule and a
  one-token call that continues a cache through the per-token rule (the script asserts
  both ran), so the comparison covers both.

Deliberate simplifications, all of them: no batch axis and no padding mask; float32
throughout; norm weights are plain gains, where the checkpoint stores its block norms,
final norm, and QK norms zero-centered (`x_normed * (1 + weight)`, which `mlx-lm` undoes
at load time and which `real-hybrid-logits` teaches); small sizes, with head counts in other ratios than the
shipped ones (the checkpoint has 4 query heads per key-value head and one key head per
value head; the tests have 2, and one key head per two value heads); text positions
only (the library's rotary embedding also carries the three position axes of image
tokens, which coincide for text); the state layout of `gated-delta-layer` for the
recurrent layers (`transformers` keeps $K$ rows of convolution state and the memory
transposed); no multi-token-prediction head and no vision tower.
