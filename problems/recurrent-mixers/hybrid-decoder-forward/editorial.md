## The idea

This model is `decoder-forward` with two changes that meet in the per-layer state.

**Each layer selects its mixer from `cfg.layers[i]`.** A linear-attention layer delegates directly to `ops.gated_delta_layer` from the kit. A full-attention layer is standard grouped-query attention with three additions:

1. An RMSNorm on queries and keys across the head dimension, keeping dot-product magnitudes controlled.
2. Partial rotary embeddings applied only to the first fraction of each head's channels.
3. A per-channel output gate computed alongside the query in `q_proj`. That is why `q_proj` has twice as many output rows as `o_proj` has input columns. Each head's slice of `q_proj` is `[query | gate]`, and the head's attention output is multiplied by `sigmoid(gate)` before `o_proj` mixes across heads.

**The state list holds two different kinds of entries.** A linear-attention layer returns `(conv_state, S)`, whose size never grows with sequence length. A full-attention layer returns `(k, v)`, which grows by one position per token. The outer layer loop stays uniform. It passes `past[i]` to layer `i` and appends the returned pair to a new state list.

Each full-attention layer also needs to know how many tokens $P$ came before the current call, both to rotate new queries and keys at positions $P, \dots, P + T - 1$ and to build the `(T, P + T)` causal mask. Because $P$ is not a separate argument, read it from the sequence dimension of the cached keys (`past[0].shape[1]`). Keys are normalized and rotated once before entering the cache and are never modified again, so cached keys match what a single full-sequence pass would have produced.

Streaming equivalence requires that after processing any prefix, the state list holds the exact same tensors that a single pass would have had at that point. For linear-attention layers, that is the last $K - 1$ projected inputs and the recurrent memory $S$. For full-attention layers, that is every normalized, rotated key and every value seen so far.

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

`gated_attention` stays concise because `ops.attention` handles grouped-query head pairing, $1 / \sqrt{d}$ scaling, and masked softmax. When $T = 1$, `mask` reduces to a single row of `True`.

## Common mistakes

**In the schedule**

- **Zero-based modulo.** Checking `i % interval == 0` places attention at layers 0, 4, 8 instead of 3, 7, 11. The checkpoint keys then mismatch because `layers.0.linear_attn.*` exists while `layers.0.self_attn.*` does not.
- **Ignoring `cfg.layers`.** `Qwen3_5TextConfig` supports an explicit `layer_types` list and falls back to the interval rule only when no list is given. Always read `cfg.layers` inside `hybrid_forward`.

**In one pass**

- **Splitting query and gate before reshaping.** Calling `chunk(2)` on the flat `(T, 2 * d * H)` projection assigns the first half of the heads to queries and the second half to gates. Reshape to `(T, H, 2 * d)` first, then split along the last axis.
- **Omitting the output gate or using SiLU.** SiLU is used inside the delta layer's norm, whereas the attention output gate uses a logistic sigmoid.
- **Omitting QK-norm or applying it after rotation.** Because the norm gains vary per channel and rotary embeddings mix channel $i$ with channel $i + r / 2$ (where $r$ is the number of rotated channels), normalizing after rotation gives a different result.
- **Relying on `ops.partial_rope` defaults.** Calling `ops.partial_rope(x, positions)` without arguments uses base `10000` and rotates every channel. Pass both `cfg.rope_base` and `cfg.partial_rotary_factor`. Also avoid `ops.rope`, which pairs adjacent channels `(0, 1), (2, 3)` instead of the half-split layout used here.
- **Leaving norm epsilons at defaults.** `ops.rms_norm` defaults to `1e-5` and `ops.gated_delta_layer` defaults to `1e-6`, while the model requires `cfg.norm_eps` everywhere.
- **Omitting `cfg.n_key_heads`.** `ops.gated_delta_layer` cannot infer the key-head count from tensor shapes alone and defaults to matching the value-head count.
- **Forgetting the final RMSNorm.** Apply `model.norm.weight` before multiplying by `embed_tokens.weight.T`.

**With state**

- **Dropping per-layer state.** Passing `None` to either the linear-attention layers or the full-attention layers passes every single-call test and fails on continuations.
- **Restarting positions at 0.** Rotating new queries and keys with `arange(T)` instead of `arange(P, P + T)` misaligns them against cached keys that kept their true positions.
- **Rotating cached keys a second time.** Concatenating new keys onto cached keys before calling `ops.partial_rope` re-rotates the cached keys.
- **Returning only new keys and values.** The attention cache must hold all $P + T$ keys and values, not just the $T$ new ones.
- **Omitting the causal mask when `past` is present.** Skipping the mask works for single-token decode steps ($T = 1$) but fails on multi-token chunked prefills where earlier tokens in the chunk would attend to later ones.
- **Mutating the caller's `state` list.** Assigning `state[i] = kept` gives the right logits on a linear run but corrupts the saved state when a caller branches twice from the same prefix.
- **Calling the kit's reference forward pass.** `hybrid_forward_ref` and `gated_attention_ref` are forbidden by the statement and rejected by the harness.

## Complexity

For $T$ new tokens after $P$ cached tokens, a linear-attention layer takes $O(T \cdot H_v d_v d_k)$ time for its recurrence regardless of $P$, while a full-attention layer takes $O(T (P + T) H d)$ time. State size is constant for linear-attention layers and $2 G d (P + T)$ floats for full-attention layers. In Qwen3.5-0.8B, that is 280,576 floats per delta layer (18 layers) versus 1,024 floats per token per attention layer (6 layers). At 4,096 tokens of context, the 6 attention caches hold about 25 million floats while all 18 recurrent states together hold 5 million.

## In the wild

Checked on 2026-11-21 by reading and running `transformers` 5.23.0, reading `mlx-lm` 0.32.2, and checking the `config.json` of `Qwen/Qwen3.5-0.8B` recorded in this project's curriculum notes (24 layers, attention at layers 3, 7, 11, 15, 19, 23, 8 query heads on 2 key-value heads of width 256 at model width 1024, `rope_theta` `1e7`, `partial_rotary_factor` `0.25`, tied embeddings, and `rms_norm_eps` `1e-6`).

- **Layer and attention.** `Qwen3_5DecoderLayer` in `transformers/models/qwen3_5/modeling_qwen3_5.py` implements the block of step 2, and `Qwen3_5Attention.forward` implements the five attention steps: splitting `q_proj` viewed as `(..., heads, 2 * head_dim)` with `torch.chunk(..., 2, dim=-1)`, applying `q_norm` and `k_norm` over the head dimension, calling `apply_rotary_pos_emb`, updating the cache, and multiplying by `torch.sigmoid(gate)` before `o_proj`.
- **Schedule.** `Qwen3_5TextConfig.__post_init__` fills `layer_types` from `full_attention_interval` (default 4) only when no explicit list is provided. `mlx-lm`'s `DecoderLayer` applies the interval rule directly.
- **State.** `mlx-lm`'s `make_cache` returns `[ArraysCache(size=2) if l.is_linear else KVCache() for l in self.layers]`, matching this problem's list layout. `transformers` stores both kinds in a single `DynamicCache` whose layers hold either `keys`/`values` or `conv_states`/`recurrent_states`.
- **Fixture verification.** `fixtures/make.py` builds `Qwen3_5ForCausalLM` from a small `Qwen3_5TextConfig`, copies the kit's weights into it, and records logits and final caches for single-call and two-chunk runs across three layer schedules. The reference agrees to within `4e-5` on logits of magnitude 7. Because the library uses a chunked delta rule for multi-token calls and a per-token rule for single-token cached calls (the script asserts both ran), the fixture checks both paths.

Deliberate simplifications in this problem:
- No batch axis and no padding mask, with float32 used throughout.
- Plain multiplicative RMSNorm gains rather than the checkpoint's zero-centered `1 + weight` convention (which `mlx-lm` folds at load time and `real-hybrid-logits` covers).
- Smaller dimensions and different head ratios than the shipped model (the checkpoint uses 4 query heads per KV head and 1 key head per value head, whereas the tests use 2 query heads per KV head and 1 key head per 2 value heads).
- Text positions only (omitting the three-axis image position IDs that coincide for text).
- The `gated-delta-layer` state layout for recurrent layers (`transformers` keeps $K$ rows of convolution state and a transposed memory matrix), with no multi-token-prediction head and no vision tower.
