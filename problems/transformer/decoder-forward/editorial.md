## The idea

For a decoder with $L = \texttt{n\_layers}$ blocks, the forward pass moves the residual
stream through a fixed sequence of stages:

$$
E[\text{tokens}] \;\to\; \underbrace{\text{block}_0 \to \dots \to \text{block}_{L-1}}_{x \,\leftarrow\, x + f(\operatorname{norm}(x)) \text{, twice per block}} \;\to\; \operatorname{RMSNorm} \;\to\; W_{\text{head}}^\top .
$$

Every individual operation is already familiar. The challenge is matching the exact
conventions that the training code used, since none of them can be inferred from the
checkpoint tensors alone:

- **Where position enters.** There is no learned position embedding. Position enters
  only through the RoPE rotation applied per head to queries and keys inside every
  attention layer. Because the dot product of a rotated query and a rotated key depends
  only on the *difference* of their positions, the values stay unrotated so that
  attention mixes content without injecting absolute position.
- **Where the norms sit.** Two norms run inside each block on the sublayer inputs, and
  one final norm runs on the residual stream after the last block. That final norm is
  needed because pre-norm blocks never normalize the stream itself.
- **How the output head works.** With tied embeddings, one matrix serves two roles. Row
  $v$ of $E$ is both the input embedding for token $v$ and the output direction whose
  dot product with the final state produces the logit for token $v$. In that case
  `lm_head.weight` is absent from `params`.
- **How heads are laid out.** Each head is a contiguous slice of $d = \texttt{head\_dim}$
  features in the projection output. Queries have $H = \texttt{n\_heads}$ heads, keys and
  values have $H_{kv} = \texttt{n\_kv\_heads}$ heads, and with group size $G = H / H_{kv}$,
  `ops.attention` pairs query head $h$ with key/value head $\lfloor h / G \rfloor$.

The causality test is a reliable sanity check that requires no reference
implementation. If you modify tokens from position $t$ onward, every logit before
position $t$ must remain unchanged. That check immediately catches a missing mask, a
transposed mask, or accidental mixing across the sequence axis.

## Reference solution (PyTorch)

```python
def forward(params, cfg, tokens):
    batch, seq = tokens.shape
    positions = torch.arange(seq)
    causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool))

    def heads(t, n):  # (B, T, n * head_dim) -> (B, n, T, head_dim)
        return t.reshape(batch, seq, n, cfg.head_dim).transpose(1, 2)

    x = params["model.embed_tokens.weight"][tokens]
    for i in range(cfg.n_layers):
        p = f"model.layers.{i}."
        h = ops.rms_norm(x, params[p + "input_layernorm.weight"], cfg.norm_eps)
        q = heads(h @ params[p + "self_attn.q_proj.weight"].T, cfg.n_heads)
        k = heads(h @ params[p + "self_attn.k_proj.weight"].T, cfg.n_kv_heads)
        v = heads(h @ params[p + "self_attn.v_proj.weight"].T, cfg.n_kv_heads)
        q = ops.rope(q, positions, cfg.rope_base)
        k = ops.rope(k, positions, cfg.rope_base)
        attended = ops.attention(q, k, v, causal)
        attended = attended.transpose(1, 2).reshape(batch, seq, cfg.d_model)
        x = x + attended @ params[p + "self_attn.o_proj.weight"].T
        h = ops.rms_norm(x, params[p + "post_attention_layernorm.weight"], cfg.norm_eps)
        x = x + ops.swiglu(h, params[p + "mlp.gate_proj.weight"],
                           params[p + "mlp.up_proj.weight"], params[p + "mlp.down_proj.weight"])
    x = ops.rms_norm(x, params["model.norm.weight"], cfg.norm_eps)
    head = params["model.embed_tokens.weight" if cfg.tie_embeddings else "lm_head.weight"]
    return x @ head.T
```

The JAX and MLX versions differ only in API spelling, using `transpose(0, 2, 1, 3)` for
the head axis and `jnp.tril` or `mx.tril` for the causal mask.

## Common mistakes

- **Forgetting the final norm.** `model.norm.weight` is a single small vector used once
  outside the layer loop. Without it, the head receives a residual stream whose norm has
  grown across every block. Early logits may still keep a similar ranking, but their
  magnitudes become far too large and sampling temperatures stop behaving as intended.
- **Rotating the values.** Applying RoPE to `v` injects absolute position into the value
  vectors that attention copies, and nothing un-rotates them afterward.
- **Silently defaulting a missing head.** Calling `params.get("lm_head.weight", zeros)`
  or loading with `strict=False` turns the missing key of a tied checkpoint into zeros
  or random weights instead of an error, making every logit uniform. Choose the head
  matrix from `cfg.tie_embeddings` and index `params` directly so a mismatch fails
  loudly.
- **Always tying the head.** Using the embedding matrix as the output head even when
  `cfg.tie_embeddings` is `False` ignores the checkpoint's separate `lm_head.weight`.
- **Using the wrong RoPE pair layout.** This model rotates adjacent pairs $(2j, 2j+1)$,
  matching Meta's original Llama code. Hugging Face's `rotate_half` pairs feature $j$
  with feature $j + d/2$. Both conventions are valid, but a checkpoint trained with one
  produces garbage under the other. The two layouts differ by a permutation of the rows
  of `q_proj` and `k_proj`, which Hugging Face's Llama conversion script applies during
  export.
- **Applying RoPE before splitting heads.** Rotating the `(B, T, n_heads * head_dim)`
  projection as one wide vector spreads the rotation frequencies across the entire width
  instead of repeating them per head. It also gives queries and keys different frequency
  tables whenever $H_{kv} \ne H$, so it only matches for a single head.
- **Omitting the causal mask.** The final token's logits are still numerically correct
  because the last position can attend to every earlier token, so a test that checks
  only the last row will miss the bug.
- **Hard-coding hyperparameters.** `norm_eps` is `1e-5` in Llama and `1e-6` in Qwen and
  Gemma. `rope_base` is `10000` in many models, `500000` in Llama 3, and `1000000` in
  Qwen. Always read both from `cfg`.

One thing that is *not* a bug is starting positions at 1 instead of 0. Because RoPE
attention scores depend only on relative position differences, shifting every position
in a row by the same constant leaves the output unchanged as long as queries, keys, and
any later appended tokens share the same offset.

## Complexity

For batch size $B$, sequence length $T$, $L$ layers, model width $D$, $H$ query heads of
width $d$, MLP width $M$, and vocabulary size $V$, the time complexity is
$O(L\,B\,T\,D\,(D + M))$ for the projections and MLPs, $O(L\,B\,H\,T^2 d)$ for
attention, and $O(B\,T\,D\,V)$ for the output head. Peak attention memory is
$O(B\,H\,T^2)$ for one layer at a time. During autoregressive generation, only the last
row of logits is needed, and recomputing the full prefix for every new token makes
generation quadratic in length. A KV cache removes that redundant work.

## In the wild

Hugging Face's `LlamaModel.forward` followed by `lm_head` implements this exact pass
with the same parameter names. `llama.cpp` builds the same per-layer graph
(`build_norm`, `build_attn` with `ggml_rope_ext` on queries and keys only, and the
feed-forward block) followed by the final norm and output projection.

Hugging Face records tied embeddings under `tie_word_embeddings`. Small models often tie
them because the embedding table is a large fraction of total parameters, whereas
Llama 3 and most large models keep a separate head. Production serving engines also
avoid computing logits for every position during generation. Instead, they slice the
final token before the final norm and output head, which is the `last-position-head`
problem.
