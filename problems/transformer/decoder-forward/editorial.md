## The idea

The forward pass is a residual stream with a fixed itinerary:

$$
E[\text{tokens}] \;\to\; \underbrace{\text{block}_0 \to \dots \to \text{block}_{L-1}}_{x \,\leftarrow\, x + f(\operatorname{norm}(x)) \text{, twice per block}} \;\to\; \operatorname{RMSNorm} \;\to\; W_{\text{head}}^\top .
$$

Nothing here is new arithmetic. What makes it a problem is that the model is defined
by a set of conventions that live in the code that trained it, and none of them can be
read off the weights:

- **Where position enters.** There is no position embedding. Position exists only as
  the RoPE rotation of queries and keys inside every attention, applied per head.
  Because the score of a rotated query and a rotated key depends only on the
  *difference* of their positions, the values must stay unrotated: they are the
  content that gets mixed, not part of the comparison.
- **Where the norms are.** Two per block, each on the input of a sublayer, plus one on
  the stream after the last block. The last one exists because pre-norm blocks never
  normalize the stream itself.
- **What the head is.** With tied embeddings the checkpoint has one matrix doing two
  jobs: row $v$ of $E$ is the input vector of token $v$ and also the direction whose
  dot product with the final state is the logit of token $v$. The key `lm_head.weight`
  is simply absent.
- **What a head is.** A slice of `head_dim` consecutive features of the projection.
  Queries have `n_heads` of them and keys and values `n_kv_heads`; `ops.attention`
  pairs query head $h$ with key/value head $\lfloor h / G \rfloor$.

A good way to gain confidence in a forward pass is the causality check in the tests:
change the tokens from position $t$ on, and the logits before $t$ must not move. It
needs no reference, and it fails for a missing mask, a mask on the wrong axis, and any
accidental mixing across time.

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

JAX and MLX differ only in spelling: `transpose(0, 2, 1, 3)` for the head axis and
`jnp.tril` / `mx.tril` for the mask.

## Common mistakes

- **Forgetting the final norm.** `model.norm.weight` is one small vector among dozens
  of tensors, used once, outside the loop. Without it the head sees a stream whose
  scale has grown through every block. The logits keep roughly their ranking early on
  and are far too large, so sampling temperatures stop meaning what they should.
- **Rotating the values.** Applying RoPE to `v` "for symmetry" injects absolute
  position into the content that attention copies. Nothing un-rotates it afterwards.
- **A head that silently defaults.** `params.get("lm_head.weight", zeros)`, or loading
  with `strict=False`, turns the missing key of a tied checkpoint into a head of
  zeros (or of random numbers) instead of an error. Every logit is then 0 and the
  model is uniform. Decide from the config, and index the dict so that a wrong
  decision fails loudly.
- **Always tying.** The opposite mistake: using the embedding matrix as the head when
  the checkpoint does have its own `lm_head.weight`.
- **The other RoPE pair layout.** This model rotates adjacent pairs $(2j, 2j+1)$, as
  Meta's original Llama code does. Hugging Face's `rotate_half` pairs feature $j$ with
  feature $j + d/2$. Both are valid conventions, and a model trained with one is
  garbage under the other. They are the same model up to a permutation of the rows of
  `q_proj` and `k_proj`, which is exactly what Hugging Face's Llama conversion script
  applies to those two matrices.
- **RoPE before the head split.** Rotating the `(B, T, n_heads * head_dim)` projection
  as one wide vector spreads the frequencies over the whole width instead of repeating
  them in every head, and gives queries and keys different frequency tables when
  $H_{kv} \ne H$. It is correct only for a single head.
- **No causal mask.** The last position's logits are still right, since it may see
  everything, so a test that looks only at the final row does not notice.
- **Hard-coded hyperparameters.** `norm_eps` is `1e-5` in Llama and `1e-6` in Qwen and
  Gemma; `rope_base` is 10,000 in most models, 500,000 in Llama 3, and 1,000,000 in
  Qwen. Read them from the config.

A mistake that is *not* one: starting the positions at 1 instead of 0. RoPE scores
depend only on position differences, so shifting every position of a row by the same
amount changes nothing. No test can detect it, and it does not matter, as long as the
shift is the same for queries and keys and consistent with the positions used for
tokens appended later.

## Complexity

With $L$ layers, width $D$, MLP width $M$, and vocabulary $V$: $O(L\,B\,T\,D\,(D + M))$
for the projections and MLPs, $O(L\,B\,H\,T^2 d)$ for attention, and $O(B\,T\,D\,V)$
for the head. Memory is $O(B\,H\,T^2)$ for the attention weights of one layer at a
time. During generation only the last row of logits is needed, and recomputing the
whole prefix for every new token makes generation quadratic in length; removing that
waste is the KV cache.

## In the wild

Hugging Face's `LlamaModel.forward` followed by `lm_head` is this function, with the
same parameter names. llama.cpp builds the same graph per layer (`build_norm`,
`build_attn` with `ggml_rope_ext` applied to the query and key tensors only, then the
feed-forward) and ends with the output norm and the output matrix. Tied embeddings are
recorded in the Hugging Face config as `tie_word_embeddings`; many small models tie
(the weight is a large share of a small model), while Llama 3 and most large models
keep a separate head. Real engines also never materialize the logits of every
position during generation: they slice the last position before the final norm and
the head, which is the `last-position-head` problem.
