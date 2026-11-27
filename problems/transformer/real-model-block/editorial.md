## The idea

A checkpoint consists of saved weights plus the conventions used to train them. Porting a model means reproducing each convention accurately. For Granite 4.0 350M, six conventions define this block, grouped by where they appear.

**From `config.json`** (using IBM's field names):

- `attention_multiplier = 0.015625`. Scores are scaled by $1/64$. Granite uses the μP parameterization (`"init_method": "mup"`), which scales attention logits by $1/d$ instead of $1/\sqrt{d}$ so logits stay bounded as width grows. For $d = 64$, the two factors differ by 8x, acting as a temperature of 8 on every attention distribution.
- `residual_multiplier = 0.263`. Both branch outputs are damped before being added to the residual stream: $x + r \cdot f(\operatorname{norm}(x))$. This factor multiplies only the branch output, never $x$.
- `rope_theta = 10000000`. Using a base 1000x larger than the standard 10,000 stretches the longest wavelength to cover a 32k context.
- `rms_norm_eps = 1e-05`.

**From tensor names and shapes:**

- `shared_mlp.input_linear.weight` has $2F$ rows, stacking the gate and up projections with the gate first. Fusing them turns two matrix multiplications into one.
- Query projections have 1024 outputs while key and value projections have 256, giving 16 query heads across 4 key/value heads.

**From the reference code alone:**

- Half-split RoPE. `transformers` uses `rotate_half`, pairing feature $i$ with feature $i + d/2$. Meta's Llama code and the kit's `ops.rope` pair $2i$ with $2i + 1$. The two layouts are equivalent up to a fixed permutation of the 64 head features, so a checkpoint can be converted by permuting rows of $W_q$ and $W_k$, but any unmodified checkpoint works with only one layout.

Notice that RoPE affects attention scores only through relative position differences. Writing each half-split pair as a complex number $z_i = a + \mathrm{i} b$, RoPE multiplies $z_i$ by $e^{\mathrm{i} p \theta_i}$. The inner product $\operatorname{Re} \sum_i q_i \overline{k_i}$ therefore depends only on $e^{\mathrm{i}(p_q - p_k)\theta_i}$. Shifting all positions by a constant leaves scores unchanged, which is why tests must include non-contiguous positions with gaps to verify that `positions` is actually used.

## Reference solution (PyTorch)

```python
def rope(x, positions):
    half = 32
    inv_freq = 1e7 ** (-torch.arange(half, dtype=torch.float32) / half)
    angles = positions.to(torch.float32)[:, None] * inv_freq
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1)
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1)
    rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * cos + rotated * sin


def granite_block(x, weights, layer, positions):
    p = f"model.layers.{layer}."
    batch, seq, width = x.shape

    def heads(h, name):
        out = h @ weights[p + f"self_attn.{name}.weight"].T
        return out.reshape(batch, seq, -1, 64).transpose(1, 2)

    h = ops.rms_norm(x, weights[p + "input_layernorm.weight"], 1e-5)
    q = rope(heads(h, "q_proj"), positions)
    k = rope(heads(h, "k_proj"), positions)
    v = heads(h, "v_proj")
    group = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(group, dim=1)
    v = v.repeat_interleave(group, dim=1)
    scores = (q @ k.transpose(-1, -2)) * (1 / 64)
    causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool))
    scores = scores.masked_fill(~causal, float("-inf"))
    mixed = (ops.softmax(scores) @ v).transpose(1, 2).reshape(batch, seq, width)
    x = x + 0.263 * (mixed @ weights[p + "self_attn.o_proj.weight"].T)

    h = ops.rms_norm(x, weights[p + "post_attention_layernorm.weight"], 1e-5)
    both = h @ weights[p + "shared_mlp.input_linear.weight"].T
    d_ff = both.shape[-1] // 2
    hidden = torch.nn.functional.silu(both[..., :d_ff]) * both[..., d_ff:]
    return x + 0.263 * (hidden @ weights[p + "shared_mlp.output_linear.weight"].T)
```

JAX and MLX differ only in API spelling: `transpose(0, 2, 1, 3)`, `jnp.repeat` or `mx.repeat` along axis 1, and `where(causal, scores, -inf)` for the causal mask.

## Common mistakes

- **Scaling by $1/\sqrt{d}$.** This is the default in `ops.attention`, `F.scaled_dot_product_attention`, and `mx.fast.scaled_dot_product_attention`, making attention 8x sharper than Granite expects.
- **Dropping the residual multiplier or applying it once.** Omitting $r = 0.263$ makes each branch 3.8x too strong. Applying it to only one branch leaves half the block wrong.
- **Multiplying the residual stream.** Computing `(x + f(x)) * r` shrinks the residual stream by roughly 14x per block.
- **Swapping gate and up.** Computing `silu(up) * gate` has the same shape and identical statistics on random weights. The first $F$ rows are the gate.
- **Using adjacent-pair RoPE.** `ops.rope` and the `rope` problem pair $(2i, 2i+1)$, which applies the wrong feature permutation for this checkpoint.
- **Rotating clockwise.** `rotate_half` maps $(a, b) \mapsto (-b, a)$. Flipping the sign negates every relative offset, making past tokens look like future tokens.
- **Using base 10,000.** Only the first frequency ($\theta_0 = 1$) matches. Every other pair rotates at the wrong rate.
- **Tiling KV heads.** Query heads 0 to 3 share KV head 0. Using `repeat` in PyTorch or `tile` in JAX/MLX produces `[0, 1, 2, 3, 0, 1, ...]` instead of `[0, 0, 0, 0, 1, ...]`.
- **Rotating values or omitting the causal mask.** Both classic decoder bugs still break this block.
- **Ignoring `positions`.** Using `arange(T)` works for contiguous prompts starting at 0, but fails for non-contiguous positions or cached decoding.
- **Omitting the trailing dot in the layer prefix.** Filtering keys with `name.startswith("model.layers.1")` also matches layers 10 through 19. A three-layer test model hides this bug, while a 28-layer checkpoint triggers it on layers 1 and 2.

## Complexity

Projections and the MLP take $O(B\,T\,D\,(D + F))$ operations, and attention takes $O(B\,H\,T^2 d)$. For Granite 4.0 350M at $T = 12$, that is roughly 100 million multiply-adds, running in a few milliseconds.

## In the wild

`transformers` implements this block as `GraniteMoeHybridDecoderLayer` (a plain transformer when configured with zero experts and no Mamba layers). Runtimes like `mlx-lm` and `llama.cpp` each re-implement the same six conventions in their Granite model files.

The hidden state entering layer 13 in the test suite has a few entries above 10,000 while most stay below 10. Massive activations like these are common in trained language models, where they act as fixed biases that layer normalization removes. They are another reason to validate ports block by block using forward hooks (as `fixtures/make.py` does) and inspect the first layer that diverges, since an absolute tolerance chosen for final logits is meaningless halfway through the network.
