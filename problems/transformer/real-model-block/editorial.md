## The idea

A checkpoint is weights plus conventions. The weights are in the file; the
conventions are in the code that trained it, and a port has to restate each one.
For this block there are six, and it helps to sort them by where they come from.

**From `config.json`** (the names are IBM's):

- `attention_multiplier = 0.015625`. The scores are $q k^\top / 64$. Granite is
  trained with the μP parameterization (`"init_method": "mup"`), in which attention
  logits scale as $1/d$ rather than $1/\sqrt{d}$ so that they stay bounded as the
  width grows. For a head of 64 dimensions the two differ by a factor of 8, which is
  a temperature of 8 on every attention distribution in the model.
- `residual_multiplier = 0.263`. Each branch is damped before it joins the stream:
  $x + r \cdot f(\operatorname{norm}(x))$. It multiplies the branch output, both
  times, and never the stream.
- `rope_theta = 10000000`, a thousand times the usual base, which stretches the
  longest wavelength to match a 32k context.
- `rms_norm_eps = 1e-05`.

**From the tensor names and shapes:**

- `shared_mlp.input_linear.weight` with $2F$ rows is the gate and the up projection
  stacked, gate first. One matmul instead of two is a small speedup that every
  SwiGLU implementation could make; this checkpoint bakes it into the file.
- 1024 query features and 256 key and value features: 16 heads on 4.

**From nowhere but the reference code:**

- The RoPE layout. `transformers` rotates with `rotate_half`, pairing feature $i$
  with $i + d/2$. Meta's Llama code and the kit's `ops.rope` pair $2i$ with $2i + 1$.
  The two are the same rotation after a fixed permutation of the 64 features, so a
  model can be converted from one to the other by permuting rows of $W_q$ and $W_k$,
  but a given checkpoint works with exactly one.

Why the rotation only shows up through differences: with half-split pairs written as
complex numbers $z_i = a + \mathrm{i} b$, RoPE multiplies $z_i$ by
$e^{\mathrm{i} p \theta_i}$, and the score $\operatorname{Re} \sum_i q_i \overline{k_i}$
picks up $e^{\mathrm{i}(p_q - p_k)\theta_i}$. That is why shifting all positions by
100 changes nothing, and why the tests need positions with gaps to see whether
`positions` is used at all.

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

JAX and MLX differ in spelling only: `transpose(0, 2, 1, 3)`, `jnp.repeat` /
`mx.repeat` along axis 1, and `where(causal, scores, -inf)` for the mask.

## Common mistakes

- **Scaling by $1/\sqrt{d}$.** The default of every attention helper, including
  `ops.attention`, `F.scaled_dot_product_attention`, and
  `mx.fast.scaled_dot_product_attention` unless told otherwise. Eight times sharper
  attention.
- **Dropping the residual multiplier**, or applying it once. Without it each branch
  is 3.8 times too strong. With it on the attention branch only, the block is half
  right, which is harder to notice.
- **Multiplying the stream.** `(x + f(x)) * r` shrinks the residual stream by a
  factor of 14 per block.
- **Swapping gate and up.** `silu(up) * gate` has the same shape and the same
  statistics on random weights. The gate rows are the first $F$.
- **Adjacent-pair RoPE**, the layout of `ops.rope` and of the `rope` problem. Right
  model, wrong permutation.
- **Rotating clockwise.** `rotate_half` is $(a, b) \mapsto (-b, a)$. With the sign on
  the other half, the model sees every relative offset negated: the previous token
  looks like the next one.
- **Base 10,000.** Only the first pair ($\theta_0 = 1$) is unaffected.
- **Tiling KV heads.** Query heads 0 to 3 share KV head 0. `repeat` in PyTorch and
  `tile` elsewhere produce the order 0, 1, 2, 3, 0, 1, ... instead of 0, 0, 0, 0, 1, ...
- **Rotating the values**, or forgetting the causal mask: both are mistakes from
  `decoder-forward`, and both still apply.
- **Ignoring `positions`.** `arange(T)` is right for a prompt and wrong for every
  token generated with a KV cache, where the new token's position is the length of
  the cache.
- **A layer prefix without its dot.** Collecting the block's tensors with
  `name.startswith("model.layers.1")` also collects layers 10 to 19, and whichever
  comes last wins. A three-layer test model cannot show this; a 28-layer checkpoint
  does, at layers 1 and 2 only, which makes it a memorable afternoon.

## Complexity

$O(B\,T\,D\,(D + F))$ for the projections and the MLP, $O(B\,H\,T^2 d)$ for the
attention. For the real model at $T = 12$ that is about 100 million multiply-adds, a
few milliseconds.

## In the wild

`transformers` implements this block as `GraniteMoeHybridDecoderLayer` (the "MoE
hybrid" architecture with zero experts and no Mamba layers is a plain transformer),
and `mlx-lm` and llama.cpp each restate the same six conventions in their own model
files. The state entering layer 13 in the tests has a few entries above 10,000 while
most are below 10. Such massive activations are common in trained language models:
they act as fixed biases that the norms then remove, and they are a second reason to
compare ports block by block, since an absolute tolerance chosen for the logits means
nothing halfway up the stack. When you debug a port this way, capture the reference
states with forward hooks, as `fixtures/make.py` does, and compare the first block
that disagrees.
