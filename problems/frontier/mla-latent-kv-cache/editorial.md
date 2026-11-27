## The idea

Standard multi-head attention caches one key and one value per head and token, totaling $2 H d_h$ numbers per token. MLA compresses both into linear functions of a single latent $c_t = W^{DKV} x_t$ of width $d_c$, with $k^C_t = W^{UK} c_t$ and $v_t = W^{UV} c_t$. Caching $c_t$ is enough to reconstruct both. However, up-projecting all $S$ cached latents back to width $H d_h$ on every decode step burns compute and temporary memory, undoing much of the win. Because every projection on the content path is linear, we can fold them into the query and output sides instead.

**Absorbing keys into the query.** For head $h$, the content attention score between query $t$ and key $j$ is:

$$
q^C_{t,h} \cdot k^C_{j,h}
= \big(W^{UQ}_h c^Q_t\big)^\top \big(W^{UK}_h c_j\big)
= \big(\underbrace{(W^{UK}_h)^\top W^{UQ}_h}_{W^{QK}_h}\, c^Q_t\big) \cdot c_j .
$$

Because $W^{QK}_h$ is constant, we precompute it once when loading the model. At inference time, we map only the new query latent into the key-value latent space and take dot products directly against the cached latents.

**Absorbing values into the output.** Because the attention probabilities $a_{t,h,j}$ are scalars, they commute with linear projections:

$$
W^O_h\, o_{t,h} = W^O_h \sum_j a_{t,h,j} W^{UV}_h c_j
= \underbrace{W^O_h W^{UV}_h}_{W^{OV}_h} \sum_j a_{t,h,j}\, c_j .
$$

Each head computes a weighted sum of the cached *latents*, and one fixed matrix $W^{OV}_h$ per head projects that mix directly to the output space. In this absorbed form, MLA behaves like multi-query attention over a single shared vector of width $d_c$ with $H$ distinct queries, while each head's private $W^{QK}_h$ and $W^{OV}_h$ preserve the expressiveness of $H$ full heads.

**Why RoPE is decoupled.** RoPE multiplies keys and queries by position-dependent rotation matrices. If RoPE were applied to the up-projected content key, the dot product would become $(W^{UQ}_h c^Q_t)^\top R_{j-t}\, W^{UK}_h c_j$. The relative rotation $R_{j-t}$ sits between $W^{UQ}_h$ and $W^{UK}_h$, so $(W^{UK}_h)^\top R_{j-t} W^{UQ}_h$ changes for every pair of positions and cannot be precomputed. MLA avoids this by splitting queries and keys into a compressed content part with no position encoding and a small uncompressed RoPE part of width $d_r$. The RoPE key is projected directly from $x_t$ and shared across all heads, adding only $d_r$ floats per token. Because a key's rotation depends only on its own position, we cache $k^R_j$ already rotated.

**Attention scaling.** In the explicit definition, each head's concatenated query and key have width $d_h + d_r$, so scores are divided by $\sqrt{d_h + d_r}$. Matrix absorption changes how the dot product is factored, not its value. Because $d_h$ no longer appears in any absorbed tensor shape, it must be passed explicitly in `weights`.

## Reference solution (PyTorch)

```python
def absorb(w_uq, w_uk, w_uv, w_o, n_heads):
    d_cq, d_c, d_model = w_uq.shape[1], w_uk.shape[1], w_o.shape[0]
    uq = w_uq.reshape(n_heads, -1, d_cq)  # (H, d_h, d_cq)
    uk = w_uk.reshape(n_heads, -1, d_c)  # (H, d_h, d_c)
    uv = w_uv.reshape(n_heads, -1, d_c)  # (H, d_h, d_c)
    o = w_o.reshape(d_model, n_heads, -1).permute(1, 0, 2)  # (H, d, d_h)
    return uk.transpose(1, 2) @ uq, o @ uv


def mla_step(x_new, weights, cache):
    batch, new, _ = x_new.shape
    heads, d_head = weights["n_heads"], weights["d_head"]
    d_rope = weights["w_kr"].shape[0]
    past = 0 if cache is None else cache[0].shape[1]
    positions = past + torch.arange(new)

    c_new = x_new @ weights["w_dkv"].T
    kr_new = ops.rope(x_new @ weights["w_kr"].T, positions)
    if cache is None:
        c, k_rope = c_new, kr_new
    else:
        c = torch.cat((cache[0], c_new), dim=1)
        k_rope = torch.cat((cache[1], kr_new), dim=1)

    c_q = x_new @ weights["w_dq"].T
    q_latent = torch.einsum("hcq,btq->bhtc", weights["w_qk"], c_q)
    q_rope = (c_q @ weights["w_qr"].T).reshape(batch, new, heads, d_rope).transpose(1, 2)
    q_rope = ops.rope(q_rope, positions)

    scores = q_latent @ c.transpose(1, 2)[:, None] + q_rope @ k_rope.transpose(1, 2)[:, None]
    scores = scores / math.sqrt(d_head + d_rope)
    visible = torch.arange(past + new)[None, :] <= past + torch.arange(new)[:, None]
    probs = ops.softmax(torch.where(visible, scores, float("-inf")))

    mixed = probs @ c[:, None]
    out = torch.einsum("hdc,bhtc->btd", weights["w_ov"], mixed)
    return out, (c, k_rope)
```

Adding the singleton axis in `c.transpose(1, 2)[:, None]` gives shape `(B, 1, d_c, S + T)`, which broadcasts the single cached latent across all $H$ heads. The JAX and MLX references use the exact same structure with `jnp` and `mx`.

## Common mistakes

- **Multiplying factors in the wrong order.** Computing $(W^{UQ}_h)^\top W^{UK}_h$ produces the transpose of $W^{QK}_h$. Applying it to $c^Q$ raises a shape error unless $d_c = d_q$, in which case it fails silently.
- **Absorbing across heads.** Multiplying the full matrices $(W^{UK})^\top W^{UQ}$ sums the per-head products together. Because each head applies its own nonlinear softmax, each head must keep its own $W^{QK}_h$ and $W^{OV}_h$.
- **Splitting $W^O$ on the wrong axis.** Up-projections produce concatenated head vectors, so heads partition their *rows*. The output projection $W^O$ consumes concatenated head vectors, so heads partition its *columns*. Reshaping `w_o` along rows yields the expected shape with scrambled weights.
- **Applying RoPE to the content path.** Rotating the absorbed query latent and cached latent breaks the absorption identity because $W^{QK}_h$ was formed assuming no rotation sits between $W^{UK}_h$ and $W^{UQ}_h$.
- **Applying RoPE before splitting heads.** The rotary query consists of $H$ vectors of width $d_r$. Rotating it as a single vector of width $H d_r$ uses frequencies $10000^{-2i/(H d_r)}$, which do not match the shared key's frequencies.
- **Resetting positions to zero.** On a cached step, new tokens start at position $S$, not 0. Using `0..T-1` passes prefill tests but breaks incremental decoding.
- **Caching unrotated RoPE keys.** If `k_rope` is cached before rotation, the current step looks correct when using the rotated local copy, while every later step attends to unrotated keys.
- **Using the wrong scale factor.** Dividing by $\sqrt{d_h}$ ignores the RoPE dimensions. Dividing by $\sqrt{d_c + d_r}$—which happens if you feed `[c; k_rope]` into a generic attention kernel—distorts the softmax temperature of every head.
- **Caching more than the latent.** Caching the raw input $x_t$ or replicating $k^R_t$ across all $H$ heads yields the right outputs while inflating the cache to $d$ or $d_c + H d_r$ floats per token.
- **Using a square causal mask.** Just as in `kv-cache-append`, the attention score matrix has shape `(T, S + T)`, and multi-token chunks still need causal masking among the new tokens.

## Complexity

The cache holds $d_c + d_r$ floats per token and layer, compared to $2 H d_h$ for standard MHA. Per decode step and head, scoring against $S$ cached tokens takes $S (d_c + d_r)$ multiply-adds, and mixing latents takes $S\,d_c$. By contrast, the explicit path requires $S (d_h + d_r)$ and $S\,d_h$ per head *plus* up-projecting all cached latents on every step for $2 S \cdot H d_h \cdot d_c$ extra operations (or keeping the up-projected tensors in memory at full MHA cache size). Weight absorption itself runs once per layer at model load time.

## In the wild

- **DeepSeek-V2 paper.** MLA was introduced in "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model" (arXiv 2405.04434), Section 2.1 and Appendix C ("Full Formulas of MLA"). That paper defines the low-rank joint KV compression, query compression, decoupled RoPE with a shared key $k^R_t = \operatorname{RoPE}(W^{KR} h_t)$, the $\sqrt{d_h + d_h^R}$ scaling factor, and the cache size of $(d_c + d_h^R)\,l$ elements per token across $l$ layers versus $2 n_h d_h l$. It also gives the non-commutativity argument for decoupling RoPE so that $W^{UK}$ can merge into the query projection and $W^{UV}$ into $W^O$. DeepSeek-V2 uses $n_h = 128$, $d_h = 128$, $d_c = 512$, $d_h^R = 64$, and a query latent width of 1536.
- **Notation mapping.** The paper's matrices $W^{DKV}, W^{UK}, W^{UV}, W^{DQ}, W^{UQ}, W^{QR}, W^{KR}, W^O$ map directly to `w_dkv`, `w_uk`, `w_uv`, `w_dq`, `w_uq`, `w_qr`, `w_kr`, and `w_o` with identical shapes. The paper's $d_c'$ corresponds to `d_cq`.
- **Simplifications in this problem.** The paper states that weight absorption is possible without writing out the explicit per-head matrices. Here `w_qk`, `w_ov`, and the einsum layout make that absorption concrete. The paper also leaves the per-head RoPE layout of the $n_h d_h^R$ query entries implicit. Here each head's $d_h^R$ slice is rotated independently to match the shared key, using the adjacent-pair layout and base 10000 from `rope`. This problem isolates a single layer with small random weights, omitting normalization, residual connections, and the MLP. Production serving engines may attend in absorbed form during decoding and up-project during long prefills, while keeping the exact same latent cache layout.
