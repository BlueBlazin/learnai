## The idea

Standard attention caches a key and a value per head and token: $2 H d_h$ numbers.
MLA makes both a linear function of one latent $c_t = W^{DKV} x_t$ of width $d_c$:
$k^C_t = W^{UK} c_t$ and $v_t = W^{UV} c_t$. Caching $c_t$ is enough to recover
them. But recovering them on every step means up-projecting all $S$ cached latents
to width $H d_h$ again and again, which gives back in compute and transient memory
what the cache saved. The better move uses the fact that everything in sight is
linear.

**Keys into the query.** For head $h$,

$$
q^C_{t,h} \cdot k^C_{j,h}
= \big(W^{UQ}_h c^Q_t\big)^\top \big(W^{UK}_h c_j\big)
= \big(\underbrace{(W^{UK}_h)^\top W^{UQ}_h}_{W^{QK}_h}\, c^Q_t\big) \cdot c_j .
$$

The matrix $W^{QK}_h$ is fixed, so the query can be mapped once into the latent
space and scored directly against the cached latents.

**Values into the output.** The softmax weights are scalars, so

$$
W^O_h\, o_{t,h} = W^O_h \sum_j a_{t,h,j} W^{UV}_h c_j
= \underbrace{W^O_h W^{UV}_h}_{W^{OV}_h} \sum_j a_{t,h,j}\, c_j .
$$

Each head mixes the *latents* and one fixed matrix per head maps the mix to the
output. In this form MLA looks like multi-query attention with a single shared key
and value of width $d_c$ and $H$ different queries, except that each head reads
the shared vector through its own matrices, which is where the capacity of $H$
full heads comes from.

**Why RoPE is kept apart.** RoPE multiplies a key by a rotation $R_j$ that depends
on its position. If it were applied to the up-projected key, the score would be
$(W^{UQ}_h c^Q_t)^\top R_{j-t}\, W^{UK}_h c_j$: a position-dependent matrix sits
between the two up-projections, the product $(W^{UK}_h)^\top R_{j-t} W^{UQ}_h$ is
different for every pair of positions, and nothing can be precomputed. So MLA
splits queries and keys into a content part, which is compressed and carries no
position, and a small RoPE part of width $d_r$, which is rotated and is not
compressed. The RoPE key is computed from $x_t$ directly and shared by all heads,
so it costs $d_r$ numbers per token, and it is cached already rotated: the rotation
of a key depends only on the key's own position.

**The scale.** The model's queries and keys are $d_h + d_r$ wide, and the scores
are divided by $\sqrt{d_h + d_r}$. Absorption rearranges how the same score is
computed; it does not change the score. After absorption $d_h$ appears in no
tensor shape, which is why it is passed as a number.

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

`c.transpose(1, 2)[:, None]` is `(B, 1, d_c, S + T)`: the head axis of size one is
what "shared by all heads" means in code. The JAX and MLX references are the same
lines with `jnp` and `mx`.

## Common mistakes

- **The product in the wrong order.** $(W^{UQ}_h)^\top W^{UK}_h$ is the transpose
  of $W^{QK}_h$. It maps a key latent toward the query latent space, and applied to
  $c^Q$ it is a shape error unless $d_c = d_q$, in which case it is a silent one.
- **Absorbing across heads.** $(W^{UK})^\top W^{UQ}$ with the full matrices is the
  sum over heads of the per-head products. It would be right if the softmax were
  linear. Each head has its own softmax, so each needs its own matrix.
- **Splitting $W^O$ on the wrong axis.** Heads are *columns* of the output
  projection (it consumes the concatenated head outputs) and *rows* of the
  up-projections (they produce them). Reshaping `w_o` as if heads were rows gives a
  tensor of the right shape with scrambled entries.
- **RoPE on the content part.** Rotating the absorbed query latent and the cached
  latent "because queries and keys get RoPE" is the very coupling the design
  removes: $W^{QK}_h$ was computed with no rotation between its factors.
- **RoPE before the head split.** The RoPE query is $H$ vectors of width $d_r$.
  Rotating it as one vector of width $H d_r$ uses frequencies
  $10000^{-2i/(H d_r)}$, which do not match the shared key's.
- **Positions that restart.** In a decode step the new token is at position $S$,
  not 0. The prefill is right and generation is wrong.
- **Caching the key before rotation.** The current call uses the rotated key and
  looks right; every later call scores against cached keys without position.
- **The wrong scale.** $\sqrt{d_h}$ forgets that the RoPE part widened queries and
  keys. $\sqrt{d_c + d_r}$, the width of what is cached (and what a generic
  attention kernel applied to the concatenation $[c; k^R]$ would divide by),
  changes the temperature of every head.
- **Caching more than the latent.** Caching the layer input and projecting it down
  on each step, or storing the shared RoPE key once per head, gives correct
  outputs and a cache of $d$ or $d_c + H d_r$ numbers per token.
- **The mask.** As in `kv-cache-append`: with a cache the mask is not square, and a
  chunk of several new tokens still needs it.

## Complexity

Cache: $d_c + d_r$ numbers per token and layer, against $2 H d_h$. Per step and
head, scoring a query against $S$ cached tokens costs $S (d_c + d_r)$
multiply-adds and mixing costs $S\,d_c$. The explicit route costs
$S (d_h + d_r)$ and $S\,d_h$ per head, after an up-projection of all cached
latents that costs $2 S \cdot H d_h \cdot d_c$ on every step, or a cache as large as
standard attention's if the up-projected tensors are kept. Absorption itself is
done once per layer when the model is loaded.

## In the wild

- **The source** is "DeepSeek-V2: A Strong, Economical, and Efficient
  Mixture-of-Experts Language Model" (arXiv 2405.04434), section 2.1 and Appendix
  C, "Full Formulas of MLA". Established there, and reproduced in this problem's
  definition: the low-rank joint compression of keys and values; the query
  compression; the decoupled RoPE with a shared key $k^R_t =
  \operatorname{RoPE}(W^{KR} h_t)$; the scale $\sqrt{d_h + d_h^R}$; that only
  $c^{KV}_t$ and $k^R_t$ need to be cached, $(d_c + d_h^R)\,l$ elements per token
  for $l$ layers against $2 n_h d_h l$; and that $W^{UK}$ can be merged into the
  query up-projection and $W^{UV}$ into $W^O$ so that keys and values need not be
  computed. The paper gives the argument for decoupling used above: with RoPE on
  the compressed keys, $W^{UK}$ could no longer be absorbed, because a rotation
  tied to the current position would sit between the matrices and matrix
  multiplication does not commute. Its sizes are $n_h = 128$, $d_h = 128$,
  $d_c = 512$, $d_h^R = 64$, and a query latent of 1536.
- **Notation.** The paper's $W^{DKV}, W^{UK}, W^{UV}, W^{DQ}, W^{UQ}, W^{QR},
  W^{KR}, W^O$ are this problem's `w_dkv`, `w_uk`, `w_uv`, `w_dq`, `w_uq`, `w_qr`,
  `w_kr`, `w_o`, with the same shapes; its $d_c'$ is `d_cq`.
- **What this problem simplifies or adds.** The paper states that the absorption
  is possible and does not write the absorbed matrices down; $W^{QK}_h$ and
  $W^{OV}_h$, their per-head layout, and the einsum formulation are this problem's
  way of making it concrete. The paper does not specify how RoPE treats the
  $n_h d_h^R$ numbers of the query's RoPE part; this problem rotates each head's
  $d_h^R$ numbers on its own, the reading under which each head's RoPE query
  matches the shared key, and uses the interleaved pairing and base 10000 of the
  `rope` problem. The weights here are small random matrices, there is one layer,
  and normalization, the residual stream, and the MLP are left out. Whether a
  production engine attends in absorbed form, or up-projects the cached latents and
  reuses its standard attention kernel, is an implementation choice that this
  problem does not describe; the cache layout is the same either way.
