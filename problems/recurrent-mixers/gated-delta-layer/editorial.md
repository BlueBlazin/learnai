## The idea

The layer is a pipeline with two memories in it.

The first is short and exact. A depthwise convolution of kernel $K$ lets every
channel of the projected input look at its own previous $K - 1$ values before the
queries, keys, and values are formed. A call that continues a sequence therefore needs
the last $K - 1$ rows of the *projected input* $P$. Not of `x`, because the projection
would have to be repeated, and not of the convolution's output, because the next
output is a function of earlier inputs, not of earlier outputs.

The second is long and lossy: the matrix $S$ of the delta rule, one per value head,
into which every earlier token has been written and partly overwritten.

Neither can be rebuilt from the other. $S$ is a fixed-size summary from which the
last inputs cannot be read back, and the last three inputs say nothing about what the
previous thousand tokens wrote. A serving system saves both, and a function with
"state in, state out" makes the invariant testable: for every split of a sequence, the
second call must receive exactly what the single call had in its hands at that
position. That is $\tilde P_{[T:]}$ and $S$ after token $T - 1$.

Taking the last $K - 1$ rows of the *concatenation* matters when a call is shorter
than the window. With $K = 4$ and a one-token call, the new state is two old rows and
one new row.

Two details of step 3 are easy to misread. The normalization puts its epsilon inside
the root, $u / \sqrt{\sum u^2 + 10^{-6}}$, which differs from the usual
$u / \max(\lVert u \rVert, \epsilon)$ only for tiny vectors. And the factor $d_k^{-1/2}$
on the query looks like it cannot matter, since $y = S q$ is linear in $q$ and the
norm of step 6 divides any constant factor out. It does matter wherever the read is
small against $\sqrt{\texttt{norm\_eps}} = 10^{-3}$, which is the case for the first
tokens of every sequence, when the memory is nearly empty: there
$y / \sqrt{\operatorname{mean}(y^2) + \epsilon} \approx y / \sqrt{\epsilon}$ and the
factor passes straight through.

## Reference solution (PyTorch)

```python
def gdn_layer(params, cfg, x, state):
    steps = x.shape[0]
    n_k, n_v, d_k, d_v = cfg.n_key_heads, cfg.n_value_heads, cfg.key_head_dim, cfg.value_head_dim
    if state is None:
        conv_state = torch.zeros(cfg.conv_kernel - 1, cfg.conv_dim)
        S = torch.zeros(n_v, d_v, d_k)
    else:
        conv_state, S = state

    projected = x @ params["in_proj_qkv.weight"].T
    padded = torch.cat((conv_state, projected), dim=0)
    w = params["conv1d.weight"][:, 0, :]
    mixed = sum(padded[j : j + steps] * w[:, j] for j in range(cfg.conv_kernel))
    conv_state = padded[steps:]
    mixed = F.silu(mixed)

    q = ops.l2norm(mixed[:, : cfg.key_dim].reshape(steps, n_k, d_k)) * d_k**-0.5
    k = ops.l2norm(mixed[:, cfg.key_dim : 2 * cfg.key_dim].reshape(steps, n_k, d_k))
    v = mixed[:, 2 * cfg.key_dim :].reshape(steps, n_v, d_v)
    q = q.repeat_interleave(n_v // n_k, dim=1)
    k = k.repeat_interleave(n_v // n_k, dim=1)

    a = x @ params["in_proj_a.weight"].T
    g = torch.exp(-torch.exp(params["A_log"]) * F.softplus(a + params["dt_bias"]))
    beta = torch.sigmoid(x @ params["in_proj_b.weight"].T)

    ys = []
    for t in range(steps):
        S = S * g[t, :, None, None]
        delta = beta[t, :, None] * (v[t] - torch.einsum("hvk,hk->hv", S, k[t]))
        S = S + delta[:, :, None] * k[t, :, None, :]
        ys.append(torch.einsum("hvk,hk->hv", S, q[t]))
    y = torch.stack(ys)

    z = (x @ params["in_proj_z.weight"].T).reshape(steps, n_v, d_v)
    y = ops.rms_norm(y, params["norm.weight"], eps=cfg.norm_eps) * F.silu(z)
    return y.reshape(steps, cfg.value_dim) @ params["out_proj.weight"].T, (conv_state, S)
```

`F.conv1d` with `groups=C` on the transposed `padded` computes the same sum; it is a
cross-correlation, so no flip is needed. The number of key heads is not recoverable
from the tensor shapes (`key_dim` could be split in several ways), which is why the
function takes `cfg`.

## Common mistakes

- **Convolution state reset on each call.** Correct for any single call, wrong for the
  first $K - 1$ tokens of every later one.
- **Memory reset on each call.** The mirror image: the convolution continues and the
  delta rule starts over.
- **Saving the convolution's outputs.** The shapes agree, since input and output have
  $C$ channels.
- **Cutting the new state from the new rows only.** `P[-(K - 1):]`, padded with zeros
  when the call is short, is right whenever $T \ge K - 1$ and loses saved rows when a
  call has fewer tokens, which is every decoding step.
- **Gate before norm.** $\operatorname{rms\_norm}(y \cdot \operatorname{silu}(z))$ is
  the order of the Mamba-2 block (`mamba2-ssd-step`); this layer normalizes first.
- **Tiling key heads.** `tile`/`repeat` gives 0, 1, 0, 1; the layer needs 0, 0, 1, 1.
  Invisible when $H_k = 1$ or $H_k = H_v$.
- **Dropping $d_k^{-1/2}$**, on the grounds that the norm cancels it. See above.
- **One RMS over the whole value width.** The norm weight has $d_v$ entries because
  the norm runs per head.
- **`normalize` for `l2norm`**, and a hard-coded `1e-6` for `cfg.norm_eps`. Both
  visible only on unusual inputs or configurations.
- **Updating the state in place.** `S.mul_(g)` on the tensor the caller passed in
  gives the right outputs and silently changes the caller's saved state, which breaks
  every use that continues twice from one state (beams, retries, a shared prefix).
- **Calling the kit's layer.** `ops.gated_delta_layer` is the answer; the statement
  rules it out and the harness rejects the call.
- **The layout of another library.** A port of the `transformers` code keeps the
  memory as $(H_v, d_k, d_v)$; it is consistent with itself and exchanges state with
  nothing else.

## Complexity

$O(T \cdot H_v\,d_v\,d_k)$ time for the recurrence plus $O(T \cdot C \cdot (D + K))$
for the projections and the convolution. The state is $(K - 1)\,C + H_v d_v d_k$
numbers whatever the sequence length. For Qwen3.5-0.8B that is
$3 \cdot 6144 + 16 \cdot 128 \cdot 128 = 280{,}576$ per layer.

## In the wild

Checked on 2026-11-21 by reading and running `transformers` 5.23.0, reading `mlx-lm`
0.32.2, and reading the `config.json` of `Qwen/Qwen3.5-0.8B` on the Hugging Face Hub.

- **The layer.** `Qwen3_5GatedDeltaNet.forward` in
  `transformers/models/qwen3_5/modeling_qwen3_5.py` performs the six steps. Its
  `l2norm` is `x * rsqrt((x * x).sum() + eps)`, its `Qwen3_5RMSNormGated` normalizes,
  multiplies by the weight, and then by `silu(gate)` (its comment: "Norm before
  gate"), and both delta-rule functions repeat key heads with `repeat_interleave`
  and multiply the query by `head_dim ** -0.5`. The public test "the same layer in
  Hugging Face Transformers" and a hidden one compare your function with that module
  holding the kit's weights, without a cache and with one
  (`fixtures/make.py` regenerates the arrays).
- **Shipped sizes.** `Qwen/Qwen3.5-0.8B`: 16 key heads and 16 value heads of 128,
  kernel 4, `rms_norm_eps` 1e-6, in 18 of 24 layers. The configuration class defaults
  to 32 value heads for 16 key heads, the case $r = 2$ of the tests.
- **State layouts.** `transformers` keeps the last $K$ projected inputs per layer,
  one more than the next token's convolution reads, as `(C, K)`, and the memory as
  `(Hv, dk, dv)`. `mlx-lm` keeps $K - 1$ rows and `(Hv, dv, dk)`, the layout used here.
- **One token or many.** With a cache and a single token `transformers` runs the
  per-token rule; every other call goes through the chunked form of
  `gated-delta-chunked`. The fixture exercises both.

Deliberate simplifications: no batch axis; no attention mask for padding tokens (the
library zeroes padded positions before the projection); float32 throughout (the
library computes the recurrence in float32 even when the model is in a narrower
type); the convolution has no bias, as shipped. The block around this sublayer (the
pre-norm, the residual, the MLP) is the subject of `hybrid-decoder-forward`.
