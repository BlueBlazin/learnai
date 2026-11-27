## The idea

A Gated DeltaNet layer carries two separate memories across calls.

The first memory is short and exact. A depthwise causal convolution of kernel size $K$ mixes each channel of the projected input $P$ with its previous $K - 1$ values before splitting out queries, keys, and values. Continuing a sequence therefore requires saving the last $K - 1$ rows of $P$. Saving raw `x` would force re-running the input projection, and saving the convolution's output would be wrong because each output depends on earlier inputs, not earlier outputs.

The second memory is long and lossy. It is the matrix $S$ of the gated delta rule, one per value head, which accumulates and overwrites associations across the entire prefix.

Neither state can be reconstructed from the other. $S$ is a fixed-size summary from which the last few inputs cannot be recovered, and the last $K - 1$ rows of $P$ know nothing about earlier tokens. Threading `(conv_state, S)` explicitly makes correctness directly testable. For any split of a sequence into two calls, the second call must receive the exact values a single call had at that boundary, namely $\tilde P_{[T:]}$ and $S$ after token $T - 1$.

Slicing the last $K - 1$ rows from the *concatenated* tensor $\tilde P$ is essential whenever a call is shorter than the kernel window. With $K = 4$ and a single-token decode step, the updated `conv_state` keeps two old rows from the previous state plus the one new row.

Two details in step 3 are easy to miss. First, L2 normalization puts its epsilon inside the square root, $u / \sqrt{\sum u^2 + 10^{-6}}$, which differs from $u / \max(\lVert u \rVert, \epsilon)$ when vectors are very small. Second, scaling queries by $d_k^{-1/2}$ may look redundant because $y = S q$ is linear in $q$ and step 6 applies RMSNorm. However, whenever $y$ is small compared to $\sqrt{\texttt{norm\_eps}} = 10^{-3}$ (which happens on the early tokens of a sequence while $S$ is nearly empty), $y / \sqrt{\operatorname{mean}(y^2) + \epsilon} \approx y / \sqrt{\epsilon}$ and the $d_k^{-1/2}$ factor passes straight through.

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

Calling `F.conv1d` with `groups=C` on the transposed `padded` tensor computes the same sum without flipping the kernel, since PyTorch's `conv1d` is a cross-correlation. The function takes `cfg` because `n_key_heads` cannot be inferred from the parameter shapes alone (`key_dim` can factor in multiple ways).

## Common mistakes

- **Resetting `conv_state` on each call.** Passes every single-call test, then corrupts the first $K - 1$ tokens of every continuation call.
- **Resetting `S` on each call.** The convolution continues cleanly across calls, but the delta-rule memory starts from zero on every continuation.
- **Saving the convolution output instead of its input.** Both tensors have $C$ channels, so the shapes match, but the next step needs the pre-convolution rows of $P$.
- **Slicing the new `conv_state` from `P` only.** Taking `P[-(K - 1):]` (with zero padding when short) works when $T \ge K - 1$, but drops saved history whenever $T < K - 1$, which includes every single-token decode step.
- **Gating before normalizing.** Computing $\operatorname{rms\_norm}(y \cdot \operatorname{silu}(z))$ follows the Mamba-2 order (`mamba2-ssd-step`), whereas Gated DeltaNet normalizes first and gates second.
- **Tiling key heads instead of repeating them.** `tile` or `repeat` produces head order `0, 1, 0, 1` instead of `0, 0, 1, 1`. This bug stays hidden when $H_k = 1$ or $H_k = H_v$.
- **Dropping the $d_k^{-1/2}$ query scale.** Appears harmless under RMSNorm, but changes outputs whenever $y$ is small relative to $\sqrt{\texttt{norm\_eps}}$.
- **Normalizing over the full value width.** `norm.weight` has length $d_v$ because RMSNorm runs independently per value head, not across all $H_v d_v$ channels.
- **Using `F.normalize` or hardcoding `1e-6` for `norm_eps`.** `F.normalize` clamps the denominator outside the square root, and hardcoding `1e-6` ignores `cfg.norm_eps`. Both fail on small-norm inputs or custom configs.
- **Mutating the input state in place.** Operations like `S.mul_(g)` produce the right outputs for one call while silently corrupting the caller's saved state, breaking beam search, retries, or shared-prefix branching.
- **Calling `ops.gated_delta_layer`.** This helper is the reference implementation and is forbidden by the harness.
- **Using another library's state layout.** Porting `transformers` directly stores `S` as `(H_v, d_k, d_v)` and `conv_state` as `(C, K)`, which is self-consistent but incompatible with this problem's state contract.

## Complexity

Let $D = \texttt{cfg.d\_model}$. The recurrence takes $O(T \cdot H_v d_v d_k)$ time, and the linear projections and depthwise convolution take $O(T \cdot C \cdot (D + K))$ time. The carried state uses $(K - 1) C + H_v d_v d_k$ floats regardless of sequence length. For Qwen3.5-0.8B, that is $3 \cdot 6144 + 16 \cdot 128 \cdot 128 = 280{,}576$ floats per layer.

## In the wild

Verified on 2026-11-21 against `transformers` 5.23.0, `mlx-lm` 0.32.2, and the `config.json` of `Qwen/Qwen3.5-0.8B` on the Hugging Face Hub.

- **Layer implementation.** `Qwen3_5GatedDeltaNet.forward` in `transformers/models/qwen3_5/modeling_qwen3_5.py` runs these six steps. Its `l2norm` computes `x * rsqrt((x * x).sum() + eps)`, `Qwen3_5RMSNormGated` normalizes before multiplying by `silu(gate)` ("Norm before gate"), and both delta-rule paths expand key heads with `repeat_interleave` and scale queries by `head_dim ** -0.5`. Public and hidden tests compare against this module both with and without cache (`fixtures/make.py` regenerates the arrays).
- **Shipped sizes.** `Qwen/Qwen3.5-0.8B` uses 16 key heads and 16 value heads of dimension 128, kernel size 4, and `rms_norm_eps = 1e-6` in 18 of its 24 layers. The config class defaults to 32 value heads for 16 key heads, matching the $r = 2$ test cases.
- **State layouts.** `transformers` keeps the last $K$ projected inputs as `(C, K)` (one extra slot beyond what the next token reads) and stores `S` as `(H_v, d_k, d_v)`. `mlx-lm` keeps $K - 1$ rows and `(H_v, d_v, d_k)`, which is the layout used here.
- **Single token vs. chunks.** With a cache and $T = 1$, `transformers` runs the recurrent per-token rule. All other calls use the chunked form (`gated-delta-chunked`), and the test fixture exercises both.

This problem makes four deliberate simplifications compared to the full model:
- No batch axis.
- No padding mask (the library zeroes padded positions before projection).
- Float32 throughout (the library runs the recurrence in float32 even when weights are in a narrower dtype).
- No convolution bias, matching the shipped weights.

The full block around this sublayer (pre-norm, residual connection, and MLP) is covered in `hybrid-decoder-forward`.
