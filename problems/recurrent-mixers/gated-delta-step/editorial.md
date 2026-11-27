## The idea

A linear-attention layer replaces the growing KV cache with a single matrix $S$ per head. The simplest version is an additive memory, $S \leftarrow S + v\,k^\top$, read with $y = S q = \sum_i v_i (k_i \cdot q)$. That memory never forgets and never overwrites, so repeated writes under the same key accumulate into a sum of values.

The delta rule replaces the additive write with an error-correcting update. It queries the memory for its current prediction $S k$ under the new key and stores only the residual:

$$
S' = S + \beta\,(v - S k)\,k^\top = S\,(I - \beta\,k k^\top) + \beta\,v\,k^\top .
$$

The factored form shows the geometry of the update. Along the direction of $k$, the existing memory is shrunk by $1 - \beta \lVert k \rVert^2$ and replaced by $\beta v$. Every direction $k'$ orthogonal to $k$ is left untouched because $(I - \beta k k^\top) k' = k'$ when $k \cdot k' = 0$. When $\lVert k \rVert = 1$ and $\beta = 1$, the old value under $k$ is completely erased and $S' k = v$. This update is one step of gradient descent on $\tfrac{1}{2} \lVert S k - v \rVert^2$ with step size $\beta$, which gives the delta rule its name.

Adding a scalar decay gate $g \in (0, 1)$ lets the head forget older associations before writing. The complete recurrence is Eq. 10 of the Gated Delta Networks paper:

$$
S_t = S_{t-1}\,\bigl(g_t\,(I - \beta_t\,k_t k_t^\top)\bigr) + \beta_t\,v_t\,k_t^\top,
\qquad y_t = S_t\,q_t .
$$

Expanding this expression yields the three steps in the specification: decay $S$, compute the correction from the *decayed* memory $\bar S$, and read from the *updated* memory $S'$.

The decay gate $g$ uses the parameterization from Mamba-2. Here $e^{A_{\log}}$ is a positive per-head decay rate and $\operatorname{softplus}(a + \texttt{dt\_bias})$ is a positive per-token step size. Their negated product is always negative, so $g$ never exceeds 1. Heads with a large rate forget within a few tokens, while heads with a small rate retain information across thousands of tokens.

## Reference solution (PyTorch)

```python
def gdn_gates(a, b, A_log, dt_bias):
    g = torch.exp(-torch.exp(A_log) * F.softplus(a + dt_bias))
    return g, torch.sigmoid(b)


def gdn_step(S, q, k, v, g, beta):
    S = S * g[:, None, None]  # decay
    predicted = torch.einsum("hvk,hk->hv", S, k)  # what the memory holds under k
    delta = beta[:, None] * (v - predicted)
    S = S + delta[:, :, None] * k[:, None, :]  # write the correction along k
    y = torch.einsum("hvk,hk->hv", S, q)  # read after the write
    return y, S
```

This problem uses the `(H, Dv, Dk)` state layout from `mlx-lm`. Hugging Face `transformers` stores the transposed layout `(H, Dk, Dv)` and writes `k.unsqueeze(-1) * delta.unsqueeze(-2)`, which computes identical values.

## Common mistakes

- **Reading before writing.** Computing $y = \bar S q$ prevents a token from seeing its own value and returns zeros on the first token of an empty memory.
- **Using an additive write.** Adding $\beta\,v\,k^\top$ without subtracting $\bar S k$ matches the delta rule on an empty memory and diverges on every later token, so single-step tests from zero state miss it.
- **Decaying after writing.** Multiplying by $g$ at the end scales the new value before it is read, giving $S' k = g v$ instead of $v$. This bug is invisible when $g = 1$.
- **Computing the correction from the undecayed memory.** Adding $\beta (v - S k) k^\top$ to $g S$ makes the memory return $v - (1 - g) S k$ under $k$. This is also invisible when $g = 1$ or when $S = 0$.
- **Applying $\beta$ only to $v$.** Writing $(\beta v - \bar S k) k^\top$ works when $\beta = 1$, but when $\beta = 0$ it erases the stored value under $k$ instead of leaving $S$ unchanged.
- **Omitting softplus in `gdn_gates`.** Computing $\exp(-e^{A_{\log}}(a + \texttt{dt\_bias}))$ produces $g > 1$ whenever $a + \texttt{dt\_bias} < 0$, causing the memory to grow exponentially.
- **Adding `dt_bias` outside softplus.** `softplus(a) + dt_bias` can also become negative, producing $g > 1$.
- **Normalizing $k$ inside `gdn_step`.** The surrounding layer normalizes keys once before the recurrence. Normalizing again inside the step changes the result whenever a caller intentionally passes non-unit keys.
- **Updating `S` in place.** Operations like `S *= g` mutate the caller's tensor. The immediate outputs look fine, but branching twice from the same saved state fails.
- **Calling the kit's oracle.** The kit helpers are the reference implementation and are forbidden by the harness.

## Complexity

One step takes $O(H\,D_v\,D_k)$ time and stores $O(H\,D_v\,D_k)$ state, independent of sequence length. In Qwen3.5-0.8B, a gated delta layer has 16 heads of $128 \times 128$, which is 262,144 floats of state. By comparison, one of the model's attention layers caches 2 key-value heads of width 256 (1,024 floats per token for keys and values combined), so the delta memory equals the size of a 256-token KV cache and never grows further.

## In the wild

Verified on 2026-11-21 against `transformers` 5.23.0 and `mlx-lm` 0.32.2.

- **Recurrence.** `torch_recurrent_gated_delta_rule` in `transformers/models/qwen3_5/modeling_qwen3_5.py` runs these exact steps per token: multiply the state by `decay.exp()`, compute `kv_mem`, form `delta = (v_t - kv_mem) * beta_t`, add the outer product, and read with `q`. The hidden six-token test compares against its output (`fixtures/make.py` regenerates the arrays). The Metal kernel in `mlx_lm/models/gated_delta.py` performs the same updates.
- **Gates.** `Qwen3_5GatedDeltaNet.forward` computes `g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)` and `beta = b.sigmoid()`, where `a` and `b` are linear projections of the layer input. It keeps `g` in log space until inside the recurrence, whereas this problem returns $g$ directly.
- **Layer preprocessing.** L2-normalizing `q` and `k`, scaling `q` by $D_k^{-1/2}$, and repeating key heads across value heads happen inside the same function in `transformers`, but are handled by the surrounding layer in `gated-delta-layer`.
- **Single token vs. chunks.** `transformers` calls the recurrent per-token function only for single-token cached decoding. Multi-token calls use a chunked formulation (`gated-delta-chunked`) that produces the same output.
- **Paper.** "Gated Delta Networks: Improving Mamba2 with Delta Rule" (arXiv 2412.06464, ICLR 2025), Eq. 10, defines this recurrence using $\alpha_t$ in place of $g_t$.
