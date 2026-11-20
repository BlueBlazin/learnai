Implement the token mixer of a Gated DeltaNet layer (the `linear_attn` sublayer of
Qwen3.5) so that it can continue a sequence: it carries the last few inputs of a short
convolution and the memory of the delta rule, and any split of a sequence into calls
must give the outputs of one call.

## Specification

```python
def gdn_layer(params, cfg, x, state): ...  # -> (out, (conv_state, S))
```

`cfg` is a [`GDNConfig`](../../../python/src/learnai/kit/gdn.py). Write $H_k$, $H_v$
for `cfg.n_key_heads`, `cfg.n_value_heads`, $d_k$, $d_v$ for `cfg.key_head_dim`,
`cfg.value_head_dim`, $K$ for `cfg.conv_kernel`, and $r = H_v / H_k$ (an integer).
`cfg.key_dim` $= H_k d_k$, `cfg.value_dim` $= H_v d_v$, and `cfg.conv_dim`
$C = 2 H_k d_k + H_v d_v$. `params` maps the checkpoint's tensor names to float32
weights; linear weights are `(out, in)` and applied as `x @ W.T`. For the $T$ rows of
`x`:

1. **Project.** $P = x\,W_{qkv}^\top$ with `in_proj_qkv.weight`, shape $(T, C)$.
2. **Convolve.** Let $\tilde P$ be the $K - 1$ rows of `conv_state` followed by the
   $T$ rows of $P$, and $w$ = `conv1d.weight`, shape $(C, 1, K)$. Each channel is
   mixed with its own past, the last tap on the current row:

$$
M_{t,c} = \operatorname{silu}\Bigl(\sum_{j=0}^{K-1} w_{c,0,j}\,\tilde P_{t+j,\,c}\Bigr),
\qquad 0 \le t < T .
$$

   The new `conv_state` is the last $K - 1$ rows of $\tilde P$.
3. **Split and normalize.** The first $H_k d_k$ columns of $M$ are the queries,
   reshaped to $(T, H_k, d_k)$; the next $H_k d_k$ the keys, likewise; the remaining
   $H_v d_v$ the values, $(T, H_v, d_v)$. Queries and keys are normalized over their
   last axis with $\operatorname{l2norm}(u) = u \,/ \sqrt{\textstyle\sum_i u_i^2 + 10^{-6}}$,
   and the queries are then multiplied by $d_k^{-1/2}$. Key head $j$ serves value heads
   $jr, \dots, jr + r - 1$: value head $h$ uses the query and key of head
   $\lfloor h / r \rfloor$.
4. **Gates.** With $a = x\,W_a^\top$ and $b = x\,W_b^\top$, both $(T, H_v)$:
   $g = \exp(-e^{\texttt{A\_log}} \cdot \operatorname{softplus}(a + \texttt{dt\_bias}))$
   and $\beta = \sigma(b)$.
5. **Recurrence.** For each value head, starting from its matrix in `S` and for
   $t = 0, \dots, T - 1$ in order (the rule of `gated-delta-step`):

$$
\bar S = g_t\,S, \qquad S = \bar S + \beta_t\,(v_t - \bar S k_t)\,k_t^\top, \qquad y_t = S\,q_t .
$$

6. **Norm, gate, project.** With $z = x\,W_z^\top$ reshaped to $(T, H_v, d_v)$ and
   $\gamma$ = `norm.weight`, per token and value head:

$$
o = \gamma \odot \frac{y}{\sqrt{\tfrac{1}{d_v}\sum_i y_i^2 + \texttt{cfg.norm\_eps}}} \odot \operatorname{silu}(z),
\qquad \text{out} = \operatorname{flatten}(o)\,W_{out}^\top .
$$

- `x`: float32, shape `(T, cfg.d_model)`, $T \ge 1$.
- `state`: `None` for a new sequence (both parts are zeros), or the pair
  `(conv_state, S)` returned by the call that read the tokens before `x`.
- Returns `(out, (conv_state, S))`: the outputs, the last $K - 1$ rows of $\tilde P$,
  and the memory after the last token.
- The state is the last argument and the second result. The tensors passed in, the
  state included, are not modified: a caller may continue twice from one state.

The kit's `ops.l2norm`, `ops.rms_norm`, `ops.softplus`, and `ops.causal_conv1d` are
allowed. Do not call `ops.gated_delta_layer`, `gdn_layer_ref`, or
`hybrid_forward_ref`.

> [!NOTE]
> The state layout is this problem's (and `mlx-lm`'s). `transformers` keeps $K$ rows
> of convolution state as $(C, K)$ and stores the memory transposed, $(H_v, d_k, d_v)$.

## Shapes

| name | shape | dtype |
|---|---|---|
| `x`, `out` | `(T, d_model)` | float32 |
| `in_proj_qkv.weight` | `(C, d_model)` | float32 |
| `conv1d.weight` | `(C, 1, K)` | float32 |
| `in_proj_a.weight`, `in_proj_b.weight` | `(Hv, d_model)` | float32 |
| `A_log`, `dt_bias` | `(Hv,)` | float32 |
| `in_proj_z.weight` | `(Hv * dv, d_model)` | float32 |
| `norm.weight` | `(dv,)` | float32 |
| `out_proj.weight` | `(d_model, Hv * dv)` | float32 |
| `conv_state` | `(K - 1, C)` | float32 |
| `S` | `(Hv, dv, dk)` | float32 |

## Example

A layer of width 4 with one key head and two value heads of width 2, kernel 2
(`conv_dim` 8), and the kit's seeded weights:

```python
cfg = GDNConfig(d_model=4, n_key_heads=1, n_value_heads=2,
                key_head_dim=2, value_head_dim=2, conv_kernel=2)
params = gdn_params(cfg, seed=3)
x = [[1.0, 0.0, -1.0, 2.0], [0.5, -0.5, 1.0, 0.0], [0.0, 2.0, 1.0, -1.0]]

out, (conv_state, S) = gdn_layer(params, cfg, x, None)
# out = [[-0.072025,  0.090774, 0.354554, 0.683884],
#        [-0.109328, -0.223080, 0.181717, 0.175837],
#        [-0.677092, -1.930008, 2.133805, 2.746412]]
# conv_state = [[1.535249, -0.848628, -0.367922, 0.590316,
#                -0.777152, 0.199583, -0.881968, 1.245803]]    (the last row of P)
```

Feeding the first two rows and then the third with the returned state gives the same
third row, `[-0.677092, -1.930008, 2.133805, 2.746412]`. Feeding the third row with
`state=None` gives `[0.581054, 1.797964, -1.971115, -2.558913]`: without the saved
row the convolution sees zeros, and without the memory the read finds only this
token's own write.

## Constraints

- $1 \le T \le 12$; $H_k$ divides $H_v$, $1 \le H_k \le H_v \le 4$; $2 \le d_k, d_v \le 8$;
  $2 \le K \le 6$; `cfg.d_model` $\le 32$. A call may have fewer than $K - 1$ tokens.
- Outputs and both parts of the state are compared with the reference, with the
  kit's `gdn_layer_ref`, and with outputs of `Qwen3_5GatedDeltaNet` in Hugging Face
  Transformers holding the same weights: relative tolerance `1e-4`, absolute `1e-5`.
- Any split of a sequence into consecutive calls, each given the state the previous
  one returned, must give the outputs and the final state of a single call, to the
  same tolerance.
- `cfg.norm_eps` is `1e-6` except where a test sets another value. Some tests scale
  `in_proj_qkv.weight` by $10^{-3}$; there $\sum_i u_i^2$ is near $10^{-6}$ and the
  placement of both epsilons decides the result.
- A Python loop over tokens is expected in step 5. No loop over heads is needed.

## Why this matters

An attention layer can rebuild its cache from the tokens; this layer cannot rebuild
either part of its state from the other, and serving it means saving, restoring, and
batching both. The errors are quiet: a convolution state that restarts from zeros
changes only the first $K - 1$ tokens after each call boundary, so a model that is
prefilled in one call and then decoded one token at a time is wrong at every step
while every single-call test passes.
