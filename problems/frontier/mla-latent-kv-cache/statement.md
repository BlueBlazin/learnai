Implement the cached attention step of Multi-Head Latent Attention (MLA), the attention mechanism used in DeepSeek-V2 and its successors. Keys and values are low-rank up-projections of a single compact latent vector per token, so the cache stores only that latent plus a short RoPE key shared across all heads. To attend **without rebuilding per-head keys and values**, you fold the key and value up-projections into the query and output weights ahead of time. Your `mla_step` function never receives those up-projection matrices.

## Specification

```python
def absorb(w_uq, w_uk, w_uv, w_o, n_heads): ...  # -> (w_qk, w_ov)
def mla_step(x_new, weights, cache): ...  # -> (out, (c, k_rope))
```

All weight matrices are stored with shape `(out_features, in_features)` and applied to row vectors as `x @ w.T`. The model dimensions are:

- $d$ (`d_model`): model hidden width
- $H$ (`n_heads`): number of attention heads
- $d_h$ (`d_head`): content head dimension
- $d_r$ (`d_rope`): RoPE dimension (even)
- $d_c$: key-value latent width
- $d_q$ (`d_cq`): query latent width

Batch size is $B$, the number of previously cached tokens is $S$, and the number of new tokens in the current call is $T \ge 1$.

**Explicit MLA definition.** For a token at position $t$ with input vector $x_t \in \mathbb{R}^d$, MLA projects into low-rank latents, up-projects content vectors, and computes separate rotary position components:

$$
\begin{aligned}
c^Q_t &= W^{DQ} x_t \in \mathbb{R}^{d_q}, &
q^C_t &= W^{UQ} c^Q_t \in \mathbb{R}^{H d_h}, &
q^R_t &= \operatorname{RoPE}_t\!\big(W^{QR} c^Q_t\big) \in \mathbb{R}^{H d_r}, \\
c_t &= W^{DKV} x_t \in \mathbb{R}^{d_c}, &
k^C_t &= W^{UK} c_t \in \mathbb{R}^{H d_h}, &
k^R_t &= \operatorname{RoPE}_t\!\big(W^{KR} x_t\big) \in \mathbb{R}^{d_r}, \\
& & v_t &= W^{UV} c_t \in \mathbb{R}^{H d_h}.
\end{aligned}
$$

Any vector of length $H d_h$ (or $H d_r$) is the concatenation of $H$ head vectors, where head $h$ occupies indices $h\,d_h, \dots, (h+1)\,d_h - 1$. The rotary key $k^R_t$ has no head index and is shared by every head. Head $h$ of token $t$ attends causally to all positions $j \le t$:

$$
s_{t,h,j} = \frac{q^C_{t,h} \cdot k^C_{j,h} \;+\; q^R_{t,h} \cdot k^R_j}{\sqrt{d_h + d_r}},
\qquad
a_{t,h,\cdot} = \operatorname{softmax}_{j \le t}\, s_{t,h,j},
\qquad
o_{t,h} = \sum_{j \le t} a_{t,h,j}\, v_{j,h}.
$$

The final layer output is $u_t = W^O\,[o_{t,0}; \dots; o_{t,H-1}] \in \mathbb{R}^d$.

Here $\operatorname{RoPE}_t$ is the same rotation as in the `rope` problem for position $t$ (adjacent pairs, $\theta_i = 10000^{-2i/d_r}$, counterclockwise). Apply it **to each head's $d_r$ entries separately** for $q^R$, and to the $d_r$ entries of $k^R$. This is provided as `ops.rope(x, positions)` in the kit, which rotates the last axis of `(..., T, D)`. The content vectors $q^C$ and $k^C$ are not rotated.

**Weight absorption (`absorb`).** Given `w_uq`, `w_uk`, `w_uv`, `w_o`, and `n_heads`, return `(w_qk, w_ov)`, containing one absorbed matrix per head. Let $W^{UQ}_h$, $W^{UK}_h$, and $W^{UV}_h$ denote head $h$'s slice of rows from each up-projection, and let $W^O_h$ denote head $h$'s slice of columns from $W^O$:

$$
W^{QK}_h = (W^{UK}_h)^\top W^{UQ}_h \in \mathbb{R}^{d_c \times d_q},
\qquad
W^{OV}_h = W^{O}_h\, W^{UV}_h \in \mathbb{R}^{d \times d_c}.
$$

These absorbed matrices satisfy $q^C_{t,h} \cdot k^C_{j,h} = (W^{QK}_h c^Q_t) \cdot c_j$ and $u_t = \sum_h W^{OV}_h \big(\sum_j a_{t,h,j}\, c_j\big)$, allowing attention to operate directly on cached latents.

**Cached attention step (`mla_step`).** Processes $T$ new tokens following $S$ cached tokens.

- `x_new`: new token representations. New token $i$ sits at absolute position $S + i$ and attends to all $S$ cached tokens, all earlier new tokens, and itself.
- `weights`: a dict containing `"w_dq"`, `"w_qr"`, `"w_dkv"`, `"w_kr"`, the absorbed `"w_qk"` and `"w_ov"` from `absorb`, and the integers `"n_heads"` and `"d_head"`. It does **not** contain $W^{UQ}$, $W^{UK}$, $W^{UV}$, or $W^O$.
- `cache`: either `None` (when $S = 0$) or a tuple `(c, k_rope)` storing the cached latents $c_j$ and the **already rotated** shared keys $k^R_j$, oldest token first.
- Returns `(out, (c, k_rope))`, where `out` contains $u_t$ for the $T$ new tokens and the returned cache covers all $S + T$ tokens. Do not modify the input `cache`.

![The cache holds only $c$ and $k^R$. The key and value up-projections are folded into $W^{QK}$ and $W^{OV}$ instead of being applied to the cache.](assets/mla-step.svg)

With this layout, the cache stores $d_c + d_r$ floats per token per layer and sequence, regardless of the number of heads.

## Shapes

| name | shape | dtype |
|---|---|---|
| `w_uq` | `(H * d_head, d_cq)` | float32 |
| `w_uk`, `w_uv` | `(H * d_head, d_c)` | float32 |
| `w_o` | `(d_model, H * d_head)` | float32 |
| `w_qk` (returned by `absorb`) | `(H, d_c, d_cq)` | float32 |
| `w_ov` (returned by `absorb`) | `(H, d_model, d_c)` | float32 |
| `weights["w_dq"]` | `(d_cq, d_model)` | float32 |
| `weights["w_qr"]` | `(H * d_rope, d_cq)` | float32 |
| `weights["w_dkv"]` | `(d_c, d_model)` | float32 |
| `weights["w_kr"]` | `(d_rope, d_model)` | float32 |
| `x_new`, `out` | `(B, T, d_model)` | float32 |
| cache `c` | `(B, S, d_c)` in, `(B, S + T, d_c)` out | float32 |
| cache `k_rope` | `(B, S, d_rope)` in, `(B, S + T, d_rope)` out | float32 |

## Example

Take $d = 2$, $H = 2$, $d_h = 1$, $d_r = 2$, and $d_c = d_q = 2$, with $W^{DQ} = W^{DKV} = W^{UQ} = W^{UV} = I$ and:

```python
w_uk = [[1, 1], [0, 2]]  # head 0: row [1, 1]; head 1: row [0, 2]
w_o = [[1, 1], [0, 1]]  # head 0: column [1, 0]; head 1: column [1, 1]
w_qr = [[1, 0], [0, 0], [0, 1], [0, 0]]  # head 0: rows 0-1; head 1: rows 2-3
w_kr = [[1, 1], [0, 0]]

w_qk, w_ov = absorb(w_uq, w_uk, w_uv, w_o, 2)
# w_qk = [[[1, 0], [1, 0]], [[0, 0], [0, 2]]]
# w_ov = [[[1, 0], [0, 0]], [[0, 1], [0, 1]]]
```

Because $d_h = 1$, the absorbed matrices are outer products: $W^{QK}_0 = [1, 1]^\top [1, 0]$, $W^{QK}_1 = [0, 2]^\top [0, 1]$, and $W^{OV}_1 = [1, 1]^\top [0, 1]$.

Now feed two tokens, $x_0 = [1, 0]$ and $x_1 = [0, 1]$, one step at a time:

```python
out, cache = mla_step([[[1.0, 0.0]]], weights, None)
# out = [[[1.0, 0.0]]]      cache = ([[[1, 0]]], [[[1, 0]]])
out, cache = mla_step([[[0.0, 1.0]]], weights, cache)
# out = [[[1.305385, 0.805385]]]
# cache = ([[[1, 0], [0, 1]]], [[[1, 0], [0.540302, 0.841471]]])
```

- **Token 0** can only attend to itself, so each head returns its own value. That gives $o = [1, 0]$ and $u_0 = W^O o = [1, 0]$. Its shared rotary key is $W^{KR} x_0 = [1, 0]$, which is unrotated at position 0.
- **Token 1** has $c^Q = c = [0, 1]$. Its shared key $W^{KR} x_1 = [1, 0]$ is rotated by 1 radian to $[\cos 1, \sin 1] = [0.5403, 0.8415]$, and that rotated vector is stored in the cache.
  - Head 0 has a zero query (content query $W^{QK}_0 c^Q = [0, 0]$ and RoPE query $[0, 0]$), so its attention weights are $[0.5, 0.5]$, producing the mixed latent $[0.5, 0.5]$.
  - Head 1 has content query $W^{QK}_1 c^Q = [0, 2]$ and RoPE query $[\cos 1, \sin 1]$. Its raw score against token 0 is $[0, 2] \cdot [1, 0] + [\cos 1, \sin 1] \cdot [1, 0] = 0.5403$, and against itself is $[0, 2] \cdot [0, 1] + 1 = 3$. Dividing by $\sqrt{d_h + d_r} = \sqrt{3}$ gives $[0.3119, 1.7321]$, softmax yields $[0.1946, 0.8054]$, and the mixed latent is $[0.1946, 0.8054]$.
  - Combining heads via $W^{OV}$ gives $u_1 = W^{OV}_0 [0.5, 0.5] + W^{OV}_1 [0.1946, 0.8054] = [0.5, 0] + [0.8054, 0.8054]$.

The cache stores $d_c + d_r = 4$ floats per token, and no key vector of width $H d_h$ is ever materialized.

## Constraints

- $1 \le B \le 2$, $1 \le T$, $S + T \le 32$, and $1 \le H \le 4$.
- $d_r$ is even with $2 \le d_r \le 4$. All other dimensions are between 1 and 20.
- Outputs and cache tensors are compared against the explicit form evaluated on the full sequence at once, with relative tolerance `1e-4` and absolute tolerance `1e-5` (`1e-3` and `1e-4` for the 30-token test case). Any split of a sequence into calls must match.
- After every call, `cache` must be a tuple of two tensors matching the shapes above and holding at most $d_c + d_r$ floats per token and sequence.
- `absorb` is checked against the reference values and the two algebraic identities above.
- You may use `ops.rope` and `ops.softmax` from `learnai.kit`. No Python loops over tokens, heads, or batch rows are needed.

## Why this matters

KV cache memory, not compute, is what caps how many long sequences fit on a GPU. MLA solves this at the architecture level. At DeepSeek-V2's dimensions ($H = 128$, $d_h = 128$, $d_c = 512$, $d_r = 64$), each token requires only 576 cached floats per layer instead of 32,768 for standard per-head keys and values.

An inference engine only captures this 57x reduction if it caches the latent directly. Porting MLA into a standard attention path by up-projecting and caching per-head keys and values produces the right numbers while silently wasting 57 times the memory. Once you cache the latent, you can either up-project every cached latent on every decode step or absorb the up-projections into the query and output weights once. Absorption works only because the position-dependent RoPE rotation is kept outside the compressed path, which is the reason MLA splits queries and keys into content and RoPE components.
