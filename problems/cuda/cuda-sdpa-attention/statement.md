Implement `attend` to compute causal attention for new tokens over a KV cache using PyTorch's fused kernel, `torch.nn.functional.scaled_dot_product_attention`. You must choose between the kernel's `is_causal=True` fast path and an explicit boolean `attn_mask`, while handling cached prefixes, padding, and grouped-query attention.

## Specification

```python
def attend(q, k, v, n_cached, key_padding): ...
```

A batch contains `B` sequences with head dimension `D`. Each sequence has `n_cached` previously cached tokens and `T` new query tokens, for a total of $S = \texttt{n\_cached} + T$ key/value slots.

- `q`: queries for the `T` new tokens across `N_q` query heads.
- `k`, `v`: keys and values across `N_kv` key/value heads. Slots $0, \dots, \texttt{n\_cached} - 1$ hold the cached tokens, and slot $\texttt{n\_cached} + i$ holds new token $i$ (already appended to `k` and `v`).
- `n_cached`: a non-negative Python `int` giving the number of cached tokens.
- `key_padding`: either `None`, or a boolean tensor on the same device as `q` where **`True` marks a padding slot**. Padding slots hold no token and must not affect the output. This matches `nn.MultiheadAttention`'s `key_padding_mask` convention.
- Returns the attention output on the same device and in the same dtype as `q`.

New token $i$ ($0 \le i < T$) may attend to slot $j$ ($0 \le j < S$) when $j \le \texttt{n\_cached} + i$ (covering the cache, earlier new tokens, and itself) and slot $j$ is not padding in its sequence. Let $A_i$ be the set of allowed slots for new token $i$. For each query head and its paired key/value head, the output $\text{out}_i$ is the softmax-weighted sum of the allowed value vectors:

$$
\text{out}_i = \sum_{j \in A_i} a_{ij}\, v_j ,
\qquad
a_{ij} = \frac{\exp\!\left(q_i \cdot k_j / \sqrt{D}\right)}{\sum_{l \in A_i} \exp\!\left(q_i \cdot k_l / \sqrt{D}\right)} .
$$

$A_i$ is never empty in the tests.

**Heads.** $N_{kv}$ divides $N_q$. When $N_{kv} < N_q$ (grouped-query attention), consecutive query heads share one key/value head, so query head $h$ uses key/value head $\lfloor h / (N_q / N_{kv}) \rfloor$.

**The kernel.** In PyTorch 2.15, the fused kernel has the following signature:

```python
F.scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
)
```

- `is_causal=True` applies a lower-triangular mask of shape `(T, S)` anchored at the top left, so query $i$ sees slots $0, \dots, i$. Passing `attn_mask` together with `is_causal=True` raises an error.
- When `attn_mask` is a **boolean** tensor, `True` means *may attend*. It must broadcast to `(B, N_q, T, S)`. Passing a float tensor adds its entries to the attention scores instead.
- `enable_gqa=True` allows `key` and `value` to have fewer heads than `query` and pairs them as defined above.
- Leaving `scale=None` uses the default scale $1/\sqrt{D}$.

**Rules.** The judge records how your function calls the kernel and enforces five rules:

1. Compute the result with **exactly one call** to `F.scaled_dot_product_attention` per call to `attend`, covering the whole batch and all heads.
2. For a **square prefill without padding** (`n_cached == 0` and `key_padding is None`), pass `is_causal=True` and do not pass `attn_mask` (no mask tensor is built).
3. In every other case, pass a **boolean** `attn_mask` and do not set `is_causal=True`. This includes when `key_padding` is provided with no slot padded. As one exception, when a single new token attends to a cache without `key_padding` (`T == 1`, `n_cached > 0`, `key_padding is None`), every slot is visible, so calling the kernel with no mask at all is also accepted.
4. Pass `k` and `v` as they are, with their original $N_{kv}$ heads.
5. Create every new tensor on the device of `q`. Calling a factory function (`torch.arange`, `torch.ones`, and similar) without `device=` fails the test.

![Two new tokens after three cached ones. `is_causal=True` hides most of the cache from them, so this case takes an explicit mask.](assets/causal-alignment.svg)

## Shapes

| name | shape | dtype |
|---|---|---|
| `q` | `(B, N_q, T, D)` | float32 or float16 |
| `k`, `v` | `(B, N_kv, S, D)`, $S = \texttt{n\_cached} + T$ | same as `q` |
| `key_padding` | `(B, S)` or `None` | bool |
| result | `(B, N_q, T, D)` | same as `q` |

## Example

Consider one sequence, one head, $D = 2$, one cached token, and two new tokens:

```python
q = [[[[1.0, 0.0], [0.0, 2.0]]]]  # (1, 1, 2, 2)
k = [[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]]  # (1, 1, 3, 2)
v = [[[[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]]]]

attend(q, k, v, n_cached=1, key_padding=None)
# [[[[0.6698, 0.3302],
#    [1.0000, 1.3374]]]]
```

- New token 0 attends to slots 0 and 1. Its scaled scores are $[1, 0] / \sqrt{2} = [0.7071, 0]$, giving weights $[0.6698, 0.3302]$ and output $0.6698\,[1, 0] + 0.3302\,[0, 1] = [0.6698, 0.3302]$.
- New token 1 attends to all three slots. Its scaled scores are $[0, 2, 2] / \sqrt{2} = [0, 1.4142, 1.4142]$, giving weights $[0.1084, 0.4458, 0.4458]$ and output $[0.1084 + 0.8916,\; 0.4458 + 0.8916] = [1.0000, 1.3374]$.

If `is_causal=True` were used instead of an explicit mask, new token 0 would see only slot 0 and return $[1, 0]$.

Now mark slot 1 as padding:

```python
attend(q, k, v, n_cached=1, key_padding=[[False, True, False]])
# [[[[1.0000, 0.0000],
#    [1.8044, 1.6088]]]]
```

New token 0 is left with slot 0 alone, returning $[1.0000, 0.0000]$. New token 1 attends to slots 0 and 2 with scaled scores $[0, 1.4142]$ and weights $[0.1956, 0.8044]$, giving $[1.8044, 1.6088]$.

## Constraints

- $1 \le B \le 4$, $1 \le T \le 24$, $0 \le \texttt{n\_cached} \le 20$, $D \in \{4, 8, 64\}$, and $N_q \le 8$.
- `float32` results are compared against a `float64` reference to a relative tolerance of `1e-4` and an absolute tolerance of `1e-5`. `float16` results use relative tolerance `1e-2` and absolute tolerance `3e-2`.
- Every query has at least one visible slot. Behavior for a row with no visible slot is not specified here.
- All kernel-call rules are exact. A second kernel call, a float mask, or an explicit mask where `is_causal=True` was required fails the test even when the output values match.
- **What runs where.** Non-GPU test cases run on the CPU using PyTorch's C++ implementation. One test runs `attend` on `torch.device("meta")`, whose tensors hold no data and cannot be mixed with CPU tensors, just like CUDA tensors. Two GPU test cases run only on machines with an NVIDIA GPU and repeat the layouts with CUDA tensors in `float32` and `float16`. The GPU cases were written from the PyTorch 2.15 documentation and were not run when the problem was authored.

## Why this matters

On a GPU, `scaled_dot_product_attention` dispatches to fused attention kernels (FlashAttention and memory-efficient attention) that never materialize the `(T, S)` score matrix. Every transformer written in PyTorch 2 calls it in every layer for every step.

Its mask and head arguments fail silently without raising an error, returning tensors of the expected shape. Passing `is_causal=True` with a cache restricts a decode step to slot 0, so the model keeps producing fluent text that ignores its context. Passing `key_padding` without inverting it makes the model attend only to padding tokens. Tiling key/value heads instead of grouping them pairs query heads with the wrong keys.
