Compute causal attention for new tokens over a KV cache with PyTorch's fused kernel,
`torch.nn.functional.scaled_dot_product_attention`, and tell it correctly what each
token may see. The kernel has two ways to take a mask, `is_causal=True` and
`attn_mask`, which cannot be combined; the first is wrong as soon as there is a cache
or padding, and the second uses the opposite convention from the padding mask you are
handed.

## Specification

```python
def attend(q, k, v, n_cached, key_padding): ...
```

`T` new tokens attend to `S` key/value slots:

- `q`: queries of the new tokens, shape `(B, N_q, T, D)`.
- `k`, `v`: keys and values, shape `(B, N_kv, S, D)` each, with
  $S = \texttt{n\_cached} + T$. Slots $0, \dots, \texttt{n\_cached} - 1$ hold the
  tokens cached earlier; slot $\texttt{n\_cached} + i$ holds new token $i$ (its key
  and value are already written).
- `n_cached`: a Python int, $\ge 0$.
- `key_padding`: `None`, or a boolean tensor of shape `(B, S)` on the device of `q`
  that is **`True` where a slot is padding**. A padding slot holds no token and must
  not influence any result. This is the convention of
  `nn.MultiheadAttention`'s `key_padding_mask`.
- Returns the attention output, shape `(B, N_q, T, D)`, in the dtype and on the
  device of `q`.

New token $i$ ($0 \le i < T$) may attend to slot $j$ when
$j \le \texttt{n\_cached} + i$ (the cache, the new tokens before it, and itself) and
slot $j$ is not padding in its row. With $A_i$ the set of such slots, for one query
head and its key/value head:

$$
\text{out}_i = \sum_{j \in A_i} a_{ij}\, v_j ,
\qquad
a_{ij} = \frac{\exp\!\left(q_i \cdot k_j / \sqrt{D}\right)}{\sum_{l \in A_i} \exp\!\left(q_i \cdot k_l / \sqrt{D}\right)} .
$$

$A_i$ is never empty in the tests.

**Heads.** $N_{kv}$ divides $N_q$. With $N_{kv} < N_q$ (grouped-query attention),
query head $h$ uses key/value head $\lfloor h / (N_q / N_{kv}) \rfloor$: consecutive
query heads share one key/value head.

**The kernel.** Its signature in PyTorch 2.15:

```python
F.scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
)
```

- `is_causal=True` applies a lower-triangular mask of shape `(T, S)` anchored at the
  top left: query $i$ sees slots $0, \dots, i$. Passing `attn_mask` together with
  `is_causal=True` is an error.
- `attn_mask` as a **boolean** tensor: `True` means *may attend*. It must broadcast
  to `(B, N_q, T, S)`. (A float tensor is added to the scores instead.)
- `enable_gqa=True` lets `key` and `value` have fewer heads than `query` and pairs
  them as described above.
- The default `scale=None` is $1/\sqrt{D}$.

**Rules.** The judge records how your function calls the kernel:

1. The result comes from **exactly one call** of `F.scaled_dot_product_attention`
   per call of `attend`, for the whole batch and all heads.
2. For a **square prefill without padding** (`n_cached == 0` and
   `key_padding is None`), the call uses `is_causal=True` and no `attn_mask`: no mask
   tensor is built.
3. In every other case the call passes a **boolean** `attn_mask` and does not set
   `is_causal`. This includes `key_padding` given with no slot padded. (One
   exception: for a single new token after a cache, without `key_padding`, every
   slot is visible, and a call with no mask at all is accepted as well.)
4. `k` and `v` are passed to the kernel as they are, with their $N_{kv}$ heads.
5. Every tensor you create is created on the device of `q`: a factory call
   (`torch.arange`, `torch.ones`, ...) without `device=` fails the case.

![Two new tokens after three cached ones. `is_causal=True` hides most of the cache from them, so this case takes an explicit mask.](assets/causal-alignment.svg)

## Shapes

| name | shape | dtype |
|---|---|---|
| `q` | `(B, N_q, T, D)` | float32 or float16 |
| `k`, `v` | `(B, N_kv, S, D)`, $S = \texttt{n\_cached} + T$ | the dtype of `q` |
| `key_padding` | `(B, S)` or `None` | bool |
| result | `(B, N_q, T, D)` | the dtype of `q` |

## Example

One sequence, one head, $D = 2$, one cached token and two new ones:

```python
q = [[[[1.0, 0.0], [0.0, 2.0]]]]  # (1, 1, 2, 2)
k = [[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]]  # (1, 1, 3, 2)
v = [[[[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]]]]

attend(q, k, v, n_cached=1, key_padding=None)
# [[[[0.6698, 0.3302],
#    [1.0000, 1.3374]]]]
```

- New token 0 attends to slots 0 and 1. Its scores are
  $[1, 0] / \sqrt{2} = [0.7071, 0]$, the weights $[0.6698, 0.3302]$, and the output
  $0.6698\,[1, 0] + 0.3302\,[0, 1]$.
- New token 1 attends to all three slots. Its scores are
  $[0, 2, 2] / \sqrt{2} = [0, 1.4142, 1.4142]$, the weights
  $[0.1084, 0.4458, 0.4458]$, and the output $[0.1084 + 0.8916,\; 0.4458 + 0.8916]$.

With `is_causal=True` instead of a mask, new token 0 would see slot 0 only and return
$[1, 0]$.

```python
attend(q, k, v, n_cached=1, key_padding=[[False, True, False]])
# [[[[1.0000, 0.0000],
#    [1.8044, 1.6088]]]]
```

Slot 1 is padding. New token 0 is left with slot 0 alone; new token 1 attends to
slots 0 and 2 with scores $[0, 1.4142]$ and weights $[0.1956, 0.8044]$.

## Constraints

- $1 \le B \le 4$, $1 \le T \le 24$, $0 \le \texttt{n\_cached} \le 20$,
  $D \in \{4, 8, 64\}$, $N_q \le 8$.
- float32 results are compared with a float64 computation to a relative tolerance of
  `1e-4` and an absolute tolerance of `1e-5`; float16 results to `1e-2` and `3e-2`.
- Every query has at least one slot it may attend to. What the kernel returns for a
  row with no visible slot is not specified here.
- The rules are exact: a second kernel call, a float mask, or a mask where
  `is_causal=True` was required fails the case even when the numbers are right.
- **What runs where.** The cases without a hardware mark run on the CPU, where the
  kernel is PyTorch's own C++ implementation; one of them runs `attend` on
  `torch.device("meta")`, whose tensors have no data and, like CUDA tensors, do not
  mix with CPU tensors. Two cases run only on a machine with an NVIDIA GPU and repeat
  the layouts with CUDA tensors in float32 and float16. The GPU cases were written
  from the PyTorch 2.15 documentation and could not be run when the problem was
  authored.

## Why this matters

On a GPU, `scaled_dot_product_attention` is the entry point to the fused attention
implementations (FlashAttention and memory-efficient attention), which never
materialize the `(T, S)` score matrix; every transformer written against PyTorch 2
calls it in each layer for each token. Its mask arguments fail silently. With
`is_causal=True` and a cache, a decode step attends to slot 0 only and the model
keeps producing fluent text that ignores its context. A padding mask passed with the
polarity of `key_padding_mask` makes the model read only the padding. Tiled instead
of grouped key/value heads pair every query head with the wrong keys. None of these
raises, and all of them return tensors of the right shape.
