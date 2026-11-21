Compute attention for new tokens over a key/value buffer with MLX's fused kernel,
`mx.fast.scaled_dot_product_attention`, and give it the right mask for the layout of
the buffer. The kernel takes its mask in three forms (nothing, the string `"causal"`,
or an array), and each of them is silently wrong for some layout.

## Specification

```python
def attend(q, k, v, n_cached, causal): ...
```

`T` new tokens attend to a buffer of `S` key/value slots:

- `q`: queries of the new tokens, shape `(B, N_q, T, D)`.
- `k`, `v`: the key and value buffers, shape `(B, N_kv, S, D)` each, with
  $S \ge$ `n_cached` $+\,T$. The slots are used from the front:
  - slots $0, \dots, \texttt{n\_cached} - 1$ hold the tokens that were cached earlier;
  - slots $\texttt{n\_cached}, \dots, \texttt{n\_cached} + T - 1$ hold the new tokens
    themselves (their keys and values are already written);
  - any slots after that are **unused capacity**. Their contents are arbitrary
    (finite) numbers and must not influence the result.
- `n_cached`: a Python int, $\ge 0$.
- `causal`: a Python bool. If `True`, new token $i$ ($0 \le i < T$) attends to slots
  $j \le \texttt{n\_cached} + i$: the cache, the new tokens before it, and itself. If
  `False`, every new token attends to all $\texttt{n\_cached} + T$ valid slots.
- Returns the attention output, shape `(B, N_q, T, D)`, in the dtype of `q`.

For one query head and its key/value head, with $A_i$ the set of slots token $i$
attends to:

$$
\text{out}_i = \sum_{j \in A_i} a_{ij}\, v_j ,
\qquad
a_{ij} = \frac{\exp\!\left(q_i \cdot k_j / \sqrt{D}\right)}{\sum_{l \in A_i} \exp\!\left(q_i \cdot k_l / \sqrt{D}\right)} .
$$

**Heads.** $N_{kv}$ divides $N_q$. With $N_{kv} < N_q$ (grouped-query attention),
query head $h$ uses key/value head $\lfloor h / (N_q / N_{kv}) \rfloor$: consecutive
query heads share one key/value head.

**The kernel.** The result must be computed by **exactly one call** of
`mx.fast.scaled_dot_product_attention` per call of `attend`, for the whole batch and
all heads; the judge counts the calls. Its signature in MLX 0.32.4:

```python
mx.fast.scaled_dot_product_attention(q, k, v, *, scale, mask=None)
```

- `scale` is required and is not applied for you: pass $1/\sqrt{D}$.
- `k` and `v` may have fewer heads than `q`; the kernel pairs them as described above.
- `mask=None`: every query attends to every slot.
- `mask="causal"`: according to the MLX documentation, it "uses lower-right alignment
  where the last query aligns with the last key". The last query sees all `S` slots,
  the one before it all but the last, and so on.
- `mask=` a boolean array: `True` means *may attend*. It must broadcast to
  `(B, N_q, T, S)`.
- `mask=` a float array: added to the scores before the softmax (`0` to attend,
  `-inf` to hide). Its dtype must not be wider than the dtype of `q`, `k`, and `v`.

![Two new tokens after three cached ones: in a full buffer the string mask is exactly right. With unused capacity the same string is aligned with the end of the buffer and lets the new tokens read slots that hold no token, so an explicit mask is needed.](assets/mask-alignment.svg)

## Shapes

| name | shape | dtype |
|---|---|---|
| `q` | `(B, N_q, T, D)` | float32 or float16 |
| `k`, `v` | `(B, N_kv, S, D)`, $S \ge$ `n_cached` $+\,T$ | the dtype of `q` |
| result | `(B, N_q, T, D)` | the dtype of `q` |

## Example

One sequence, one head, $D = 2$, one cached token and two new ones in a buffer of
exactly three slots:

```python
q = [[[[1.0, 0.0], [0.0, 2.0]]]]  # (1, 1, 2, 2)
k = [[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]]  # (1, 1, 3, 2)
v = [[[[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]]]]

attend(q, k, v, n_cached=1, causal=True)
# [[[[0.6698, 0.3302],
#    [1.0000, 1.3374]]]]
```

- New token 0 attends to slots 0 and 1. Its scores are
  $[1, 0] / \sqrt{2} = [0.7071, 0]$, the weights $[0.6698, 0.3302]$, and the output
  $0.6698\,[1, 0] + 0.3302\,[0, 1]$.
- New token 1 attends to all three slots. Its scores are
  $[0, 2, 2] / \sqrt{2} = [0, 1.4142, 1.4142]$, the weights
  $[0.1084, 0.4458, 0.4458]$, and the output $[0.1084 + 0.8916,\; 0.4458 + 0.8916]$.

Now put the same three tokens into a buffer of five slots, with anything in the last
two:

```python
k = [[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [9.0, 9.0], [-7.0, 3.0]]]]
v = [[[[1.0, 0.0], [0.0, 1.0], [2.0, 2.0], [50.0, -50.0], [80.0, 80.0]]]]

attend(q, k, v, n_cached=1, causal=True)  # the same result
```

Here `mask="causal"` would let token 0 read slots 0 to 3 and token 1 slots 0 to 4.

## Constraints

- $1 \le B \le 4$, $1 \le N_{kv} \le N_q \le 8$, $1 \le T \le 24$, $T \le S \le 48$,
  $D \in \{2, 4, 8, 64\}$.
- float32 results are compared with a float64 computation to a relative tolerance of
  `1e-4` and an absolute tolerance of `1e-5`; float16 results to `1e-2` and `2e-2`.
  The result must have the dtype of `q`.
- Exactly one call of `mx.fast.scaled_dot_product_attention` is made per call of
  `attend`. A solution that computes the attention with `matmul` and `softmax` fails
  even when its numbers are right.
- No Python loops over batch items, heads, or positions are needed.

## Why this matters

Attention is where a decoder spends its memory bandwidth, and the hand-written
version materializes a `(T, S)` score matrix for every head of every layer: for a
2,048-token prefill with 8 heads that is 128 MiB of float32 scores per layer, on this
machine a peak of 136 MiB against 4 MiB for the fused kernel, which never stores
them. The models in `mlx-lm` call this kernel for that reason, and the mask is the
one argument that depends on how the cache is laid out. A growing cache is always
exactly full, so `"causal"` (or nothing, for one token) is right. A preallocated
buffer, the layout that lets a decode step be compiled once, is not full, and the
same string then attends to slots that hold zeros or the remains of an earlier
request, with no error and plausible-looking output.
