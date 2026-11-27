Implement a preallocated key-value (KV) cache whose buffers never change shape. You allocate two fixed-size buffers once at full capacity and track how many slots hold tokens with an integer length. Writing fills the next free slots, and attention must behave as if the unused slots did not exist, regardless of what values they contain.

![This problem is the KV cache, here two buffers allocated once at full capacity.](assets/map.svg)

## Specification

```python
def cache_init(capacity, batch, heads, dim): ...  # returns cache
def cache_write(cache, k_new, v_new): ...  # returns cache
def cache_attend(q, cache): ...  # returns out
```

A **cache** is a tuple `(k, v, length)`:

- `k` and `v` are float32 buffers of shape `(B, H, C, D)`, where `B` is batch size, `H` is the number of attention heads, `C` is capacity, and `D` is the head dimension. Slot `j` along axis 2 holds the key or value for the token at position `j`.
- `length` is a Python `int` satisfying $0 \le \text{length} \le C$. Slots `0` through `length - 1` hold the keys and values of the tokens written so far, oldest first. Slots `length` through `C - 1` are **unused**, and their contents are arbitrary.

![A buffer of capacity $C = 8$ before and after a write of two tokens. Attention reads only the slots below the length.](assets/buffer-layout.svg)

**`cache_init(capacity, batch, heads, dim)`** returns an empty cache where `k` and `v` are zero-filled float32 tensors of shape `(batch, heads, capacity, dim)` and `length` is `0`.

**`cache_write(cache, k_new, v_new)`** writes the keys and values of `T` new tokens into slots `length` through `length + T - 1` and returns the updated cache `(k, v, length + T)`.

- `k_new` and `v_new` have shape `(B, H, T, D)`.
- The returned buffers must keep shape `(B, H, C, D)` on every call, and slots `0` through `length - 1` must keep their previous contents.
- If `length + T > C`, raise `ValueError`. A write that fills the buffer exactly (`length + T == C`) is valid.

**`cache_attend(q, cache)`** computes causal scaled dot-product attention for the `T` most recently written tokens. The query tensor `q` has shape `(B, H, T, D)` with $1 \le T \le \text{length}$. Query $i$ (for $0 \le i < T$) corresponds to the token in slot $\text{length} - T + i$, so it attends to itself and every earlier slot $j \le \text{length} - T + i$.

Using $q_i$, $k_j$, and $v_j$ for the vectors of a single batch row and head, attention computes the scaled dot-product scores $s_{ij}$, softmax weights $a_{ij}$, and weighted value sum $\text{out}_i$ as

$$
\text{out}_i = \sum_{j \le \text{length} - T + i} a_{ij}\, v_j,
\qquad
a_{ij} = \frac{e^{s_{ij}}}{\sum_{l \le \text{length} - T + i} e^{s_{il}}},
\qquad
s_{ij} = \frac{q_i \cdot k_j}{\sqrt{D}},
$$

and returns `out` of shape `(B, H, T, D)`. This matches the computation in `kv-cache-append`, split into two functions. A decoder calls `cache_write` first and then calls `cache_attend` with the queries for the tokens it just wrote.

> [!WARNING]
> The unused slots may hold anything, including zeros, leftover entries from an earlier sequence, huge numbers, infinities, or NaN. The output of `cache_attend` must match what you would get if those slots did not exist. The tests fill unused slots with NaN to verify this.

**State in, state out.** `cache_write` returns the cache to continue with, and the caller never uses the previous cache again. Under this contract, you may either update the buffers in place (using slice assignment in PyTorch or MLX) or return new buffers of the same shape (as required in JAX, where arrays are immutable). Both approaches are valid and indistinguishable to the tests. Returning a buffer whose shape depends on `length` is not allowed.

You may use [`ops.attention`](../../../python/src/learnai/kit/ops/__init__.py) from the kit (`from learnai.kit import ops`) for the attention arithmetic.

## Shapes

| name | shape | dtype |
|---|---|---|
| `k`, `v` in a cache | `(B, H, C, D)` | float32 |
| `length` in a cache | Python `int` | |
| `k_new`, `v_new` | `(B, H, T, D)` | float32 |
| `q` | `(B, H, T, D)` | float32 |
| result of `cache_attend` | `(B, H, T, D)` | float32 |

## Example

Consider one batch row (`B = 1`), one head (`H = 1`), head dimension `D = 2`, capacity `C = 4`, and the three tokens from the `kv-cache-append` example:

```python
q = [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]
k = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
v = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]  # each of shape (1, 1, 3, 2) once batched

cache = cache_init(4, 1, 1, 2)
# k = v = [[[[0, 0], [0, 0], [0, 0], [0, 0]]]], length = 0

cache = cache_write(cache, k[:, :, :2], v[:, :, :2])
# k = [[[[1, 0], [0, 1], [0, 0], [0, 0]]]]
# v = [[[[1, 2], [3, 4], [0, 0], [0, 0]]]], length = 2

cache_attend(q[:, :, :2], cache)
# [[[[1.000000, 2.000000],
#    [2.608860, 3.608860]]]]

cache = cache_write(cache, k[:, :, 2:], v[:, :, 2:])
# k = [[[[1, 0], [0, 1], [1, 1], [0, 0]]]]
# v = [[[[1, 2], [3, 4], [5, 6], [0, 0]]]], length = 3

cache_attend(q[:, :, 2:], cache)
# [[[[3.510469, 4.510469]]]]
```

The buffers keep four slots throughout. In the last call, the query $[1, 1]$ has scores $[0.7071, 0.7071, 1.4142]$ against slots 0 through 2, giving softmax weights $[0.2483, 0.2483, 0.5035]$ and output $[3.5105, 4.5105]$.

Slot 3 is unused. Its key is zero, so its raw dot-product score would be $0$, not $-\infty$. A solution that attends over all four slots gives slot 3 an unnormalized weight of $e^0 = 1$ in the softmax and returns $[3.1276, 4.0186]$. A zero key is not an absent key. Once the buffer is full (`length = 4`), any further `cache_write` raises `ValueError`.

## Constraints

- $1 \le B, H \le 4$, $1 \le C \le 8$, $1 \le T \le C$, and $2 \le D \le 16$.
- `cache_attend` is compared against the reference, and against a full causal pass over the tokens written so far, with a relative tolerance of `1e-4` and an absolute tolerance of `1e-5`. Any NaN fails.
- After every write, the tests verify the shape of both buffers, confirm that `length` is a Python `int` with the expected value, and check that slots `0` through `length - 1` match what was written. The contents of unused slots are not checked.
- `cache_init` is checked for exact equality.
- No Python loops over tokens, heads, or batch rows are needed.

## Why this matters

A cache that grows by concatenation copies every stored key and value on every decode step and passes a differently shaped tensor to the attention kernel each time. Inference engines preallocate instead. Hugging Face Transformers' `StaticCache` allocates `(batch, heads, max_cache_len, head_dim)` zeros per layer and writes new entries by index so the model can be compiled with fixed shapes. Similarly, `mlx-lm`'s `KVCache` allocates 256 slots at a time, writes by slice assignment, and slices up to its current offset.

Both designs are vulnerable to the bug this problem tests. Reading one slot past `length`, or assuming the unused tail is harmless, makes the model attend to tokens that were never written. With a zero-filled tail, that produces slightly wrong logits on every step. With a reused buffer, it leaks context from a previous request.
