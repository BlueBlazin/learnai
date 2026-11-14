Implement a KV cache that never changes shape: two buffers allocated once at full
capacity, and a length that says how many slots hold tokens. Writing fills the next
free slots; attention must behave as if the unused tail did not exist, whatever it
happens to contain.

## Specification

```python
def cache_init(capacity, batch, heads, dim): ...  # returns cache
def cache_write(cache, k_new, v_new): ...  # returns cache
def cache_attend(q, cache): ...  # returns out
```

A **cache** is a tuple `(k, v, length)`:

- `k` and `v` are float32 buffers of shape `(B, H, C, D)`: batch, heads, capacity,
  head dimension. Slot `j` along axis 2 belongs to the token at position `j`.
- `length` is a Python int, $0 \le \text{length} \le C$. Slots `0` to `length - 1` hold
  the keys and values of the tokens written so far, oldest first. Slots `length` to
  `C - 1` are **unused**, and their contents are arbitrary.

![A buffer of capacity $C = 8$ holding five tokens. A write of two tokens fills slots 5 and 6 and moves the length to 7; attention reads the slots below the length and nothing else.](assets/buffer-layout.svg)

**`cache_init(capacity, batch, heads, dim)`** returns an empty cache: `k` and `v` are
zeros of shape `(batch, heads, capacity, dim)`, and `length` is `0`.

**`cache_write(cache, k_new, v_new)`** stores the keys and values of `T` new tokens,
of shape `(B, H, T, D)`, in slots `length` to `length + T - 1`, and returns the cache
`(k, v, length + T)`.

- The returned buffers have the same shape `(B, H, C, D)` as before, on every call.
  Slots `0` to `length - 1` keep their contents.
- If `length + T > C`, raise `ValueError`. A write that fills the buffer exactly
  (`length + T == C`) is valid.

**`cache_attend(q, cache)`** computes attention for the `T` most recently written
tokens. `q` has shape `(B, H, T, D)` with $1 \le T \le \text{length}$, and query $i$
belongs to the token in slot $\text{length} - T + i$. It attends to slots
$j \le \text{length} - T + i$: itself and everything before it,

$$
\text{out}_i = \sum_{j \le \text{length} - T + i} a_{ij}\, v_j,
\qquad
a_{ij} = \frac{e^{s_{ij}}}{\sum_{l \le \text{length} - T + i} e^{s_{il}}},
\qquad
s_{ij} = \frac{q_i \cdot k_j}{\sqrt{D}},
$$

and returns `out` of shape `(B, H, T, D)`. This is the step of
`kv-cache-append` with the write and the attention
separated: a decoder calls `cache_write` and then `cache_attend` with the queries of
the tokens it just wrote.

> [!WARNING]
> The unused slots may hold anything: zeros, the entries of an earlier sequence, huge
> numbers, infinities, NaN. The output of `cache_attend` must be exactly what it would
> be if those slots did not exist, and the tests fill them with NaN to check.

**State in, state out.** `cache_write` returns the cache to continue with, and the
caller never uses the one it passed in again. Under that contract a solution may fill
the buffers it was given in place (slice assignment in PyTorch and MLX) or build
updated ones of the same shape (JAX arrays are immutable, so an update there returns
a new array); both are correct, and the tests cannot tell them apart. What is not
allowed is a buffer whose shape depends on `length`.

You may use [`ops.attention`](../../../python/src/whetstone/kit/ops/__init__.py) from
the kit (`from whetstone.kit import ops`) for the attention arithmetic.

## Shapes

| name | shape | dtype |
|---|---|---|
| `k`, `v` in a cache | `(B, H, C, D)` | float32 |
| `length` in a cache | Python `int` | |
| `k_new`, `v_new` | `(B, H, T, D)` | float32 |
| `q` | `(B, H, T, D)` | float32 |
| result of `cache_attend` | `(B, H, T, D)` | float32 |

## Example

One batch row, one head, `D = 2`, capacity 4, and the three tokens of the
`kv-cache-append` example:

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

The buffers have four slots throughout. In the last call the query $[1, 1]$ has
scores $[0.7071, 0.7071, 1.4142]$ against slots 0 to 2, weights
$[0.2483, 0.2483, 0.5035]$, and output $[3.5105, 4.5105]$.

Slot 3 is unused. Its key is zero, so its score would be $0$, not $-\infty$: a
solution that attends over all four slots gives it the weight $e^0$ in the softmax
and returns $[3.1276, 4.0186]$. A zero key is not an absent key. A fifth token
cannot be written: with `length = 4`, any `cache_write` raises `ValueError`.

## Constraints

- $1 \le B, H \le 4$; $1 \le C \le 8$; $1 \le T \le C$; $2 \le D \le 16$.
- `cache_attend` is compared with the reference, and with one full causal pass over
  the tokens written so far, to a relative tolerance of `1e-4` and an absolute
  tolerance of `1e-5`. A NaN is always wrong.
- After every write the tests check the shape of both buffers, that `length` is a
  Python int with the right value, and that slots `0` to `length - 1` hold exactly
  what was written. The contents of the unused slots are not checked.
- `cache_init` is compared exactly.
- No Python loops over tokens, heads, or batch rows are needed.

## Why this matters

A cache that grows by concatenation copies every key and value it holds on every
decode step, and hands the attention kernel a tensor of a new shape each time.
Engines preallocate instead. Hugging Face Transformers' `StaticCache` allocates
`(batch, heads, max_cache_len, head_dim)` zeros per layer and writes new entries by
index, which is what makes the model compilable with fixed shapes; mlx-lm's `KVCache`
allocates 256 slots at a time, writes by slice assignment, and returns the slice
below its offset. Both carry the bug this problem is about one line away: read one
slot too many, or trust the tail to be harmless, and the model attends to tokens
that were never written. With a zero tail the result is slightly wrong logits on
every step; with a reused buffer it is text leaking from the previous request.
