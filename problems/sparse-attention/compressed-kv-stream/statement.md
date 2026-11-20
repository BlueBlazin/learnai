Compress a stream of key-value rows into one cache entry per window of `rate` rows,
when the stream arrives in pieces of any size. The windows belong to the stream, not
to the calls, so rows of an unfinished window have to wait in a state between calls.

## Specification

```python
def compress(kv, gate, position_bias, gain, state, rate): ...  # -> (entries, state)
def visible_entries(t, rate): ...  # -> int
```

**The entries.** Number the rows of the whole stream $0, 1, 2, \dots$ in arrival order,
across all calls. Window $w$ is the rows $w \cdot \texttt{rate}, \dots,
(w + 1) \cdot \texttt{rate} - 1$; the row at offset $j$ of a window is its $j$-th row,
$0 \le j < \texttt{rate}$. With $c_j, z_j \in \mathbb{R}^d$ the `kv` and `gate` rows of
window $w$ and $b_j$ row $j$ of `position_bias`, the entry of window $w$ is

$$
p = \sum_{j=0}^{\texttt{rate}-1} a_j \odot c_j, \qquad
a_j = \frac{\exp(z_j + b_j)}{\sum_{l=0}^{\texttt{rate}-1} \exp(z_l + b_l)}, \qquad
e_w = \texttt{gain} \odot \frac{p}{\sqrt{\tfrac{1}{d}\sum_{i} p_i^2 + 10^{-6}}} .
$$

All operations on vectors are per channel: the softmax runs **over the `rate` rows of
the window, separately for each channel**, and $\odot$ is the elementwise product.
The norm is one root mean square over the $d$ channels of $p$.

**A call** brings the next $n$ rows of the stream. `state` is
`(buf_kv, buf_gate, count)`:

- `buf_kv`, `buf_gate`: the rows received so far that belong to the unfinished
  window, oldest first, exactly as they were received (no bias added): float32 of
  shape `(r, d)` each, $0 \le r < \texttt{rate}$.
- `count`: a Python int, the number of entries emitted so far.

Before the first call the state is two tensors of shape `(0, d)` and `0`.

`compress` returns `(entries, state)`. `entries` holds one row for each window that
this call completes, oldest window first, as float32 of shape `(m, d)` with
$m = (r + n) \mathbin{//} \texttt{rate}$; a call that completes no window returns
shape `(0, d)`. An unfinished window is never emitted. The new state holds the
$(r + n) - m \cdot \texttt{rate}$ rows that are left and `count + m`.

**`visible_entries(t, rate)`** returns, as a Python int, how many entries the query at
position $t \ge 0$ may attend to: entry $w$ is visible to it exactly when the last row
of window $w$ is at a position $\le t$. (Row $i$ of the stream is the token at
position $i$.) The count is exact for every $t$ in the stated range and is computed
in constant time: a loop over windows does not finish for the largest $t$.

Arguments:

- `kv`, `gate`: float32, shape `(n, d)`, $n \ge 1$.
- `position_bias`: float32, shape `(rate, d)`. `gain`: float32, shape `(d,)`.
- `rate`: a Python int $\ge 1$. `t`: a Python int $\ge 0$.
- The tensors and the state passed in are not modified: the same state can be
  continued again.

## Shapes

| name | shape | dtype |
|---|---|---|
| `kv`, `gate` | `(n, d)` | float32 |
| `position_bias` | `(rate, d)` | float32 |
| `gain` | `(d,)` | float32 |
| `buf_kv`, `buf_gate` | `(r, d)`, $0 \le r <$ `rate` | float32 |
| `entries` | `(m, d)`, $m \ge 0$ | float32 |

## Example

A window of 2, width 2, four rows that arrive as three and then one:

```python
kv = [[4, 2], [0, 6], [1, 1], [-3, 3]]
gate = [[0, 0], [0, 0], [0, 0], [log(3), 0]]
position_bias = [[log(3), 0], [0, 0]]
gain = [1, 2]
state = (zeros((0, 2)), zeros((0, 2)), 0)

entries, state = compress(kv[:3], gate[:3], position_bias, gain, state, 2)
# entries = [[0.848528, 2.262742]]; state = ([[1, 1]], [[0, 0]], 1)
entries, state = compress(kv[3:], gate[3:], position_bias, gain, state, 2)
# entries = [[-0.632455, 2.529822]]; state = (shape (0, 2), shape (0, 2), 2)
```

Window 0 is rows 0 and 1. In channel 0 the gates plus bias are $(\ln 3, 0)$, so the
weights are $(0.75, 0.25)$; in channel 1 they are $(0.5, 0.5)$. That gives
$p = (0.75 \cdot 4,\ 0.5 \cdot 2 + 0.5 \cdot 6) = (3, 4)$, with root mean square
$\sqrt{12.5} = 3.5355$, and $e_0 = (1, 2) \odot (0.8485, 1.1314)$. Row 2 opens window 1
and waits.

The second call completes window 1 with the waiting row at offset 0 and the new row
at offset 1. In channel 0 the gates plus bias are $(0 + \ln 3,\ \ln 3 + 0)$: equal
weights. $p = (-1, 2)$, root mean square $\sqrt{2.5}$.

`visible_entries(t, 2)` is 0, 1, 1, 2 for $t = 0, 1, 2, 3$: the query at position 1,
the last row of window 0, already sees entry 0.

## Constraints

- $1 \le n \le 512$, $1 \le d \le 64$, $1 \le \texttt{rate} \le 128$, streams of at
  most 512 rows; $|z_j + b_j| \le 400$; $0 \le t < 2^{31}$.
- `entries` and the buffers in the returned state are compared with the definition
  evaluated in float64, and for two streams with the output of
  `DeepseekV4HCACompressor` in Hugging Face Transformers: relative tolerance `1e-4`,
  absolute tolerance `1e-5`. `count` and `visible_entries` are compared exactly.
- Results must be finite for every valid input, including a window whose `kv` rows
  are all zero (its entry is zero).
- However a stream is cut into calls, the concatenated entries are the same.
- You may use your framework's `softmax`.

## Why this matters

DeepSeek-V4 keeps two kinds of memory per layer: recent tokens as they are, and the
distant past compressed by windows, 128 tokens to one entry in its heavily compressed
layers. The entry count grows by one per 128 tokens, which is what makes a context of
a million tokens fit. A server never sees the context at once: it arrives as a prefill
chunk, then one token per step. An implementation that restarts its windows at each
call, emits a partial window, or lets a query see the entry of a window it is still
inside gives different entries for the same text, depending on how the request was
batched.
