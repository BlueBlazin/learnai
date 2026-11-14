Implement the mask of causal sliding-window attention and attention under it: every
query sees itself and a fixed number of the tokens just before it, and nothing older.
The mask is two inequalities, and each of them has an off-by-one that returns a
plausible band of the right shape.

## Specification

```python
def window_mask(T, window): ...
def sliding_window_attention(q, k, v, window): ...
```

**The window.** Query position $i$ may attend to key position $j$ exactly when

$$
i - \text{window} < j \le i .
$$

So a query sees `window` keys, counting its own position: $j = i - \text{window} + 1,
\dots, i$. Near the start of the sequence there are fewer than `window` keys in the
past, and the query sees all of them: positions $0, \dots, i$.

**`window_mask(T, window)`** returns the boolean matrix of that rule: shape `(T, T)`,
with entry `[i, j]` equal to `True` where query $i$ may attend to key $j$ and `False`
elsewhere. Rows are queries and columns are keys.

**`sliding_window_attention(q, k, v, window)`** is scaled dot-product attention
restricted to the window. `q`, `k`, and `v` have shape `(B, H, T, d)`: batch, heads,
sequence, head width. For one batch item and one head:

$$
s_{ij} = \frac{q_i \cdot k_j}{\sqrt{d}}, \qquad
a_{ij} = \begin{cases}
\dfrac{e^{s_{ij}}}{\sum_{l = \max(0,\, i - \text{window} + 1)}^{i} e^{s_{il}}} & i - \text{window} < j \le i, \\[2ex]
0 & \text{otherwise},
\end{cases}
\qquad
y_i = \sum_{j} a_{ij}\, v_j .
$$

- `T`: a Python int, $T \ge 1$.
- `window`: a Python int, $\text{window} \ge 1$. It may be larger than `T`; the mask is
  then the plain causal mask.
- `q`, `k`, `v`: float32, shape `(B, H, T, d)`.
- `window_mask` returns a bool tensor of shape `(T, T)`.
- `sliding_window_attention` returns float32, shape `(B, H, T, d)`.

Two consequences of the definition are tested directly:

- **Locality.** `y[b, h, i]` does not depend on any key or value at a position
  $j \le i - \text{window}$ or $j > i$.
- **Causal limit.** With $\text{window} \ge T$ the result is ordinary causal attention.

You may use your framework's `softmax`, matrix multiplication, and `einsum`. Do not call
a built-in attention function or layer (`scaled_dot_product_attention`,
`dot_product_attention`, `MultiheadAttention`, and the like).

## Shapes

| name | shape | dtype |
|---|---|---|
| `q`, `k`, `v` | `(B, H, T, d)` | float32 |
| result of `window_mask` | `(T, T)` | bool |
| result of `sliding_window_attention` | `(B, H, T, d)` | float32 |

## Example

```python
window_mask(4, 2)
# [[ True, False, False, False],
#  [ True,  True, False, False],
#  [False,  True,  True, False],
#  [False, False,  True,  True]]
```

Row 0 has a single visible key, because nothing precedes position 0. From row 1 on,
every row has exactly `window = 2` visible keys: the diagonal and one key before it.

Attention with one batch item, one head, three tokens, $d = 2$, and the same tensor as
queries, keys, and values:

```python
x = [[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]]  # shape (1, 1, 3, 2)

sliding_window_attention(x, x, x, window=2)
# [[[[1.000000, 0.000000],
#    [0.330238, 0.669762],
#    [0.669762, 1.000000]]]]
```

- Token 0 sees only itself: $y_0 = x_0$.
- Token 1 sees tokens 0 and 1, with scores $[0, 0.7071]$ and weights
  $[0.3302, 0.6698]$.
- Token 2 sees tokens 1 and 2 only. Their scores are $[0.7071, 1.4142]$, the weights
  are $[0.3302, 0.6698]$, and $y_2 = 0.3302\,[0, 1] + 0.6698\,[1, 1] = [0.6698, 1]$.
  Token 0 has dropped out of the window. With `window=3` it would be visible and the
  last row would be $[0.751745, 0.751745]$, the result of causal attention.

## Constraints

- The mask is compared exactly. Attention results are compared with the reference to
  a relative tolerance of `1e-4` and an absolute tolerance of `1e-5`. A NaN is always
  wrong.
- $1 \le B, H \le 8$; $1 \le T \le 32$; $1 \le d \le 16$; $1 \le \text{window} \le 64$.
- Scores satisfy $|s_{ij}| \le 100$.
- No Python loops over positions are needed.

## Why this matters

Full causal attention makes every new token look at the whole context, so the KV cache
grows without bound and each step costs more than the one before. A sliding window
caps both: a layer that only ever reads the last `window` keys needs to store only
those, in a buffer of fixed size. Mistral 7B was introduced with a window of 4,096
tokens in every layer, and Gemma 3 interleaves five local layers with a span of 1,024
tokens for every global one, which is where most of its long-context memory saving
comes from. The engine must reproduce the window the model was trained with to the
token. One key too many or too few changes every attention distribution slightly, and
it disagrees with a rolling cache that holds exactly `window` entries, so a full
forward pass and cached decoding stop matching.
