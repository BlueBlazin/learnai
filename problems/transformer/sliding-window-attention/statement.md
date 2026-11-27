Implement the boolean visibility mask for causal sliding-window attention and compute scaled dot-product attention using that mask. Under a sliding window, each query token attends only to itself and a fixed number of immediately preceding tokens, ignoring all future tokens and all older tokens outside the window.

![This problem is attention, limited to a window of the most recent tokens.](assets/map.svg)

## Specification

```python
def window_mask(T, window): ...
def sliding_window_attention(q, k, v, window): ...
```

Let `T` ($T \ge 1$) be the sequence length and `window` ($\text{window} \ge 1$) be the window size, both Python integers. For a query at position $i$ and a key at position $j$ (with $0 \le i, j < T$), query $i$ may attend to key $j$ if and only if

$$
i - \text{window} < j \le i .
$$

Each query therefore attends to at most `window` keys, including its own position: $j = i - \text{window} + 1, \dots, i$. Near the start of the sequence, when fewer than `window` positions exist in the past, the query attends to all available positions $0, \dots, i$. When $\text{window} \ge T$, the rule reduces to standard causal attention.

**`window_mask(T, window)`** returns the boolean mask of this rule with shape `(T, T)`. Rows correspond to queries and columns correspond to keys. Entry `[i, j]` is `True` when query $i$ may attend to key $j$ and `False` otherwise.

**`sliding_window_attention(q, k, v, window)`** computes scaled dot-product attention restricted to this window. The inputs `q`, `k`, and `v` have shape `(B, H, T, d)`, representing batch size `B`, number of heads `H`, sequence length `T`, and head dimension `d`. For a single batch item and head, the scaled scores $s_{ij}$, attention weights $a_{ij}$, and output vectors $y_i$ are

$$
s_{ij} = \frac{q_i \cdot k_j}{\sqrt{d}}, \qquad
a_{ij} = \begin{cases}
\dfrac{e^{s_{ij}}}{\sum_{l = \max(0,\, i - \text{window} + 1)}^{i} e^{s_{il}}} & i - \text{window} < j \le i, \\[2ex]
0 & \text{otherwise},
\end{cases}
\qquad
y_i = \sum_{j} a_{ij}\, v_j .
$$

Two properties of this definition are tested directly:

- **Locality.** Output `y[b, h, i]` does not depend on any key or value at position $j \le i - \text{window}$ or $j > i$.
- **Causal limit.** When $\text{window} \ge T$, the output matches standard causal attention.

You may use your framework's `softmax`, matrix multiplication, and `einsum`. Do not call a built-in attention function or layer (`scaled_dot_product_attention`, `dot_product_attention`, `MultiheadAttention`, or similar).

## Shapes

| name | shape | dtype |
|---|---|---|
| `q`, `k`, `v` | `(B, H, T, d)` | float32 |
| result of `window_mask` | `(T, T)` | bool |
| result of `sliding_window_attention` | `(B, H, T, d)` | float32 |

## Example

For `T = 4` and `window = 2`:

```python
window_mask(4, 2)
# [[ True, False, False, False],
#  [ True,  True, False, False],
#  [False,  True,  True, False],
#  [False, False,  True,  True]]
```

Row 0 has one visible key because no tokens precede position 0. From row 1 onward, every row has exactly `window = 2` visible keys, namely the diagonal and one position before it.

Now consider attention with $B = 1$, $H = 1$, $T = 3$, and $d = 2$, using the same tensor for queries, keys, and values:

```python
x = [[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]]  # shape (1, 1, 3, 2)

sliding_window_attention(x, x, x, window=2)
# [[[[1.000000, 0.000000],
#    [0.330238, 0.669762],
#    [0.669762, 1.000000]]]]
```

- Token 0 attends only to itself, so $y_0 = x_0$.
- Token 1 attends to tokens 0 and 1, with scaled scores $[0, 0.7071]$ and softmax weights $[0.3302, 0.6698]$.
- Token 2 attends only to tokens 1 and 2, since token 0 falls outside the window. Their scaled scores are $[0.7071, 1.4142]$ and their weights are $[0.3302, 0.6698]$, giving $y_2 = 0.3302\,[0, 1] + 0.6698\,[1, 1] = [0.6698, 1]$. With `window=3`, token 0 would remain visible and the last row would be $[0.751745, 0.751745]$, matching full causal attention.

## Constraints

- `window_mask` is compared for exact equality. Attention outputs are compared against the reference with a relative tolerance of `1e-4` and an absolute tolerance of `1e-5`. Any NaN fails the test.
- $1 \le B, H \le 8$, $1 \le T \le 32$, $1 \le d \le 16$, $1 \le \text{window} \le 64$.
- Scaled attention scores satisfy $|s_{ij}| \le 100$.
- No Python loops over positions are needed.

## Why this matters

In full causal attention, every new token attends to the entire preceding context, so the KV cache grows linearly with sequence length and each decoding step does more work than the last. A sliding window caps both. A layer that reads only the most recent `window` keys needs a fixed-size rolling buffer to store them.

Mistral 7B launched with a 4,096-token window in every layer, and Gemma 3 interleaves five 1,024-token local layers for every global layer to save most of its long-context KV cache memory. An inference engine must match the training window to the exact token. Including one key too many or too few shifts every attention distribution and causes a full forward pass to disagree with a rolling KV cache that holds exactly `window` entries.
