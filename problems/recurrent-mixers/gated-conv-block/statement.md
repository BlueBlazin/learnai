Implement the gated convolution block that LFM3 uses in place of attention in most of its layers. The block wraps a short causal depthwise convolution in two multiplicative gates and carries a small fixed-size state across calls.

## Specification

```python
def conv_block(x, w_in, w_conv, w_out, state): ...  # -> (y, state)
```

Let $T \ge 1$ be the number of new tokens, $D$ the channel width, and $L$ the number of convolution taps. Given input `x`, linear projection weights $W_{in}$ (`w_in`) and $W_{out}$ (`w_out`), per-channel convolution kernels $w$ (`w_conv`), and the previous `state`, compute the block output `y` and the updated state as follows:

1. **Project and split.** Compute $x\,W_{in}^\top$ of shape `(T, 3 * D)` and split its columns into three contiguous blocks of width $D$. The first $D$ columns are $B$, the next $D$ are $C$, and the last $D$ are $h$.
2. **Gate the input.** Compute the elementwise product $u = B \odot h$, of shape `(T, D)`.
3. **Convolve causally per channel.** If `state` is `None` (a new sequence), initialize it as $L - 1$ rows of zeros of shape `(L - 1, D)`. Concatenate the $L - 1$ rows of `state` before the $T$ rows of $u$ to form $\tilde{u}$ of shape `(L - 1 + T, D)`. For each token index $t$ and channel $c$, compute:

$$
v_{t,c} = \sum_{j=0}^{L-1} w_{c,j}\;\tilde{u}_{t+j,\,c}, \qquad 0 \le t < T .
$$

   Row $t + L - 1$ of $\tilde{u}$ is the current token at step $t$. The last tap ($j = L - 1$) therefore multiplies the current token, and the first tap ($j = 0$) multiplies the token $L - 1$ positions earlier. Each channel is filtered independently, with no bias and no activation function.
4. **Gate the output and project.** Compute $y = (C \odot v)\,W_{out}^\top$.
5. **Update state.** Return the last $L - 1$ rows of $\tilde{u}$ as the new `state`.

`conv_block` returns `(y, state)`. Do not modify any input tensor, including `state`, in place.

Splitting a sequence across consecutive calls and passing each returned `state` into the next call must produce the exact same outputs and final state as a single call on the full sequence. A call may receive fewer than $L - 1$ tokens.

You may use your framework's built-in convolution functions. Do not call `ops.causal_conv1d` from the kit.

## Shapes

| name | shape | dtype |
|---|---|---|
| `x`, `y` | `(T, D)` | float32 |
| `w_in` | `(3 * D, D)` | float32 |
| `w_conv` | `(D, L)` | float32 |
| `w_out` | `(D, D)` | float32 |
| `state` (argument and result) | `(L - 1, D)` | float32 |

## Example

Consider channel width $D = 1$ and a kernel of $L = 3$ taps:

```python
w_in = [[1.0], [2.0], [1.0]]     # B = x, C = 2x, h = x
w_conv = [[1.0, 10.0, 100.0]]
w_out = [[1.0]]

y, state = conv_block([[1.0], [2.0], [3.0]], w_in, w_conv, w_out, None)
# y = [[200.0], [1640.0], [5646.0]]
# state = [[4.0], [9.0]]

y, state = conv_block([[1.0]], w_in, w_conv, w_out, state)
# y = [[388.0]]
# state = [[9.0], [1.0]]
```

In the first call, $u = B \odot h = x^2 = (1, 4, 9)$. Prepending $L - 1 = 2$ zeros gives $\tilde{u} = (0, 0, 1, 4, 9)$. The tap of weight $100$ always multiplies the current token, so the convolution produces:

- $v_0 = 100 \cdot 1 = 100$
- $v_1 = 10 \cdot 1 + 100 \cdot 4 = 410$
- $v_2 = 1 \cdot 1 + 10 \cdot 4 + 100 \cdot 9 = 941$

Multiplying by the output gate $C = 2x$ gives $y = (2 \cdot 100,\ 4 \cdot 410,\ 6 \cdot 941) = (200, 1640, 5646)$. The returned state holds the last two rows of $u$, not of $x$.

In the second call, prepending `state` to the new $u = (1)$ gives $\tilde{u} = (4, 9, 1)$. The convolution produces $v = 1 \cdot 4 + 10 \cdot 9 + 100 \cdot 1 = 194$, and gating by $C = 2$ gives $y = 388$. Passing `state=None` instead would have produced $2 \cdot 100 = 200$.

## Constraints

- $1 \le T \le 12$, $1 \le D \le 6$, $2 \le L \le 7$.
- Outputs and state are checked against the reference (and for one sequence, against `Lfm3ShortConv` in Hugging Face Transformers with $L = 7$) with relative tolerance `1e-4` and absolute tolerance `1e-5`.
- Kernels are not symmetric, so reversing the taps changes the result.
- Outputs for tokens before position $t$ must not change when tokens at position $t$ or later change (absolute tolerance `1e-6`).
- No Python loop over tokens or channels is needed. A loop over the $L$ taps is fine.

## Why this matters

Thirteen of the 24 layers in LFM3-1.7B use this block. Each layer stores only six rows of state, with no key-value cache and no memory growth as the context lengthens.

The block is short, but two subtle bugs are easy to miss. Deep-learning frameworks implement cross-correlation under the name "convolution", so flipping the kernel or indexing taps backward from the current token produces well-scaled numbers that silently disagree with trained weights. Storing the wrong tensor in the state is also invisible until a sequence is split across calls.
