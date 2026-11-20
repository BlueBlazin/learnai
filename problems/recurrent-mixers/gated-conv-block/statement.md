Implement the convolution block that LFM3 uses in place of attention in most of its
layers: two gates around a short causal depthwise convolution, with the last few
convolution inputs carried from call to call.

## Specification

```python
def conv_block(x, w_in, w_conv, w_out, state): ...  # -> (y, state)
```

For $T$ new tokens of width $D$ and a kernel of $L$ taps:

1. **Project and split.** $P = x\,W_{in}^\top$ has shape $(T, 3D)$. Its first $D$
   columns are $B$, the next $D$ are $C$, and the last $D$ are $h$.
2. **Gate the input.** $u = B \odot h$, shape $(T, D)$.
3. **Convolve causally, per channel.** Let $\tilde u$ be the $L - 1$ rows of `state`
   followed by the $T$ rows of $u$. With $w$ = `w_conv`:

$$
v_{t,c} = \sum_{j=0}^{L-1} w_{c,j}\;\tilde u_{t+j,\,c}, \qquad 0 \le t < T .
$$

   Row $t + L - 1$ of $\tilde u$ is the current token, so the **last** tap multiplies
   the current token and the first tap the token $L - 1$ positions back. Channels do
   not mix, there is no bias, and no activation follows.
4. **Gate the output and project.** $y = (C \odot v)\,W_{out}^\top$.
5. **New state.** The last $L - 1$ rows of $\tilde u$.

- `x`: float32, shape `(T, D)`, $T \ge 1$.
- `w_in`: float32, shape `(3D, D)`; `w_out`: float32, shape `(D, D)`.
- `w_conv`: float32, shape `(D, L)`, one kernel per channel.
- `state`: `None` for a new sequence (the rows before the first token are zeros), or
  float32 of shape `(L - 1, D)` as returned by the call that read the tokens before
  `x`.
- Returns `(y, state)`: float32 of shape `(T, D)` and float32 of shape `(L - 1, D)`.
  The state is the last argument and the second result; the tensors passed in, the
  state included, are not modified.

Any split of a sequence into consecutive calls, each given the state the previous one
returned, gives the outputs and the final state of a single call. A call may have
fewer than $L - 1$ tokens.

Your framework's convolution functions are allowed. Do not call the kit's
`ops.causal_conv1d`.

## Shapes

| name | shape | dtype |
|---|---|---|
| `x`, `y` | `(T, D)` | float32 |
| `w_in` | `(3 * D, D)` | float32 |
| `w_conv` | `(D, L)` | float32 |
| `w_out` | `(D, D)` | float32 |
| `state` (argument and result) | `(L - 1, D)` | float32 |

## Example

Width $D = 1$ and a kernel of three taps:

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

Here $u = x^2 = (1, 4, 9)$ and $\tilde u = (0, 0, 1, 4, 9)$. The convolution gives
$v_0 = 100 \cdot 1$, $v_1 = 10 \cdot 1 + 100 \cdot 4 = 410$, and
$v_2 = 1 \cdot 1 + 10 \cdot 4 + 100 \cdot 9 = 941$: the tap of weight 100 always meets
the current token. The output gate is $C = 2x$, so $y = (2 \cdot 100,\ 4 \cdot 410,\
6 \cdot 941)$. The state is the last two rows of $u$, not of $x$.

The second call sees $\tilde u = (4, 9, 1)$: $v = 4 + 90 + 100 = 194$ and
$y = 2 \cdot 194$. With `state=None` it would have returned $2 \cdot 100 = 200$.

## Constraints

- $1 \le T \le 12$, $1 \le D \le 6$, $2 \le L \le 7$.
- Outputs and the state are compared with the reference, and for one sequence with
  the outputs of `Lfm3ShortConv` in Hugging Face Transformers with $L = 7$: relative
  tolerance `1e-4`, absolute `1e-5`.
- Kernels are not symmetric: reversing the taps changes the result.
- Outputs for tokens before position $t$ must not change when tokens at $t$ or later
  change (absolute tolerance `1e-6`).
- No Python loop over tokens or channels is needed; a loop over the $L$ taps is fine.

## Why this matters

Thirteen of the 24 layers of LFM3-1.7B are this block, and its whole memory of the
prefix is six rows per layer: no key-value cache, no growth with context. It is cheap
to get subtly wrong. Deep-learning libraries call the operation of step 3 a
convolution and implement a cross-correlation, so a port through a library that flips
the kernel, or a hand-written loop that indexes taps from the current token
backwards, runs and returns well-scaled numbers that match nothing; and a state that
stores the wrong tensor is invisible until a sequence is split across calls.
