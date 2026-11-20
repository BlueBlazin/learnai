## The idea

The block is a multiplicative gate, a short filter, and another gate:

$$
y = \bigl(C \odot \operatorname{conv}(B \odot h)\bigr)\,W_{out}^\top,
\qquad [\,B \mid C \mid h\,] = x\,W_{in}^\top .
$$

One projection produces all three parts. $B$ decides, per channel, how much of the
signal $h$ enters the filter; the filter mixes each channel with its own last
$L - 1$ values; $C$ decides how much of the result leaves. There is no nonlinearity
other than the two products, and no mixing across channels inside the convolution:
that is left to the projections.

Because output $t$ reads $u_{t-L+1}, \dots, u_t$, the block is causal, and continuing
a sequence needs exactly the last $L - 1$ rows of $u$. That is the entire state:
$(L - 1) \cdot D$ numbers, independent of the context length. It must hold $u$, the
tensor the filter reads. Rows of $x$ would do only if the projection and the gate were
recomputed, and rows of the filter's output are of no use at all, since a finite
impulse response filter has no feedback.

The alignment of the taps is a convention, and the one used here is the one in the
weights: `w_conv[c, L - 1]` multiplies the current token. That is what every deep
learning library computes under the name "convolution" (a cross-correlation over a
left-padded input). The convolution of signal processing would reverse the taps.

$B$ and $h$ enter only through their product, so exchanging them changes nothing; $C$
is not exchangeable with either.

## Reference solution (PyTorch)

```python
def conv_block(x, w_in, w_conv, w_out, state):
    steps, kernel = x.shape[0], w_conv.shape[1]
    B, C, h = torch.chunk(x @ w_in.T, 3, dim=-1)
    u = B * h
    if state is None:
        state = torch.zeros(kernel - 1, u.shape[1])
    padded = torch.cat((state, u), dim=0)
    conv = sum(padded[j : j + steps] * w_conv[:, j] for j in range(kernel))
    return (C * conv) @ w_out.T, padded[steps:]
```

`F.conv1d(padded.T[None], w_conv[:, None, :], groups=D)[0].T` is the same sum.

## Common mistakes

- **Symmetric padding.** "Same" padding centers the window, and output $t$ reads up
  to $(L - 1) / 2$ future tokens. Every output of a prompt changes when a token is
  appended.
- **A flipped kernel.** Indexing taps from the current token backwards
  (`w[:, j] * u[t - j]`), or using a routine that implements the mathematical
  convolution. Invisible with a symmetric kernel, which is why the tests use none.
- **State holds `x`.** Same shape as the right answer. Correct for one call.
- **State holds the convolution's output.** Same shape again.
- **State reset on each call,** or **cut from the new rows only.** The second is right
  whenever a call has at least $L - 1$ tokens and wrong for every decoding step.
- **Updating the state in place.** Shifting the caller's state tensor row by row
  gives the right outputs and changes a tensor the caller still holds.
- **Calling the kit's convolution.** `ops.causal_conv1d` does steps 3 and 5; the
  statement rules it out and the harness rejects the call.
- **The thirds in the wrong roles:** filtering $B \odot C$ and gating with $h$, or
  multiplying $C$ in before the filter. The split itself is three consecutive blocks
  of columns, not every third column.

## Complexity

$O(T \cdot D \cdot (D + L))$: the projections dominate, the filter costs $L$
multiply-adds per channel and token. State: $(L - 1) \cdot D$ numbers, 12,288 per
layer for LFM3-1.7B ($D = 2048$, $L = 7$).

## In the wild

Checked on 2026-11-21 by reading and running `transformers` 5.23.0 and by reading the
`config.json` of `LiquidAI/LFM3-1.7B` on the Hugging Face Hub (`conv_L_cache` 7,
`conv_bias` false, hidden size 2048, 13 `conv` layers of 24, `use_sandwich_norm`
true).

- `Lfm3ShortConv.forward` in `transformers/models/lfm3/modeling_lfm3.py`:
  `B, C, x = in_proj(hidden_states).chunk(3)`, `hidden_states = B * x`, a depthwise
  `F.conv1d` over the left-padded (or cached) sequence, `y = C * hidden_states`,
  `out_proj`. The public test "the block of LFM3" and a hidden one compare your
  function with that module, without a cache and with a `DynamicCache`
  (`fixtures/make.py` regenerates the arrays). Running it against a reversed kernel
  confirmed that the library does not flip.
- The library's cache keeps $L$ rows of $u$ per layer as `(D, L)`, one more than the
  next token reads; this problem keeps $L - 1$ as `(L - 1, D)`.
- The Gated DeltaNet layer of Qwen3.5 (`gated-delta-layer`) and the Mamba-2 block of
  Nemotron contain a causal depthwise convolution of the same form with kernel 4 (Nemotron's has a bias), there followed
  by SiLU and used as a preprocessing step for a recurrence.

Deliberate simplifications: no batch axis; no attention mask for padding tokens (the
library zeroes padded positions before the projection); no bias, as shipped. Around
the block, the layer applies a norm before it, and with `use_sandwich_norm` another
after it, then the residual connection; those are not part of this function. No
paper for this architecture was read, and the license of the weights was not checked;
no weights are used.
