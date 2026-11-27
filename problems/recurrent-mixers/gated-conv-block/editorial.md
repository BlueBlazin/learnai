## The idea

The block combines an input gate, a short depthwise filter, and an output gate:

$$
y = \bigl(C \odot \operatorname{conv}(B \odot h)\bigr)\,W_{out}^\top,
\qquad [\,B \mid C \mid h\,] = x\,W_{in}^\top .
$$

A single linear projection produces $B$, $C$, and $h$. For each channel, $B$ controls how much of the signal $h$ enters the filter. The filter then mixes each channel with its own previous $L - 1$ values, and $C$ controls how much of the filtered result passes to the output projection. There is no nonlinearity other than the two elementwise products, and no cross-channel mixing inside the convolution itself. Cross-channel mixing happens only in $W_{in}$ and $W_{out}$.

Because output $t$ depends only on $u_{t-L+1}, \dots, u_t$ (where $u = B \odot h$), the block is causal and requires only the last $L - 1$ rows of $u$ to continue a sequence. That fixed-size tensor of $(L - 1) \cdot D$ numbers is the entire state, independent of context length. The state must store $u$, which is the exact tensor the filter reads. Storing rows of $x$ would work only if the input projection and gate were recomputed, and storing the filter output $v$ is useless because a finite impulse response filter has no feedback.

Trained weights fix the tap alignment so that `w_conv[c, L - 1]` multiplies the current token. Every deep-learning framework computes this exact cross-correlation over a left-padded input under the name "convolution". A signal-processing convolution would reverse the taps.

Note also that $B$ and $h$ enter only through their product $B \odot h$, so swapping them does not change the result. By contrast, $C$ sits outside the filter and cannot be swapped with either.

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

`F.conv1d(padded.T[None], w_conv[:, None, :], groups=D)[0].T` (with `D = u.shape[1]`) computes the same sum.

## Common mistakes

- **Symmetric padding.** Using "same" padding centers the kernel window, so output $t$ reads up to $(L - 1) / 2$ future tokens. Appending a new token then changes earlier outputs.
- **Flipped kernel.** Indexing taps backward from the current token (`w[:, j] * u[t - j]`), or using a routine that flips the kernel, fails on asymmetric kernels. The tests use only asymmetric kernels for this reason.
- **Storing `x` in the state.** This tensor has the right shape and passes every single-call test, but fails on multi-call continuations.
- **Storing the convolution output `v` in the state.** This also has the right shape and fails only when state is carried across calls.
- **Resetting the state or slicing only from `u`.** Slicing `u[-(L - 1):]` works whenever a call has at least $L - 1$ tokens, but drops earlier state rows on shorter calls such as single-token decoding steps. Slicing `padded[T:]` handles every call length.
- **Updating the state in place.** Shifting the caller's state tensor row by row produces the right outputs while mutating a tensor the caller still holds.
- **Calling the kit's convolution.** `ops.causal_conv1d` implements steps 3 and 5, so the harness forbids it.
- **Misassigning the three projected blocks.** Filtering $B \odot C$ and gating with $h$, or multiplying by $C$ before the filter, gives the wrong result. The split must also take three contiguous blocks of $D$ columns rather than strided slices.

## Complexity

The time complexity is $O(T \cdot D \cdot (D + L))$. The linear projections dominate, while the depthwise filter adds $L$ multiply-adds per channel and token. The state holds $(L - 1) \cdot D$ numbers, which is 12,288 floats per layer for LFM3-1.7B ($D = 2048$, $L = 7$).

## In the wild

Checked on 2026-11-21 by reading and running `transformers` 5.23.0 and by reading the `config.json` of `LiquidAI/LFM3-1.7B` on the Hugging Face Hub (`conv_L_cache` 7, `conv_bias` false, hidden size 2048, 13 `conv` layers out of 24, `use_sandwich_norm` true).

- `Lfm3ShortConv.forward` in `transformers/models/lfm3/modeling_lfm3.py` runs `B, C, x = in_proj(hidden_states).chunk(3)`, computes `hidden_states = B * x`, applies a depthwise `F.conv1d` over the left-padded or cached sequence, computes `y = C * hidden_states`, and applies `out_proj`. Both the public test "the block of LFM3" and a hidden test compare your function against this module without a cache and with a `DynamicCache` (`fixtures/make.py` regenerates the arrays). Running the module against a reversed kernel confirmed that it does not flip the taps.
- The library's cache stores $L$ rows of $u$ per layer with shape `(D, L)`, which is one row more than the next token needs. This problem stores $L - 1$ rows with shape `(L - 1, D)`.
- The Gated DeltaNet layer in Qwen3.5 (`gated-delta-layer`) and the Mamba-2 block in Nemotron both include a causal depthwise convolution of this form with kernel size 4 (Nemotron adds a bias). In those architectures, the convolution is followed by SiLU and serves as a preprocessing step before a recurrence.

Deliberate simplifications in this problem: there is no batch axis, no padding mask (the library zeroes padded positions before the input projection), and no bias (matching the shipped model). In the full LFM3 layer, a normalization sits before the block, and with `use_sandwich_norm` another normalization sits after it before the residual connection; those are omitted here. No paper for this architecture was read, no weight license was checked, and no model weights are used.
