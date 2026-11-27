## The idea

Let $H$ be the hidden-state tensor of shape `(B, T, D)` and $W_{\text{out}}$ be the output matrix of shape `(V, D)`. Because the output layer is linear and acts on each position independently ($\text{logits}_{b,t} = H_{b,t}\, W_{\text{out}}^\top$), selecting a position commutes with the projection:

$$
\big(H\, W_{\text{out}}^\top\big)_{b,\,T-1} \;=\; H_{b,\,T-1}\; W_{\text{out}}^\top .
$$

Both sides produce the exact same values at very different costs. The left side multiplies $B \cdot T$ vectors by a $V \times D$ matrix and materializes $B \cdot T \cdot V$ logits. The right side multiplies only $B$ vectors and materializes $B \cdot V$ logits, reducing both work and memory by a factor of $T$, the sequence length.

Selecting position $T - 1$ follows from how autoregressive decoders are trained. Position $t$ has attended to tokens $0 \dots t$ and its logits are scored against token $t + 1$. The continuation of the sequence is therefore predicted at position $T - 1$, whereas position 0 predicts the second prompt token, which is already known.

## Reference solution (PyTorch)

```python
def next_token_logits(hidden, w_out):
    last = hidden[:, -1]
    return last @ w_out.T
```

The JAX and MLX versions are identical. Integer indexing along axis 1 drops the time dimension, yielding shape `(B, D)`, and multiplying by the transposed `(V, D)` matrix produces shape `(B, V)`.

## Common mistakes

- **Projecting everything, then slicing.** Computing `(hidden @ w_out.T)[:, -1]` gives the right numbers but does $T$ times too much work. None of the frameworks eliminate this waste automatically. Eager PyTorch completes the full matrix product before slicing, and the lazy graphs in JAX and MLX also compute the full product.
- **Selecting the wrong position.** `hidden[:, 0]` selects the first position instead of the last. `hidden[:, T - 2]` comes from confusing the sequence length with the last index. When `T = 1`, `T - 2` evaluates to `-1` and wraps around to the right element, so single-token tests miss this bug.
- **Indexing the wrong axis.** `hidden[-1]` selects the last *sequence* rather than the last position, producing shape `(T, D)`. When $B = T$, the output even has the expected shape.
- **Flattening batch and time incorrectly.** After `hidden.reshape(B * T, D)`, the last position of sequence $b$ sits at row $b\,T + T - 1$. Taking the last `B` rows instead grabs the last $B$ positions of the final sequence, a bug that passes when $B = 1$.
- **Keeping the time axis.** Slicing with `hidden[:, -1:]` preserves a length-1 time axis and returns shape `(B, 1, V)`. Downstream code that calls `argmax` over the last axis still runs, hiding the stray dimension until later.
- **Forgetting to transpose `w_out`.** Computing `last @ w_out` raises a shape mismatch whenever $V \ne D$ and silently computes the wrong product when $V = D$.

## Complexity

Slicing first takes $O(B\,D\,V)$ time and $O(B\,V)$ memory, independent of $T$. Projecting first takes $O(B\,T\,D\,V)$ time and $O(B\,T\,V)$ memory.

In practice, the measured speedup is smaller than $T$ when the single matrix product is small enough that fixed per-call overhead shows up. On the test machine for $B = 8$, $T = 512$, $D = 64$, $V = 8192$, projecting first took about 100 times longer in PyTorch on CPU, 20 times longer in JAX, and 8 times longer in MLX on GPU (where GPU dispatch overhead dominates the small product). At realistic model sizes, fixed overhead vanishes and the full factor appears in both latency and memory. For example, a 4,096-token prompt with a 128,000-entry vocabulary produces roughly $5 \times 10^8$ logits per sequence, or 2 GB in float32.

## In the wild

Training implementations return logits for every position because the cross-entropy loss scores the whole sequence. Inference paths slice the hidden states before the output head. Hugging Face Transformers exposes a `logits_to_keep` argument (formerly `num_logits_to_keep`) on many causal models for this exact purpose. Dedicated serving engines apply the final normalization layer and output head only to the last position during prefill, while decoding steps already have a single position per sequence.

Two practical cases extend this pattern. In a padded batch, each sequence ends at a different index, so slicing becomes a gather indexed by sequence length (covered in `last-token-gather`). Also, when a user requests log-probabilities for the prompt tokens themselves, the server must unembed every position, which is why prompt logprobs are expensive.
