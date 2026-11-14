## The idea

The output layer is linear and acts on each position independently:
$\text{logits}_{b,t} = W_{\text{out}}\, h_{b,t}$. Selecting a position therefore commutes
with projecting:

$$
\big(H\, W_{\text{out}}^\top\big)_{b,\,T-1} \;=\; H_{b,\,T-1}\; W_{\text{out}}^\top .
$$

The two sides are the same numbers at very different costs. The left side multiplies
$B \cdot T$ vectors by a $V \times D$ matrix and materializes $B \cdot T \cdot V$
logits; the right side multiplies $B$ vectors and materializes $B \cdot V$. The ratio is
exactly $T$, the prompt length.

Which position to select follows from how a decoder is trained: position $t$ has seen
tokens $0..t$ and its output is scored against token $t + 1$. The continuation of the
whole sequence is thus predicted at position $T - 1$. Position 0 predicts the second
token of the prompt, which is already known.

## Reference solution (PyTorch)

```python
def next_token_logits(hidden, w_out):
    last = hidden[:, -1]
    return last @ w_out.T
```

The JAX and MLX versions are character for character the same. Integer indexing drops
the time axis, giving `(B, D)`, and the product with the transposed `(V, D)` matrix
gives `(B, V)`.

## Common mistakes

- **Projecting everything, then slicing.** `(hidden @ w_out.T)[:, -1]` is what you get
  by calling a model that returns full logits and indexing the result. It is correct
  and $T$ times too expensive. No framework here removes the waste for you: eager
  PyTorch has already done the product when the slice runs, and the lazy graphs of
  JAX and MLX compute the full product too.
- **The wrong position.** `hidden[:, 0]` is the first position. `hidden[:, T - 2]`
  comes from confusing the length with the last index; with `T = 1` the index is `-1`
  and wraps around to the right answer, so a single-token test does not catch it.
- **The wrong axis.** `hidden[-1]` is the last *sequence*, shape `(T, D)`. When
  $B = T$ the result even has the right shape.
- **Flattening batch and time.** After `hidden.reshape(B * T, D)` the last position of
  sequence $b$ is row $b\,T + T - 1$. Taking the last `B` rows instead returns the last
  $B$ positions of the final sequence. With a batch of one it is correct.
- **Keeping the time axis.** `hidden[:, -1:]` keeps a time axis of length 1, and the
  result is `(B, 1, V)`. Downstream code that takes an argmax over the last axis still
  runs, and the stray axis surfaces somewhere else.
- **Not transposing the weights.** `last @ w_out` fails on shapes unless $V = D$.

## Complexity

$O(B\,D\,V)$ time and $O(B\,V)$ memory, independent of $T$. The wasteful order is
$O(B\,T\,D\,V)$ time and $O(B\,T\,V)$ memory.

The measured gap is smaller than the factor $T$ in arithmetic, because a product of
this size is fast enough that fixed per-call costs show. On the machine these tests
were written on, for $B = 8$, $T = 512$, $D = 64$, $V = 8192$ the wasteful order took
about 100 times as long in PyTorch on the CPU, 20 times in JAX, and 8 times in MLX on
the GPU, where the single small product is dominated by the cost of dispatching to
the GPU. With a real vocabulary and hidden width the fixed costs vanish and the full
factor appears, along with the memory: a 4,096-token prompt and a 128,000-entry
vocabulary make about $5 \times 10^8$ logits per sequence, 2 GB in float32.

## In the wild

Model code written for training returns logits at every position, since the loss
needs them all. Inference code paths cut the hidden states before the head. Hugging
Face Transformers gives many causal models an argument for exactly this
(`logits_to_keep`, earlier named `num_logits_to_keep`), which slices the hidden states
before the head is applied. Engines built for serving apply the final norm and the
head to the last position only during prefill, and decoding steps have a single
position anyway.

Two refinements go beyond this problem. In a padded batch the last real token of each
row is at a different index, so the selection becomes a gather by sequence length
(the `last-token-gather` problem). And when a request asks for the log-probabilities
of its prompt tokens, every position is needed after all, which is one reason that
option is expensive.
