Run a decoder on a batch of sequences of different lengths, in one pass, so that every
sequence gets exactly the logits it would get alone. Batching needs a rectangular
tensor, so the shorter sequences are padded, and the whole problem is making sure the
padding changes nothing: not through attention, not through positions, and not through
where the answer is read.

## Specification

```python
def batched_forward(params, cfg, seqs, pad_id): ...
```

`params` and `cfg` describe the decoder of the `decoder-forward` problem (the kit's
[`DecoderConfig` and `decoder_params`](../../../python/src/whetstone/kit/decoder.py)):
embedding, `cfg.n_layers` pre-norm blocks with RoPE and grouped-query attention, final
RMSNorm, and a head that is the embedding matrix when `cfg.tie_embeddings` is `True`
and `lm_head.weight` otherwise. Write $\operatorname{forward}(s)$ for that model's
logits on the single sequence $s$, with a causal mask and positions $0, 1, 2, \dots$

`seqs` is a list of $B$ sequences of token ids with lengths $L_0, \dots, L_{B-1}$,
each at least 1. The function returns a tensor of shape `(B, vocab_size)` whose row
$b$ is the next-token logits of sequence $b$:

$$
\text{result}[b] = \operatorname{forward}(\text{seqs}[b])[L_b - 1] .
$$

That equation is the entire requirement on the output. The function must produce it
from **one** batched pass over a padded tensor, built as follows. Let
$T = \max_b L_b$ and $P_b = T - L_b$, the number of padding slots in row $b$.

1. **Pad on the left.** Row $b$ of the `(B, T)` token tensor is $P_b$ copies of
   `pad_id` followed by `seqs[b]`. Column $j$ of row $b$ is *real* when $j \ge P_b$.
   The last real token of every row is then in the last column, $T - 1$.
2. **Mask.** Query position $i$ of row $b$ may attend to key position $j$ exactly when
   $j \le i$ and column $j$ of row $b$ is real. A padding query may therefore attend
   to nothing, and its output is never used.
3. **Positions.** The token in column $j$ of row $b$ has RoPE position $j - P_b$, so
   that the first real token of every row is at position 0, as it is when the sequence
   runs alone. The positions of padding columns do not matter.
4. **Read out** the logits of the last column of every row.

Whether a column is padding is decided by the lengths, not by the token values:
`pad_id` is an ordinary token id and may also occur in `seqs` as a real token. (Many
models have no dedicated padding token and reuse the end-of-sequence id.)

The starter contains a complete `decoder_block(cfg, block, x, mask, positions)`: one
block of the model, taking a boolean mask that broadcasts to `(B, n_heads, T, T)`
(`True` where the query on the third axis may attend to the key on the last axis) and
integer positions of shape `(T,)` or `(B, T)`. Its attention gives a query that may
attend to nothing an output of zero rather than NaN. You write everything around it.

- `params`: a dict from checkpoint-style names to float32 tensors, as in
  `decoder-forward`. `layer_params(params, i)` from `whetstone.kit` returns the
  weights of layer `i` with the `model.layers.{i}.` prefix removed, which is what
  `decoder_block` takes.
- `cfg`: the `DecoderConfig`.
- `seqs`: a list of `B` lists of Python ints in `[0, vocab_size)`; $1 \le L_b$.
- `pad_id`: a Python int in `[0, vocab_size)`.
- Returns float32, shape `(B, vocab_size)`.

Do not call the kit's own forward pass, `decoder_forward_ref` or `decoder_block_ref`.

## Shapes

| name | shape | dtype |
|---|---|---|
| `seqs` | `B` lists of lengths $L_b$ | Python ints |
| padded tokens (internal) | `(B, T)`, $T = \max_b L_b$ | integer |
| mask (internal) | broadcastable to `(B, n_heads, T, T)` | bool |
| positions (internal) | `(B, T)` | integer |
| result | `(B, vocab_size)` | float32 |

## Example

A two-layer model of width 4 with the kit's seeded weights, and three sequences of
lengths 3, 1, and 2:

```python
from whetstone.kit import DecoderConfig, decoder_params

cfg = DecoderConfig(vocab_size=5, d_model=4, n_layers=2, n_heads=2, n_kv_heads=1, d_mlp=6)
params = decoder_params(cfg, seed=0)

batched_forward(params, cfg, [[1, 3, 4], [2], [4, 0]], pad_id=0)
# [[-0.525848, -3.150511,  2.285472,  0.695466, -1.054566],
#  [-0.656692, -3.295562,  2.559741,  2.571999, -1.360637],
#  [ 0.285671, -1.890282, -2.674756, -4.402627, -2.909733]]
```

Here $T = 3$, and the padded batch, its real columns, and its positions are

```python
tokens    = [[1, 3, 4],   real = [[True,  True,  True],   positions = [[ 0,  1, 2],
             [0, 0, 2],           [False, False, True],                [-2, -1, 0],
             [0, 4, 0]]           [False, True,  True]]                [-1,  0, 1]]
```

Row 1 is the sequence `[2]` alone: its only real token attends to itself and nothing
else. If it also attended to the two padding tokens in front of it, the model would be
computing the logits of the three-token sequence `[0, 0, 2]`, which are
`[0.136466, -1.268586, 2.554092, 2.141590, -0.205061]`: plausible numbers, and wrong.

Row 2 ends in a real token `0`, the same id as the padding. Its last column is real
and its first column is not, although both hold a `0`.

## Constraints

- Every row is compared with the kit's oracle on that sequence alone, to a relative
  tolerance of `1e-3` and an absolute tolerance of `1e-4`. A NaN is always wrong.
- $1 \le B \le 4$, $1 \le L_b \le 8$, `n_layers` $\le 4$.
- Compute all rows in one batched pass over the `(B, T)` tensor; do not loop over the
  sequences.
- Do not call `decoder_forward_ref` or `decoder_block_ref`.

## Why this matters

A serving engine never runs one request at a time: throughput comes from batching, and
real requests never have the same length. The contract of a batched engine is that
batching is invisible, so a request must get the same logits whether it shared a
forward pass with a long neighbor, a short one, or none. When padding leaks, it leaks
quietly. Outputs stay fluent, and they depend on what else was in the batch, so the
same prompt gives different completions under load than in testing, and the bug
cannot be reproduced with a single request. Left padding is the convention for
generation because the next token is appended on the right, where every row ends.
