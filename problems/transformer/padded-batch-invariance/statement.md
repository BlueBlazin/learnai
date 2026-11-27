Run a transformer decoder on a batch of sequences with different lengths in a single forward pass so that every sequence produces the exact same next-token logits it would produce alone. Because batching requires a rectangular tensor, shorter sequences must be padded without letting padding affect attention, rotary positions, or output readout.

![This problem is attention and RoPE, which must ignore the padding of a batch of unequal lengths.](assets/map.svg)

## Specification

```python
def batched_forward(params, cfg, seqs, pad_id): ...
```

The arguments `params` and `cfg` define the decoder from `decoder-forward` (see [`DecoderConfig` and `decoder_params`](../../../python/src/learnai/kit/decoder.py)). The model consists of a token embedding table, `cfg.n_layers` pre-norm transformer blocks with RoPE and grouped-query attention, a final RMSNorm, and an output head (using the embedding matrix when `cfg.tie_embeddings` is `True`, or `lm_head.weight` otherwise). Let $\operatorname{forward}(s)$ denote this model's output logits on a single sequence $s$ with a causal mask and positions $0, 1, 2, \dots$.

`seqs` is a list of $B$ token-id sequences with lengths $L_0, \dots, L_{B-1}$ (each $L_b \ge 1$ and each token id in `[0, vocab_size)`). `pad_id` is a Python int in `[0, vocab_size)`. Return a float32 tensor of shape `(B, vocab_size)` whose row $b$ equals the next-token logits at the last token of sequence $b$:

$$
\text{result}[b] = \operatorname{forward}(\text{seqs}[b])[L_b - 1].
$$

Compute all rows in **one** batched pass over a padded `(B, T)` tensor, where $T = \max_b L_b$ is the maximum sequence length and $P_b = T - L_b$ is the number of padding slots in row $b$:

1. **Pad on the left.** Fill row $b$ with $P_b$ copies of `pad_id` followed by `seqs[b]`. Column $j$ of row $b$ is *real* when $j \ge P_b$, placing every sequence's final real token in column $T - 1$.
2. **Build the attention mask.** Query position $i$ in row $b$ may attend to key position $j$ if and only if $j \le i$ and column $j$ of row $b$ is real. A padding query therefore attends to no keys, and its output is never read.
3. **Compute per-row positions.** Assign column $j$ of row $b$ the RoPE position $j - P_b$, so the first real token of every sequence starts at position 0 just as in a solo run. Positions assigned to padding columns do not affect the output.
4. **Read out the last column.** Extract the logits from column $T - 1$ of every row.

Determine which columns are padding from the sequence lengths, never by comparing token ids against `pad_id`. `pad_id` is a valid token id and may appear inside `seqs` as a real token (many tokenizers have no dedicated pad token and reuse the end-of-sequence id).

The starter code provides a complete `decoder_block(cfg, block, x, mask, positions)`. It accepts a boolean `mask` broadcastable to `(B, n_heads, T, T)` (`True` where the query on axis 2 may attend to the key on axis 3) and integer `positions` of shape `(T,)` or `(B, T)`. When a query has no valid keys to attend to, the provided block outputs zero instead of NaN. Use `layer_params(params, i)` from `learnai.kit` to strip the `model.layers.{i}.` prefix from `params` for layer `i`.

Do not call the kit's reference forward pass (`decoder_forward_ref` or `decoder_block_ref`).

## Shapes

| name | shape | dtype |
|---|---|---|
| `seqs` | `B` lists of lengths $L_b$ | Python ints |
| padded tokens (internal) | `(B, T)`, $T = \max_b L_b$ | integer |
| mask (internal) | broadcastable to `(B, n_heads, T, T)` | bool |
| positions (internal) | `(B, T)` | integer |
| result | `(B, vocab_size)` | float32 |

## Example

Consider a two-layer decoder of width 4 with seeded weights from the kit and three sequences of lengths 3, 1, and 2:

```python
from learnai.kit import DecoderConfig, decoder_params

cfg = DecoderConfig(vocab_size=5, d_model=4, n_layers=2, n_heads=2, n_kv_heads=1, d_mlp=6)
params = decoder_params(cfg, seed=0)

batched_forward(params, cfg, [[1, 3, 4], [2], [4, 0]], pad_id=0)
# [[-0.525848, -3.150511,  2.285472,  0.695466, -1.054566],
#  [-0.656692, -3.295562,  2.559741,  2.571999, -1.360637],
#  [ 0.285671, -1.890282, -2.674756, -4.402627, -2.909733]]
```

Here $T = 3$. The left-padded tokens, real-token indicators, and per-row positions are:

```python
tokens    = [[1, 3, 4],   real = [[True,  True,  True],   positions = [[ 0,  1, 2],
             [0, 0, 2],           [False, False, True],                [-2, -1, 0],
             [0, 4, 0]]           [False, True,  True]]                [-1,  0, 1]]
```

Row 1 corresponds to the single-token sequence `[2]`. Its only real token attends strictly to itself. If it also attended to the two leading padding tokens, the model would instead compute the logits of `[0, 0, 2]`, which are `[0.136466, -1.268586, 2.554092, 2.141590, -0.205061]`.

Row 2 ends with a real token `0`, which matches `pad_id`. Column 2 is real while column 0 is padding, even though both contain `0`.

## Constraints

- Every row is compared with the kit's oracle run on that sequence alone, to a relative tolerance of `1e-3` and an absolute tolerance of `1e-4`. A NaN is always wrong.
- $1 \le B \le 4$, $1 \le L_b \le 8$, and `n_layers` $\le 4$.
- Compute all rows in one batched pass over the `(B, T)` tensor. Do not loop over the sequences.
- Do not call `decoder_forward_ref` or `decoder_block_ref`.

## Why this matters

Inference servers batch requests together for throughput, and incoming prompts almost never share the same length. Batching must be invisible to the caller, meaning a prompt should produce identical logits whether it runs alone or alongside much longer requests.

When padding leaks into attention or positions, it fails silently. Generations remain fluent, yet outputs vary depending on what other requests happened to share the batch. Because single-request tests never pad, these bugs can slip into production and become very hard to reproduce. Left padding is the standard convention for autoregressive generation because each new token is appended on the right, where every sequence ends.
