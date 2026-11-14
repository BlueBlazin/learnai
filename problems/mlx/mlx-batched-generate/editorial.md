## The idea

A transformer row is a function of three things: the tokens, which keys each query
may attend to, and the positions. Padding changes none of the real tokens, so a
padded row equals its solo run if the other two are restored:

- **Mask.** No real query attends to a padding key. Then padding contributes nothing
  to any real token's attention, in any layer, whatever garbage the padding slots
  compute.
- **Positions.** Real tokens are numbered from 0.

Left padding is what makes the rest easy. The last real token of every row is in the
last column, so the next token is `logits[:, -1]` for all rows. And every generated
token is appended after the real tokens of its row, so a row's real tokens stay
contiguous for the whole generation; with right padding the padding would end up in
the middle of the sequence.

Three details decide whether it actually works.

**The mask must follow the row into the cache.** During decoding there is one query
per row, attending to all cached columns. The first $P_b$ of those are the row's
padding keys, still in the cache, and they must stay hidden at every step. The decode
mask has shape `(B, 1, 1, C + 1)`: `-inf` on the padding columns, `0` elsewhere, one
more `0` per step. Passing no mask on the decode steps is the most natural bug in
this problem, because a single-row decode step needs none.

**A fully masked row is NaN, and NaN gets through an additive mask.** A padding query
with nothing to attend to has scores of all `-inf`; the softmax is `0 / 0`. Nobody
reads that row's output, but in the next layer it becomes that slot's *key*, and a
real query's score against it is `NaN`. The mask then adds `-inf`, and
`NaN + (-inf)` is `NaN`: the padding key is no longer hidden, and the whole softmax
row is NaN. With one layer nothing goes wrong, which makes this easy to miss; with
two, every padded row of the batch is destroyed. Letting each padding query attend
to itself (the diagonal) keeps it finite. A boolean mask applied with `mx.where`
would also survive, because it *replaces* the score; an additive mask cannot.

**Positions: what RoPE forgives and absolute positions do not.** RoPE scores depend
only on the difference between the query's and the key's position. Shifting a whole
row by $P_b$ (using the column number as the position, in the prompt call *and* in
the decode steps) changes nothing, and such a solution passes every RoPE test. It
fails with absolute positions, where a row starting at position 3 is a different
input, and it fails with RoPE as soon as the shift is inconsistent: positions
$j - P_b$ for the prompt and $T + s$ for the decode steps put a gap of $P_b$ between
a short row's prompt and its continuation. Numbering each row from 0 is right for
both.

## Reference solution (MLX)

```python
def batch_generate(model, prompts, max_new_tokens, eos_id):
    batch = len(prompts)
    lengths = [len(prompt) for prompt in prompts]
    width = max(lengths)
    pad = mx.array([width - n for n in lengths])

    tokens = mx.array([[0] * (width - len(p)) + list(p) for p in prompts])
    cols = mx.arange(width)
    real = cols[None, :] >= pad[:, None]  # (B, T)

    causal = cols[None, :] <= cols[:, None]
    allowed = (causal[None] & real[:, None, :]) | mx.eye(width, dtype=mx.bool_)[None]
    mask = mx.where(allowed, 0.0, -mx.inf)[:, None]  # (B, 1, T, T)
    positions = mx.maximum(cols[None, :] - pad[:, None], 0)
    logits, cache = model(tokens, mask=mask, positions=positions)

    key_mask = mx.where(real, 0.0, -mx.inf)[:, None, None, :]  # (B, 1, 1, T)
    visible = mx.zeros((batch, 1, 1, 1))
    row_lengths = mx.array(lengths)

    out = [[] for _ in prompts]
    done = [False] * batch
    for step in range(max_new_tokens):
        token = mx.argmax(logits[:, -1, :], axis=-1)  # (B,)
        mx.eval(token, cache)
        for row, value in enumerate(token.tolist()):
            if not done[row]:
                out[row].append(value)
                done[row] = value == eos_id or len(out[row]) == max_new_tokens
        if all(done):
            break
        key_mask = mx.concatenate([key_mask, visible], axis=-1)
        logits, cache = model(
            token[:, None], cache, mask=key_mask, positions=(row_lengths + step)[:, None]
        )
    return out
```

The bookkeeping of finished rows lives in Python, on values read once per step.
`token.tolist()` after `mx.eval(token, cache)` is one transfer of `B` integers. A
finished row keeps its slot and is fed its own argmax: its cache keeps growing, its
logits keep being computed, and nothing is appended. Removing the row instead would
save that work and change the batch dimension of every cached array; real servers do
it at the level of the scheduler, not inside the loop.

The loop tests `all(done)` before building the next call. Since every row finishes
at `max_new_tokens` at the latest, the loop always ends through that `break`, and the
model is called exactly $g_{\max}$ times.

## What batching buys

Measured on this machine (M1 Max, MLX 0.32.4) on an `MLXTinyLM` with 6 layers of
width 256 and a vocabulary of 8192, prompts of 4 to 39 tokens, 64 new tokens per row:

| | 8 prompts | 32 prompts |
|---|---|---|
| one batch, one read per step (the reference) | 0.20 s | 0.16 s |
| one batch, `.item()` per row | 0.29 s | 0.61 s |
| one prompt at a time | 1.08 s | 4.32 s |

A model this small is far from filling the GPU, so 32 rows cost the same as 8 and
the batch is 27 times faster than the sequential loop. The middle row is the same
batched computation with `B` reads per step: each `.item()` is a separate round trip
to the device, and at 32 rows the round trips cost more than the model.

## Common mistakes

- **No padding mask.** A causal mask alone lets every short row attend to its
  padding. All of its tokens change.
- **Padding on the right** and reading `logits[:, -1]`. For a short row that column
  is a padding slot.
- **A fully masked padding query.** NaN through the second layer, as above.
- **The mask dropped on the decode steps.** The first token of every row is right;
  from the second on, short rows see their cached padding.
- **Positions from the padded layout.** `arange(T)` for every row is invisible to
  RoPE and wrong for absolute positions. A shared `T + step` during decoding is wrong
  for both, from the second token on.
- **Padding found by token value** (`tokens != pad_id`). A prompt that contains the
  padding id loses real tokens. Use the lengths.
- **Finished rows that keep collecting**, or an EOS that is left out of the row.
- **Stopping when the first row finishes** (`any(done)`), which truncates the rest.
- **One more call after the last token.** Checking `all(done)` after building the
  next step costs a full batched forward pass per generation.
- **`.item()` per row**, or a loop over prompts: right tokens, `B` times the reads
  or the model calls.

## Complexity

$g_{\max}$ model calls on $B \cdot T + B \cdot (g_{\max} - 1)$ tokens, one evaluation
and one read of `B` integers per step. The cache holds $T + g_{\max} - 1$ columns for
every row, including the padding columns of short rows and the filler of finished
ones.

## In the wild

`mlx-lm`'s `BatchKVCache` stores `left_padding` per row, builds the mask
`(query >= key) & (key >= left_padding)` from it on every call, and gives each row
the RoPE offset `-left_padding`, so padding slots get negative positions and real
tokens start at 0; `mx.fast.rope` accepts a vector of per-row offsets for exactly
this. Its mask is boolean, which is how it survives the fully masked rows. Hugging
Face `transformers` asks for `padding_side="left"` for generation for the reason in
the first section, and un-masks the fully masked rows of an additive mask before
attention, for the same reason as the diagonal here.
