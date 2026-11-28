## The idea

A transformer row's output depends on three things: its token sequence, which keys each query may attend to, and its token positions. Padding leaves the real tokens intact, so a padded row matches its solo run if you preserve the other two properties:

- **Attention mask:** Prevent every real query from attending to any padding key. Padding then contributes nothing to any real token's representations in any layer.
- **Positions:** Number every row's real tokens starting from 0.

Left padding simplifies the rest of the loop. Every row's last real token lands in the final column, so `logits[:, -1]` always scores the next token for all rows. Every newly generated token is then appended directly after the real tokens of its row, keeping all real tokens contiguous throughout generation. With right padding, the padding would sit between the prompt and the generated tokens.

Three details determine whether the implementation works across all test cases:

**Carry the padding mask into every decode step.** During decoding, each row has one query attending to all cached columns plus the new column. The first $P_b$ cached columns are still row $b$'s padding keys and must stay masked at every step. The decode mask has shape `(B, 1, 1, C + 1)` with `-inf` on the padding columns and `0` elsewhere, appending one `0` column per step. Dropping the mask during decode is a common bug because single-row decoding needs no mask.

**Prevent fully masked padding queries from producing NaN.** A padding query whose mask row is all `-inf` produces `0 / 0 = NaN` in softmax. Even though you discard that query's logits, its NaN hidden state becomes a key in the next layer. When a real query computes its attention score against that NaN key, `NaN + (-inf)` evaluates to `NaN`, so the additive mask fails to hide it and the entire softmax row turns into NaN. Single-layer models hide this bug, while two-layer models break on every padded row. Allowing each padding query to attend to itself (setting the diagonal to `0`) keeps its softmax finite. A boolean mask applied with `mx.where` avoids this issue because it replaces scores instead of adding to them, whereas an additive mask cannot.

**Number positions from 0 for both RoPE and absolute embeddings.** RoPE attention scores depend only on relative position differences between queries and keys. Shifting an entire row by $P_b$ (using raw column indices for both the prompt and decode steps) leaves RoPE outputs unchanged and passes RoPE-only tests. However, it fails on absolute sinusoidal embeddings, where starting a sequence at position 3 changes the input embeddings. It also fails on RoPE if the prompt uses shifted positions $j - P_b$ while decoding uses unshifted positions `T + step`, which introduces a gap of $P_b$ between a short prompt and its continuation. Starting each row at position 0 works for both.

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

Finished-row bookkeeping stays in Python using values read once per step. Calling `token.tolist()` after `mx.eval(token, cache)` transfers all `B` integers in a single host read. A finished row keeps its slot in the batch and receives its own argmax as filler while nothing further is appended to its output list. Compacting finished rows out of the batch would change the batch dimension of every cached tensor, so production servers handle compaction in an outer scheduler rather than inside the inner decode loop.

The loop checks `all(done)` before launching the next forward pass. Because every row finishes by `max_new_tokens` at the latest, the loop always exits through that `break`, calling the model exactly $g_{\max}$ times.

## What batching buys

Measured on an M1 Max (MLX 0.32.4) with a 6-layer `MLXTinyLM` (width 256, vocabulary 8192), prompt lengths from 4 to 39, and 64 generated tokens per row:

| approach | 8 prompts | 32 prompts |
|---|---|---|
| one batch, one read per step (reference) | 0.20 s | 0.16 s |
| one batch, `.item()` per row | 0.29 s | 0.61 s |
| one prompt at a time | 1.08 s | 4.32 s |

A small model underutilizes the GPU, so 32 rows run in roughly the same time as 8 and beat sequential decoding by 27x. In the middle row, the GPU work is identical to the reference, but calling `.item()` per row issues `B` separate device-to-host round trips per step. At 32 rows, those round trips cost more than the transformer itself.

## Common mistakes

- **Omitting the padding mask.** Using only a causal mask allows short rows to attend to their padding slots, changing all of their generated tokens.
- **Right-padding while reading `logits[:, -1]`.** For any shorter row, the last column is a padding slot rather than the last prompt token.
- **Leaving padding queries fully masked.** All-`-inf` rows produce NaN in softmax, which propagates through keys in the next layer and corrupts the batch.
- **Dropping the mask on decode steps.** The first generated token is right, but every subsequent token of a short row attends to cached padding keys and drifts.
- **Using padded column indices as positions.** Passing `arange(T)` for every row works under RoPE if decode steps also shift, but fails on absolute sinusoidal positions. Passing a shared `T + step` during decoding breaks both RoPE and absolute positions starting at the second token.
- **Detecting padding by token value (`tokens != pad_id`).** If a real prompt contains `pad_id`, those real tokens get masked out. Always derive masks from prompt lengths.
- **Appending to finished rows or excluding EOS.** Finished rows must stop collecting tokens immediately after recording `eos_id`.
- **Stopping when the first row finishes (`any(done)`).** This truncates every longer row prematurely.
- **Running one extra forward pass after the last token.** Checking `all(done)` after issuing the next model call wastes a full batched forward pass.
- **Calling `.item()` per row or looping over prompts.** Both produce the right tokens but exceed the read or forward-call limits by a factor of `B`.

## Complexity

The loop makes $g_{\max}$ model calls processing $B \cdot T + B \cdot (g_{\max} - 1)$ tokens in total, with one evaluation and one device read of `B` integers per step. The KV cache stores $T + g_{\max} - 1$ columns per row, including padding slots for short rows and filler steps for early-finishing rows.

## In the wild

`mlx-lm`'s `BatchKVCache` tracks `left_padding` per row, constructs the mask `(query >= key) & (key >= left_padding)` on each call, and passes per-row RoPE offsets of `-left_padding` via `mx.fast.rope` so real tokens start at position 0. It uses a boolean mask, which avoids NaN propagation from fully masked rows. Hugging Face `transformers` similarly requires `padding_side="left"` for batched generation and unmasks fully masked rows before applying an additive attention mask.
