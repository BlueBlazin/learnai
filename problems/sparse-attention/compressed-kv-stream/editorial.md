## The idea

Heavily compressed attention replaces every `rate` consecutive tokens of the distant past with a single learned summary. Each summary is a per-channel weighted average of the window's `kv` rows. For every channel independently, a softmax over the `rate` rows combines a token-dependent `gate` with a learned `position_bias` for each offset in the window. RMS normalization then gives every compressed entry a consistent scale regardless of what the window contains.

The main challenge is that windows are defined over the entire stream, whereas `compress` only receives one chunk at a time. Call boundaries do not reset windows. Any trailing rows that do not yet fill a complete window must wait in the state, and the next call must resume that window at the right offset.

A small state solves this cleanly. Store the waiting rows (always fewer than `rate`) and the running entry count. This maintains a simple invariant: **the buffer always starts at a window boundary**. When a new chunk arrives, concatenating the buffer in front of the new rows produces a tensor whose row 0 is at window offset 0. From there, the computation is identical to processing a single contiguous stream. Every full block of `rate` rows becomes a window, and the trailing remainder shorter than `rate` becomes the new buffer, which again starts at a window boundary.

This invariant also explains why `position_bias` cannot be added when rows arrive. A row's offset is its stream position modulo `rate`, not its index within the current call. Reshaping the concatenated full blocks into `(windows, rate, d)` (where `windows` is the number of completed windows) aligns axis 1 with the window offset, so adding `position_bias` of shape `(rate, d)` broadcasts each offset automatically.

**Visibility.** Entry $w$ summarizes token positions $w \cdot \texttt{rate}$ through $(w + 1) \cdot \texttt{rate} - 1$. A query at position $t$ may attend to entry $w$ only if all summarized positions are at or before $t$, which requires:

$$
(w + 1) \cdot \texttt{rate} - 1 \le t \iff w + 1 \le \frac{t + 1}{\texttt{rate}}
$$

Since $w$ is a non-negative integer, the number of valid entries is `(t + 1) // rate`. The query at the last position of a window is part of that window and may see its entry because the entry contains nothing from the future. One position earlier, it cannot see that entry without leaking future tokens.

## Reference solution (PyTorch)

```python
def compress(kv, gate, position_bias, gain, state, rate):
    buf_kv, buf_gate, count = state
    kv, gate = torch.cat((buf_kv, kv)), torch.cat((buf_gate, gate))
    windows = kv.shape[0] // rate
    usable, dim = windows * rate, kv.shape[1]
    win_kv = kv[:usable].reshape(windows, rate, dim)
    win_gate = gate[:usable].reshape(windows, rate, dim) + position_bias
    pooled = (torch.softmax(win_gate, dim=1) * win_kv).sum(dim=1)
    entries = gain * pooled * torch.rsqrt(pooled.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    return entries, (kv[usable:], gate[usable:], count + windows)


def visible_entries(t, rate):
    return (t + 1) // rate
```

When `windows == 0`, every line still executes cleanly on tensors with an empty leading axis, returning `entries` of shape `(0, d)`. The `1e-6` epsilon keeps an all-zero window finite and equal to zero.

## Common mistakes

- **Taking softmax over channels (`dim=-1`).** Standard attention normalizes over the last axis, but here weights compete across the `rate` rows of a window (`dim=1`) independently for each channel.
- **Taking softmax over the entire call.** Without reshaping into `(windows, rate, dim)`, a call that completes multiple windows normalizes across all of them. Tests with only one window per call will not catch this.
- **Omitting `position_bias`, or adding it on arrival.** Counting offsets from the start of each call works only when every call begins on a window boundary.
- **Emitting the unfinished window.** Compressing a partial window at the end of a call creates an entry that the full stream never contains and shifts all later windows.
- **Dropping the remainder.** Discarding leftover rows restarts windows at every call boundary.
- **Concatenating the buffer after the new rows.** Row order inside a window matters because `position_bias` depends on each row's offset.
- **Returning only this call's window count instead of `count + windows`.** Nothing inside `compress` reads `count`, so this bug is easy to miss, but downstream rotary embeddings depend on the cumulative entry count.
- **Using `t // rate` for `visible_entries`.** This hides a window's entry from its own last token even though the entry is already complete.
- **Rounding `visible_entries` up.** This lets a query attend to the entry of a window it is still inside, leaking future tokens.
- **Computing `visible_entries` in float32.** Integers above $2^{24}$ are not exact in float32, so `floor((t + 1) / rate)` can be off by one for large positions. Always use integer division.
- **Normalization bugs.** Omitting `gain`, omitting the `1e-6` epsilon (producing `NaN` on all-zero windows and wrong results on small values), or normalizing rows before pooling instead of after all break the specification.
- **Using a naive `exp(gate)` softmax.** Without subtracting the maximum logit, `exp` overflows float32 near 88 (see `stable-softmax`).

## Complexity

Each call runs in $O((r + n)d)$ time and maintains $O(\texttt{rate} \cdot d)$ state regardless of total stream length. The compressed cache grows by $d$ floats per `rate` tokens, so a million-token context at `rate = 128` takes only 7,812 entries per layer.

## In the wild

Verified on 2026-11-21 against `transformers` 5.23.0 (`models/deepseek_v4/modeling_deepseek_v4.py`) and the `config.json` of `deepseek-ai/DeepSeek-V4-Pro-0813`.

- `DeepseekV4HCACache.store_compression_weights` prepends buffered `kv` and `gate` rows to incoming rows, returns the longest prefix whose length is a multiple of `rate`, and stores the remainder. `update_compressor_states` appends the new entries and increments `entry_count`.
- `DeepseekV4HCACompressor.forward` reshapes rows into windows, adds `position_bias`, applies `softmax(dim=2)` over the window axis (with a leading batch dimension), sums, and applies `kv_norm` (RMSNorm with learned weight and `eps = rms_norm_eps`). It masks entry `w` for query position `t` whenever `w >= (t + 1) // rate`.
- The checkpoint configures `compress_ratios` of 128 on heavily compressed layers and 4 on compressed sparse layers, with `rms_norm_eps = 1e-6`, head width 512, and a sliding window of 128 tokens.
- The public `DeepseekV4HCACompressor` tests run the library module and cache on chunked streams for `rate = 4` and `rate = 128` (head width 16), comparing your `entries`, buffers, and `count` against its outputs, and comparing `visible_entries` against the number of unmasked entries in the returned mask (`fixtures/make.py` regenerates these fixtures).
- Reference paper: *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence* (arXiv:2606.19348). Only the abstract was inspected directly. The library implementation cites Section 2.3.2, equations 20–23.

This problem omits several surrounding parts of the shipped compressor:

- **Entry rotary embeddings.** After normalization, the shipped module rotates the trailing `qk_rope_head_dim` channels of entry `w` using position `w * rate` (computed from `entry_count * rate` for the first entry of a call) with base `compress_rope_theta`. The test fixtures capture entries before this rotation.
- **Input projections.** In the full model, `kv` and `gate` come from `kv_proj(h)` and `gate_proj(h)` applied to hidden states. Here they arrive already projected.
- **Batch axis and mixed precision.** The shipped module includes a batch dimension and upcasts to float32 for the softmax inside a 16-bit model.
- **Cache ownership.** The shipped `forward` appends to the cache and returns all accumulated entries plus the attention mask, whereas `compress` returns only the newly completed entries for the caller to append.
- **Stateless calls.** Without a cache object, the shipped module compresses only complete windows in the current input and discards the remainder.
- **Single-token decode shortcut.** For a single query token, the shipped module skips building a mask because every entry currently in the cache is visible, matching `(t + 1) // rate`.
- **Rest of the attention layer.** Recent tokens stay uncompressed in a sliding window concatenated with the compressed entries, and the attention layer reuses one tensor as both key and value (see `shared-kv-rotation`).
