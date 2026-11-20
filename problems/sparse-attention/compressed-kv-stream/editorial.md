## The idea

Heavily compressed attention replaces every `rate` consecutive tokens of the distant
past by one learned summary. The summary is a weighted average of the window's rows
in which every channel chooses its own weights: a softmax over the window, per
channel, of a gate that depends on the token (`gate`) plus a learned preference for
each offset in the window (`position_bias`). The average is then normalized, so every
entry has the same scale whatever the window contained.

The arithmetic is three lines. The problem is that the definition is about the
*stream* and the function is called on *pieces*:

- A window is a fixed block of row numbers. A call boundary means nothing to it.
- So rows that arrive before their window is complete must wait, and the next call
  has to continue the window where it stood, with the next row at the right offset.

The state that makes this work is small: the waiting rows (fewer than `rate`), and how
many entries exist. The invariant is that **the buffer always starts at a window
boundary**. Given that, concatenating the buffer in front of the new rows produces a
tensor whose row 0 is at offset 0, and the rest is the single-call computation:
blocks of `rate` rows are windows, and the tail shorter than `rate` is the new buffer,
which again starts at a window boundary.

That invariant is also why the bias cannot be added when a row arrives. A row's offset
is its position in the stream modulo `rate`, not its position in the call. Adding the
bias after the reshape gets the offset from the layout instead of from arithmetic on
indices.

**Visibility.** Entry $w$ summarizes positions up to $(w + 1) \cdot \texttt{rate} - 1$.
A query at position $t$ may use it only if all of those are at or before $t$, which is
$w < (t + 1) \mathbin{//} \texttt{rate}$. The query at the last position of a window is
itself part of the summary and may see it: an entry contains nothing later than its
last row. One position earlier it may not, or it would read a summary of its own
future.

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

With `windows = 0` every line still runs, on tensors with an empty first axis, and
`entries` has shape `(0, d)`. The epsilon keeps an all-zero window at zero.

## Common mistakes

- **Softmax over channels.** `dim=-1` is the habit from attention. Here the weights
  compete across the rows of a window, and each channel's weights sum to 1 over them.
- **Softmax over the whole call.** Without the reshape, a call that completes several
  windows normalizes across all of them. Calls of one window at a time hide it.
- **No position bias,** or **the bias added on arrival** with offsets counted from
  the first row of the call. The second is right exactly when every call starts at a
  window boundary.
- **Emitting the unfinished window.** A partial window compressed at the end of a
  call produces an entry that the complete stream never contains, and shifts every
  later window.
- **Dropping the remainder.** Windows then restart at every call.
- **The buffer after the new rows.** The order inside a window matters because of
  the bias.
- **The count of this call instead of the total.** Nothing in this function reads
  the count, which is why it goes wrong unnoticed. The shipped model derives each
  entry's position from it.
- **Threshold `t // rate`.** The last token of a window cannot see its own window's
  entry, although the entry is complete and the cache contains it.
- **Threshold rounded up.** A token sees the entry of the window it is still inside.
- **The threshold in floating point.** Positions above $2^{24}$ are not exact in
  float32, so `floor((t + 1) / rate)` computed there is off by one for a million-token
  context times a small rate. Use integer arithmetic.
- **The norm:** no gain, no epsilon (NaN for a zero window, and wrong for small
  entries), or the rows normalized before pooling.
- **A naive softmax.** `exp(gate)` overflows float32 near 88; see `stable-softmax`.

## Complexity

$O(n\,d)$ time per call and $O(\texttt{rate} \cdot d)$ state, whatever the length of
the stream. The cache grows by $d$ numbers per `rate` tokens: at a rate of 128, a
million tokens are 7812 entries per layer.

## In the wild

Checked on 2026-11-21 in `transformers` 5.23.0,
`models/deepseek_v4/modeling_deepseek_v4.py`, and in the `config.json` of
`deepseek-ai/DeepSeek-V4-Pro-0813`.

- `DeepseekV4HCACache.store_compression_weights` concatenates the buffered `kv` and
  `gate` in front of the new ones, returns the longest prefix that is a multiple of
  the rate, and keeps the rest; `update_compressor_states` appends the new entries and
  advances `entry_count`. `DeepseekV4HCACompressor.forward` reshapes to windows, adds
  `position_bias`, takes `softmax(dim=2)` (the window axis, there with a batch axis in
  front), sums, and applies `kv_norm`, an RMS norm with a weight and
  `eps = rms_norm_eps`. It masks entry `w` for query position `t` when
  `w >= (t + 1) // rate`.
- The checkpoint's configuration has `compress_ratios` of 128 on its heavily
  compressed layers and 4 on its compressed sparse layers, `rms_norm_eps` 1e-6, a head
  width of 512, and a sliding window of 128.
- The public tests "DeepseekV4HCACompressor" run that compressor with its cache on a
  stream in pieces, for a window of 4 and for the shipped window of 128 (head width
  16), and compare your entries, your buffers, and your count with its own.
  `visible_entries` is compared with the number of unmasked entries in the mask the
  module returns. `fixtures/make.py` regenerates them.
- "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence"
  (arXiv 2606.19348). Only the abstract was read; the code cites its section 2.3.2,
  equations 20 to 23, for this computation.

What this problem leaves out of the shipped compressor, completely:

- **The rotary embedding of each entry.** After the norm, the shipped code rotates
  the trailing `qk_rope_head_dim` channels of entry `w` by the angles of position
  `w * rate` (`entry_count * rate` for the first entry of a call, which is what the
  count is for), with base `compress_rope_theta`. The fixture holds the entries as
  they are before that rotation.
- **The projections.** `kv` and `gate` are `kv_proj(h)` and `gate_proj(h)` of the
  hidden states; here they arrive projected.
- **The batch axis,** and the cast: the shipped softmax is computed in float32 inside
  a model that otherwise runs in a 16-bit format.
- **What is returned.** The shipped forward returns every entry emitted so far (it
  appends to the cache) together with the mask; this function returns the entries
  of the call, and the caller appends them.
- **The call without a cache.** With no cache the shipped module compresses the
  complete windows of its input and discards the remainder.
- **The single-token shortcut.** For one query the shipped module returns no mask:
  every entry in the cache is visible, which agrees with `(t + 1) // rate` because the
  cache then holds exactly that many.
- **The rest of the layer:** the recent tokens kept uncompressed in a sliding window
  and concatenated with the entries, and the attention itself, in which one tensor
  serves as both key and value (`shared-kv-rotation`).
