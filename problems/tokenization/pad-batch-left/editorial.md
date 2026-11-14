## The idea

A decoder-only model predicts token $t + 1$ from the hidden state at position $t$. In
a generation loop that means: run the batch, take the logits of the last column, pick a
token per row, append it as a new column. That only works if the last column of every
row is that row's last real token. Left padding guarantees it for every row at once;
right padding guarantees it only for the longest row.

The attention mask completes the picture. Pad positions are real inputs as far as the
network is concerned: they have embeddings and would be attended to. The mask is what
removes them, and with left padding and a causal model it is all that is needed, since
the real tokens then see exactly their own prefix. (Position handling is the other half
of making padded rows match unpadded ones; with relative schemes such as RoPE the
offset introduced by left padding cancels, with learned absolute positions it must be
corrected. That is the subject of `padded-batch-invariance`.)

The mask must be derived from the lengths. The ids alone cannot say which positions are
padding when `pad_id` is also a legitimate token, and that is the normal situation, not
an exotic one: many open models ship without a pad token and are configured with
`pad_token = eos_token`.

The empty sequence is rejected rather than padded because nothing downstream can use
it: its mask is all zeros, so every attention row is fully masked (a softmax over
nothing, typically NaN), and it has no last token to continue from. Failing at the
boundary gives a clear error instead of NaN logits several layers later. A server
normally cannot receive an empty prompt anyway, because the chat template or the BOS
token contributes at least one token.

## Reference solution (Python)

```python
def pad_batch(seqs, pad_id, side="left"):
    if side not in ("left", "right"):
        raise ValueError(f'side must be "left" or "right", got {side!r}')
    if not seqs:
        raise ValueError("cannot pad an empty batch")
    lengths = [len(seq) for seq in seqs]
    if min(lengths) == 0:
        raise ValueError("every sequence needs at least one token")
    width = max(lengths)
    ids, attention_mask = [], []
    for seq, n in zip(seqs, lengths, strict=True):
        pads, zeros = [pad_id] * (width - n), [0] * (width - n)
        ones = [1] * n
        if side == "left":
            ids.append(pads + list(seq))
            attention_mask.append(zeros + ones)
        else:
            ids.append(list(seq) + pads)
            attention_mask.append(ones + zeros)
    return ids, attention_mask, lengths
```

## Common mistakes

- **Padding on the wrong side.** Swapping the two branches, or defaulting to
  `"right"` because that is what tokenizer libraries default to. The longest row is
  unaffected, so a test with one prompt passes.
- **An inverted mask.** PyTorch's `key_padding_mask` uses `True` for positions to
  ignore; the attention mask of tokenizers and generation APIs uses `1` for positions
  to keep. Mixing the two conventions masks out the prompt and attends to the padding.
- **Mask from `ids != pad_id`.** Correct until a real token equals `pad_id`, which
  happens in every conversation that contains an end-of-turn token when pad is EOS.
- **`lengths` filled with the padded width.** After padding, every row has length `T`;
  the caller wants the number of real tokens, for example to find where each row's
  prompt starts or to compute per-row token counts.
- **Accepting an empty sequence.** `max()` and list multiplication handle it without
  complaint and produce a row of padding.
- **Not validating `side`.** An `if side == "left": ... else: ...` treats `"Left"`,
  `"centre"`, and every typo as right padding.
- **Padding in place.** `seq[:0] = pads` or `seq.extend(pads)` modifies the caller's
  token lists; the next call with the same prompts sees sequences that already contain
  padding, and a retry pads them again.

## Complexity

$O(B \cdot T)$ time and memory, the size of the output.

## In the wild

Hugging Face tokenizers have a `padding_side` attribute that defaults to `"right"`, and
`generate()` on a decoder-only model logs a warning when it detects right-padded input,
telling the user to set `padding_side='left'`. Engines with a paged KV cache, such as
vLLM, avoid padding in the decode loop by tracking each sequence's own cache blocks,
but any code that runs several prompts through the model as one tensor, including
reranking and batched scoring, goes through exactly this function. The cost of padding is real
too: a batch is as wide as its longest row, which is why batchers group prompts of
similar length.
