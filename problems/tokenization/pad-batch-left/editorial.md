## The idea

A decoder-only model predicts token $t + 1$ from the hidden state at position $t$. During batched generation, each step runs the batch through the model, takes the logits from the last column, samples one token per row, and appends that token as a new column. For this to work, the last column of every row must hold that row's final real token. Left padding guarantees this for every row, while right padding guarantees it only for the longest row.

The attention mask completes the setup. To the network, pad tokens still have embeddings and would be attended to if left unmasked. Masking them out lets every real token in a left-padded causal batch attend only to its own prefix. (Position indices are the other half of making padded rows match unpadded ones. Relative schemes like RoPE cancel the offset introduced by left padding, while learned absolute positions must be shifted explicitly. That is covered in `padded-batch-invariance`.)

The mask must come from the sequence lengths rather than from comparing token values against `pad_id`. Many open models ship without a dedicated pad token and set `pad_token = eos_token`, so `pad_id` regularly appears as a real token inside a prompt.

Empty sequences are rejected up front instead of padded. A row of all padding has an all-zero attention mask, which forces softmax over zero valid keys and typically produces NaN logits several layers later. Rejecting empty sequences at the boundary gives a clear error immediately. In practice, a server rarely receives an empty prompt anyway because the chat template or BOS token adds at least one token.

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

- **Padding on the wrong side.** Swapping the `"left"` and `"right"` branches, or defaulting to `"right"` like many tokenizer libraries, breaks shorter rows. The longest row is unaffected, so single-prompt tests still pass.
- **Inverting the mask.** PyTorch's `key_padding_mask` uses `True` for positions to ignore, whereas tokenizer and generation APIs use `1` for positions to keep. Mixing the two masks out the prompt and attends to the padding.
- **Deriving the mask from `ids != pad_id`.** This works until a real token equals `pad_id`, which happens in any conversation containing an end-of-turn token when pad is EOS.
- **Filling `lengths` with the padded width.** Every padded row has the same padded width, but callers need the original unpadded lengths to locate where each prompt starts or to track per-row token counts.
- **Accepting an empty sequence.** `max()` and list repetition silently turn an empty list into a row of pure padding.
- **Skipping validation on `side`.** Writing `if side == "left": ... else: ...` silently treats `"Left"`, `"center"`, and any other typo as right padding.
- **Padding in place.** Mutating the caller's lists with `seq[:0] = pads` or `seq.extend(pads)` corrupts the input sequences, so any reuse or retry sees already-padded data.

## Complexity

$O(B \cdot T)$ time and memory, where $B$ is the number of sequences and $T$ is the maximum sequence length. This matches the size of the output.

## In the wild

Hugging Face tokenizers expose a `padding_side` attribute that defaults to `"right"`. Calling `generate()` on a decoder-only model logs a warning if it detects right-padded input and tells you to set `padding_side='left'`.

Serving engines with a paged KV cache, such as vLLM, avoid padding during decoding by tracking each sequence's cache blocks independently. Still, any code that runs multiple prompts through a model as one dense tensor (including batched scoring and reranking) uses this exact padding logic. Because a batch is always as wide as its longest row, batching layers typically group prompts of similar length to minimize wasted compute.
