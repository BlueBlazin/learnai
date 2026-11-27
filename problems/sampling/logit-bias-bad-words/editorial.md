## The idea

The three controls fall into two distinct categories.

- **Logit bias** is arithmetic (`logits[v] += bias[v]`). It shifts a token's log-probability while keeping the token available.
- **The allow-list and banned sequences** are set membership. A removed token must have probability 0, which is `-inf` in logit space.

Separating these two categories makes the precedence rules automatic. First build the boolean mask of removed tokens from the allow-list and banned sequences, and build the additive bias vector from `bias`. Then add the bias to `logits` and select `-inf` wherever the removal mask is true. Because the bias never modifies the mask, a `+100` bias can never revive a removed token.

A banned sequence describes multi-token text, while the logits processor runs one token at a time. A sequence $(w_0, \ldots, w_{L-1})$ would be completed on the current step only if the last $L - 1$ tokens of `history` equal $(w_0, \ldots, w_{L-2})$ and the model now emits $w_{L-1}$. The processor therefore removes only $w_{L-1}$, and only when that tail matches. Earlier tokens are left untouched so the model can still emit them and continue in a different direction. For a one-token sequence, the prefix is empty. Every history ends with the empty prefix, so the token is removed at every step.

## Reference solution (PyTorch)

```python
def apply_bias(logits, bias, allowed_token_ids, bad_words, history):
    vocab = logits.shape[0]
    delta = [0.0] * vocab
    for token, value in bias.items():
        delta[token] = value

    removed = [allowed_token_ids is not None] * vocab
    if allowed_token_ids is not None:
        for token in allowed_token_ids:
            removed[token] = False

    for word in bad_words:
        prefix = word[:-1]
        start = len(history) - len(prefix)
        if start >= 0 and history[start:] == prefix:
            removed[word[-1]] = True

    biased = logits + torch.tensor(delta, dtype=logits.dtype)
    return torch.where(torch.tensor(removed), float("-inf"), biased)
```

- Because the request metadata is small and irregular, building `delta` and `removed` in Python and touching the tensor with one addition and one `where` keeps the code simple. The JAX and MLX implementations are identical except for the tensor constructor, since nothing mutates a tensor in place.
- Checking `start >= 0` rejects any prefix longer than `history`. When `prefix` is empty, `start == len(history)` and `history[start:]` is `[]`, which matches `prefix`.
- Since `-inf + delta` is still `-inf`, any input logit that was already masked stays masked.

## Common mistakes

- **Banning every token of a banned sequence.** If "New York Times" is banned and `history` ends with "New York", only "Times" should be removed. Removing "New" and "York" restricts valid continuations, and banning them unconditionally deletes common subwords from the vocabulary.
- **Ignoring the history.** Removing the last token of every banned sequence at every step applies the one-token rule to multi-token phrases, making "Times" unavailable everywhere.
- **Matching the prefix out of order.** Comparing sets or sorted lists treats "York New" as matching "New York", even though "Times" would not complete the banned sequence.
- **Matching the prefix anywhere in the history.** Searching for the prefix as a general sublist bans the final token for the rest of generation once the prefix appears anywhere earlier.
- **Slicing with `history[-len(prefix):]`.** For a one-token sequence, `len(prefix)` is `0`, so `history[-0:]` returns the entire history instead of an empty list. The ban then silently stops working as soon as `history` is non-empty.
- **Applying bias after masking via assignment.** Writing `out[token] = logits[token] + bias[token]` onto an already-masked tensor overwrites `-inf` with a finite value. Adding in place preserves `-inf`, whereas assigning from the original `logits` does not.
- **Assigning the bias instead of adding it.** OpenAI's `logit_bias` is defined as an additive offset. Values like `-100` and `+100` act like bans or forced choices only because their magnitudes are large.
- **Letting the allow-list override banned sequences.** Computing the allow-list mask after processing `bad_words` can accidentally un-mask tokens that complete a banned sequence.

## Complexity

Time and memory are $O(V + |\text{bias}| + |\text{allowed}| + \sum_w |w|)$ per step, where $V$ is the vocabulary size. Each banned sequence of length $L$ compares at most $L - 1$ tokens against the tail of `history`, so `history` is never scanned in full. Production servers precompute the history-independent parts (the bias vector, allow-list mask, and one-token bans) once per request and only run the multi-token tail checks on each step.

## In the wild

vLLM implements banned sequences in `NoBadWordsLogitsProcessor` (`vllm/logits_process.py` in the 0.8 series). It folds one-token bans into a constant `-inf` bias vector once per request. For longer sequences, it skips any sequence longer than `len(history) + 1`, compares `past_tokens_ids[-prefix_length:]` with `bad_word_ids[:prefix_length]`, and adds `-inf` to the final token on a match. Its comments credit `NoBadWordsLogitsProcessor` and `SequenceBiasLogitsProcessor` from Hugging Face `transformers`, where banning is a special case of applying a `-inf` sequence bias. The general `SequenceBiasLogitsProcessor` also supports finite biases to promote phrases.

Real implementations differ in two main ways:

- **Strings versus token ids.** vLLM's `bad_words` API accepts strings and tokenizes each one both with and without a leading space, because `"word"` and `" word"` map to different token sequences. Hugging Face `transformers` takes `bad_words_ids` directly and leaves tokenization to the caller. Because token-id bans cannot catch alternative tokenizations of the same text, string-level stop conditions exist as a separate mechanism.
- **Bias range.** OpenAI documents `logit_bias` values in `[-100, 100]`, and vLLM clamps inputs to that range when constructing sampling parameters. Hugging Face's `sequence_bias` accepts any float.
