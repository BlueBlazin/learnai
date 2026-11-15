## The idea

The three controls act on one row of logits, and they are of two kinds.

- A **bias** is arithmetic: `logits[v] += bias[v]`. It shifts a token's
  log-probability and leaves the token in play.
- The **allow-list** and the **banned sequences** are set membership: a token is
  either removed or not. A removed token must have probability exactly 0, which in
  logit space is `-inf`.

Treating them as two kinds gives the combination rule for free. Compute the set of
removed tokens from the allow-list and the banned sequences, compute the biased
logits, and select between `-inf` and the biased logit. No order of operations can
then let a `+100` bias resurrect a removed token, because the bias never touches the
mask.

A banned sequence is a statement about text, and the logits processor sees one step
at a time. The sequence $w_1 \ldots w_L$ would be completed at this step only if the
last $L - 1$ tokens of the history are exactly $w_1 \ldots w_{L-1}$ and the model now
emits $w_L$. So the processor removes $w_L$, and only under that condition. The
earlier tokens are never touched: they are ordinary tokens that happen to start a
banned sequence, and the model may still emit them and continue differently. A
one-token sequence has an empty prefix, which every history ends with, so its token
is removed at every step.

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

- The request data is tiny and irregular, so it is processed in Python; the logits
  are touched by one addition and one select. The JAX and MLX references are the
  same code with a different constructor, which is the point of building the tables
  first: nothing is written into a tensor in place.
- `start >= 0` rejects a prefix longer than the history. `history[start:]` with
  `start == len(history)` is the empty list, which equals the empty prefix.
- `-inf + delta` is `-inf`, so an input logit that was already masked stays masked.

## Common mistakes

- **Banning every token of a matching sequence.** If "New York Times" is banned and
  the history ends with "New York", only "Times" must go. Removing "New" and "York"
  as well changes what the model can say next for no reason; done unconditionally,
  it deletes common word pieces from the vocabulary.
- **Ignoring the history.** Removing the last token of every banned sequence at every
  step is the one-token rule applied to all sequences. The word piece "Times" is then
  never available.
- **Matching the prefix out of order** (comparing sets, or sorted lists). After
  "York New" the token "Times" is not completing anything.
- **Matching the prefix anywhere in the history.** The condition is about the *end*
  of the history. A substring search bans the last token for the rest of the
  generation once the prefix has occurred.
- **`history[-len(prefix):]`.** For a one-token sequence this is `history[-0:]`, the
  whole history, which equals the empty prefix only when the history is empty: the
  ban silently stops working after the first token.
- **Bias after the mask.** `out[token] = logits[token] + bias[token]` on the masked
  tensor writes a finite number over `-inf`. Adding in place would have been safe;
  assigning is not.
- **Assigning the bias** instead of adding it. OpenAI's parameter is documented as a
  value added to the logit; `-100` and `+100` behave like a ban and a forced choice
  only because the numbers are large.
- **Letting the allow-list protect a token.** Recomputing the mask from the
  allow-list after the banned sequences drops their removals.

## Complexity

$O(V + |\text{bias}| + |\text{allowed}| + \sum_w |w|)$ per step: each banned sequence
costs one comparison of at most $L - 1$ tokens against the tail of the history, and
the history is never scanned. A server precomputes what does not depend on the
history (the bias vector, the allow mask, the one-token bans) once per request and
does only the multi-token comparison per step.

## In the wild

vLLM implements the banned sequences as `NoBadWordsLogitsProcessor`
(`vllm/logits_process.py` in the 0.8 series). It handles one-token sequences once,
as a constant `-inf` bias vector, and for each longer sequence compares
`past_tokens_ids[-prefix_length:]` with `bad_word_ids[:prefix_length]` after skipping
sequences longer than the history plus one; on a match it adds `-inf` to the last
token. Its own comments say the code is based on `NoBadWordsLogitsProcessor` and
`SequenceBiasLogitsProcessor` of Hugging Face `transformers`, where banning is the
special case of a sequence bias of `-inf`: the general form adds a finite bias to the
last token of a matching sequence, which is how `transformers` can also *promote* a
phrase. This problem follows those semantics.

Where implementations differ:

- **Strings versus token ids.** vLLM's `bad_words` request parameter takes strings
  and tokenizes each one with and without a leading space, since "word" and " word"
  are different token sequences; `transformers` takes `bad_words_ids` and leaves that
  to the caller. A ban on token ids cannot see the same text spelled with different
  tokens, which is why string-level stop conditions exist as a separate mechanism.
- **Range of the bias.** OpenAI documents `logit_bias` values from `-100` to `100`;
  vLLM clamps each value to that range when it builds the sampling parameters, while
  `transformers`' `sequence_bias` accepts any float.
