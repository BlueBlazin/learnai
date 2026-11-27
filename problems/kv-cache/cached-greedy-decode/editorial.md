## The idea

In a causal transformer, the output at position $i$ depends only on tokens $0..i$. Appending a new token never changes the attention keys and values of earlier positions. A KV cache stores those earlier keys and values so each new call only computes the new positions and attends over the cached prefix. Appending one token therefore costs one token of model work instead of the entire sequence.

This splits the decoding loop into two phases:

- **Prefill.** `lm(prompt)` processes all $P$ prompt tokens in a single call. The model returns logits for every position, and the last row (`logits[-1]`) scores the first generated token.
- **Decode.** Each step picks the argmax of `logits[-1]`. If generation should continue, the loop passes only that single token along with the latest cache to get the next logits.

The loop stays correct if it maintains one invariant. Before choosing a token, `cache` covers the exact sequence so far, and `logits[-1]` came from its final token.

The cost limits follow directly from this structure. Choosing the $k$-th generated token requires the model to have seen the prompt and the first $k - 1$ generated tokens. Returning $g$ tokens therefore requires $P + g - 1$ processed tokens, with no token passed twice. The first token comes from the prefill call, and each remaining token requires one call after its predecessor is known, giving $g$ calls in total.

## Reference solution (PyTorch)

```python
def greedy_decode(lm, prompt, max_new_tokens, eos_id):
    logits, cache = lm(prompt)
    generated = []
    while True:
        token = int(torch.argmax(logits[-1]))
        generated.append(token)
        if token == eos_id or len(generated) == max_new_tokens:
            return generated
        logits, cache = lm([token], cache)
```

Placing the stop check between choosing a token and calling `lm` ensures that the final token (whether EOS or the `max_new_tokens` limit) is never fed back into the model. When `eos_id` is `None`, `token == eos_id` is always `False`. The JAX and MLX solutions differ only in the argmax call.

## Common mistakes

- **Re-feeding the whole sequence.** Calling `lm(prompt + generated)` on every step produces the right tokens, so it often goes unnoticed in real code. However, it processes $\sum_k (P + k - 1)$ tokens, which is quadratic in length (2,130 tokens instead of 65 for 60 new tokens).
- **Dropping or reusing a stale cache.** Calling `lm([token])` without a cache makes the model see a sequence of length one, producing plausible-looking tokens unrelated to the prompt. Forgetting to rebind `cache` to the newly returned value fails in a similar way.
- **Feeding the wrong token after prefill.** Some loops initialize `token = prompt[-1]` and feed it at the top of the loop. That only works if prefill stopped at `prompt[:-1]`. After a full prefill, it duplicates the last prompt token in the context.
- **Feeding the prompt token by token.** This produces the right tokens and the same `tokens_processed` count, but it makes $P - 1$ extra forward calls. On real hardware, one call over $P$ tokens runs in parallel while $P$ sequential calls multiply time-to-first-token.
- **Feeding the final token.** Checking the stop condition after the forward call wastes one call on logits that are never read.
- **Mishandling EOS.** Ignoring `eos_id` keeps generating past the end of the sequence. Dropping the EOS token from the returned list makes it impossible for the caller to tell whether generation finished naturally or hit `max_new_tokens`.

## Complexity

The loop makes $g$ forward calls and processes $P + g - 1$ tokens. Inside the model, a decode step at sequence length $L$ still attends over all $L$ cached positions, so attention takes $O(L)$ time per step and $O((P + g)^2)$ time overall. The constant factor is much smaller than uncached decoding because the cache avoids recomputing the keys, values, and MLP layers for every earlier token. Memory usage is $O(L)$ keys and values per layer and head.

## In the wild

Every inference engine implements this loop. Hugging Face `generate` passes `past_key_values` back into the model and slices the input to the newest token. llama.cpp evaluates the prompt as one batch with `llama_decode` and then processes one token per call against its KV cache. vLLM and TGI schedule prefill and decode as separate phases because their compute profiles differ, reporting time-to-first-token and time-per-output-token separately.

Production engines use mutable, preallocated buffers instead of the immutable caches `TinyLM` returns. Tracking which tokens a mutable cache covers becomes a major source of bugs once batching, beam search, or speculative decoding (which must roll the cache back after rejected draft tokens) are added.
