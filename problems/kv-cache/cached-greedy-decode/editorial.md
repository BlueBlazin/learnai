## The idea

A causal transformer's output at position $i$ depends on tokens $0..i$ only. So when
a token is appended, nothing computed for earlier positions changes: their keys and
values in every attention layer are the same as before. The KV cache stores them, and
a call with a cache computes only the new positions, attending to the cached keys and
values. Appending one token costs one token of model work instead of the whole
prefix.

The decoding loop then has two phases with different shapes:

- **Prefill.** `lm(prompt)` processes all $P$ prompt tokens in one call. The call
  returns logits for every position, but only the last row matters: it scores the
  first new token.
- **Decode.** Each step picks the argmax of the last row, and, if generation goes on,
  feeds that single token with the cache to obtain the logits for the next step.

The invariant that keeps the loop right: *before choosing a token, the cache covers
exactly the sequence so far, and `logits[-1]` was computed from its last token.* Every
bug in this problem is a violation of that sentence.

Counting the cost follows from the loop. Choosing token $k$ needs the model to have
seen the prompt and tokens $1..k-1$. To return $g$ tokens the model must therefore see
$P + g - 1$ tokens, and no token twice. The first token needs one call, and each
later token needs a call after its predecessor is known, so $g$ calls is the
minimum.

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

The stop check sits between choosing a token and feeding it, so the final token (EOS
or the one that reaches the limit) is never fed. `token == eos_id` is simply false
when `eos_id` is `None`. The JAX and MLX versions differ only in the argmax call.

## Common mistakes

- **Re-feeding the whole sequence** (`lm(prompt + generated)` every step). The tokens
  are right, which is why this survives in real code: nothing fails, it is only
  slow. The cost is $\sum_k (P + k - 1)$, quadratic in the length; 2130 tokens instead
  of 65 for 60 new tokens.
- **Dropping the cache** (`lm([token])`). The model then sees a sequence of length
  one. The output is still plausible-looking tokens, just unrelated to the prompt.
  Keeping a stale cache (not rebinding the returned one) fails in a similar way.
- **Feeding the wrong token after prefill.** A common loop shape starts with
  `token = prompt[-1]` and feeds it at the top of the loop. That is correct only if
  the prefill stopped at `prompt[:-1]`; after a full prefill the last prompt token is
  in the context twice.
- **Feeding the prompt token by token.** Same tokens, same number of tokens processed,
  but $P - 1$ extra calls. On a real accelerator one call over $P$ tokens runs in
  parallel, while $P$ calls run in sequence, so this multiplies the time to the first
  token.
- **Feeding the final token.** Checking the stop condition after the forward call
  spends one call on logits nobody reads.
- **EOS handling.** Ignoring `eos_id` generates past the end of the answer; dropping
  the EOS token from the result makes it impossible for the caller to tell "finished"
  from "ran out of budget".

## Complexity

$g$ forward calls and $P + g - 1$ tokens processed. Inside the model, a decode step
at length $L$ still attends over $L$ cached positions, so the attention work is
$O(L)$ per step and $O((P + g)^2)$ over the generation, but with a small constant:
the cache removes the recomputation of every layer's keys, values, and MLP for the
prefix. Memory is $O(L)$ keys and values per layer and head.

## In the wild

Every engine has this loop. Hugging Face `generate` passes `past_key_values` back
into the model and slices the input to the new token; llama.cpp evaluates the prompt
as one batch with `llama_decode` and then one token per call against its KV cache;
vLLM and TGI schedule prefill and decode as distinct phases because their costs
differ so much, and report them separately as time to first token and time per
output token. Production caches are mutable, preallocated buffers instead of the
immutable values `TinyLM` returns, which makes the bookkeeping of "what does the cache
cover" the main source of bugs once batching, beam search, or speculative decoding
(which must roll the cache back after a rejected draft) enter the picture.
