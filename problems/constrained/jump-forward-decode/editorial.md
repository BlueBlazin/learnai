## The idea

In constrained greedy decoding, each next token is the argmax over the permitted set, which depends only on the constraint state. When the permitted set has a single element, the choice is already determined before the model runs. Every other logit would be masked to $-\infty$, so a forward pass cannot change the outcome. Jump-forward decoding appends that token directly.

Skipping the forward pass does not mean skipping the token in the model's context. Every later free step is conditioned on the entire sequence generated so far, including forced tokens, so those tokens must enter the KV cache before the next free step. Because nothing needs the logits of forced tokens, you do not have to feed them one by one. Instead, buffer them and pass them together in the single forward call required by the next free step. In a causal language model, position $i$ attends only to positions $\le i$. Feeding $k$ tokens in one call therefore produces the same KV cache state as $k$ single-token calls, and its last row of logits matches the distribution the ordinary loop would see.

This yields a simple loop invariant. Maintain a `pending` list holding all tokens the model has not yet seen, in order. Initialize `pending` with `prompt`, append every generated token to it, and clear it whenever you call `lm(pending, cache)` at a free step. Let $P$ be the prompt length, $f$ the number of free steps, and $L$ the number of generated tokens before the last free step. This invariant guarantees three properties:

- The model runs $f$ times, once per free step.
- The model processes $P + L$ tokens (or $0$ if $f = 0$), because anything still in `pending` when the loop ends is never fed.
- The output matches standard constrained greedy decoding token for token, because forced steps pick the only permitted token and free steps take the argmax over identical logits.

Always check whether a step is forced using the full `permitted` set, including `eos_id`. A state with one allowed token that is also accepting gives the model a real choice between continuing and stopping. Conversely, an accepting state with no allowed tokens permits only `eos_id`, making the final stop token forced and eliminating the trailing model call.

## Reference solution (PyTorch)

```python
def jump_forward_decode(lm, prompt, constraint, max_new_tokens, eos_id):
    state = constraint.start()
    generated = []
    cache = None
    pending = list(prompt)
    while len(generated) < max_new_tokens:
        permitted = list(constraint.allowed(state))
        if constraint.is_accepting(state):
            permitted.append(eos_id)
        if not permitted:
            break
        if len(permitted) == 1:
            token = permitted[0]
        else:
            logits, cache = lm(pending, cache)
            pending = []
            keep = [False] * lm.vocab_size
            for candidate in permitted:
                keep[candidate] = True
            masked = torch.where(torch.tensor(keep), logits[-1], float("-inf"))
            token = int(torch.argmax(masked))
        generated.append(token)
        if token == eos_id:
            break
        state = constraint.advance(state, token)
        pending.append(token)
    return generated
```

The JAX and MLX versions differ only in the names of `where` and `argmax`.

## Common mistakes

- **Jumping over forced text instead of forced tokens.** After `{"safe": fal`, the text `se}` is certain, and `{"` is certain at the start. However, when the vocabulary offers both `{` and `{"`, two tokens are permitted and the model's preference between them affects the output. Picking the longest piece or re-encoding the forced string is a different algorithm with a different output, even though some engines choose to do it in practice.
- **Appending forced tokens to the output without adding them to `pending`.** Because the constraint still enforces valid syntax, this bug often escapes casual testing. Every free choice after the first forced run is computed from a corrupted context that omits the forced tokens.
- **Calling the model once per forced token.** Picking the forced token without reading logits and immediately calling `lm([token], cache)` to keep the cache up to date saves only an argmax and violates the forward-call limit.
- **Running a dedicated forward pass for the prompt.** If the first generated token is forced, the prompt's logits are never used. Keep `prompt` in `pending` like any other unseen token.
- **Flushing `pending` after the loop.** Once the last free step is over, no further decisions are made, so the last free token and any trailing forced tokens do not need to be fed. Flushing `pending` at the end wastes a call (this function does not return the cache).
- **Checking `len(allowed)` instead of `len(permitted)`.** Treating `len(allowed) == 1` as forced skips the choice to emit `eos_id` in an accepting state. Requiring `not is_accepting(state)` to jump pays an unnecessary call for a forced `eos_id`.
- **Forgetting `max_new_tokens` inside an inner jump loop.** Forced tokens count toward `max_new_tokens`. If you write an inner `while forced:` loop, it must enforce the same length bound as the outer loop.

## Complexity

For $g$ generated tokens with $f$ free steps, prompt length $P$, and $L$ tokens generated before the last free step, this loop makes $f$ forward calls and processes $P + L$ tokens (compared to $g$ calls and $P + g - 1$ tokens for standard constrained decoding). Constraint overhead is one `allowed`, one `is_accepting`, and at most one `advance` call per generated token.

The speedup comes from reducing forward calls rather than token positions, since every forced token before the last free step still requires one attention position. On a GPU, prefilling a small batch of tokens in one pass takes roughly the same wall-clock time as decoding a single token, so the speedup approaches the fraction of forced steps. For JSON with a fixed schema, that is most of the output.

## In the wild

- **SGLang** introduced the name jump-forward decoding (in "SGLang: Efficient Execution of Structured Language Model Programs", arXiv 2312.07104, and the LMSYS blog post "Fast JSON Decoding for Local LLMs with Compressed Finite State Machine"). SGLang operates at the text level, compressing consecutive singular transitions in the regex automaton into one edge and prefilling that string. Because text-level jumping bypasses token choices, SGLang re-tokenizes the combined text so the model continues from its tokenizer's native segmentation. The LMSYS post notes that alternate tokenizations shift downstream distributions and puts the re-tokenization overhead at about 4 percent.
- **llguidance** (the constraint engine behind Guidance) exposes forced tokens to the host as fast-forward tokens. After a token is committed, the parser returns any subsequent tokens forced by the grammar, and the host appends them in one batch. It enables this only for tokenizers with a canonical tokenization.
- These two designs split exactly on the distinction in the specification note. Token-level forcing (this problem) reproduces masked decoding exactly at the cost of jumping less often, while text-level forcing jumps over every forced string and must resolve tokenization boundaries. A related boundary fix at the start of generation is `token-healing`.
- Under sampling instead of greedy decoding, the logic is identical. A distribution masked down to one token has probability 1 on that token, so a forced step consumes no randomness.
