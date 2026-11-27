## The idea

At every step, a language model scores every token in the vocabulary, while the constraint partitions the vocabulary into tokens that keep the output valid and tokens that do not. Constrained decoding masks out the invalid tokens *before* selecting the next token, so greedy decoding always picks the highest-scoring valid continuation. By induction, every prefix stays valid, and because `eos_id` is permitted only in accepting states, any sequence ending with `eos_id` is a complete member of the target language.

The loop maintains two pieces of state that must describe the exact same prefix: the KV cache (what the model has processed) and the constraint state (where the generated output sits in the grammar). Each generated token updates both. The constraint advances immediately via `advance`, while the model receives the token lazily on the next step when new logits are needed.

Because `eos_id` has no text, `constraint.allowed(state)` never includes it. Adding `eos_id` to the permitted set whenever `constraint.is_accepting(state)` is `True` handles all three stopping cases cleanly. It blocks `eos_id` mid-structure, lets `eos_id` compete with continuation tokens when a complete output can still grow (such as `12` vs. `123`), and forces `eos_id` when the output is complete and no extension is legal.

The remaining case, where `allowed(state)` is empty and `is_accepting(state)` is `False`, cannot happen for a well-formed grammar over a complete vocabulary. It can still occur in practice when a vocabulary lacks a required character or a grammar has a dead state. Calling `argmax` on an all-$-\infty$ row would silently emit token `0` forever, so breaking out of the loop is the right behavior.

## Reference solution (PyTorch)

```python
def constrained_greedy(lm, prompt, constraint, max_new_tokens, eos_id):
    state = constraint.start()
    generated = []
    cache = None
    pending = prompt
    while len(generated) < max_new_tokens:
        permitted = list(constraint.allowed(state))
        if constraint.is_accepting(state):
            permitted.append(eos_id)
        if not permitted:
            break
        logits, cache = lm(pending, cache)
        keep = [False] * lm.vocab_size
        for token in permitted:
            keep[token] = True
        masked = torch.where(torch.tensor(keep), logits[-1], float("-inf"))
        token = int(torch.argmax(masked))
        generated.append(token)
        if token == eos_id:
            break
        state = constraint.advance(state, token)
        pending = [token]
    return generated
```

The order of operations inside the loop guarantees both cost limits. Because the permitted set depends only on `state`, checking for an empty set before calling `lm` ensures a dead end costs zero forward calls. Deferring each new token in `pending` until the next iteration ensures the final token is never fed back to `lm`, regardless of why the loop ended. Producing $g$ tokens from a prompt of length $P$ therefore takes $g$ forward calls and $P + g - 1$ processed tokens (or $0$ when $g = 0$), matching unconstrained cached decoding. The JAX and MLX implementations differ only in their `where` and `argmax` calls. Building a fixed-shape boolean mask of length `vocab_size` also prevents JAX from recompiling for every distinct length of `permitted`.

## Common mistakes

- **Not advancing the state.** The first token is valid, and every later step masks as if it were still at the start, producing outputs like `[[[[[`. Discarding the return value of `advance` or advancing only on some branches fails the same way.
- **Always permitting EOS.** Leaving `eos_id` unmasked (or adding it unconditionally) lets the model stop early, producing truncated JSON that was supposed to be valid.
- **Never permitting EOS, or permitting it only when nothing else is allowed.** Masking with `allowed` alone prevents the model from ending any extensible output (such as a number or string), forcing it to run to `max_new_tokens`. Permitting `eos_id` only when `allowed` is empty has the same bug whenever a complete output can still be extended.
- **Multiplying logits by a 0/1 mask.** `logits * mask` is a probability-space operation applied to logits. It sets forbidden tokens to `0.0`, which beats any permitted token whose logit is negative. Logit masking must set forbidden entries to $-\infty$.
- **Validating after the forward pass.** Feeding the unconstrained argmax into the model and checking validity afterward wastes a forward call on every rejected token (173 calls instead of 9 in the worked example). Checking candidates *before* feeding them to the model avoids extra model calls.
- **Calling the model before the dead-end check.** This wastes a forward call on every dead end, including a full prompt prefill when nothing is permitted at the start.
- **Calling `advance` on EOS.** `eos_id` is not a grammar token, so `constraint.advance(state, eos_id)` raises `ValueError`. Break on `token == eos_id` before advancing.

## Complexity

Let $P$ be the prompt length, $g$ be the number of generated tokens, and $V$ be `lm.vocab_size`. The loop makes $g$ forward calls and processes $P + g - 1$ tokens (or $0$ if $g = 0$), along with one `allowed` call and at most one `advance` call per step. Constructing the dense mask takes $O(V)$ time per step. For real vocabularies with 100,000+ tokens, computing `allowed` dominates the overhead, which is why serving engines precompute allowed token sets per automaton state (`dfa-token-index`) and store masks as bitsets.

## In the wild

- **Outlines** (Willard and Louf, 2023) compiles a regex or JSON Schema into a finite automaton and precomputes the allowed token set for every state. Each decode step reduces to a table lookup and a logit mask, just like `DFAConstraint` here.
- **vLLM** supports multiple structured-output backends (XGrammar, Guidance, Outlines, and lm-format-enforcer). Each backend produces a per-step token bitmask that the engine applies to the logits before sampling.
- **llama.cpp** optimizes the common case by sampling first from the unconstrained distribution, checking that single token against the grammar, and applying the full vocabulary mask only if the token is rejected. Because the logits are already computed, no extra model forward pass is needed.
- **Local vs. global optimality.** Constrained greedy decoding picks the highest-scoring valid *next* token, not the highest-probability valid string overall. A tight mask can force the model down a low-probability path, which engines accept in exchange for syntactic validity. The `max_tokens` limit remains outside the guarantee, so a truncated response can still be invalid JSON.
- **Jump-forward decoding.** Steps where only a single token is permitted do not need logits from the model. Skipping those forward calls and feeding the forced tokens together as one chunk is the optimization implemented in `jump-forward-decode`.
