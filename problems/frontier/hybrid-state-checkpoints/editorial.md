## The idea

An attention layer stores a prefix as one key and one value per token. Rolling back to a shorter prefix simply drops trailing entries, so any cached sequence serves all of its prefixes, and prefix reuse reduces to finding the longest common prefix, truncating, and continuing.

A recurrent layer (such as a state-space or linear-attention layer) folds each token $x_t$ into a fixed-size state:

$$
s_t = f(s_{t-1}, x_t).
$$

The state $s_t$ has the same size as $s_1$, and its entries cannot be separated by token. Because $f$ has no inverse, a saved state represents only the exact prefix it has read. Reusing a prefix requires a copy of the state saved at that exact boundary. Since you do not know in advance where future prompts will diverge, you save checkpoints at regular intervals and at natural boundaries.

This leads to three design decisions.

**Which checkpoint to resume.** Choose the longest stored key that is a proper prefix of the prompt. Looking for the entry that shares the most tokens is wrong. A stored sequence that shares six tokens and then diverges cannot be used at all, whereas a stored four-token prefix is directly reusable. The prefix must be *proper* because generating the first token requires logits, and `lm` only produces logits when fed at least one token.

**Where to save checkpoints.** Save at multiples of `interval` counted from the start of the sequence, and also at the end of the prompt. Counting from index 0 ensures that different requests align on the same checkpoint positions regardless of where they resumed. Saving at the end of the prompt covers the common case where a follow-up turn continues immediately from that prompt. During decoding, every step is already a single-token call, so saving a state costs nothing extra. During prefill, a state is only exposed at the end of a call, so the remaining prompt must be fed in segments ending at each target position. With interval $n$, any prompt sharing $m$ tokens with a stored sequence resumes at $\lfloor m / n \rfloor \cdot n$ or later and recomputes fewer than $n$ shared tokens.

**What to keep.** Each checkpoint stores a full copy of the recurrent state. Under a fixed capacity budget, the most valuable checkpoints are the ones requests keep revisiting, such as a shared system prompt. Ordering the store by recency handles this cleanly. Resuming refreshes a key, saving appends a key, and eviction drops from the oldest end. A Python `dict` maintains insertion order and is sufficient.

Because the model folds tokens in one at a time using identical arithmetic, resuming from a checkpoint matches a fresh run bit for bit, so the tests check both tokens and states for exact equality.

## Reference solution (PyTorch)

```python
def generate_with_checkpoints(lm, store, prompt, max_new_tokens, interval, max_checkpoints):
    new_store = dict(store)  # same order: least recently used first

    def save(sequence, state):
        key = tuple(sequence)
        new_store.pop(key, None)
        new_store[key] = state

    best = None
    for key in store:
        usable = len(key) < len(prompt) and tuple(prompt[: len(key)]) == key
        if usable and (best is None or len(key) > len(best)):
            best = key
    state, done = None, 0
    if best is not None:
        state, done = store[best], len(best)
        save(best, state)

    while done < len(prompt):
        stop = min(len(prompt), (done // interval + 1) * interval)
        logits, state = lm(prompt[done:stop], state)
        done = stop
        save(prompt[:done], state)

    sequence, generated = list(prompt), []
    while True:
        token = int(torch.argmax(logits[-1]))
        generated.append(token)
        if len(generated) == max_new_tokens:
            break
        sequence.append(token)
        logits, state = lm([token], state)
        if len(sequence) % interval == 0:
            save(sequence, state)

    while len(new_store) > max_checkpoints:
        del new_store[next(iter(new_store))]
    return generated, new_store
```

Because $r < P$, the prompt loop always executes at least once, so `logits` is always initialized before decoding starts. The JAX and MLX solutions differ only in the `argmax` line.

## Common mistakes

- **Treating the store as a KV prefix cache.** Picking the entry with the longest common prefix and continuing from its state feeds the remaining tokens of that entry followed by the rest of the new prompt. No error is raised, and the model silently generates tokens for the wrong prompt.
- **Rounding the shared length up.** The nearest checkpoint to where two prompts diverge must be at or before the divergence point. A checkpoint past that point has already absorbed tokens that do not appear in the new prompt.
- **Resuming from the full prompt.** When a prompt is repeated, a key of length $P$ already exists in the store. Resuming from it leaves zero prompt tokens to feed, and feeding the last prompt token again to obtain logits causes the model to read that token twice.
- **Skipping checkpoints during generation.** A follow-up turn begins with the previous prompt plus the generated reply. If checkpoints are only saved during prefill, the next turn resumes at the end of the prompt and re-reads the entire reply.
- **Saving only at the end of the prompt.** Saving a single checkpoint per request only helps exact continuations. A regenerated answer, an edited message, or another user sharing the same system prompt will find no usable checkpoint.
- **Skipping the checkpoint at the end of the prompt.** Saving only at multiples of `interval` forces a direct continuation to recompute up to `interval - 1` prompt tokens unnecessarily.
- **Counting the interval from the resume point.** After resuming at length 5 with `interval = 4`, checkpoints belong at 8 and 12 rather than 9 and 13. Relative offsets depend on request history and prevent different requests with the same prefix from sharing checkpoints.
- **Saving the wrong state under a key.** Saving the prompt checkpoint after the decode loop stores the post-generation state under the prompt's key, causing the next request to silently inherit the generated reply.
- **Prefilling one token at a time.** Calling `lm` on each token individually produces the right states and tokens, but uses $P - r$ forward calls instead of a few segmented calls.
- **Evicting without tracking recency.** FIFO eviction drops the oldest entry, which is often a frequently reused system prompt. Refusing to insert new entries when full freezes the store on stale prefixes, and skipping eviction altogether allows memory to grow without bound.

## Complexity

When resuming from length $r$, the model processes $P - r + g - 1$ tokens across $s + g - 1$ forward calls, where $s \le \lceil (P - r) / n \rceil + 1$. One request creates at most $\lfloor (P + g - 1) / n \rfloor + 1$ checkpoints, each consuming `state.nbytes` bytes regardless of sequence length. The interval $n$ trades memory for recomputation. A diverging prompt recomputes roughly $n / 2$ shared tokens on average, while the number of stored checkpoints scales as $1 / n$.

## In the wild

- **SGLang's `MambaRadixCache`.** The PyTorch blog post "Hybrid Models Meet SGLang: More than Full Attention" (December 2025) explains why recurrent states are updated in place, cannot be rolled back to earlier prefixes, and are orders of magnitude larger than a single token's KV entry. SGLang addresses this with `MambaRadixCache`, a radix tree whose lookup finds the deepest node holding a state whose key is a prefix of the input and copies that state out. Its insert step forks checkpoints after chunked prefill or decoding, and it evicts recurrent states and KV blocks separately by recency. That mirrors this problem's lookup rule, segment-boundary saving, and LRU eviction.
- **llama.cpp context checkpoints.** Pull request #16382 in `ggml-org/llama.cpp` ("implement context checkpointing for hybrid and recurrent models") generalized sliding-window checkpoints to recurrent and hybrid architectures. The server exposes `--ctx-checkpoints N` for the maximum checkpoints per slot (default 32) and `--checkpoint-min-step N` for the minimum token spacing between checkpoints (default 8192), which correspond to `max_checkpoints` and, loosely, `interval`.
- **Recurrent layer architectures.** These layers appear in "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (arXiv 2312.00752) and "Gated Delta Networks: Improving Mamba2 with Delta Rule" (arXiv 2412.06464). The Gated Delta Network linear-attention layer is used in Qwen3-Next (featured in the SGLang post) and in three out of every four layers of Qwen3.5, and is built in `gated-delta-step` and `gated-delta-layer`.
- **Simplifications in this problem.** The model here is purely recurrent. A hybrid model also contains attention layers whose KV cache *can* be truncated, so a real engine restores the recurrent state from a checkpoint and trims the KV cache to that same length. The store here is a flat dict scanned linearly, whereas a radix tree finds the deepest usable prefix in time proportional to the prompt length. Production engines also choose checkpoint positions around prefill chunks and request boundaries rather than a fixed interval, and their eviction policies are more nuanced than a single global LRU list.
