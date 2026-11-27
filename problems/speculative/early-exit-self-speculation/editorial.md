## The idea

A forward pass of an $L$-layer decoder over a sequence is a chain: embedding, layers $0 \dots L - 1$, final norm, and output head. An early exit cuts this chain after layer $E - 1$ and applies the same norm and head there. When the model is trained so that this exit is a reasonable predictor (as in LayerSkip, which uses layer dropout and an early-exit loss), layers $0 \dots E - 1$ plus the head act as a draft model that is a prefix of the target's own forward pass.

This changes two things compared with a separate draft model.

**Verification is a continuation, not a second pass.** To verify $k'$ drafts, the target needs its full-depth output at each drafted position and the position right before them. The first $E$ layers of that pass are identical to what the draft loop just ran, because causal attention at any position depends only on earlier tokens, and those tokens are the same. Verification therefore starts at layer $E$ using the saved exit states as input, paying $L - E$ layers instead of $L$. One call over all $k' + 1$ positions yields $g_0, \dots, g_{k'}$, the full model's greedy tokens conditioned on the preceding drafts.

**There is one cache.** Layers $0 \dots E - 1$ append their keys and values while drafting, and layers $E \dots L - 1$ append theirs for those exact same positions during verification. Between the two phases, the early layers are ahead of the late layers, which is why the cache tracks lengths per layer. After verification finishes, all layers cover the same prefix again.

**Correctness.** The proof matches `speculative-decode-loop`. Token $g_0$ is the full model's greedy choice given the verified sequence $S$. If $d_1 = g_0$, then $g_1$ was computed on $S, d_1$ and is therefore the full model's true next token. The same holds for every consecutive match up to the first mismatch, where $g_n$ gives the full model's true token. Every token after the mismatch was conditioned on a rejected draft and is discarded. The emitted tokens $d_1 \dots d_n, g_n$ are therefore identical to greedy decoding, regardless of exit quality. The exit only determines how many rounds are needed.

**Rollback.** At the end of a round, every layer has cached keys and values for all $k'$ drafts, including the $k' - n$ rejected ones. All layers must be truncated back to the verified tokens they have already processed. After appending the round's $n + 1$ emitted tokens to $S$, that length is $|S| - 1$, because the newest emitted token (the correction or bonus token) has not been embedded yet and will be the first input of the next round. Leaving a rejected draft in any layer's cache allows every later position to attend to it and corrupts the output.

**What is saved.** On a GPU, decoding latency tracks sequential layer evaluations (`layer_calls`), where one call over a short batch of tokens costs about the same as one call over a single token. Plain greedy decoding pays $L$ layer calls per token. One round here pays $E(k' + 1)$ calls for drafting plus $L - E$ calls for verification, emitting $n + 1$ tokens. When all $k$ drafts are accepted, the cost drops to $E + (L - E)/(k + 1)$ layer calls per token. The early layers still run once per token, while the late layers run once per round. Total token work (`layer_tokens`) does not decrease, because every scored position passes through all $L$ layers whether accepted or rejected. Self-speculation trades extra batched compute in the late layers for fewer sequential steps.

## Reference solution (PyTorch)

```python
def self_speculative_greedy(lm, prompt, exit_layer, k, max_new_tokens):
    seq = list(prompt)
    cache = lm.empty_cache()
    while True:
        seen = cache.length
        n_draft = min(k, max_new_tokens - (len(seq) - len(prompt)) - 1)

        feed, position = seq[seen:], seen
        exits, drafts = [], []
        while True:
            h, cache = lm.layers(lm.embed(feed, offset=position), 0, exit_layer, cache)
            exits.append(h)
            position += len(feed)
            if len(drafts) == n_draft:
                break
            feed = [int(torch.argmax(lm.head(lm.norm(h[-1:]))))]
            drafts.append(feed[0])

        h, cache = lm.layers(torch.cat(exits), exit_layer, lm.n_layers, cache)
        greedy = torch.argmax(lm.head(lm.norm(h[-(n_draft + 1) :])), dim=-1).tolist()
        n = 0
        while n < n_draft and drafts[n] == greedy[n]:
            n += 1
        seq += [*drafts[:n], greedy[n]]
        if len(seq) - len(prompt) == max_new_tokens:
            return seq[len(prompt) :]
        cache = cache.truncate(len(seq) - 1)
```

The draft loop runs `n_draft + 1` early-layer steps. The final step feeds the last draft solely to produce its exit state for the bonus position. In the first round, `seq[seen:]` is the entire prompt, so every prompt token passes through each layer once. Calling `cache.truncate(len(seq) - 1)` rolls back every layer together, preserving the invariant at the top of the loop that all layers have processed `seq[:seen]`.

## Common mistakes

- **Recomputing the early layers.** Verifying with a full forward pass from layer 0 (porting a two-model loop with separate draft and target caches) returns the right tokens but pays $E$ extra layers on every scored position, increasing `layer_tokens` from $L$ to $L + E$ per position.
- **Rolling back only the early cache.** When keeping two separate cache objects, it is easy to truncate the draft cache and forget the late layers. The late layers then attend to rejected tokens in the next round, producing wrong output with no runtime error.
- **No rollback at all.** Failing to truncate the cache causes the same silent corruption across every layer.
- **An unnormalized exit.** Calling `lm.head(h)` directly on the raw exit state skips the learned gain in `lm.norm`. The full model still corrects the mistakes, so the output stays exact while the acceptance rate collapses.
- **Verifying position by position.** Calling the late layers once per drafted token gives the right output but uses as many layer calls as standard greedy decoding.
- **Dropping the bonus token.** Emitting only $k'$ tokens when all drafts match forces the next round to re-derive a token the full model already computed.
- **Drafting past the budget.** With $r$ tokens left to generate, at most $r - 1$ drafts can be emitted (the $r$-th comes from verification). Drafting $k$ tokens anyway and slicing the output wastes layer calls in the final round, and wastes all of them when `max_new_tokens == 1`.
- **Rebuilding the cache every round.** Clearing the cache instead of truncating makes total compute quadratic in the sequence length.

## Complexity

Across $R$ rounds and $T = \sum_i (k'_i + 1)$ scored positions, the algorithm uses $E \cdot T + (L - E) \cdot R$ layer calls and $L \cdot (P - 1 + T)$ layer-tokens. By comparison, cached greedy decoding of $g$ tokens uses $L \cdot g$ layer calls and $L \cdot (P + g - 1)$ layer-tokens. Memory is one KV cache across all $L$ layers plus the saved exit states for the current round (at most $k + 1$ positions, or $P + k$ positions in the first round).

## In the wild

- "LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding" (arXiv:2404.16710) introduces this method. It decodes with early layers, verifies and corrects with the remaining layers, and shares one LM head across all layers. Because the first $E$ layers are identical in both stages, a single KV cache serves both, and the exit layer's hidden state is saved (called the *exit query cache*) so verification continues from layer $E$ without recomputing early layers. The paper refers to the combined storage as the KVQ cache. The `exits` list in this problem plays the role of that exit query cache.
- The kit's `residual_scale` simulates LayerSkip's training recipe by making late layers refine early representations so the exit agrees with the full model. A model trained without early-exit supervision behaves like a low-agreement fixture, where self-speculation adds overhead instead of speed.
- The same cost accounting applies to any self-speculation scheme (such as layer skipping or a pruned/quantized copy sharing a cache). The speedup comes from whatever work verification can reuse, and disappears whenever verification restarts from token ids.
