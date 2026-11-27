In `speculative-decode-loop`, the draft model and the target model were separate, each with its own cache, and the target started from token ids. Here the draft model is layers `0..E-1` of the target itself. When verification begins, what has already been computed for the drafted positions, and which layer still has work to do? What must you save during drafting so that verification can pick up from that layer?

---

Save the exit hidden states from the draft loop. Each draft step runs `h, cache = lm.layers(lm.embed(feed, offset=position), 0, E, cache)`. Append `h` to a list, take `argmax(lm.head(lm.norm(h[-1:])))` as the next draft token, and feed it on the next step. Run one final early-layer step on the last draft token so its exit state is also saved. Verification is then a single call to `lm.layers(concatenation of saved states, E, L, cache)`. Passing the last `k' + 1` rows through `lm.norm` and `lm.head` gives the full model's greedy tokens `g_0 .. g_k'`. One cache object serves all layers. The early layers advance during drafting, and the late layers catch up in that single verification call.

---

Track the per-round bookkeeping using `seen = cache.length` at the start of each round (`0` in the first round, `len(seq) - 1` afterwards):

- Start the draft loop by feeding `seq[seen:]` at `offset=seen`, then feed each new draft token one at a time at the next position.
- Set `n_draft = min(k, max_new_tokens - generated - 1)`. When `n_draft == 0`, the draft loop runs just the single early-layer step on the unprocessed tokens.
- Accept the longest prefix where `drafts[j] == greedy[j]`, then append `greedy[n]` (the correction or bonus token).
- Roll every layer back with `cache = cache.truncate(len(seq) - 1)` so only processed verified tokens remain. The newest emitted token has not passed through any layer yet and becomes the first input of the next round.
