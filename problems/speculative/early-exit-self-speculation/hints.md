In `speculative-decode-loop` the draft model and the target were separate, each
with its own cache, and the target started from the token ids. Here the "draft
model" is layers `0..E-1` of the target. When verification begins, what has
already been computed for the drafted positions, and what is the first layer that
still has work to do? What must you keep from the draft loop so that this layer
can start?

---

Keep the exit states. Every draft step is
`h, cache = lm.layers(lm.embed(feed, offset=position), 0, E, cache)`: store `h`,
take the argmax of `lm.head(lm.norm(h[-1:]))` as the next draft, and feed that
token next. Feed the last draft too (one more early step) so that its exit state
exists. Then verification is a single call:
`lm.layers(concatenation of the stored states, E, L, cache)`. Its last `k' + 1`
rows, through `lm.norm` and `lm.head`, are the full model's tokens `g_0 .. g_k'`.
One cache object serves all layers: the early ones run ahead during the draft,
and the late ones catch up in that one call.

---

Bookkeeping per round, with `seen = cache.length` at its start (0 in the first
round, `len(seq) - 1` afterwards):

- first feed: `seq[seen:]` at `offset=seen`; each later feed: the previous draft
  at the next position;
- `n_draft = min(k, max_new_tokens - generated - 1)`, and with `n_draft == 0` the
  draft loop is just the one early step for the unprocessed tokens;
- accept the longest prefix with `drafts[j] == greedy[j]`, then append
  `greedy[n]`;
- `cache = cache.truncate(len(seq) - 1)`: every layer back to the verified
  tokens that have been processed. The newest token has not been through any
  layer; it is the first feed of the next round.
