## The idea

A forward pass of an $L$-layer decoder over one position is a chain: embed, layers
$0..L-1$, norm, head. Early exit cuts the chain after layer $E - 1$ and applies
the norm and the head there. If the model was trained so that this is a decent
predictor (LayerSkip trains with layer dropout and an early-exit loss for that
purpose), then layers $0..E-1$ plus the head are a draft model that is *a prefix
of the target's own computation*.

That changes two things compared with a separate draft model.

**Verification is a continuation, not a second pass.** To verify drafts, the
target needs its full-depth output at each drafted position. The first $E$ layers
of that computation are exactly what the draft loop already ran: the state after
layer $E - 1$ at a position depends only on the tokens up to that position, and
those are the same tokens. So verification starts at layer $E$ with the saved exit
states as input, and costs $L - E$ layers, not $L$. In one call over all
$k' + 1$ positions, the late layers produce $g_0, \dots, g_{k'}$, the full
model's greedy tokens, each conditioned on the drafts before it.

**There is one cache.** Layers $0..E-1$ append their keys and values while
drafting; layers $E..L-1$ append theirs for the same positions during
verification. Between the two phases the early layers are ahead, which is why the
kit's cache has per-layer lengths; after verification all layers cover the same
positions again.

**Correctness** is the argument of `speculative-decode-loop`. $g_0$ is the full
model's token given $S$; if $d_1 = g_0$, then $g_1$, computed given $S, d_1$, is
the full model's token given the true continuation, and so on up to the first
mismatch. Everything after it was computed on top of a token that is not emitted.
The emitted tokens $d_1..d_n, g_n$ are therefore the full model's greedy tokens,
whatever the exit predicts. The exit only decides how many rounds it takes.

**Rollback.** After a round, every layer has keys and values for the $k'$ drafts,
and $k' - n$ of them are rejected. All layers must go back to the verified
tokens they have processed: $|S| - 1$ after the round's tokens are appended,
since the newest token (the correction or bonus) has not been embedded yet. It is
the first thing the early layers are fed in the next round. A rejected draft left
in *any* layer's cache is attended to by every later position.

**What is saved.** Count layer evaluations the way latency sees them on a GPU, one
per layer per call. Greedy decoding pays $L$ per token. A round here pays
$E (k' + 1)$ for drafting and $L - E$ for verifying, and emits $n + 1$ tokens.
With everything accepted that is $E + (L - E)/(k + 1)$ per token: the early
layers still run once per token, the late layers once per round. The token count
goes the other way: every scored position passes through all $L$ layers whether
its draft is accepted or not, so `layer_tokens` is never below greedy decoding's.
Self-speculation trades batched work in the late layers for fewer sequential
steps.

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

The draft loop runs `n_draft + 1` early steps: the last one feeds the final draft
only to obtain its exit state, which the bonus position needs. In the first round
the first step carries the whole prompt, so the prompt passes through each layer
exactly once. `cache.truncate` rolls back every layer at once; the invariant at
the top of the loop is that all layers have processed `seq[:seen]`.

## Common mistakes

- **Recomputing the early layers.** Verifying with a full forward pass over the
  drafts (the natural port of a two-model loop, with one cache for the "draft
  model" and one for the "target") is correct and pays $E$ extra layers on every
  scored position: `layer_tokens` grows from $L$ to $L + E$ per position.
- **Rolling back only the early cache.** With two cache objects it is easy to
  truncate the one the draft loop uses and forget the other. The late layers then
  verify the next round against keys and values of rejected tokens: wrong output,
  no error.
- **No rollback at all**, the same failure in every layer.
- **An unnormalized exit.** `lm.head(h)` on the raw exit state. With a learned
  gain in the final norm this is a different predictor; the full model still
  corrects it, so the tokens are right and the acceptance rate collapses.
- **Verifying position by position.** One late-layer call per drafted token gives
  the right tokens with the layer calls of ordinary decoding.
- **Dropping the bonus token**, so that a fully accepted round emits $k'$ tokens
  instead of $k' + 1$ and the next round re-derives a token that was already
  known.
- **Drafting past the budget.** With $r$ tokens left, at most $r - 1$ drafts can
  be emitted; drafting $k$ regardless and cutting the result wastes layer calls in
  the last round (and all of them when `max_new_tokens` is 1).
- **Rebuilding the cache every round**, which makes the work quadratic in the
  sequence length.

## Complexity

With $R$ rounds and $T = \sum_i (k'_i + 1)$ scored positions:
$E \cdot T + (L - E) \cdot R$ layer calls and $L \cdot (P - 1 + T)$ layer-tokens,
against $L \cdot g$ and $L \cdot (P + g - 1)$ for cached greedy decoding of $g$
tokens. Memory: one KV cache for all layers, plus the exit states of at most
$k + 1$ positions (or of the prompt in the first round).

## In the wild

- "LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding"
  (arXiv:2404.16710) is the method: decode with earlier layers, verify and
  correct with the remaining layers, with a shared LM head for all layers. The
  paper states what is reused. Because the first $E$ layers are the same in both
  stages, a single KV cache serves draft and verification; and the exit layer's
  query is saved (the *exit query cache*) so that verification can continue from
  layer $E$ instead of recomputing the early layers. It calls the union of the
  two the KVQ cache. The exit states this problem stores play the role of
  that exit query cache.
- The kit's `residual_scale` plays the role of LayerSkip's training recipe: it
  makes the late layers refine what the early ones computed, so that an exit has
  something to agree with. A model trained without early-exit supervision
  behaves like the low-agreement fixture, and self-speculation makes it slower.
- The same accounting applies to any scheme that drafts with part of the target
  (skipping layers, a pruned or quantized copy sharing the cache): the saving is
  whatever the draft and the verification can share, and it is lost wherever the
  verification starts from the token ids again.
